"""Executed-operation FLOP accounting and calibrated state-conditioned estimates.

The convention is scalar arithmetic operations, MAC=2, transcendental=1. Memory
movement/indexing are zero FLOPs, but still consume metered wall time and bytes.
Unknown arithmetic operators invalidate complete coverage; they never become 0.
"""
from __future__ import annotations
import contextlib
import contextvars
import json
import math
import time
from collections import Counter
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from .state import digest, canonical
from .resources import Cost, ExecutionStopped

CURRENT_COUNTER = contextvars.ContextVar("softcas_flop_counter", default=None)


def _tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [t for v in value for t in _tensors(v)]
    if isinstance(value, dict):
        return [t for v in value.values() for t in _tensors(v)]
    return []


def _numel(value):
    return sum(t.numel() for t in _tensors(value) if t.is_floating_point() or t.is_complex())


ZERO = {
    "alias", "detach", "view", "_unsafe_view", "reshape", "as_strided", "slice", "select", "squeeze", "unsqueeze",
    "expand", "permute", "transpose", "t", "contiguous", "clone", "copy_", "_to_copy", "to", "empty", "empty_like",
    "empty_strided", "zeros", "zeros_like", "ones", "ones_like", "full", "full_like", "new_zeros", "new_empty",
    "new_empty_strided", "lift_fresh", "lift_fresh_copy", "fill_", "zero_", "cat", "stack", "split", "split_with_sizes",
    "unbind", "index", "index_put", "index_put_", "index_select", "gather", "scatter", "scatter_", "masked_select",
    "masked_fill", "masked_fill_", "where", "nonzero", "argmax", "argmin", "sort", "argsort", "topk", "unique_dim",
    "_unique2", "repeat", "repeat_interleave", "constant_pad_nd", "pad", "roll", "flip", "arange", "linspace",
    "scalar_tensor", "_local_scalar_dense", "is_nonzero", "isfinite", "isinf", "isnan", "any", "all", "eq", "ne",
    "lt", "le", "gt", "ge", "logical_and", "logical_or", "logical_not", "bitwise_and", "bitwise_or", "bitwise_not",
    "sym_size", "sym_numel", "size", "numel", "stride", "storage_offset", "record_stream", "_assert_async",
    "resolve_conj", "resolve_neg", "conj", "conj_physical", "detach_", "set_", "resize_", "resize_as_",
    "searchsorted", "bucketize", "cummax", "cummin", "floor", "floor_", "ceil", "ceil_", "round", "round_",
    "trunc", "trunc_", "sign", "sign_", "_index_put_impl_", "slice_backward", "select_backward", "unbind_copy",
}
ELEMENTWISE = {"add": 1, "add_": 1, "sub": 1, "sub_": 1, "mul": 1, "mul_": 1, "div": 1, "div_": 1,
               "true_divide": 1, "reciprocal": 1, "neg": 1, "abs": 1, "square": 1, "sqrt": 1, "rsqrt": 2,
               "exp": 1, "log": 1, "log1p": 2, "pow": 1, "maximum": 0, "minimum": 0, "clamp": 0,
               "clamp_": 0, "clamp_min": 0, "clamp_max": 0, "relu": 0, "relu_": 0,
               "sigmoid": 4, "sigmoid_backward": 3, "silu": 5, "silu_": 5, "silu_backward": 9,
               "tanh": 1, "tanh_backward": 3, "threshold_backward": 0, "leaky_relu": 1,
               "addcmul": 2, "addcmul_": 2, "addcdiv": 2, "addcdiv_": 2,
               "uniform_": 2, "normal_": 5, "lerp": 3, "lerp_": 3, "remainder": 1, "fmod": 1}


def operator_flops(name, args, kwargs, out):
    if not any(t.is_floating_point() or t.is_complex() for t in _tensors((args, out))):
        return 0
    if name in ZERO:
        return 0
    if name in ELEMENTWISE:
        return ELEMENTWISE[name] * _numel(out)
    if name in ("mm", "matmul", "bmm"):
        return 2 * _numel(out) * args[0].shape[-1]
    if name in ("addmm", "baddbmm"):
        return 2 * _numel(out) * args[1].shape[-1] + _numel(out)
    if name in ("convolution", "_convolution", "conv2d"):
        x, weight = args[:2]
        kernel = math.prod(weight.shape[2:]) * weight.shape[1]
        return _numel(out) * (2 * kernel + int(len(args) > 2 and args[2] is not None))
    if name == "convolution_backward":
        grad, x, weight = args[:3]
        mask = args[10]
        one = 2 * grad.numel() * math.prod(weight.shape[2:]) * weight.shape[1]
        return one * (int(mask[0]) + int(mask[1])) + ((grad.numel() - grad.shape[1]) if mask[2] else 0)
    if name in ("sum", "mean", "prod"):
        n, outputs = args[0].numel(), _numel(out)
        return max(0, n - outputs) + (outputs if name == "mean" else 0)
    if name in ("cumsum", "cumprod"):
        dim = int(args[1])
        return args[0].numel() - args[0].numel() // args[0].shape[dim]
    if name in ("amax", "amin", "max", "min"):
        return 0  # comparisons are not floating-point arithmetic in this convention
    if name in ("_softmax", "_log_softmax", "softmax", "log_softmax"):
        return 4 * _numel(out)
    if name in ("_softmax_backward_data", "_log_softmax_backward_data"):
        return 4 * _numel(out)
    if name in ("native_batch_norm", "_native_batch_norm_legit", "_native_batch_norm_legit_no_training", "batch_norm"):
        return 4 * args[0].numel() + 2 * args[0].shape[1]
    if name in ("native_batch_norm_backward", "_native_batch_norm_legit_functional"):
        return 8 * args[0].numel()
    if name in ("avg_pool2d", "adaptive_avg_pool2d", "_adaptive_avg_pool2d"):
        return args[0].numel() + _numel(out)
    if name in ("avg_pool2d_backward", "_adaptive_avg_pool2d_backward"):
        return _numel(out)
    if name in ("max_pool2d_with_indices", "max_pool2d_with_indices_backward", "upsample_nearest2d", "upsample_nearest2d_backward"):
        return 0
    if name in ("upsample_bilinear2d", "upsample_bilinear2d_backward"):
        return 7 * _numel(out)
    if name in ("mse_loss", "mse_loss_backward", "smooth_l1_loss", "smooth_l1_loss_backward"):
        return 4 * args[0].numel()
    if name in ("binary_cross_entropy_with_logits", "binary_cross_entropy", "binary_cross_entropy_backward"):
        return 10 * args[0].numel()
    if name in ("nll_loss_forward", "nll_loss_backward", "nll_loss2d_forward", "nll_loss2d_backward"):
        return 2 * args[1].numel()
    if name.startswith("_foreach_"):
        inner = name.removeprefix("_foreach_")
        if inner in ELEMENTWISE:
            return ELEMENTWISE[inner] * (_numel(out) or _numel(args[0]))
        if inner in ("zero_", "copy_", "maximum_"):
            return 0
    return None


class ExecutedFlops(TorchDispatchMode):
    def __init__(self, strict=False, overrides=None, operation_limit=None):
        super().__init__()
        self.strict, self.overrides, self.operation_limit = strict, overrides or {}, operation_limit
        self.total = 0
        self.unknown = Counter()
        self.operators = Counter()

    def add_external(self, executor, count):
        self.total += count
        self.operators[executor] += 1
        if self.operation_limit is not None and self.total > self.operation_limit:
            raise ExecutionStopped("External executor computation allowance reached")

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = func._schema.name.split("::")[-1]
        self.operators[str(func)] += 1
        count = operator_flops(name, args, kwargs, out)
        if count is None and str(func) in self.overrides:
            rule = self.overrides[str(func)]
            count = rule["per_output_element"] * _numel(out) + rule.get("constant", 0)
        if count is None:
            self.unknown[str(func)] += 1
            if self.strict:
                raise ExecutionStopped("Uncalibrated executed arithmetic operator: " + str(func))
        else:
            self.total += count
            if self.operation_limit is not None and self.total > self.operation_limit:
                raise ExecutionStopped("Executed-operation computation allowance reached; overrun is retained")
        return out


class CostProfile:
    def __init__(self, path, runtime_identity, strict=False):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text(encoding="utf-8"))
        if self.data["runtime_hash"] != digest({k: v for k, v in runtime_identity.items() if k != "seed"}):
            raise ValueError("Cost profile belongs to a different hardware/runtime/precision context")
        if strict and (self.data.get("status") != "validated" or self.data.get("unknown_operators")):
            raise ValueError("Formal accounting requires validated operator coverage")

    def estimate(self, signature, fit_steps, acquisition_bytes=0, evaluation_new=True, descriptors=None):
        row = self.data["models"].get(signature)
        if row is None and self.data.get("component_model") and descriptors:
            cm = self.data["component_model"]
            mode = descriptors["mode"]
            row = dict(cm["base"][mode])
            for port, ratio, owner in descriptors["branches"]:
                part = cm["branches"][f"{port}/{ratio}/{owner}/{mode}"]
                for key in ("fit_step_F", "fit_step_W", "evaluation_F", "evaluation_W", "peak_additional_bytes"):
                    row[key] += part[key]
        if row is None:
            raise ValueError("Missing architecture/composition calibration: " + signature)
        safety = self.data["admission_safety"]
        return Cost(F=(row["fit_step_F"] * fit_steps + row["evaluation_F"]) * safety["F"],
                    E=int(evaluation_new), B=acquisition_bytes,
                    W=(row["fit_step_W"] * fit_steps + row["evaluation_W"]) * safety["W"])


def calibrate_states(backend, cache, data, states, config, output):
    """Explicit later command. Inputs must be independent development states/data."""
    if config["purpose"] != "DEVELOPMENT":
        raise ValueError("Cost calibration is fixed on independent development trajectories")
    from .resources import Ledger
    results, unknown = {}, Counter()
    repeats = config["calibration"]["repeats"]
    for state in states:
        for mode in ("configuration", "collaboration"):
            ledger = Ledger(Cost(1e300, 1e300, 1e300, 1e300), Cost(1e300, 1e300, 1e300, 1e300))
            ledger.runtime = backend.runtime
            ledger.meter_policy = {**config["meter"], "strict": False, "enforce_F": False}
            with measured_work(ledger, "construct"):
                model = backend.assembly(state, cache, mode="refine" if mode == "configuration" else "collaboration")
            construction = ledger.used["search"]
            optimizer = backend.optimizer(model, state, config)
            scaler = backend.new_scaler()
            times, flops = [], []
            if optimizer:
                for i in range(repeats):
                    before = ledger.used["search"]
                    with measured_work(ledger, "fit"):
                        optimizer.zero_grad(set_to_none=True)
                        backend.fit_batch(model, data.D.batches[i % len(data.D.batches)], state.h.residual, optimizer, scaler)
                    delta = ledger.used["search"] - before
                    times.append(delta.W)
                    flops.append(delta.F)
            before = ledger.used["search"]
            with measured_work(ledger, "evaluation"):
                _ = backend.full_base_identity, data.V.identity
                backend.score(model, data.V)
            evaluation = ledger.used["search"] - before
            signature = backend.candidate_signature(state, cache, mode)
            from .runtime import process_memory
            results[signature] = {"fit_step_F": max(flops, default=0), "fit_step_W": max(times, default=0),
                                  "evaluation_F": evaluation.F + construction.F, "evaluation_W": evaluation.W + construction.W,
                                  "peak_additional_bytes": process_memory()["rss_bytes"],
                                  "z": __import__("dataclasses").asdict(state.z),
                                  "members": [(cache.get(k).interface.port, cache.get(k).ratio) for k in state.C], "mode": mode}
            for event in ledger.events:
                unknown.update(event.get("unknown_operators", {}))
    value = {"schema": 1, "status": "coverage_complete_unreviewed" if not unknown else "incomplete_coverage",
             "counting_convention": "scalar_ops_mac2_v1", "runtime_hash": digest({k: v for k, v in backend.runtime.identity().items() if k != "seed"}),
             "base_sha256": backend.identity, "execution": backend.execution,
             "port_interfaces": [__import__("dataclasses").asdict(p) for p in backend.specs],
             "runtime": backend.runtime.identity(), "models": results, "unknown_operators": dict(unknown),
             "overrides": config["meter"].get("overrides", {}),
             "admission_safety": config["calibration"]["admission_safety"], "data": {"D": data.D.identity, "V": data.V.identity},
             "validation_note": "Review operator rules, peak bounds and safety on held-out development before setting status=validated"}
    value["component_model"] = fit_components(results)
    from .storage import atomic_write
    atomic_write(output, canonical(value))
    return value


def fit_components(rows):
    """Nonnegative development cost model, with a positive-residual envelope.

    This is an admission estimator, never a replacement for executed accounting
    or a proof of an unseen hard upper bound.
    """
    import numpy as np
    from scipy.optimize import nnls
    features = [(port, ratio, owner) for owner in ("local", "external") for port in range(6) for ratio in (1 / 16, 1 / 8)]
    base, branches = {}, {}
    for mode in ("configuration", "collaboration"):
        selected = [r for r in rows.values() if r["mode"] == mode]
        X = []
        for r in selected:
            descriptors = [(i, r["z"]["ratio"], "local") for i, bit in enumerate(r["z"]["mask"]) if bit]
            descriptors += [(int(port), ratio, "external") for port, ratio in r["members"]]
            X.append([1.0] + [float(descriptors.count(f)) for f in features])
        X = np.asarray(X)
        if not len(X):
            raise ValueError("Missing development observations for " + mode)
        supported = np.flatnonzero(X[:, 1:].sum(0) > 0)
        base[mode] = {}
        for f in supported:
            port, ratio, owner = features[f]
            branches[f"{port}/{ratio}/{owner}/{mode}"] = {}
        for field in ("fit_step_F", "fit_step_W", "evaluation_F", "evaluation_W", "peak_additional_bytes"):
            y = np.asarray([r[field] for r in selected])
            active = np.r_[0, supported + 1]
            beta, _ = nnls(X[:, active], y, maxiter=10000)
            excess = max(0.0, float(np.max(y - X[:, active] @ beta)))
            base[mode][field] = float(beta[0] + excess)
            for j, f in enumerate(supported):
                port, ratio, owner = features[f]
                branches[f"{port}/{ratio}/{owner}/{mode}"][field] = float(beta[j + 1])
    return {"base": base, "branches": branches, "rule": "nonnegative_additive_plus_development_residual_envelope",
            "hard_bound_claim": False, "uncalibrated_features": [f for f in features if not any(k.startswith(f"{f[0]}/{f[1]}/{f[2]}/") for k in branches)]}


@contextlib.contextmanager
def measured_work(ledger, reason, scope="search", selection=False):
    ledger.guard(Cost(E=int(selection)), scope)
    runtime = getattr(ledger, "runtime", None)
    if runtime is not None:
        runtime.synchronize()
    start = time.perf_counter()
    policy = getattr(ledger, "meter_policy", {})
    counter = ExecutedFlops(strict=policy.get("strict", False), overrides=policy.get("overrides"),
                            operation_limit=max(0, ledger.remaining(scope).F) if policy.get("enforce_F") else None)
    try:
        token = CURRENT_COUNTER.set(counter)
        with counter:
            yield
    finally:
        CURRENT_COUNTER.reset(token)
        try:
            if runtime is not None:
                runtime.synchronize()
        finally:
            ledger.charge(reason, Cost(F=counter.total, E=int(selection), W=time.perf_counter() - start), scope,
                          meter="scalar_ops_mac2_v1", coverage_complete=not bool(counter.unknown),
                          unknown_operators=dict(counter.unknown), operators=dict(counter.operators))
