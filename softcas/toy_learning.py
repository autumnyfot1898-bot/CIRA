"""A teacher-student CPU fixture with observable residual learning; never AP.

The ordinary TinyBase fixture tests control flow. This alternative preserves
the six-port Assembly contract but removes attenuation through six random
convolutional blocks. Targets are computed from a fixed teacher, not from
desired gains. D/V/A/T use disjoint, deterministic random draws.
"""
from __future__ import annotations

from dataclasses import asdict
import inspect
import math
from pathlib import Path
import hashlib
import torch
from torch import nn

from .backend import Backend
from .data import Split
from .state import PortSpec, canonical, digest, freeze_tensors


class IdentityPortBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(6)])
        self.head = nn.Linear(8, 2)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()
            self.head.weight[0, 0] = self.head.weight[1, 1] = 1.0

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x.mean((2, 3)))


class TeacherBackend(Backend):
    metric = "synthetic_quality_NOT_AP"
    execution = "cpu_float32_teacher_residual_fixture_v1"

    def __init__(self):
        # No activation is injected into the six-port execution path. Local
        # and external adapters use the same implementation as the main loop.
        with torch.random.fork_rng():
            torch.manual_seed(42)
            self.base = IdentityPortBase().eval().requires_grad_(False)
        self.identity = digest({"fixture": "identity_port_teacher_v1", "base":
            [(k, asdict(v)) for k, v in freeze_tensors(self.base.state_dict())]})
        self.port_modules = tuple(f"blocks.{i}" for i in range(6))
        self.specs = tuple(PortSpec(self.identity, ("toy0", "toy1"),
                                   "teacher_fixture_8channels_v1", i, 8, 1)
                           for i in range(6))
        self.evaluator = digest({"metric": self.metric,
                                 "implementation": inspect.getsource(type(self)),
                                 "scope": "SYNTHETIC_BEHAVIOR_ONLY_NOT_AP"})

    @property
    def execution_identity(self):
        return digest({"backend": super().execution_identity,
                       "teacher_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})

    def score(self, model, split):
        """Same monotone quality mapping, with double-precision accumulation.

        Forward/training remain float32. The double accumulation avoids hiding
        small changes behind a float32 MSE reduction; it does not add gains.
        """
        model.eval()
        squared, count = 0.0, 0
        with torch.no_grad():
            for batch in split.batches:
                error = model(batch["img"]).double() - batch["target"].double()
                squared += float(error.square().sum())
                count += error.numel()
        if not count:
            raise ValueError("Empty teacher split")
        return 1.0 / (1.0 + squared / count)


def teacher_split(role, client, stage, domain, seed, batches=2, batch_size=4,
                  target_direction=1.0):
    """Generate independent examples for the declared role and teacher domain.

    target_direction=-1 deliberately reverses the D teacher in the negative
    transfer probe. It is never selected by looking at a resulting V gain.
    """
    if role not in ("D", "V", "A", "T"):
        raise ValueError("A declared D/V/A/T role is required")
    if batches <= 0 or batch_size <= 0 or target_direction not in (-1.0, 1.0):
        raise ValueError("Invalid teacher fixture size/direction")
    generator = torch.Generator().manual_seed(int(seed))
    values = []
    angle = 0.2 * int(domain)
    for _ in range(batches):
        centers = 0.4 + 0.35 * torch.randn(batch_size, 8, 1, 1, generator=generator)
        image = centers + 0.1 * torch.randn(batch_size, 8, 6, 6, generator=generator)
        features = image.mean((2, 3))
        residual = 0.32 + 0.12 * torch.tanh(features[:, 2:4] + angle)
        target = features[:, :2] + float(target_direction) * residual
        values.append({"img": image, "target": target})
    manifest = canonical({"kind": "SYNTHETIC_TEACHER_NOT_AP", "role": role,
        "seed": int(seed), "domain": str(domain), "batches": batches,
        "batch_size": batch_size, "direction": target_direction}).decode()
    return Split(role, str(client), int(stage), str(domain), tuple(values), manifest)


def fitting_probe(backend, state, cache, D, V, steps=24, optimizer_settings=None):
    """Real gradients/updates and held-out scores; no controller label injection."""
    model = backend.assembly(state, cache, 42, "refine")
    before_base = {k: v.detach().clone() for k, v in model.base.state_dict().items()}
    before_local = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    optimizer = model.optimizer(state.h.lr, 0.01, optimizer_settings)
    if optimizer is None:
        raise ValueError("Probe needs trainable parameters")
    score_before = backend.score(model, V)
    with torch.no_grad():
        predictions_before = torch.cat([model(b["img"]).flatten() for b in V.batches])
        targets = torch.cat([b["target"].flatten() for b in V.batches])
    loss_before = float(backend.loss(model, D.batches[0]).detach())
    rows = []
    for i in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = backend.loss(model, D.batches[i % len(D.batches)]) + state.h.residual * model.penalty()
        loss.backward()
        gradients = {n: float(p.grad.detach().double().norm())
                     for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}
        if not all(math.isfinite(v) for v in gradients.values()):
            raise AssertionError("Nonfinite fixture gradient")
        old = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        optimizer.step()
        delta = math.sqrt(sum(float((p.detach().double() - old[n].double()).square().sum())
                            for n, p in model.named_parameters() if p.requires_grad))
        rows.append({"step": i + 1, "D_loss": float(loss.detach()),
                     "gradient_l2": math.sqrt(sum(v * v for v in gradients.values())),
                     "parameter_update_l2": delta, "gradient_by_parameter": gradients})
    score_after = backend.score(model, V)
    with torch.no_grad():
        predictions_after = torch.cat([model(b["img"]).flatten() for b in V.batches])
    parameter_delta = math.sqrt(sum(float((p.detach().double() - before_local[n].double()).square().sum())
                                   for n, p in model.named_parameters() if p.requires_grad))
    base_unchanged = all(torch.equal(value, before_base[name])
                         for name, value in model.base.state_dict().items())
    result = {"purpose": "SYNTHETIC_BEHAVIOR_ONLY_NOT_AP", "metric": backend.metric,
              "score_before": score_before, "score_after": score_after,
              "V_gain": score_after - score_before, "D_loss_before": loss_before,
              "V_prediction_change_l2": float((predictions_after.double() - predictions_before.double()).norm()),
              "V_loss_float64_before": float((predictions_before.double() - targets.double()).square().mean()),
              "V_loss_float64_after": float((predictions_after.double() - targets.double()).square().mean()),
              "D_loss_after": float(backend.loss(model, D.batches[0]).detach()),
              "local_parameter_delta_l2": parameter_delta,
              "base_unchanged": base_unchanged, "steps": rows,
              "D_identity": D.identity, "V_identity": V.identity}
    return model.snapshot(state), result
