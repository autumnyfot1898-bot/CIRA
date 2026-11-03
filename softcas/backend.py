"""Backend contract, synthetic behavior fixture, native YOLO loss integration."""
from __future__ import annotations
import hashlib
import inspect
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import torch
from torch import nn
from .state import PortSpec, State, canonical, digest
from .model import Assembly
from .resources import Cost


@contextmanager
def work(ledger, reason, scope="search", selection=False):
    """Measure executed supported CPU-kernel FLOPs and elapsed wall time.

    torch.profiler omits some backward/elementwise/controller/checksum operations.
    This is a disclosed partial FLOP meter, NOT a formal calibrated total-F meter.
    Runtime caps for E/B and inter-operation W are enforced independently.
    """
    if getattr(ledger, "meter_policy", None) is not None:
        from .metering import measured_work
        with measured_work(ledger, reason, scope, selection):
            yield
        return
    ledger.guard(Cost(E=int(selection)), scope)
    start = time.perf_counter()
    profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True)
    profiler.__enter__()
    try:
        yield
    finally:
        profiler.__exit__(None, None, None)
        flops = sum(event.flops for event in profiler.key_averages())
        ledger.charge(reason, Cost(F=flops, E=int(selection), W=time.perf_counter() - start), scope,
                      meter="torch.profiler_supported_cpu_flops_partial", completion="executed")


class TinyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv2d(3 if i == 0 else 8, 8, 3, padding=1),
                                                   nn.BatchNorm2d(8), nn.SiLU()) for i in range(6)])
        self.head = nn.Linear(8, 2)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x.mean((2, 3)))


class Backend:
    metric = "synthetic_quality_NOT_AP"
    execution = "cpu_float32_eager_batch2"

    def optimizer(self, model, state, config):
        return model.optimizer(state.h.lr, config["beta_lr"], config.get("optimizer"))

    def new_scaler(self):
        runtime = getattr(self, "runtime", None)
        return runtime.scaler() if runtime else None

    def fit_batch(self, model, batch, residual_weight, optimizer, scaler=None):
        import contextlib
        runtime = getattr(self, "runtime", None)
        context = runtime.autocast() if runtime else contextlib.nullcontext()
        with context:
            loss = self.loss(model, batch) + residual_weight * model.penalty()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite native fitting loss")
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        return float(loss.detach())

    def candidate_signature(self, state, cache, mode="configuration"):
        return self.signature_parts(state.z, sorted((cache.get(k).interface.port, cache.get(k).ratio) for k in state.C),
                                    __import__("json").loads(state.nu).get("receiving_rule", "soft"), mode)

    def signature_parts(self, z, members, receiving_rule, mode="configuration"):
        return digest({"z": asdict(z), "members": sorted(members), "mode": mode,
            "receiving_rule": receiving_rule,
            "execution": self.execution})

    def before_candidate(self, state, cache):
        guard = getattr(self, "memory_guard", None)
        profile = getattr(self, "cost_profile", None)
        if guard:
            row = profile.data["models"].get(self.candidate_signature(state, cache)) if profile else None
            guard.check(row["peak_additional_bytes"] if row else 0)

    @property
    def full_base_identity(self):
        from .state import freeze_tensors
        return digest({"declared": self.identity, "actual": [(k, asdict(v)) for k, v in freeze_tensors(self.base.state_dict())]})

    @property
    def execution_identity(self):
        return digest({"execution": self.execution, "torch": torch.__version__,
                       "threads": torch.get_num_threads(), "platform": __import__("platform").platform(),
                       "sources": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                   for name in ("model.py", "backend.py", "evaluation.py", "postprocess.py")}})

    def __init__(self, seed=42):
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.base = TinyBase().eval()
        self.base.requires_grad_(False)
        from .state import freeze_tensors
        self.identity = digest([(k, asdict(v)) for k, v in freeze_tensors(self.base.state_dict())])
        self.port_modules = tuple(f"blocks.{i}" for i in range(6))
        self.specs = tuple(PortSpec(self.identity, ("toy0", "toy1"), "synthetic_v1", i, 8, 1) for i in range(6))
        self.evaluator = digest({"metric": self.metric, "code": inspect.getsource(type(self)),
                                 "torch": torch.__version__})

    def assembly(self, state, cache, seed=0, mode="import"):
        return Assembly(self.base, self.port_modules, self.specs, state, cache, seed, mode)

    def loss(self, model, batch):
        return torch.nn.functional.mse_loss(model(batch["img"]), batch["target"])

    def score(self, model, split):
        model.eval()
        with torch.no_grad():
            losses = [float(self.loss(model, b)) for b in split.batches]
        return 1.0 / (1.0 + sum(losses) / len(losses))

    def qualify(self, model, example, repeats=5):
        model.eval()
        times = []
        with torch.no_grad():
            model(example)
            for _ in range(repeats):
                start = time.perf_counter()
                model(example)
                times.append((time.perf_counter() - start) * 1000)
        import numpy as np
        size = sum(t.numel() * t.element_size() for t in model.state_dict().values())
        # RSS is conservative whole-process peak, not tracemalloc Python allocations.
        import ctypes
        from ctypes import wintypes
        if __import__("os").name == "nt":
            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (k, ctypes.c_size_t) for k in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            handle = ctypes.windll.kernel32.GetCurrentProcess
            handle.restype = wintypes.HANDLE
            getmem = ctypes.windll.psapi.GetProcessMemoryInfo
            getmem.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
            if not getmem(handle(), ctypes.byref(counters), counters.cb):
                raise RuntimeError("Cannot measure process peak RAM")
            ram = counters.PeakWorkingSetSize
        else:
            import resource
            ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        return {"deployed_tensor_bytes": size, "peak_process_ram_bytes": ram,
                "p95_ms": float(np.percentile(times, 95)), "repeats": repeats,
                "execution": self.execution, "active_external_paths": sum(
                    int(p.discrete_choice in p.external) if p.is_discrete else len(p.external) for p in model.ports.values()),
                "device": __import__("platform").platform(), "ram_scope": "process lifetime peak working set",
                "size_scope": "uncompressed full state_dict tensors; packaging overhead separate"}


class YoloBackend(Backend):
    metric = "AP50:95"

    def __init__(self, checkpoint, port_indices, labels, preprocessing, image_size=64, runtime=None):
        import ultralytics
        from ultralytics import YOLO
        from ultralytics.cfg import get_cfg
        if ultralytics.__version__ != "8.4.0":
            raise RuntimeError("Native loss contract verified for ultralytics==8.4.0; revalidate other versions")
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)  # no silent checkpoint download
        self.identity = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        self.checkpoint = str(checkpoint.resolve())
        self.image_size = image_size
        from .runtime import RuntimeSpec
        self.runtime = RuntimeSpec(**runtime) if isinstance(runtime, dict) else (runtime or RuntimeSpec())
        self.device = self.runtime.activate()
        self.base = YOLO(str(checkpoint)).model.float().cpu().eval()
        self.base.requires_grad_(False)
        self.base.args = get_cfg()
        if tuple(self.base.names.values()) != tuple(labels):
            raise ValueError("Fixed label vocabulary/output interpretation must match checkpoint")
        self.label_count = len(self.base.names)
        self.end2end = getattr(self.base, "end2end", False)
        self.port_modules = tuple(f"model.{i}" for i in port_indices)
        if len(set(port_indices)) != 6:
            raise ValueError("Six distinct declared YOLO feature ports required")
        modules = dict(self.base.named_modules())
        observed, handles = {}, []
        for name in self.port_modules:
            handles.append(modules[name].register_forward_hook(lambda m, a, o, n=name: observed.__setitem__(n, tuple(o.shape))))
        try:
            with torch.no_grad():
                self.base(torch.zeros(1, 3, image_size, image_size))
        finally:
            for handle in handles:
                handle.remove()
        self.specs = tuple(PortSpec(self.identity, tuple(labels), preprocessing, i,
                                   observed[name][1], image_size // observed[name][-1]) for i, name in enumerate(self.port_modules))
        self.base = self.base.to(self.device)
        self.execution = f"{self.runtime.device}_{self.runtime.precision}_eager_img{image_size}_frozen_eval_native_raw_loss"
        self.evaluator = digest({"ultralytics": ultralytics.__version__, "torch": torch.__version__,
                                 "code": inspect.getsource(type(self)), "metric": self.metric,
                                 "ap_implementation": "softcas.detection_ap_torch_101point_v1"})

    def assembly(self, state, cache, seed=0, mode="import"):
        model = Assembly(self.base, self.port_modules, self.specs, state, cache, seed, mode).to(self.device)
        if self.runtime.channels_last:
            model = model.to(memory_format=torch.channels_last)
        return model

    @property
    def execution_identity(self):
        return digest({"native": super().execution_identity, "runtime": self.runtime.identity()})

    def loss(self, model, batch):
        batch = self.runtime.batch(batch)
        outputs = model(batch["img"])
        # Frozen eval head returns (inference, raw). v8 raw is list[Tensor];
        # end-to-end heads return a raw dictionary. Never no_grad here.
        if not isinstance(outputs, tuple) or len(outputs) < 2:
            raise RuntimeError("Unexpected pinned YOLO eval/raw loss contract")
        raw = outputs[1]
        loss, components = model.base.loss(batch, preds=raw)
        return loss.sum() / batch["img"].shape[0]

    def predict(self, model, image, return_cpu=True):
        from .postprocess import decode
        image = self.runtime.batch(image)
        with self.runtime.autocast():
            raw = model(image)
        output = decode(raw, self.label_count, end2end=self.end2end)
        return [p.detach().float().cpu() for p in output] if return_cpu else output

    def score(self, model, split):
        from .evaluation import detection_ap_torch
        from ultralytics.utils.ops import xywh2xyxy
        predictions, targets = [], []
        model.eval()
        with torch.no_grad():
            for batch in split.batches:
                batch = self.runtime.batch(batch)
                out = self.predict(model, batch["img"], return_cpu=False)
                for i, pred in enumerate(out):
                    keep = batch["batch_idx"] == i
                    boxes = xywh2xyxy(batch["bboxes"][keep]) * batch["img"].shape[-1]
                    targets.append(torch.cat((batch["cls"][keep].reshape(-1, 1), boxes), dim=1).detach().cpu())
                    predictions.append(pred.detach().float().cpu())
        return detection_ap_torch(predictions, targets)

    def qualify(self, model, example, repeats=100):
        from .device import qualify_local
        report = qualify_local(model, example.detach().cpu(), self.runtime, repeats=repeats,
                               predict=lambda image: self.predict(model, image))
        report["latency_scope"] = "preprocessed CPU tensor, H2D, complete detector, fixed NMS, D2H; synchronized CUDA"
        return report
