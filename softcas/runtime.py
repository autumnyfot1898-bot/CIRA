"""Explicit CUDA runtimes for Windows RTX 5080 and Jetson Orin.

No hardware query or environment mutation happens at import time.
"""
from __future__ import annotations
import contextlib
import os
import platform
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import torch


@dataclass(frozen=True)
class RuntimeSpec:
    device: str = "cpu"
    precision: str = "float32"
    family: str = "cpu"
    deterministic: bool = True
    allow_tf32: bool = False
    channels_last: bool = False
    seed: int = 1001

    def validate(self):
        if self.precision not in ("float32", "float16", "bfloat16"):
            raise ValueError("Unknown inference/training precision")
        device = torch.device(self.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required by this profile; CPU fallback is disabled")
            capability = torch.cuda.get_device_capability(device)
            if self.family == "rtx5080" and (capability != (12, 0) or "5080" not in torch.cuda.get_device_name(device)):
                raise RuntimeError("RTX 5080 profile requires the specified Blackwell device")
            if self.family == "rtx5080" and "sm_120" not in torch.cuda.get_arch_list():
                raise RuntimeError("Install a Blackwell-enabled PyTorch CUDA wheel; the CPU/old CUDA environment is unsuitable")
            if self.family == "jetson_orin" and (platform.machine().lower() not in ("aarch64", "arm64") or capability != (8, 7)):
                raise RuntimeError("Orin profile requires aarch64 and compute capability 8.7")
            if self.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
                raise RuntimeError("Selected device/runtime does not support BF16")
        elif self.family != "cpu" or self.precision != "float32":
            raise ValueError("A GPU profile cannot silently execute on CPU")
        return device

    def activate(self):
        if self.deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        device = self.validate()
        if device.type == "cuda":
            torch.cuda.set_device(device)
            torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
            torch.backends.cudnn.allow_tf32 = self.allow_tf32
        torch.backends.cudnn.benchmark = not self.deterministic
        torch.use_deterministic_algorithms(self.deterministic)
        torch.manual_seed(self.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
        return device

    def autocast(self):
        if self.precision == "float32":
            return contextlib.nullcontext()
        dtype = torch.float16 if self.precision == "float16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def synchronize(self):
        if torch.device(self.device).type == "cuda":
            torch.cuda.synchronize(torch.device(self.device))

    def batch(self, value):
        if isinstance(value, torch.Tensor):
            return value.to(self.device, non_blocking=True)
        if isinstance(value, dict):
            return {k: self.batch(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return tuple(self.batch(v) for v in value)
        if isinstance(value, list):
            return [self.batch(v) for v in value]
        return value

    def scaler(self):
        # Scaler is fresh for every fitting block, alongside its optimizer.
        if self.precision != "float16":
            return None
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            return torch.amp.GradScaler("cuda")
        return torch.cuda.amp.GradScaler()

    def identity(self):
        info = {**asdict(self), "torch": torch.__version__, "cuda": torch.version.cuda,
                "python": platform.python_version(), "os": platform.platform(), "machine": platform.machine()}
        if torch.device(self.device).type == "cuda":
            info.update(name=torch.cuda.get_device_name(self.device),
                        capability=list(torch.cuda.get_device_capability(self.device)),
                        cudnn=torch.backends.cudnn.version())
        release = Path("/etc/nv_tegra_release")
        if release.is_file():
            info["l4t"] = release.read_text().strip()
        return info


def process_memory():
    """Current/peak RSS plus unified-memory availability, without double counting CUDA."""
    import psutil
    process = psutil.Process()
    memory = process.memory_info()
    system = psutil.virtual_memory()
    return {"rss_bytes": memory.rss, "peak_rss_bytes": getattr(memory, "peak_wset", memory.rss),
            "system_available_bytes": system.available, "system_total_bytes": system.total,
            "system_used_bytes": system.total - system.available}


class MemoryGuard:
    def __init__(self, process_limit, minimum_system_free=0, cuda_limit=None):
        self.process_limit = process_limit
        self.minimum_system_free = minimum_system_free
        self.cuda_limit = cuda_limit
        self.peak_rss = 0

    def check(self, expected_additional=0):
        from .resources import ExecutionStopped
        info = process_memory()
        self.peak_rss = max(self.peak_rss, info["rss_bytes"])
        if (self.process_limit is not None and info["rss_bytes"] + expected_additional > self.process_limit) or info["system_available_bytes"] - expected_additional < (self.minimum_system_free or 0):
            raise ExecutionStopped("Working-memory allowance reached")
        if self.cuda_limit is not None and torch.cuda.is_available():
            if torch.cuda.memory_reserved() + expected_additional > self.cuda_limit:
                raise ExecutionStopped("CUDA working-memory allowance reached")
        return info


class Deadline:
    def __init__(self, seconds):
        self.expires = time.monotonic() + seconds

    def check(self):
        from .resources import ExecutionStopped
        if time.monotonic() >= self.expires:
            raise ExecutionStopped("Monotonic execution deadline reached")
