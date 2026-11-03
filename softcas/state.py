"""Immutable, content-addressed X=(z,h,C,phi,beta,nu) and packages."""
from __future__ import annotations
import base64
import hashlib
import json
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import numpy as np
import torch


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


@dataclass(frozen=True)
class TensorBlob:
    dtype: str
    shape: tuple[int, ...]
    data: str

    @classmethod
    def take(cls, tensor):
        array = tensor.detach().cpu().contiguous().numpy()
        return cls(array.dtype.str, tuple(array.shape), base64.b64encode(array.tobytes()).decode())

    def tensor(self):
        a = np.frombuffer(base64.b64decode(self.data), dtype=np.dtype(self.dtype)).copy()
        return torch.from_numpy(a.reshape(self.shape))


def freeze_tensors(tensors) -> tuple[tuple[str, TensorBlob], ...]:
    return tuple((k, TensorBlob.take(v)) for k, v in sorted(tensors.items()))


def thaw_tensors(blobs):
    return {k: v.tensor() for k, v in blobs}


@dataclass(frozen=True)
class Structure:
    mask: tuple[int, ...]
    ratio: float

    def __post_init__(self):
        if len(self.mask) != 6 or any(m not in (0, 1) for m in self.mask):
            raise ValueError("Exactly six binary ports required")
        if self.ratio not in (1 / 16, 1 / 8):
            raise ValueError("Ratio outside eq:search_space")
        if sum(self.mask) > 2:
            raise ValueError("At most two local ports")


@dataclass(frozen=True)
class FitSettings:
    lr: float
    residual: float

    def __post_init__(self):
        if self.lr not in (1e-4, 3e-4, 1e-3) or self.residual not in (1e-4, 1e-3):
            raise ValueError("Settings outside eq:search_space")


@dataclass(frozen=True)
class PortSpec:
    base: str
    labels: tuple[str, ...]
    preprocess: str
    port: int
    channels: int
    stride: int
    layout: str = "NCHW"
    precision: str = "float32"


@dataclass(frozen=True)
class Package:
    owner: str
    version: str
    created: float
    available: float
    interface: PortSpec
    ratio: float
    tensors: tuple[tuple[str, TensorBlob], ...]
    provenance: str

    @property
    def hash(self):
        return digest(asdict(self))

    @property
    def wire(self):
        return canonical(asdict(self))

    @classmethod
    def read(cls, wire: bytes):
        d = json.loads(wire)
        d["interface"]["labels"] = tuple(d["interface"]["labels"])
        d["interface"] = PortSpec(**d["interface"])
        d["tensors"] = tuple((k, TensorBlob(v["dtype"], tuple(v["shape"]), v["data"])) for k, v in d["tensors"])
        return cls(**d)


@dataclass(frozen=True)
class State:
    z: Structure
    h: FitSettings
    C: tuple[str, ...]
    phi: tuple[tuple[str, TensorBlob], ...]
    beta: tuple[tuple[str, float], ...]
    nu: str

    def __post_init__(self):
        if self.C != tuple(sorted(set(self.C))) or tuple(k for k, _ in self.beta) != self.C:
            raise ValueError("Membership must be sorted, unique, and equal logit keys")
        canonical(dict(self.beta))
        json.loads(self.nu)

    @property
    def hash(self):
        return digest(asdict(self))

    def model_hash(self, base_identity: str):
        # h and history distinguish checkpoints, but do not change the forward map.
        execution = json.loads(self.nu)
        return digest({"base": base_identity, "z": asdict(self.z), "C": self.C,
                       "receiving_rule": execution.get("receiving_rule", "soft"),
                       "discrete_choices": execution.get("discrete_choices", {}),
                       "phi": [(k, asdict(v)) for k, v in self.phi], "beta": self.beta})

    def lineage(self, **changes):
        d = json.loads(self.nu)
        d.update(changes)
        return replace(self, nu=canonical(d).decode())

    def save(self, path):
        # Use the same atomic replacement as the persistent content store.
        # A failed write must leave an already deployed checkpoint readable.
        from .storage import atomic_write
        atomic_write(path, canonical({"hash": self.hash, "state": asdict(self)}))

    @classmethod
    def load(cls, path):
        return cls.read(Path(path).read_bytes())

    @classmethod
    def read(cls, wire, expected_hash=None):
        """Decode the verified payload and optionally bind its requested identity."""
        wrapper = json.loads(wire)
        d = wrapper["state"]
        d["z"] = Structure(tuple(d["z"]["mask"]), d["z"]["ratio"])
        d["h"] = FitSettings(**d["h"])
        d["C"] = tuple(d["C"])
        d["beta"] = tuple((k, v) for k, v in d["beta"])
        d["phi"] = tuple((k, TensorBlob(v["dtype"], tuple(v["shape"]), v["data"])) for k, v in d["phi"])
        state = cls(**d)
        if state.hash != wrapper["hash"]:
            raise ValueError("Checkpoint checksum mismatch")
        if expected_hash is not None and state.hash != expected_hash:
            raise ValueError("Checkpoint identity differs from requested state hash")
        return state
