"""Atomic immutable files, bounded checkpoint retention, and deployment bundles."""
from __future__ import annotations
import hashlib
import io
import json
import os
import tempfile
import time
import zipfile
from pathlib import Path
from .state import State, Package, canonical, digest
from .resources import Cost, ExecutionStopped


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class BoundedStore:
    def __init__(self, root, capacity, transient_capacity=None, ledger=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity
        self.transient_capacity = transient_capacity or capacity
        self.ledger = ledger
        self.pins = {}
        self.touched = {}
        self.counter = 0

    def _path(self, key):
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Content store requires a SHA-256 key")
        return self.root / key

    def keys(self):
        return sorted(p.name for p in self.root.iterdir() if p.is_file() and len(p.name) == 64)

    @property
    def bytes(self):
        return sum(self._path(k).stat().st_size for k in self.keys())

    def pin(self, owner, keys):
        self.pins[owner] = frozenset(keys)

    def unpin(self, owner):
        self.pins.pop(owner, None)

    def get(self, key):
        start = time.perf_counter()
        self.counter += 1
        self.touched[key] = self.counter
        data = self._path(key).read_bytes()
        if hashlib.sha256(data).hexdigest() != key:
            raise ValueError("Persistent object checksum mismatch")
        if self.ledger:
            self.ledger.charge("persistent_storage_read", Cost(W=time.perf_counter() - start),
                               disk_bytes=len(data), object_hash=key)
        return data

    def put(self, data, scope="search"):
        key = hashlib.sha256(data).hexdigest()
        path = self._path(key)
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError("Immutable persistent object conflict")
            return key
        if len(data) > self.transient_capacity or len(data) > self.capacity:
            raise ExecutionStopped("Object/transient-storage limit")
        pinned = set().union(*self.pins.values()) if self.pins else set()
        candidates = sorted((k for k in self.keys() if k not in pinned), key=lambda k: (self.touched.get(k, 0), k))
        total, victims = self.bytes + len(data), []
        for victim in candidates:
            if total <= self.capacity:
                break
            total -= self._path(victim).stat().st_size
            victims.append(victim)
        if total > self.capacity:
            raise ExecutionStopped("Protected checkpoint storage cannot be evicted")
        start = time.perf_counter()
        for victim in victims:
            self._path(victim).unlink()
        try:
            atomic_write(path, data)
        finally:
            if self.ledger:
                self.ledger.charge("persistent_storage_write", Cost(W=time.perf_counter() - start), scope,
                                   disk_bytes=len(data), object_hash=key, evicted=victims)
        self.counter += 1
        self.touched[key] = self.counter
        return key


class CheckpointStore:
    def __init__(self, store):
        self.store = store
        self.index = {}
        self.index_file = store.root / "checkpoint-index.json"
        if self.index_file.is_file():
            self.index = json.loads(self.index_file.read_bytes())

    def save(self, state):
        payload = canonical({"hash": state.hash, "state": __import__("dataclasses").asdict(state)})
        key = self.store.put(payload)
        self.index[state.hash] = key
        self.index = {s: k for s, k in self.index.items() if self.store._path(k).is_file()}
        atomic_write(self.index_file, canonical(self.index))
        return key

    def pin(self, token, states):
        states = tuple(states)
        payloads = [canonical({"hash": s.hash, "state": __import__("dataclasses").asdict(s)}) for s in states]
        if sum(len(p) for p in {hashlib.sha256(p).hexdigest(): p for p in payloads}.values()) > self.store.capacity:
            raise ExecutionStopped("Protected archive checkpoints exceed persistent capacity")
        existing = [self.index[s.hash] for s in states if s.hash in self.index and self.store._path(self.index[s.hash]).is_file()]
        self.store.pin(token, existing)
        keys = []
        for state in states:
            # Protect the existing set before admitting another checkpoint.
            key = self.save(state)
            keys.append(key)
            self.store.pin(token, existing + keys)
        self.store.pin(token, keys)

    def load(self, state_hash):
        key = self.index[state_hash]
        data = self.store.get(key)
        # The index is a lookup aid, not authority for the requested state.
        # Decode these checksum-verified bytes rather than reopening the file.
        return State.read(data, expected_hash=state_hash)


def bundle_bytes(backend, candidate, incumbent, cache, context):
    files = {}
    for state in (candidate, incumbent):
        files["states/" + state.hash + ".json"] = canonical({"hash": state.hash, "state": __import__("dataclasses").asdict(state)})
        for key in state.C:
            files["packages/" + key + ".json"] = cache.get(key).wire
    manifest = {"schema": 1, "base_sha256": backend.identity, "image_size": backend.image_size,
                "ports": [int(name.split(".")[-1]) for name in backend.port_modules],
                "labels": list(backend.specs[0].labels), "preprocessing": backend.specs[0].preprocess,
                "candidate": candidate.hash, "incumbent": incumbent.hash, "context": context,
                "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}}
    files["bundle.json"] = canonical(manifest)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return output.getvalue()


def unpack_bundle(data, root, maximum_bytes):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        if len(members) > 2048 or sum(m.file_size for m in members) > maximum_bytes:
            raise ValueError("Deployment bundle exceeds transient-storage limit")
        names = [m.filename for m in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate bundle members")
        manifest = json.loads(archive.read("bundle.json"))
        if set(names) != set(manifest["files"]) | {"bundle.json"}:
            raise ValueError("Unlisted bundle member")
        for name, checksum in manifest["files"].items():
            target = (root / name).resolve()
            if root not in target.parents or target.is_symlink():
                raise ValueError("Bundle member escapes deployment root")
            content = archive.read(name)
            if hashlib.sha256(content).hexdigest() != checksum:
                raise ValueError("Bundle content checksum mismatch")
            atomic_write(target, content)
        atomic_write(root / "bundle.json", canonical(manifest))
    return manifest
