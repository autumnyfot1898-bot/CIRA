"""Observed advertisements, verified immutable versions, pins, and restart transfers."""
from __future__ import annotations
import time
import hashlib
from dataclasses import dataclass, asdict
from .state import Package, canonical
from .resources import Cost, ExecutionStopped, TransferJournal


class PackageCache:
    def __init__(self, capacity_bytes, interfaces, store=None, allowed_package=None):
        self.capacity = capacity_bytes
        self.interfaces = tuple(interfaces)
        self._wire = {}
        self.pins = {}
        self.touched = {}
        self.clock = 0
        self.transitions = []
        self.store = store
        self.allowed_package = allowed_package or (lambda key: True)

    def __contains__(self, key):
        return self.allowed_package(key) and (key in self._wire or (self.store is not None and self.store._path(key).is_file()))

    @property
    def bytes(self):
        return sum(map(len, self._wire.values()))

    def keys(self):
        return tuple(k for k in sorted(set(self._wire) | (set(self.store.keys()) if self.store else set())) if self.allowed_package(k))

    def get(self, key):
        if not self.allowed_package(key):
            raise ExecutionStopped('External package excluded by donor policy')
        if key not in self._wire and self.store:
            self.put(key, self.store.get(key))
        self.clock += 1
        self.touched[key] = self.clock
        if self.store:
            self.store.touched[key] = self.clock
        return Package.read(self._wire[key])

    def pin(self, token, keys):
        keys = tuple(keys)
        if any(not self.allowed_package(k) for k in keys):
            raise ValueError('Pinned state violates donor policy')
        self.pins[token] = frozenset(keys)
        if self.store:
            self.store.pin(token, self.pins[token])

    def unpin(self, token):
        self.pins.pop(token, None)
        if self.store:
            self.store.unpin(token)

    def put(self, key, wire, now=None):
        if not self.allowed_package(key):
            raise ExecutionStopped('External package excluded by donor policy')
        now = time.time() if now is None else now
        if hashlib.sha256(wire).hexdigest() != key:
            raise ValueError("Wire-byte checksum mismatch")
        package = Package.read(wire)
        if package.hash != key:
            raise ValueError("Package checksum mismatch")
        if package.interface not in self.interfaces:
            raise ValueError("Incompatible exact port interface")
        if package.created > package.available or package.created > now or package.available > now:
            raise ValueError("Future or invalid package publication time")
        if len(wire) > self.capacity:
            raise ExecutionStopped("Package exceeds cache/transient storage allowance")
        import math
        if not math.isfinite(package.ratio) or package.ratio <= 0:
            raise ValueError("Invalid external width")
        hidden = max(1, int(package.interface.channels * package.ratio))
        expected = {"down.weight": (hidden, package.interface.channels, 1, 1),
                    "dw.weight": (hidden, 1, 3, 3), "up.weight": (package.interface.channels, hidden, 1, 1), "gate": ()}
        if {k: blob.shape for k, blob in package.tensors} != expected:
            raise ValueError("Package architecture/shape mismatch")
        # Check bytes/shapes/dtype directly; do not initialize random neural
        # modules merely to validate an immutable received tensor payload.
        import torch
        if len(package.tensors) != len(expected) or any(blob.dtype != "<f4" for _, blob in package.tensors):
            raise ValueError("Declared float32 package must contain each expected tensor exactly once")
        if not all(torch.isfinite(blob.tensor()).all() for _, blob in package.tensors):
            raise ValueError("Nonfinite received package weights/gate")
        if key in self._wire:
            if self._wire[key] != wire:
                raise ValueError("Published versions are immutable")
            return
        protected = set().union(*self.pins.values()) if self.pins else set()
        victims = sorted((k for k in self._wire if k not in protected),
                         key=lambda k: (self.touched[k], k))
        # Plan eviction atomically: failed admission cannot evict dependencies.
        evict = []
        size = self.bytes + len(wire)
        for victim in victims:
            if size <= self.capacity:
                break
            size -= len(self._wire[victim])
            evict.append(victim)
        if size > self.capacity:
            raise ExecutionStopped("Pinned cache capacity")
        if self.store and self.store.put(wire) != key:
            raise ValueError("Persistent package checksum mismatch")
        for victim in evict:
            del self._wire[victim]
            self.transitions.append({"event": "evict", "hash": victim})
        self._wire[key] = bytes(wire)
        self.clock += 1
        self.touched[key] = self.clock
        self.transitions.append({"event": "verified", "hash": key, "bytes": len(wire)})


@dataclass(frozen=True)
class Advertisement:
    hash: str
    owner: str
    observed: float
    available: float
    size: int
    interface: object
    ratio: float | None = None


class ContactExchange:
    """Replay/transport adapter. Quota describes current observed contact only.

    A live transport can implement the same acquire contract; future duration is
    never a proposal feature. Each interrupted request restarts from byte zero.
    """
    def __init__(self, receiver, cache, receiver_ledger, journal=None, catalogue_limit=32):
        self.receiver, self.cache, self.ledger = receiver, cache, receiver_ledger
        self.journal = journal or TransferJournal()
        self.catalogue = {}
        self.sources = {}
        self.connected = set()
        self.sequence = 0
        self.records = []
        self.catalogue_limit = catalogue_limit

    def observe(self, owner, packages, now, quota, source_ledger):
        self.connected.add(owner)
        self.sources[owner] = (dict(packages), quota, source_ledger)
        observed = {}
        for key, wire in sorted(packages.items())[:self.catalogue_limit]:
            p = Package.read(wire)
            if key != p.hash or p.owner != owner or p.available > now or p.created > now:
                continue
            observed[key] = Advertisement(key, owner, now, p.available, len(wire), p.interface, p.ratio)
        message = canonical({"owner": owner, "quota": quota, "versions": [asdict(v) for v in observed.values()]})
        self._transmit(owner, len(message), "catalogue", now, source_ledger)
        self.catalogue.update(observed)

    def disconnect(self, owner):
        self.connected.discard(owner)

    def _transmit(self, owner, size, kind, now, source_ledger):
        # Account request separately as receiver Tx / source Rx.
        sender, receiver = ((self.receiver, owner) if kind == "request" else (owner, self.receiver))
        self.sequence += 1
        tid = f"tx{len(self.journal.records)}:{self.receiver}:{self.sequence}:{sender}:{receiver}"
        start = time.perf_counter()
        self.ledger.guard(Cost(B=size))
        source_ledger.guard(Cost(B=size))
        self.ledger.charge(kind, Cost(B=size, W=time.perf_counter() - start),
                           transfer_id=tid, direction="tx" if kind == "request" else "rx")
        source_ledger.charge("serve_" + kind, Cost(B=size, W=time.perf_counter() - start), transfer_id=tid,
                             direction="rx" if kind == "request" else "tx")
        self.journal.add(tid, sender, receiver, size, kind, now)
        self.records.append({"transfer_id": tid, "time": now, "kind": kind, "bytes": size})

    def acquire(self, key, now):
        if key in self.cache:
            return self.cache.get(key)
        ad = self.catalogue.get(key)
        if ad is None or ad.observed > now or ad.available > now or ad.owner not in self.connected:
            raise ExecutionStopped("No observed available exact version/contact")
        packages, quota, source_ledger = self.sources[ad.owner]
        request = canonical({"hash": key, "requester": self.receiver, "source": ad.owner, "restart_offset": 0})
        self._transmit(ad.owner, len(request), "request", now, source_ledger)
        wire = packages[key]
        sent = min(len(wire), quota, max(0, int(self.ledger.remaining().B)), max(0, int(source_ledger.remaining().B)))
        self._transmit(ad.owner, sent, "payload", now, source_ledger)
        self.sources[ad.owner] = (packages, quota - sent, source_ledger)
        if sent < len(wire):
            self.records.append({"time": now, "hash": key, "kind": "interruption", "bytes": 0,
                                 "status": "interrupted_restart_required"})
            raise ExecutionStopped("Interrupted payload; retry restarts")
        start = time.perf_counter()
        try:
            self.cache.put(key, wire, now)
        finally:
            self.ledger.charge("checksum_interface", Cost(W=time.perf_counter() - start))
        return self.cache.get(key)

    @staticmethod
    def order_requests(requests):
        # One sender services this queue sequentially; callers supply observed requests.
        return sorted(requests, key=lambda r: (r["time"], r["requester"]))
