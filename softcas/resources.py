"""Actual ledger with separate release reserve and unique transfer accounting."""
from __future__ import annotations
import math
import time
import threading
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Cost:
    F: float = 0.0
    E: float = 0.0
    B: float = 0.0
    W: float = 0.0

    def __post_init__(self):
        if any(not math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Nonfinite cost")

    def __add__(self, other):
        return Cost(*(a + b for a, b in zip(self.values(), other.values())))

    def __sub__(self, other):
        return Cost(*(a - b for a, b in zip(self.values(), other.values())))

    def values(self):
        return (self.F, self.E, self.B, self.W)

    def fits(self, remaining):
        return all(c >= 0 and c <= r for c, r in zip(self.values(), remaining.values()))


class ExecutionStopped(RuntimeError):
    pass


class Ledger:
    def __init__(self, search: Cost, release: Cost, enforce=("E", "B", "W")):
        self.limits = {"search": search, "release": release}
        self.used = {"search": Cost(), "release": Cost()}
        self.events = []
        self.enforce = tuple(enforce)
        self.lock = threading.RLock()
        self.runtime = None
        self.meter_policy = None
        self.wall_anchors = {}

    def begin(self, scope="search"):
        with self.lock:
            self.wall_anchors[scope] = (time.monotonic(), self.used[scope].W)

    def finish(self, scope="search"):
        with self.lock:
            anchor = self.wall_anchors.pop(scope, None)
            if anchor is not None:
                gap = max(0, time.monotonic() - anchor[0] + anchor[1] - self.used[scope].W)
                if gap:
                    self.charge("wall_clock_unbilled", Cost(W=gap), scope)

    def remaining(self, scope="search"):
        with self.lock:
            remaining = self.limits[scope] - self.used[scope]
            if scope in self.wall_anchors:
                anchor, base_used = self.wall_anchors[scope]
                elapsed = base_used + time.monotonic() - anchor
                remaining = Cost(remaining.F, remaining.E, remaining.B,
                                 self.limits[scope].W - max(self.used[scope].W, elapsed))
            return remaining

    def guard(self, next_cost=Cost(), scope="search"):
        remaining = self.remaining(scope)
        if any(getattr(next_cost, r) > getattr(remaining, r) for r in self.enforce):
            raise ExecutionStopped("Execution allowance/deadline reached")

    def charge(self, reason, cost: Cost, scope="search", **metadata):
        if any(v < 0 for v in cost.values()):
            raise ValueError("Negative charge")
        with self.lock:
            self.used[scope] = self.used[scope] + cost
            self.events.append({"reason": reason, "scope": scope, "cost": asdict(cost),
                                "time": time.time(), **metadata})

    def violations(self):
        return {scope: [r for r in ("F", "E", "B", "W")
                        if getattr(self.used[scope], r) > getattr(limit, r)]
                for scope, limit in self.limits.items()}


class TransferJournal:
    """Network count counts sender Tx once; node ledgers independently keep Tx/Rx."""
    def __init__(self):
        self.records = {}

    def add(self, transfer_id, sender, receiver, size, kind, timestamp):
        record = dict(sender=sender, receiver=receiver, bytes=size, kind=kind, time=timestamp)
        if transfer_id in self.records:
            if self.records[transfer_id] != record:
                raise ValueError("Transfer ID reused for different transmission")
            return
        self.records[transfer_id] = record

    @property
    def network_bytes(self):
        return sum(r["bytes"] for r in self.records.values())
