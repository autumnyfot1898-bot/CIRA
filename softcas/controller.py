"""eq:reward/ridge/admissible_actions/priority/action_selection."""
from __future__ import annotations
import math
import random
from dataclasses import asdict, dataclass
import numpy as np
from .actions import TYPES


@dataclass(frozen=True)
class Scored:
    state: object
    ap: float
    utility: float
    family: str


class Archive:
    def __init__(self, incumbent, ap, capacity, cache, truncation="utility_hash", tournament_tie="hash", state_store=None):
        if capacity < 2 or truncation != "utility_hash" or tournament_tie != "hash":
            raise ValueError("Inject/configure supported archive and tournament strategies explicitly")
        self.incumbent = incumbent.hash
        self.capacity, self.cache = capacity, cache
        self.state_store = state_store
        self.items = {incumbent.hash: Scored(incumbent, ap, 0.0, "incumbent")}
        self._pin()

    @property
    def champion(self):
        return sorted(self.items.values(), key=lambda s: (-s.utility, s.state.hash))[0]

    def _pin(self):
        self.cache.pin("archive", [k for s in self.items.values() for k in s.state.C])
        if self.state_store:
            self.state_store.pin("archive", [s.state for s in self.items.values()])

    def insert(self, scored):
        if scored.state.hash in self.items:
            # Exact re-import cannot rewrite an already scored state's family;
            # in particular X0 must retain universal parent eligibility.
            return
        previous = dict(self.items)
        self.items[scored.state.hash] = scored
        protected = {self.incumbent, self.champion.state.hash}
        best = sorted((s for k, s in self.items.items() if k not in protected),
                      key=lambda s: (-s.utility, s.state.hash))
        retain = protected | {s.state.hash for s in best[:self.capacity - len(protected)]}
        self.items = {k: v for k, v in self.items.items() if k in retain}
        try:
            self._pin()
        except (RuntimeError, OSError):
            self.items = previous
            self._pin()
            raise

    def parent(self, rng, family=None):
        pool = sorted((s for s in self.items.values() if family is None or s.state.hash == self.incumbent or s.family == family),
                      key=lambda s: s.state.hash)
        # Seeded size-two tournament with replacement; tie is complete-state hash.
        pair = [rng.choice(pool), rng.choice(pool)]
        return sorted(pair, key=lambda s: (-s.utility, s.state.hash))[0]


class Ridge:
    def __init__(self, dim, regularization):
        if (not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 or
                not math.isfinite(regularization) or regularization <= 0):
            raise ValueError("lambda > 0")
        self.L = np.eye(dim, dtype=np.float64) * regularization
        self.b = np.zeros(dim, dtype=np.float64)
        self.n = 0
        self.flops = 0

    def predict(self, feature):
        f = np.asarray(feature, dtype=np.float64)
        if f.shape != self.b.shape or not np.isfinite(f).all():
            raise ValueError("Invalid pre-execution feature vector")
        mu = float(f @ np.linalg.solve(self.L, self.b))
        s = math.sqrt(max(0.0, float(f @ np.linalg.solve(self.L, f))))
        d = len(f)
        lu = d * (d - 1) * (2 * d - 1) // 3 + d * (d - 1) // 2
        triangular = 2 * d * d - d
        self.flops += 2 * (lu + triangular) + 4 * d - 1
        return mu, s

    def update(self, feature, target):
        f = np.array(feature, dtype=np.float64, copy=True)
        if f.shape != self.b.shape or not np.isfinite(f).all() or not math.isfinite(target):
            raise ValueError("Invalid saved feature/target")
        self.L += np.outer(f, f)
        self.b += f * target
        self.n += 1
        self.flops += 2 * len(f)**2 + 2 * len(f)


def priority(parent_utility, champion_utility, mu, coverage, estimate, remaining, kappa, epsilon):
    if kappa < 0 or epsilon <= 0:
        raise ValueError("Invalid priority constants")
    if not estimate.fits(remaining):
        return 0.0
    fractions = [0.0 if c == 0 else c / r for c, r in zip(estimate.values(), remaining.values())]
    return max(0, parent_utility + mu + kappa * coverage - champion_utility) / (epsilon + max(fractions))


class FeatureEncoder:
    """Explicit FIXTURE encoding, replace/lock on independent development.

    No child result, actual cost, or future contact duration is accepted here.
    """
    def __init__(self, fields, scales):
        self.fields, self.scales = tuple(fields), dict(scales)
        if (not fields or len(set(fields)) != len(fields) or set(fields) != set(scales) or
                any(not math.isfinite(v) or v <= 0 for v in scales.values())):
            raise ValueError("Every ordered field requires a positive scale")

    def encode(self, proposal, scored, remaining, cache, catalogue):
        parent = scored.state
        meta = catalogue.get(proposal.add) if proposal.add else None
        package_bytes = 0
        if proposal.add is not None:
            if proposal.add in cache:
                package_bytes = len(cache.get(proposal.add).wire)
            elif meta is not None:
                package_bytes = meta.size
            else:
                raise ValueError("Requested exact version requires observable size metadata")
        values = {"bias": 1.0, **{"type_" + t: float(proposal.type == t) for t in TYPES},
                  **{"kind_" + t: float(proposal.kind == t) for t in ("fresh", "refine", "add", "remove", "replace", "import", "discrete")},
                  "parent_utility": scored.utility, "members": len(parent.C),
                  "beta_sum": sum(v for _, v in parent.beta),
                  "fit_steps": __import__("json").loads(parent.nu).get("fit_steps", 0),
                  "fidelity": proposal.fidelity, "package_bytes": package_bytes,
                  "package_missing": float(proposal.add is not None and proposal.add not in cache),
                  "local_ports": sum((proposal.z or parent.z).mask),
                  "ratio": (proposal.z or parent.z).ratio,
                  **{"remaining_" + r: getattr(remaining, r) for r in ("F", "E", "B", "W")}}
        result = tuple(values[name] / self.scales[name] for name in self.fields)
        if not all(math.isfinite(value) for value in result):
            raise ValueError("Nonfinite pre-execution feature")
        return result


class Selector:
    def __init__(self, cold_order, type_order, kappa, epsilon, policy="softcas", no_gain=False,
                 proposal_protocol="legacy"):
        if (len(cold_order) != len(TYPES) or len(type_order) != len(TYPES) or
                set(cold_order) != set(TYPES) or set(type_order) != set(TYPES)):
            raise ValueError("Explicit scheduling enumeration/order required")
        if proposal_protocol not in ("legacy", "native_r1000"):
            raise ValueError("Unknown proposal protocol")
        self.cold_order, self.type_order = tuple(cold_order), tuple(type_order)
        self.kappa, self.epsilon, self.policy, self.no_gain = kappa, epsilon, policy, no_gain
        self.proposal_protocol = proposal_protocol
        self.cold_index = 0
        self.observed = set()
        self.local_fresh_offered = False
        self.cycle = 0
        self.flops = 0
        self.last_pool = []

    def choose(self, proposals, archive, ridge, encoder, remaining, cache, catalogue):
        # A read-only trace of exactly the bounded, affordable pool scored here.
        # Replays must not run the predictor again or evaluate unselected children.
        self.last_pool = []
        admitted = sorted((a for a in proposals if a.estimate.fits(remaining)),
                          key=lambda a: a.tie_key(self.type_order))
        ranked = []
        for a in admitted:
            f = encoder.encode(a, archive.items[a.parent], remaining, cache, catalogue)
            mu, s = ridge.predict(f)
            J = priority(archive.items[a.parent].utility, archive.champion.utility,
                         0.0 if self.no_gain else mu, s, a.estimate, remaining, self.kappa, self.epsilon)
            self.flops += len(f) + len(archive.items[a.parent].state.beta)
            self.flops += 6 + sum(c > 0 for c in a.estimate.values())
            ranked.append((a, f, mu, s, J))
            self.last_pool.append({
                "proposal": a.hash, "kind": a.kind, "type": a.type,
                "parent": a.parent, "parent_utility": archive.items[a.parent].utility,
                "champion_utility": archive.champion.utility,
                "features": list(f), "prediction": mu, "coverage": s,
                "estimate": dict(zip(("F", "E", "B", "W"), a.estimate.values())),
                "remaining": dict(zip(("F", "E", "B", "W"), remaining.values())),
                "priority": J, "tie_rank": len(self.last_pool),
                "z": asdict(a.z) if a.z else None,
                "h": asdict(a.h) if a.h else None,
                "add": a.add, "remove": a.remove, "secondary": a.secondary,
                "fidelity": a.fidelity,
            })
        # Local-only Fresh uses a scored local-only parent; no hidden C removal.
        if not self.local_fresh_offered:
            local = [r for r in ranked if r[0].kind == "fresh" and not archive.items[r[0].parent].state.C]
            if self.proposal_protocol == "native_r1000":
                self.local_fresh_offered = True
            if local:
                self.local_fresh_offered = True
                return local[0], "cold_local_fresh"
        if self.proposal_protocol == "native_r1000":
            while self.cold_index < len(self.cold_order):
                t = self.cold_order[self.cold_index]
                self.cold_index += 1
                if t not in self.observed:
                    matches = [r for r in ranked if r[0].type == t]
                    if matches:
                        return matches[0], "cold_type"
        else:
            for t in self.cold_order:
                if t not in self.observed:
                    matches = [r for r in ranked if r[0].type == t]
                    if matches:
                        return matches[0], "cold_type"
        if self.policy == "round_robin":
            for _ in self.type_order:
                t = self.type_order[self.cycle % len(self.type_order)]
                self.cycle += 1
                matches = [r for r in ranked if r[0].type == t]
                if matches:
                    return matches[0], "round_robin"
            return None, "keep"
        positive = [r for r in ranked if r[4] > 0]
        return (sorted(positive, key=lambda r: (-r[4], r[0].tie_key(self.type_order)))[0], "priority") if positive else (None, "keep")
