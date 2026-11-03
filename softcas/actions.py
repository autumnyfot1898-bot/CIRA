"""Recorded proposals and exact state inheritance, separate from execution."""
from __future__ import annotations
import random
from dataclasses import dataclass, asdict, replace
from .state import Structure, FitSettings, State, digest
from .resources import Cost

TYPES = ("configuration", "cached", "acquisition", "import")


@dataclass(frozen=True)
class Proposal:
    parent: str
    kind: str
    seed: int
    estimate: Cost
    z: Structure | None = None
    h: FitSettings | None = None
    add: str | None = None
    remove: str | None = None
    imported: State | None = None
    secondary: str | None = None
    fidelity: int = 0
    acquisition: bool = False
    discrete: tuple[tuple[int, str], ...] | None = None

    @property
    def type(self):
        if self.kind in ("fresh", "refine"):
            return "configuration"
        if self.kind == "import":
            return "import"
        return "acquisition" if self.acquisition else "cached"

    @property
    def family(self):
        if self.kind == "import" and self.imported is not None:
            return __import__("json").loads(self.imported.nu).get("family", "import")
        return "configuration" if self.kind in ("fresh", "refine") else "collaboration"

    @property
    def hash(self):
        return digest(asdict(self))

    def tie_key(self, type_order=TYPES):
        return (type_order.index(self.type), self.add or self.remove or "", self.parent, self.hash)


def vary(primary, secondary, seed, fix_z=False, fix_h=False):
    rng = random.Random(seed)
    p = list(primary.z.mask) + [primary.z.ratio, primary.h.lr, primary.h.residual]
    q = list(secondary.z.mask) + [secondary.z.ratio, secondary.h.lr, secondary.h.residual]
    choices = [[0, 1]] * 6 + [[1 / 16, 1 / 8], [1e-4, 3e-4, 1e-3], [1e-4, 1e-3]]
    genes = [rng.choice((a, b)) for a, b in zip(p, q)]
    mutable = list(range(7) if not fix_z else []) + ([7, 8] if not fix_h else [])
    if not mutable:
        raise ValueError("No mutable genes")
    j = rng.choice(mutable)
    genes[j] = rng.choice([v for v in choices[j] if v != genes[j]])
    if fix_z:
        genes[:7] = p[:7]
    if fix_h:
        genes[7:] = p[7:]
    active = [i for i, m in enumerate(genes[:6]) if m]
    rng.shuffle(active)
    if not active:
        active = [rng.randrange(6)]
    genes[:6] = [int(i in active[:2]) for i in range(6)]
    return Structure(tuple(genes[:6]), genes[6]), FitSettings(*genes[7:])


def construct(parent, proposal, cache):
    if proposal.parent != parent.hash:
        raise ValueError("Primary parent hash mismatch")
    if proposal.kind == "import":
        if proposal.imported is None:
            raise ValueError("Missing complete checkpoint")
        return proposal.imported  # immutable, exact: no lineage rewriting
    if proposal.kind == "fresh":
        if proposal.z is None or proposal.h is None or sum(proposal.z.mask) not in (1, 2):
            raise ValueError("Fresh must replay legal recorded genes")
        return replace(parent, z=proposal.z, h=proposal.h, phi=())
    if proposal.kind == "refine":
        if not parent.phi and not parent.beta:
            raise ValueError("Refine requires trainable variables")
        return parent
    if proposal.kind not in ("add", "remove", "replace", "discrete"):
        raise ValueError(proposal.kind)
    members = set(parent.C)
    if proposal.remove:
        if proposal.remove not in members:
            raise ValueError("Removed member absent")
        members.remove(proposal.remove)
    if proposal.add:
        if proposal.add in members:
            raise ValueError("Added member already selected")
        if proposal.remove and cache.get(proposal.add).interface.port != cache.get(proposal.remove).interface.port:
            raise ValueError("Replace must use same port")
        members.add(proposal.add)
    if proposal.kind == "add" and (not proposal.add or proposal.remove):
        raise ValueError("Invalid Add")
    if proposal.kind == "remove" and (not proposal.remove or proposal.add):
        raise ValueError("Invalid Remove")
    if proposal.kind == "replace" and (not proposal.add or not proposal.remove):
        raise ValueError("Invalid Replace")
    for port in range(6):
        if sum(cache.get(k).interface.port == port for k in members) > 2:
            raise ValueError("Two external paths per port")
    logits = dict(parent.beta)
    return replace(parent, C=tuple(sorted(members)), beta=tuple((k, logits.get(k, 0.0)) for k in sorted(members)))
