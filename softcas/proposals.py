"""Bounded legal proposals from a shared scored archive and observed metadata."""
from __future__ import annotations
import random
import hashlib
import json
import math
from dataclasses import asdict, replace
import numpy as np
from .actions import Proposal, vary, TYPES
from .state import Structure, FitSettings, canonical
from .resources import Cost


class ProposalFactory:
    def __init__(self, config, backend, exchange, retained=()):
        self.c, self.backend, self.exchange = config, backend, exchange
        self.rng = random.Random(config["seed"])
        self.retained = tuple(retained)
        self.phase = 0
        self.phase_anchor = None
        self.phase_origin = Cost()
        self.phase_transition = None
        self.generation_index = 0
        self.phase_fit_origin = None
        self.randomization_records = []

    def _stream(self, operation, slot=0):
        """Independent PCG64 stream; never advances another operation's RNG."""
        context = self.c.get("randomization", {})
        key = (self.c["seed"], context.get("dataset", self.c.get("dataset", "unspecified")),
               context.get("detector", getattr(self.backend, "identity", "unspecified")),
               context.get("client", getattr(self.exchange, "receiver", "unspecified")),
               context.get("stage", self.c.get("stage", 0)), operation,
               self.c.get("_attempt_index", self.generation_index), slot)
        seed = int.from_bytes(hashlib.sha256(canonical(key)).digest()[:8], "big")
        self.randomization_records.append({"seed_tuple": key, "seed": seed, "generator": "numpy.PCG64"})
        return np.random.Generator(np.random.PCG64(seed))

    @staticmethod
    def _identity(proposal):
        # Estimates change with resource profiles; construction identity does not.
        fields = asdict(proposal)
        fields.pop("estimate")
        return canonical(fields)

    def _fidelity(self, kind, parent, members=None):
        if kind == "import":
            return 0
        members = parent.C if members is None else members
        configuration = kind in ("fresh", "refine")
        trainable = ((configuration and any(parent.z.mask)) or
                     (bool(members) and self.c["ablation"] != "discrete"))
        if not trainable:
            return 0
        fitting = self.c.get("fitting", {})
        if fitting.get("unit") != "epochs":
            return self.c["phi_steps" if configuration else "beta_steps"]
        batches = self.c.get("_fit_batches_per_epoch")
        remaining = self.c.get("_fit_steps_remaining")
        if not isinstance(batches, int) or isinstance(batches, bool) or batches <= 0:
            raise ValueError("Epoch fidelity requires the stage's positive D batch count")
        if not isinstance(remaining, int) or isinstance(remaining, bool) or remaining < 0:
            raise ValueError("Epoch fidelity requires the remaining stage optimizer-step allowance")
        allowed = remaining
        if self.c["method"] == "serial":
            total = fitting["stage_epochs"] * batches
            first = math.floor(total * self.c["serial"]["fraction"])
            if self.phase == 0:
                allowed = min(allowed, max(0, first - (total - remaining)))
            elif self.phase_fit_origin is not None:
                allowed = min(allowed, max(0, total - first - (self.phase_fit_origin - remaining)))
        epochs = fitting["local_epochs" if configuration else "receiving_epochs"]
        return min(epochs * batches, allowed)

    def _make(self, kind, seed, parent, archive, z=None, h=None, add=None, remove=None,
              imported=None, secondary=None, discrete=None):
        members = (set(parent.C) - {remove}) | ({add} if add else set())
        # A Fresh structure can activate parameters absent in a bare parent.
        fidelity_parent = replace(parent, z=z) if z is not None else parent
        fidelity = self._fidelity(kind, fidelity_parent, members)
        if discrete is not None:
            fidelity = 0
        target = imported if kind == "import" else parent
        estimate = self.estimate(kind, fidelity, add, archive, target, z=z, remove=remove)
        return Proposal(parent.hash, kind, seed, estimate, z, h, add, remove, imported,
                        secondary, fidelity, bool(add and add not in self.exchange.cache), discrete)

    def _requires_fitting(self, proposal, archive):
        if proposal.kind == "import" or proposal.discrete is not None:
            return False
        parent = archive.items[proposal.parent].state
        if proposal.kind in ("fresh", "refine"):
            return bool(any((proposal.z or parent.z).mask) or
                        (parent.C and self.c["ablation"] != "discrete"))
        members = (set(parent.C) - {proposal.remove}) | ({proposal.add} if proposal.add else set())
        return bool(members) and self.c["ablation"] != "discrete"

    def serial_family(self, archive, ledger):
        c = self.c
        order = c["serial"]["order"]
        if self.phase == 0:
            if c.get("fitting", {}).get("unit") == "epochs":
                total = c["fitting"]["stage_epochs"] * c["_fit_batches_per_epoch"]
                first = math.floor(total * c["serial"]["fraction"])
                if total - c["_fit_steps_remaining"] >= first:
                    self.advance_serial(archive, ledger)
            else:
                fractions = c["serial"]["fraction"]
                phase_limit = Cost(*(v * fractions for v in ledger.limits["search"].values()))
                phase_remaining = phase_limit - ledger.used["search"]
                if any(v <= 0 for v, total in zip(phase_remaining.values(), phase_limit.values()) if total > 0):
                    self.advance_serial(archive, ledger)
        return order[self.phase]

    def advance_serial(self, archive, ledger):
        if self.phase == 0:
            self.phase = 1
            self.phase_anchor = archive.champion.state.hash
            self.phase_origin = ledger.used["search"]
            self.phase_fit_origin = self.c.get("_fit_steps_remaining")
            self.phase_transition = {"anchor": self.phase_anchor, "used": self.phase_origin.__dict__}

    def _parent(self, archive, family, rng=None):
        rng = rng or self.rng
        restrict = family if self.c["ablation"] == "no_handoff" else None
        if self.c["method"] == "serial" and self.phase == 1:
            # Second phase starts from the first phase's selected complete state;
            # later second-phase outputs may continue that context.
            eligible = [s for s in archive.items.values() if s.state.hash == self.phase_anchor or
                        __import__("json").loads(s.state.nu).get("serial_phase") == 1]
            if not eligible:
                raise RuntimeError("Serial anchor disappeared")
            pool = sorted(eligible, key=lambda s: s.state.hash)
            pair = [pool[int(rng.integers(len(pool)))] if hasattr(rng, "integers") else rng.choice(pool) for _ in range(2)]
            return sorted(pair, key=lambda s: (-s.utility, s.state.hash))[0]
        if hasattr(rng, "integers"):
            pool = sorted((s for s in archive.items.values() if restrict is None or
                           s.state.hash == archive.incumbent or s.family == restrict), key=lambda s:s.state.hash)
            pair = [pool[int(rng.integers(len(pool)))] for _ in range(2)]
            return min(pair, key=lambda s:(-s.utility, s.state.hash))
        return archive.parent(rng, restrict)

    def _native_variation(self, parent, secondary, slot, fix_z=False, fix_h=False):
        primary = list(parent.z.mask) + [parent.z.ratio, parent.h.lr, parent.h.residual]
        other = list(secondary.z.mask) + [secondary.z.ratio, secondary.h.lr, secondary.h.residual]
        alleles = [[0, 1]] * 6 + [[1/16, 1/8], [1e-4, 3e-4, 1e-3], [1e-4, 1e-3]]
        mutable = ([] if fix_z else list(range(7))) + ([] if fix_h else [7, 8])
        if not mutable:
            raise ValueError("No mutable genes")
        cross = self._stream("variation.crossover", slot)
        genes = primary[:]
        for i in mutable:
            genes[i] = (primary[i], other[i])[int(cross.integers(2))]
        position = self._stream("variation.mutation_position", slot)
        i = mutable[int(position.integers(len(mutable)))]
        options = [v for v in alleles[i] if v != genes[i]]
        allele = self._stream("variation.mutation_allele", slot)
        genes[i] = options[int(allele.integers(len(options)))]
        active = [i for i, bit in enumerate(genes[:6]) if bit]
        repair = self._stream("variation.mask_repair", slot)
        if not active:
            active = [int(repair.integers(6))]
        elif len(active) > 2:
            active = list(repair.choice(active, size=2, replace=False))
        genes[:6] = [int(i in active) for i in range(6)]
        return Structure(tuple(genes[:6]), genes[6]), FitSettings(*genes[7:])

    def estimate(self, kind, steps, add, archive, parent, z=None, remove=None):
        table = self.c["estimates"]
        b = table["control_bytes"] + (self.exchange.catalogue[add].size if add and add not in self.exchange.cache else 0)
        profile = getattr(self.backend, "cost_profile", None)
        if profile:
            members = (set(parent.C) - {remove}) | ({add} if add else set())
            descriptors, branches = [], []
            for key in sorted(members):
                if key in self.exchange.cache:
                    package = self.exchange.cache.get(key)
                    port, ratio = package.interface.port, package.ratio
                else:
                    ad = self.exchange.catalogue[key]
                    port, ratio = ad.interface.port, ad.ratio
                    if ratio is None:
                        raise ValueError("Observed package architecture metadata required for cost admission")
                descriptors.append((port, ratio))
                branches.append((port, ratio, "external"))
            structure = z or parent.z
            branches.extend((i, structure.ratio, "local") for i, enabled in enumerate(structure.mask) if enabled)
            mode = "configuration" if kind in ("fresh", "refine") else "collaboration"
            rule = "discrete" if self.c["ablation"] == "discrete" else "soft"
            signature = self.backend.signature_parts(structure, descriptors, rule, mode)
            return profile.estimate(signature, steps if kind != "import" else 0,
                                    b if add and add not in self.exchange.cache else 0,
                                    descriptors={"mode": mode, "branches": branches})
        return Cost(F=table["evaluation_F"] + steps * table["step_F"], E=1,
                    B=b if add and add not in self.exchange.cache else 0,
                    W=table["evaluation_W"] + steps * table["step_W"])

    def generate(self, archive, ledger):
        c, cache = self.c, self.exchange.cache
        method, ablation = c["method"], c["ablation"]
        native = c.get("proposal_protocol", "legacy") == "native_r1000"
        allowed_family = self.serial_family(archive, ledger) if method == "serial" else None
        raw = []
        for slot in range(c["proposals_per_type"]):
            seed = (int(self._stream("proposal.initialization", slot).integers(2**63)) if native
                    else self.rng.randrange(2**31))
            if allowed_family != "collaboration":
                parent = self._parent(archive, "configuration", self._stream("configuration.primary", slot) if native else None).state
                secondary = self._parent(archive, "configuration", self._stream("configuration.secondary", slot) if native else None).state
                fixed_z = Structure(tuple(c["fixed"]["mask"]), c["fixed"]["ratio"])
                fixed_h = FitSettings(c["fixed"]["lr"], c["fixed"]["residual"])
                if method != "fixed_collaboration":
                    if method == "local":
                        # Only the initial empty local structure needs initialization.
                        z, h = fixed_z, fixed_h
                    else:
                        if native:
                            z, h = self._native_variation(parent, secondary, slot, fix_z=ablation == "hpo_only", fix_h=ablation == "nas_only")
                        else:
                            z, h = vary(parent, secondary, seed, fix_z=ablation == "hpo_only", fix_h=ablation == "nas_only")
                        if ablation == "hpo_only":
                            z = fixed_z
                        if ablation == "nas_only":
                            h = fixed_h
                    if method != "local" or not parent.phi:
                        raw.append(self._make("fresh", seed, parent, archive, z=z, h=h, secondary=secondary.hash))
                if (any(parent.z.mask) or (parent.beta and ablation != "discrete")) and ablation != "no_refine":
                    raw.append(self._make("refine", seed, parent, archive))
            if method in ("local", "local_nas_hpo") or allowed_family == "configuration":
                continue
            parent = self._parent(archive, "collaboration", self._stream("collaboration.primary", slot) if native else None).state
            known = sorted(set(cache.keys()) | set(self.exchange.catalogue))
            for key in known:
                ad = self.exchange.catalogue.get(key)
                spec = cache.get(key).interface if key in cache else ad.interface
                if spec not in self.backend.specs or key in parent.C:
                    continue
                if key not in cache and ad.owner not in self.exchange.connected:
                    continue
                at_port = [k for k in parent.C if cache.get(k).interface.port == spec.port]
                if len(at_port) < 2 and len(parent.C) < c["max_external_paths"]:
                    raw.append(self._make("add", seed, parent, archive, add=key))
                for old in at_port:
                    raw.append(self._make("replace", seed, parent, archive, add=key, remove=old))
            for key in parent.C:
                raw.append(self._make("remove", seed, parent, archive, remove=key))
            for saved in self.retained:
                if all(k in cache for k in saved.C):
                    raw.append(self._make("import", seed, parent, archive, imported=saved))
        # Restrict per scheduling TYPE, not per membership edit or construction mode.
        if ablation == "discrete":
            expanded = []
            import json
            for a in raw:
                parent = archive.items[a.parent].state
                choices = json.loads(parent.nu).get("discrete_choices", {})
                if a.kind in ("add", "replace", "remove"):
                    port = (cache.get(a.add).interface.port if a.add in cache else self.exchange.catalogue[a.add].interface.port) if a.add else cache.get(a.remove).interface.port
                    members = (set(parent.C) - {a.remove}) | ({a.add} if a.add else set())
                    members_here = [k for k in sorted(members) if
                                    (cache.get(k).interface if k in cache else self.exchange.catalogue[k].interface).port == port]
                    for selected in [None] + members_here:
                        new = {**choices, str(port): selected}
                        expanded.append(self._make(a.kind, a.seed, parent, archive, add=a.add, remove=a.remove,
                                                   discrete=tuple(sorted(new.items()))))
                else:
                    expanded.append(a)
            raw = expanded
        if c.get("fitting", {}).get("unit") == "epochs":
            # Budget exhaustion cannot turn a trainable operation into an
            # untrained structural trial. Inherently zero-fit edits stay legal.
            raw = [a for a in raw if a.fidelity > 0 or not self._requires_fitting(a, archive)]
        result = []
        remaining = ledger.remaining()
        epoch_mode = c.get("fitting", {}).get("unit") == "epochs"
        if method == "serial" and self.phase == 0 and not epoch_mode:
            phase_cap = Cost(*(v * c["serial"]["fraction"] for v in ledger.limits["search"].values()))
            phase_remaining = phase_cap - ledger.used["search"]
            remaining = Cost(*(min(a, b) for a, b in zip(remaining.values(), phase_remaining.values())))
        for t in TYPES:
            # Affordability precedes the bounded draw, so rejected expensive
            # proposals cannot crowd valid cheaper candidates out of a pool.
            pool = [a for a in raw if a.type == t and a.estimate.fits(remaining)]
            if native:
                unique = {self._identity(a): a for a in pool}
                pool = [unique[key] for key in sorted(unique)]
                order = self._stream("pool.permutation." + t).permutation(len(pool))
                pool = [pool[int(i)] for i in order]
            else:
                pool.sort(key=lambda a: a.hash)
                self.rng.shuffle(pool)
            result.extend(pool[:c["proposals_per_type"]])
        if method == "serial" and self.phase == 0:
            if not epoch_mode:
                phase_cap = Cost(*(v * c["serial"]["fraction"] for v in ledger.limits["search"].values()))
                remaining = phase_cap - ledger.used["search"]
                result = [a for a in result if a.estimate.fits(remaining)]
            if not result:
                self.advance_serial(archive, ledger)
                return self.generate(archive, ledger)
        self.generation_index += 1
        return result
