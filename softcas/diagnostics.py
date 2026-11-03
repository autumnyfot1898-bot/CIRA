"""Two-structure x two-context diagnostic with fixed seeds, fidelity and weights."""
from __future__ import annotations
import math
from dataclasses import replace
from pathlib import Path
from .state import Structure, FitSettings, State, Package, canonical
from .engine import initial_state
from .backend import work
from .resources import Cost, Ledger
from .storage import atomic_write


def configuration_composition(backend, cache, D, V, T, spec, algorithm, output):
    if D.role != "D" or V.role != "V" or T.role != "T" or len(spec["structures"]) != 2:
        raise ValueError("Diagnostic requires two locked structures and role-separated D/V/T")
    if spec.get("development_record_sha256") is None:
        raise ValueError("Lock the structures/context/coefficients on independent development")
    C = []
    for file in spec["peer_packages"]:
        package = Package.read(Path(file).read_bytes())
        if package.created > spec["snapshot_time"] or package.available > spec["snapshot_time"]:
            raise ValueError("Diagnostic donor version was unavailable at the declared snapshot")
        cache.put(package.hash, package.wire)
        C.append(package.hash)
    if not C:
        raise ValueError("The registered peer context requires at least one valid independently produced donor version")
    alphas = spec["receiving_coefficients"]
    betas = {}
    for port in range(6):
        keys = [k for k in C if cache.get(k).interface.port == port]
        if len(keys) > 2:
            raise ValueError("Diagnostic composition exceeds shared path cap")
        null = 1 - sum(alphas[k] for k in keys)
        if null <= 0 or any(alphas[k] <= 0 for k in keys):
            raise ValueError("Fixed soft coefficients must have positive null mass")
        betas.update({k: math.log(alphas[k] / null) for k in keys})
    h = FitSettings(spec["h"]["lr"], spec["h"]["residual"])
    ledger = Ledger(Cost(**spec["budget"]), Cost(**spec["budget"]))
    ledger.runtime, ledger.meter_policy = getattr(backend, "runtime", None), algorithm.get("meter")
    states, scores = {}, {}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for context in ("empty", "peer"):
        for i, structure in enumerate(spec["structures"]):
            z = Structure(tuple(structure["mask"]), structure["ratio"])
            state = initial_state(backend, cache, z, h, spec["initialization_seed"])
            if context == "peer":
                state = replace(state, C=tuple(sorted(C)), beta=tuple(sorted(betas.items())))
            with work(ledger, "diagnostic_fresh_construct"):
                model = backend.assembly(state, cache, spec["initialization_seed"], "fresh")
                for port in model.ports.values():
                    port.beta.requires_grad_(False)
                optimizer = backend.optimizer(model, state, algorithm)
                scaler = backend.new_scaler()
            for step in range(spec["fit_steps"]):
                with work(ledger, "diagnostic_fit"):
                    optimizer.zero_grad(set_to_none=True)
                    backend.fit_batch(model, D.batches[step % len(D.batches)], h.residual, optimizer, scaler)
            state = model.snapshot(state).lineage(diagnostic=True, context=context, structure=i,
                    seed=spec["initialization_seed"], fit_steps=spec["fit_steps"], beta_fixed=True, fit_data=D.identity)
            with work(ledger, "diagnostic_V"):
                v = backend.score(model, V)
            states[(context, i)] = state
            scores[(context, i)] = {"V": v}
            state.save(output / (context + "_" + str(i) + ".json"))
    # Lock choices on V before accessing a single T score.
    chosen = {context: min(range(2), key=lambda i: (-scores[(context, i)]["V"], i)) for context in ("empty", "peer")}
    atomic_write(output / "V_choices.json", canonical(chosen))
    consequences = {}
    for target in ("empty", "peer"):
        for i in range(2):
            model = backend.assembly(states[(target, i)], cache)
            with work(ledger, "diagnostic_independent_T"):
                scores[(target, i)]["T"] = backend.score(model, T)
        other = "peer" if target == "empty" else "empty"
        consequences[target] = scores[(target, chosen[target])]["T"] - scores[(target, chosen[other])]["T"]
    result = {"kind": "configuration_composition_diagnostic", "purpose": algorithm.get("purpose"),
              "formal": algorithm.get("purpose") == "FORMAL", "metric": backend.metric,
              "V_choices": chosen, "T_same_context_choice_effect": consequences,
              "cells": [{"context": c, "structure": i, "state": states[(c, i)].hash, **s} for (c, i), s in scores.items()],
              "ledger": ledger.events, "actual": __import__("dataclasses").asdict(ledger.used["search"]), "spec": spec,
              "seed_unit": "aggregate receiving-client effects within the same trajectory seed"}
    atomic_write(output / "diagnostic.json", canonical(result))
    return result


def run_diagnostic_job(spec):
    import json
    from .config import load_config, effective_formal
    from .backend import YoloBackend
    from .cache import PackageCache
    from .data import load_detection_split
    from .metering import CostProfile
    from dataclasses import asdict
    config = load_config(spec["config"], spec.get("formal", False), overrides=spec.get("overrides"))
    formal = effective_formal(config, spec.get("formal", False))
    interface = json.loads(Path(spec["detector_config"]).read_bytes())
    backend = YoloBackend(spec["checkpoint"], interface["ports"], interface["labels"], interface["preprocessing"],
                          interface["image_size"], config["runtime"])
    if config.get("cost_profile"):
        backend.cost_profile = CostProfile(config["cost_profile"], backend.runtime.identity(), formal)
        profile = backend.cost_profile.data
        if (profile.get("base_sha256") != backend.identity or profile.get("execution") != backend.execution or
                canonical(profile.get("port_interfaces")) != canonical([asdict(p) for p in backend.specs])):
            raise ValueError("Cost calibration must use this exact frozen base and declared six ports")
    cache = PackageCache(config["cache_bytes"], backend.specs)
    splits = [load_detection_split(spec["registry"], spec["client"], spec["stage"], role, interface["image_size"],
                                  interface["labels"], formal=formal, batch_size=config["fit_batch_size"] if role == "D" else config["evaluation_batch_size"],
                                  domain=spec["domain"] if role == "T" else None) for role in ("D", "V", "T")]
    return configuration_composition(backend, cache, *splits, spec["diagnostic"], config, spec["output"])
