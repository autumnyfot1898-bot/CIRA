"""Deduplicated native study registration, not an execution/performance claim.

Each analysis view references a canonical complete federation trajectory. RGB
endpoint evaluations are registered separately and do not create training runs.
Unbound assets and unimplemented literature adapters remain explicit blockers.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import math
from collections import Counter

from .state import canonical

INTERNAL = ("local", "local_nas_hpo", "fixed_collaboration", "serial", "round_robin", "softcas")
EXTERNAL = ("DPFL", "pFedDIL", "DA-DPFL", "DeModCL", "DCFCL", "dFLMoE", "PerFedSDE-NAS", "DSpodFL")
METHODS = INTERNAL + EXTERNAL
ABLATIONS = ("nas_only", "hpo_only", "no_refine", "no_handoff", "discrete", "no_gain")
SERIAL_RULES = ("cfg25_collab75", "cfg50_collab50", "cfg75_collab25", "collab50_cfg50")
GRID = {
    "Q0": dict(ridge_lambda=1., kappa=.01, beta_lr=.01, attempts=64),
    "Q1": dict(ridge_lambda=.1, kappa=.01, beta_lr=.01, attempts=64),
    "Q2": dict(ridge_lambda=10., kappa=.01, beta_lr=.01, attempts=64),
    "Q3": dict(ridge_lambda=1., kappa=0., beta_lr=.01, attempts=64),
    "Q4": dict(ridge_lambda=1., kappa=.1, beta_lr=.01, attempts=64),
    "Q5": dict(ridge_lambda=1., kappa=.01, beta_lr=.003, attempts=64),
    "Q6": dict(ridge_lambda=1., kappa=.01, beta_lr=.03, attempts=64),
    "Q7": dict(ridge_lambda=1., kappa=.01, beta_lr=.01, attempts=32),
    "Q8": dict(ridge_lambda=1., kappa=.01, beta_lr=.01, attempts=48),
}


def _distinct(values, name):
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{name} must be nonempty and unique")
    return tuple(values)


def build_study(spec):
    datasets = _distinct(spec["datasets"], "datasets")
    detectors = _distinct(spec["detectors"], "detectors")
    seeds = _distinct(spec["seeds"], "seeds")
    if any(not isinstance(s, int) or isinstance(s, bool) for s in seeds):
        raise ValueError("Use explicit integer trajectory seeds")
    selected_q = spec["development_selection"]["controller_q"]
    if selected_q not in tuple(GRID)[:7]:
        raise ValueError("Controller development selection must be Q0--Q6")
    serial = spec["development_selection"]["serial_rules"]
    cells = list(itertools.product(datasets, detectors))
    if set(serial) != {d + "/" + p for d, p in cells} or any(r not in SERIAL_RULES for r in serial.values()):
        raise ValueError("Bind one prespecified Serial rule per dataset/detector cell")
    selection_locked = spec["development_selection"].get("locked", False)
    if type(selection_locked) is not bool:
        raise ValueError("Development locked must be a boolean")
    lock_hash = spec["development_selection"].get("sha256")
    if selection_locked and (not isinstance(lock_hash, str) or len(lock_hash) != 64 or
                             any(c not in "0123456789abcdef" for c in lock_hash)):
        raise ValueError("A locked selection requires its manifest SHA-256")
    trajectories, views, blocks = {}, [], []
    baseline = dict(rounds=1000, compute=1., data_fraction=1., alpha=1.,
                    order="independent", contacts="normal", donors="normal")

    def add(block, methods, levels, ablations=("none",)):
        before, start = len(trajectories), len(views)
        for (dataset, detector), seed, method, level, ablation in itertools.product(cells, seeds, methods, levels, ablations):
            factors = {**baseline, **level}
            if method == "softcas":
                factors.setdefault("controller_q", selected_q)
                factors["controller_parameters"] = dict(GRID[factors["controller_q"]])
            if method == "serial":
                factors.setdefault("serial_rule", serial[dataset + "/" + detector])
            # Local neither requests nor inherits peers, hence donor conditions
            # are exact anchor views rather than duplicate independent runs.
            if method == "local":
                factors["donors"] = "normal"
            recipe = dict(dataset=dataset, detector=detector, seed=seed,
                          method=method, ablation=ablation, factors=factors)
            identity = hashlib.sha256(canonical(recipe)).hexdigest()
            if identity not in trajectories:
                blockers = ["native input/calibration/topology bindings not verified"]
                if not selection_locked:
                    blockers.append("development selection is an unlocked example")
                if method in EXTERNAL:
                    blockers.append("external native adapter and qualification not implemented")
                trajectories[identity] = dict(id=identity, recipe=recipe,
                    status="planned_not_run", runnable=False, blockers=blockers,
                    first_block=block)
            views.append(dict(id=f"{block}:{len(views)-start+1}", block=block,
                              trajectory_id=identity, requested_level=copy.deepcopy(level),
                              alias=trajectories[identity]["first_block"] != block))
        blocks.append(dict(block=block, new_trajectories=len(trajectories)-before,
                           analysis_views=len(views)-start))

    add("N01", METHODS, [{}])
    add("N02", ("softcas",), [{}], ("none",) + ABLATIONS)
    add("N03", ("softcas",), [dict(controller_q=q) for q in GRID])
    add("N04", ("softcas", "serial", "fixed_collaboration", "local_nas_hpo", "DeModCL", "DSpodFL"),
        [dict(compute=f) for f in (.5, 1., 2.)])
    add("N05", ("softcas", "serial"), [dict(data_fraction=f) for f in (.5, .75, 1.)])
    add("N06", ("softcas", "serial", "fixed_collaboration", "DeModCL", "DCFCL"),
        [dict(alpha=a) for a in (.1, .3, 1., "equal")])
    add("N07", ("softcas", "serial", "DeModCL", "DCFCL"), [dict(order=o) for o in ("independent", "common")])
    add("N08", ("softcas", "serial", "DeModCL", "DCFCL", "DSpodFL"),
        [dict(contacts=c) for c in ("normal", "half_visibility", "one_stage2_half_payload")])
    add("N09", ("softcas", "serial", "fixed_collaboration", "local"),
        [dict(donors=d) for d in ("normal", "empty", "weak")])
    add("N10", ("softcas", "serial", "fixed_collaboration", "DeModCL"),
        [dict(alpha=a, compute=f) for a, f in itertools.product((.1, 1.), (.5, 1.))])
    add("N11", ("serial",), [dict(serial_rule=r) for r in SERIAL_RULES])
    add("N12", METHODS, [dict(rounds=r) for r in (300, 500, 1000)])
    rgb = [dict(trajectory_id=row["trajectory_id"], sigma=sigma,
                status="planned_not_evaluated", extra_training=False)
           for row in views if row["block"] == "N01" for sigma in (0., .01, .03)]
    return dict(schema="cira.native_study.v1", purpose="PLAN_ONLY_NOT_EXECUTED",
                datasets=list(datasets), detectors=list(detectors), seeds=list(seeds),
                independent_unit="complete federation trajectory",
                development_selection=copy.deepcopy(spec["development_selection"]),
                blocks=blocks, unique_trajectories=len(trajectories), analysis_views=len(views),
                trajectories=list(trajectories.values()), views=views, rgb_endpoint_views=rgb,
                extra_rgb_training=0, external_adapters_implemented=[],
                binding_contract={
                    "compute": "bind calibrated F limits; do not scale actual work labels",
                    "data_fraction": "bind nested whole-scene D registries, preserve V/A/T",
                    "alpha": "bind achieved whole-scene assignments and class histograms",
                    "order": "bind causal per-client order and stage events",
                    "contacts": "bind actual exogenous opportunity mask or one stage-two payload interruption",
                    "donors": "bind all catalogue/cache/inherited-member restrictions or DEV-locked weak donors",
                    "rounds": "independent complete four-stage 300/500/1000 runs; not truncation of test outputs",
                    "runtime": "bind bases, ports, features, cost profiles, hardware/software, topology and DEV locks"})


def summarize_completed(study, results, metric="seen"):
    """Join explicit trajectory identities; aliases never increase sample n."""
    known = {}
    for row in study["trajectories"]:
        identity = row["id"]
        if identity in known or hashlib.sha256(canonical(row["recipe"])).hexdigest() != identity:
            raise ValueError("Study registry contains a duplicate or corrupted recipe identity")
        known[identity] = row
    if len(known) != study["unique_trajectories"]:
        raise ValueError("Study registry trajectory count does not match its records")
    if any(v["trajectory_id"] not in known for v in study["views"]):
        raise ValueError("Study view references an unknown trajectory")
    completed = {}
    for row in results:
        identity = row["study_trajectory_id"]
        if identity not in known or identity in completed:
            raise ValueError("Unknown or duplicate trajectory result identity")
        recipe = known[identity]["recipe"]
        if row.get("study_recipe") != recipe:
            raise ValueError("Result recipe must match the registered complete condition")
        value = row.get(metric)
        valid = (row.get("complete") is True and isinstance(value, (int, float)) and
                 not isinstance(value, bool) and math.isfinite(value))
        if row.get('metric') is not None and row['metric']!='AP50:95':
            valid=False
        if valid and metric in ("seen", "current", "forget", "minimum_client_seen", "ap50", "ap75") and not 0 <= value <= 1:
            raise ValueError("AP/forget endpoints must use fractional units [0,1]")
        if valid and metric == "bwt" and not -1 <= value <= 1:
            raise ValueError("BWT must use fractional units [-1,1]")
        completed[identity] = value if valid else None
    block_rows = []
    for block in study["blocks"]:
        ids = {v["trajectory_id"] for v in study["views"] if v["block"] == block["block"]}
        block_rows.append(dict(block=block["block"], distinct_trajectories=len(ids),
            complete=sum(completed.get(i) is not None for i in ids),
            missing=sum(completed.get(i) is None for i in ids)))
    return dict(schema="cira.native_study.summary.v1", metric=metric,
                independent_completed=sum(v is not None for v in completed.values()),
                unique_registered=len(known), blocks=block_rows,
                missing_results_stay_missing=True)


def primary_contrasts(study, results, metric="seen"):
    """Prespecified CIRA--Serial / CIRA--DeModCL 12-cell Holm family."""
    from .reporting import paired_effects
    results = list(results)
    summarize_completed(study, results, metric)  # Validate exact identities first.
    values = {r["study_trajectory_id"]: r[metric] for r in results
              if r.get("complete") is True and isinstance(r.get(metric), (int, float))
              and not isinstance(r[metric], bool) and math.isfinite(r[metric])}
    recipes = {r["id"]: r["recipe"] for r in study["trajectories"]}
    main = {}
    for view in study["views"]:
        if view["block"] != "N01":
            continue
        r = recipes[view["trajectory_id"]]
        main[(r["dataset"], r["detector"], r["seed"], r["method"])] = values.get(view["trajectory_id"])
    left, right, cells = {}, {}, []
    for dataset, detector, control in itertools.product(study["datasets"], study["detectors"], ("serial", "DeModCL")):
        cell = f"{dataset}/{detector}/CIRA-minus-{control}"
        cells.append(cell)
        for seed in study["seeds"]:
            left[(cell, seed)] = main.get((dataset, detector, seed, "softcas"))
            right[(cell, seed)] = main.get((dataset, detector, seed, control))
    return dict(schema="cira.native_study.primary.v1", metric=metric,
                independent_unit=study["independent_unit"],
                contrasts=paired_effects(left, right, cells, study["seeds"]))
