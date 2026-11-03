"""Compile deduplicated study recipes into input-bound execution jobs.

Compilation reads local manifests only. Execution is a separate explicit call.
Every non-default condition must be materialized, not merely written in a label.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from .config import load_config, merge_dict
from .state import canonical, digest
from .study import INTERNAL, GRID, SERIAL_RULES, summarize_completed

CONDITION_FIELDS = ("alpha", "data_fraction", "order", "contacts", "donors")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def condition_identity(recipe):
    return {**{k: recipe[k] for k in ("dataset", "detector", "seed")},
            **{k: recipe["factors"][k] for k in CONDITION_FIELDS}}


def condition_key(recipe):
    return digest(condition_identity(recipe))


def file_binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _asset(binding, inputs):
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError("Every asset needs exactly path and sha256")
    path = Path(binding["path"]).resolve()
    actual = file_binding(path)
    if actual["sha256"] != binding["sha256"]:
        raise ValueError("Bound asset content differs: " + str(path))
    if str(path) in inputs and inputs[str(path)] != actual["sha256"]:
        raise ValueError("Conflicting asset binding")
    inputs[str(path)] = actual["sha256"]
    return str(path)


def _config_inputs(path, inputs, parents=()):
    path = Path(path).resolve()
    if path in parents:
        raise ValueError("Configuration inheritance cycle")
    inputs[str(path)] = file_binding(path)["sha256"]
    raw = read(path)
    if "extends" in raw:
        _config_inputs(path.parent / raw["extends"], inputs, (*parents, path))


def _serial(rule):
    if rule not in SERIAL_RULES:
        raise ValueError("Unknown registered Serial rule")
    first, second = rule.split("_")
    order = ["configuration", "collaboration"] if first.startswith("cfg") else ["collaboration", "configuration"]
    fraction = int("".join(c for c in first if c.isdigit())) / 100
    return dict(order=order, fraction=fraction, attempt_fraction=fraction)


def _select(study, selection):
    known = {r["id"] for r in study["trajectories"]}
    ids = set(selection.get("ids", known))
    if not ids <= known:
        raise ValueError("Selection contains unknown trajectory IDs")
    if "blocks" in selection:
        blocks = set(selection["blocks"])
        if not blocks <= {r["block"] for r in study["blocks"]}:
            raise ValueError("Unknown study block")
        ids &= {v["trajectory_id"] for v in study["views"] if v["block"] in blocks}
    if "methods" in selection:
        methods = set(selection["methods"])
        if not methods <= {r["recipe"]["method"] for r in study["trajectories"]}:
            raise ValueError("Unknown registered method")
        ids &= {r["id"] for r in study["trajectories"] if r["recipe"]["method"] in methods}
    return [r for r in study["trajectories"] if r["id"] in ids]


def binding_template(study, selection=None):
    """Write once per shared input condition, rather than once per method."""
    summarize_completed(study, [])
    rows = _select(study, selection or {})
    conditions = {condition_key(r["recipe"]): {"condition": condition_identity(r["recipe"]),
                  "manifest": {"path": None, "sha256": None}} for r in rows}
    return {"schema": "cira.study_bindings.v1", "formal": True,
            "study_sha256": digest(study), "selection": selection or {},
            "output_root": None, "clients": ["0", "1", "2", "3"],
            "development_lock": {"path": None, "sha256": None},
            "cells": {r["recipe"]["dataset"] + "/" + r["recipe"]["detector"]: {
                "config": {"path": None, "sha256": None},
                "checkpoint": {"path": None, "sha256": None},
                "detector_config": {"path": None, "sha256": None},
                "overrides": {}, "budgets_by_round": {str(n): {"path": None, "sha256": None} for n in (300, 500, 1000)}} for r in rows},
            "conditions": conditions, "external_adapters": {},
            "note": "Null assets are unbound. Compile reports blockers; it never silently executes a default condition."}


def _development(study, spec, inputs):
    path = _asset(spec["development_lock"], inputs)
    lock = read(path)
    if lock.get("schema") != "cira.development_lock.v1" or lock.get("role") != "DEV":
        raise ValueError("An explicit native DEV selection lock is required")
    if lock.get("selection") != study["development_selection"] or not lock["selection"].get("locked"):
        raise ValueError("Study and actual development selection differ or are unlocked")
    if set(lock.get("seeds", [])) != {9001, 9002, 9003}:
        raise ValueError("Development lock requires the declared independent DEV seeds")
    return lock


def _compile_one(study, row, spec, lock, common_inputs):
    from .experiment import trajectory_config
    recipe = row["recipe"]
    f, inputs = recipe["factors"], dict(common_inputs)
    cell = recipe["dataset"] + "/" + recipe["detector"]
    bound = spec["cells"][cell]
    condition = spec["conditions"][condition_key(recipe)]
    if condition["condition"] != condition_identity(recipe):
        raise ValueError("Condition binding identity differs from recipe")
    manifest_file = _asset(condition["manifest"], inputs)
    materialized = read(manifest_file)
    if materialized.get("condition") != condition_identity(recipe):
        raise ValueError("Materialized condition differs from the requested recipe")
    if materialized.get("schema") != "cira.condition.v1" or materialized.get("complete") is not True:
        raise ValueError("Condition materialization did not complete")
    registry = _asset(materialized["outputs"]["registry"], inputs)
    trace = _asset(materialized["outputs"]["contact_trace"], inputs)
    checkpoint = _asset(bound["checkpoint"], inputs)
    detector = _asset(bound["detector_config"], inputs)
    config = _asset(bound["config"], inputs)
    _config_inputs(config, inputs)
    budgets = read(_asset(bound["budgets_by_round"][str(f["rounds"])], inputs))
    if budgets.get("rounds") != f["rounds"] or budgets.get("cell") != cell:
        raise ValueError("Calibrated allowance table belongs to a different cell/round condition")
    if not 0 < budgets["search"]["F"] or not 0 < budgets["search"]["W"]:
        raise ValueError("Bind nonzero calibrated F/W allowances")
    q = GRID[study["development_selection"]["controller_q"]]
    overrides = merge_dict(bound.get("overrides", {}), q)
    if recipe["method"] == "softcas":
        if f["controller_parameters"] != GRID[f["controller_q"]]:
            raise ValueError("Recipe grid parameters differ from the versioned implementation")
        overrides = merge_dict(overrides, f["controller_parameters"])
    fixed = lock["cells"][cell]["fixed"]
    search = copy.deepcopy(budgets["search"])
    search["F"] *= f["compute"]
    overrides = merge_dict(overrides, {"fixed": fixed,
        "round_budget": {"per_client": f["rounds"], "stages": 4, "epochs_per_round": 2},
        "fitting": {"unit": "epochs", "stage_epochs": f["rounds"] // 2},
        "budgets": {"search": search, "release": budgets["release"]}})
    if recipe["method"] == "serial":
        overrides["serial"] = _serial(f["serial_rule"])
    name = row["id"]
    job = {"name": name, **{k: recipe[k] for k in ("dataset", "detector", "seed", "method", "ablation")},
           "config": config, "overrides": overrides, "checkpoint": checkpoint, "detector_config": detector,
           "registry": registry, "registry_id": condition_key(recipe), "contact_trace": trace,
           "clients": list(spec["clients"]), "formal": spec.get("formal", True),
           "output": str(Path(spec["output_root"]).resolve() / name),
           "study_trajectory_id": row["id"], "study_recipe": copy.deepcopy(recipe),
           "condition_manifest": manifest_file, "interventions": materialized.get("interventions", {}),
           "input_manifest": inputs}
    if recipe["method"] not in INTERNAL:
        # External algorithms have their own executor and qualification; they
        # must not be passed through the CIRA method switch.
        from .external import validate_adapter
        adapter = spec["external_adapters"][recipe["method"]]
        validate_adapter(adapter, job)
        job.update(executor="external", adapter=copy.deepcopy(adapter))
    else:
        job["executor"] = "internal"
        effective, job["formal"] = trajectory_config(job)
        if effective.get("cost_profile"):
            profile = Path(effective["cost_profile"]).resolve()
            inputs[str(profile)] = file_binding(profile)["sha256"]
        job["expected_configuration_sha256"] = digest(effective)
    return job


def compile_study(study, spec):
    """Return ready jobs plus exact blockers. No backend, socket, or trainer starts."""
    summarize_completed(study, [])
    if spec.get("study_sha256") != digest(study):
        raise ValueError("Binding file does not identify this study registry")
    if type(spec.get("formal", True)) is not bool:
        raise ValueError("formal must be a boolean")
    clients = spec.get("clients", [])
    if not clients or len(set(map(str, clients))) != len(clients):
        raise ValueError("Bind distinct federation clients")
    rows = _select(study, spec.get("selection", {}))
    inputs, lock, common_error = {}, None, None
    try:
        if not spec.get("output_root"):
            raise ValueError("Bind an explicit output_root")
        lock = _development(study, spec, inputs)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        common_error = str(exc)
    jobs, blocked = [], []
    for row in rows:
        try:
            if common_error:
                raise ValueError(common_error)
            jobs.append(_compile_one(study, row, spec, lock, inputs))
        except (KeyError, TypeError, ValueError, OSError) as exc:
            blocked.append({"study_trajectory_id": row["id"], "reason": type(exc).__name__ + ": " + str(exc)})
    plan = {"schema": "cira.compiled_study.v1", "execution_topology": "single_host_logical_clients",
            "study_sha256": digest(study), "binding_spec_sha256": digest(spec),
            "selected_trajectories": len(rows), "ready_trajectories": len(jobs), "blocked": blocked,
            "jobs": jobs, "unit": "complete federation trajectory", "executed": False}
    if "physical_nodes" in spec and jobs:
        from .experiment import _physical_plan, trajectory_config
        if any(j["executor"] != "internal" for j in jobs):
            raise ValueError("External executors must declare and manage their native physical topology separately")
        plan["jobs"], plan["trajectory_groups"] = _physical_plan(jobs, spec)
        for job in plan["jobs"]:
            _config_inputs(job["config"], job["input_manifest"])
            config, job["formal"] = trajectory_config(job)
            job["expected_configuration_sha256"] = digest(config)
        plan.update(execution_topology="physical_nodes", node_ids=list(map(str, clients)))
    if spec.get("require_all_ready", False) and blocked:
        raise ValueError(f"{len(blocked)} selected trajectories have unresolved bindings; inspect a non-strict compile report")
    return plan


def preflight_job(job):
    """Pin input and configuration identities again immediately before execution."""
    if digest(job["study_recipe"]) != job["study_trajectory_id"]:
        raise ValueError("Corrupted study recipe")
    for path, expected in job["input_manifest"].items():
        if file_binding(path)["sha256"] != expected:
            raise ValueError("Compiled input changed before execution: " + path)
    if job.get("executor") == "internal":
        from .experiment import trajectory_config
        configuration, _ = trajectory_config(job)
        if digest(configuration) != job["expected_configuration_sha256"]:
            raise ValueError("Effective configuration differs from the compiled job")
    return True


def execute_study(plan, resume=False, node_id=None):
    from .experiment import execute_trajectory
    if plan.get("schema") != "cira.compiled_study.v1":
        raise ValueError("Explicit compiled study plan required")
    if plan.get("execution_topology") == "physical_nodes":
        if str(node_id) not in plan["node_ids"]:
            raise ValueError("Physical study execution requires this peer's explicit node-id")
        jobs = [j for j in plan["jobs"] if j["node_id"] == str(node_id)]
    else:
        if node_id is not None:
            raise ValueError("node-id is only valid for physical study plans")
        jobs = plan["jobs"]
    if not jobs:
        raise ValueError("No fully bound jobs to execute; inspect compile blockers")
    # Validate all selected inputs before the first costly trajectory starts.
    for job in jobs:
        preflight_job(job)
    results = []
    for job in jobs:
        if job["executor"] == "external":
            from .external import execute_external
            result = execute_external(job, resume=resume)
        else:
            result = execute_trajectory(job, resume)
        results.append(result)
    return results
