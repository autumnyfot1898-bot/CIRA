"""Materialize registered data/contact conditions without changing raw images.

All output registries refer to the original immutable image bytes. Scene
selection, client assignment and order are based on metadata, never outcomes.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from .data import validate_registry
from .partition import assign_groups, shared_targets
from .state import canonical, digest
from .storage import atomic_write

CONDITION_FIELDS = {"dataset", "detector", "seed", "alpha", "data_fraction", "order", "contacts", "donors"}


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + " must be an explicit integer >= " + str(minimum))
    return value


def _binding(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _input(value):
    if isinstance(value, dict):
        if set(value) != {"path", "sha256"}:
            raise ValueError("A bound input requires exactly path and sha256")
        actual = _binding(value["path"])
        if value != actual:
            raise ValueError("Input file identity differs from binding")
    else:
        actual = _binding(value)
    return actual, json.loads(Path(actual["path"]).read_text(encoding="utf-8"))


def _vocabulary(document):
    labels = document.get("labels", [])
    if len(labels) != 8 or len(set(labels)) != 8:
        raise ValueError("Eight distinct fixed-vocabulary labels required")
    for record in document["records"]:
        for box in record.get("boxes", []):
            if (len(box) != 5 or isinstance(box[0], bool) or not isinstance(box[0], (int, float)) or
                    box[0] != int(box[0]) or not 0 <= box[0] < len(labels) or
                    any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in box) or
                    any(not 0 <= v <= 1 for v in box[1:]) or any(v <= 0 for v in box[3:])):
                raise ValueError("Invalid fixed-vocabulary normalized box")


def _contexts(document, domains, clients):
    if not domains or len(set(domains)) != len(domains) or not clients or len(set(clients)) != len(clients):
        raise ValueError("Explicit distinct domains and clients required")
    rows = [r for r in document["records"] if r["role"] in ("D", "V", "A")]
    if {r["domain"] for r in rows} != set(domains):
        raise ValueError("Declared domains must exactly cover online domains")
    return rows


def scene_alpha(document, domains, clients, seed, alpha, max_passes):
    """Actual whole-scene assignment using one shared class/client target matrix."""
    _integer(seed, "seed")
    _integer(max_passes, "max_passes")
    clients = tuple(map(str, clients))
    _contexts(document, domains, clients)
    equal = alpha == "equal"
    if not equal and (isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or
                      not math.isfinite(alpha) or alpha <= 0):
        raise ValueError("alpha must be positive or equal")
    result = copy.deepcopy(document)
    targets = shared_targets(seed, len(clients), 8, 1.0 if equal else alpha, equal)
    diagnostics = []
    for role in ("D", "V", "A"):
        for domain in domains:
            rows = [r for r in result["records"] if r["role"] == role and r["domain"] == domain]
            scenes = defaultdict(list)
            for row in rows:
                scenes[row["scene_id"]].append(row)
            groups = []
            for scene, records in sorted(scenes.items()):
                counts = [0] * 8
                for r in records:
                    for box in r["boxes"]:
                        counts[int(box[0])] += 1
                groups.append({"id": scene, "images": len(records), "class_boxes": counts})
            assigned = assign_groups(groups, targets, seed, max_passes)
            for row in rows:
                row["client"] = clients[assigned["assignment"][row["scene_id"]]]
            diagnostics.append({"role": role, "domain": domain, **assigned})
    # Stages are remapped subsequently by the requested order. Scene roles,
    # domains and images are unchanged by client reassignment.
    return result, {"algorithm": "partition.shared_targets + assign_groups", "alpha": alpha,
                    "targets": targets.tolist(), "max_passes": max_passes,
                    "diagnostics": diagnostics}


def domain_order(document, domains, clients, seed, kind, common=None):
    """Change only online D/V/A stage indices; T and DEV remain untouched."""
    clients = tuple(map(str, clients))
    _contexts(document, domains, clients)
    _integer(seed, "seed")
    if kind not in ("independent", "common"):
        raise ValueError("Order must be independent or common")
    if common is not None and (len(common) != len(domains) or set(common) != set(domains)):
        raise ValueError("common_order must be an exact permutation of domains")
    orders = {}
    for index, client in enumerate(clients):
        order = list(domains)
        random.Random(seed + index if kind == "independent" else seed).shuffle(order)
        orders[client] = list(common) if kind == "common" and common is not None else order
    result = copy.deepcopy(document)
    for row in result["records"]:
        if row["role"] in ("D", "V", "A"):
            client = str(row["client"])
            if client not in orders:
                raise ValueError("Online record belongs to an undeclared client")
            row["stage"] = orders[client].index(row["domain"]) + 1
    return result, {"kind": kind, "orders": orders, "algorithm": "Random(seed+client_index), common uses seed"}


def nested_d(document, fraction, seed):
    """Scene-prefix subsets: larger fractions are supersets under a fixed seed.

    Each client/domain cell uses the prefix closest to its requested image count,
    with a nonempty prefix. Image count can differ from a nominal fraction due to
    indivisible scenes; the exact image/box counts and fractions are recorded.
    """
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("D fraction must lie in (0,1]")
    _integer(seed, "seed")
    groups = defaultdict(lambda: defaultdict(list))
    for row in document["records"]:
        if row["role"] == "D":
            groups[(str(row["client"]), row["domain"])][row["scene_id"]].append(row)
    retained = set()
    diagnostics = []
    for (client, domain), scenes in sorted(groups.items()):
        ordered = sorted(scenes, key=lambda scene: (digest({"seed": seed, "operation": "nested_whole_scene_D",
                                                           "client": client, "domain": domain, "scene": scene}), scene))
        total = sum(len(rows) for rows in scenes.values())
        sizes = [sum(len(scenes[s]) for s in ordered[:n]) for n in range(1, len(ordered) + 1)]
        k = min(range(1, len(ordered) + 1), key=lambda n: (abs(sizes[n - 1] - fraction * total), n))
        chosen = ordered[:k]
        retained.update((client, domain, scene) for scene in chosen)
        diagnostics.append({"client": client, "domain": domain, "requested_fraction": fraction,
                            "all_scenes": ordered, "retained_scenes": chosen, "total_images": total,
                            "retained_images": sizes[k - 1], "achieved_fraction": sizes[k - 1] / total})
    if not groups:
        raise ValueError("Registry has no fitting scenes")
    result = copy.deepcopy(document)
    result["records"] = [r for r in result["records"] if r["role"] != "D" or
                         (str(r["client"]), r["domain"], r["scene_id"]) in retained]
    return result, {"algorithm": "seeded whole-scene prefix closest to nominal image count; ties use shorter prefix",
                    "nominal_fraction": fraction, "cells": diagnostics}


def _trace(events):
    rows = copy.deepcopy(events)
    if not isinstance(rows, list):
        raise ValueError("Contact trace must be an event list")
    seen = set()
    for e in rows:
        t = e.get("offset_seconds")
        ids_ok = all(isinstance(e.get(name), (str, int)) and not isinstance(e.get(name), bool) and
                     str(e[name]).strip() for name in ("owner", "receiver"))
        if (isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0 or
                type(e.get("online")) is not bool or not ids_ok or
                str(e["owner"]) == str(e["receiver"])):
            raise ValueError("Invalid contact event")
        key = (str(e["receiver"]), str(e["owner"]), t)
        if key in seen:
            raise ValueError("Ambiguous simultaneous contact events")
        seen.add(key)
        e["receiver"], e["owner"] = str(e["receiver"]), str(e["owner"])
    return sorted(rows, key=lambda e: (e["offset_seconds"], e["receiver"], e["owner"]))


def contact_condition(events, seed, kind, half_payload_event=None):
    """Half-mask complete eligible opportunities; never create an online edge."""
    _integer(seed, "seed")
    rows = _trace(events)
    state, active, opportunities = {}, {}, []
    event_opportunity = {}
    for index, e in enumerate(rows):
        pair = (e["receiver"], e["owner"])
        if e["online"]:
            if not state.get(pair, False):
                identifier = digest({"receiver": pair[0], "owner": pair[1], "offset_seconds": e["offset_seconds"]})
                opportunities.append({"id": identifier, "receiver": pair[0], "owner": pair[1],
                                      "offset_seconds": e["offset_seconds"]})
                active[pair] = identifier
            event_opportunity[index] = active[pair]
        state[pair] = e["online"]
    evidence = {"kind": kind, "opportunities": opportunities, "eligible_opportunity_count": len(opportunities)}
    intervention = {}
    if kind == "half_visibility":
        order = sorted(opportunities, key=lambda o: (digest({"seed": seed, "operation": "contact_half_mask", "id": o["id"]}), o["id"]))
        retained = {o["id"] for o in order[:len(order) // 2]}
        for index, identity in event_opportunity.items():
            if identity not in retained:
                rows[index]["online"] = False
        evidence.update(retained_opportunity_ids=sorted(retained), retained_opportunity_count=len(retained),
                        rule="retain floor(N/2) seed-fixed complete opportunities; disabled events remain offline")
    elif kind == "one_stage2_half_payload":
        event = copy.deepcopy(half_payload_event or {})
        if (event.get("stage") != 2 or event.get("receiver") is None or event.get("owner") is None or
                isinstance(event.get("contact_offset_seconds"), bool) or
                not isinstance(event.get("contact_offset_seconds"), (int, float)) or
                not math.isfinite(event["contact_offset_seconds"])):
            raise ValueError("Bind one explicit stage-two receiver/owner/contact_offset_seconds")
        match = [o for o in opportunities if (o["receiver"], o["owner"], o["offset_seconds"]) ==
                 (str(event["receiver"]), str(event["owner"]), event["contact_offset_seconds"])]
        if len(match) != 1:
            raise ValueError("Registered interruption must name an actual eligible contact opportunity")
        event.update(receiver=str(event["receiver"]), owner=str(event["owner"]), kind="single_half_payload",
                     payload_fraction=.5, event_id=digest({"seed": seed, "stage": 2, "contact": match[0]["id"]}))
        intervention = event
        evidence["interruption"] = event
    elif kind != "normal":
        raise ValueError("Unknown contact condition")
    return rows, evidence, intervention


def donor_condition(kind, weak_lock=None):
    if kind == "normal":
        return {}, {"mode": "normal"}
    if kind == "empty":
        return {"mode": "empty"}, {"mode": "empty", "scope": "catalogue, cache and inherited members throughout trajectory"}
    if kind != "weak":
        raise ValueError("Unknown donor condition")
    binding, lock = _input(weak_lock)
    if lock.get("role") != "DEV" or lock.get("schema") != "cira.weak_donor_lock.v1":
        raise ValueError("Weak donor construction must be locked using DEV, not formal T outcomes")
    allowed = lock.get("allowlist")
    if (not isinstance(allowed, list) or not allowed or len(set(allowed)) != len(allowed) or
            any(not isinstance(k, str) or len(k) != 64 or any(c not in "0123456789abcdef" for c in k) for k in allowed) or
            not isinstance(lock.get("construction"), str) or not lock["construction"].strip()):
        raise ValueError("Weak donor lock requires exact unique package hashes and a DEV construction")
    policy = {"mode": "allowlist", "allowlist": sorted(allowed), "development_lock": binding, "role": "DEV"}
    return policy, {**policy, "construction": lock["construction"]}


def achieved_histograms(document, domains, clients):
    rows = []
    for client in map(str, clients):
        for domain in domains:
            for role in ("D", "V", "A"):
                selected = [r for r in document["records"] if str(r["client"]) == client and r["domain"] == domain and r["role"] == role]
                counts = [0] * 8
                for r in selected:
                    for b in r["boxes"]:
                        counts[int(b[0])] += 1
                rows.append({"client": client, "domain": domain, "role": role, "images": len(selected),
                             "scenes": len({r["scene_id"] for r in selected}), "class_boxes": counts})
    if any(row["images"] == 0 for row in rows):
        raise ValueError("Insufficient disjoint scene support in a client/domain/DVA cell; no held-out repair performed")
    return rows


def build_condition(spec, output=None):
    condition = copy.deepcopy(spec["condition"])
    if set(condition) != CONDITION_FIELDS:
        raise ValueError("Condition must contain exactly dataset/detector/seed/alpha/data_fraction/order/contacts/donors")
    _integer(condition["seed"], "seed")
    for key in ("dataset", "detector"):
        if not isinstance(condition[key], str) or not condition[key].strip():
            raise ValueError("Explicit dataset/detector identities required")
    registry_input, document = _input(spec["source_registry"])
    trace_input, trace = _input(spec["source_contact_trace"])
    formal = spec.get("formal", False)
    if type(formal) is not bool:
        raise ValueError("formal must be a boolean")
    _vocabulary(document)
    validate_registry(document["records"], formal=formal)
    domains, clients = list(spec["domains"]), tuple(map(str, spec.get("clients", ["0", "1", "2", "3"])))
    before_test = canonical([r for r in document["records"] if r["role"] in ("T", "DEV")])
    assigned, alpha = scene_alpha(document, domains, clients, condition["seed"], condition["alpha"], spec["max_passes"])
    ordered, order = domain_order(assigned, domains, clients, condition["seed"], condition["order"], spec.get("common_order"))
    before_subset = canonical([r for r in ordered["records"] if r["role"] != "D"])
    result, subset = nested_d(ordered, condition["data_fraction"], condition["seed"])
    assert before_subset == canonical([r for r in result["records"] if r["role"] != "D"])
    assert before_test == canonical([r for r in result["records"] if r["role"] in ("T", "DEV")])
    validate_registry(result["records"], formal=formal)
    histograms = achieved_histograms(result, domains, clients)
    contacts, contact_evidence, interruption = contact_condition(trace, condition["seed"], condition["contacts"], spec.get("half_payload_event"))
    donors, donor_evidence = donor_condition(condition["donors"], spec.get("weak_donor_lock"))
    # Relocate relative references only, never read/write/resize source images.
    source_root = Path(registry_input["path"]).parent
    for r in result["records"]:
        if not Path(r["image"]).is_absolute():
            r["image"] = str((source_root / r["image"]).resolve())
    destination = Path(output or spec["output"]).resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError("Condition output must be new or empty")
    destination.mkdir(parents=True, exist_ok=True)
    result["condition_materialization"] = {"condition_key": digest(condition), "source_registry_sha256": registry_input["sha256"]}
    atomic_write(destination / "registry.json", canonical(result))
    atomic_write(destination / "contact_trace.json", canonical(contacts))
    manifest = {"schema": "cira.condition.v1", "complete": True, "condition": condition,
                "condition_key": digest(condition), "formal_registry_checked": formal,
                "inputs": {"registry": registry_input, "contact_trace": trace_input},
                "outputs": {"registry": _binding(destination / "registry.json"),
                            "contact_trace": _binding(destination / "contact_trace.json")},
                "interventions": {"donors": donors, "contacts": interruption},
                "evidence": {"alpha_assignment": alpha, "domain_order": order, "nested_D": subset,
                             "contacts": contact_evidence, "donors": donor_evidence,
                             "achieved_histograms": histograms, "D_subset_preserves_VAT": True,
                             "all_conditions_preserve_T_DEV": True, "raw_images_modified": False}}
    atomic_write(destination / "condition_manifest.json", canonical(manifest))
    return manifest
