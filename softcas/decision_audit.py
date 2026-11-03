"""Offline diagnostics of observed actions and the saved pre-execution pools.

Only selected children have outcomes. A same-pool priority replay measures a
decision change, never an unobserved counterfactual performance improvement.
"""
from __future__ import annotations

import json
import hashlib
import math
from collections import defaultdict
from pathlib import Path

from .controller import priority
from .resources import Cost


def _finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def replay_choice(pool, constants, no_gain=False):
    """Replay priority only; tie_rank is the actual sorted enumeration order."""
    ranked = []
    for row in pool:
        j = priority(row["parent_utility"], row["champion_utility"],
                     0.0 if no_gain else row["prediction"], row["coverage"],
                     Cost(**row["estimate"]), Cost(**row["remaining"]),
                     constants["kappa"], constants["epsilon"])
        if j > 0:
            ranked.append((j, row["tie_rank"], row["proposal"]))
    return min(ranked, key=lambda item: (-item[0], item[1]))[2] if ranked else None


def audit_actions(actions):
    actions = list(actions)
    if any(type(a.get("attempt")) is not int or a["attempt"] != index
           for index, a in enumerate(actions, 1)):
        raise ValueError("A stage trace must contain each attempt exactly once in execution order 1..N")
    errors, zero_errors, mean_errors, complete_errors = [], [], [], []
    previous_targets, decisions, collisions = [], [], []
    omitted_pools = 0
    negative_retained = negative_reused = negative_count = 0
    for index, action in enumerate(actions):
        target, prediction = action.get("gain"), action.get("prediction")
        if _finite(target) and _finite(prediction):
            # Running baseline uses previous observed targets only, including
            # failure targets if those were fed to the actual online learner.
            baseline = sum(previous_targets) / len(previous_targets) if previous_targets else 0.0
            errors.append(abs(prediction - target))
            zero_errors.append(abs(target))
            mean_errors.append(abs(baseline - target))
            if action.get("status") == "complete":
                complete_errors.append(abs(prediction - target))
            previous_targets.append(target)
        if action.get("status") == "complete" and _finite(target) and target < 0:
            negative_count += 1
            child = action.get("child")
            retained = child in action.get("archive_after", [])
            negative_retained += retained
            negative_reused += retained and any(row.get("parent") == child for row in actions[index + 1:])
        if "candidate_pool" not in action:
            omitted_pools += 1
            continue
        pool = action["candidate_pool"]
        by_feature = defaultdict(list)
        for row in pool:
            by_feature[tuple(row["features"])].append(row["proposal"])
        for proposals in by_feature.values():
            unique = sorted(set(proposals))
            if len(unique) > 1:
                collisions.append({"attempt": action.get("attempt"), "proposals": unique})
        # Cold start and round-robin do not use priority to choose. Do not
        # mislabel their replay as an actual policy comparison.
        if action.get("decision") != "priority":
            continue
        constants = action["priority_constants"]
        full = replay_choice(pool, constants)
        no_gain = replay_choice(pool, constants, no_gain=True)
        expected = no_gain if action.get("no_gain", False) else full
        decisions.append({"attempt": action.get("attempt"), "full": full,
                          "no_gain": no_gain, "changed": full != no_gain,
                          "selected": action["proposal"],
                          "selected_matches_replay": expected == action["proposal"]})
    def average(values):
        return sum(values) / len(values) if values else None
    return {
        "schema": "cira.decision_audit.v1",
        "scope": "observed selected targets and pre-execution same-pool decisions; no unselected outcomes",
        "actions": len(actions), "failed_targets": sum(a.get("status") != "complete" for a in actions),
        "observed_targets": len(errors), "prediction_mae": average(errors),
        "zero_baseline_mae": average(zero_errors), "past_mean_baseline_mae": average(mean_errors),
        "completed_prediction_mae": average(complete_errors),
        "nonzero_predictions": sum(_finite(a.get("prediction")) and a["prediction"] != 0 for a in actions),
        "nonzero_targets": sum(_finite(a.get("gain")) and a["gain"] != 0 for a in actions),
        "missing_candidate_pools": omitted_pools,
        "priority_decisions": len(decisions),
        "changed_priority_decisions": sum(d["changed"] for d in decisions),
        "changed_fraction": average([int(d["changed"]) for d in decisions]),
        "all_selected_replays_match": all(d["selected_matches_replay"] for d in decisions) if decisions else None,
        "decision_replays": decisions, "feature_collision_groups": collisions,
        "negative_completed": negative_count, "negative_retained": negative_retained,
        "negative_reused_as_parent": negative_reused,
    }


def audit_files(action_files):
    # Predictor and archive reset at each stage. Never mix stage histories for
    # the online-mean baseline or for negative-state reuse.
    files = [Path(path) for path in action_files]
    if not files or len(set(p.resolve() for p in files)) != len(files):
        raise ValueError("Provide distinct stage actions.jsonl files")
    stages, seen_contents = [], set()
    for p in files:
        actions = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        digest = hashlib.sha256(json.dumps(actions, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode()).hexdigest()
        if actions and digest in seen_contents:
            raise ValueError("Copied stage traces cannot be counted as independent stages")
        seen_contents.add(digest)
        stages.append({"file": str(p.resolve()), "content_sha256": digest, **audit_actions(actions)})
    return {"schema": "cira.decision_audit.files.v1", "stages": stages,
            "actions": sum(s["actions"] for s in stages),
            "priority_decisions": sum(s["priority_decisions"] for s in stages),
            "changed_priority_decisions": sum(s["changed_priority_decisions"] for s in stages)}
