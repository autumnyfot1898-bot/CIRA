"""Appendix eq:partition_balance, whole-scene allocation before model outcomes.

The grouping/role registry is an external input. This module never guesses clips
from recording-folder names and never reads detector performance.
"""
from __future__ import annotations
import random
import numpy as np


def shared_targets(seed, clients=4, classes=8, alpha=1.0, equal=False):
    if clients < 1 or classes < 1 or alpha <= 0:
        raise ValueError("Invalid partition dimensions/Dirichlet concentration")
    return (np.full((classes, clients), 1 / clients) if equal else
            np.random.default_rng(seed).dirichlet(np.full(clients, alpha), size=classes))


def balance_objective(assignments, counts, images, targets):
    classes, clients = targets.shape
    totals = counts.sum(0)
    support = totals > 0
    by_client = np.zeros((clients, classes))
    image_client = np.zeros(clients)
    for i, client in enumerate(assignments):
        if client >= 0:
            by_client[client] += counts[i]
            image_client[client] += images[i]
    class_term = float(np.mean((by_client[:, support] / totals[support] - targets[support].T) ** 2)) if support.any() else 0
    image_term = float(np.mean((image_client / images.sum() - 1 / clients) ** 2))
    return class_term + image_term


def assign_groups(groups, targets, seed, max_passes):
    """Seeded greedy pass, then globally best improving whole-group move/swap.

    targets is the SAME externally created class x client matrix for every
    domain-role block. max_passes and the tie order are part of the config record.
    """
    targets = np.asarray(targets, dtype=float)
    if targets.ndim != 2 or not np.allclose(targets.sum(1), 1) or (targets < 0).any():
        raise ValueError("Invalid shared target matrix")
    if not groups or max_passes < 0 or len({g["id"] for g in groups}) != len(groups):
        raise ValueError("Nonempty unique whole groups and nonnegative pass limit required")
    order = sorted(range(len(groups)), key=lambda i: groups[i]["id"])
    random.Random(seed).shuffle(order)
    ordered = [groups[i] for i in order]
    counts = np.array([g["class_boxes"] for g in ordered], dtype=float)
    images = np.array([g["images"] for g in ordered], dtype=float)
    if counts.shape[1] != len(targets) or (counts < 0).any() or (images <= 0).any():
        raise ValueError("Invalid whole-group image/class counts")
    assignments = [-1] * len(groups)
    for i in range(len(groups)):
        choices = []
        for client in range(targets.shape[1]):
            trial = assignments.copy()
            trial[i] = client
            choices.append((balance_objective(trial, counts, images, targets), client))
        assignments[i] = min(choices)[1]
    initial = balance_objective(assignments, counts, images, targets)
    changes = []
    terminated = "pass_limit"
    for pass_id in range(max_passes):
        current = balance_objective(assignments, counts, images, targets)
        best = None
        for i in range(len(groups)):
            for client in range(targets.shape[1]):
                if client == assignments[i]:
                    continue
                trial = assignments.copy()
                trial[i] = client
                item = (balance_objective(trial, counts, images, targets), ("move", i, client), trial)
                if item[0] < current - 1e-14 and (best is None or item[:2] < best[:2]):
                    best = item
            for j in range(i):
                if assignments[i] == assignments[j]:
                    continue
                trial = assignments.copy()
                trial[i], trial[j] = trial[j], trial[i]
                item = (balance_objective(trial, counts, images, targets), ("swap", j, i), trial)
                if item[0] < current - 1e-14 and (best is None or item[:2] < best[:2]):
                    best = item
        if best is None:
            terminated = "no_improving_move_or_swap"
            break
        assignments = best[2]
        changes.append({"pass": pass_id, "operation": best[1], "objective": best[0]})
    return {"assignment": {g["id"]: client for g, client in zip(ordered, assignments)},
            "seed": seed, "targets": targets.tolist(), "initial_objective": initial,
            "objective": balance_objective(assignments, counts, images, targets), "changes": changes,
            "termination": terminated, "group_order": [g["id"] for g in ordered],
            "client_images": [int(sum(images[i] for i, v in enumerate(assignments) if v == c)) for c in range(targets.shape[1])],
            "tie": "objective_then_operation_then_seeded_group_index_then_client"}
