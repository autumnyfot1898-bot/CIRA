"""Declared epoch budgets and replayable, stage-local fitting data order."""
from __future__ import annotations
import numpy as np
from .state import digest


def epoch_mode(config):
    return config.get("fitting", {}).get("unit") == "epochs"


def stage_steps(config, batches_per_epoch):
    if not epoch_mode(config):
        return None
    return config["fitting"]["stage_epochs"] * batches_per_epoch


def epoch_seed(config, proposal, split, epoch):
    # Excludes estimated costs, elapsed time and evaluation outcomes.
    context = config.get("randomization", {})
    key = {"seed": config["seed"], "dataset": context.get("dataset"),
           "detector": context.get("detector"), "client": str(split.client),
           "stage": split.stage, "operation": "fit_epoch_shuffle", "epoch": epoch,
           "attempt": config.get("_attempt_index", 0), "construction_seed": proposal.seed,
           "parent": proposal.parent, "kind": proposal.kind}
    return int(digest(key)[:16], 16)


def shuffled_batches(batches, seed):
    """Native records are reshuffled before batching; fixture batches by index."""
    rng = np.random.Generator(np.random.PCG64(seed))
    if hasattr(batches, "shuffled"):
        return batches.shuffled(rng.permutation(len(batches.records)).tolist())
    return tuple(batches[int(i)] for i in rng.permutation(len(batches)))
