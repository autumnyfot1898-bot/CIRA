"""Deployed stage-by-domain endpoints, trajectory-level summaries and exports."""
from __future__ import annotations
import csv
from collections import defaultdict
import numpy as np
from scipy import stats
from .state import canonical


def retention(rows):
    """One deployed hash per client/stage; no domain-specific model selection."""
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["seed"], r["client"])].append(r)
    output = []
    for (seed, client), records in grouped.items():
        by_stage = defaultdict(list)
        for r in records:
            by_stage[r["stage"]].append(r)
        history, first = defaultdict(list), {}
        for stage in sorted(by_stage):
            group = by_stage[stage]
            if len({r["deployed"] for r in group}) != 1:
                raise ValueError("Different historical models selected for test domains")
            current = {r["current_domain"] for r in group}
            if len(current) != 1 or len({r["domain"] for r in group}) != len(group):
                raise ValueError("Invalid stage-by-domain rows")
            current = next(iter(current))
            scores = {r["domain"]: r["score"] for r in group}
            expected = set(first) | {current}
            if set(scores) != expected:
                raise ValueError("Missing/extra seen domain in deployed matrix")
            if not all(np.isfinite(v) and 0 <= v <= 1 for v in scores.values()):
                raise ValueError("Invalid endpoint")
            old = list(first)
            forget = np.mean([max(0, max(history[k]) - scores[k]) for k in old]) if old else None
            bwt = np.mean([scores[k] - first[k] for k in old]) if old else None
            output.append({"seed": seed, "client": client, "stage": stage,
                           "deployed": group[0]["deployed"], "current_domain": current,
                           "current": scores[current], "seen": float(np.mean(list(scores.values()))),
                           "forget": float(forget) if forget is not None else None,
                           "bwt": float(bwt) if bwt is not None else None})
            for domain, score in scores.items():
                first.setdefault(domain, score)
                history[domain].append(score)
    return output


def macro_endpoints(rows, expected_clients=None):
    points = retention(rows)
    expected = set(expected_clients) if expected_clients is not None else {p["client"] for p in points}
    grouped = defaultdict(list)
    for p in points:
        grouped[(p["seed"], p["stage"])].append(p)
    if any({p["client"] for p in group} != expected for group in grouped.values()):
        raise ValueError("Incomplete client trajectory cannot produce equal-client macro endpoint")
    return [{"seed": seed, "stage": stage,
             **{key: float(np.mean([p[key] for p in group])) if all(p[key] is not None for p in group) else None
                for key in ("seen", "current", "forget", "bwt")},
             "minimum_client_seen": min(p["seen"] for p in group)}
            for (seed, stage), group in sorted(grouped.items())]


def paired_effects(full, serial, cells, seeds=(1001, 1002, 1003, 1004, 1005)):
    """Keys are (cell, seed); missing pairs stay missing. Holm across declared cells."""
    cells, seeds = tuple(cells), tuple(seeds)
    if len(set(cells)) != len(cells) or len(set(seeds)) != len(seeds):
        raise ValueError("Declared contrast cells and independent seeds must be unique")
    result = []
    for cell in cells:
        paired = []
        for seed in seeds:
            left, right = full.get((cell, seed)), serial.get((cell, seed))
            if left is None or right is None:
                paired.append(None)
                continue
            if not np.isfinite(left) or not np.isfinite(right):
                raise ValueError("Nonfinite paired endpoint; use None for a missing result")
            paired.append(float(left - right))
        diffs = [value for value in paired if value is not None]
        n = len(diffs)
        mean = float(np.mean(diffs)) if n else None
        sd = float(np.std(diffs, ddof=1)) if n > 1 else None
        if n > 1:
            half = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
            ci = [mean - half, mean + half]
            if sd == 0:
                p = 1.0 if mean == 0 else 0.0
            else:
                p = float(stats.ttest_1samp(diffs, 0).pvalue)
        else:
            ci, p = None, None
        result.append({"cell": cell, "paired_differences": paired,
                       "n_pair": n, "mean": mean, "sample_sd": sd, "ci95_unadjusted": ci, "p": p})
    # Untestable contrasts remain in the declared family as p=1 for adjustment.
    order = sorted(range(len(result)), key=lambda i: result[i]["p"] if result[i]["p"] is not None else 1)
    maximum = 0
    for rank, i in enumerate(order):
        p = result[i]["p"]
        if p is not None:
            maximum = max(maximum, min(1.0, (len(cells) - rank) * p))
            result[i]["p_holm"] = maximum
        else:
            result[i]["p_holm"] = None
    return result


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_matrix(directory, rows, expected_clients=None):
    from pathlib import Path
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "test_matrix.jsonl").write_bytes(b"\n".join(canonical(r) for r in rows))
    values = macro_endpoints(rows, expected_clients)
    write_csv(directory / "deployed_endpoints.csv", values,
              ("seed", "stage", "seen", "current", "forget", "bwt", "minimum_client_seen"))
    return values
