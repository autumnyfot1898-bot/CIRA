"""Pure data interface for the seven reference table fragments; never edits TeX."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .config import METHODS
from .reporting import write_csv, paired_effects
from .state import canonical

CELLS = tuple(f"{dataset}/{detector}" for dataset in ("ControlledFog", "ACDC")
              for detector in ("YOLOv8n", "YOLO11n", "YOLO26n"))


def export_tables(trajectories, output):
    """Input: one record per method/cell/independent seed, endpoints in [0,1].

    Synthetic results are rejected rather than relabeled as AP. Missing values
    remain blank. Clients/devices/repeats never increase n here.
    """
    if any(r.get("purpose") != "FORMAL" or r.get("metric") != "AP50:95" for r in trajectories):
        raise ValueError("Table result interface accepts identified formal AP trajectories only")
    for record in trajectories:
        if record.get("complete") is not True:
            raise ValueError("Incomplete trajectories must not enter formal endpoint tables")
        if record["method"] not in METHODS or record["cell"] not in CELLS:
            raise ValueError("Unknown method/cell would be omitted from the declared table layout")
        if record.get("ablation", "none") != "none":
            raise ValueError("Intervention trajectories belong to the ablation table, not the main table")
        seen = record.get("seen")
        if seen is not None and (not np.isfinite(seen) or not 0 <= seen <= 1):
            raise ValueError("SeenAP endpoint must be finite and in [0,1], or missing")
    keys = [(r["method"], r["cell"], r["seed"]) for r in trajectories]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate trajectory-level record")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    main, retention, resources, device = [], [], [], []
    for method in METHODS:
        for cell in CELLS:
            records = [r for r in trajectories if r["method"] == method and r["cell"] == cell]
            values = [r["seen"] for r in records if r.get("seen") is not None]
            main.append({"method": method, "cell": cell, "n": len(values),
                         "mean_percent": float(np.mean(values) * 100) if values else None,
                         "sample_sd_percent": float(np.std(values, ddof=1) * 100) if len(values) > 1 else None})
            for name, destination, fields in (
                ("retention", retention, ("current", "forget", "bwt", "minimum_client_seen")),
                ("resources", resources, ("F_total", "selection_E", "network_tx_bytes", "max_client_update_seconds")),
                ("device", device, ("active_external_paths", "deployed_tensor_bytes", "peak_process_ram_bytes", "p95_ms"))):
                for r in records:
                    destination.append({"method": method, "cell": cell, "seed": r["seed"],
                                        **{f: r.get(f) for f in fields}})
    write_csv(output / "T2_primary.csv", main, ("method", "cell", "n", "mean_percent", "sample_sd_percent"))
    for filename, data, fields in (("T3_retention.csv", retention, ("current", "forget", "bwt", "minimum_client_seen")),
                                   ("T4_resources.csv", resources, ("F_total", "selection_E", "network_tx_bytes", "max_client_update_seconds")),
                                   ("T5_device.csv", device, ("active_external_paths", "deployed_tensor_bytes", "peak_process_ram_bytes", "p95_ms"))):
        write_csv(output / filename, data, ("method", "cell", "seed", *fields))
    full = {(r["cell"], r["seed"]): r.get("seen") for r in trajectories if r["method"] == "softcas"}
    serial = {(r["cell"], r["seed"]): r.get("seen") for r in trajectories if r["method"] == "serial"}
    (output / "T7_paired.json").write_bytes(canonical(paired_effects(full, serial, CELLS)))
    # T1 and T6 require explicit config/intervention records, separate from AP values.
    (output / "T1_setup_required_fields.json").write_bytes(canonical([
        "dataset", "detector_checkpoint_sha256", "class_vocabulary", "scene_registry_sha256", "seed",
        "DVA_counts", "fitting_config", "controller_config", "resource_allowances", "device_identity"]))
    (output / "T6_ablation_input_fields.json").write_bytes(canonical([
        "variant", "only_changed_rule", "cell", "seed", "seen", "F_total", "config_sha256"]))
