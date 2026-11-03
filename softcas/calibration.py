"""Independent-development calibration command; not executed during authoring."""
from __future__ import annotations
import json
from pathlib import Path
from dataclasses import replace
from itertools import combinations
from .backend import YoloBackend
from .config import load_config
from .data import StageData, load_detection_split
from .cache import PackageCache
from .engine import initial_state
from .state import State, Structure, FitSettings, Package
from .metering import calibrate_states


def calibrate_job(spec, output):
    config = load_config(spec["config"])
    if spec["seed"] not in (9001, 9002, 9003) or config["purpose"] != "DEVELOPMENT":
        raise ValueError("Use the registered independent development seeds, not formal evaluation trajectories")
    config["runtime"]["seed"] = spec["seed"]
    detector = json.loads(Path(spec["detector_config"]).read_bytes())
    backend = YoloBackend(spec["checkpoint"], detector["ports"], detector["labels"], detector["preprocessing"], detector["image_size"], config["runtime"])
    cache = PackageCache(config["cache_bytes"], backend.specs)
    for file in spec.get("packages", []):
        package = Package.read(Path(file).read_bytes())
        cache.put(package.hash, package.wire)
    registry = json.loads(Path(spec["registry"]).read_bytes())
    if registry.get("assignment", {}).get("seed") != spec["seed"]:
        raise ValueError("Development registry must identify its actual independent assignment seed")
    data = StageData(*(load_detection_split(spec["registry"], spec["client"], spec["stage"], role, detector["image_size"], detector["labels"],
                       formal=True, batch_size=config["fit_batch_size"] if role == "D" else config["evaluation_batch_size"])
                       for role in ("D", "V", "A")))
    states = [State.load(file) for file in spec.get("states", [])]
    if spec.get("include_external_components", True):
        from .model import Adapter
        from .state import freeze_tensors
        import torch
        for port in range(6):
            for ratio in (1 / 16, 1 / 8):
                with torch.random.fork_rng():
                    torch.manual_seed(spec["seed"] + port)
                    branch = Adapter(backend.specs[port].channels, ratio)
                    torch.nn.init.normal_(branch.up.weight, std=0.01)
                package = Package("CALIBRATION_ONLY_NOT_A_PEER", f"{port}-{ratio}", 0, 0, backend.specs[port], ratio,
                                  freeze_tensors(branch.state_dict()), "Synthetic arithmetic specimen; never published or used as method performance evidence")
                cache.put(package.hash, package.wire)
                fixed = config["fixed"]
                anchor = initial_state(backend, cache, Structure(tuple(fixed["mask"]), fixed["ratio"]),
                                       FitSettings(fixed["lr"], fixed["residual"]), spec["seed"])
                states.append(replace(anchor, C=(package.hash,), beta=((package.hash, 0.0),)))
    if spec.get("include_search_grid"):
        h = FitSettings(config["fixed"]["lr"], config["fixed"]["residual"])
        for ports in list(combinations(range(6), 1)) + list(combinations(range(6), 2)):
            for ratio in (1 / 16, 1 / 8):
                mask = tuple(int(i in ports) for i in range(6))
                state = initial_state(backend, cache, Structure(mask, ratio), h, spec["seed"])
                states.append(state)
                for members in spec.get("composition_sets", []):
                    if any(k not in cache for k in members):
                        raise ValueError("Calibration needs exact observed compatible donor versions")
                    states.append(replace(state, C=tuple(sorted(members)), beta=tuple((k, 0) for k in sorted(members))))
    if not states:
        raise ValueError("Supply states and/or request the declared local search grid")
    return calibrate_states(backend, cache, data, states, config, output)
