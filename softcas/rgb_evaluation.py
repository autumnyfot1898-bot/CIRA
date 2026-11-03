"""Frozen deployed-state RGB perturbation readout; no fitting or selection."""
from __future__ import annotations

from pathlib import Path
from dataclasses import asdict
import json

from .data import DetectionBatches, ManifestSplit, detection_record_batch, load_detection_split
from .state import canonical, digest
from .storage import atomic_write


class PerturbedBatches(DetectionBatches):
    def __init__(self, source, sigma, seed):
        super().__init__(source.records, source.root, source.image_size, source.batch_size)
        self.sigma, self.seed = sigma, seed

    def __getitem__(self, index):
        import torch
        if not 0 <= index < len(self):
            raise IndexError(index)
        records = self.records[index*self.batch_size:(index+1)*self.batch_size]
        batches = [detection_record_batch(r, self.root, self.image_size,
                                         noise_sigma=self.sigma, noise_seed=self.seed) for r in records]
        return {"img": torch.cat([b["img"] for b in batches]),
                "batch_idx": torch.cat([b["batch_idx"]+i for i,b in enumerate(batches)]),
                "cls": torch.cat([b["cls"] for b in batches]),
                "bboxes": torch.cat([b["bboxes"] for b in batches])}


class PerturbedSplit(ManifestSplit):
    @property
    def identity(self):
        return digest({"clean": super().identity, "sigma": self.sigma, "seed": self.noise_seed,
                       "protocol": "float32_pre_resize_gaussian_clip_bilinear_v1"})


def perturbed_split(source, sigma, seed):
    if source.role != "T" or sigma not in (0., .01, .03) or isinstance(sigma, bool):
        raise ValueError("Frozen T readout requires registered sigma 0/.01/.03")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("Integer evaluation seed required")
    # Preserve the exact fixed labels/boxes and use the same float preprocessing
    # as the clean native loader. A unique evaluation identity prevents reuse.
    split = PerturbedSplit(source.role, source.client, source.stage, source.domain,
                          source.records, source.root, source.labels, source.image_size, source.batch_size)
    split.sigma, split.noise_seed = sigma, seed
    split.batches = PerturbedBatches(source.batches, sigma, seed)
    split.manifest = canonical({"clean": source.identity, "sigma": sigma, "seed": seed,
                               "protocol": "float32_pre_resize_gaussian_clip_bilinear_v1"}).decode()
    return split


def evaluate_rgb(spec):
    from .backend import YoloBackend, work
    from .cache import PackageCache
    from .config import load_config, effective_formal
    from .state import State, Package
    from .resources import Ledger, Cost
    output = Path(spec["output"])
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("RGB output must be new or empty")
    config = load_config(spec["config"], spec.get("formal", False), spec.get("overrides"))
    formal = effective_formal(config, spec.get("formal", False))
    detector = json.loads(Path(spec["detector_config"]).read_text(encoding="utf-8"))
    backend = YoloBackend(spec["checkpoint"], detector["ports"], detector["labels"], detector["preprocessing"],
                          detector["image_size"], config["runtime"])
    state = State.load(spec["state"])
    if state.hash != spec["deployed_sha256"]:
        raise ValueError("Frozen readout must use the exact deployed state")
    cache = PackageCache(config["cache_bytes"], backend.specs)
    for file in spec.get("packages", []):
        p = Package.read(Path(file).read_bytes())
        cache.put(p.hash, p.wire)
    ledger = Ledger(Cost(**config["offline_budget"]), Cost())
    ledger.runtime, ledger.meter_policy = backend.runtime, config.get("meter")
    if config.get("cost_profile"):
        from .metering import CostProfile
        backend.cost_profile = CostProfile(config["cost_profile"], backend.runtime.identity(), formal)
        profile = backend.cost_profile.data
        if (profile.get("base_sha256") != backend.identity or profile.get("execution") != backend.execution or
                canonical(profile.get("port_interfaces")) != canonical([asdict(p) for p in backend.specs])):
            raise ValueError("RGB calibration must identify this exact frozen base/runtime/ports")
    source = load_detection_split(spec["registry"], spec["client"], spec["stage"], "T", detector["image_size"],
                                  detector["labels"], formal=formal, batch_size=config["evaluation_batch_size"], domain=spec["domain"])
    rows = []
    output.mkdir(parents=True, exist_ok=True)
    for sigma in (0., .01, .03):
        split = perturbed_split(source, sigma, spec["seed"])
        with work(ledger, "frozen_RGB_readout"):
            model = backend.assembly(state, cache, mode="import")
            value = backend.score(model, split)
        rows.append({"sigma": sigma, "score": value, "test_identity": split.identity, "deployed": state.hash})
    result = {"schema": "cira.frozen_rgb.v1", "purpose": config["purpose"], "metric": backend.metric,
              "actual_training_steps": 0, "model_selection": False, "rows": rows,
              "actual": asdict(ledger.used["search"]), "budget_violations": ledger.violations(),
              "execution_scope": "readout on configured local runtime; not remote Jetson measurement unless executed there",
              "preprocessing": "float32/255, seed/image-keyed noise, clip, float bilinear letterbox",
              "clean_anchor_reused": False}
    atomic_write(output/"rgb_evaluation.json", canonical(result))
    return result
