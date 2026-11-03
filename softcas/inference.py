"""Run the currently committed model on an Orin, preserving fixed receiving weights."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import torch
from .backend import YoloBackend
from .cache import PackageCache
from .data import detection_record_batch
from .runtime import RuntimeSpec
from .state import State, Package
from .storage import atomic_write
from .state import canonical
from .tensorrt_backend import install_adapter_engines


def predict_deployed(spec):
    agent = json.loads(Path(spec["agent_config"]).read_bytes())
    namespace = spec["deployment_id"]
    if not namespace or any(not (c.isalnum() or c in "-_") for c in namespace):
        raise ValueError("Invalid deployment namespace")
    pointer = json.loads((Path(agent["job_store"]) / "current" / (namespace + ".json")).read_bytes())
    if pointer.get("qualification_report"):
        from .device import target_code_identity
        qualification = json.loads(Path(pointer["qualification_report"]).read_bytes())
        if qualification["code_identity"] != target_code_identity() or qualification["runtime"] != RuntimeSpec(**agent["runtime"]).identity():
            raise RuntimeError("The serving executor differs from the qualified deployment")
    bundle = Path(pointer["bundle_root"])
    manifest = json.loads((bundle / "bundle.json").read_bytes())
    state = State.load(bundle / "states" / (pointer["candidate"] + ".json"))
    backend = YoloBackend(agent["bases"][manifest["base_sha256"]], manifest["ports"], manifest["labels"], manifest["preprocessing"],
                          manifest["image_size"], agent["runtime"])
    if backend.identity != manifest["base_sha256"]:
        raise ValueError("Serving base differs from the committed frozen checkpoint")
    cache = PackageCache(agent["cache_bytes"], backend.specs)
    for key in state.C:
        package = Package.read((bundle / "packages" / (key + ".json")).read_bytes())
        cache.put(package.hash, package.wire)
    model = backend.assembly(state, cache)
    model.eval()
    if agent["executor"] == "tensorrt_adapters":
        example = torch.zeros(1, 3, manifest["image_size"], manifest["image_size"], device=backend.device)
        install_adapter_engines(model, example, backend.runtime, Path(agent["engine_store"]) / namespace,
                                agent.get("trt_fp16", False), agent["trt_workspace_bytes"])
    backend.base = None
    records = []
    with torch.no_grad():
        for file in spec["images"]:
            file = Path(file)
            record = {"image": str(file.resolve()), "image_sha256": hashlib.sha256(file.read_bytes()).hexdigest(), "boxes": []}
            batch = detection_record_batch(record, file.parent, manifest["image_size"])
            predictions = backend.predict(model, batch["img"])[0]
            from PIL import Image
            with Image.open(file) as image:
                width, height = image.size
            size = manifest["image_size"]
            ratio = min(size / width, size / height)
            nw, nh = round(width * ratio), round(height * ratio)
            left, top = (size - nw) // 2, (size - nh) // 2
            original = predictions.clone()
            original[:, [0, 2]] = ((original[:, [0, 2]] - left) * width / nw).clamp(0, width)
            original[:, [1, 3]] = ((original[:, [1, 3]] - top) * height / nh).clamp(0, height)
            records.append({"image": str(file), "deployed": state.hash, "detections_letterbox_xyxy_score_class": predictions.tolist(),
                            "detections_original_xyxy_score_class": original.tolist(),
                            "image_size": manifest["image_size"], "labels": manifest["labels"]})
    atomic_write(spec["output"], canonical(records))
    return {"images": len(records), "deployed": state.hash, "output": spec["output"]}
