"""Offline dataset conversion and source/development-only frozen-base preparation."""
from __future__ import annotations
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from .data import validate_registry
from .partition import shared_targets, assign_groups
from .state import canonical, digest
from .storage import atomic_write

CLASSES = ("person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle")


def convert_coco(annotation_files, image_roots, scene_map, destination, aliases=None):
    """scene_map explicitly supplies scene/counterpart/appearance/role/domain.

    It is evidence, not a folder-name heuristic. No model score participates.
    annotation_files and image_roots use the same dataset/domain keys.
    """
    aliases = aliases or {}
    groups = json.loads(Path(scene_map).read_text(encoding="utf-8"))
    records = []
    for domain, annotation_file in annotation_files.items():
        document = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
        names = {c["id"]: aliases.get(c["name"], c["name"]) for c in document["categories"]}
        annotations = defaultdict(list)
        for annotation in document["annotations"]:
            if annotation.get("iscrowd", 0):
                continue
            name = names[annotation["category_id"]]
            if name not in CLASSES:
                raise ValueError("Explicitly resolve unsupported annotation class: " + name)
            annotations[annotation["image_id"]].append(annotation)
        for image in document["images"]:
            key = domain + "/" + image["file_name"]
            if key not in groups:
                raise ValueError("Missing verified scene/counterpart mapping: " + key)
            group = groups[key]
            source = (Path(image_roots[domain]) / image["file_name"]).resolve()
            boxes = []
            for annotation in annotations[image["id"]]:
                x, y, w, h = annotation["bbox"]
                if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > image["width"] + 1e-4 or y + h > image["height"] + 1e-4:
                    raise ValueError("Resolve out-of-image/invalid source boxes before conversion")
                boxes.append([CLASSES.index(names[annotation["category_id"]]), (x + w / 2) / image["width"],
                              (y + h / 2) / image["height"], w / image["width"], h / image["height"]])
            records.append({"image": str(source), "image_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                            "scene_id": group["scene_id"], "role": group["role"], "domain": domain,
                            "client": group.get("client", "unassigned"), "stage": group.get("stage", 0),
                            "verified_grouping": group.get("verified_grouping", False), "boxes": boxes,
                            "source_annotation": str(Path(annotation_file).resolve()), "source_image_id": image["id"]})
    value = {"schema": 1, "labels": CLASSES, "records": records,
             "sources": {k: hashlib.sha256(Path(v).read_bytes()).hexdigest() for k, v in annotation_files.items()},
             "scene_map_sha256": hashlib.sha256(Path(scene_map).read_bytes()).hexdigest()}
    atomic_write(destination, canonical(value))
    return value


def assign_clients(registry, output, domains, seed, alpha, max_passes, clients=4, equal=False):
    data = json.loads(Path(registry).read_bytes())
    targets = shared_targets(seed, clients, len(data["labels"]), alpha, equal)
    records = data["records"]
    group_context = defaultdict(set)
    for r in records:
        if r["role"] in ("D", "V", "A"):
            group_context[r["scene_id"]].add((r["role"], r["domain"]))
    if any(len(v) != 1 for v in group_context.values()):
        raise ValueError("Choose one online appearance/domain and one global role per whole scene before client assignment")
    orders = {}
    for i in range(clients):
        order = list(domains)
        random.Random(seed + i).shuffle(order)
        orders[str(i)] = order
    diagnostics = []
    for role in ("D", "V", "A"):
        for domain in domains:
            selected = [r for r in records if r["role"] == role and r["domain"] == domain]
            scenes = defaultdict(list)
            for record in selected:
                scenes[record["scene_id"]].append(record)
            groups = []
            for scene, rows in sorted(scenes.items()):
                counts = [0] * len(data["labels"])
                for r in rows:
                    for b in r["boxes"]:
                        counts[int(b[0])] += 1
                groups.append({"id": scene, "images": len(rows), "class_boxes": counts})
            result = assign_groups(groups, targets, seed, max_passes)
            for record in selected:
                client = str(result["assignment"][record["scene_id"]])
                record.update(client=client, stage=orders[client].index(domain) + 1)
            diagnostics.append({"role": role, "domain": domain, **result})
    data.update(assignment={"seed": seed, "targets": targets.tolist(), "orders": orders, "diagnostics": diagnostics})
    validate_registry(records, formal=True)
    # No missing cell is silently repaired by moving held-out scenes.
    counts = {f"{i}/{t}/{r}": sum(str(x["client"]) == str(i) and x["stage"] == t and x["role"] == r for x in records)
              for i in range(clients) for t in range(1, len(domains) + 1) for r in ("D", "V", "A")}
    if any(v == 0 for v in counts.values()):
        raise ValueError("Insufficient disjoint scene groups for every client-stage-role: " + str(counts))
    data["executed_split_counts"] = counts
    atomic_write(output, canonical(data))
    return data


def prepare_base(config):
    """Explicit command: prepare once on SOURCE/DEV, never formal online D/V/A/T."""
    from ultralytics import YOLO
    from .runtime import RuntimeSpec
    manifest = json.loads(Path(config["source_registry"]).read_bytes())
    if tuple(manifest["labels"]) != CLASSES or any(r["role"] not in ("SOURCE", "DEV") for r in manifest["records"]):
        raise ValueError("Base preparation needs separate source/development-only eight-class data")
    runtime = RuntimeSpec(**config["runtime"])
    runtime.activate()
    data_yaml = materialize_source_dataset(manifest, Path(config["output_root"]) / (config["name"] + "_source_data"))
    # Standard Ultralytics training prepares the frozen base only. SoftCAS updates
    # never call this high-level trainer or allow it to unfreeze the detector.
    model = YOLO(config["architecture"])
    if config.get("initial_weights"):
        model.load(config["initial_weights"])
    results = model.train(data=str(data_yaml), epochs=config["epochs"], imgsz=config["image_size"],
                          device=runtime.device, batch=config["batch_size"], workers=config["workers"],
                          project=config["output_root"], name=config["name"], exist_ok=False,
                          seed=config["seed"], deterministic=runtime.deterministic,
                          amp=runtime.precision != "float32", **config.get("trainer", {}))
    checkpoint = Path(model.trainer.best)
    frozen = YOLO(str(checkpoint)).model
    if tuple(frozen.names.values()) != CLASSES:
        raise ValueError("Prepared detector head has the wrong class vocabulary/order")
    metadata = {"checkpoint": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "labels": CLASSES, "source_registry_sha256": hashlib.sha256(Path(config["source_registry"]).read_bytes()).hexdigest(),
                "config": config, "runtime": runtime.identity(), "frozen_for_adaptation": True}
    atomic_write(checkpoint.with_suffix(".softcas.json"), canonical(metadata))
    return metadata


def materialize_source_dataset(manifest, output):
    """Materialize only SOURCE/DEV under the output root; never writes beside originals."""
    import os
    import shutil
    import yaml
    output = Path(output).resolve()
    lists = {"SOURCE": [], "DEV": []}
    for r in manifest["records"]:
        if r["role"] not in lists:
            raise ValueError("Online/test records are forbidden during frozen-base preparation")
        source = Path(r["image"])
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != r["image_sha256"]:
            raise ValueError("Source image checksum mismatch")
        filename = r["image_sha256"] + source.suffix.lower()
        target = output / "images" / r["role"] / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        label = output / "labels" / r["role"] / (r["image_sha256"] + ".txt")
        atomic_write(label, ("\n".join(" ".join(str(v) for v in b) for b in r["boxes"]) + "\n").encode())
        lists[r["role"]].append(str(target))
    if not lists["SOURCE"] or not lists["DEV"]:
        raise ValueError("Declare separate source training and development validation scenes")
    if {r["scene_id"] for r in manifest["records"] if r["role"] == "SOURCE"} & {r["scene_id"] for r in manifest["records"] if r["role"] == "DEV"}:
        raise ValueError("Source/development scene leakage")
    for role, images in lists.items():
        atomic_write(output / (role + ".txt"), ("\n".join(images) + "\n").encode())
    config = {"path": str(output), "train": str(output / "SOURCE.txt"), "val": str(output / "DEV.txt"),
              "names": dict(enumerate(manifest["labels"]))}
    destination = output / "data.yaml"
    atomic_write(destination, yaml.safe_dump(config, allow_unicode=True).encode())
    return destination


def relocate_registry(source, destination, replacements):
    """Path relocation changes no image/scene/role/annotation identity."""
    document = json.loads(Path(source).read_bytes())
    for r in document["records"]:
        name = r["image"].replace("\\", "/")
        for old, new in sorted(replacements.items(), key=lambda p: -len(p[0])):
            old = old.replace("\\", "/").rstrip("/")
            if name == old or name.startswith(old + "/"):
                r["image"] = new.rstrip("/") + name[len(old):]
                break
    atomic_write(destination, canonical(document))
    return document


def convert_cityscapes_polygons(spec):
    """Explicit polygon-envelope box derivation for ControlledFog preparation.

    The author supplies appearance/scene/role correspondence. Other rendered
    appearances are not automatically duplicated into online stages.
    """
    rows = json.loads(Path(spec["appearance_scene_map"]).read_bytes())
    records = []
    for row in rows:
        annotation = json.loads(Path(row["polygon_annotation"]).read_bytes())
        width, height = annotation["imgWidth"], annotation["imgHeight"]
        boxes = []
        for obj in annotation["objects"]:
            if obj.get("deleted", False):
                continue
            label = obj["label"]
            if label.endswith("group") or label not in CLASSES:
                continue
            polygon = obj["polygon"]
            if not polygon:
                continue
            x0, x1 = min(p[0] for p in polygon), max(p[0] for p in polygon)
            y0, y1 = min(p[1] for p in polygon), max(p[1] for p in polygon)
            x0, x1 = max(0, x0), min(width, x1)
            y0, y1 = max(0, y0), min(height, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            boxes.append([CLASSES.index(label), (x0 + x1) / (2 * width), (y0 + y1) / (2 * height),
                          (x1 - x0) / width, (y1 - y0) / height])
        path = Path(row["image"]).resolve()
        records.append({"image": str(path), "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "scene_id": row["scene_id"], "role": row["role"], "domain": row["domain"],
                        "client": row.get("client", "unassigned"), "stage": row.get("stage", 0),
                        "verified_grouping": row.get("verified_grouping", False), "boxes": boxes,
                        "source_annotation": str(Path(row["polygon_annotation"]).resolve()),
                        "box_derivation": "visible_polygon_envelope_continuous_coordinates_no_plus_one"})
    document = {"schema": 1, "labels": CLASSES, "records": records,
                "annotation_kind": "derived_detection_boxes_from_cityscapes_polygons",
                "scene_map_sha256": hashlib.sha256(Path(spec["appearance_scene_map"]).read_bytes()).hexdigest()}
    atomic_write(spec["output"], canonical(document))
    return document
