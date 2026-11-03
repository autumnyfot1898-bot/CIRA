"""Role/scene gates and explicit current-stage batches. T stays outside search."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import torch
from .state import digest, freeze_tensors, canonical


@dataclass(frozen=True)
class Split:
    role: str
    client: str
    stage: int
    domain: str
    batches: tuple
    manifest: str

    @property
    def identity(self):
        # Include actual batch contents, not only a user-supplied display ID.
        return digest({"role": self.role, "client": self.client, "stage": self.stage,
                       "domain": self.domain, "manifest": self.manifest,
                       "batches": [[(k, v.__dict__) for k, v in freeze_tensors(b)] for b in self.batches]})


@dataclass(frozen=True)
class StageData:
    D: Split
    V: Split
    A: Split

    def __post_init__(self):
        if (self.D.role, self.V.role, self.A.role) != ("D", "V", "A"):
            raise ValueError("Strict D/V/A roles required")
        contexts = {(s.client, s.stage, s.domain) for s in (self.D, self.V, self.A)}
        if len(contexts) != 1 or any(not s.batches for s in (self.D, self.V, self.A)):
            raise ValueError("Only current stage, nonempty D/V/A may enter online search")


def validate_registry(records, formal=False):
    """Scene_id must already merge clips, counterparts and rendered variants.

    We cannot infer correct scene identity from folder names. Formal execution
    requires explicit verified_grouping and never silently upgrades candidate data.
    """
    roles, online, images, hashes = {}, {}, set(), set()
    for r in records:
        required = ("image", "image_sha256", "scene_id", "role", "domain", "client", "stage")
        if any(k not in r for k in required) or r["role"] not in ("DEV", "D", "V", "A", "T"):
            raise ValueError("Incomplete scene/role registry")
        if formal and not r.get("verified_grouping", False):
            raise ValueError("Formal execution requires verified scene/counterpart grouping")
        if r["image"] in images or r["image_sha256"] in hashes:
            raise ValueError("Duplicate image/derived appearance")
        images.add(r["image"])
        hashes.add(r["image_sha256"])
        scene = r["scene_id"]
        if scene in roles and roles[scene] != r["role"]:
            raise ValueError("Global cross-client scene role leakage")
        roles[scene] = r["role"]
        if r["role"] in ("D", "V", "A"):
            context = (r["client"], r["stage"], r["domain"])
            if scene in online and online[scene] != context:
                raise ValueError("Online scene reused across stages/clients/appearances")
            online[scene] = context
    return {"records": len(records), "scenes": len(roles), "hash": digest(records), "formal": formal}


def synthetic_split(role, client, stage, domain, seed, batches=1):
    g = torch.Generator().manual_seed(seed)
    values = tuple({"img": torch.randn(2, 3, 8, 8, generator=g),
                    "target": torch.randn(2, 2, generator=g)} for _ in range(batches))
    return Split(role, client, stage, domain, values,
                 canonical({"kind": "SYNTHETIC_BEHAVIOR_ONLY", "seed": seed, "role": role}).decode())


def _legacy_load_detection_stage(manifest_path, client, stage, image_size, labels, formal=False):
    """Load a verified explicit registry; no automatic D/V/A/T reassignment.

    Boxes use [class_id,cx,cy,w,h] normalized to the original RGB image.
    Letterbox (114), bilinear RGB resize, float32/255 is recorded in the base
    preprocessing identity. Empty annotation lists are retained.
    """
    import hashlib
    import numpy as np
    from PIL import Image
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if tuple(manifest["labels"]) != tuple(labels):
        raise ValueError("Manifest label vocabulary differs from frozen head")
    records = manifest["records"]
    validate_registry(records, formal)
    roles = {r: [] for r in ("D", "V", "A")}
    role_records = {r: [] for r in roles}
    domains = set()
    for record in records:
        if record["role"] not in roles or str(record["client"]) != str(client) or record["stage"] != stage:
            continue
        image_path = Path(record["image"])
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != record["image_sha256"]:
            raise ValueError("Image content does not match registry checksum")
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        factor = min(image_size / width, image_size / height)
        new_w, new_h = round(width * factor), round(height * factor)
        left, top = (image_size - new_w) // 2, (image_size - new_h) // 2
        canvas = Image.new("RGB", (image_size, image_size), (114, 114, 114))
        canvas.paste(image.resize((new_w, new_h), Image.Resampling.BILINEAR), (left, top))
        img = torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).float()[None] / 255
        boxes = torch.tensor(record["boxes"], dtype=torch.float32).reshape(-1, 5)
        if len(boxes):
            if (boxes[:, 0] < 0).any() or (boxes[:, 0] >= len(labels)).any() or not torch.equal(boxes[:, 0], boxes[:, 0].round()):
                raise ValueError("Class ID outside fixed vocabulary")
            if not torch.isfinite(boxes).all() or (boxes[:, 1:] < 0).any() or (boxes[:, 1:] > 1).any() or (boxes[:, 3:] <= 0).any():
                raise ValueError("Invalid normalized detection box")
            boxes[:, 1] = (boxes[:, 1] * new_w + left) / image_size
            boxes[:, 2] = (boxes[:, 2] * new_h + top) / image_size
            boxes[:, 3] *= new_w / image_size
            boxes[:, 4] *= new_h / image_size
        roles[record["role"]].append({"img": img, "batch_idx": torch.zeros(len(boxes)),
                                       "cls": boxes[:, :1], "bboxes": boxes[:, 1:]})
        role_records[record["role"]].append(record)
        domains.add(record["domain"])
    if len(domains) != 1:
        raise ValueError("Stage must have exactly one supplied domain")
    domain = next(iter(domains))
    return StageData(*(Split(r, str(client), stage, domain, tuple(roles[r]),
                            canonical({"labels": labels, "records": role_records[r]}).decode()) for r in ("D", "V", "A")))


def detection_record_batch(record, root, image_size, *, noise_sigma=0.0, noise_seed=0):
    import hashlib
    import numpy as np
    from PIL import Image
    path = Path(record["image"])
    path = path if path.is_absolute() else Path(root) / path
    if hashlib.sha256(path.read_bytes()).hexdigest() != record["image_sha256"]:
        raise ValueError("Image content differs from immutable registry: " + str(path))
    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = image.size
    if (isinstance(noise_sigma, bool) or not isinstance(noise_sigma, (int, float)) or
            not __import__("math").isfinite(noise_sigma) or noise_sigma < 0):
        raise ValueError("Noise sigma must be finite and nonnegative")
    if not isinstance(noise_seed, int) or isinstance(noise_seed, bool):
        raise ValueError("Noise seed must be an integer")
    ratio = min(image_size / width, image_size / height)
    nw, nh = round(width * ratio), round(height * ratio)
    left, top = (image_size - nw) // 2, (image_size - nh) // 2
    pixels = np.asarray(image).astype(np.float32) / np.float32(255)
    if noise_sigma:
        # No method, profile, node, or file path enters the shared image key.
        key = hashlib.sha256(canonical({"seed": noise_seed, "image": record["image_sha256"],
                                        "sigma": float(noise_sigma)})).digest()
        rng = np.random.Generator(np.random.PCG64(int.from_bytes(key[:8], "big")))
        pixels = np.clip(pixels + rng.normal(0, noise_sigma, pixels.shape).astype(np.float32), 0, 1)
    tensor = torch.from_numpy(pixels.copy()).permute(2, 0, 1)[None]
    resized = torch.nn.functional.interpolate(tensor, size=(nh, nw), mode="bilinear", align_corners=False)
    image_tensor = torch.nn.functional.pad(resized, (left, image_size-nw-left, top, image_size-nh-top),
                                           value=114/255)
    boxes = torch.tensor(record["boxes"], dtype=torch.float32).reshape(-1, 5)
    if len(boxes):
        if not torch.isfinite(boxes).all() or (boxes[:, 1:] < 0).any() or (boxes[:, 1:] > 1).any() or (boxes[:, 3:] <= 0).any():
            raise ValueError("Invalid normalized boxes")
        boxes[:, 1] = (boxes[:, 1] * nw + left) / image_size
        boxes[:, 2] = (boxes[:, 2] * nh + top) / image_size
        boxes[:, 3] *= nw / image_size
        boxes[:, 4] *= nh / image_size
    return {"img": image_tensor, "batch_idx": torch.zeros(len(boxes)), "cls": boxes[:, :1], "bboxes": boxes[:, 1:]}


class DetectionBatches:
    def __init__(self, records, root, image_size, batch_size):
        self.records, self.root, self.image_size, self.batch_size = records, root, image_size, batch_size
        if batch_size < 1:
            raise ValueError("Positive batch size required")

    def __len__(self):
        return (len(self.records) + self.batch_size - 1) // self.batch_size

    def shuffled(self, order):
        if sorted(order) != list(range(len(self.records))):
            raise ValueError("An epoch must contain every fitting record exactly once")
        return DetectionBatches(tuple(self.records[i] for i in order), self.root,
                                self.image_size, self.batch_size)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        records = self.records[index * self.batch_size:(index + 1) * self.batch_size]
        batches = [detection_record_batch(r, self.root, self.image_size) for r in records]
        return {"img": torch.cat([b["img"] for b in batches]),
                "batch_idx": torch.cat([b["batch_idx"] + i for i, b in enumerate(batches)]),
                "cls": torch.cat([b["cls"] for b in batches]), "bboxes": torch.cat([b["bboxes"] for b in batches])}


class ManifestSplit:
    def __init__(self, role, client, stage, domain, records, root, labels, image_size, batch_size, strict_identity=True):
        self.role, self.client, self.stage, self.domain = role, str(client), stage, domain
        self.records, self.root, self.labels = tuple(records), Path(root), tuple(labels)
        self.image_size, self.batch_size, self.strict_identity = image_size, batch_size, strict_identity
        self.batches = DetectionBatches(self.records, self.root, image_size, batch_size)
        self.manifest = canonical({"labels": labels, "records": records}).decode()
        self._stats = None
        if not records:
            raise ValueError("Missing required " + role + " data support")

    @property
    def identity(self):
        import hashlib
        stats = []
        for record in self.records:
            path = Path(record["image"])
            path = path if path.is_absolute() else self.root / path
            stat = path.stat()
            stats.append((record["image_sha256"], stat.st_size, stat.st_mtime_ns))
            if self.strict_identity or self._stats is None or tuple(stats) != self._stats[:len(stats)]:
                if hashlib.sha256(path.read_bytes()).hexdigest() != record["image_sha256"]:
                    raise ValueError("Data identity changed after registry construction")
        self._stats = tuple(stats)
        portable_records = [{k: v for k, v in r.items() if k not in ("image", "source_annotation")} for r in self.records]
        return digest({"role": self.role, "domain": self.domain, "records": portable_records,
                       "labels": self.labels, "image_size": self.image_size, "batch_size": self.batch_size,
                       "preprocessing": "rgb_float32_div255_clip_bilinear_letterbox114_v2"})


def load_detection_split(manifest_path, client, stage, role, image_size, labels, formal=False, batch_size=1,
                         lazy=True, domain=None, strict_identity=True):
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if tuple(manifest["labels"]) != tuple(labels):
        raise ValueError("Registry label order differs from frozen base")
    validate_registry(manifest["records"], formal)
    selected = []
    for r in manifest["records"]:
        if r["role"] != role:
            continue
        if role == "T":
            if str(r["client"]) not in (str(client), "shared", "*") or (domain is not None and r["domain"] != domain):
                continue
        elif str(r["client"]) != str(client) or r["stage"] != stage:
            continue
        for box in r["boxes"]:
            if len(box) != 5 or box[0] != int(box[0]) or not 0 <= box[0] < len(labels):
                raise ValueError("Annotation outside fixed label vocabulary")
        selected.append(r)
    domains = {r["domain"] for r in selected}
    if len(domains) != 1:
        raise ValueError("Each requested online/test split must have exactly one identified domain")
    split = ManifestSplit(role, client, stage, next(iter(domains)),
                          sorted(selected, key=lambda r: canonical({k: v for k, v in r.items() if k not in ("image", "source_annotation")})),
                          path.parent, labels, image_size, batch_size, strict_identity)
    if lazy:
        return split
    return Split(role, str(client), stage, split.domain, tuple(split.batches), split.manifest)


def load_detection_stage(manifest_path, client, stage, image_size, labels, formal=False, batch_size=1, lazy=True):
    return StageData(*(load_detection_split(manifest_path, client, stage, r, image_size, labels, formal,
                                           batch_size, lazy) for r in ("D", "V", "A")))
