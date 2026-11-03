"""Fixed, deterministic detector postprocessing without time-truncated NMS."""
from __future__ import annotations
import torch


def fixed_nms(boxes, scores, classes, iou_threshold, maximum):
    if not len(boxes):
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    order = torch.argsort(scores, descending=True, stable=True)
    area = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(1)
    kept = []
    while order.numel() and len(kept) < maximum:
        current = order[0]
        kept.append(current)
        others = order[1:]
        if not others.numel():
            break
        lower = torch.maximum(boxes[current, :2], boxes[others, :2])
        upper = torch.minimum(boxes[current, 2:], boxes[others, 2:])
        intersection = (upper - lower).clamp(min=0).prod(1)
        iou = intersection / (area[current] + area[others] - intersection + 1e-7)
        suppress = (iou > iou_threshold) & (classes[others] == classes[current])
        order = others[~suppress]
    return torch.stack(kept).long()


def decode(predictions, classes, end2end=False, confidence=0.001, iou=0.7, maximum=300, maximum_candidates=30000):
    raw = predictions[0] if isinstance(predictions, (tuple, list)) else predictions
    if end2end or raw.shape[-1] == 6:
        outputs = []
        for values in raw:
            values = values[values[:, 4] > confidence]
            order = torch.argsort(values[:, 4], descending=True, stable=True)[:maximum]
            outputs.append(values[order])
        return outputs
    if raw.ndim != 3 or raw.shape[1] != 4 + classes:
        raise ValueError("Pinned detector inference output contract changed")
    outputs = []
    for values in raw.transpose(1, 2):
        scores, labels = values[:, 4:].max(1)
        keep = scores > confidence
        values, scores, labels = values[keep], scores[keep], labels[keep]
        order = torch.argsort(scores, descending=True, stable=True)[:maximum_candidates]
        values, scores, labels = values[order], scores[order], labels[order]
        boxes = torch.cat((values[:, :2] - values[:, 2:4] / 2, values[:, :2] + values[:, 2:4] / 2), dim=1)
        selected = fixed_nms(boxes, scores, labels, iou, maximum)
        outputs.append(torch.cat((boxes[selected], scores[selected, None], labels[selected, None].float()), dim=1))
    return outputs
