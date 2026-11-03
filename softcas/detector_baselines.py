"""Full-detector DPFL/DSpodFL operations and a bounded native interface probe.

This module deliberately does not return a formal external study result. A probe
cannot qualify a baseline, create a DEV lock, or manufacture Orin calibration.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch

from .baseline_algorithms import average_states, dpfl_select, dspodfl_step

LABELS = ('person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')
ARCHITECTURES = {'YOLOv8n': 'yolov8n.yaml', 'YOLO11n': 'yolo11n.yaml', 'YOLO26n': 'yolo26n.yaml'}


def detector_loss(model, batch):
    """Ultralytics 8.4.0 native detection loss, normalized per image."""
    if model.training:
        losses, _ = model.loss(batch)
    else:
        predictions = model(batch['img'])
        if not isinstance(predictions, tuple) or len(predictions) < 2:
            raise RuntimeError('Pinned detector raw evaluation contract changed')
        losses, _ = model.loss(batch, preds=predictions[1])
    return losses.sum() / batch['img'].shape[0]


def new_detector(architecture, device, *, checkpoint=None, seed=0):
    import ultralytics
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel
    if ultralytics.__version__ != '8.4.0':
        raise RuntimeError('Detector ports require validated ultralytics==8.4.0')
    if architecture not in ARCHITECTURES:
        raise ValueError('Unknown registered detector architecture')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if checkpoint is None:
            model = DetectionModel(ARCHITECTURES[architecture], nc=len(LABELS), verbose=False)
        else:
            from ultralytics import YOLO
            path = Path(checkpoint)
            if not path.is_file():
                raise FileNotFoundError(path)
            model = YOLO(str(path)).model
            if tuple(model.names.values()) != LABELS:
                raise ValueError('Native checkpoint must have the fixed eight-class vocabulary')
            # Reject a checkpoint from a different architecture with coincident labels.
            expected = DetectionModel(ARCHITECTURES[architecture], nc=len(LABELS), verbose=False)
            shapes = lambda m: {n: tuple(t.shape) for n, t in m.state_dict().items()}
            if shapes(model) != shapes(expected):
                raise ValueError('Checkpoint architecture differs from the declared detector')
            del expected
    model.names = dict(enumerate(LABELS))
    model.args = get_cfg()
    model = model.float().to(device)
    model.requires_grad_(True)
    # The DFL projection is a fixed discretization, not a trainable detector layer.
    for name, parameter in model.named_parameters():
        if '.dfl.' in name:
            parameter.requires_grad_(False)
    return model


def dpfl_round(models, fit_batches, selection_batches, *, learning_rate, cardinality,
               variant, rng, neighborhoods=None, local_steps=1, client_weights=None):
    """Local detector SGD, receiver-V subset selection, then personalized averaging.

Every oracle uses immutable post-fit snapshots. A receiver's BN integer counter
remains local. All model copies are independent. This provides the round core;
continual stage history, contact enforcement and budget admission belong to the
study runner and must be qualified before using this as a formal comparison.
"""
    n = len(models)
    if not n or len(fit_batches) != n or len(selection_batches) != n:
        raise ValueError('Every client requires its own fitting and selection batches')
    if not math.isfinite(learning_rate) or learning_rate <= 0 or type(local_steps) is not int or local_steps < 1:
        raise ValueError('Positive learning rate and integer local step count required')
    neighborhoods = [list(range(n)) for _ in models] if neighborhoods is None else neighborhoods
    if len(neighborhoods) != n or any(i not in row or len(row) != len(set(row)) or any(type(j) is not int or not 0 <= j < n for j in row) for i, row in enumerate(neighborhoods)):
        raise ValueError('Explicit valid neighborhood including self required for every client')
    if not 1 <= cardinality <= min(len(row) for row in neighborhoods):
        raise ValueError('Subset cardinality exceeds an available neighborhood')
    client_weights = [1.0] * n if client_weights is None else list(client_weights)
    if len(client_weights) != n or any(not math.isfinite(w) or w <= 0 for w in client_weights):
        raise ValueError('Positive finite per-client sample weights required')
    local_losses = []
    for model, batch in zip(models, fit_batches):
        model.train()
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
        for _ in range(local_steps):
            optimizer.zero_grad(set_to_none=True)
            loss = detector_loss(model, batch)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite local detector loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Nonfinite local detector gradient')
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        local_losses.append(float(loss.detach()))
    snapshots = [{k: v.detach().clone() for k, v in m.state_dict().items()} for m in models]
    chosen, oracle_records, updates = [], [], []
    for i, model in enumerate(models):
        replica = copy.deepcopy(model).eval()
        def oracle(group):
            replica.load_state_dict(average_states([snapshots[j] for j in group],
                [client_weights[j] for j in group], buffer_reference=group.index(i)))
            with torch.no_grad():
                loss = detector_loss(replica, selection_batches[i])
            return (10.0 - float(loss)) / 10.0
        selection = dpfl_select(i, neighborhoods[i], cardinality, oracle, variant=variant, rng=rng)
        chosen.append(list(selection.clients))
        oracle_records.append([{'subset': list(group), 'reward': reward} for group, reward in selection.oracle_calls])
        updates.append(average_states([snapshots[j] for j in selection.clients],
            [client_weights[j] for j in selection.clients], buffer_reference=selection.clients.index(i)))
        del replica
    for model, state in zip(models, updates):
        model.load_state_dict(state)
    return {'local_losses': local_losses, 'selected_clients': chosen, 'selection_oracles': oracle_records,
            'selection_metric': '(10 - receiver V detection loss) / 10',
            'selection_variant': variant, 'counter_policy': 'receiver-local integer BN counters',
            'client_averaging_weights': client_weights,
            'communication_scope': 'in-process native round probe, not physical transmission'}


def _load_batch(records, size, device):
    from PIL import Image, ImageOps
    import numpy as np
    images, classes, boxes, indices = [], [], [], []
    for index, record in enumerate(records):
        path = Path(record['image'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('Probe image content changed')
        # Fixed square letterbox; labels originate from normalized original boxes.
        image = Image.open(path).convert('RGB')
        width, height = image.size
        gain = min(size / width, size / height)
        resized = (round(width * gain), round(height * gain))
        image = image.resize(resized, Image.Resampling.BILINEAR)
        left, top = (size - resized[0]) // 2, (size - resized[1]) // 2
        image = ImageOps.expand(image, (left, top, size - resized[0] - left, size - resized[1] - top), fill=(114, 114, 114))
        images.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255)
        for cls, cx, cy, w, h in record['boxes']:
            if type(cls) is not int or not 0 <= cls < 8 or not all(math.isfinite(x) for x in (cx, cy, w, h)):
                raise ValueError('Invalid eight-class probe labels')
            if w <= 0 or h <= 0 or min(cx - w / 2, cy - h / 2) < -1e-6 or max(cx + w / 2, cy + h / 2) > 1 + 1e-6:
                raise ValueError('Invalid normalized probe box')
            # Rounding can give a slightly different actual horizontal/vertical gain.
            sx, sy = resized[0] / size, resized[1] / size
            boxes.append([cx * sx + left / size, cy * sy + top / size, w * sx, h * sy])
            classes.append([cls])
            indices.append(index)
    if not images:
        raise ValueError('Nonempty probe image batch required')
    return {'img': torch.stack(images).to(device),
            'cls': torch.tensor(classes, dtype=torch.float32, device=device).reshape(-1, 1),
            'bboxes': torch.tensor(boxes, dtype=torch.float32, device=device).reshape(-1, 4),
            'batch_idx': torch.tensor(indices, dtype=torch.int64, device=device)}


def run_probe(manifest_path, output, *, architecture='YOLOv8n', device='cuda:0', size=640, seed=1701):
    """Actual four-replica native loss/update/decode; NOT a performance experiment."""
    from .postprocess import decode
    from .storage import atomic_write
    manifest_path, output = Path(manifest_path), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new or empty output directory')
    if size < 64 or size % 32:
        raise ValueError('Probe input size must be a multiple of 32 and at least 64')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('scope') != 'native_interface_probe_NOT_FORMAL' or len(manifest.get('clients', [])) != 4:
        raise ValueError('Explicit four-client non-formal probe manifest required')
    output.mkdir(parents=True, exist_ok=True)
    report = {'schema': 'cira.native_baseline_probe.v1', 'scope': 'native_interface_probe_NOT_FORMAL',
              'status': 'started', 'architecture': architecture, 'seed': seed, 'input_size': size,
              'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              'torch': torch.__version__, 'device': str(device),
              'initialization': 'random eight-class detector; NOT SOURCE-trained',
              'formal_AP': None, 'native_DEV_lock': None, 'Orin_measurements': None, 'methods': {}}
    report['source_files_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('baseline_algorithms.py', 'detector_baselines.py', 'postprocess.py')}
    started = time.perf_counter()
    try:
        torch.manual_seed(seed)
        if str(device).startswith('cuda'):
            report['device_name'] = torch.cuda.get_device_name(device)
        fit, select = [], []
        for client in manifest['clients']:
            if not client.get('fit') or not client.get('select') or set(x['sha256'] for x in client['fit']) & set(x['sha256'] for x in client['select']):
                raise ValueError('Distinct nonempty probe fit/select images required')
            fit.append(_load_batch(client['fit'], size, device))
            select.append(_load_batch(client['select'], size, device))
        base = new_detector(architecture, device, seed=seed)
        initial = {name: value.detach().clone() for name, value in base.named_parameters()}
        # Hash/checkpoint are real initialized weights, not reported as SOURCE assets.
        checkpoint = output / 'random_init_NOT_SOURCE.pt'
        torch.save({'state_dict': base.state_dict(), 'architecture': architecture, 'labels': LABELS,
                    'scope': report['scope'], 'seed': seed}, checkpoint)
        report['initial_checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        graph = [[i != j for j in range(4)] for i in range(4)]
        for method in ('DSpodFL', 'DPFL'):
            models = [copy.deepcopy(base) for _ in range(4)]
            for model in models:
                model.train()
            if method == 'DSpodFL':
                result = dspodfl_step(models, fit, detector_loss, graph, graph, [True] * 4, 1e-4)
            else:
                result = dpfl_round(models, fit, select, learning_rate=1e-4, cardinality=2,
                                    variant='rgl', rng=random.Random(seed))
            result['detectors'] = []
            for i, model in enumerate(models):
                model.eval()
                with torch.no_grad():
                    predictions = decode(model(select[i]['img']), 8, end2end=getattr(model, 'end2end', False))
                if any(not torch.isfinite(values).all() or (len(values) and (values[:, 5].min() < 0 or values[:, 5].max() >= 8)) for values in predictions):
                    raise ValueError('Invalid native decoded detections')
                delta = sum(float((value.detach() - initial[name]).abs().sum()) for name, value in model.named_parameters())
                if delta <= 0:
                    raise ValueError('Detector parameters did not update')
                target = output / (method + '_client_' + str(i) + '.pt')
                torch.save({'state_dict': model.state_dict(), 'architecture': architecture, 'labels': LABELS,
                            'scope': report['scope'], 'method': method, 'client': i}, target)
                restored = torch.load(target, map_location='cpu', weights_only=True)
                if any(not torch.equal(value.detach().cpu(), restored['state_dict'][name])
                       for name, value in model.state_dict().items()):
                    raise RuntimeError('Native detector checkpoint roundtrip differs')
                result['detectors'].append({'client': i, 'parameter_L1_change': delta,
                    'decoded_counts': [len(values) for values in predictions],
                    'checkpoint_roundtrip_passed': True,
                    'checkpoint': str(target.resolve()), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
            report['methods'][method] = result
            del models
        report['status'] = 'passed_native_loss_update_decode_probe'
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        if str(device).startswith('cuda'):
            torch.cuda.synchronize(device)
        report['elapsed_wall_seconds'] = time.perf_counter() - started
        atomic_write(output / 'probe_report.json', json.dumps(report, indent=2).encode())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--architecture', choices=ARCHITECTURES, default='YOLOv8n')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--image-size', type=int, default=640)
    parser.add_argument('--seed', type=int, default=1701)
    args = parser.parse_args()
    result = run_probe(args.manifest, args.output, architecture=args.architecture,
                       device=args.device, size=args.image_size, seed=args.seed)
    print(json.dumps({k: result[k] for k in ('status', 'scope', 'architecture', 'elapsed_wall_seconds')}))


if __name__ == '__main__':
    main()
