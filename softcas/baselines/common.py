"""Native detector operations shared as infrastructure, not as an algorithm."""
from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math

import torch
from torch import nn


@dataclass(frozen=True)
class Batches:
    role: str
    values: tuple

    def require(self, role):
        if self.role != role or not self.values:
            raise ValueError('Nonempty ' + role + ' batches required for this operation')
        return self.values


@contextmanager
def evaluating(model):
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for module, mode in modes:
            module.training = mode


def snapshot(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def model_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def trainable(model, enabled=True):
    model.requires_grad_(enabled)
    if enabled:
        for name, parameter in model.named_parameters():
            if '.dfl.' in name:
                parameter.requires_grad_(False)


def native_loss(model, batch, predictions=None):
    if predictions is None:
        predictions = model(batch['img'])
        if not model.training:
            predictions = predictions[1]
    losses, _ = model.loss(batch, preds=predictions)
    return losses.sum() / batch['img'].shape[0]


def fit(model, data, steps, learning_rate, *, loss_fn=native_loss, extra_loss=None,
        masks=None, momentum=0.0, weight_decay=0.0):
    batches = data.require('D')
    if type(steps) is not int or steps < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Positive fitting steps and learning rate required')
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    losses = []
    gradients = {}
    for step in range(steps):
        batch = batches[step % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model, batch)
        if extra_loss is not None:
            loss = loss + extra_loss(model, batch)
        if loss.numel() != 1 or not torch.isfinite(loss):
            raise ValueError('Nonfinite/scalar detector training objective')
        loss.backward()
        gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
        if any(not torch.isfinite(g).all() for g in gradients.values()):
            raise ValueError('Nonfinite detector gradient')
        optimizer.step()
        if masks is not None:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in masks:
                        parameter.mul_(masks[name].to(parameter.device))
        losses.append(float(loss.detach()))
    optimizer.zero_grad(set_to_none=True)
    return {'losses': losses, 'gradients': gradients, 'fitting_steps': steps}


def validation_loss(model, data, loss_fn=native_loss):
    batches = data.require('V')
    with evaluating(model), torch.no_grad():
        values = [float(loss_fn(model, batch)) for batch in batches]
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Nonfinite detector selection objective')
    return sum(values) / len(values)


class DetectorTask:
    """Ultralytics 8.4 raw-head adaptation, including YOLO26 one-to-one heads."""
    def __init__(self, feature_index=6):
        self.feature_index = feature_index

    loss = staticmethod(native_loss)

    def forward_features(self, model, images):
        observed = []
        handle = model.model[self.feature_index].register_forward_hook(lambda module, args, value: observed.append(value))
        try:
            predictions = model(images)
        finally:
            handle.remove()
        if len(observed) != 1 or not isinstance(observed[0], torch.Tensor) or observed[0].ndim != 4:
            raise ValueError('Declare a single four-dimensional native feature layer')
        return predictions, observed[0].mean((2, 3))

    def features(self, model, images):
        with evaluating(model):
            return self.forward_features(model, images)[1]

    def loss_features(self, model, batch):
        raw, features = self.forward_features(model, batch['img'])
        if not model.training:
            raw = raw[1]
        return native_loss(model, batch, raw), features

    @staticmethod
    def head_features(model, images):
        observed = []
        head = model.model[-1]
        handle = head.register_forward_pre_hook(lambda module, args: observed.append(list(args[0])))
        try:
            predictions = model(images)
        finally:
            handle.remove()
        if len(observed) != 1:
            raise ValueError('Native detector must execute its declared head exactly once')
        return predictions, observed[0]

    @staticmethod
    def dense(model, images):
        with evaluating(model):
            output = model(images)
        head = model.model[-1]
        raw = output[1]
        return head._inference(raw['one2one'] if head.end2end else raw)

    @staticmethod
    def predictions_from_dense(model, dense):
        from ..postprocess import decode
        head = model.model[-1]
        if head.end2end:
            return decode(head.postprocess(dense.permute(0, 2, 1)), head.nc, end2end=True)
        return decode(dense, head.nc, end2end=False)

    @staticmethod
    def predictions_from_raw(model, raw):
        head = model.model[-1]
        dense = head._inference(raw['one2one'] if head.end2end else raw)
        return DetectorTask.predictions_from_dense(model, dense)

    def ap(self, model, data, role='V', predict=None):
        from ..evaluation import detection_ap_torch
        if role not in ('V','A','T'):
            raise ValueError('AP evaluation requires an explicit V, A or T role')
        predictions, targets = [], []
        with evaluating(model), torch.no_grad():
            for batch in data.require(role):
                if predict is None:
                    dense = self.dense(model,batch['img'])
                    decoded = self.predictions_from_dense(model,dense)
                else:decoded = predict(batch['img'])
                predictions.extend([value.detach().float().cpu() for value in decoded])
                height,width = batch['img'].shape[-2:]
                for index in range(batch['img'].shape[0]):
                    mask = batch['batch_idx']==index
                    boxes = batch['bboxes'][mask]
                    center,half = boxes[:,:2],boxes[:,2:]/2
                    xyxy = torch.cat([center-half,center+half],dim=1)*boxes.new_tensor([width,height,width,height])
                    targets.append(torch.cat([batch['cls'][mask].reshape(-1,1),xyxy],dim=1).detach().float().cpu())
        return detection_ap_torch(predictions,targets)


def restore_models(models, states):
    if len(models) != len(states):
        raise ValueError('Checkpoint client count differs')
    # Validate shapes on replicas first, avoiding a partially restored federation.
    replicas = [copy.deepcopy(model) for model in models]
    for model, state in zip(replicas, states):
        model.load_state_dict(state, strict=True)
    for model, state in zip(models, states):
        model.load_state_dict(state, strict=True)


def parameter_vector(model):
    return torch.cat([p.detach().float().cpu().flatten() for p in model.parameters()])


def independent_models(models):
    if len(models) < 1:
        raise ValueError('Nonempty model federation required')
    pointers = set()
    for model in models:
        for value in model.parameters():
            address = (value.device.type, value.device.index, value.data_ptr())
            if address in pointers:
                raise ValueError('Client detectors must not share parameter storage')
            pointers.add(address)
