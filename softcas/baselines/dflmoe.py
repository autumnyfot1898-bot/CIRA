"""dFLMoE detector port: feature transforms, head exchange and cross attention.

CVPR 2025 Eq. 2--6. The explicit detector adaptation fuses aligned native
regression/class logits before loss/decode, not unrelated final bounding boxes.
"""
from __future__ import annotations
import copy
import math

import torch
from torch import nn

from .common import DetectorTask, independent_models, native_loss, snapshot, trainable


class MoEClient(nn.Module):
    def __init__(self, owner, models):
        super().__init__()
        self.owner, self.client_ids = owner, list(range(len(models)))
        self.local_model = models[owner]
        head = self.local_model.model[-1]
        channels = []
        for layer in head.cv2:
            # First Conv wrapper exposes its input Conv2d channels.
            conv = next(module for module in layer.modules() if isinstance(module, nn.Conv2d))
            channels.append(conv.in_channels)
        device = next(self.local_model.parameters()).device
        self.common = nn.ModuleList([nn.Conv2d(c, c, 1) for c in channels]).to(device)
        self.transforms = nn.ModuleDict({str(i): nn.ModuleList([nn.Conv2d(c, c, 1) for c in channels])
                                        for i in self.client_ids}).to(device)
        self.query = nn.ModuleList([nn.Conv2d(c, head.no, 1) for c in channels]).to(device)
        self.experts = nn.ModuleDict({str(i): copy.deepcopy(model.model[-1]).eval().requires_grad_(False)
                                     for i, model in enumerate(models) if i != owner})
        self.active = list(self.client_ids)

    def transform_parameters(self):
        return list(self.common.parameters()) + list(self.transforms.parameters()) + list(self.query.parameters())

    def _branch(self, features, branch):
        selected = [value.detach() for value in features] if branch == 'one2one' else features
        common = [layer(value) for layer, value in zip(self.common, selected)]
        outputs = []
        local_head = self.local_model.model[-1]
        for client in self.active:
            mapped = [layer(value) for layer, value in zip(self.transforms[str(client)], common)]
            head = local_head if client == self.owner else self.experts[str(client)]
            raw = head.forward_head(mapped, **getattr(head, branch))
            outputs.append(torch.cat((raw['boxes'], raw['scores']), dim=1))
        # [B,E,C,A]: attention over experts for each aligned detection location.
        values = torch.stack(outputs, dim=1)
        query = torch.cat([layer(value).flatten(2) for layer, value in zip(self.query, selected)], dim=2)
        logits = (values * query[:, None]).sum(2) / math.sqrt(query.shape[1])
        weights = logits.softmax(dim=1)
        fused = (values * weights[:, :, None]).sum(1)
        split = local_head.reg_max * 4
        return {'boxes': fused[:, :split], 'scores': fused[:, split:], 'feats': selected}, weights

    def raw(self, features):
        head = self.local_model.model[-1]
        many, weights = self._branch(features, 'one2many')
        if head.end2end:
            one, one_weights = self._branch(features, 'one2one')
            return {'one2many': many, 'one2one': one}, {'one2many': weights, 'one2one': one_weights}
        return many, {'one2many': weights}

    def phase(self, local):
        trainable(self.local_model, local)
        self.local_model.train(local)
        for parameter in self.transform_parameters():
            parameter.requires_grad_(not local)
        for expert in self.experts.values():
            expert.eval().requires_grad_(False)

    def exchange_heads(self, models, available=None):
        self.active = [self.owner]
        for client, model in enumerate(models):
            if client != self.owner and (available is None or available[self.owner][client]):
                self.experts[str(client)].load_state_dict(snapshot(model.model[-1]))
                self.experts[str(client)].eval().requires_grad_(False)
                self.active.append(client)
        self.active.sort()

    def predict(self, images):
        self.local_model.eval()
        with torch.no_grad():
            _, features = DetectorTask.head_features(self.local_model, images)
            raw, _ = self.raw(features)
            return DetectorTask.predictions_from_raw(self.local_model, raw)


class DFLMoE:
    method = 'dFLMoE'
    def __init__(self, models, *, learning_rate, local_steps, moe_steps,
                 local_loss_weight=1., moe_loss_weight=1.):
        independent_models(models)
        if learning_rate <= 0 or type(local_steps) is not int or local_steps < 1 or type(moe_steps) is not int or moe_steps < 1:
            raise ValueError('Positive learning rate and both training phase lengths required')
        if min(local_loss_weight, moe_loss_weight) < 0 or local_loss_weight + moe_loss_weight == 0:
            raise ValueError('Nonnegative dFLMoE loss weights with a positive sum required')
        self.models = models
        self.clients = nn.ModuleList([MoEClient(i, models) for i in range(len(models))])
        self.lr, self.local_steps, self.moe_steps = learning_rate, local_steps, moe_steps
        self.local_weight, self.moe_weight = local_loss_weight, moe_loss_weight
        self.round_index = 0

    def _phase(self, client, data, local):
        batches = data.require('D')
        client.phase(local)
        parameters = [p for p in client.local_model.parameters() if p.requires_grad] if local else client.transform_parameters()
        optimizer = torch.optim.SGD(parameters, lr=self.lr)
        losses = []
        for step in range(self.local_steps if local else self.moe_steps):
            batch = batches[step % len(batches)]
            optimizer.zero_grad(set_to_none=True)
            if local:
                native, features = DetectorTask.head_features(client.local_model, batch['img'])
                local_loss = native_loss(client.local_model, batch, native)
            else:
                with torch.no_grad():
                    _, features = DetectorTask.head_features(client.local_model, batch['img'])
                local_loss = batch['img'].new_zeros(())
            mixed, _ = client.raw(features)
            mixture_loss = native_loss(client.local_model, batch, mixed)
            loss = self.local_weight * local_loss + self.moe_weight * mixture_loss if local else mixture_loss
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite dFLMoE objective')
            loss.backward()
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all() for parameter in parameters):
                raise ValueError('Nonfinite dFLMoE gradient')
            optimizer.step()
            losses.append(float(loss.detach()))
        optimizer.zero_grad(set_to_none=True)
        return losses

    def round(self, fit_data, select_data=None, available=None):
        if len(fit_data) != len(self.clients):
            raise ValueError('One D stream per dFLMoE client required')
        # All clients finish local training before any head snapshot is replaced.
        local = [self._phase(client, data, True) for client, data in zip(self.clients, fit_data)]
        for client in self.clients:
            client.exchange_heads(self.models, available)
        mixture = [self._phase(client, data, False) for client, data in zip(self.clients, fit_data)]
        self.round_index += 1
        return {'round': self.round_index, 'local_losses': local, 'mixture_losses': mixture,
                'active_experts': [list(client.active) for client in self.clients], 'selection_E': 0}

    def predict(self, client, images):
        return self.clients[client].predict(images)

    def state_dict(self):
        return {'method': self.method, 'clients': [snapshot(client) for client in self.clients],
                'active': [list(client.active) for client in self.clients], 'round_index': self.round_index}

    def load_state_dict(self, state):
        if state['method'] != self.method or len(state['clients']) != len(self.clients):
            raise ValueError('Wrong mixture algorithm checkpoint')
        for client, saved, active in zip(self.clients, state['clients'], state['active']):
            client.load_state_dict(saved)
            client.active = list(active)
        self.round_index = state['round_index']
