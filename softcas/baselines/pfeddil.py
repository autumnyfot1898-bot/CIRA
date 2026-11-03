"""pFedDIL detector portfolio, shared-feature classifiers and knowledge migration.

ECCV 2024 Eq. 2--8. This comparator retains server aggregation. Binary task
membership targets must be supplied explicitly on D batches; detection class IDs
are never reinterpreted as domain-membership labels.
"""
from __future__ import annotations
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from ..baseline_algorithms import average_states
from .common import DetectorTask, evaluating, independent_models, snapshot, restore_models, trainable


class Portfolio:
    def __init__(self, prototype, task, threshold, learning_rate, local_steps, migration_weight=1.):
        self.prototype, self.task = copy.deepcopy(prototype), task
        self.threshold, self.lr, self.steps = threshold, learning_rate, local_steps
        self.migration_weight = migration_weight
        self.models, self.auxiliary = [], nn.ModuleList()
        self.active, self.references, self.intensities = None, [], []
        self.stage = 0

    def _head(self, model, example):
        with torch.no_grad():
            feature = self.task.features(model, example['img'])
        return nn.Linear(feature.shape[1], 1).to(feature.device)

    def matching(self, data):
        batches = data.require('D')
        result = []
        for model, auxiliary in zip(self.models, self.auxiliary):
            values = []
            with evaluating(model), torch.no_grad():
                for batch in batches:
                    values.append(auxiliary(self.task.features(model, batch['img'])).sigmoid().flatten())
            result.append(float(torch.cat(values).mean()))
        return result

    def begin_stage(self, data, global_model):
        batches = data.require('D')
        intensity = self.matching(data)
        self.references = [snapshot(model) for model in self.models]
        self.intensities = intensity
        if not intensity or max(intensity) < self.threshold:
            self.models.append(copy.deepcopy(global_model))
            self.auxiliary.append(self._head(self.models[-1], batches[0]))
            self.active = len(self.models) - 1
            decision = 'new'
        else:
            self.active = max(range(len(intensity)), key=lambda i: (intensity[i], -i))
            # References remain frozen; updating this slot does not mutate the
            # historical targets used by Eq. 5 in the current stage.
            self.models[self.active] = copy.deepcopy(self.models[self.active])
            decision = 'inherit'
        self.stage += 1
        return {'decision': decision, 'active_model': self.active, 'matching_intensities': intensity}

    def fit(self, data):
        batches = data.require('D')
        if self.active is None:
            raise ValueError('pFedDIL requires a stage matching decision first')
        for batch in batches:
            target = batch.get('auxiliary_target')
            if target is None or target.numel() != batch['img'].shape[0] or not torch.all((target == 0) | (target == 1)):
                raise ValueError('D-only binary auxiliary_target per image is required for pFedDIL')
        model, auxiliary = self.models[self.active], self.auxiliary[self.active]
        model.train()
        trainable(model)
        auxiliary.train().requires_grad_(True)
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad] + list(auxiliary.parameters()), lr=self.lr)
        named = dict(model.named_parameters())
        targets = [{name: value.to(named[name].device) for name, value in state.items() if name in named}
                   for state in self.references]
        losses = []
        for step in range(self.steps):
            batch = batches[step % len(batches)]
            optimizer.zero_grad(set_to_none=True)
            detection_loss, features = self.task.loss_features(model, batch)
            logits = auxiliary(features).flatten()
            auxiliary_loss = F.binary_cross_entropy_with_logits(logits, batch['auxiliary_target'].to(logits).flatten())
            migration = detection_loss.new_zeros(())
            for intensity, old in zip(self.intensities, targets):
                difference = torch.cat([(p - old[name]).flatten() for name, p in named.items() if p.requires_grad])
                migration = migration + intensity * torch.linalg.vector_norm(difference)
            loss = detection_loss + auxiliary_loss + self.migration_weight * migration
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite pFedDIL joint objective')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in list(model.parameters()) + list(auxiliary.parameters())):
                raise ValueError('Nonfinite pFedDIL gradient')
            optimizer.step()
            losses.append({'detection': float(detection_loss.detach()), 'auxiliary': float(auxiliary_loss.detach()),
                           'migration': float(migration.detach())})
        optimizer.zero_grad(set_to_none=True)
        return losses

    def predict(self, images):
        if not self.models:
            raise ValueError('Empty pFedDIL portfolio')
        dense, probabilities = [], []
        with torch.no_grad():
            for model, auxiliary in zip(self.models, self.auxiliary):
                with evaluating(model):
                    dense.append(self.task.dense(model, images))
                    probabilities.append(auxiliary(self.task.features(model, images)).sigmoid().flatten())
            probability = torch.stack(probabilities, dim=1)
            normalizer = probability.sum(1, keepdim=True)
            weights = torch.where(normalizer > 0, probability / normalizer.clamp_min(torch.finfo(probability.dtype).tiny),
                                  torch.full_like(probability, 1 / len(self.models)))
            if any(value.shape != dense[0].shape for value in dense):
                raise ValueError('pFedDIL dense detection grids must align across portfolio models')
            mixed = sum(value * weights[:, i, None, None] for i, value in enumerate(dense))
            return self.task.predictions_from_dense(self.models[0], mixed)

    def state_dict(self):
        return {'models': [snapshot(model) for model in self.models], 'auxiliary': [snapshot(head) for head in self.auxiliary],
                'active': self.active, 'references': self.references, 'intensities': self.intensities, 'stage': self.stage}

    def load_state_dict(self, state):
        self.models = [copy.deepcopy(self.prototype) for _ in state['models']]
        restore_models(self.models, state['models'])
        heads = []
        for saved in state['auxiliary']:
            output, size = saved['weight'].shape
            head = nn.Linear(size, output).to(next(self.prototype.parameters()).device)
            head.load_state_dict(saved)
            heads.append(head)
        self.auxiliary = nn.ModuleList(heads)
        self.active, self.references = state['active'], state['references']
        self.intensities, self.stage = state['intensities'], state['stage']


class PFedDIL:
    method = 'pFedDIL'
    def __init__(self, models, *, threshold, learning_rate, local_steps, task=None,
                 migration_weight=1., sample_weights=None):
        independent_models(models)
        if not 0 <= threshold <= 1 or learning_rate <= 0 or type(local_steps) is not int or local_steps < 1:
            raise ValueError('Valid matching threshold, learning rate and local step count required')
        self.models = models
        self.task = task or DetectorTask()
        self.global_model = copy.deepcopy(models[0])
        self.portfolios = [Portfolio(model, self.task, threshold, learning_rate, local_steps, migration_weight) for model in models]
        self.weights = [1.] * len(models) if sample_weights is None else list(sample_weights)
        self.round_index, self.stage = 0, 0

    def begin_stage(self, fit_data):
        if len(fit_data) != len(self.models):
            raise ValueError('One fitting stream per pFedDIL client required')
        decisions = [portfolio.begin_stage(data, self.global_model) for portfolio, data in zip(self.portfolios, fit_data)]
        self.models[:] = [portfolio.models[portfolio.active] for portfolio in self.portfolios]
        self.stage += 1
        return {'stage': self.stage, 'matching': decisions, 'topology': 'server_aggregation_reference'}

    def round(self, fit_data, select_data=None, available=None):
        if not self.stage:
            raise ValueError('pFedDIL stage has not started')
        results = [portfolio.fit(data) for portfolio, data in zip(self.portfolios, fit_data)]
        states = [snapshot(model) for model in self.models]
        global_state = average_states(states, self.weights)
        self.global_model.load_state_dict(global_state)
        # Server returns the common active-model update; earlier portfolio slots
        # remain private and are not overwritten by aggregation.
        restore_models(self.models, [global_state] * len(self.models))
        self.round_index += 1
        return {'round': self.round_index, 'local_losses': results, 'selection_E': 0,
                'topology': 'server_aggregation_reference', 'portfolio_sizes': [len(p.models) for p in self.portfolios]}

    def predict(self, client, images):
        return self.portfolios[client].predict(images)

    def state_dict(self):
        return {'method': self.method, 'global': snapshot(self.global_model),
                'portfolios': [p.state_dict() for p in self.portfolios], 'round_index': self.round_index, 'stage': self.stage}

    def load_state_dict(self, state):
        if state['method'] != self.method or len(state['portfolios']) != len(self.portfolios):
            raise ValueError('pFedDIL checkpoint differs')
        self.global_model.load_state_dict(state['global'])
        for portfolio, saved in zip(self.portfolios, state['portfolios']):
            portfolio.load_state_dict(saved)
        self.models[:] = [p.models[p.active] for p in self.portfolios]
        self.round_index, self.stage = state['round_index'], state['stage']
