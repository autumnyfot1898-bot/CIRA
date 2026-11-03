"""DA-DPFL detector adaptation: sparse training, reuse, PQ trigger and regrowth.

Matches the scheduled author variant at a7f93b8b...; detector Conv/Linear weights
are selected by module type, excluding the fixed DFL projection. No topology or
pruning decision is delegated to CIRA.
"""
from __future__ import annotations
import copy
import math
import random

import torch
from torch import nn

from ..baseline_algorithms import dadpfl_wait_schedule, pq_prune_layer, sparse_average
from .common import evaluating, fit, independent_models, native_loss, restore_models, snapshot


def layer_densities(parameters, density, distribution='ERK', power=1.):
    if not 0 < density <= 1 or not parameters or distribution not in ('ERK', 'uniform'):
        raise ValueError('Nonempty prunable layers, density in (0,1], ERK/uniform required')
    if distribution == 'uniform' or density == 1:
        return {name: density for name in parameters}
    raw = {name: (sum(value.shape) / value.numel()) ** power for name, value in parameters.items()}
    total = sum(value.numel() for value in parameters.values())
    dense = set()
    while True:
        remaining = [name for name in parameters if name not in dense]
        if not remaining:
            return {name: 1. for name in parameters}
        rhs = total * density - sum(parameters[name].numel() for name in dense)
        epsilon = rhs / sum(raw[name] * parameters[name].numel() for name in remaining)
        over = [name for name in remaining if epsilon * raw[name] > 1]
        if not over:
            return {name: 1. if name in dense else max(0., epsilon * raw[name]) for name in parameters}
        dense.update(over)


def rewire_mask(weights, mask, gradients, drop_fraction, rng=None, rigl=True):
    if not 0 <= drop_fraction <= 1 or weights.shape != mask.shape:
        raise ValueError('Valid removal fraction and equal-shaped masks required')
    flat = mask.detach().flatten().bool().clone()
    active = flat.nonzero().flatten()
    remove = min(math.ceil(drop_fraction * len(active)), len(active))
    if remove == 0:
        return flat.reshape(mask.shape).to(mask.dtype), 0
    order = torch.argsort(weights.detach().abs().flatten()[active], stable=True)
    flat[active[order[:remove]]] = False
    inactive = (~flat).nonzero().flatten()
    if rigl:
        if gradients is None or gradients.shape != weights.shape or not torch.isfinite(gradients).all():
            raise ValueError('Finite dense gradients required for RigL regrowth')
        order = torch.argsort(gradients.detach().square().flatten()[inactive], descending=True, stable=True)
        selected = inactive[order[:remove]]
    else:
        if rng is None:
            raise ValueError('Explicit random generator required for random regrowth')
        selected = inactive[torch.tensor(rng.sample(range(len(inactive)), remove), device=inactive.device)]
    flat[selected] = True
    return flat.reshape(mask.shape).to(mask.dtype), remove


class DADPFL:
    method = 'DA-DPFL'
    def __init__(self, models, *, density, target_sparsity, total_rounds, max_wait,
                 learning_rate, local_steps, seed, distribution='ERK', p=.5, q=1.,
                 eta=.1, prune_scale=1., early_prune_threshold=.001,
                 reconfiguration_reduce=3., rewire_interval=10, anneal_factor=.3,
                 rigl=True, lr_decay=1., momentum=0., weight_decay=0., loss_fn=native_loss):
        independent_models(models)
        if not 0 <= target_sparsity < 1 or type(rewire_interval) is not int or rewire_interval < 1:
            raise ValueError('Valid target sparsity and rewire interval required')
        if reconfiguration_reduce <= 1 or not 0 <= anneal_factor <= 1:
            raise ValueError('Pruning reduction >1 and anneal factor in [0,1] required')
        self.models, self.loss_fn = models, loss_fn
        self.lr, self.steps, self.total = learning_rate, local_steps, total_rounds
        self.wait = dadpfl_wait_schedule(max_wait, total_rounds)
        self.rng = random.Random(seed)
        self.pq = dict(p=p, q=q, eta=eta, scale=prune_scale)
        self.target = target_sparsity
        self.threshold, self.reduce = early_prune_threshold, reconfiguration_reduce
        self.rewire_interval, self.anneal, self.rigl = rewire_interval, anneal_factor, rigl
        self.lr_decay, self.momentum, self.weight_decay = lr_decay, momentum, weight_decay
        prunable = {}
        for name, module in models[0].named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and '.dfl.' not in name + '.':
                if module.weight.requires_grad:
                    prunable[(name + '.' if name else '') + 'weight'] = module.weight
        densities = layer_densities(prunable, density, distribution)
        generator = torch.Generator().manual_seed(seed)
        self.masks = []
        # Common initial masks, a documented option of the author implementation.
        initial_mask = {}
        for name, value in prunable.items():
            mask = torch.zeros(value.numel())
            mask[torch.randperm(value.numel(), generator=generator)[:int(densities[name] * value.numel())]] = 1
            initial_mask[name] = mask.reshape(value.shape)
        for model in models:
            masks = {name: value.to(next(model.parameters()).device).clone() for name, value in initial_mask.items()}
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in masks:
                        parameter.mul_(masks[name])
            self.masks.append(masks)
        self.initial = [snapshot(model) for model in models]
        self.norm_history = [[] for _ in models]
        self.trigger_round, self.pruning_rounds, self.round_index = None, set(), 0

    def round(self, fit_data, select_data=None, available=None):
        r, n = self.round_index, len(self.models)
        if r >= self.total or len(fit_data) != n:
            raise ValueError('DA-DPFL round exceeds configured schedule or missing client D stream')
        original = [snapshot(model) for model in self.models]
        updated = copy.deepcopy(original)
        reports, thresholds = [], []
        pq_event = r in self.pruning_rounds
        for i, model in enumerate(self.models):
            peers = [j for j in range(n) if i != j and (available is None or available[i][j])]
            peers.sort()
            wait = min(self.wait[r], len(peers))
            states = [original[i]] + [updated[j] if rank < wait else original[j] for rank, j in enumerate(peers)]
            masks = [self.masks[i]] + [self.masks[j] for j in peers]
            # Aggregation counts active nonzero coordinates in the received states.
            masks_cpu = [{name: (state[name] != 0).float() for name in self.masks[i]} for state in states]
            merged = sparse_average(states, masks_cpu, {name: mask.cpu() for name, mask in self.masks[i].items()})
            model.load_state_dict(merged)
            result = fit(model, fit_data[i], self.steps, self.lr * self.lr_decay ** r,
                         loss_fn=self.loss_fn, masks=self.masks[i], momentum=self.momentum, weight_decay=self.weight_decay)
            state = snapshot(model)
            total = sum(mask.numel() for mask in self.masks[i].values())
            sparsity = 1 - sum(float(mask.sum()) for mask in self.masks[i].values()) / total
            if pq_event and sparsity < self.target:
                for name, parameter in model.named_parameters():
                    if name in self.masks[i]:
                        self.masks[i][name], _ = pq_prune_layer(parameter.detach(), self.masks[i][name], **self.pq)
            rewired = 0
            if (r + 1) % self.rewire_interval == 0:
                # G2 importance is the square of the gradient averaged over the
                # complete local D stream, not only the final fitting minibatch.
                parameters = [(name,p) for name,p in model.named_parameters() if name in self.masks[i]]
                gradients = {name:torch.zeros_like(p) for name,p in parameters}
                batches = fit_data[i].require('D')
                with evaluating(model):
                    for batch in batches:
                        objective = self.loss_fn(model,batch)
                        values = torch.autograd.grad(objective,[p for _,p in parameters],allow_unused=True)
                        for (name,_),value in zip(parameters,values):
                            if value is not None:gradients[name].add_(value.detach()/len(batches))
                drop = self.anneal / 2 * (1 + math.cos(r * math.pi / self.total))
                for name, parameter in model.named_parameters():
                    if name in self.masks[i]:
                        self.masks[i][name], count = rewire_mask(parameter, self.masks[i][name],
                            gradients.get(name), drop, self.rng, self.rigl)
                        rewired += count
            # The next mask is projected before exporting/predicting the deployed state.
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in self.masks[i]:
                        parameter.mul_(self.masks[i][name])
            updated[i] = snapshot(model)
            norm = sum(float(torch.linalg.vector_norm((state[name] - self.initial[i][name]).float()))
                       for name in state if state[name].is_floating_point())
            history = self.norm_history[i]
            history.append(norm)
            if len(history) >= 2:
                # An unchanged client has ratio zero; nonzero changes from a zero
                # initial displacement remain nonconverged rather than NaN.
                change = abs(history[-1] - history[-2])
                thresholds.append(change / history[0] if history[0] > 0 else (0. if change == 0 else math.inf))
            reports.append({'client': i, 'losses': result['losses'], 'same_round_reuse': peers[:wait],
                            'PQ_pruning': pq_event, 'rewired': rewired})
        if self.trigger_round is None and len(thresholds) == n and sum(x <= self.threshold for x in thresholds) / n >= .5:
            self.trigger_round = r
            intervals, interval = [], r
            for _ in range(21):
                intervals.append(interval)
                interval = round(interval / self.reduce)
            cumulative = 0
            for interval in intervals:
                cumulative += interval
                # Author loop uses the event on the next iteration. A trigger
                # in the past is not replayed as an immediate pruning event.
                self.pruning_rounds.add(cumulative + 1)
        restore_models(self.models, updated)
        self.round_index += 1
        return {'round': self.round_index, 'clients': reports, 'trigger_round': self.trigger_round,
                'selection_E': 0, 'mask_density': [sum(float(x.sum()) for x in m.values()) / sum(x.numel() for x in m.values()) for m in self.masks]}

    def state_dict(self):
        return {'method': self.method, 'models': [snapshot(m) for m in self.models],
                'masks': [{name: value.cpu().clone() for name, value in mask.items()} for mask in self.masks],
                'initial': self.initial, 'norm_history': copy.deepcopy(self.norm_history),
                'trigger_round': self.trigger_round, 'pruning_rounds': sorted(self.pruning_rounds),
                'round_index': self.round_index, 'rng': self.rng.getstate()}

    def load_state_dict(self, state):
        if state['method'] != self.method:
            raise ValueError('Wrong sparse algorithm checkpoint')
        restore_models(self.models, state['models'])
        self.masks = [{name: mask.to(next(model.parameters()).device) for name, mask in masks.items()}
                      for model, masks in zip(self.models, state['masks'])]
        self.initial, self.norm_history = state['initial'], copy.deepcopy(state['norm_history'])
        self.trigger_round, self.pruning_rounds = state['trigger_round'], set(state['pruning_rounds'])
        self.round_index = state['round_index']
        self.rng.setstate(state['rng'])
