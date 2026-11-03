"""DSpodFL synchronous event process, updates and restartable state."""
from __future__ import annotations
import copy
import math
import random

from ..baseline_algorithms import _graph, dspodfl_step
from .common import independent_models, native_loss, restore_models, snapshot


class DSpodFL:
    method = 'DSpodFL'
    def __init__(self, models, *, adjacency, gradient_probabilities, edge_probabilities,
                 learning_rate, seed, combine_then_adapt=False, loss_fn=native_loss):
        independent_models(models)
        self.models, self.graph = models, copy.deepcopy(adjacency)
        _graph(self.graph, self.graph, len(models))
        self.pg, self.pe = list(gradient_probabilities), copy.deepcopy(edge_probabilities)
        n = len(models)
        if len(self.pg) != n or len(self.pe) != n or any(len(row) != n for row in self.pe):
            raise ValueError('One gradient probability per client and square edge probabilities required')
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in self.pg + [p for row in self.pe for p in row]):
            raise ValueError('Finite probabilities in [0,1] required')
        if any(self.pe[i][j] != self.pe[j][i] or (not adjacency[i][j] and self.pe[i][j] != 0)
               for i in range(n) for j in range(n)):
            raise ValueError('Symmetric probabilities confined to the fixed graph required')
        self.lr, self.cta, self.loss_fn = learning_rate, combine_then_adapt, loss_fn
        self.rng, self.round_index = random.Random(seed), 0

    def round(self, fit_data, select_data=None, available=None):
        n = len(self.models)
        if len(fit_data) != n:
            raise ValueError('One D stream per client required')
        batches = [self.rng.choice(data.require('D')) for data in fit_data]
        gradients = [self.rng.random() < p for p in self.pg]
        edges = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                active = self.graph[i][j] and self.rng.random() < self.pe[i][j]
                if available is not None:
                    active = active and available[i][j] and available[j][i]
                edges[i][j] = edges[j][i] = bool(active)
        report = dspodfl_step(self.models, batches, self.loss_fn, self.graph, edges,
                             gradients, self.lr, self.cta)
        self.round_index += 1
        report.update(round=self.round_index, selection_E=0)
        return report

    def state_dict(self):
        return {'method': self.method, 'models': [snapshot(m) for m in self.models],
                'round_index': self.round_index, 'rng': self.rng.getstate(), 'graph': self.graph,
                'gradient_probabilities': self.pg, 'edge_probabilities': self.pe, 'cta': self.cta}

    def load_state_dict(self, state):
        if (state['method'] != self.method or state['graph'] != self.graph or state['cta'] != self.cta or
                state['gradient_probabilities'] != self.pg or state['edge_probabilities'] != self.pe):
            raise ValueError('DSpodFL checkpoint event process differs')
        restore_models(self.models, state['models'])
        self.round_index = state['round_index']
        self.rng.setstate(state['rng'])
