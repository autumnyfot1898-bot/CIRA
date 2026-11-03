"""DPFL: streamed BGGC preprocessing, fixed graph and per-round GGC.

Primary definition: AISTATS 2025, Algorithms 1--3. The fetch callback is the
transport boundary: BGGC makes two explicit passes and retains only an aggregate
and a batch of at most Bc peer states per receiver.
"""
from __future__ import annotations

import copy
import math
import random

import torch

from ..baseline_algorithms import average_states, dpfl_select
from .common import fit, independent_models, native_loss, restore_models, snapshot, validation_loss


def batched_graph(owner, peers, owner_state, weights, cardinality, fetch, oracle, rng):
    peers = list(peers)
    if owner in peers or len(set(peers)) != len(peers) or not 1 <= cardinality <= len(peers) + 1:
        raise ValueError('BGGC needs unique peers excluding owner and a valid total cardinality')
    if any(not math.isfinite(weights[i]) or weights[i] <= 0 for i in [owner] + peers):
        raise ValueError('Positive finite BGGC sample weights required')
    names = list(owner_state)
    floating = [name for name in names if owner_state[name].is_floating_point()]
    x = {name: owner_state[name].clone() * weights[owner] for name in floating}
    y = {name: value.clone() for name, value in x.items()}
    total_x, total_y = weights[owner], weights[owner]
    fetched, calls = [], []

    def checked_fetch(peer):
        state = fetch(peer)
        if state.keys() != owner_state.keys() or any(state[name].shape != owner_state[name].shape or
                state[name].dtype != owner_state[name].dtype or not torch.isfinite(state[name]).all() for name in names):
            raise ValueError('Incompatible or nonfinite BGGC peer model')
        fetched.append(peer)
        return state

    def score(sums, total, members):
        state = {name: owner_state[name].clone() for name in names}
        for name in floating:
            state[name] = sums[name] / total
        value = float(oracle(state))
        if not math.isfinite(value):
            raise ValueError('Nonfinite BGGC validation reward')
        calls.append({'subset': list(members), 'reward': value})
        return value

    for peer in peers:
        state = checked_fetch(peer)
        for name in floating:
            y[name].add_(state[name].to(y[name].device), alpha=weights[peer])
        total_y += weights[peer]
    order = peers.copy()
    rng.shuffle(order)
    lower, upper = [owner], [owner] + order
    if cardinality == 1:
        return lower, {'fetches': fetched, 'oracles': calls, 'peak_peer_batch': 0}
    # X and Y are running weighted sums; no all-peer model bank is retained here.
    for start in range(0, len(order), cardinality):
        batch = [(peer, checked_fetch(peer)) for peer in order[start:start + cardinality]]
        for peer, state in batch:
            plus = {name: x[name] + weights[peer] * state[name].to(x[name].device) for name in floating}
            minus = {name: y[name] - weights[peer] * state[name].to(y[name].device) for name in floating}
            low = score(x, total_x, lower)
            add = score(plus, total_x + weights[peer], lower + [peer])
            high = score(y, total_y, upper)
            removed = [i for i in upper if i != peer]
            remove = score(minus, total_y - weights[peer], removed)
            a, b = max(add - low, 0), max(remove - high, 0)
            probability = a / (a + b) if a + b > 0 else 1.
            if rng.random() < probability:
                x, total_x = plus, total_x + weights[peer]
                lower.append(peer)
            else:
                y, total_y, upper = minus, total_y - weights[peer], removed
            if len(lower) == cardinality:
                return lower, {'fetches': fetched, 'oracles': calls, 'peak_peer_batch': min(cardinality, len(peers))}
    return lower, {'fetches': fetched, 'oracles': calls, 'peak_peer_batch': min(cardinality, len(peers))}


class DPFL:
    method = 'DPFL'
    def __init__(self, models, *, loss_fn=native_loss, learning_rate, cardinality,
                 pretrain_steps, local_steps, seed, sample_weights=None):
        independent_models(models)
        self.models, self.loss_fn = models, loss_fn
        self.lr, self.cardinality = learning_rate, cardinality
        self.pretrain_steps, self.local_steps = pretrain_steps, local_steps
        self.rng = random.Random(seed)
        self.weights = [1.] * len(models) if sample_weights is None else list(sample_weights)
        if len(self.weights) != len(models) or any(not math.isfinite(w) or w <= 0 for w in self.weights):
            raise ValueError('Positive client sample weights required')
        self.graph, self.round_index = None, 0

    def prepare(self, fit_data, select_data, available=None):
        if self.graph is not None:
            raise ValueError('DPFL fixed graph is already prepared; restore it from a checkpoint')
        n = len(self.models)
        if len(fit_data) != n or len(select_data) != n:
            raise ValueError('One D/V stream per DPFL client required')
        logs = [fit(m, data, self.pretrain_steps, self.lr, loss_fn=self.loss_fn)
                for m, data in zip(self.models, fit_data)]
        states = [snapshot(m) for m in self.models]
        graph, diagnostics, updates = [], [], []
        for i, model in enumerate(self.models):
            replica = copy.deepcopy(model)
            peers = [j for j in range(n) if j != i and (available is None or available[i][j])]
            def oracle(state):
                replica.load_state_dict(state)
                return -validation_loss(replica, select_data[i], self.loss_fn)
            chosen, diagnostic = batched_graph(i, peers, states[i], self.weights,
                min(self.cardinality, len(peers) + 1), lambda j: states[j], oracle, self.rng)
            graph.append(chosen)
            diagnostics.append(diagnostic)
            updates.append(average_states([states[j] for j in chosen], [self.weights[j] for j in chosen], chosen.index(i)))
        restore_models(self.models, updates)
        self.graph = graph
        return {'pretraining_steps': sum(x['fitting_steps'] for x in logs), 'graph': copy.deepcopy(graph),
                'BGGC': diagnostics, 'selection_E': sum(len(d['oracles']) for d in diagnostics)}

    def round(self, fit_data, select_data, available=None):
        if self.graph is None:
            raise ValueError('DPFL requires BGGC preparation first')
        local = [fit(m, data, self.local_steps, self.lr, loss_fn=self.loss_fn)
                 for m, data in zip(self.models, fit_data)]
        states = [snapshot(m) for m in self.models]
        updates, choices, calls = [], [], []
        for i, model in enumerate(self.models):
            replica = copy.deepcopy(model)
            candidates = [j for j in self.graph[i] if j == i or available is None or available[i][j]]
            def oracle(group):
                replica.load_state_dict(average_states([states[j] for j in group], [self.weights[j] for j in group], group.index(i)))
                return -validation_loss(replica, select_data[i], self.loss_fn)
            selected = dpfl_select(i, candidates, min(self.cardinality, len(candidates)), oracle, rng=self.rng)
            group = selected.clients
            updates.append(average_states([states[j] for j in group], [self.weights[j] for j in group], group.index(i)))
            choices.append(list(group))
            calls.append(list(selected.oracle_calls))
        restore_models(self.models, updates)
        self.round_index += 1
        return {'round': self.round_index, 'selected_clients': choices, 'oracles': calls,
                'local_losses': [item['losses'] for item in local], 'selection_E': sum(map(len, calls))}

    def state_dict(self):
        return {'method': self.method, 'models': [snapshot(m) for m in self.models],
                'graph': copy.deepcopy(self.graph), 'round_index': self.round_index, 'rng': self.rng.getstate()}

    def load_state_dict(self, state):
        if state['method'] != self.method:
            raise ValueError('Wrong algorithm checkpoint')
        restore_models(self.models, state['models'])
        self.graph, self.round_index = copy.deepcopy(state['graph']), state['round_index']
        self.rng.setstate(state['rng'])
