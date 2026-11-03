"""DCFCL detector adaptation using paper Eq. 4--12, MBA and DCE.

Dense per-anchor class logits replace classification logits for consistency
distillation. Coalition utility is calculated from the two distinct vector
cosines in Eq. 11--12, not by applying one norm to an already combined affinity.
"""
from __future__ import annotations
import copy
from itertools import combinations
import math

import torch
from torch.nn import functional as F

from ..baseline_algorithms import average_states
from .common import DetectorTask, evaluating, fit, independent_models, native_loss, parameter_vector, restore_models, snapshot


def coalitions(clients):
    clients = tuple(sorted(clients))
    return [tuple(group) for size in range(1, len(clients) + 1) for group in combinations(clients, size)]


def canonical_partition(partition):
    return tuple(sorted(tuple(sorted(group)) for group in partition if group))


def benefit_table(gradients, parameters, sample_weights, similarity_weight):
    n = len(gradients)
    if not n or n != len(parameters) or len(sample_weights) != n or similarity_weight < 0:
        raise ValueError('Matched client vectors and nonnegative model-similarity coefficient required')
    if any(not math.isfinite(w) or w <= 0 for w in sample_weights):
        raise ValueError('Positive finite sample weights required')
    def cosine(a, b):
        return float(F.cosine_similarity(a.reshape(1, -1).double(), b.reshape(1, -1).double(), dim=1))
    table = {}
    for group in coalitions(range(n)):
        utility = {}
        for owner in group:
            peers = [j for j in group if j != owner]
            if not peers:
                utility[owner] = 0.
                continue
            total = sum(sample_weights[j] for j in peers)
            g = sum((gradients[j].double() * sample_weights[j] / total for j in peers), torch.zeros_like(gradients[0], dtype=torch.float64))
            p = sum((parameters[j].double() * sample_weights[j] / total for j in peers), torch.zeros_like(parameters[0], dtype=torch.float64))
            utility[owner] = cosine(g, gradients[owner]) + similarity_weight * cosine(p, parameters[owner])
        if any(not math.isfinite(value) for value in utility.values()):
            raise ValueError('Nonfinite coalition benefit')
        table[group] = utility
    return table


def current_utilities(partition, table):
    return {client: table[tuple(group)][client] for group in partition for client in group}


def blocks(group, partition, table, tolerance=1e-12):
    old = current_utilities(partition, table)
    gains = [table[tuple(group)][i] - old[i] for i in group]
    return all(gain >= -tolerance for gain in gains) and any(gain > tolerance for gain in gains)


def form_coalition(partition, group):
    members = set(group)
    return canonical_partition([group] + [tuple(i for i in old if i not in members) for old in partition])


def merge_blocking(initial, table):
    """MBA coalition traversal and stable-coalition count pruning (Algorithm 1)."""
    partition = canonical_partition(initial)
    clients = {i for group in partition for i in group}
    if sum(map(len, partition)) != len(clients):
        raise ValueError('Coalition partition contains duplicate clients')
    if set(table) != set(coalitions(clients)):
        raise ValueError('Complete coalition benefit table required')
    frozen, trace = [], []
    while partition:
        counts = {}
        active = {i for group in partition for i in group}
        candidates = coalitions(active)
        for group in candidates:
            if blocks(group, partition, table):
                partition = form_coalition(partition, group)
                trace.append({'blocking': list(group), 'partition': partition})
                counts = {key: count for key, count in counts.items() if key in partition}
                for item in partition:
                    counts[item] = counts.get(item, 0) + 1
        if not counts:
            frozen.extend(partition)
            partition = ()
        else:
            stable = min(counts, key=lambda key: (-counts[key], key))
            frozen.append(stable)
            partition = tuple(group for group in partition if group != stable)
    result = canonical_partition(frozen)
    unresolved = [group for group in coalitions(clients) if blocks(group, result, table)]
    # Preserve the paper's MBA output rather than replacing it with another
    # optimization algorithm. An independent certificate exposes any remaining
    # blocking coalition; completion does not imply a mathematical guarantee.
    trace.append({'equilibrium_verified':not unresolved,'remaining_blocking_coalitions':unresolved})
    return result, trace


def dynamic_evolution(previous, table):
    """DCE starts from the prior coalition partition, retaining its warm start."""
    return merge_blocking(previous, table)


class DCFCL:
    method = 'DCFCL'
    def __init__(self, models, *, learning_rate, local_steps, distillation_weight,
                 temperature, model_similarity_weight, sample_weights=None,
                 loss_fn=native_loss, task=None):
        independent_models(models)
        if temperature <= 0 or distillation_weight < 0:
            raise ValueError('Positive temperature and nonnegative distillation coefficient required')
        self.models, self.loss_fn, self.task = models, loss_fn, task or DetectorTask()
        self.lr, self.steps = learning_rate, local_steps
        self.kd, self.temperature, self.epsilon = distillation_weight, temperature, model_similarity_weight
        self.weights = [1.] * len(models) if sample_weights is None else list(sample_weights)
        self.partition = canonical_partition([(i,) for i in range(len(models))])
        self.round_index = 0

    def _class_logits(self, model, images):
        output = model(images)
        raw = output[1] if not model.training else output
        raw = raw['one2many'] if 'one2many' in raw else raw
        return raw['scores']

    def round(self, fit_data, select_data=None, available=None):
        before = [parameter_vector(model) for model in self.models]
        reports = []
        for model, data in zip(self.models, fit_data):
            teacher = copy.deepcopy(model).eval().requires_grad_(False)
            def objective(student, batch):
                # Reuse the student's single native forward for supervised and
                # distillation terms, including BN statistics and the head grid.
                if self.loss_fn is native_loss:
                    output=student(batch['img'])
                    supervised=native_loss(student,batch,output)
                    raw=output['one2many'] if 'one2many' in output else output
                    logits=raw['scores']/self.temperature
                else:
                    supervised=self.loss_fn(student,batch)
                    logits=self._class_logits(student,batch['img'])/self.temperature
                with torch.no_grad():
                    # Use the same temperature on teacher/student (Eq. 4).
                    target = (self._class_logits(teacher, batch['img']) / self.temperature).softmax(dim=1)
                return supervised+self.kd * -(target * logits.log_softmax(dim=1)).sum(1).mean()
            result = fit(model, data, self.steps, self.lr, loss_fn=objective if self.kd else self.loss_fn)
            reports.append(result['losses'])
        parameters = [parameter_vector(model) for model in self.models]
        # The author client stores post-fit minus pre-fit model differences as
        # the gradient-coherence representation; sign cancels pairwise cosines.
        gradients = [after - old for after, old in zip(parameters, before)]
        table = benefit_table(gradients, parameters, self.weights, self.epsilon)
        if available is not None:
            # Unavailable pairs cannot be placed into the same exchange group.
            for group, utility in table.items():
                if any(i != j and not (available[i][j] and available[j][i]) for i in group for j in group):
                    for i in group:
                        utility[i] = -1e100
        partition, trace = dynamic_evolution(self.partition, table)
        states = [snapshot(model) for model in self.models]
        updates = [None] * len(states)
        for group in partition:
            for owner in group:
                updates[owner] = average_states([states[i] for i in group], [self.weights[i] for i in group], group.index(owner))
        restore_models(self.models, updates)
        self.partition = partition
        self.round_index += 1
        return {'round': self.round_index, 'partition': partition, 'transitions': trace,
                'blocking_equilibrium_verified':trace[-1]['equilibrium_verified'],
                'local_losses': reports, 'selection_E': 0}

    def state_dict(self):
        return {'method': self.method, 'models': [snapshot(m) for m in self.models],
                'partition': self.partition, 'round_index': self.round_index}

    def load_state_dict(self, state):
        if state['method'] != self.method:
            raise ValueError('Wrong coalition algorithm checkpoint')
        restore_models(self.models, state['models'])
        self.partition, self.round_index = canonical_partition(state['partition']), state['round_index']
