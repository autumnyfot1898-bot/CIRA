"""Independent tensor algorithms used by external detector adaptations.

These functions do not construct CIRA states, adapters, predictors or archives.
References and the precise extent of each implementation are in
docs/DETECTOR_BASELINES.md. Sparse/coalition helpers are not full DA-DPFL/DCFCL.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Callable

import torch


def _finite(value, name):
    if not math.isfinite(float(value)):
        raise ValueError(name + ' must be finite')
    return float(value)


def _compatible(states):
    if not states:
        raise ValueError('At least one model is required')
    first = states[0]
    for state in states:
        if state.keys() != first.keys():
            raise ValueError('Model parameter names differ')
        for key in first:
            if state[key].shape != first[key].shape or state[key].dtype != first[key].dtype:
                raise ValueError('Model parameter shape/dtype differs: ' + key)
            if not torch.isfinite(state[key]).all():
                raise ValueError('Nonfinite model tensor: ' + key)


def average_states(states, weights=None, buffer_reference=0):
    """Average floating tensors; retain local integer counters explicitly.

This counter policy avoids averaging BN counters into invalid integer values.
It is a detector-port choice, recorded separately from the original algorithm.
"""
    _compatible(states)
    if not 0 <= buffer_reference < len(states):
        raise ValueError('Invalid local buffer reference')
    weights = [1.0] * len(states) if weights is None else list(weights)
    if len(weights) != len(states) or any(_finite(w, 'Weight') < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError('Nonnegative weights with a positive sum are required')
    total = sum(weights)
    result = {}
    for name, first in states[0].items():
        if not (first.is_floating_point() or first.is_complex()):
            result[name] = states[buffer_reference][name].detach().clone()
        else:
            result[name] = torch.zeros_like(first)
            for weight, state in zip(weights, states):
                result[name].add_(state[name].to(first.device), alpha=weight / total)
    return result


def _graph(adjacency, active_edges, count):
    if len(adjacency) != count or len(active_edges) != count:
        raise ValueError('Graph size differs from the model count')
    for matrix in (adjacency, active_edges):
        if any(len(row) != count for row in matrix):
            raise ValueError('Square graph matrices required')
        for i in range(count):
            for j in range(count):
                if type(matrix[i][j]) is not bool or matrix[i][j] != matrix[j][i] or (i == j and matrix[i][j]):
                    raise ValueError('An undirected boolean graph without self-loops is required')
    if any(active_edges[i][j] and not adjacency[i][j] for i in range(count) for j in range(count)):
        raise ValueError('Active communication edge is absent from the declared graph')


def dspodfl_step(models, batches, loss_fn, adjacency, active_edges, active_gradients,
                  learning_rate, combine_then_adapt=False):
    """One synchronous DSpodFL step with explicit realized Bernoulli indicators.

All neighbor differences use the same pre-step snapshots. Metropolis weights
use degrees in the fixed graph, not the currently active subgraph. In ATC,
gradients are evaluated at the pre-mixing point. In CTA they use the mixed point.
The upstream algorithm updates parameters only; gradient-copy BN buffers are
therefore discarded and receiver BN buffers remain local. No optimizer momentum.
"""
    count = len(models)
    if not count or len(batches) != count or len(active_gradients) != count:
        raise ValueError('One model, batch and gradient indicator per client required')
    if any(type(flag) is not bool for flag in active_gradients):
        raise ValueError('Gradient indicators must be boolean')
    learning_rate = _finite(learning_rate, 'Learning rate')
    if learning_rate <= 0 or type(combine_then_adapt) is not bool:
        raise ValueError('Positive learning rate and explicit CTA/ATC flag required')
    _graph(adjacency, active_edges, count)
    snapshots = [{k: v.detach().clone() for k, v in m.named_parameters()} for m in models]
    _compatible(snapshots)
    degrees = [sum(row) for row in adjacency]
    results, losses = [], []
    for i, model in enumerate(models):
        mixed = {k: v.clone() for k, v in snapshots[i].items()}
        for j in range(count):
            if active_edges[i][j]:
                weight = 1.0 / (1 + max(degrees[i], degrees[j]))
                for name in mixed:
                    mixed[name].add_(snapshots[j][name].to(mixed[name].device) - snapshots[i][name], alpha=weight)
        if active_gradients[i]:
            replica = copy.deepcopy(model)
            if combine_then_adapt:
                with torch.no_grad():
                    for name, parameter in replica.named_parameters():
                        parameter.copy_(mixed[name])
            trainable = [(name, parameter) for name, parameter in replica.named_parameters() if parameter.requires_grad]
            if not trainable:
                raise ValueError('No trainable detector parameters')
            loss = loss_fn(replica, batches[i])
            if loss.numel() != 1 or not torch.isfinite(loss):
                raise ValueError('Scalar finite native training loss required')
            gradients = torch.autograd.grad(loss, [p for _, p in trainable], allow_unused=True)
            for (name, _), gradient in zip(trainable, gradients):
                if gradient is not None:
                    if not torch.isfinite(gradient).all():
                        raise ValueError('Nonfinite detector gradient')
                    mixed[name].add_(gradient, alpha=-learning_rate)
            losses.append(float(loss.detach()))
            del replica, gradients
        else:
            losses.append(None)
        results.append(mixed)
    _compatible(results)  # Reject nonfinite updates before changing any client.
    with torch.no_grad():
        for model, state in zip(models, results):
            for name, parameter in model.named_parameters():
                parameter.copy_(state[name])
    directed_transfers = sum(sum(row) for row in active_edges)
    tensor_bytes = sum(t.numel() * t.element_size() for t in snapshots[0].values())
    return {'local_losses': losses, 'active_gradients': list(active_gradients),
            'active_undirected_edges': [[i, j] for i in range(count) for j in range(i + 1, count) if active_edges[i][j]],
            'parameter_tensor_bytes_sent': directed_transfers * tensor_bytes,
            'communication_scope': 'logical tensor payload; no socket/header/device measurement',
            'update_order': 'CTA' if combine_then_adapt else 'ATC'}


@dataclass(frozen=True)
class Selection:
    clients: tuple[int, ...]
    oracle_calls: tuple[tuple[tuple[int, ...], float], ...]


def dpfl_select(owner, available, cardinality, oracle: Callable, *, variant='rgl', rng=None):
    """DPFL subset oracle with owner retention and an explicit size bound.

`rgl` uses randomized double-greedy add/remove marginals from optimized_rgl_p.
`non_monotone_greedy` preserves that separately named upstream variant. Reward
is supplied by the receiver's V detection loss, never by a test AP readout.
Positive equal marginals use 1/2 (the normalized mathematical rule); the upstream
Python leaves p undefined in that branch. Two zero marginals use upstream p=1.
"""
    available = list(available)
    if len(set(available)) != len(available) or owner not in available:
        raise ValueError('Unique available client IDs including the owner required')
    if type(cardinality) is not int or not 1 <= cardinality <= len(available):
        raise ValueError('Cardinality includes the owner and must fit the available set')
    if variant not in ('rgl', 'non_monotone_greedy') or (variant == 'rgl' and rng is None):
        raise ValueError('Declare the DPFL selection variant and seeded RNG')
    calls = []

    def evaluate(group):
        group = tuple(group)
        value = _finite(oracle(group), 'Subset reward')
        calls.append((group, value))
        return value

    lower = [owner]
    if cardinality == 1:
        return Selection(tuple(lower), tuple(calls))
    others = [i for i in available if i != owner]
    if variant == 'non_monotone_greedy':
        while len(lower) < cardinality and others:
            base = evaluate(lower)
            gains = [(evaluate(lower + [i]) - base, i) for i in others]
            gain, candidate = max(gains, key=lambda value: (value[0], -value[1]))
            if gain < 0:
                break
            lower.append(candidate)
            others.remove(candidate)
    else:
        rng.shuffle(others)
        upper = [owner] + others
        low_reward, high_reward = evaluate(lower), evaluate(upper)
        for candidate in others:
            plus = evaluate(lower + [candidate])
            removed = [i for i in upper if i != candidate]
            minus = evaluate(removed)
            add_gain, remove_gain = max(plus - low_reward, 0), max(minus - high_reward, 0)
            probability = add_gain / (add_gain + remove_gain) if add_gain + remove_gain > 0 else 1.0
            if rng.random() < probability:
                lower.append(candidate)
                low_reward = plus
            else:
                upper = removed
                high_reward = minus
            if len(lower) == cardinality:
                break
    return Selection(tuple(lower), tuple(calls))


def sparse_average(states, masks, local_mask, local_reference=0):
    """DA-DPFL masked coordinate average, followed by receiver mask projection.

The count of nonzero contributors is the denominator per coordinate. This is a
kernel only: initialization, pruning triggers, regrowth and scheduling remain
the responsibility of the complete method implementation.
"""
    _compatible(states)
    if len(masks) != len(states) or not 0 <= local_reference < len(states):
        raise ValueError('One mask per contributor and a valid receiver are required')
    result = average_states(states, buffer_reference=local_reference)
    for name, projection in local_mask.items():
        if name not in result or projection.shape != result[name].shape:
            raise ValueError('Receiver mask shape/name differs')
        numerator, denominator = torch.zeros_like(result[name]), torch.zeros_like(result[name])
        for state, mask in zip(states, masks):
            value = mask[name].to(result[name].device)
            if value.shape != result[name].shape or not torch.all((value == 0) | (value == 1)):
                raise ValueError('Binary coordinate masks required')
            numerator.add_(state[name].to(numerator.device) * value)
            denominator.add_(value)
        if not torch.all((projection == 0) | (projection == 1)):
            raise ValueError('Binary receiver mask required')
        result[name] = torch.where(denominator > 0, numerator / denominator.clamp_min(1), 0) * projection.to(numerator.device)
    return result


def dadpfl_wait_schedule(largest_wait, total_rounds):
    """Author stepwise max-wait schedule; one value for each communication round."""
    if type(largest_wait) is not int or type(total_rounds) is not int or not 0 <= largest_wait < total_rounds:
        raise ValueError('Nonnegative max-wait smaller than the round count required')
    per_step, remainder = divmod(total_rounds, largest_wait + 1)
    return [n for n in range(largest_wait, -1, -1) for _ in range(per_step)] + [0] * remainder


def pq_prune_layer(weights, mask, *, p, q, eta, scale):
    """DA-DPFL layer-scope PQ-index sparsification with magnitude ties retained."""
    if not 0 < p < q or eta < 0 or scale < 0 or not all(math.isfinite(v) for v in (p, q, eta, scale)):
        raise ValueError('Require finite 0<p<q, nonnegative eta and scale')
    if weights.shape != mask.shape or not torch.all((mask == 0) | (mask == 1)) or not torch.isfinite(weights).all():
        raise ValueError('Finite weights and a shape-matched binary mask required')
    active = weights[mask.bool()].abs()
    count = active.numel()
    if not count or torch.all(active == 0):
        return torch.zeros_like(mask), {'active_before': count, 'active_after': 0, 'pq_index': 0.0}
    ratio = (active.pow(p).mean().pow(1 / p) / active.pow(q).mean().pow(1 / q)).clamp(0, 1)
    index = 1 - ratio
    retained_bound = torch.ceil(count * (1 + eta) ** (q / (p - q)) * ratio.pow(q * p / (q - p)))
    prune_ratio = (scale * (1 - retained_bound / count)).clamp(0, 0.9)
    position = min(int(torch.floor(count * prune_ratio)), count - 1)
    threshold = active.sort().values[position]
    result = mask.bool() & (weights.abs() >= threshold)
    return result.to(mask.dtype), {'active_before': count, 'active_after': int(result.sum()), 'pq_index': float(index)}


def dcfcl_affinity(gradients, parameters, model_similarity_weight):
    """DCFCL gradient cosine + sw * parameter cosine; no coalition/trainer claim."""
    weight = _finite(model_similarity_weight, 'Model-similarity weight')
    if weight < 0 or len(gradients) != len(parameters) or not gradients:
        raise ValueError('Matched nonempty client vectors and nonnegative sw required')
    for vectors in (gradients, parameters):
        if any(v.ndim != 1 or v.shape != vectors[0].shape or not torch.isfinite(v).all() for v in vectors):
            raise ValueError('Equal-size finite vectors required within each representation')
    def cosine(values):
        matrix = torch.stack(values).float()
        matrix = torch.nn.functional.normalize(matrix, dim=1)
        return matrix @ matrix.T
    return cosine(gradients) + weight * cosine(parameters)
