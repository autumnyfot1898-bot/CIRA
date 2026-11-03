import copy
import json
import random
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from softcas.actions import Proposal, construct, vary, TYPES
from softcas.backend import Backend
from softcas.cache import PackageCache, ContactExchange
from softcas.cli import ROOT
from softcas.config import load_config
from softcas.controller import Archive, Scored, Ridge, FeatureEncoder, Selector, priority
from softcas.data import StageData, synthetic_split, validate_registry
from softcas.engine import Executor, initial_state, run_stage
from softcas.evaluation import EvaluationCache, detection_ap
from softcas.model import Adapter, PortComposition
from softcas.proposals import ProposalFactory
from softcas.reporting import retention, paired_effects, macro_endpoints
from softcas.partition import shared_targets, assign_groups, balance_objective
from softcas.tables import export_tables
from softcas.resources import Cost, Ledger, ExecutionStopped, TransferJournal
from softcas.state import State, Structure, FitSettings, Package, freeze_tensors, canonical

torch.set_num_threads(1)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.c = load_config(ROOT / "configs" / "fixture.json")
        self.backend = Backend()
        self.cache = PackageCache(1000000, self.backend.specs)
        self.state = initial_state(self.backend, self.cache, Structure((1, 0, 0, 0, 0, 0), 1 / 8), FitSettings(1e-3, 1e-3), 42)
        self.data = StageData(*(synthetic_split(r, "0", 1, "a", 10 + i) for i, r in enumerate(("D", "V", "A"))))
        self.ledger = Ledger(Cost(1e12, 100, 1e7, 1000), Cost(1e12, 0, 1e7, 100))
        self.source = Ledger(Cost(1e12, 100, 1e7, 1000), Cost())
        self.exchange = ContactExchange("0", self.cache, self.ledger)
        self.evaluator = EvaluationCache()
        self.executor = Executor(self.backend, self.exchange, self.evaluator, self.c, self.ledger)

    def package(self, port=0, seed=9, owner="peer"):
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            adapter = Adapter(8, 1 / 8)
            torch.nn.init.normal_(adapter.up.weight, 0, 0.2)
        return Package(owner, str(seed), 1, 2, self.backend.specs[port], 1 / 8,
                       freeze_tensors(adapter.state_dict()), "fixture independent donor")

    def put(self, package):
        self.cache.put(package.hash, package.wire)
        return package.hash

    def proposal(self, kind, parent=None, **kwargs):
        return Proposal((parent or self.state).hash, kind, 19, Cost(1e6, 1, 0, 1), **kwargs)


class StateAndModelTests(Fixture):
    def test_checkpoint_roundtrip_checksum_and_no_shared_storage(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "state.json")
            self.state.save(path)
            restored = State.load(path)
            self.assertEqual(restored, self.state)
            tensor = restored.phi[0][1].tensor()
            tensor.add_(100)
            self.assertEqual(restored.hash, self.state.hash)
            raw = json.loads(path.read_text())
            raw["state"]["nu"] = "{}"
            path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                State.load(path)

    def test_zero_up_gate_half_and_six_ports(self):
        adapter = Adapter(8, 1 / 16)
        self.assertEqual(float(adapter.gate.sigmoid()), 0.5)
        self.assertEqual(adapter.down.out_channels, 1)
        self.assertTrue(torch.equal(adapter(torch.randn(2, 8, 3, 3)), torch.zeros(2, 8, 3, 3)))
        self.assertEqual(len(self.backend.specs), 6)

    def test_parallel_same_input_and_null_normalization(self):
        p = self.package()
        local = Adapter(8, 1 / 8)
        torch.nn.init.ones_(local.up.weight)
        port = PortComposition(p.interface, local, [p], {p.hash: 0})
        x = torch.randn(2, 8, 4, 4)
        expected = x + local(x) + 0.5 * port.external[p.hash](x)
        self.assertTrue(torch.allclose(port(x), expected))
        self.assertTrue(torch.allclose(port.weights()[1], torch.tensor([0.5, 0.5])))
        port.beta[p.hash].data.fill_(-50)
        port(x)
        self.assertEqual(port.path_calls, 2)

    def test_input_gradient_through_frozen_external_and_base(self):
        p = self.package(port=2)
        key = self.put(p)
        state = replace(self.state, C=(key,), beta=((key, 0),))
        model = self.backend.assembly(state, self.cache, mode="refine")
        frozen = {k: v.clone() for k, v in model.base.state_dict().items()}
        self.backend.loss(model, self.data.D.batches[0]).backward()
        self.assertGreater(float(model.ports["0"].local.up.weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in model.base.parameters()))
        self.assertTrue(all(p.grad is None for port in model.ports.values() for p in port.external.parameters()))
        self.assertTrue(all(torch.equal(v, frozen[k]) for k, v in model.base.state_dict().items()))

    def test_penalty_normalizes_per_image_then_averages(self):
        p = self.package()
        local = Adapter(8, 1 / 8)
        local.load_state_dict({k: v.tensor() for k, v in p.tensors})
        port = PortComposition(p.interface, local, [], {})
        x = torch.randn(2, 8, 3, 3)
        x[1] *= 20
        residual = local(x)
        port(x)
        expected = (residual.square().sum((1, 2, 3)) / (x.square().sum((1, 2, 3)) + 1e-8)).mean()
        self.assertTrue(torch.allclose(port.last_penalty, expected))

    def test_fresh_reinitializes_all_phi_and_keeps_composition(self):
        key = self.put(self.package())
        parent = replace(self.state, C=(key,), beta=((key, 0.7),))
        model = self.backend.assembly(parent, self.cache, mode="refine")
        model.ports["0"].local.up.weight.data.fill_(3)
        parent = model.snapshot(parent)
        before = parent.hash
        a = self.proposal("fresh", parent, z=parent.z, h=parent.h)
        child = construct(parent, a, self.cache)
        fresh = self.backend.assembly(child, self.cache, a.seed, "fresh")
        self.assertEqual(float(fresh.ports["0"].local.up.weight.abs().sum()), 0)
        self.assertEqual(float(fresh.ports["0"].local.gate), 0)
        self.assertAlmostEqual(float(fresh.ports["0"].beta[key]), 0.7, places=6)
        self.assertEqual(before, parent.hash)

    def test_refine_copies_and_optimizer_resets(self):
        a = self.proposal("refine")
        child = construct(self.state, a, self.cache)
        model = self.backend.assembly(child, self.cache, mode="refine")
        self.assertEqual(model.snapshot(child).phi, self.state.phi)
        opt1, opt2 = model.optimizer(1e-3, 1e-2), model.optimizer(1e-3, 1e-2)
        self.assertIsNot(opt1, opt2)
        self.assertEqual(len(opt2.state), 0)

    def test_add_remove_replace_inheritance_and_mask(self):
        k1, k2 = self.put(self.package(seed=1)), self.put(self.package(seed=2))
        parent = replace(self.state, C=(k1,), beta=((k1, 1.2),))
        child = construct(parent, self.proposal("add", parent, add=k2), self.cache)
        self.assertEqual(dict(child.beta)[k1], 1.2)
        self.assertEqual(dict(child.beta)[k2], 0)
        model = self.backend.assembly(child, self.cache, mode="collaboration")
        self.assertTrue(all(not p.requires_grad for p in model.ports["0"].local.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.ports["0"].beta.parameters()))
        replacement = construct(parent, self.proposal("replace", parent, add=k2, remove=k1), self.cache)
        self.assertEqual(replacement.C, (k2,))
        removed = construct(parent, self.proposal("remove", parent, remove=k1), self.cache)
        self.assertFalse(removed.beta)
        self.assertIsNone(self.backend.assembly(removed, self.cache, mode="collaboration").optimizer(1e-3, 1e-3))
        wrong = self.put(self.package(port=1, seed=3))
        with self.assertRaises(ValueError):
            construct(parent, self.proposal("replace", parent, add=wrong, remove=k1), self.cache)

    def test_external_two_path_cap(self):
        keys = sorted(self.put(self.package(seed=i)) for i in range(3))
        parent = replace(self.state, C=tuple(keys[:2]), beta=tuple((k, 0) for k in keys[:2]))
        with self.assertRaises(ValueError):
            construct(parent, self.proposal("add", parent, add=keys[2]), self.cache)

    def test_import_is_exact_no_fitting(self):
        saved = self.state.lineage(unique="exact complete history")
        result = self.executor.run(self.proposal("import", imported=saved), self.state, self.data)
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.state, saved)
        self.assertFalse(any(e["reason"] == "fit" for e in self.ledger.events))

    def test_complete_checkpoint_rejects_wrong_base_and_ports(self):
        with self.assertRaises(ValueError):
            Backend(seed=123).assembly(self.state, self.cache)
        wrong = self.state.lineage(ports_signature="different preprocessing")
        with self.assertRaises(ValueError):
            self.backend.assembly(wrong, self.cache)

    def test_seeded_variation_replay_and_search_grid(self):
        for seed in range(40):
            z, h = vary(self.state, self.state, seed)
            self.assertIn(sum(z.mask), (1, 2))
            self.assertEqual((z, h), vary(self.state, self.state, seed))

    def test_discrete_receiving_is_one_hot_no_beta_fit(self):
        k = self.put(self.package())
        state = replace(self.state, C=(k,), beta=((k, 9),)).lineage(receiving_rule="discrete", discrete_choices={"0": k})
        model = self.backend.assembly(state, self.cache, mode="collaboration")
        port = model.ports["0"]
        x = torch.randn(1, 8, 4, 4)
        self.assertTrue(torch.allclose(port(x), x + port.local(x) + port.external[k](x)))
        self.assertFalse(port.beta[k].requires_grad)
        self.assertIsNone(model.optimizer(1e-3, 1e-3))
        null = state.lineage(discrete_choices={"0": None})
        self.assertNotEqual(state.model_hash(self.backend.identity), null.model_hash(self.backend.identity))
        null_port = self.backend.assembly(null, self.cache).ports["0"]
        null_port(x)
        self.assertEqual(null_port.path_calls, 0)

    def test_frozen_local_penalty_retains_beta_gradient(self):
        key = self.put(self.package(port=0))
        state = initial_state(self.backend, self.cache, Structure((0, 0, 1, 0, 0, 0), 1 / 8), self.state.h, 4)
        state = replace(state, C=(key,), beta=((key, 0),))
        model = self.backend.assembly(state, self.cache, mode="collaboration")
        model.ports["2"].local.up.weight.data.fill_(0.3)
        model(self.data.D.batches[0]["img"])
        model.penalty().backward()
        self.assertGreater(float(model.ports["0"].beta[key].grad.abs()), 0)
        self.assertTrue(all(p.grad is None for p in model.ports["2"].local.parameters()))


class CacheAndBudgetTests(Fixture):
    def test_partial_restart_cost_and_offline_cached_use(self):
        p = self.package()
        self.exchange.observe("peer", {p.hash: p.wire}, 3, 10, self.source)
        before = self.ledger.used["search"].B
        with self.assertRaises(ExecutionStopped):
            self.exchange.acquire(p.hash, 3)
        self.assertNotIn(p.hash, self.cache)
        request_bytes = sum(r["bytes"] for r in self.exchange.records if r["kind"] == "request")
        self.assertEqual(self.ledger.used["search"].B - before, request_bytes + 10)
        self.exchange.observe("peer", {p.hash: p.wire}, 4, 100000, self.source)
        self.exchange.acquire(p.hash, 4)
        self.assertIn(p.hash, self.cache)
        self.assertEqual(sum(r["bytes"] for r in self.exchange.records if r["kind"] == "payload"), 10 + len(p.wire))
        self.exchange.disconnect("peer")
        self.assertEqual(self.exchange.acquire(p.hash, 5), p)
        self.assertEqual(self.exchange.journal.network_bytes, self.ledger.used["search"].B)

    def test_future_unadvertised_incompatible_and_checksum_rejected(self):
        p = self.package()
        with self.assertRaises(ExecutionStopped):
            self.exchange.acquire(p.hash, 4)
        future = replace(p, available=20)
        self.exchange.observe("peer", {future.hash: future.wire}, 3, 100000, self.source)
        self.assertNotIn(future.hash, self.exchange.catalogue)
        with self.assertRaises(ValueError):
            self.cache.put("wrong", p.wire)
        wrong = replace(p, interface=replace(p.interface, base="other detector"))
        with self.assertRaises(ValueError):
            self.cache.put(wrong.hash, wrong.wire)
        bad = replace(p, tensors=p.tensors[:-1])
        with self.assertRaises(ValueError):
            self.cache.put(bad.hash, bad.wire)

    def test_pin_protects_incumbent_and_atomic_admission(self):
        p, q = self.package(seed=1), self.package(seed=2)
        cache = PackageCache(max(len(p.wire), len(q.wire)) + 5, self.backend.specs)
        cache.put(p.hash, p.wire)
        cache.pin("incumbent", [p.hash])
        with self.assertRaises(ExecutionStopped):
            cache.put(q.hash, q.wire)
        self.assertIn(p.hash, cache)
        cache.unpin("incumbent")
        cache.put(q.hash, q.wire)
        self.assertNotIn(p.hash, cache)

    def test_full_acquisition_survives_later_fit_failure(self):
        p = self.package()
        self.exchange.observe("peer", {p.hash: p.wire}, 3, 100000, self.source)
        a = replace(self.proposal("add", add=p.hash), acquisition=True)
        before = self.state.hash
        with patch.object(self.backend, "loss", side_effect=RuntimeError("injected fitting failure")):
            result = self.executor.run(a, self.state, self.data)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.state, self.state)
        self.assertIn(p.hash, self.cache)
        self.assertGreater(result.cost.B, 0)
        self.assertGreater(result.cost.W, 0)
        self.assertEqual(self.state.hash, before)

    def test_returned_delta_never_double_charged(self):
        before = self.ledger.used["search"]
        result = self.executor.run(self.proposal("refine"), self.state, self.data)
        self.assertEqual(result.status, "complete")
        self.assertEqual(self.ledger.used["search"] - before, result.cost)
        total = Cost()
        for e in self.ledger.events:
            if e["scope"] == "search":
                total += Cost(**e["cost"])
        self.assertEqual(total, self.ledger.used["search"])

    def test_reserved_resources_actual_overrun_and_limits(self):
        ledger = Ledger(Cost(F=1, E=1), Cost(F=100, E=0, W=100))
        self.assertEqual(ledger.remaining().F, 1)
        ledger.charge("actual", Cost(F=2, E=1))
        self.assertIn("F", ledger.violations()["search"])
        with self.assertRaises(ExecutionStopped):
            ledger.guard(Cost(E=1))
        self.assertEqual(ledger.remaining("release").F, 100)
        with self.assertRaises(ValueError):
            ledger.charge("refund", Cost(F=-1))

    def test_network_unique_directed_ids_and_sender_order(self):
        journal = TransferJournal()
        journal.add("id", "a", "b", 10, "payload", 1)
        journal.add("id", "a", "b", 10, "payload", 1)
        self.assertEqual(journal.network_bytes, 10)
        requests = [{"time": 1, "requester": "b"}, {"time": 0, "requester": "z"}, {"time": 1, "requester": "a"}]
        self.assertEqual([r["requester"] for r in ContactExchange.order_requests(requests)], ["z", "a", "b"])


class ControllerAndStageTests(Fixture):
    def test_ridge_exact_statistics_solve_and_failure_target(self):
        ridge = Ridge(2, 2)
        f = (1.0, 2.0)
        with patch("numpy.linalg.inv", side_effect=AssertionError("inverse forbidden")):
            ridge.update(f, 0.3)
            np.testing.assert_allclose(ridge.L, 2 * np.eye(2) + np.outer(f, f))
            np.testing.assert_allclose(ridge.b, np.array(f) * 0.3)
            mu, s = ridge.predict(f)
            self.assertAlmostEqual(mu, 0.3 * 5 / 7)
            self.assertAlmostEqual(s, np.sqrt(5 / 7))
            ridge.update(f, 0)
            np.testing.assert_allclose(ridge.b, np.array(f) * 0.3)

    def test_priority_parent_relative_champion_and_zero_allowance(self):
        value = priority(0.1, 0.3, 0.25, 0, Cost(F=1), Cost(F=2), 0, 0.1)
        self.assertAlmostEqual(value, 0.05 / 0.6)
        self.assertEqual(priority(0, 0, 1, 0, Cost(B=1), Cost(B=0), 0, 0.1), 0)
        self.assertEqual(priority(0, 0, 1, 0, Cost(), Cost(), 0, 0.1), 10)

    def test_archive_protection_and_seeded_tournament(self):
        archive = Archive(self.state, 0.5, 2, self.cache)
        best = self.state.lineage(id=1)
        archive.insert(Scored(best, 0.8, 0.3, "configuration"))
        archive.insert(Scored(self.state.lineage(id=2), 0.6, 0.1, "collaboration"))
        self.assertEqual(set(archive.items), {self.state.hash, best.hash})
        r1, r2 = random.Random(2), random.Random(2)
        self.assertEqual([archive.parent(r1).state.hash for _ in range(10)], [archive.parent(r2).state.hash for _ in range(10)])

    def test_reimport_incumbent_preserves_both_family_parent_eligibility(self):
        archive = Archive(self.state, 0.5, 2, self.cache)
        archive.insert(Scored(self.state, 0.5, 0, "configuration"))
        self.assertEqual(archive.parent(random.Random(1), "collaboration").state, self.state)

    def test_failed_attempt_slot_target_and_preexecution_features(self):
        self.c["attempts"] = 2
        captured = []
        def provider(archive, attempt):
            p = self.proposal("refine")
            captured.append(p)
            return [p]
        self.evaluator.evaluate(self.backend, self.state, self.cache, self.data.V, self.ledger)
        with patch.object(self.backend, "loss", side_effect=RuntimeError("injected")):
            result = run_stage(self.backend, self.state, self.data, self.cache, self.exchange, self.ledger,
                               self.c, proposal_provider=provider, evaluator=self.evaluator)
        self.assertEqual(len(result.actions), 2)
        self.assertEqual(result.predictor.n, 2)
        self.assertTrue(all(r["gain"] == 0 and r["status"] == "failed" and r["child"] == self.state.hash for r in result.actions))
        self.assertTrue(all(r["predictor_n_before"] + 1 == r["predictor_n_after"] for r in result.actions))
        self.assertEqual(result.actions[0]["features"][self.c["features"]["fields"].index("parent_utility")], 0)

    def test_audit_failure_locks_one_and_keeps_incumbent(self):
        self.c["attempts"] = 2
        counters = {"V": 0, "A": 0}
        def score(model, split):
            counters[split.role] += 1
            return (0.4 + 0.1 * counters["V"]) if split.role == "V" else (0.1 if counters["A"] == 1 else 0.9)
        with patch.object(self.backend, "score", side_effect=score):
            result = run_stage(self.backend, self.state, self.data, self.cache, self.exchange, self.ledger,
                               self.c, proposal_provider=lambda archive, n: [self.proposal("refine", archive.champion.state)])
        self.assertNotEqual(result.champion.hash, self.state.hash)
        self.assertEqual(result.deployed, self.state)
        self.assertEqual(counters["A"], 2)
        self.assertFalse(result.published)
        self.assertEqual(result.predictor.n, 2)
        self.assertEqual(self.ledger.used["search"].E, 3)
        self.assertAlmostEqual(result.actions[1]["gain"], 0.1)
        self.assertAlmostEqual(result.actions[1]["parent_utility"], 0.1)

    def test_collaboration_release_does_not_republish_unchanged_local_or_peer(self):
        key = self.put(self.package())
        self.c["attempts"] = 1
        calls = {"V": 0, "A": 0}
        def score(model, split):
            calls[split.role] += 1
            return (0.4 if calls["V"] == 1 else 0.8) if split.role == "V" else (0.8 if calls["A"] == 1 else 0.4)
        with patch.object(self.backend, "score", side_effect=score):
            result = run_stage(self.backend, self.state, self.data, self.cache, self.exchange, self.ledger,
                               self.c, proposal_provider=lambda archive, n: [self.proposal("add", add=key)])
        self.assertTrue(result.audit["released"])
        self.assertEqual(result.deployed.C, (key,))
        self.assertEqual(result.deployed.phi, self.state.phi)
        self.assertFalse(result.published)

    def test_no_gain_is_distinct_from_round_robin(self):
        archive = Archive(self.state, 0.5, 2, self.cache)
        ridge = Ridge(len(self.c["features"]["fields"]), 1)
        ridge.b.fill(-1)
        encoder = FeatureEncoder(**self.c["features"])
        a = self.proposal("refine")
        no_gain = Selector(TYPES, TYPES, 0, 0.001, no_gain=True)
        rr = Selector(TYPES, TYPES, 0, 0.001, policy="round_robin")
        for selector in (no_gain, rr):
            selector.observed = set(TYPES)
            selector.local_fresh_offered = True
        decision, reason = no_gain.choose([a], archive, ridge, encoder, self.ledger.remaining(), self.cache, {})
        self.assertIsNone(decision)
        decision, reason = rr.choose([a], archive, ridge, encoder, self.ledger.remaining(), self.cache, {})
        self.assertEqual(reason, "round_robin")
        self.assertIsNotNone(decision)

    def test_cold_local_fresh_first_and_unavailable_types_skipped(self):
        archive = Archive(self.state, 0.5, 2, self.cache)
        encoder = FeatureEncoder(**self.c["features"])
        ridge = Ridge(len(encoder.fields), 1)
        selector = Selector(TYPES, TYPES, 0.1, 0.001)
        a = self.proposal("fresh", z=self.state.z, h=self.state.h)
        decision, reason = selector.choose([a, self.proposal("refine")], archive, ridge, encoder, self.ledger.remaining(), self.cache, {})
        self.assertEqual(reason, "cold_local_fresh")
        self.assertEqual(decision[0].kind, "fresh")

    def test_no_handoff_and_bounded_proposals(self):
        archive = Archive(self.state, 0.5, 6, self.cache)
        config_state = self.state.lineage(id="config")
        collab_state = self.state.lineage(id="collab")
        archive.insert(Scored(config_state, 0.6, 0.1, "configuration"))
        archive.insert(Scored(collab_state, 0.9, 0.4, "collaboration"))
        self.c["ablation"] = "no_handoff"
        factory = ProposalFactory(self.c, self.backend, self.exchange)
        for _ in range(20):
            self.assertNotEqual(factory._parent(archive, "configuration").state.hash, collab_state.hash)
        proposals = factory.generate(archive, self.ledger)
        self.assertTrue(all(sum(a.type == t for a in proposals) <= self.c["proposals_per_type"] for t in TYPES))

    def test_formal_config_rejects_test_choices(self):
        with self.assertRaises(ValueError):
            load_config(ROOT / "configs" / "fixture.json", formal=True)


class EvaluationAndReportingTests(Fixture):
    def test_exact_reuse_identity_h_history_and_data(self):
        score, reused = self.evaluator.evaluate(self.backend, self.state, self.cache, self.data.V, self.ledger)
        history = self.state.lineage(history="different")
        _, reused = self.evaluator.evaluate(self.backend, history, self.cache, self.data.V, self.ledger)
        self.assertTrue(reused)
        self.assertEqual(self.ledger.used["search"].E, 1)
        data2 = synthetic_split("V", "0", 1, "a", 88)
        self.evaluator.evaluate(self.backend, history, self.cache, data2, self.ledger)
        self.assertEqual(self.ledger.used["search"].E, 2)
        self.backend.execution += "changed"
        _, reused = self.evaluator.evaluate(self.backend, history, self.cache, self.data.V, self.ledger)
        self.assertFalse(reused)
        with self.assertRaises(ValueError):
            self.evaluator.evaluate(self.backend, history, self.cache, self.data.A, self.ledger)

    def test_model_base_and_tensor_changes_invalidate_reuse(self):
        initial = self.evaluator.key(self.backend, self.state, self.data.V)
        model = self.backend.assembly(self.state, self.cache, mode="refine")
        model.ports["0"].local.up.weight.data.fill_(1)
        changed = model.snapshot(self.state)
        self.assertNotEqual(initial, self.evaluator.key(self.backend, changed, self.data.V))
        self.data.V.batches[0]["img"].add_(1)
        self.assertNotEqual(initial, self.evaluator.key(self.backend, self.state, self.data.V))
        key = self.evaluator.key(self.backend, self.state, self.data.V)
        next(self.backend.base.parameters()).data.add_(1)
        self.assertNotEqual(key, self.evaluator.key(self.backend, self.state, self.data.V))

    def test_incomplete_evaluation_charges_work_without_completed_E(self):
        with patch.object(self.backend, "score", side_effect=RuntimeError("interrupted")):
            with self.assertRaises(RuntimeError):
                self.evaluator.evaluate(self.backend, self.state, self.cache, self.data.V, self.ledger)
        self.assertEqual(self.ledger.used["search"].E, 0)
        self.assertGreater(self.ledger.used["search"].W, 0)
        self.evaluator.evaluate(self.backend, self.state, self.cache, self.data.V, self.ledger)
        self.evaluator.evaluate(self.backend, self.state, self.cache, self.data.V, self.ledger, reuse=False)
        self.assertEqual(self.ledger.used["search"].E, 1)

    def test_scene_role_and_online_history_gates(self):
        base = dict(image="a", image_sha256="x", scene_id="s", role="D", domain="fog", client="0", stage=1)
        self.assertEqual(validate_registry([base])["scenes"], 1)
        with self.assertRaises(ValueError):
            validate_registry([base], formal=True)
        with self.assertRaises(ValueError):
            validate_registry([base, {**base, "image": "b", "role": "T", "client": "1"}])
        with self.assertRaises(ValueError):
            validate_registry([base, {**base, "image": "b", "stage": 2}])
        with self.assertRaises(ValueError):
            StageData(self.data.D, replace(self.data.V, stage=2), self.data.A)

    def test_ap50_95_perfect_wrong_and_unsupported_class(self):
        target = torch.tensor([[0, 0, 0, 10, 10]], dtype=torch.float32)
        perfect = torch.tensor([[0, 0, 10, 10, 0.9, 0]], dtype=torch.float32)
        self.assertGreater(detection_ap([perfect], [target]), 0.99)
        wrong = perfect.clone()
        wrong[:, 5] = 1
        self.assertEqual(detection_ap([wrong], [target]), 0)
        with self.assertRaises(ValueError):
            detection_ap([perfect], [target[:0]])

    def test_deployed_retention_values_and_model_identity(self):
        rows = [dict(seed=1001, client="0", stage=t, current_domain=c, domain=d, score=s, deployed=h)
                for t, c, d, s, h in [(1, "a", "a", 0.7, "m1"), (2, "b", "a", 0.6, "m2"),
                                      (2, "b", "b", 0.8, "m2"), (3, "c", "a", 0.5, "m3"),
                                      (3, "c", "b", 0.9, "m3"), (3, "c", "c", 0.4, "m3")]]
        metrics = retention(rows)
        self.assertAlmostEqual(metrics[-1]["seen"], 0.6)
        self.assertAlmostEqual(metrics[-1]["forget"], 0.1)
        self.assertAlmostEqual(metrics[-1]["bwt"], -0.05)
        self.assertIsNone(metrics[0]["forget"])
        bad = copy.deepcopy(rows)
        bad[1]["deployed"] = "picked_old_best"
        with self.assertRaises(ValueError):
            retention(bad)
        with self.assertRaises(ValueError):
            macro_endpoints(rows, expected_clients={"0", "1"})

    def test_paired_missing_not_zero_sample_sd_holm(self):
        full = {("a", 1001): 0.8, ("a", 1002): 0.7}
        serial = {("a", 1001): 0.7, ("a", 1002): 0.5}
        result = paired_effects(full, serial, ["a", "b"])
        self.assertEqual(result[0]["n_pair"], 2)
        self.assertEqual(result[0]["paired_differences"][2:], [None, None, None])
        self.assertAlmostEqual(result[0]["sample_sd"], np.std([0.1, 0.2], ddof=1))
        self.assertGreaterEqual(result[0]["p_holm"], result[0]["p"])
        self.assertIsNone(result[1]["ci95_unadjusted"])

    def test_whole_group_partition_shared_targets_and_replay(self):
        targets = shared_targets(1001, 2, 2)
        np.testing.assert_allclose(targets.sum(1), 1)
        groups = [{"id": str(i), "images": i + 1, "class_boxes": [i + 1, 5 - i]} for i in range(4)]
        a = assign_groups(groups, targets, 1001, 20)
        b = assign_groups(groups, targets, 1001, 20)
        self.assertEqual(a, b)
        self.assertEqual(sum(a["client_images"]), 10)
        self.assertEqual(set(a["assignment"]), {g["id"] for g in groups})
        self.assertLessEqual(a["objective"], a["initial_objective"])
        np.testing.assert_allclose(shared_targets(1, 2, 2, equal=True), 0.5)

    def test_table_interface_keeps_missing_blank_and_rejects_fixture_AP(self):
        with tempfile.TemporaryDirectory() as folder:
            export_tables([], folder)
            self.assertIn("softcas,ControlledFog/YOLOv8n,0,,", Path(folder, "T2_primary.csv").read_text(encoding="utf-8-sig"))
            with self.assertRaises(ValueError):
                export_tables([{"purpose": "SYNTHETIC_BEHAVIOR_ONLY", "metric": "AP50:95"}], folder)


if __name__ == "__main__":
    unittest.main()
