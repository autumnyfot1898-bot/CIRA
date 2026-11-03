"""Regression checks for admitted work and native epoch ordering (CPU only)."""
import copy
from unittest.mock import patch
from test_invariants import Fixture
from softcas.fitting import stage_steps, shuffled_batches, epoch_seed
from softcas.data import DetectionBatches
from softcas.engine import run_stage
from softcas.resources import Cost
from softcas.config import validate_config


class FittingProtocolTests(Fixture):
    def test_executor_obeys_positive_admitted_fidelity(self):
        self.c["phi_steps"] = 6
        a = self.proposal("refine", fidelity=1)
        with patch.object(self.backend, "fit_batch", wraps=self.backend.fit_batch) as fit:
            result = self.executor.run(a, self.state, self.data)
        self.assertEqual(result.status, "complete", result.error)
        self.assertEqual(fit.call_count, 1)
        self.assertEqual((result.fit_steps, result.completed_fit_steps), (1, 1))

    def native(self):
        self.c.update(proposal_protocol="native_r1000", fitting={
            "unit": "epochs", "stage_epochs": 2, "local_epochs": 3, "receiving_epochs": 2},
            _fit_steps_remaining=2, _attempt_index=0)

    def test_native_admitted_zero_is_not_legacy_default(self):
        self.native()
        with patch.object(self.backend, "fit_batch", wraps=self.backend.fit_batch) as fit:
            result = self.executor.run(self.proposal("refine", fidelity=0), self.state, self.data)
        self.assertEqual(result.status, "complete", result.error)
        self.assertEqual(fit.call_count, 0)

    def test_unadmitted_over_budget_block_never_fits(self):
        self.native()
        with patch.object(self.backend, "fit_batch", wraps=self.backend.fit_batch) as fit:
            result = self.executor.run(self.proposal("refine", fidelity=3), self.state, self.data)
        self.assertEqual(result.status, "failed")
        self.assertIn("remainder", result.error)
        self.assertEqual(fit.call_count, 0)

    def test_partial_failed_block_counts_attempted_work(self):
        self.native()
        original = self.backend.fit_batch
        calls = []
        def fail_second(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("interrupted batch")
            return original(*args, **kwargs)
        with patch.object(self.backend, "fit_batch", side_effect=fail_second):
            result = self.executor.run(self.proposal("refine", fidelity=2), self.state, self.data)
        self.assertEqual(result.status, "failed")
        self.assertIsNone(result.score)
        self.assertEqual((result.fit_steps, result.completed_fit_steps), (2, 1))
        self.assertGreater(result.cost.F, 0)
        self.assertEqual(len(result.epoch_seeds), 2)

    def test_native_sample_order_precedes_batching(self):
        data = DetectionBatches(tuple(range(19)), None, 640, 8)
        first = shuffled_batches(data, 31)
        self.assertEqual(first.records, shuffled_batches(data, 31).records)
        self.assertNotEqual(first.records, shuffled_batches(data, 32).records)
        self.assertEqual(sorted(first.records), list(range(19)))
        self.assertEqual(len(first), 3)  # final partial batch retained

    def test_epoch_seed_replays_without_global_rng(self):
        self.native()
        a = self.proposal("refine", fidelity=2)
        seed = epoch_seed(self.c, a, self.data.D, 0)
        self.assertEqual(seed, epoch_seed(copy.deepcopy(self.c), a, self.data.D, 0))
        self.assertNotEqual(seed, epoch_seed(self.c, a, self.data.D, 1))
        self.assertEqual(stage_steps(self.c, 7), 14)

    def test_controller_work_rechecks_admission(self):
        a = self.proposal("refine")
        from dataclasses import replace
        a = replace(a, estimate=Cost(B=self.ledger.remaining().B + 1))
        with patch("softcas.engine.Selector.choose", return_value=((a, (0.0,)*25, 0., 0., 1.), "fixture")):
            result = run_stage(self.backend, self.state, self.data, self.cache, self.exchange,
                               self.ledger, self.c, proposal_provider=lambda *_: [a])
        self.assertEqual(result.stop, "controller_budget")
        self.assertEqual(result.actions, [])

    def test_controller_does_not_recharge_nested_refresh_time(self):
        import itertools
        clock = itertools.count()
        self.exchange.refresh = lambda: self.ledger.charge("refresh_fixture", Cost(W=0.25))
        with patch("softcas.engine.time.perf_counter", side_effect=lambda: next(clock)*0.5), \
             patch("softcas.engine.Selector.choose", return_value=(None, "keep")):
            run_stage(self.backend, self.state, self.data, self.cache, self.exchange,
                      self.ledger, self.c, proposal_provider=lambda *_: [])
        event = next(e for e in self.ledger.events if e['reason']=='controller')
        self.assertEqual(event['cost']['W'], 0.25)


class ConfigProtocolTests(Fixture):
    def test_formal_purpose_cannot_bypass_gate_via_false_flag(self):
        self.c["purpose"] = "FORMAL"
        with self.assertRaisesRegex(ValueError, "Formal execution blocked"):
            validate_config(self.c, formal=False)

    def test_legacy_batch_steps_are_not_formal_epoch_training(self):
        from softcas.config import PENDING
        self.c.update(purpose='FORMAL',confirmed={k:True for k in PENDING})
        with self.assertRaisesRegex(ValueError,'epoch semantics'):
            validate_config(self.c)

    def test_invalid_native_or_nonfinite_settings_rejected(self):
        for name, value in [("ridge_lambda", float('nan')), ("priority_epsilon", float('inf')),
                            ("kappa", -1), ("archive_capacity", 1), ("attempts", True)]:
            config = copy.deepcopy(self.c);config[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_config(config)
        config = copy.deepcopy(self.c)
        config['fitting'] = {'unit':'epochs','stage_epochs':500,'local_epochs':32,'receiving_epochs':16}
        with self.assertRaisesRegex(ValueError,'native_r1000'):
            validate_config(config)
