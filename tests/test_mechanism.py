import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from softcas.backend import Backend
from softcas.cache import PackageCache
from softcas.config import load_config
from softcas.data import synthetic_split
from softcas.engine import initial_state
from softcas.mechanism import (continuation_states, continuation_comparison,
                               receiving_comparison, match_completed_work, run_mechanism_job)
from softcas.model import Adapter
from softcas.state import Structure, FitSettings, Package, freeze_tensors, State

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[1]


class ObservedBackend(Backend):
    def __init__(self):
        super().__init__()
        self.roles, self.fit_modes = [], []
        self.output = None
        self.fail_fit = False
        self.discrete_V_preferred = None

    def fit_batch(self, model, *args, **kwargs):
        self.fit_modes.append({"local_trainable": any(p.requires_grad for port in model.ports.values()
                                                       if port.local is not None for p in port.local.parameters()),
                               "beta_trainable": any(p.requires_grad for port in model.ports.values() for p in port.beta.values())})
        if self.fail_fit:
            # A real forward is executed before a controlled failure.
            self.loss(model, args[0])
            raise RuntimeError("controlled fit failure")
        return super().fit_batch(model, *args, **kwargs)

    def score(self, model, split):
        if split.role == "T":
            lock = self.output / "V_choices.json"
            assert lock.is_file(), "T accessed before the V decision lock"
            saved = json.loads(lock.read_bytes())
            assert saved["T_read"] is False
        self.roles.append(split.role)
        discrete = any(port.is_discrete for port in model.ports.values())
        if discrete and self.discrete_V_preferred:
            selected = any(port.discrete_choice == self.discrete_V_preferred for port in model.ports.values())
            # Deliberately reversed held-out preference proves the choice uses V.
            return (0.9 if selected else 0.1) if split.role == "V" else (0.2 if selected else 0.8)
        return super().score(model, split)


class MechanismTests(unittest.TestCase):
    def setUp(self):
        self.c = load_config(ROOT / "configs" / "fixture.json")
        self.backend = ObservedBackend()
        self.cache = PackageCache(1_000_000, self.backend.specs)
        self.parent = initial_state(self.backend, self.cache, Structure((1, 0, 0, 0, 0, 0), 1 / 8),
                                    FitSettings(1e-3, 1e-3), 41)
        with torch.random.fork_rng():
            torch.manual_seed(7)
            adapter = Adapter(8, 1 / 8)
            torch.nn.init.normal_(adapter.up.weight, std=0.2)
        self.package = Package("peer", "independent", 1, 2, self.backend.specs[2], 1 / 8,
                               freeze_tensors(adapter.state_dict()), "independently generated test branch")
        self.cache.put(self.package.hash, self.package.wire)
        self.parent = replace(self.parent, C=(self.package.hash,), beta=((self.package.hash, 0.7),))
        model = self.backend.assembly(self.parent, self.cache, mode="refine")
        model.ports["0"].local.up.weight.data.fill_(0.7)
        self.parent = model.snapshot(self.parent)
        self.D, self.V, self.T = [synthetic_split(role, "0", 2, "a", 100 + i) for i, role in enumerate(("D", "V", "T"))]
        self.spec = {"snapshot_time": 3, "fit_epochs": 2, "initialization_seed": 50,
                     "max_onehot_candidates": 8, "budget": {"F": 1e12, "E": 100, "B": 0, "W": 120}}

    def test_native_continuation_fields_are_explicit_and_membership_is_fixed(self):
        branches = continuation_states(self.parent, self.backend, self.cache, 50)
        self.assertEqual(branches["complete_state"].phi, self.parent.phi)
        self.assertEqual(branches["complete_state"].beta, self.parent.beta)
        self.assertEqual(branches["reset_receiving"].phi, self.parent.phi)
        self.assertEqual(branches["reset_receiving"].beta, ((self.package.hash, 0.0),))
        self.assertNotEqual(branches["cold_local"].phi, self.parent.phi)
        self.assertEqual(branches["cold_local"].beta, branches["reset_receiving"].beta)
        for state in branches.values():
            self.assertEqual((state.z, state.h, state.C), (self.parent.z, self.parent.h, self.parent.C))
            self.assertNotIn("momentum", json.loads(state.nu))
        again = continuation_states(self.parent, self.backend, self.cache, 50)
        self.assertEqual(branches, again)

    def test_receiving_preserves_local_weights_and_locks_V_before_T(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.backend.output = Path(tmp)
            self.backend.discrete_V_preferred = self.package.hash
            result = receiving_comparison(self.backend, self.cache, self.D, self.V, self.T,
                                          self.parent, self.spec, self.c, tmp)
            self.assertEqual(result["status"], "completed")
            rows = {row["branch"]: row for row in result["rows"]}
            self.assertEqual(rows["uniform"]["completed_fit_steps"], 0)
            self.assertFalse(rows["uniform"]["optimizer_created"])
            self.assertEqual(rows["fitted"]["completed_fit_steps"], 2 * len(self.D.batches))
            self.assertEqual(len(rows["onehot_V_selected"]["V_candidates"]), 2)
            self.assertEqual(rows["onehot_V_selected"]["V"], 0.9)
            self.assertEqual(rows["onehot_V_selected"]["T"], 0.2)
            chosen = State.load(Path(tmp) / "onehot_V_selected.json")
            self.assertIn(self.package.hash, json.loads(chosen.nu)["discrete_choices"].values())
            for name in ("uniform", "fitted", "onehot_V_selected"):
                state = State.load(Path(tmp) / (name + ".json"))
                self.assertEqual(state.phi, self.parent.phi)
                self.assertEqual(state.C, self.parent.C)
            first_T = self.backend.roles.index("T")
            self.assertNotIn("V", self.backend.roles[first_T:])
            self.assertTrue(self.backend.fit_modes)
            self.assertTrue(all(not modes["local_trainable"] and modes["beta_trainable"] for modes in self.backend.fit_modes))
            self.assertFalse(result["contract"]["same_compute_claim"])

    def test_onehot_cap_refuses_full_cartesian_product_before_fitting(self):
        spec = {**self.spec, "max_onehot_candidates": 1}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "result"
            with self.assertRaisesRegex(ValueError, "requires 2 candidates"):
                receiving_comparison(self.backend, self.cache, self.D, self.V, self.T,
                                     self.parent, spec, self.c, output)
            self.assertFalse(output.exists())
            self.assertFalse(self.backend.fit_modes)

    def test_receiving_empty_members_remain_registered_with_zero_fit(self):
        parent = replace(self.parent, C=(), beta=())
        with tempfile.TemporaryDirectory() as tmp:
            self.backend.output = Path(tmp)
            result = receiving_comparison(self.backend, self.cache, self.D, self.V, self.T,
                                          parent, self.spec, self.c, tmp)
            self.assertTrue(result["contract"]["empty_membership"])
            self.assertEqual(result["contract"]["onehot_candidates"], 1)
            self.assertTrue(all(row["completed_fit_steps"] == 0 for row in result["rows"]))
            self.assertFalse(self.backend.fit_modes)

    def test_failed_fit_keeps_work_and_does_not_get_a_zero_T_score(self):
        self.backend.fail_fit = True
        with tempfile.TemporaryDirectory() as tmp:
            self.backend.output = Path(tmp)
            result = receiving_comparison(self.backend, self.cache, self.D, self.V, self.T,
                                          self.parent, self.spec, self.c, tmp)
            fitted = result["rows"][1]
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(fitted["status"], "failed")
            self.assertEqual(fitted["attempted_fit_steps"], 1)
            self.assertEqual(fitted["completed_fit_steps"], 0)
            self.assertGreater(fitted["actual"]["F"], 0)
            self.assertGreater(fitted["actual"]["W"], 0)
            self.assertNotIn("T", fitted)

    def test_continuation_shares_schedule_but_reports_measured_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.backend.output = Path(tmp)
            result = continuation_comparison(self.backend, self.cache, self.D, self.V, self.T,
                                             self.parent, self.spec, self.c, tmp)
            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["matched_work_eligible"])
            self.assertEqual(len(result["rows"]), 3)
            schedules = [row["epoch_seeds"] for row in result["rows"]]
            self.assertTrue(all(schedule == schedules[0] for schedule in schedules))
            self.assertTrue(all(row["completed_fit_steps"] == 2 * len(self.D.batches) for row in result["rows"]))
            self.assertIn("momentum", result["contract"]["historical_difference"])
            for row in result["rows"]:
                self.assertGreater(row["actual"]["F"], 0)
                self.assertIn("mechanism_" + row["branch"] + "_initialize", [event["reason"] for event in row["ledger"]])

    def test_cold_initialization_failure_is_recorded_in_its_branch_ledger(self):
        original = self.backend.assembly
        def assembly(state, cache, seed=0, mode="import"):
            if mode == "fresh":
                raise RuntimeError("controlled initialization failure")
            return original(state, cache, seed, mode)
        with tempfile.TemporaryDirectory() as tmp, patch.object(self.backend, "assembly", side_effect=assembly):
            self.backend.output = Path(tmp)
            result = continuation_comparison(self.backend, self.cache, self.D, self.V, self.T,
                                             self.parent, self.spec, self.c, tmp)
            cold = next(row for row in result["rows"] if row["branch"] == "cold_local")
            self.assertEqual(cold["failure_phase"], "initialization")
            self.assertEqual(cold["status"], "failed")
            self.assertFalse(result["matched_work_eligible"])
            self.assertIsNone(result["contract"]["initial_intervention_changes"]["cold_local_changes_phi"])
            self.assertIn("mechanism_cold_local_initialize", [event["reason"] for event in cold["ledger"]])
            self.assertNotIn("T", cold)

    def test_continuation_failure_is_not_eligible_matched_work(self):
        self.backend.fail_fit = True
        with tempfile.TemporaryDirectory() as tmp:
            self.backend.output = Path(tmp)
            result = continuation_comparison(self.backend, self.cache, self.D, self.V, self.T,
                                             self.parent, self.spec, self.c, tmp)
            self.assertFalse(result["matched_work_eligible"])
            self.assertTrue(all("T" not in row for row in result["rows"]))

    def test_invalid_role_and_future_version_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "D/V/T"):
                receiving_comparison(self.backend, self.cache, self.D, self.T, self.V,
                                     self.parent, self.spec, self.c, tmp)
            with self.assertRaisesRegex(ValueError, "unavailable"):
                receiving_comparison(self.backend, self.cache, self.D, self.V, self.T,
                                     self.parent, {**self.spec, "snapshot_time": 1}, self.c, tmp)

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            sentinel = Path(tmp) / "old.json"
            sentinel.write_text("old")
            with self.assertRaises(FileExistsError):
                continuation_comparison(self.backend, self.cache, self.D, self.V, self.T,
                                         self.parent, self.spec, self.c, tmp)
            self.assertEqual(sentinel.read_text(), "old")

    def test_discrete_parent_is_not_silently_changed_to_soft(self):
        parent = self.parent.lineage(receiving_rule="discrete", discrete_choices={"2": self.package.hash})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "soft-receiving parent"):
                continuation_comparison(self.backend, self.cache, self.D, self.V, self.T,
                                         parent, self.spec, self.c, tmp)

    def work_row(self, policy, seed=1, **kwargs):
        return {"dataset": "d", "detector": "p", "seed": seed, "policy": policy,
                "status": "completed", "work_unit": "executed_FLOPs", "target_work": 100,
                "actual_work": 99, "value": 0.3 if policy == "Full" else 0.4, **kwargs}

    def test_work_match_preserves_negative_effect_and_rejects_missing_and_underwork(self):
        rows = [self.work_row("Full"), self.work_row("Control"),
                self.work_row("Full", 2), self.work_row("Control", 2, actual_work=50),
                self.work_row("Full", 3), self.work_row("Control", 3, value=None), self.work_row("Full", 4)]
        result = match_completed_work(rows, ["Full", "Control"])
        self.assertEqual(result["registered_pairs"], 4)
        self.assertEqual(result["eligible_pairs"], 1)
        self.assertAlmostEqual(result["pairs"][0]["differences"]["Control"], -0.1)
        self.assertTrue(all(row["differences"] is None for row in result["pairs"][1:]))
        self.assertFalse(result["missing_values_imputed"])

    def test_work_match_rejects_failed_over_ceiling_different_targets_and_duplicate(self):
        for kwargs in ({"status": "failed"}, {"actual_work": 101}, {"target_work": 101}, {"work_unit": "epochs"}):
            result = match_completed_work([self.work_row("Full"), self.work_row("Control", **kwargs)], ["Full", "Control"])
            self.assertEqual(result["eligible_pairs"], 0)
        result = match_completed_work([self.work_row("Full"), self.work_row("Full"), self.work_row("Control")], ["Full", "Control"])
        self.assertEqual(result["eligible_pairs"], 0)

    def test_matched_work_job_requires_no_native_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = Path(tmp) / "records.json"
            records.write_text(json.dumps([self.work_row("Full"), self.work_row("Control")]))
            result = run_mechanism_job({"kind": "matched_work", "records": str(records), "policies": ["Full", "Control"],
                                        "output": str(Path(tmp) / "new")})
            self.assertEqual(result["eligible_pairs"], 1)
            self.assertTrue((Path(tmp) / "new" / "mechanism.json").is_file())

    def test_expected_pairs_keep_wholly_missing_pairs_without_imputation(self):
        expected = [{"dataset": "d", "detector": "p", "seed": seed} for seed in (1, 2)]
        rows = [self.work_row("Full"), self.work_row("Control")]
        result = match_completed_work(rows, ["Full", "Control"], expected_pairs=expected)
        self.assertEqual(result["registered_pairs"], 2)
        self.assertEqual(result["eligible_pairs"], 1)
        missing = next(item for item in result["pairs"] if item["pair"]["seed"] == 2)
        self.assertFalse(missing["eligible"])
        self.assertIsNone(missing["differences"])
        self.assertEqual(missing["observations"], [])
        self.assertEqual(len(missing["reasons"]), 2)
        empty = match_completed_work([], ["Full", "Control"], expected_pairs=expected)
        self.assertEqual(empty["registered_pairs"], 2)
        self.assertEqual(empty["eligible_pairs"], 0)
        with self.assertRaisesRegex(ValueError, "outside"):
            match_completed_work([self.work_row("Full", seed=3)], ["Full", "Control"], expected_pairs=expected)

    def test_matched_work_job_passes_expected_pair_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = Path(tmp) / "records.json"
            records.write_text("[]")
            expected = [{"dataset": "d", "detector": "p", "seed": 1}]
            result = run_mechanism_job({"kind": "matched_work", "records": str(records), "policies": ["Full", "Control"],
                                        "expected_pairs": expected, "output": str(Path(tmp) / "new")})
            self.assertEqual(result["registered_pairs"], 1)
            self.assertTrue(result["expected_pairs_locked"])
            self.assertFalse(result["pairs"][0]["eligible"])


if __name__ == "__main__":
    unittest.main()
