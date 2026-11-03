import copy
import json
import tempfile
import unittest
from pathlib import Path

from softcas.decision_audit import audit_actions, audit_files, replay_choice


class DecisionAuditTests(unittest.TestCase):
    def setUp(self):
        self.constants = {"kappa": .01, "epsilon": .001}
        common = {"parent_utility": 0., "champion_utility": 0., "coverage": 1.,
                  "estimate": dict(F=1., E=0., B=0., W=0.),
                  "remaining": dict(F=10., E=10., B=10., W=10.), "features": [1.]}
        self.pool = [dict(common, proposal="a", prediction=-.1, tie_rank=0),
                     dict(common, proposal="b", prediction=.2, tie_rank=1)]

    def action(self, **updates):
        return dict(dict(attempt=1, candidate_pool=copy.deepcopy(self.pool),
                         priority_constants=self.constants, decision="priority", proposal="b",
                         gain=.1, prediction=.2, status="complete", no_gain=False), **updates)

    def test_replay_detects_changed_selection_without_child_labels(self):
        self.assertEqual(replay_choice(self.pool, self.constants), "b")
        self.assertEqual(replay_choice(self.pool, self.constants, True), "a")
        result = audit_actions([self.action()])
        self.assertEqual(result["changed_fraction"], 1.)
        self.assertTrue(result["all_selected_replays_match"])
        self.assertNotIn("counterfactual_gain", result)

    def test_cold_choice_is_excluded_and_old_logs_remain_missing(self):
        old = self.action(decision="cold_type")
        old.pop("candidate_pool")
        result = audit_actions([old, self.action(attempt=2, decision="cold_local_fresh")])
        self.assertEqual(result["priority_decisions"], 0)
        self.assertIsNone(result["changed_fraction"])
        self.assertEqual(result["missing_candidate_pools"], 1)

    def test_running_mean_only_uses_past_targets(self):
        first = self.action(gain=1., prediction=0.)
        second = self.action(attempt=2, gain=3., prediction=1., status="failed")
        result = audit_actions([first, second])
        self.assertEqual(result["past_mean_baseline_mae"], 1.5)
        self.assertEqual(result["zero_baseline_mae"], 2.)
        self.assertEqual(result["completed_prediction_mae"], 1.)

    def test_feature_collisions_and_negative_reuse_are_observed(self):
        first = self.action(gain=-.1, child="negative", archive_after=["negative"])
        second = self.action(attempt=2, parent="negative")
        result = audit_actions([first, second])
        self.assertEqual(len(result["feature_collision_groups"]), 2)
        self.assertEqual(result["negative_retained"], 1)
        self.assertEqual(result["negative_reused_as_parent"], 1)

    def test_no_positive_priorities_returns_keep(self):
        pool = copy.deepcopy(self.pool)
        for row in pool:
            row.update(prediction=-1., coverage=0.)
        self.assertIsNone(replay_choice(pool, self.constants))

    def test_reordered_or_duplicated_attempts_rejected(self):
        with self.assertRaises(ValueError): audit_actions([self.action(), self.action()])
        with self.assertRaises(ValueError): audit_actions([self.action(attempt=2)])

    def test_copied_stage_file_cannot_be_double_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp)/"first.jsonl", Path(tmp)/"copy.jsonl"]
            for path in paths:
                path.write_text(json.dumps(self.action()) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError): audit_files(paths)


if __name__ == "__main__":
    unittest.main()
