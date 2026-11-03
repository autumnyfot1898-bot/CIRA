import copy
import unittest

from softcas.study import build_study, primary_contrasts, summarize_completed


def example():
    datasets = ["ControlledFog", "ACDC"]
    detectors = ["YOLOv8n", "YOLO11n", "YOLO26n"]
    return dict(datasets=datasets, detectors=detectors, seeds=list(range(1001, 1006)),
                development_selection=dict(controller_q="Q0", locked=False, sha256=None,
                    serial_rules={d+"/"+p: "cfg50_collab50" for d in datasets for p in detectors}))


class StudyRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.study = build_study(example())

    def test_counts_match_prespecified_matrix(self):
        expected_new = [420, 180, 240, 360, 120, 450, 120, 300, 180, 120, 90, 840]
        expected_views = [420, 210, 270, 540, 180, 600, 240, 450, 360, 480, 120, 1260]
        self.assertEqual([b["new_trajectories"] for b in self.study["blocks"]], expected_new)
        self.assertEqual([b["analysis_views"] for b in self.study["blocks"]], expected_views)
        self.assertEqual(self.study["unique_trajectories"], 3420)
        self.assertEqual(self.study["analysis_views"], 5130)
        self.assertEqual(len(self.study["rgb_endpoint_views"]), 1260)
        self.assertEqual(self.study["extra_rgb_training"], 0)

    def test_selected_alternative_changes_anchor_not_count(self):
        spec = example()
        spec["development_selection"]["controller_q"] = "Q2"
        spec["development_selection"]["serial_rules"]["ACDC/YOLOv8n"] = "cfg25_collab75"
        other = build_study(spec)
        self.assertEqual(other["unique_trajectories"], 3420)
        self.assertNotEqual(other["trajectories"][0]["id"], other["trajectories"][-1]["id"])
        self.assertNotEqual({r["id"] for r in other["trajectories"]}, {r["id"] for r in self.study["trajectories"]})

    def test_no_unbound_or_external_recipe_is_runnable(self):
        self.assertTrue(all(not r["runnable"] for r in self.study["trajectories"]))
        self.assertTrue(all(r["status"] == "planned_not_run" for r in self.study["trajectories"]))
        self.assertEqual(self.study["external_adapters_implemented"], [])

    def test_local_donor_levels_share_exact_anchor(self):
        rows = {r["id"]: r for r in self.study["trajectories"]}
        views = [v for v in self.study["views"] if v["block"] == "N09" and
                 rows[v["trajectory_id"]]["recipe"]["method"] == "local"]
        self.assertEqual(len(views), 90)
        self.assertEqual(len({v["trajectory_id"] for v in views}), 30)

    def test_duplicate_seed_and_unqualified_lock_rejected(self):
        spec = example()
        spec["seeds"] += [1001]
        with self.assertRaises(ValueError): build_study(spec)
        spec = example()
        spec["development_selection"]["locked"] = "false"
        with self.assertRaises(ValueError): build_study(spec)
        spec = example()
        spec["development_selection"]["locked"] = True
        with self.assertRaises(ValueError): build_study(spec)

    def result(self):
        row = self.study["trajectories"][0]
        return dict(study_trajectory_id=row["id"], study_recipe=copy.deepcopy(row["recipe"]),
                    complete=True, seen=.5)

    def test_aliases_do_not_increase_independent_count(self):
        result = summarize_completed(self.study, [self.result()])
        self.assertEqual(result["independent_completed"], 1)
        self.assertEqual(result["blocks"][0]["complete"], 1)
        self.assertEqual(summarize_completed(self.study, []) ["independent_completed"], 0)

    def test_duplicate_mismatched_and_incomplete_results(self):
        row = self.result()
        with self.assertRaises(ValueError): summarize_completed(self.study, [row, row])
        row["study_recipe"]["factors"]["rounds"] = 500
        with self.assertRaises(ValueError): summarize_completed(self.study, [row])
        row = self.result()
        row["complete"] = False
        self.assertEqual(summarize_completed(self.study, [row])["independent_completed"], 0)

    def test_missing_baseline_stays_inestimable_in_holm_family(self):
        rows = primary_contrasts(self.study, [self.result()])["contrasts"]
        self.assertEqual(len(rows), 12)
        self.assertTrue(all(r["n_pair"] == 0 and r["mean"] is None and r["p_holm"] is None for r in rows))

    def test_registry_tampering_and_endpoint_units_rejected(self):
        study = copy.deepcopy(self.study)
        study["trajectories"][0]["recipe"]["factors"]["rounds"] = 17
        with self.assertRaises(ValueError): summarize_completed(study, [])
        study = copy.deepcopy(self.study)
        study["trajectories"].append(study["trajectories"][0])
        with self.assertRaises(ValueError): summarize_completed(study, [])
        row = self.result()
        row["seen"] = 50.
        with self.assertRaises(ValueError): summarize_completed(self.study, [row])


if __name__ == "__main__":
    unittest.main()
