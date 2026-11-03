import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from softcas.config import load_config
from softcas.state import canonical, digest
from softcas.study import build_study
from softcas.study_execution import (binding_template, compile_study, condition_key,
                                    condition_identity, execute_study, file_binding, preflight_job)

ROOT = Path(__file__).resolve().parents[1]


class StudyExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.selection = dict(controller_q="Q2", locked=True, sha256="a" * 64,
                              serial_rules={"ControlledFog/YOLOv8n": "cfg50_collab50"})
        self.study = build_study(dict(datasets=["ControlledFog"], detectors=["YOLOv8n"], seeds=[1001],
                                     development_selection=self.selection))
        self.spec = binding_template(self.study, {"blocks": ["N01"], "methods": ["softcas"]})
        self.spec.update(formal=False, output_root=str(self.root / "runs"), clients=["0", "1"])
        cfg = load_config(ROOT / "configs/fixture.json")
        cfg.update(proposal_protocol="native_r1000", fitting=dict(unit="epochs",stage_epochs=500,local_epochs=32,receiving_epochs=16))
        cell = "ControlledFog/YOLOv8n"
        dev = dict(schema="cira.development_lock.v1", role="DEV", selection=self.selection,
                   seeds=[9001, 9002, 9003], cells={cell: {"fixed": cfg["fixed"]}})
        self.spec["development_lock"] = self.asset("dev.json", dev)
        bound = self.spec["cells"][cell]
        bound.update(config=self.asset("cfg.json", cfg), checkpoint=self.asset("base.pt", {"test_only": True}),
                     detector_config=self.asset("detector.json", {"test_only": True}))
        for rounds in (300, 500, 1000):
            bound["budgets_by_round"][str(rounds)] = self.asset(f"budget{rounds}.json", dict(
                rounds=rounds, cell=cell, search=dict(F=rounds*10.,E=65,B=1024,W=200.),
                release=dict(F=10.,E=0,B=1024,W=10.)))
        self.registry = self.asset("registry.json", {"test_only": True})
        self.trace = self.asset("trace.json", [])
        self.bind_conditions()

    def asset(self, name, value):
        path = self.root / name
        path.write_bytes(canonical(value))
        return file_binding(path)

    def bind_conditions(self):
        for row in self.study["trajectories"]:
            key = condition_key(row["recipe"])
            if key in self.spec["conditions"]:
                self.spec["conditions"][key]["manifest"] = self.asset(key+".json", dict(
                    schema="cira.condition.v1", complete=True, condition=condition_identity(row["recipe"]),
                    outputs={"registry": self.registry, "contact_trace": self.trace}, interventions={}))

    def compile(self):
        plan = compile_study(self.study, self.spec)
        self.assertEqual(plan["blocked"], [])
        return plan

    def test_compile_pins_actual_assets_and_selected_q_without_starting_training(self):
        with patch("softcas.experiment.execute_trajectory") as run:
            job = self.compile()["jobs"][0]
            self.assertEqual(run.call_count, 0)
        self.assertEqual(job["overrides"]["ridge_lambda"], 10.)
        self.assertEqual(job["overrides"]["fitting"]["stage_epochs"], 500)
        self.assertTrue(preflight_job(job))
        self.assertEqual(job["executor"], "internal")

    def test_changed_inputs_or_configuration_cannot_execute_old_plan(self):
        job = self.compile()["jobs"][0]
        Path(job["registry"]).write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed before execution"): preflight_job(job)

    def test_short_round_and_compute_are_real_overrides(self):
        wanted = next(r for r in self.study["trajectories"] if r["recipe"]["method"] == "softcas" and r["recipe"]["factors"]["rounds"] == 300)
        self.spec["selection"] = {"ids": [wanted["id"]]}
        job = self.compile()["jobs"][0]
        self.assertEqual(job["overrides"]["fitting"]["stage_epochs"], 150)
        self.assertEqual(job["overrides"]["budgets"]["search"]["F"], 3000.)
        wanted = next(r for r in self.study["trajectories"] if r["recipe"]["method"] == "softcas" and r["recipe"]["factors"]["compute"] == .5)
        self.spec["selection"] = {"ids": [wanted["id"]]}
        self.assertEqual(self.compile()["jobs"][0]["overrides"]["budgets"]["search"]["F"], 5000.)

    def test_missing_or_mismatched_condition_stays_blocked(self):
        entry = next(iter(self.spec["conditions"].values()))
        entry["condition"]["contacts"] = "half_visibility"
        plan = compile_study(self.study, self.spec)
        self.assertEqual(plan["ready_trajectories"], 0)
        self.assertIn("Condition binding", plan["blocked"][0]["reason"])
        with self.assertRaisesRegex(ValueError, "No fully bound"): execute_study(plan)

    def test_unlocked_or_wrong_study_lock_is_not_ready(self):
        self.spec["study_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "this study"): compile_study(self.study, self.spec)
        self.spec["study_sha256"] = digest(self.study)
        self.spec["development_lock"] = dict(path=None, sha256=None)
        plan = compile_study(self.study, self.spec)
        self.assertEqual(len(plan["blocked"]), 1)

    def test_explicit_dispatch_and_same_recipe_result(self):
        plan = self.compile()
        def run(job, resume):
            return dict(study_trajectory_id=job["study_trajectory_id"], study_recipe=job["study_recipe"], complete=True)
        with patch("softcas.experiment.execute_trajectory", side_effect=run) as runner:
            result = execute_study(plan)
        self.assertEqual(runner.call_count, 1)
        self.assertEqual(result[0]["study_recipe"], plan["jobs"][0]["study_recipe"])

    def test_physical_plan_has_four_nodes_and_requires_node_id(self):
        self.spec.update(clients=["0","1","2","3"], epoch_start_unix=1900000000,
                         physical_nodes={str(i): {"endpoint": {"host":"127.0.0.1","port":18000+i}} for i in range(4)})
        plan = self.compile()
        self.assertEqual(len(plan["jobs"]), 4)
        with self.assertRaisesRegex(ValueError, "node-id"): execute_study(plan)
        with patch("softcas.experiment.execute_trajectory", return_value={}) as runner:
            execute_study(plan, node_id="2")
            self.assertEqual(runner.call_args[0][0]["clients"], ["2"])


if __name__ == "__main__": unittest.main()
