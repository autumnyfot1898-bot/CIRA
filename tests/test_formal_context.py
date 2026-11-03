"""Formal context must follow effective config through execution (no GPU runs)."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from softcas.config import PENDING, effective_formal, load_config
from softcas.diagnostics import run_diagnostic_job
from softcas.experiment import build_plan, execute_trajectory, trajectory_config


ROOT = Path(__file__).resolve().parents[1]


class FormalContextTests(unittest.TestCase):
    def formal_config(self):
        value = load_config(ROOT / "configs/native_r1000.json")
        value.update(purpose="FORMAL", confirmed={key: True for key in PENDING},
                     cost_profile="bound_cost.json", flop_meter="scalar_ops_mac2_v1")
        value["meter"]["strict"] = True
        return value

    def write(self, root, name, value):
        path = root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return str(path)

    def job(self, root, config):
        labels = [str(i) for i in range(8)]
        return dict(config=self.write(root, "algorithm.json", config), method="softcas", seed=1001,
                    detector="test_detector", dataset="test_dataset", clients=["0"], registry_id="test",
                    registry=self.write(root, "registry.json", dict(labels=labels, records=[])),
                    detector_config=self.write(root, "detector.json", dict(labels=labels, ports=list(range(6)),
                                              preprocessing="test", image_size=16)),
                    checkpoint=self.write(root, "checkpoint.json", {}),
                    contact_trace=self.write(root, "trace.json", []), output=str(root / "run"))

    def test_effective_context_never_downgrades_formal_and_rejects_truthy_flags(self):
        for flag in (False, True):
            self.assertTrue(effective_formal({"purpose": "FORMAL"}, flag))
        self.assertFalse(effective_formal({"purpose": "DEVELOPMENT"}))
        for flag in ("false", "true", 0, 1, None, [], {}):
            with self.subTest(flag=flag), self.assertRaisesRegex(ValueError, "boolean"):
                effective_formal({"purpose": "FORMAL"}, flag)

    def test_final_inherited_config_is_validated_after_job_bindings(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parent = self.formal_config()
            parent["confirmed"] = {}
            self.write(root, "parent.json", parent)
            child = self.write(root, "child.json", {"extends": "parent.json"})
            with self.assertRaisesRegex(ValueError, "Formal execution blocked"):
                load_config(child)
            result = load_config(child, overrides={"confirmed": {k: True for k in PENDING}})
            self.assertEqual(result["purpose"], "FORMAL")
            self.assertTrue(all(result["confirmed"].values()))

    def test_trajectory_overrides_are_applied_before_gate_without_mutating_job(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = self.formal_config()
            config["confirmed"] = {}
            job = self.job(root, config)
            job.update(formal=False, overrides={"confirmed": {k: True for k in PENDING}, "runtime": {"seed": 9}})
            original = copy.deepcopy(job)
            resolved, formal = trajectory_config(job)
            self.assertTrue(formal)
            self.assertEqual(resolved["runtime"]["seed"], 1001)
            self.assertEqual(job, original)

    def test_formal_config_with_missing_or_false_job_flag_enforces_scene_grouping(self):
        for explicit in (None, False):
            with self.subTest(explicit=explicit), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                job = self.job(root, self.formal_config())
                if explicit is not None:
                    job["formal"] = explicit
                self.write(root, "registry.json", dict(labels=[str(i) for i in range(8)], records=[
                    dict(image="image.png", image_sha256="hash", scene_id="s", role="D",
                         domain="fog", client="0", stage=1)]))
                with patch("softcas.experiment.YoloBackend") as backend:
                    with self.assertRaisesRegex(ValueError, "verified scene"):
                        execute_trajectory(job)
                    backend.assert_not_called()

    def test_formal_cost_profile_cannot_be_loaded_in_nonstrict_mode(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            job = self.job(root, self.formal_config())
            job["formal"] = False
            backend = SimpleNamespace(runtime=SimpleNamespace(identity=lambda: {"device": "bound"}))
            with patch("softcas.experiment.YoloBackend", return_value=backend), \
                 patch("softcas.experiment.CostProfile", side_effect=RuntimeError("cost sentinel")) as profile:
                with self.assertRaisesRegex(RuntimeError, "cost sentinel"):
                    execute_trajectory(job)
            self.assertIs(profile.call_args.args[2], True)

    def test_trajectory_forwards_formal_to_online_roles_and_remote_release(self):
        from softcas.data import synthetic_split
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            job = self.job(root, self.formal_config())
            job.update(formal=False, stage_events=[dict(client="0", stage=1, domain="fog")])
            backend = SimpleNamespace(identity="base", execution="bound", specs=(),
                                      runtime=SimpleNamespace(identity=lambda: {}))
            profile = SimpleNamespace(data=dict(base_sha256="base", execution="bound", port_interfaces=[]))
            state = SimpleNamespace(C=())
            def split_loader(*args, **kwargs):
                return synthetic_split(args[3], "0", 1, "fog", 1)
            with patch("softcas.experiment.YoloBackend", return_value=backend), \
                 patch("softcas.experiment.CostProfile", return_value=profile), \
                 patch("softcas.experiment.MemoryGuard"), patch("softcas.experiment.PackageCache"), \
                 patch("softcas.experiment.CheckpointStore"), patch("softcas.experiment.PeerService"), \
                 patch("softcas.experiment.PeerServer"), patch("softcas.experiment.threading.Thread"), \
                 patch("softcas.experiment.initial_state", return_value=state), \
                 patch("softcas.experiment.load_detection_split", side_effect=split_loader) as splits, \
                 patch("softcas.experiment.RemoteReleaseService", side_effect=RuntimeError("release sentinel")) as remote:
                with self.assertRaisesRegex(RuntimeError, "release sentinel"):
                    execute_trajectory(job)
            self.assertEqual([call.args[3] for call in splits.call_args_list], ["D", "V", "A"])
            self.assertTrue(all(call.kwargs["formal"] is True for call in splits.call_args_list))
            self.assertIs(remote.call_args.args[3], True)

    def test_diagnostic_forwards_formal_to_all_splits_and_cost_profile(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            job = self.job(root, self.formal_config())
            job.update(formal=False, client="0", stage=1, domain="fog", diagnostic={})
            backend = SimpleNamespace(identity="base", execution="bound", specs=(),
                                      runtime=SimpleNamespace(identity=lambda: {"device": "bound"}))
            profile = SimpleNamespace(data=dict(base_sha256="base", execution="bound", port_interfaces=[]))
            with patch("softcas.backend.YoloBackend", return_value=backend), \
                 patch("softcas.metering.CostProfile", return_value=profile) as cost, \
                 patch("softcas.cache.PackageCache"), \
                 patch("softcas.data.load_detection_split", return_value=None) as split, \
                 patch("softcas.diagnostics.configuration_composition", return_value={"test": True}):
                self.assertEqual(run_diagnostic_job(job), {"test": True})
            self.assertIs(cost.call_args.args[2], True)
            self.assertEqual([call.args[3] for call in split.call_args_list], ["D", "V", "T"])
            self.assertTrue(all(call.kwargs["formal"] is True for call in split.call_args_list))

    def test_diagnostic_checks_cost_profile_exact_base_and_interfaces(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            job = self.job(root, self.formal_config())
            job.update(client="0", stage=1, domain="fog", diagnostic={})
            backend = SimpleNamespace(identity="base", execution="bound", specs=(),
                                      runtime=SimpleNamespace(identity=lambda: {}))
            wrong = SimpleNamespace(data=dict(base_sha256="different", execution="bound", port_interfaces=[]))
            with patch("softcas.backend.YoloBackend", return_value=backend), \
                 patch("softcas.metering.CostProfile", return_value=wrong), \
                 patch("softcas.data.load_detection_split") as split:
                with self.assertRaisesRegex(ValueError, "exact frozen base"):
                    run_diagnostic_job(job)
                split.assert_not_called()

    def test_plan_resolves_readable_formal_config_and_requires_serial_development(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            job = self.job(root, self.formal_config())
            key = "test_dataset/test_detector"
            spec = dict(datasets={"test_dataset": dict(registries={"1001": job["registry"]}, contact_trace=job["contact_trace"])},
                        detectors={"test_detector": dict(config=job["detector_config"], checkpoint=job["checkpoint"])},
                        seeds=[1001], clients=["0"], primary_detector="test_detector", algorithm_config=job["config"],
                        output_root=str(root / "plan"), formal=False,
                        serial_selection={key: {"rule": dict(order=["configuration", "collaboration"], fraction=.5, attempt_fraction=.5)}})
            with self.assertRaisesRegex(ValueError, "development selection record"):
                build_plan(spec)
            spec["serial_selection"][key]["development_record_sha256"] = "locked-independent-record"
            plan = build_plan(spec)
            self.assertTrue(all(job["formal"] is True for job in plan["jobs"]))


if __name__ == "__main__":
    unittest.main()
