"""Regression checks for orchestration/reporting; no native training or device work."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from softcas import cli, production
from softcas.backend import Backend
from softcas.cache import PackageCache
from softcas.config import load_config
from softcas.data import synthetic_split
from softcas.diagnostics import configuration_composition
from softcas.experiment import join_physical_nodes, build_plan, execute_plan, execution_scope
from softcas.model import Adapter
from softcas.reporting import paired_effects
from softcas.state import Package, State, freeze_tensors
from softcas.tables import CELLS, export_tables


class ReportingRepairTests(unittest.TestCase):
    def record(self, **fields):
        return dict(purpose="FORMAL", metric="AP50:95", complete=True,
                    method="softcas", cell=CELLS[0], seed=1001, seen=None, **fields)

    def test_missing_paired_endpoints_stay_missing_and_do_not_change_n(self):
        values = paired_effects({("cell", 1001): None, ("cell", 1002): 0.8},
                                {("cell", 1001): 0.7, ("cell", 1002): 0.6}, ["cell"])[0]
        self.assertEqual(values["n_pair"], 1)
        self.assertIsNone(values["paired_differences"][0])
        self.assertAlmostEqual(values["mean"], 0.2)
        self.assertIsNone(values["ci95_unadjusted"])

    def test_invalid_pairs_and_duplicate_seed_list_are_rejected(self):
        with self.assertRaises(ValueError):
            paired_effects({("cell", 1001): float("nan")}, {("cell", 1001): 0.5}, ["cell"])
        with self.assertRaises(ValueError):
            paired_effects({}, {}, ["cell"], seeds=(1001, 1001))

    def test_table_export_handles_none_and_keeps_blank_primary_cell(self):
        records = [self.record(), {**self.record(), "method": "serial", "seen": 0.7}]
        with tempfile.TemporaryDirectory() as folder:
            export_tables(records, folder)
            paired = json.loads(Path(folder, "T7_paired.json").read_bytes())[0]
            self.assertEqual(paired["n_pair"], 0)
            with Path(folder, "T2_primary.csv").open(encoding="utf-8-sig", newline="") as f:
                full = next(r for r in csv.DictReader(f) if r["method"] == "softcas" and r["cell"] == CELLS[0])
            self.assertEqual((full["n"], full["mean_percent"]), ("0", ""))

    def test_table_rejects_incomplete_out_of_range_or_silently_omitted_records(self):
        for fields in [dict(complete=False), dict(method="unknown"), dict(cell="unknown"),
                       dict(seen=70.0), dict(seen=float("inf")), dict(ablation="no_gain")]:
            with self.subTest(fields=fields), tempfile.TemporaryDirectory() as folder:
                with self.assertRaises(ValueError):
                    export_tables([{**self.record(), **fields}], folder)


class CliRepairTests(unittest.TestCase):
    def test_directory_output_is_not_overwritten_as_a_json_file(self):
        for command in ["join-nodes", "diagnostic"]:
            with self.subTest(command=command), tempfile.TemporaryDirectory() as folder:
                output = Path(folder, "results")
                spec = Path(folder, "spec.json")
                spec.write_text(json.dumps(dict(node_results=[], output="unused")), encoding="utf-8")
                def run(*args):
                    actual_output = args[1] if command == "join-nodes" else args[0]["output"]
                    self.assertEqual(Path(actual_output), output)
                    output.mkdir()
                    (output / "result.json").write_text("{}", encoding="utf-8")
                    return {"complete": True}
                symbol = "softcas.experiment.join_physical_nodes" if command == "join-nodes" else "softcas.diagnostics.run_diagnostic_job"
                with patch(symbol, side_effect=run), patch("sys.argv", ["softcas.production", command, "--spec", str(spec), "--output", str(output)]), contextlib.redirect_stdout(io.StringIO()):
                    production.main()
                self.assertTrue((output / "result.json").is_file())

    def test_effective_cli_method_override_is_validated(self):
        config = load_config(cli.ROOT / "configs/fixture.json")
        config["method"] = "softcas"
        config["serial"]["fraction"] = 0
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "config.json")
            path.write_text(json.dumps(config), encoding="utf-8")
            with patch("sys.argv", ["softcas.cli", "check-config", "--config", str(path), "--method", "serial"]):
                with self.assertRaises(ValueError):
                    cli.main()


class NodeJoinRepairTests(unittest.TestCase):
    @staticmethod
    def node(root, client, federation=("0", "1"), row_seed=1001):
        root.mkdir()
        row = dict(seed=row_seed, client=client, stage=1, current_domain="fog", domain="fog", score=0.5, deployed="state")
        (root / "test_matrix.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        (root / "all_transfers.json").write_text("[]", encoding="utf-8")
        value = dict(node_complete=True, cell=CELLS[0], seed=1001, method="softcas", ablation="none",
                     purpose="FORMAL", metric="AP50:95", federation_clients=list(federation), local_clients=[client],
                     output_root=str(root), max_client_update_seconds=1, computation_accounting_complete=True,
                     F_known_lower_bound=1, selection_E=1, network_accounting_complete=True,
                     physical_training={}, active_external_paths=0, deployed_tensor_bytes=1, deployed_bytes=1,
                     peak_process_ram_bytes=1, p95_ms=1)
        path = root / "node.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_node_federation_disagreement_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            left = self.node(root / "left", "0")
            right = self.node(root / "right", "1", federation=("1", "2"))
            with self.assertRaisesRegex(ValueError, "same nonempty federation"):
                join_physical_nodes([left, right], root / "joined")

    def test_node_matrix_cannot_silently_use_another_seed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            nodes = [self.node(root / "left", "0", row_seed=1002), self.node(root / "right", "1", row_seed=1002)]
            with self.assertRaisesRegex(ValueError, "declared seed"):
                join_physical_nodes(nodes, root / "joined")

    def test_valid_nodes_form_one_equal_client_seed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            nodes = [self.node(root / "left", "0"), self.node(root / "right", "1")]
            result = join_physical_nodes(nodes, root / "joined")
            self.assertTrue(result["complete"])
            self.assertEqual(result["seed"], 1001)
            self.assertAlmostEqual(result["seen"], 0.5)
            self.assertEqual(result["F_total"], 2)


class MatrixRepairTests(unittest.TestCase):
    def spec(self):
        return dict(datasets={"ControlledFog":dict(registries={"1001":"registry.json"}, contact_trace="trace.json")},
                    detectors={d:dict(checkpoint=d+".pt", config="detector.json") for d in ("YOLOv8n","YOLO11n","YOLO26n")},
                    seeds=[1001], clients=["0","1","2","3"], primary_detector="YOLOv8n", output_root="not_executed",
                    algorithm_config="config.json", formal=False,
                    serial_selection={"ControlledFog/"+d:dict(rule=dict(order=["configuration","collaboration"],fraction=0.5,attempt_fraction=0.5)) for d in ("YOLOv8n","YOLO11n","YOLO26n")})

    def test_ablation_profiles_are_explicit_and_legacy_default_is_preserved(self):
        spec=self.spec()
        legacy=build_plan(spec)["jobs"]
        self.assertEqual(len([j for j in legacy if j["ablation"]!="none"]),6)
        spec["ablation_detectors"]=list(spec["detectors"])
        expanded=build_plan(spec)["jobs"]
        self.assertEqual(len([j for j in expanded if j["ablation"]!="none"]),18)
        self.assertEqual({j["detector"] for j in expanded if j["ablation"]!="none"},set(spec["detectors"]))

    def test_duplicate_seeds_and_unknown_ablation_detector_do_not_silently_expand_plan(self):
        for fields in [dict(seeds=[1001,1001]),dict(clients=["0",0]),dict(ablation_detectors=["unknown"])]:
            with self.subTest(fields=fields):
                with self.assertRaises(ValueError):build_plan({**self.spec(),**fields})

    def test_recorded_scope_uses_runtime_family_and_peer_topology(self):
        self.assertEqual(execution_scope(dict(external_peers={"1":{}}),dict(family="jetson_orin")),
                         "distributed peer node on jetson_orin; target audit/test on identified deployment endpoint")
        self.assertIn("single-host logical federation on rtx5080",execution_scope({},dict(family="rtx5080")))

    def physical_spec(self):
        return {**self.spec(), "epoch_start_unix":1900000000, "trajectory_spacing_seconds":600,
                "physical_nodes":{str(i):dict(endpoint=dict(host="127.0.0.1",port=18760+i),
                    config="configs/jetson_updates.json",overrides=dict(deployment=dict(default_endpoint=dict(host="127.0.0.1",port=18770+i)))) for i in range(4)}}

    def test_physical_plan_expands_to_four_single_node_jobs_with_shared_epochs(self):
        plan=build_plan(self.physical_spec())
        self.assertEqual(plan["execution_topology"],"physical_nodes")
        self.assertEqual(len(plan["jobs"]),4*len(plan["trajectory_groups"]))
        group=plan["trajectory_groups"][0]
        jobs=[j for j in plan["jobs"] if j["trajectory_group"]==group["trajectory"]]
        self.assertEqual({j["epoch_start_unix"] for j in jobs},{1900000000})
        self.assertEqual({j["clients"][0] for j in jobs},{"0","1","2","3"})
        self.assertEqual(plan["trajectory_groups"][1]["epoch_start_unix"],1900000600)
        for job in jobs:
            self.assertEqual(job["clients"],[job["node_id"]])
            self.assertEqual(set(job["external_peers"]),set(job["federation_clients"])-set(job["clients"]))
            self.assertTrue(job["output"].endswith("node"+job["node_id"]))
            self.assertEqual(job["overrides"]["transport"]["base_port"],18760+int(job["node_id"]))
        self.assertEqual(len(group["node_results"]),4)

    def test_physical_plan_never_guesses_unbound_endpoint_or_schedule(self):
        for fields in [dict(epoch_start_unix=None),dict(trajectory_spacing_seconds=None),dict(physical_nodes={})]:
            with self.subTest(fields=fields):
                with self.assertRaises(ValueError):build_plan({**self.physical_spec(),**fields})
        spec=self.physical_spec();spec["physical_nodes"]["0"]["endpoint"]["host"]="SET_ORIN_0"
        with self.assertRaisesRegex(ValueError,"actual host"):
            build_plan(spec)

    def test_physical_execution_requires_node_id_and_runs_only_that_peers_jobs(self):
        plan=build_plan(self.physical_spec())
        with patch("softcas.experiment.execute_trajectory",side_effect=lambda job,resume:job["node_id"]) as run:
            with self.assertRaisesRegex(ValueError,"--node-id"):
                execute_plan(plan)
            self.assertEqual(run.call_count,0)
            values=execute_plan(plan,node_id="2")
            self.assertEqual(values,["2"]*len(plan["trajectory_groups"]))
            self.assertEqual(run.call_count,len(plan["trajectory_groups"]))

    def test_cli_passes_node_id_to_physical_plan_executor(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder,"plan.json");path.write_text(json.dumps(build_plan(self.physical_spec())),encoding="utf-8")
            with patch("softcas.experiment.execute_plan",return_value=[]) as execute, patch("sys.argv",["softcas.production","run-plan","--spec",str(path),"--node-id","1"]),contextlib.redirect_stdout(io.StringIO()):
                production.main()
            self.assertEqual(execute.call_args.kwargs,{"node_id":"1"})


class DiagnosticRepairTests(unittest.TestCase):
    def test_validation_tie_chooses_z0_before_any_test_readout(self):
        backend = Backend()
        cache = PackageCache(1000000, backend.specs)
        package = Package("fixture_peer", "v1", 0, 0, backend.specs[2], 1/8,
                          freeze_tensors(Adapter(8, 1/8).state_dict()), "UNIT_TEST_ONLY")
        algorithm = load_config(cli.ROOT / "configs/fixture.json")
        splits = [synthetic_split(role, "0", 1, "a", 11+i) for i, role in enumerate(("D", "V", "T"))]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);peer = root / "peer.json";peer.write_bytes(package.wire)
            spec = dict(structures=[dict(mask=[1,0,0,0,0,0], ratio=1/8), dict(mask=[0,1,0,0,0,0], ratio=1/8)],
                        development_record_sha256="fixture", peer_packages=[str(peer)], receiving_coefficients={package.hash:0.2},
                        snapshot_time=1, h=dict(lr=1e-3, residual=1e-3), initialization_seed=1001,
                        fit_steps=0, budget=dict(F=1e20,E=0,B=0,W=100))
            output = root / "diagnostic"
            readouts=[]
            def score(model, split):
                if split.role == "T":
                    self.assertEqual(json.loads((output / "V_choices.json").read_bytes()), {"empty":0,"peer":0})
                readouts.append(split.role)
                return 0.5
            # Reverse the state-hash order to ensure the registered z0 rule,
            # rather than a serialization artifact, determines an exact tie.
            hash_property=property(lambda state: ("f" if json.loads(state.nu).get("structure")==0 else "0")*64)
            with patch.object(backend,"score",side_effect=score), patch.object(State,"hash",hash_property), patch("softcas.diagnostics.work",side_effect=lambda *a,**k:contextlib.nullcontext()):
                result=configuration_composition(backend,cache,*splits,spec,algorithm,output)
            self.assertEqual(result["V_choices"], {"empty":0,"peer":0})
            self.assertEqual(readouts, ["V"]*4+["T"]*4)
            self.assertEqual(result["purpose"], "SYNTHETIC_BEHAVIOR_ONLY")


if __name__ == "__main__":
    unittest.main()
