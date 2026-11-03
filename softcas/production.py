"""Explicit RTX/Orin command line. Importing this module starts no work."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .state import canonical
from .storage import atomic_write


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(prog="python -m softcas.production")
    parser.add_argument("command", choices=("doctor", "prepare-data", "partition", "relocate-registry", "prepare-base",
                                            "plan", "study-plan", "study-bindings", "study-bind", "development-lock", "external-source", "external-bind", "study-compile", "study-run", "study-report", "conditions", "rgb-readout", "external-check", "external-run", "decision-audit", "mechanism", "run", "run-plan", "join-nodes", "serve-agent", "calibrate", "register-trt-cost", "diagnostic", "report", "predict", "target-status"))
    parser.add_argument("--spec", required=True, help="Explicit JSON input; no inference/training starts without a command")
    parser.add_argument("--output")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--node-id", help="For physical run-plan: execute only this peer's explicitly scheduled jobs")
    args = parser.parse_args()
    if args.node_id is not None and args.command not in ("run-plan", "study-run"):
        parser.error("--node-id is only valid with run-plan or study-run")
    spec = load(args.spec)
    if args.command == "doctor":
        from .runtime import RuntimeSpec, process_memory
        runtime = RuntimeSpec(**spec["runtime"])
        runtime.validate()
        result = {"runtime": runtime.identity(), "memory": process_memory()}
    elif args.command == "prepare-data":
        from .prepare import convert_coco, convert_cityscapes_polygons
        if spec.get("format", "coco") == "cityscapes_polygons":
            result = convert_cityscapes_polygons({**spec, "output": args.output or spec["output"]})
        else:
            result = convert_coco(spec["annotation_files"], spec["image_roots"], spec["scene_map"], args.output or spec["output"], spec.get("aliases"))
    elif args.command == "partition":
        from .prepare import assign_clients
        result = assign_clients(spec["registry"], args.output or spec["output"], spec["domains"], spec["seed"],
                                spec["alpha"], spec["max_passes"], spec.get("clients", 4), spec.get("equal", False))
    elif args.command == "relocate-registry":
        from .prepare import relocate_registry
        result = relocate_registry(spec["registry"], args.output or spec["output"], spec["path_map"])
    elif args.command == "prepare-base":
        from .prepare import prepare_base
        result = prepare_base(spec)
    elif args.command == "plan":
        from .experiment import build_plan
        result = build_plan(spec)
    elif args.command == "study-plan":
        from .study import build_study
        if spec.get('development_lock'):
            from .bindings import native_study
            result=native_study(spec)
        else:
            result = build_study(spec)
    elif args.command == "study-bindings":
        from .study_execution import binding_template
        result = binding_template(load(spec["study_registry"]), spec.get("selection"))
    elif args.command == 'study-bind':
        from .bindings import bind_study, write_binding_outputs
        result = bind_study(load(spec['study_registry']),spec)
        if spec.get('binding_output_directory'):
            result=write_binding_outputs(result,spec['binding_output_directory'])
    elif args.command == 'development-lock':
        from .bindings import lock_development
        result = lock_development(spec)
    elif args.command == 'external-bind':
        from .bindings import bind_external
        result = bind_external(spec)
    elif args.command == 'external-source':
        from .bindings import external_source_snapshot
        result = external_source_snapshot(spec)
    elif args.command == "study-compile":
        from .study_execution import compile_study
        result = compile_study(load(spec["study_registry"]), load(spec["bindings"]))
    elif args.command == "study-run":
        from .study_execution import execute_study
        result = execute_study(spec, args.resume, node_id=args.node_id)
    elif args.command == "conditions":
        from .conditions import build_condition
        result = build_condition(spec, args.output or spec["output"])
    elif args.command == "rgb-readout":
        from .rgb_evaluation import evaluate_rgb
        result = evaluate_rgb({**spec, "output": args.output or spec["output"]})
    elif args.command == "external-check":
        from .external import validate_adapter
        if isinstance(spec['adapter'],str):
            value=load(spec['adapter']);spec['adapter']=value.get('adapter',value)
        result = validate_adapter(spec["adapter"], spec)
    elif args.command == "external-run":
        from .external import execute_external
        if isinstance(spec['adapter'],str):
            value=load(spec['adapter']);spec['adapter']=value.get('adapter',value)
        result = execute_external(spec, resume=args.resume)
    elif args.command == "study-report":
        from .study import summarize_completed, primary_contrasts
        study = load(spec["study_registry"])
        records = [load(path) for path in spec["trajectory_files"]]
        result = {"summary": summarize_completed(study, records, spec.get("metric", "seen")),
                  "primary": primary_contrasts(study, records, spec.get("metric", "seen"))}
    elif args.command == "decision-audit":
        from .decision_audit import audit_files
        result = audit_files(spec["action_files"])
    elif args.command == "mechanism":
        from .mechanism import run_mechanism_job
        result = run_mechanism_job({**spec, "output": args.output or spec["output"]})
    elif args.command == "run":
        from .experiment import execute_trajectory
        result = execute_trajectory(spec, args.resume)
    elif args.command == "run-plan":
        from .experiment import execute_plan
        result = execute_plan(spec, args.resume, node_id=args.node_id)
    elif args.command == "join-nodes":
        from .experiment import join_physical_nodes
        result = join_physical_nodes(spec["node_results"], args.output or spec["output"])
    elif args.command == "target-status":
        from .resources import Ledger, Cost
        from .transport import TcpClient
        ledger = Ledger(Cost(**spec["budget"]), Cost())
        rpc = TcpClient("reconciliation", spec["host"], spec["port"], ledger, spec["token_env"], spec["timeout_seconds"])
        result, _ = rpc.call("status", {"deployment_id": spec["deployment_id"]})
        result["query_cost"] = __import__("dataclasses").asdict(ledger.used["search"])
    elif args.command == "serve-agent":
        from .device import AgentJobs
        from .resources import Ledger, Cost
        from .transport import PeerService, PeerServer
        from .journal import DurableRecords
        from .runtime import RuntimeSpec, process_memory
        RuntimeSpec(**spec["runtime"]).validate()
        if spec.get("worker_memory_bytes") is None:
            spec["worker_memory_bytes"] = int(process_memory()["system_total_bytes"] * spec["operational_memory_fraction"])
        ledger = Ledger(Cost(**spec["serving_budget"]), Cost())
        ledger.events = DurableRecords(Path(spec["job_store"]) / "agent_ledger.jsonl")
        for event in ledger.events:
            ledger.used[event["scope"]] += Cost(**event["cost"])
        service = PeerService(spec["device_id"], ledger, spec["token_env"], spec["publication_store"],
                              spec["source_payload_quota"], spec["maximum_bundle_bytes"], AgentJobs(spec))
        service.records = DurableRecords(Path(spec["job_store"]) / "transport.jsonl")
        service.restore_quota()
        with PeerServer((spec["bind"], spec["port"]), service, spec["job_timeout_seconds"] * 3) as server:
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
        result = {"stopped": True, "device_id": spec["device_id"]}
    elif args.command == "calibrate":
        from .calibration import calibrate_job
        result = calibrate_job(spec, args.output or spec["output"])
    elif args.command == "register-trt-cost":
        from .runtime import RuntimeSpec
        from .tensorrt_backend import register_plan_cost
        result = register_plan_cost(spec["plan"], RuntimeSpec(**spec["runtime"]), load(spec["reviewed_layers"]))
    elif args.command == "diagnostic":
        from .diagnostics import run_diagnostic_job
        result = run_diagnostic_job({**spec, "output": args.output or spec["output"]})
    elif args.command == "report":
        from .tables import export_tables
        from .reporting import write_csv
        records = [load(path) for path in spec["trajectory_files"]]
        if any(not r.get("complete", False) for r in records):
            raise ValueError("Incomplete seed trajectories must remain missing, not be filled with zero")
        output = Path(args.output or spec["output"])
        main = [r for r in records if r.get("ablation", "none") == "none"]
        export_tables(main, output)
        interventions = {"nas_only": "fix h", "hpo_only": "fix z", "no_handoff": "restrict parent families", "no_refine": "remove Refine",
                         "discrete": "one-hot receiving with no continuous beta fitting", "no_gain": "set predicted mean only to zero"}
        rows = [{**r, "variant": r["ablation"], "only_changed_rule": interventions[r["ablation"]]} for r in records if r.get("ablation", "none") != "none"]
        write_csv(output / "T6_ablation.csv", rows, ("variant", "only_changed_rule", "cell", "seed", "seen", "F_total"))
        result = {"main_trajectories": len(main), "ablation_trajectories": len(rows), "output": str(output)}
    else:
        from .inference import predict_deployed
        result = predict_deployed(spec)
    if args.output and args.command not in ("prepare-data", "partition", "relocate-registry", "report", "join-nodes", "diagnostic", "mechanism", "conditions", "rgb-readout"):
        atomic_write(args.output, canonical(result))
    # Large data/results are saved, not dumped into the console.
    print(json.dumps({"command": args.command, "output": args.output or spec.get("output"),
                      "status": "complete", "executed_by_explicit_cli": True}))


if __name__ == "__main__":
    main()
