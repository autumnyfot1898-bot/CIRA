"""Real-detector federation and experiment matrix entrypoints.

Windows RTX execution emulates logical clients; Orin qualification is physical.
Those two scopes are recorded separately rather than relabeling RTX training as
measured on-device adaptation. All orchestration starts only from explicit CLI.
"""
from __future__ import annotations
import copy
import hashlib
import json
import os
import math
import re
import threading
import time
from pathlib import Path
from dataclasses import asdict
from .backend import YoloBackend, work
from .cache import PackageCache
from .config import load_config, effective_formal, merge_dict, METHODS, ABLATIONS
from .controller import FeatureEncoder
from .data import StageData, load_detection_split, validate_registry
from .device import RemoteReleaseService, DeploymentOutcomeUnknown
from .engine import initial_state, run_stage, export_stage
from .evaluation import EvaluationCache
from .journal import Journal, DurableRecords
from .metering import CostProfile
from .reporting import export_matrix, write_csv
from .resources import Cost, Ledger
from .runtime import MemoryGuard
from .state import Structure, FitSettings, State, canonical, digest
from .storage import BoundedStore, CheckpointStore, atomic_write
from .transport import PeerService, PeerServer, LiveExchange, TcpClient


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def stage_events(registry, clients):
    contexts = {}
    for r in registry["records"]:
        if r["role"] in ("D", "V", "A"):
            key = (str(r["client"]), r["stage"])
            if key in contexts and contexts[key] != r["domain"]:
                raise ValueError("A supplied stage has conflicting domain boundaries")
            contexts[key] = r["domain"]
    return [{"client": c, "stage": t, "domain": contexts[(c, t)]}
            for t in sorted({t for c, t in contexts}) for c in clients if (c, t) in contexts]


class ContactTrace:
    def __init__(self, events, start):
        self.events = sorted(events, key=lambda e: (e["offset_seconds"], e["receiver"], e["owner"]))
        self.start = start

    def gate(self, receiver):
        def visible(owner, now):
            online = False
            elapsed = now - self.start
            for e in self.events:
                if e["offset_seconds"] > elapsed:
                    break
                if str(e["receiver"]) == str(receiver) and str(e["owner"]) == str(owner):
                    online = bool(e["online"])
            return online
        return visible


def trajectory_config(job):
    """Resolve overrides and the effective formal context before any execution."""
    overrides = merge_dict(job.get("overrides", {}), {
        "method": job["method"], "ablation": job.get("ablation", "none"),
        "seed": job["seed"], "runtime": {"seed": job["seed"]}})
    config = load_config(job["config"], formal=job.get("formal", False), overrides=overrides)
    config["randomization"] = {**config.get("randomization", {}),
                               "dataset": job.get("dataset", job.get("cell", "unspecified")),
                               "detector": job.get("detector", "unspecified")}
    return config, effective_formal(config, job.get("formal", False))


def execute_trajectory(job, resume=False):
    job = copy.deepcopy(job)
    if "study_trajectory_id" in job:
        from .study_execution import preflight_job
        preflight_job(job)
    config, formal = trajectory_config(job)
    job["formal"] = formal
    root = Path(job["output"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    identity = digest({"job": {k: v for k, v in job.items() if k != "output"}, "configuration": config,
                       "inputs": {key: hashlib.sha256(Path(job[key]).read_bytes()).hexdigest()
                                  for key in ("checkpoint", "detector_config", "registry", "contact_trace")},
                       "implementation": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}})
    progress_file = root / "progress.json"
    if progress_file.exists() and not resume:
        raise FileExistsError("Trajectory output already exists; use explicit resume or another output directory")
    if not progress_file.exists() and any(root.iterdir()):
        raise FileExistsError("Unidentified existing output must not be mixed with a new trajectory")
    progress = read_json(progress_file) if progress_file.exists() else {"job": identity, "completed": [], "states": {}, "status": "new"}
    if progress["job"] != identity:
        raise ValueError("Resume job/data/method identity differs")
    registry = read_json(job["registry"])
    validate_registry(registry["records"], formal=formal)
    detector = read_json(job["detector_config"])
    if tuple(detector["labels"]) != tuple(registry["labels"]) or len(detector["labels"]) != 8:
        raise ValueError("Eight-class fixed-vocabulary detector and registry required")
    backend = YoloBackend(job["checkpoint"], detector["ports"], detector["labels"], detector["preprocessing"],
                          detector["image_size"], config["runtime"])
    if config.get("cost_profile"):
        backend.cost_profile = CostProfile(config["cost_profile"], backend.runtime.identity(), formal)
        profile = backend.cost_profile.data
        if (profile.get("base_sha256") != backend.identity or profile.get("execution") != backend.execution or
                canonical(profile.get("port_interfaces")) != canonical([asdict(p) for p in backend.specs])):
            raise ValueError("Cost calibration must use this exact frozen base and declared six ports")
    backend.memory_guard = MemoryGuard(config["working_memory"]["process_bytes"],
                                       config["working_memory"]["minimum_free_bytes"], config["working_memory"].get("cuda_bytes"))
    clients = tuple(str(c) for c in job["clients"])
    if len(set(clients)) != len(clients):
        raise ValueError("Duplicate logical clients")
    events = job.get("stage_events") or stage_events(registry, clients)
    for client in clients:
        sequence = [e["stage"] for e in events if str(e["client"]) == client]
        if not sequence or sequence != list(range(1, len(sequence) + 1)):
            raise ValueError("Supplied asynchronous event order skips/reverses a client stage")
    if job.get("external_peers") and not isinstance(job.get("epoch_start_unix"), (int, float)):
        raise ValueError("Distributed physical clients require an explicit shared contact-trace epoch")
    origin = job.get("epoch_start_unix")
    if origin is None:
        origin = time.time()
    trace = ContactTrace(read_json(job["contact_trace"]), origin)
    serving, services, servers, caches, stores, states, evaluators = {}, {}, [], {}, {}, {}, {}
    all_transfers, matrix, stage_reports = [], [], []
    offline = Ledger(Cost(**config["offline_budget"]), Cost(**config["offline_budget"]))
    offline.events = DurableRecords(root / "offline_ledger.jsonl")
    for event in offline.events:
        offline.used[event["scope"]] += Cost(**event["cost"])
    offline.runtime, offline.meter_policy = backend.runtime, config["meter"]
    try:
        for index, client in enumerate(clients):
            directory = root / ("client" + client)
            source = Ledger(Cost(**config["serving_budget"]), Cost())
            source.events = DurableRecords(directory / "source_ledger.jsonl")
            for event in source.events:
                source.used[event["scope"]] += Cost(**event["cost"])
            serving[client] = source
            package_store = BoundedStore(directory / "cache", config["storage"]["cache_bytes"], config["storage"]["transient_bytes"])
            caches[client] = PackageCache(config["cache_bytes"], backend.specs, package_store)
            stores[client] = CheckpointStore(BoundedStore(directory / "retained", config["storage"]["checkpoint_bytes"],
                                                       config["storage"]["transient_bytes"]))
            fixed = config["fixed"]
            states[client] = State.load(progress["states"][client]) if client in progress["states"] else initial_state(
                backend, caches[client], Structure(tuple(fixed["mask"]), fixed["ratio"]), FitSettings(fixed["lr"], fixed["residual"]), job["seed"])
            caches[client].pin("deployed", states[client].C)
            stores[client].pin("deployed", [states[client]])
            evaluators[client] = EvaluationCache()
            service = PeerService(client, source, config["transport"]["token_env"], directory / "publications",
                                  config["transport"]["source_payload_quota"])
            service.records = DurableRecords(directory / "source_transfers.jsonl")
            service.restore_quota()
            services[client] = service
            address = (config["transport"]["bind"], config["transport"]["base_port"] + index)
            server = PeerServer(address, service, config["transport"]["timeout_seconds"])
            servers.append(server)
            threading.Thread(target=server.serve_forever, daemon=True).start()
        for event in events:
            client, stage = str(event["client"]), event["stage"]
            marker = client + "/" + str(stage)
            directory = root / ("client" + client) / ("stage" + str(stage))
            if marker in progress["completed"]:
                stage_reports.append(read_json(directory / "stage.json"))
                if (directory / "independent_test.json").exists():
                    matrix.extend(read_json(directory / "independent_test.json"))
                continue
            if (directory / "journal.jsonl").exists():
                raise RuntimeError("Interrupted stage journal exists. Reconcile its locked audit/target status before continuation; automatic re-search is disabled")
            if "offset_seconds" in event or job.get("external_peers"):
                remaining = origin + event.get("offset_seconds", 0) - time.time()
                if remaining > 0:
                    time.sleep(remaining)
            ledger = Ledger(Cost(**config["budgets"]["search"]), Cost(**config["budgets"]["release"]))
            ledger.events = DurableRecords(directory / "ledger_live.jsonl")
            ledger.runtime, ledger.meter_policy = backend.runtime, config["meter"]
            ledger.begin("search")
            cache = caches[client]
            cache.store.ledger = ledger
            backend.state_store = stores[client]
            backend.state_store.store.ledger = ledger
            backend.journal = Journal(directory / "journal.jsonl")
            data = StageData(*(load_detection_split(job["registry"], client, stage, role, detector["image_size"], detector["labels"],
                                 formal=formal, batch_size=config["fit_batch_size"] if role == "D" else config["evaluation_batch_size"])
                               for role in ("D", "V", "A")))
            peers = {other: {"host": config["transport"]["connect_host"], "port": config["transport"]["base_port"] + i}
                     for i, other in enumerate(clients) if other != client}
            peers.update({str(owner): address for owner, address in job.get("external_peers", {}).items() if str(owner) != client})
            exchange = LiveExchange(client, cache, ledger, peers, config["transport"]["token_env"], trace.gate(client),
                                    config["transport"]["timeout_seconds"])
            exchange.records = DurableRecords(directory / "controller_transfers.jsonl")
            endpoint = config["deployment"]["endpoints"].get(client) or config["deployment"]["default_endpoint"]
            release = RemoteReleaseService(endpoint, job["registry_id"], client, formal, identity[:20] + "-" + client)
            release.records = DurableRecords(directory / "target_transfers.jsonl")
            backend.release_service = release
            release.ensure_incumbent(backend, states[client], cache, ledger, config)
            retained = []
            for state_hash in sorted(stores[client].index):
                try:
                    saved = stores[client].load(state_hash)
                    if all(k in cache for k in saved.C):
                        retained.append(saved)
                except (OSError, KeyError):
                    continue
            progress.update(status="active", active=marker)
            atomic_write(progress_file, canonical(progress))
            result = run_stage(backend, states[client], data, cache, exchange, ledger, copy.deepcopy(config),
                               retained=retained, evaluator=evaluators[client], owner=client)
            states[client] = result.deployed
            stores[client].pin("deployed", [result.deployed])
            for package in result.published:
                services[client].publish(package)
            export_stage(result, directory, backend, data, config, evaluators[client], cache, exchange)
            stage_reports.append(read_json(directory / "stage.json"))
            # Independent T goes only to the reporting evaluator, after deployment.
            seen = []
            for previous in events:
                if str(previous["client"]) == client and previous["stage"] <= stage and previous["domain"] not in seen:
                    seen.append(previous["domain"])
            stage_matrix = []
            for domain in seen:
                split = load_detection_split(job["registry"], client, stage, "T", detector["image_size"], detector["labels"],
                                              formal=formal, batch_size=config["evaluation_batch_size"], domain=domain)
                score, report = release.evaluate(backend, result.deployed, split, cache, offline, config, "T")
                row = {"seed": job["seed"], "client": client, "stage": stage, "current_domain": event["domain"],
                       "domain": domain, "deployed": result.deployed.hash, "score": score, "test_manifest": split.identity,
                       "metric": "AP50:95", "purpose": config["purpose"], "target_executor": report["executor"],
                       "target_runtime": report["runtime"], "first_stage": min(e["stage"] for e in events if str(e["client"]) == client and e["domain"] == domain)}
                stage_matrix.append(row)
            matrix.extend(stage_matrix)
            atomic_write(directory / "independent_test.json", canonical(stage_matrix))
            # A rejected champion's measurements must never describe the
            # incumbent that actually remains deployed. Measure that exact
            # state in another isolated worker, outside all online decisions.
            from .measurement import summarize_sessions
            measurements = []
            count = config.get("report_measurement_sessions", 1)
            for repeat in range(count):
                _, measurement = release.evaluate(backend, result.deployed, split, cache, offline, config,
                    "T", measure=True, measurement_repeat=repeat if count > 1 else None)
                measurements.append(measurement)
            atomic_write(directory / "deployed_device.json", canonical(summarize_sessions(measurements)))
            atomic_write(directory / "deployed_device_sessions.json", canonical(measurements))
            all_transfers.extend(exchange.records + release.records)
            progress["states"][client] = str(directory / "checkpoints" / (result.deployed.hash + ".json"))
            progress["completed"].append(marker)
            progress.update(status="stage_complete", active=None)
            atomic_write(progress_file, canonical(progress))
        endpoints = export_matrix(root, matrix, set(clients))
        # Network totals use sender Tx once per directed leg, including remote
        # agent replies and partial transmissions, excluding independent tests.
        local_tx = {}
        for file in root.glob("client*/stage*/*_transfers.jsonl"):
            all_transfers.extend(json.loads(line) for line in file.read_bytes().splitlines())
        for service in services.values():
            all_transfers.extend(service.records)
        accounting_complete = True
        requested = {r["request"] for r in all_transfers if "request" in r}
        endpoints_seen = set()
        for client in clients:
            endpoint = config["deployment"]["endpoints"].get(client) or config["deployment"]["default_endpoint"]
            address = (endpoint["host"], endpoint["port"])
            if address in endpoints_seen:
                continue
            endpoints_seen.add(address)
            try:
                rpc = TcpClient("reporting", *address, offline, endpoint["token_env"], config["remote_timeout_seconds"])
                cursor = 0
                while True:
                    page, _ = rpc.call("accounting", {"cursor": cursor})
                    all_transfers.extend(r for r in page["records"] if r.get("request") in requested)
                    cursor = page["cursor"]
                    if not page["more"]:
                        break
            except (OSError, RuntimeError):
                accounting_complete = False
        for record in all_transfers:
            if record.get("leg", "").endswith(":tx") and record.get("operation") not in ("test", "measure", "accounting"):
                local_tx[(record["node"], record["leg"])] = record["bytes"]
        final = max(endpoints, key=lambda r: r["stage"])
        total_F = sum(r["actual"][s]["F"] for r in stage_reports for s in ("search", "release"))
        total_F += sum(l.used["search"].F for l in serving.values())
        computation_complete = all(r.get("flop_accounting", {}).get("coverage_complete", False) for r in stage_reports)
        # The client association is recorded in the stage files' directory paths.
        client_times = {c: sum(read_json(root / ("client" + c) / ("stage" + str(e["stage"])) / "stage.json")["actual"][s]["W"]
                               for e in events if str(e["client"]) == c for s in ("search", "release")) for c in clients}
        final_devices = {c: read_json(root / ("client" + c) / ("stage" + str(max(e["stage"] for e in events if str(e["client"]) == c))) /
                                           "deployed_device.json") for c in clients}
        device_fields = ("active_external_paths", "deployed_tensor_bytes", "deployed_bytes", "peak_process_ram_bytes", "p95_ms")
        trajectory = {"purpose": config["purpose"], "formal": formal, "metric": "AP50:95", "method": job["method"], "ablation": job.get("ablation", "none"),
                      "cell": job["dataset"] + "/" + job["detector"], "seed": job["seed"],
                      **{key: final[key] for key in ("seen", "current", "forget", "bwt", "minimum_client_seen")},
                      "F_total": total_F if computation_complete else None, "F_known_lower_bound": total_F,
                      "computation_accounting_complete": computation_complete,
                      "selection_E": sum(r["actual"]["search"]["E"] for r in stage_reports),
                      "network_tx_bytes": sum(local_tx.values()) if accounting_complete else None,
                      "network_tx_bytes_observed": sum(local_tx.values()), "network_accounting_complete": accounting_complete,
                      "max_client_update_seconds": max(client_times.values()), "physical_training": backend.runtime.identity(),
                      "execution_scope": execution_scope(job, config["runtime"]),
                      **{k: max(d[k] for d in final_devices.values()) for k in device_fields},
                      "device_aggregation": "maximum over clients at each client's final deployed stage",
                      "deployed_devices": final_devices,
                      "offline_cost": asdict(offline.used["search"]),
                      "local_clients": list(clients), "federation_clients": job.get("federation_clients", list(clients)),
                      "output_root": str(root), "node_complete": True,
                      "complete": set(clients) == set(str(c) for c in job.get("federation_clients", clients))}
        if "study_trajectory_id" in job:
            trajectory.update(study_trajectory_id=job["study_trajectory_id"],
                              study_recipe=job["study_recipe"], input_manifest=job["input_manifest"])
        atomic_write(root / "trajectory.json", canonical(trajectory))
        atomic_write(root / "all_transfers.json", canonical(all_transfers))
        atomic_write(root / "source_ledgers.json", canonical({c: l.events for c, l in serving.items()}))
        progress.update(status="complete")
        atomic_write(progress_file, canonical(progress))
        return trajectory
    except BaseException as exc:
        if 'ledger' in locals():
            ledger.finish("search")
            ledger.finish("release")
        progress.update(status="deployment_unresolved" if isinstance(exc, DeploymentOutcomeUnknown) else "incomplete", error=str(exc))
        atomic_write(progress_file, canonical(progress))
        raise
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()


def execution_scope(job, runtime):
    family = runtime.get("family", "unspecified")
    topology = "distributed peer node" if job.get("external_peers") else "single-host logical federation"
    return f"{topology} on {family}; target audit/test on identified deployment endpoint"


def _physical_plan(jobs, spec):
    """Expand one trajectory into independently launched, explicitly timed nodes."""
    if not isinstance(spec["physical_nodes"], dict) or not spec["physical_nodes"]:
        raise ValueError("physical_nodes must map client IDs to explicit node configurations")
    nodes = {str(client): node for client, node in spec["physical_nodes"].items()}
    clients = [str(client) for client in spec["clients"]]
    if len(nodes) != len(spec["physical_nodes"]) or set(nodes) != set(clients):
        raise ValueError("physical_nodes must identify exactly the declared federation clients")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", client) for client in clients):
        raise ValueError("Physical client IDs must be safe single directory names")
    endpoints = {}
    for client, node in nodes.items():
        if not isinstance(node, dict) or not isinstance(node.get("endpoint"), dict):
            raise ValueError(f"Bind physical_nodes[{client}] to a node configuration with an endpoint")
        endpoint = node.get("endpoint", {})
        host, port = endpoint.get("host"), endpoint.get("port")
        if (not isinstance(host, str) or not host.strip() or host.startswith(("SET_", "REQUIRED")) or
                not isinstance(port, int) or isinstance(port, bool) or not 0 < port < 65536):
            raise ValueError(f"Bind physical_nodes[{client}].endpoint to an actual host and TCP port before planning")
        endpoints[client] = {"host": host, "port": port}
    if len({(e["host"], e["port"]) for e in endpoints.values()}) != len(endpoints):
        raise ValueError("Physical nodes require distinct peer endpoints")
    epochs = spec.get("trajectory_start_epochs")
    if epochs is not None:
        if not isinstance(epochs, dict) or set(epochs) != {job["name"] for job in jobs}:
            raise ValueError("trajectory_start_epochs must bind every trajectory name exactly once")
    else:
        start, spacing = spec.get("epoch_start_unix"), spec.get("trajectory_spacing_seconds")
        if (not isinstance(start, (int, float)) or isinstance(start, bool) or
                not math.isfinite(start) or start <= 0):
            raise ValueError("Bind epoch_start_unix explicitly (or provide trajectory_start_epochs); no start time is inferred")
        if len(jobs) > 1 and (not isinstance(spacing, (int, float)) or isinstance(spacing, bool) or
                              not math.isfinite(spacing) or spacing <= 0):
            raise ValueError("Multiple physical trajectories require explicit positive trajectory_spacing_seconds or per-trajectory start epochs")
        epochs = {job["name"]: start + index * (spacing or 0) for index, job in enumerate(jobs)}
    if any(not isinstance(epoch, (int, float)) or isinstance(epoch, bool) or not math.isfinite(epoch) or epoch <= 0
           for epoch in epochs.values()):
        raise ValueError("Every physical trajectory needs an explicit finite positive shared epoch")
    expanded, groups = [], []
    for job in jobs:
        node_results, node_names = [], []
        for client in clients:
            node = nodes[client]
            node_job = copy.deepcopy(job)
            node_job.update(name=job["name"] + "__node" + client,
                            trajectory_group=job["name"], node_id=client, clients=[client],
                            federation_clients=clients, epoch_start_unix=epochs[job["name"]],
                            external_peers={c: address for c, address in endpoints.items() if c != client},
                            output=str(Path(job["output"]) / ("node" + client)),
                            config=node.get("config", job["config"]))
            node_job["overrides"] = merge_dict(job["overrides"], node.get("overrides", {}))
            node_job["overrides"] = merge_dict(node_job["overrides"], {
                "transport": {"bind": node.get("bind", "0.0.0.0"), "base_port": endpoints[client]["port"]}})
            expanded.append(node_job)
            node_results.append(str(Path(node_job["output"]) / "trajectory.json"))
            node_names.append(node_job["name"])
        groups.append(dict(trajectory=job["name"], node_jobs=node_names,
                           epoch_start_unix=epochs[job["name"]], node_results=node_results,
                           joined_output=str(Path(job["output"]) / "joined")))
    return expanded, groups


def build_plan(spec):
    # Validate flag types even when planning with deliberately unbound paths.
    declared_formal = effective_formal({}, spec.get("formal", False))
    if not spec["datasets"] or not spec["detectors"] or not spec["seeds"] or not spec["clients"]:
        raise ValueError("Experiment matrix needs datasets, detectors, independent seeds and clients")
    if len(set(spec["seeds"])) != len(spec["seeds"]) or len({str(c) for c in spec["clients"]}) != len(spec["clients"]):
        raise ValueError("Duplicate seed/client identities would repeat a trajectory in the matrix")
    ablation_detectors = spec.get("ablation_detectors", [spec["primary_detector"]])
    if (len(set(ablation_detectors)) != len(ablation_detectors) or
            any(detector not in spec["detectors"] for detector in ablation_detectors)):
        raise ValueError("Ablation detectors must be unique members of the declared detector matrix")
    jobs = []
    for dataset, dataset_spec in spec["datasets"].items():
        for detector, detector_spec in spec["detectors"].items():
            variants = [(method, "none") for method in METHODS]
            if detector in ablation_detectors:
                variants += [("softcas", a) for a in ABLATIONS if a != "none"]
            for seed in spec["seeds"]:
                for method, ablation in variants:
                    name = f"{dataset}__{detector}__{method}__{ablation}__{seed}"
                    job = {"name": name, "dataset": dataset, "detector": detector, "method": method, "ablation": ablation,
                           "seed": seed, "checkpoint": detector_spec["checkpoint"], "detector_config": detector_spec["config"],
                           "registry": dataset_spec["registries"][str(seed)], "registry_id": dataset + "_" + str(seed),
                           "contact_trace": dataset_spec["contact_trace"], "clients": spec["clients"],
                           "config": spec["algorithm_config"], "formal": declared_formal,
                           "output": str(Path(spec["output_root"]) / name),
                           "overrides": copy.deepcopy(spec.get("cell_overrides", {}).get(dataset + "/" + detector, {}))}
                    if method == "serial":
                        selection = spec["serial_selection"][dataset + "/" + detector]
                        job["overrides"] = merge_dict(job["overrides"], {"serial": selection["rule"]})
                    jobs.append(job)
    plan = {"schema": 1, "jobs": jobs, "unit": "one seed includes all clients and stages", "spec_sha256": digest(spec),
            "execution_topology": "single_host_logical_clients"}
    if "physical_nodes" in spec:
        plan["jobs"], plan["trajectory_groups"] = _physical_plan(jobs, spec)
        plan.update(execution_topology="physical_nodes", node_ids=[str(c) for c in spec["clients"]],
                    unit="join all physical node results to form one independent seed trajectory")
    for job in plan["jobs"]:
        # A plan may refer to paths not mounted on the planning host. Execution
        # always resolves again; readable configs already enforce their purpose.
        if Path(job["config"]).is_file():
            _, job["formal"] = trajectory_config(job)
        if job["formal"] and job["method"] == "serial":
            selection = spec["serial_selection"][job["dataset"] + "/" + job["detector"]]
            if not selection.get("development_record_sha256"):
                raise ValueError("Serial choice must cite its independent development selection record")
    return plan


def execute_plan(plan, resume=False, node_id=None):
    if plan.get("execution_topology") == "physical_nodes":
        if node_id is None or str(node_id) not in plan.get("node_ids", []):
            raise ValueError("Physical plans require an explicit --node-id on each peer; nodes cannot be executed serially on one host")
        jobs = [job for job in plan["jobs"] if job.get("node_id") == str(node_id)]
        if not jobs:
            raise ValueError("Physical plan has no jobs for the selected node")
    else:
        if node_id is not None:
            raise ValueError("--node-id is only valid for a physical_nodes plan")
        jobs = plan["jobs"]
    values = []
    for job in jobs:
        # Completed stages are reused only after checking the full job/input/code
        # identity, just like an interrupted trajectory.
        values.append(execute_trajectory(job, resume))
    return values


def join_physical_nodes(node_files, output):
    """Aggregate one seed after all physical nodes finish; nodes are not seeds."""
    nodes = [read_json(file) for file in node_files]
    if not nodes or any(not n.get("node_complete") for n in nodes):
        raise ValueError("Every physical node must finish its local stages")
    contract = ("cell", "seed", "method", "ablation", "purpose", "metric")
    if any(any(n[k] != nodes[0][k] for k in contract) for n in nodes):
        raise ValueError("Cannot join different seeds, bases/cells, methods or interventions")
    if any("study_trajectory_id" in n for n in nodes):
        if any(n.get("study_trajectory_id") != nodes[0].get("study_trajectory_id") or
               n.get("study_recipe") != nodes[0].get("study_recipe") for n in nodes):
            raise ValueError("Cannot join different study recipes/conditions or unidentified nodes")
        if digest(nodes[0]["study_recipe"]) != nodes[0]["study_trajectory_id"]:
            raise ValueError("Corrupted study recipe identity in node result")
    expected = set(str(c) for c in nodes[0]["federation_clients"])
    if not expected or any(set(str(c) for c in n["federation_clients"]) != expected for n in nodes):
        raise ValueError("Every physical node must declare the same nonempty federation")
    clients = [str(c) for n in nodes for c in n["local_clients"]]
    if set(clients) != expected or len(clients) != len(set(clients)):
        raise ValueError("Distributed federation has missing/duplicate clients")
    rows, tx, times = [], {}, []
    for node in nodes:
        root = Path(node["output_root"])
        node_rows = [json.loads(line) for line in (root / "test_matrix.jsonl").read_bytes().splitlines()]
        local_clients = {str(c) for c in node["local_clients"]}
        if not node_rows or any(r["seed"] != node["seed"] or str(r["client"]) not in local_clients for r in node_rows):
            raise ValueError("Node test matrix must match its declared seed and local clients")
        rows.extend({**r, "client": str(r["client"])} for r in node_rows)
        for record in read_json(root / "all_transfers.json"):
            if record.get("leg", "").endswith(":tx") and record.get("operation") not in ("test", "measure", "accounting"):
                tx[(record["node"], record["leg"])] = record["bytes"]
        times.append(node["max_client_update_seconds"])
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Join output must be new or empty; previous results are preserved")
    endpoints = export_matrix(output, rows, expected)
    final = max(endpoints, key=lambda r: r["stage"])
    complete_F = all(n["computation_accounting_complete"] for n in nodes)
    known_F = sum(n["F_known_lower_bound"] for n in nodes)
    value = {**{k: nodes[0][k] for k in contract},
             **{k: final[k] for k in ("seen", "current", "forget", "bwt", "minimum_client_seen")},
             "F_total": known_F if complete_F else None, "F_known_lower_bound": known_F,
             "computation_accounting_complete": complete_F,
             "selection_E": sum(n["selection_E"] for n in nodes), "network_tx_bytes": sum(tx.values()),
             "network_accounting_complete": all(n["network_accounting_complete"] for n in nodes),
             "max_client_update_seconds": max(times), "physical_training": [n["physical_training"] for n in nodes],
             **{k: max(n[k] for n in nodes) for k in ("active_external_paths", "deployed_tensor_bytes", "deployed_bytes", "peak_process_ram_bytes", "p95_ms")},
             "device_aggregation": "maximum over final deployed clients across nodes",
             "complete": True, "clients": sorted(expected), "node_files": [str(p) for p in node_files]}
    if not value["network_accounting_complete"]:
        value["network_tx_bytes_observed"] = value["network_tx_bytes"]
        value["network_tx_bytes"] = None
    if "study_trajectory_id" in nodes[0]:
        value.update(study_trajectory_id=nodes[0]["study_trajectory_id"], study_recipe=nodes[0]["study_recipe"],
                     input_manifests=[n.get("input_manifest", {}) for n in nodes])
    atomic_write(output / "trajectory.json", canonical(value))
    return value
