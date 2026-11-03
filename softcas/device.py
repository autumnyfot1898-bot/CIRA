"""Isolated Orin qualification, target audit, atomic release and reconciliation."""
from __future__ import annotations
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
import numpy as np
import torch
from .resources import Cost, Ledger, ExecutionStopped
from .state import State, Package, canonical, digest
from .storage import atomic_write, bundle_bytes, unpack_bundle
from .runtime import RuntimeSpec, MemoryGuard, process_memory


class DeploymentOutcomeUnknown(Exception):
    """Never convert an unacknowledged commit into a false incumbent assertion."""


class WorkerFailure(RuntimeError):
    def __init__(self, message, partial_cost, cost_complete=False, remote_job_id=None):
        super().__init__(message)
        self.partial_cost, self.cost_complete = partial_cost, cost_complete
        self.remote_job_id = remote_job_id


def target_code_identity():
    directory = Path(__file__).parent
    return digest({name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                   for name in ("model.py", "backend.py", "evaluation.py", "postprocess.py", "tensorrt_backend.py", "device.py", "runtime.py", "data.py")})


def qualify_local(model, example, runtime, repeats=100, warmup=20, predict=None):
    if repeats < 2 or warmup < 0:
        raise ValueError("Declare positive measurement repeats and warmup")
    model.eval()
    predictor = predict or model
    stop, samples = threading.Event(), []
    def sample():
        while not stop.is_set():
            samples.append(process_memory())
            stop.wait(0.01)
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        if torch.device(runtime.device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(runtime.device)
        times = []
        with torch.no_grad():
            for _ in range(warmup):
                with runtime.autocast():
                    predictor(example)
            runtime.synchronize()
            for _ in range(repeats):
                runtime.synchronize()
                start = time.perf_counter()
                with runtime.autocast():
                    predictor(example)
                runtime.synchronize()
                times.append((time.perf_counter() - start) * 1000)
    finally:
        stop.set()
        sampler.join()
    memory = process_memory()
    peak = max([memory["peak_rss_bytes"], memory["rss_bytes"]] + [r["rss_bytes"] for r in samples])
    report = {"deployed_tensor_bytes": sum(v.numel() * v.element_size() for v in model.state_dict().values()),
              "peak_process_ram_bytes": peak, "p95_ms": float(np.percentile(times, 95)),
              "peak_system_used_bytes": max([memory["system_used_bytes"]] + [r["system_used_bytes"] for r in samples]),
              "p50_ms": float(np.percentile(times, 50)), "maximum_ms": max(times),
              "latency_samples_ms": times, "warmup": warmup, "repeats": repeats,
              "batch_size": example.shape[0],
              "ram_samples": len(samples), "ram_scope": "isolated worker RSS peak; CUDA is reported separately, not added on unified memory",
              "active_external_paths": sum(int(p.discrete_choice in p.external) if p.is_discrete else len(p.external) for p in model.ports.values()),
              "runtime": runtime.identity(), "device": platform.platform(), "execution": "native_composition",
              "latency_scope": "preprocessed tensor to detector output; synchronized CUDA; caller may provide postprocessing"}
    if torch.device(runtime.device).type == "cuda":
        report.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(runtime.device),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(runtime.device))
    return report


def worker_job(job_file):
    """Called only by a later explicit edge-worker command, not on import."""
    from .backend import YoloBackend, work
    from .cache import PackageCache
    from .data import load_detection_split
    from .tensorrt_backend import install_adapter_engines
    job_file = Path(job_file)
    job = json.loads(job_file.read_bytes())
    cfg = job["agent"]
    bundle_root = job_file.parent / "bundle"
    manifest = json.loads((bundle_root / "bundle.json").read_bytes())
    checkpoint = Path(cfg["bases"][manifest["base_sha256"]])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != manifest["base_sha256"]:
        raise ValueError("Target frozen base checksum mismatch")
    runtime = RuntimeSpec(**cfg["runtime"])
    runtime.activate()
    ledger = Ledger(Cost(**cfg["worker_budget"]), Cost(**cfg["worker_budget"]))
    from .journal import DurableRecords
    ledger.events = DurableRecords(job_file.parent / (job["kind"] + "-ledger.jsonl"))
    ledger.runtime, ledger.meter_policy = runtime, cfg["meter"]
    ledger.begin("release")
    with work(ledger, "target_setup", "release"):
        backend = YoloBackend(checkpoint, manifest["ports"], manifest["labels"], manifest["preprocessing"], manifest["image_size"], runtime)
    cache = PackageCache(cfg["cache_bytes"], backend.specs)
    for file in (bundle_root / "packages").glob("*.json"):
        p = Package.read(file.read_bytes())
        cache.put(p.hash, p.wire)
    candidate = State.load(bundle_root / "states" / (manifest["candidate"] + ".json"))
    incumbent = State.load(bundle_root / "states" / (manifest["incumbent"] + ".json"))
    context = manifest["context"]
    registry = cfg["registries"][context["registry_id"]]
    role = {"audit": "A", "qualify": "A", "evaluate": "V", "test": "T", "measure": "T"}[job["kind"]]
    split = load_detection_split(registry, context["client"], context["stage"], role,
                                 manifest["image_size"], manifest["labels"], formal=context.get("formal", False),
                                 batch_size=context.get("batch_size", 1), lazy=True,
                                 domain=context["domain"] if role == "T" else None)
    if split.identity != context["data_identity"] or split.domain != context["domain"]:
        raise ValueError("Target data/evaluator context differs from the locked requested split")
    guard = MemoryGuard(cfg["worker_memory_bytes"], cfg.get("minimum_free_bytes", 0), cfg.get("cuda_memory_bytes"))
    scores, plans, models = {}, {}, {}
    states = (candidate, incumbent) if job["kind"] == "audit" else (candidate,)
    for state in states:
        if state.hash in scores:
            continue
        guard.check()
        with work(ledger, "target_construct", "release"):
            model = backend.assembly(state, cache)
            model.eval()
        if cfg["executor"] == "tensorrt_adapters":
            with work(ledger, "target_engine_build", "release"):
                plans[state.hash] = install_adapter_engines(model, runtime.batch(split.batches[0]["img"]), runtime,
                                    Path(cfg["engine_store"]) / context["deployment_id"], cfg.get("trt_fp16", False), cfg["trt_workspace_bytes"])
        elif cfg["executor"] != "torch":
            raise ValueError("Unknown target executor")
        if job["kind"] not in ("qualify", "measure"):
            with work(ledger, "target_" + role, "release"):
                scores[state.hash] = backend.score(model, split)
        models[state.hash] = model
    report = {"candidate": candidate.hash, "incumbent": incumbent.hash,
              "code_identity": target_code_identity(),
              "data_identity": split.identity, "runtime": runtime.identity(),
              "candidate_score": scores.get(candidate.hash), "executor": cfg["executor"],
              "counting_coverage_complete": all(e.get("coverage_complete", True) for e in ledger.events)}
    if job["kind"] == "audit":
        report["incumbent_score"] = scores[incumbent.hash]
    if job["kind"] in ("qualify", "measure"):
        model = models[candidate.hash]
        # This fresh process has loaded only the one locked candidate. Release
        # the assembly template before measuring the serving detector itself.
        backend.base = None
        cache._wire.clear()
        import gc
        gc.collect()
        if torch.device(runtime.device).type == "cuda":
            torch.cuda.empty_cache()
        with work(ledger, "target_device_check", "release"):
            report["device"] = qualify_local(model, split.batches[0]["img"], runtime,
                                             cfg["measurement_repeats"], cfg["measurement_warmup"],
                                             predict=lambda image: backend.predict(model, image))
        report["device"]["latency_scope"] = "preprocessed CPU tensor, H2D, detector, fixed NMS, D2H; isolated candidate process"
        # Actual deployment storage counts the selected state, dependencies, base and plans.
        files = [checkpoint, bundle_root / "states" / (candidate.hash + ".json")]
        files += [bundle_root / "packages" / (key + ".json") for key in candidate.C]
        files += [Path(p["plan"]) for p in plans.get(candidate.hash, [])]
        report["device"]["deployed_bytes"] = sum(p.stat().st_size for p in {p.resolve() for p in files})
        report["device"]["tensor_size_scope"] = "runtime tensors; TRT plan bytes separately included in deployed_bytes"
        report["device"]["plans"] = plans.get(candidate.hash, [])
        report["device"]["executor"] = cfg["executor"]
        report["device"]["physical_device_id"] = cfg["device_id"]
    ledger.finish("release")
    report["counting_coverage_complete"] = all(e.get("coverage_complete", True) for e in ledger.events)
    report["actual_cost"] = __import__("dataclasses").asdict(ledger.used["release"])
    report["ledger"] = ledger.events
    atomic_write(job_file.parent / ("qualification.json" if job["kind"] == "qualify" else "audit_or_selection.json"), canonical(report))
    return report


class AgentJobs:
    def __init__(self, config):
        self.config = config
        self.root = Path(config["job_store"]).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()  # one CUDA qualification worker on this physical board

    def _check_storage(self, incoming=0):
        # Keep receipts, interrupted audits and committed bundles. Exhaustion
        # rejects new jobs; evidence is never silently evicted or re-searched.
        for directory, field in ((self.root, "maximum_job_store_bytes"),
                                 (Path(self.config["engine_store"]), "maximum_engine_store_bytes")):
            used = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()) if directory.exists() else 0
            allowance = self.config[field]
            reservation = incoming if field == "maximum_job_store_bytes" else 0
            if used + reservation > allowance:
                raise ExecutionStopped("Target persistent storage allowance reached: " + field)

    def _known_cost(self, root):
        total = {r: 0.0 for r in ("F", "E", "B", "W")}
        for file in root.glob("*-ledger.jsonl"):
            for line in file.read_bytes().splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                for r in total:
                    total[r] += event["cost"][r]
        return total

    def _launch(self, root, kind, payload, expected):
        job_file = root / (kind + "-job.json")
        atomic_write(job_file, canonical({"kind": kind, "agent": self.config, "payload": payload}))
        with open(root / (kind + ".stdout.txt"), "wb") as stdout, open(root / (kind + ".stderr.txt"), "wb") as stderr:
            process = subprocess.Popen([sys.executable, "-m", "softcas.edge_worker", str(job_file)],
                                       stdout=stdout, stderr=stderr, start_new_session=True)
            deadline = time.monotonic() + self.config["job_timeout_seconds"]
            failure = None
            try:
                import psutil
                while process.poll() is None:
                    self._check_storage()
                    try:
                        rss = psutil.Process(process.pid).memory_info().rss
                    except psutil.NoSuchProcess:
                        rss = 0
                    if time.monotonic() >= deadline or rss > self.config["worker_memory_bytes"]:
                        raise ExecutionStopped("Isolated worker deadline/RAM limit")
                    if sum(p.stat().st_size for p in root.glob("*.txt")) > self.config["maximum_log_bytes"]:
                        raise ExecutionStopped("Worker log storage bound")
                    time.sleep(0.05)
            except (RuntimeError, OSError) as exc:
                failure = exc
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
            if failure or process.returncode or not expected.is_file():
                partial = self._known_cost(root)
                raise WorkerFailure("Target worker failed; inspect job " + root.name + (": " + str(failure) if failure else ""), partial,
                                    cost_complete=False, remote_job_id=root.name)

    def execute(self, kind, payload, data):
        if hashlib.sha256(data).hexdigest() != payload["bundle_hash"]:
            raise ValueError("Incomplete or corrupt deployment bundle")
        job_id = digest({"kind": kind, "bundle": payload["bundle_hash"], "request": payload["request_id"]})
        root = self.root / job_id
        root.mkdir(exist_ok=True)
        report_file = root / "report.json"
        with self.lock:
            if not report_file.is_file():
                if not (root / "bundle" / "bundle.json").is_file():
                    self._check_storage(self.config["maximum_bundle_expanded_bytes"] + 2 * self.config["maximum_log_bytes"])
                manifest = unpack_bundle(data, root / "bundle", self.config["maximum_bundle_expanded_bytes"])
                if manifest["context"]["role"] != {"audit": "A", "evaluate": "V", "test": "T", "measure": "T"}[kind]:
                    raise ValueError("Audit/selection role mismatch")
                if manifest["context"]["registry_id"] not in self.config["registries"]:
                    raise ValueError("Target registry alias is not provisioned")
                audit_file = root / "audit_or_selection.json"
                if not audit_file.is_file():
                    if kind == "audit" and (root / "audit-ledger.jsonl").exists():
                        raise WorkerFailure("Interrupted locked audit is not repeated; retain incumbent and inspect " + job_id,
                                            self._known_cost(root), False, job_id)
                    self._launch(root, kind, payload, audit_file)
                report = json.loads(audit_file.read_bytes())
                if kind == "audit":
                    qualification_file = root / "qualification.json"
                    if not qualification_file.is_file():
                        self._launch(root, "qualify", payload, qualification_file)
                    qualification = json.loads(qualification_file.read_bytes())
                    if qualification["runtime"] != report["runtime"] or qualification["code_identity"] != report["code_identity"]:
                        raise WorkerFailure("Target executor changed between locked audit and qualification", self._known_cost(root), False, job_id)
                    report["device"] = qualification["device"]
                    report["qualification_ledger"] = qualification["ledger"]
                    report["actual_cost"] = {r: report["actual_cost"][r] + qualification["actual_cost"][r] for r in ("F", "E", "B", "W")}
                    report["counting_coverage_complete"] &= qualification["counting_coverage_complete"]
                atomic_write(report_file, canonical(report))
            manifest = json.loads((root / "bundle" / "bundle.json").read_bytes())
            report = json.loads(report_file.read_bytes())
            if report["runtime"] != RuntimeSpec(**self.config["runtime"]).identity() or report["code_identity"] != target_code_identity():
                raise RuntimeError("Saved locked result belongs to a different target executor; it is not silently re-audited")
            report["job_id"] = job_id
            if kind == "audit":
                limits = payload["device_limits"]
                eligible = report["candidate_score"] - report["incumbent_score"] >= payload["audit_epsilon"]
                eligible = eligible and all(report["device"][k] <= v for k, v in limits.items() if v is not None)
                eligible = eligible and (report["counting_coverage_complete"] or not payload.get("formal", False))
                receipt = {"job": job_id, "candidate": report["candidate"], "incumbent": report["incumbent"],
                           "client": manifest["context"]["client"], "deployment_id": manifest["context"]["deployment_id"],
                           "eligible": eligible, "report_sha256": hashlib.sha256(report_file.read_bytes()).hexdigest()}
                atomic_write(root / "receipt.json", canonical(receipt))
                report["receipt"] = job_id
            return report

    def commit(self, payload):
        receipt_id = payload["receipt"]
        if len(receipt_id) != 64 or any(c not in "0123456789abcdef" for c in receipt_id):
            raise ValueError("Invalid audit receipt")
        with self.lock:
            receipt = json.loads((self.root / receipt_id / "receipt.json").read_bytes())
            if not receipt["eligible"] or receipt["candidate"] != payload["candidate"]:
                raise ValueError("Only the locked qualified candidate can be activated")
            deployment_id = str(receipt["deployment_id"])
            if not deployment_id or any(not (c.isalnum() or c in "-_") for c in deployment_id):
                raise ValueError("Invalid deployment namespace")
            pointer = self.root / "current" / (deployment_id + ".json")
            current = json.loads(pointer.read_bytes()) if pointer.is_file() else None
            if current and current["candidate"] == receipt["candidate"] and current["receipt"] == receipt_id:
                return current  # idempotent recovery after a lost ACK
            if current and current["candidate"] != receipt["incumbent"]:
                raise ValueError("Deployment compare-and-swap incumbent mismatch")
            value = {"candidate": receipt["candidate"], "receipt": receipt_id, "activated": time.time(),
                     "bundle_root": str(self.root / receipt_id / "bundle"),
                     "qualification_report": str(self.root / receipt_id / "report.json")}
            atomic_write(pointer, canonical(value))
            return value

    def status(self, payload):
        client = str(payload["deployment_id"])
        if not client or any(not (c.isalnum() or c in "-_") for c in client):
            raise ValueError("Invalid logical client ID")
        pointer = self.root / "current" / (client + ".json")
        return json.loads(pointer.read_bytes()) if pointer.is_file() else {"candidate": None}

    def provision(self, payload, data):
        if hashlib.sha256(data).hexdigest() != payload["bundle_hash"]:
            raise ValueError("Initial bundle checksum mismatch")
        root = self.root / ("initial-" + payload["bundle_hash"])
        self._check_storage(self.config["maximum_bundle_expanded_bytes"])
        manifest = unpack_bundle(data, root / "bundle", self.config["maximum_bundle_expanded_bytes"])
        state = State.load(root / "bundle" / "states" / (manifest["candidate"] + ".json"))
        if state.C or manifest["candidate"] != manifest["incumbent"]:
            raise ValueError("Only the declared local-only base-equivalent incumbent can be initially provisioned")
        if any(torch.count_nonzero(blob.tensor()) for key, blob in state.phi if key.endswith("up.weight")):
            raise ValueError("Initial baseline residual must be zero; trained candidates require the locked audit")
        if manifest["base_sha256"] not in self.config["bases"]:
            raise ValueError("Initial base is not provisioned")
        base_file = Path(self.config["bases"][manifest["base_sha256"]])
        if hashlib.sha256(base_file.read_bytes()).hexdigest() != manifest["base_sha256"]:
            raise ValueError("Target initial base file checksum mismatch")
        client = str(manifest["context"]["deployment_id"])
        if not client or any(not (c.isalnum() or c in "-_") for c in client):
            raise ValueError("Invalid logical client ID")
        pointer = self.root / "current" / (client + ".json")
        with self.lock:
            if pointer.exists():
                current = json.loads(pointer.read_bytes())
                if current["candidate"] != state.hash:
                    raise ValueError("An existing deployment cannot be overwritten by baseline provisioning")
                return current
            value = {"candidate": state.hash, "receipt": None, "bundle_root": str(root / "bundle"), "initialized": time.time()}
            atomic_write(pointer, canonical(value))
            return value


class RemoteReleaseService:
    def __init__(self, endpoint, registry_id, client, formal=False, deployment_id=None):
        self.endpoint, self.registry_id, self.client, self.formal = endpoint, registry_id, client, formal
        self.receipt = None
        self.deployment_id = deployment_id or str(client)
        self.ledger = None
        self.records = []

    def _rpc(self, ledger, config):
        from .transport import TcpClient
        return TcpClient(self.client, self.endpoint["host"], self.endpoint["port"], ledger,
                         self.endpoint["token_env"], config["remote_timeout_seconds"], self.records)

    def audit(self, backend, candidate, incumbent, split, cache, ledger, config):
        context = {"registry_id": self.registry_id, "client": split.client, "stage": split.stage,
                   "domain": split.domain, "role": "A", "data_identity": split.identity, "formal": self.formal,
                   "batch_size": getattr(split, "batch_size", 1), "deployment_id": self.deployment_id}
        payload = bundle_bytes(backend, candidate, incumbent, cache, context)
        ledger.guard(Cost(B=len(payload)), "release")
        request = {"bundle_hash": hashlib.sha256(payload).hexdigest(), "request_id": digest(context | {"candidate": candidate.hash, "incumbent": incumbent.hash}),
                   "audit_epsilon": config["audit_epsilon"], "device_limits": config["device_limits"], "formal": self.formal}
        try:
            report, _ = self._rpc(ledger, config).call("audit", request, payload, "release")
        except (RuntimeError, OSError) as exc:
            response = getattr(exc, "response", {})
            partial, job_id = response.get("partial_cost"), response.get("remote_job_id")
            already = sum(e["cost"]["F"] for e in ledger.events if job_id is not None and e.get("remote_job_id") == job_id)
            ledger.charge("failed_target_known_computation", Cost(F=max(0, partial["F"] - already) if partial else 0), "release",
                           target_partial_cost=partial, coverage_complete=False, remote_job_id=job_id,
                           note="F is only the known lower bound; unobserved in-flight work remains unavailable")
            raise
        if report["candidate"] != candidate.hash or report["incumbent"] != incumbent.hash or report["data_identity"] != split.identity:
            raise ValueError("Target audit response identity mismatch")
        already = sum(e["cost"]["F"] for e in ledger.events if e.get("remote_job_id") == report["job_id"])
        ledger.charge("target_audit_computation", Cost(F=max(0, report["actual_cost"]["F"] - already)), "release",
                      physical_executor=report["device"]["physical_device_id"], target_cost=report["actual_cost"], remote_job_id=report["job_id"],
                      coverage_complete=report["counting_coverage_complete"])
        self.receipt, self.ledger = report["receipt"], ledger
        return report

    def activate(self, candidate, config):
        rpc = self._rpc(self.ledger, config)
        try:
            result, _ = rpc.call("commit", {"receipt": self.receipt, "candidate": candidate}, scope="release")
            if result["candidate"] != candidate:
                raise ValueError("Activation acknowledgement differs from locked candidate")
            return result
        except (OSError, RuntimeError) as error:
            try:
                result, _ = rpc.call("status", {"deployment_id": self.deployment_id}, scope="release")
            except (OSError, RuntimeError) as followup:
                raise DeploymentOutcomeUnknown("Commit acknowledgement lost; reconcile target status before any new stage") from followup
            if result.get("candidate") == candidate and result.get("receipt") == self.receipt:
                return result
            raise RuntimeError("Target did not activate the qualified candidate") from error

    def ensure_incumbent(self, backend, state, cache, ledger, config):
        rpc = self._rpc(ledger, config)
        current, _ = rpc.call("status", {"deployment_id": self.deployment_id}, scope="release")
        if current.get("candidate") == state.hash:
            return current
        if current.get("candidate") is not None:
            raise DeploymentOutcomeUnknown("Target/source incumbents differ; reconcile before adapting")
        context = {"client": self.client, "role": "baseline", "registry_id": self.registry_id, "deployment_id": self.deployment_id}
        data = bundle_bytes(backend, state, state, cache, context)
        value, _ = rpc.call("provision", {"bundle_hash": hashlib.sha256(data).hexdigest()}, data, "release")
        return value

    def evaluate(self, backend, state, split, cache, ledger, config, role="T", measure=False, measurement_repeat=None):
        if role not in ("T", "V") or split.role != role:
            raise ValueError("Explicit independent-test or selection role required")
        context = {"registry_id": self.registry_id, "client": split.client, "stage": split.stage,
                   "domain": split.domain, "role": role, "data_identity": split.identity, "formal": self.formal,
                   "batch_size": getattr(split, "batch_size", 1), "deployment_id": self.deployment_id}
        if measurement_repeat is not None:
            if not measure or isinstance(measurement_repeat, bool) or not isinstance(measurement_repeat, int) or measurement_repeat < 0:
                raise ValueError("Measurement repetition identity requires an independent measurement request")
            context["measurement_repeat"] = measurement_repeat
        data = bundle_bytes(backend, state, state, cache, context)
        if measure and role != "T":
            raise ValueError("Post-deployment measurement uses the independent reporting scope")
        request = {"bundle_hash": hashlib.sha256(data).hexdigest(), "request_id": digest(context | {"state": state.hash, "measure": measure})}
        operation = "measure" if measure else ("test" if role == "T" else "evaluate")
        report, _ = self._rpc(ledger, config).call(operation, request, data,
                                               "search" if role == "T" else "release")
        if report["candidate"] != state.hash or report["data_identity"] != split.identity:
            raise ValueError("Independent target evaluation identity mismatch")
        already = sum(e["cost"]["F"] for e in ledger.events if e.get("remote_job_id") == report["job_id"])
        ledger.charge("independent_target_test" if role == "T" else "target_selection",
                      Cost(F=max(0, report["actual_cost"]["F"] - already)), "search" if role == "T" else "release",
                      target_cost=report["actual_cost"], remote_job_id=report["job_id"],
                      coverage_complete=report["counting_coverage_complete"])
        return report["candidate_score"], report
