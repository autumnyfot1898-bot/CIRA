"""Algorithms softcas and child: execution, feedback, one locked audit, release."""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import torch
from .actions import construct
from .backend import work
from .controller import Archive, Scored, Ridge, Selector, FeatureEncoder
from .evaluation import EvaluationCache
from .proposals import ProposalFactory
from .resources import Cost, ExecutionStopped
from .state import State, Package, canonical, freeze_tensors
from .fitting import epoch_mode, stage_steps, epoch_seed, shuffled_batches


@dataclass(frozen=True)
class AttemptResult:
    state: State
    score: float | None
    cost: Cost
    status: str
    error: str | None
    reuse: bool = False
    fit_steps: int = 0
    completed_fit_steps: int = 0
    epoch_seeds: tuple = ()


class Executor:
    def __init__(self, backend, exchange, evaluator, config, ledger):
        self.backend, self.exchange, self.evaluator = backend, exchange, evaluator
        self.config, self.ledger = config, ledger

    def run(self, proposal, parent, data):
        before = self.ledger.used["search"]
        start = time.perf_counter()
        token = "running:" + proposal.hash
        cache = self.exchange.cache
        cache.pin(token, parent.C + ((proposal.add,) if proposal.add else ()) +
                  (proposal.imported.C if proposal.imported else ()))
        attempted, executed, shuffle_seeds = 0, 0, []
        try:
            self.ledger.guard()
            if proposal.add and proposal.add not in cache:
                self.exchange.acquire(proposal.add, time.time())
            child = construct(parent, proposal, cache)
            if self.config["ablation"] == "discrete" and proposal.kind != "import":
                import_choices = dict(proposal.discrete) if proposal.discrete is not None else json.loads(parent.nu).get("discrete_choices", {})
                child = child.lineage(receiving_rule="discrete", discrete_choices=import_choices)
            if any(k not in cache for k in child.C):
                raise ExecutionStopped("Exact checkpoint dependency unavailable")
            self.backend.before_candidate(child, cache)
            # Snapshot import verbatim; every fitting block starts a fresh optimizer.
            if proposal.kind != "import":
                mode = proposal.kind if proposal.kind in ("fresh", "refine") else "collaboration"
                with work(self.ledger, "construct"):
                    model = self.backend.assembly(child, cache, proposal.seed, mode)
                    for port in model.ports.values():
                        port.epsilon = self.config["residual_epsilon"]
                optimizer = self.backend.optimizer(model, child, self.config)
                scaler = self.backend.new_scaler()
                default_steps = self.config["phi_steps"] if mode in ("fresh", "refine") else self.config["beta_steps"]
                # Legacy providers used fidelity=0 as an unspecified value. The
                # explicit native protocol binds even zero before cost ranking.
                steps = proposal.fidelity if epoch_mode(self.config) or proposal.fidelity else default_steps
                if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
                    raise ValueError("Fitting fidelity must be a nonnegative integer")
                if epoch_mode(self.config) and steps > self.config.get("_fit_steps_remaining", steps):
                    raise ValueError("Proposal exceeds the admitted stage fitting remainder")
                if epoch_mode(self.config):
                    key = "local_epochs" if mode in ("fresh", "refine") else "receiving_epochs"
                    if steps > self.config["fitting"][key] * len(data.D.batches):
                        raise ValueError("Proposal exceeds the declared per-block epoch cap")
                if optimizer is not None:
                    batches = data.D.batches
                    for step in range(steps):
                        self.ledger.guard()
                        if epoch_mode(self.config) and step % len(data.D.batches) == 0:
                            seed = epoch_seed(self.config, proposal, data.D, step // len(data.D.batches))
                            batches = shuffled_batches(data.D.batches, seed)
                            shuffle_seeds.append(seed)
                        with work(self.ledger, "fit"):
                            batch = batches[step % len(batches)]
                            optimizer.zero_grad(set_to_none=True)
                            attempted += 1
                            self.backend.fit_batch(model, batch, child.h.residual, optimizer, scaler)
                            executed += 1
                child = model.snapshot(child)
                history = json.loads(parent.nu)
                child = child.lineage(parent=parent.hash, secondary=proposal.secondary, action=proposal.kind,
                                      family=proposal.family, seed=proposal.seed, fit_steps=history.get("fit_steps", 0) + executed,
                                      last_fit_steps=executed, optimizer_reset=optimizer is not None,
                                      fit_data=data.D.identity, stage=data.D.stage, client=data.D.client,
                                      execution=self.backend.execution,
                                      serial_phase=self.config.get("_serial_phase", 0),
                                      **({"epoch_shuffle_seeds": shuffle_seeds} if epoch_mode(self.config) else {}))
            else:
                with work(self.ledger, "construct_import"):
                    model = self.backend.assembly(child, cache)
            if len(child.C) > self.config["max_external_paths"]:
                raise ExecutionStopped("Total path cap")
            # Capacity checks are conservative tensor-size checks before full V evaluation.
            size = sum(v.numel() * v.element_size() for v in model.state_dict().values())
            size_limit = self.config["device_limits"].get("deployed_tensor_bytes")
            if size_limit is not None and size > size_limit:
                raise ExecutionStopped("Deployment size capacity")
            self.ledger.guard()
            score, reused = self.evaluator.evaluate(self.backend, child, cache, data.V, self.ledger, prepared_model=model)
            status, error = "complete", None
        except (RuntimeError, ValueError, KeyError, OSError) as exc:
            child, score, status, error, reused = parent, None, "failed", f"{type(exc).__name__}: {exc}", False
        finally:
            cache.unpin(token)
            # Inner work/transfer scopes already charged wall time. Charge only the
            # unaccounted difference; returning L-L0 is NEVER a second charge.
            elapsed = time.perf_counter() - start
            charged = (self.ledger.used["search"] - before).W
            self.ledger.charge("execution_bookkeeping", Cost(W=max(0, elapsed - charged)))
        return AttemptResult(child, score, self.ledger.used["search"] - before, status, error, reused,
                             attempted, executed, tuple(shuffle_seeds))


@dataclass
class StageResult:
    incumbent: State
    champion: State
    deployed: State
    actions: list
    audit: dict
    ledger: object
    published: tuple
    archive: object
    predictor: object
    stop: str


def initial_state(backend, cache, z, h, seed=0):
    from .state import digest
    state = State(z, h, (), (), (), canonical({"fit_steps": 0, "seed": seed, "base": backend.identity,
                    "initializer": "port_seed_sha256_v1",
                    "ports_signature": digest([asdict(spec) for spec in backend.specs])}).decode())
    model = backend.assembly(state, cache, seed, "fresh")
    return model.snapshot(state)


def run_stage(backend, incumbent, data, cache, exchange, ledger, config,
              retained=(), evaluator=None, proposal_provider=None, owner="client0"):
    config = dict(config)
    config["randomization"] = {**config.get("randomization", {}),
        "client": str(data.D.client), "stage": data.D.stage}
    config["randomization"].setdefault("detector", backend.identity)
    config["randomization"].setdefault("dataset", data.D.domain)
    config["_fit_batches_per_epoch"] = len(data.D.batches)
    fit_limit = stage_steps(config, len(data.D.batches))
    fit_used = 0
    stage_start, stage_W = time.perf_counter(), ledger.used["search"].W
    journal = getattr(backend, "journal", None)
    if journal:
        journal.emit("stage_entry", incumbent=incumbent.hash, client=data.D.client, stage=data.D.stage)
    evaluator = evaluator or EvaluationCache()
    # Stage-local predictor/selector reset even when raw evaluation cache persists.
    encoder = FeatureEncoder(config["features"]["fields"], config["features"]["scales"])
    ridge = Ridge(len(encoder.fields), config["ridge_lambda"])
    selector = Selector(config["cold_order"], config["type_order"], config["kappa"], config["priority_epsilon"],
                        "round_robin" if config["method"] == "round_robin" else "softcas",
                        config["ablation"] == "no_gain",
                        proposal_protocol=config.get("proposal_protocol", "legacy"))
    cache.pin("incumbent", incumbent.C)
    if any(k not in cache for k in incumbent.C):
        raise ValueError("Stage-entry incumbent requires all exact verified dependencies")
    for token in tuple(cache.pins):
        if token in ("locked_champion", "deployed", "archive") or token.startswith("retained:"):
            cache.unpin(token)
    for i, state in enumerate(retained):
        cache.pin("retained:" + str(i), state.C)
    v0, _ = evaluator.evaluate(backend, incumbent, cache, data.V, ledger)
    if journal:
        journal.emit("incumbent_selection", incumbent=incumbent.hash, score=v0, V=data.V.identity)
    archive = Archive(incumbent, v0, config["archive_capacity"], cache,
                      config["archive_truncation"], config["tournament_tie"], getattr(backend, "state_store", None))
    factory = ProposalFactory(config, backend, exchange, retained)
    executor = Executor(backend, exchange, evaluator, config, ledger)
    actions = []
    stop = "attempt_limit"
    while len(actions) < config["attempts"]:
        config["_attempt_index"] = len(actions)
        if fit_limit is not None:
            config["_fit_steps_remaining"] = max(0, fit_limit - fit_used)
        start, used_before = time.perf_counter(), ledger.used["search"]
        numeric_before = ridge.flops + selector.flops
        try:
            ledger.guard()
            if hasattr(exchange, "refresh"):
                exchange.refresh()
            proposals = proposal_provider(archive, len(actions)) if proposal_provider else factory.generate(archive, ledger)
            if journal and config.get("proposal_protocol") == "native_r1000":
                journal.emit("proposal_randomization", attempt=len(actions),
                             streams=getattr(factory, "randomization_records", []))
            # Resolve already known IMPORT execution identities before encoding/ranking.
            resolved = []
            for a in proposals:
                if a.kind == "import" and evaluator.key(backend, a.imported, data.V) in evaluator.scores:
                    score = evaluator.scores[evaluator.key(backend, a.imported, data.V)]
                    archive.insert(Scored(a.imported, score, score - v0, json.loads(a.imported.nu).get("family", "incumbent")))
                else:
                    resolved.append(a)
            decision, reason = selector.choose(resolved, archive, ridge, encoder, ledger.remaining(), cache, exchange.catalogue)
        except ExecutionStopped:
            decision, reason = None, "execution_limit"
        # Closed-form ridge operation count is a disclosed controller proxy; W measured.
        dim = len(encoder.fields)
        ledger.charge("controller", Cost(F=ridge.flops + selector.flops - numeric_before,
                                         W=max(0, time.perf_counter() - start -
                                               (ledger.used["search"] - used_before).W)),
                      meter="LAPACK_LU_scalar_equivalent_plus_explicit_feature_priority_arithmetic")
        if decision is not None and not decision[0].estimate.fits(ledger.remaining()):
            decision, reason = None, "controller_budget"
        if decision is None:
            if config["method"] == "serial" and factory.phase == 0:
                factory.advance_serial(archive, ledger)
                continue
            stop = reason
            break
        a, f, mu, coverage, J = decision
        parent = archive.items[a.parent]
        # f is a tuple; saved before any acquisition, fitting, child score or update.
        record = {"attempt": len(actions) + 1, "proposal": a.hash, "kind": a.kind, "type": a.type,
                  "parent": a.parent, "parent_utility": parent.utility, "features": list(f),
                  "feature_fields": list(encoder.fields), "prediction": mu, "coverage": coverage,
                  "priority": J, "decision": reason, "estimate": asdict(a.estimate),
                  "champion_before": archive.champion.state.hash,
                  "champion_utility_before": archive.champion.utility,
                  "seed": a.seed, "add": a.add, "remove": a.remove, "secondary": a.secondary,
                  "proposed_z": asdict(a.z) if a.z else None, "proposed_h": asdict(a.h) if a.h else None,
                  "fidelity": a.fidelity, "pre_ledger": asdict(ledger.used["search"]),
                  "proposal_streams": list(getattr(factory, "randomization_records", [])),
                  "candidate_pool": selector.last_pool,
                  "priority_constants": {"kappa": config["kappa"], "epsilon": config["priority_epsilon"]},
                  "no_gain": config["ablation"] == "no_gain",
                  "cache_before": cache.keys(), "predictor_n_before": ridge.n}
        config["_serial_phase"] = factory.phase
        if journal:
            journal.emit("before_attempt", **record)
        result = executor.run(a, parent.state, data)
        fit_used += result.fit_steps
        if fit_limit is not None:
            config["_fit_steps_remaining"] = max(0, fit_limit - fit_used)
        utility = result.score - v0 if result.status == "complete" else parent.utility
        gain = utility - parent.utility if result.status == "complete" else 0.0
        if result.status == "complete":
            try:
                archive.insert(Scored(result.state, result.score, utility, a.family))
            except (RuntimeError, OSError) as exc:
                result = replace(result, state=parent.state, score=None, status="failed",
                                 error="archive persistence: " + str(exc))
                utility, gain = parent.utility, 0.0
        update_start = time.perf_counter()
        feedback_before = ridge.flops
        ridge.update(f, gain)
        selector.observed.add(a.type)
        ledger.charge("predictor_feedback", Cost(F=ridge.flops - feedback_before + (2 if result.status == "complete" else 0), W=time.perf_counter() - update_start),
                      meter="outer_product_operation_count")
        record.update(child=result.state.hash, child_utility=utility, score=result.score, gain=gain,
                      cost=asdict(result.cost), status=result.status, error=result.error, reuse=result.reuse,
                      cache_after=cache.keys(), packages=result.state.C,
                      predictor_n_after=ridge.n, post_ledger=asdict(ledger.used["search"]),
                      fit_steps=result.fit_steps, completed_fit_steps=result.completed_fit_steps,
                      stage_fit_steps=fit_used, stage_fit_step_limit=fit_limit,
                      epoch_shuffle_seeds=list(result.epoch_seeds),
                      archive_after=sorted(archive.items),
                      champion_after=archive.champion.state.hash,
                      feasibility=ledger.violations())
        actions.append(record)
        if journal:
            journal.emit("attempt_result", **record)
        if config["method"] == "serial" and factory.phase == 0 and not epoch_mode(config):
            phase_attempts = max(1, int(config["attempts"] * config["serial"]["attempt_fraction"]))
            if len(actions) >= phase_attempts:
                factory.advance_serial(archive, ledger)
    champion = archive.champion.state  # lock once; audit never feeds back into search
    if journal:
        journal.emit("locked", incumbent=incumbent.hash, champion=champion.hash)
    ledger.charge("stage_search_bookkeeping", Cost(W=max(0, time.perf_counter() - stage_start -
                                                       (ledger.used["search"].W - stage_W))))
    ledger.finish("search")
    cache.pin("locked_champion", champion.C)
    audit = {"locked": champion.hash, "incumbent": incumbent.hash, "released": False,
             "threshold": config["audit_epsilon"], "metric": backend.metric}
    deployed, published = incumbent, []
    release_start, release_W = time.perf_counter(), ledger.used["release"].W
    ledger.begin("release")
    try:
        release_service = getattr(backend, "release_service", None)
        if release_service is not None:
            report = release_service.audit(backend, champion, incumbent, data.A, cache, ledger, config)
            plus, old, device = report["candidate_score"], report["incumbent_score"], report["device"]
            audit["target_report"] = report
            if config["purpose"] == "FORMAL" and not report["counting_coverage_complete"]:
                raise RuntimeError("Target accounting coverage is incomplete; the locked candidate is not activated")
        else:
            plus, _ = evaluator.evaluate(backend, champion, cache, data.A, ledger, "release")
            old, _ = evaluator.evaluate(backend, incumbent, cache, data.A, ledger, "release")
            with work(ledger, "device_qualification", "release"):
                model = backend.assembly(champion, cache)
                device = backend.qualify(model, data.A.batches[0]["img"], config["qualification_repeats"])
        limits = config["device_limits"]
        device_ok = all(device[k] <= limits[k] for k in limits if limits[k] is not None)
        audit.update(candidate_score=plus, incumbent_score=old, difference=plus-old, device=device,
                     device_pass=device_ok)
        if plus - old >= config["audit_epsilon"] and device_ok and not ledger.violations()["release"]:
            deployed = champion
            audit["released"] = champion.hash != incumbent.hash
        if journal:
            journal.emit("audit_result", audit=audit)
        # Only self-owned local functions of a new release are published.
        if audit["released"]:
            with work(ledger, "release_publication_prepare", "release"):
                model = backend.assembly(deployed, cache)
                now = time.time()
                for i, port in model.ports.items():
                    if port.local is not None:
                        previous_local = tuple((k, v) for k, v in incumbent.phi if k.startswith(i + "."))
                        current_local = tuple((k, v) for k, v in deployed.phi if k.startswith(i + "."))
                        if previous_local == current_local:
                            continue
                        published.append(Package(owner, deployed.hash + ":" + i, now, now,
                                                 backend.specs[int(i)], deployed.z.ratio,
                                                 freeze_tensors(port.local.state_dict()),
                                                 canonical({"assembly": deployed.hash, "roles": data.D.identity}).decode()))
    except (RuntimeError, ValueError, KeyError, OSError) as exc:
        audit["error"] = f"{type(exc).__name__}: {exc}"
        deployed, published = incumbent, []
        audit["released"] = False
    ledger.charge("release_bookkeeping", Cost(W=max(0, time.perf_counter() - release_start -
                                                   (ledger.used["release"].W - release_W))), "release")
    if ledger.violations()["release"]:
        deployed, published = incumbent, []
        audit["released"] = False
    audit["deployed"] = deployed.hash
    if audit["released"] and getattr(backend, "release_service", None) is not None:
        try:
            audit["activation"] = backend.release_service.activate(champion.hash, config)
        except (RuntimeError, ValueError, OSError) as exc:
            deployed, published = incumbent, []
            audit.update(deployed=incumbent.hash, released=False, activation_error=str(exc))
    ledger.finish("release")
    if config["method"] == "serial":
        audit["serial_phase_transition"] = factory.phase_transition
    cache.pin("deployed", deployed.C)
    if journal:
        journal.emit("stage_complete", deployed=deployed.hash, audit=audit)
    return StageResult(incumbent, champion, deployed, actions, audit, ledger, tuple(published), archive, ridge, stop)


def export_stage(result, directory, backend, data, config, evaluator, cache, exchange):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for state in {s.hash: s for s in (result.incumbent, result.champion, result.deployed,
                                    *(s.state for s in result.archive.items.values()))}.values():
        state.save(directory / "checkpoints" / (state.hash + ".json"))
    for key in cache.keys():
        p = directory / "packages" / (key + ".json")
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(cache.get(key).wire)
    for package in result.published:
        p = directory / "published" / (package.hash + ".json")
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(package.wire)
    for name, rows in (("actions", result.actions), ("ledger", result.ledger.events),
                       ("evaluations", evaluator.records), ("transfers", exchange.records),
                       ("cache_transitions", cache.transitions)):
        (directory / (name + ".jsonl")).write_bytes(b"\n".join(canonical(r) for r in rows))
    (directory / "stage.json").write_bytes(canonical({"purpose": config["purpose"], "metric": backend.metric,
        "incumbent": result.incumbent.hash, "champion": result.champion.hash, "deployed": result.deployed.hash,
        "stop": result.stop, "audit": result.audit, "actual": {k: asdict(v) for k, v in result.ledger.used.items()},
        "violations": result.ledger.violations(), "base": backend.identity, "execution": backend.execution,
        "data": {r: getattr(data, r).identity for r in ("D", "V", "A")}, "config": config,
        "flop_accounting": {"policy": config.get("flop_meter", "torch_profiler_partial"),
                            "coverage_complete": all(e.get("coverage_complete", True) for e in result.ledger.events),
                            "controller": "LU scalar equivalent, no hardware-issued instruction claim"},
        "published": [p.hash for p in result.published]}))
