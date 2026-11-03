"""CPU fixture federation, suite, and genuine YOLO interface smoke entrypoints."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from dataclasses import replace, asdict
from pathlib import Path
import torch
from .backend import Backend, YoloBackend, work
from .cache import PackageCache, ContactExchange
from .config import load_config, validate_config, METHODS, ABLATIONS
from .data import StageData, synthetic_split, load_detection_stage
from .engine import initial_state, run_stage, export_stage
from .evaluation import EvaluationCache
from .resources import Cost, Ledger, TransferJournal
from .reporting import export_matrix, write_csv
from .state import Structure, FitSettings, Package, freeze_tensors, canonical


ROOT = Path(__file__).resolve().parents[1]


def environment():
    def version(name):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "dependencies": {m: version(m) for m in ("torch", "torchvision", "ultralytics", "numpy", "scipy")},
            "torch_path": torch.__file__, "threads": torch.get_num_threads()}


def demo(config, output, stages=2, clients=2):
    """Sequential causal contact trace, same method's own earlier publications.

    All data and quality scores are synthetic. Per-client stage order is seeded;
    source-serving costs are kept in source node ledgers, separate from a
    receiver's stage search allowance. Production is endogenous to each method.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    backend = Backend()
    caches = [PackageCache(config["cache_bytes"], backend.specs) for _ in range(clients)]
    fixed = config["fixed"]
    states = [initial_state(backend, cache, Structure(tuple(fixed["mask"]), fixed["ratio"]),
                            FitSettings(fixed["lr"], fixed["residual"]), config["seed"]) for cache in caches]
    evaluators = [EvaluationCache() for _ in range(clients)]
    published = [{} for _ in range(clients)]
    source_ledgers = [Ledger(Cost(F=1e30, E=1e30, B=1e30, W=1e30), Cost(), enforce=()) for _ in range(clients)]
    journal, rows, stage_rows = TransferJournal(), [], []
    offline_ledger = Ledger(Cost(1e30, 1e30, 1e30, 1e30), Cost(), enforce=())
    orders = []
    import random
    for i in range(clients):
        order = list(range(stages))
        random.Random(config["seed"] + i).shuffle(order)
        orders.append(order)
    for stage in range(1, stages + 1):
        for client in range(clients):
            domain = str(orders[client][stage - 1])
            data = StageData(*(synthetic_split(r, str(client), stage, domain,
                              config["seed"] + 100 * stage + 1000 * client + j)
                              for j, r in enumerate(("D", "V", "A"))))
            ledger = Ledger(Cost(**config["budgets"]["search"]), Cost(**config["budgets"]["release"]))
            exchange = ContactExchange(str(client), caches[client], ledger, journal)
            for peer in range(clients):
                if peer != client:
                    exchange.observe(str(peer), published[peer], time.time(), config["contact_quota"], source_ledgers[peer])
            previous = states[client]
            result = run_stage(backend, previous, data, caches[client], exchange, ledger,
                               copy.deepcopy(config), retained=(previous,), evaluator=evaluators[client], owner=str(client))
            states[client] = result.deployed
            for p in result.published:
                published[client][p.hash] = p.wire
            export_stage(result, output / f"client{client}" / f"stage{stage}", backend, data,
                         config, evaluators[client], caches[client], exchange)
            stage_rows.append({"client": client, "stage": stage, "method": config["method"], "ablation": config["ablation"],
                               "attempts": len(result.actions), "released": result.audit["released"],
                               "search_F_partial": ledger.used["search"].F, "selection_E": ledger.used["search"].E,
                               "node_B": ledger.used["search"].B, "update_W": ledger.used["search"].W,
                               "deployed": result.deployed.hash})
            # Independent reporting AFTER locked release; score never passed to search.
            for past_domain in orders[client][:stage]:
                split = synthetic_split("T", str(client), 0, str(past_domain), config["seed"] + 90000 + past_domain)
                model = backend.assembly(result.deployed, caches[client])
                with work(offline_ledger, "independent_historical_test"), torch.no_grad():
                    score = backend.score(model, split)
                rows.append({"seed": config["seed"], "client": str(client), "stage": stage,
                             "current_domain": domain, "domain": str(past_domain), "deployed": result.deployed.hash,
                             "test_manifest": split.identity, "first_stage": orders[client].index(past_domain) + 1,
                             "score": score, "metric": backend.metric, "purpose": "SYNTHETIC_BEHAVIOR_ONLY"})
    endpoints = export_matrix(output, rows, {str(i) for i in range(clients)})
    write_csv(output / "stage_resources.csv", stage_rows, tuple(stage_rows[0]))
    summary = {"purpose": "SYNTHETIC_BEHAVIOR_ONLY", "metric": backend.metric,
               "method": config["method"], "ablation": config["ablation"],
               "network_tx_bytes": journal.network_bytes, "source_serving": [asdict(l.used["search"]) for l in source_ledgers],
               "events": len(stage_rows), "attempts": sum(r["attempts"] for r in stage_rows),
               "published_versions": sum(len(p) for p in published), "endpoints": endpoints,
               "offline_test_cost_partial": asdict(offline_ledger.used["search"]),
               "environment": environment(), "formal_performance_result": False}
    (output / "summary.json").write_bytes(canonical(summary))
    (output / "network_transfers.json").write_bytes(canonical(journal.records))
    return summary


def yolo_smoke(checkpoint, output, image=None):
    from ultralytics import YOLO
    # This only probes the checkpoint's own label space, not an approved 8-class base.
    labels = tuple(YOLO(str(checkpoint)).model.names.values())
    backend = YoloBackend(checkpoint, (2, 4, 6, 8, 15, 18), labels,
                          "interface_smoke_rgb_resize64_no_formal_preprocessing", 64)
    cache = PackageCache(10000000, backend.specs)
    state = initial_state(backend, cache, Structure((1, 0, 0, 0, 0, 1), 1 / 8), FitSettings(1e-3, 1e-3), 77)
    # A synthetic external branch exercises frozen-operator gradients and beta.
    # Its provenance is explicit; it is not an empirical peer-produced adapter.
    from .model import Adapter
    with torch.random.fork_rng():
        torch.manual_seed(91)
        donor = Adapter(backend.specs[2].channels, 1 / 16)
        torch.nn.init.normal_(donor.up.weight, 0, 0.001)
    package = Package("interface_fixture_peer", "fixture_v1", 0, 0, backend.specs[2], 1 / 16,
                      freeze_tensors(donor.state_dict()), "SYNTHETIC_INTERFACE_ONLY")
    cache.put(package.hash, package.wire)
    state = replace(state, C=(package.hash,), beta=((package.hash, 0),))
    model = backend.assembly(state, cache, 77, "refine")
    original = {k: v.clone() for k, v in model.base.state_dict().items()}
    if image:
        import PIL.Image
        import numpy as np
        img = torch.tensor(np.asarray(PIL.Image.open(image).convert("RGB").resize((64, 64))).copy()).permute(2, 0, 1).float()[None] / 255
    else:
        img = torch.rand(1, 3, 64, 64)
    # Synthetic box: only interface/gradient behavior, never an AP result.
    batch = {"img": img, "batch_idx": torch.tensor([0.]), "cls": torch.tensor([[0.]]),
             "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.4]])}
    optimizer = model.optimizer(1e-3, 1e-3)
    losses = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = backend.loss(model, batch) + state.h.residual * model.penalty()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    from .data import Split
    selection_probe = Split("V", "interface", 1, "probe", (batch,), "SYNTHETIC_BOX_NOT_BENCHMARK")
    decoder_score = backend.score(model, selection_probe)
    finite_grad = all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    nonzero = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0]
    base_unchanged = all(torch.equal(v, original[k]) for k, v in model.base.state_dict().items())
    report = {"purpose": "REAL_CHECKPOINT_INTERFACE_ONLY_SYNTHETIC_BOX_NOT_PERFORMANCE", "checkpoint": str(checkpoint),
              "checkpoint_sha256": backend.identity, "image": str(image) if image else None,
              "label_count": len(labels), "losses": losses, "finite_gradients": bool(finite_grad),
              "external_fixture_hash": package.hash,
              "external_parameters_and_gate_frozen": all(p.grad is None and not p.requires_grad
                    for port in model.ports.values() for p in port.external.parameters()),
              "complete_detector_ap_interface_executed": bool(0 <= decoder_score <= 1),
              "base_parameters_and_buffers_unchanged": base_unchanged, "trainable_nonzero_gradients": nonzero,
              "ports": [asdict(s) for s in backend.specs], "environment": environment()}
    if not finite_grad or not base_unchanged or not nonzero:
        raise RuntimeError("YOLO native loss/freeze smoke failed")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical(report))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("demo", "suite", "yolo-smoke", "run-stage", "check-config", "environment"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "fixture.json"))
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "demo"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--image")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--ablation", choices=ABLATIONS)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--manifest")
    parser.add_argument("--detector-config")
    parser.add_argument("--incumbent")
    parser.add_argument("--client", default="0")
    parser.add_argument("--stage", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "environment":
        print(json.dumps(environment(), indent=2))
        return
    if args.command == "yolo-smoke":
        report = yolo_smoke(args.checkpoint, args.output, args.image)
        print(json.dumps({k: report[k] for k in ("purpose", "checkpoint_sha256", "label_count", "losses", "finite_gradients", "base_parameters_and_buffers_unchanged")}))
        return
    c = load_config(args.config, args.formal)
    if args.method:
        c["method"] = args.method
    if args.ablation:
        c["ablation"] = args.ablation
    validate_config(c, args.formal)
    if args.command == "check-config":
        print(json.dumps({"valid": True, "purpose": c["purpose"], "formal": args.formal}))
    elif args.command == "demo":
        result = demo(c, args.output, args.stages, args.clients)
        print(json.dumps({k: result[k] for k in ("purpose", "method", "ablation", "events", "attempts", "published_versions")}))
    elif args.command == "run-stage":
        detector = json.loads(Path(args.detector_config).read_text(encoding="utf-8"))
        if detector["image_size"] % 32 or detector["image_size"] < 32:
            raise ValueError("YOLO input size must be a positive multiple of 32")
        if detector["preprocessing"] != "rgb_letterbox114_bilinear_float32_div255":
            raise ValueError("Declare the preprocessing actually implemented by this loader")
        backend = YoloBackend(args.checkpoint, tuple(detector["ports"]), tuple(detector["labels"]),
                              detector["preprocessing"], detector["image_size"])
        data = load_detection_stage(args.manifest, args.client, args.stage, detector["image_size"], detector["labels"], args.formal)
        cache = PackageCache(c["cache_bytes"], backend.specs)
        for file in detector.get("cached_packages", []):
            package = Package.read(Path(file).read_bytes())
            cache.put(package.hash, package.wire)
        fixed = c["fixed"]
        state = __import__("softcas.state", fromlist=["State"]).State.load(args.incumbent) if args.incumbent else initial_state(
            backend, cache, Structure(tuple(fixed["mask"]), fixed["ratio"]), FitSettings(fixed["lr"], fixed["residual"]), c["seed"])
        if any(k not in cache for k in state.C):
            raise ValueError("Missing exact incumbent dependencies")
        ledger = Ledger(Cost(**c["budgets"]["search"]), Cost(**c["budgets"]["release"]))
        exchange = ContactExchange(args.client, cache, ledger)
        evaluator = EvaluationCache()
        result = run_stage(backend, state, data, cache, exchange, ledger, c, evaluator=evaluator, owner=args.client)
        export_stage(result, args.output, backend, data, c, evaluator, cache, exchange)
        print(json.dumps({"output": args.output, "metric": backend.metric, "purpose": c["purpose"], "attempts": len(result.actions)}))
    else:
        summaries = []
        variants = [(m, "none") for m in METHODS] + [("softcas", a) for a in ABLATIONS if a != "none"]
        for method, ablation in variants:
            run_config = copy.deepcopy(c)
            run_config.update(method=method, ablation=ablation)
            summaries.append(demo(run_config, Path(args.output) / (method + "__" + ablation), args.stages, args.clients))
        Path(args.output, "suite.json").write_bytes(canonical(summaries))
        print(json.dumps({"variants": len(summaries), "purpose": c["purpose"], "formal_performance_result": False}))


if __name__ == "__main__":
    main()
