"""Run genuine residual learning on CPU toy data (not detector evidence)."""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(output, clients=4, stages=2, seed=1001):
    import torch
    from softcas.backend import Backend, work
    from softcas.cache import ContactExchange, PackageCache
    from softcas.cli import environment
    from softcas.config import load_config, validate_config
    from softcas.data import StageData, synthetic_split
    from softcas.decision_audit import audit_files
    from softcas.engine import initial_state, run_stage, export_stage
    from softcas.evaluation import EvaluationCache
    from softcas.reporting import export_matrix
    from softcas.resources import Cost, Ledger, TransferJournal
    from softcas.state import Structure, FitSettings, canonical
    from softcas.toy_learning import TeacherBackend, fitting_probe, teacher_split
    from run_cira_demo import PAGE

    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Use a new or empty output directory")
    if not 1 <= clients <= 8 or not 1 <= stages <= 4:
        raise ValueError("Supported demo scale: 1-8 clients, 1-4 stages")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    config = load_config(ROOT / "configs/fixture.json")
    config.update(seed=seed, proposal_protocol="native_r1000", attempts=12,
                  archive_capacity=8, optimizer={"name": "AdamW", "weight_decay": 0.0},
                  fitting={"unit": "epochs", "stage_epochs": 12,
                           "local_epochs": 4, "receiving_epochs": 2})
    validate_config(config)
    (output / "demo_config.json").write_bytes(canonical(config))

    # Diagnose the existing fixture at its initial local block without changing
    # that fixture, its records, thresholds, gain labels or numeric precision.
    old_backend = Backend()
    old_cache = PackageCache(config["cache_bytes"], old_backend.specs)
    old_state = initial_state(old_backend, old_cache, Structure((1, 0, 0, 0, 0, 0), 1/8),
                              FitSettings(1e-3, 1e-3), seed)
    _, old_probe = fitting_probe(old_backend, old_state, old_cache,
        synthetic_split("D", "0", 1, "0", seed + 100),
        synthetic_split("V", "0", 1, "0", seed + 101), steps=4)
    old_probe["description"] = ("Controlled initial-block probe of the legacy TinyBase fixture, "
        "using its SGD, learning rate and four-step block. This is not a replay of all 51 actions.")

    backend = TeacherBackend()
    probe_cache = PackageCache(config["cache_bytes"], backend.specs)
    probe_state = initial_state(backend, probe_cache, Structure((1, 0, 0, 0, 0, 0), 1/8),
                                FitSettings(1e-3, 1e-4), 42)
    probe_v = teacher_split("V", "probe", 1, "0", 500)
    probes = {"legacy_initial_block": old_probe}
    for direction, name in ((1.0, "aligned_teacher"), (-1.0, "opposed_D_teacher")):
        _, probes[name] = fitting_probe(backend, probe_state, probe_cache,
            teacher_split("D", "probe", 1, "0", 200, target_direction=direction),
            probe_v, steps=32, optimizer_settings={"name": "AdamW", "weight_decay": 0.0})
    # Zero is a score identity check, not a fitting outcome manufactured to
    # complete three categories of results.
    probe_model = backend.assembly(probe_state, probe_cache, mode="import")
    score_a, score_b = backend.score(probe_model, probe_v), backend.score(probe_model, probe_v)
    probes["unchanged_state_repeat"] = {"score_a": score_a, "score_b": score_b,
                                         "difference": score_b - score_a}
    (output / "learning_probes.json").write_bytes(canonical(probes))
    if not (probes["aligned_teacher"]["V_gain"] > 0 and
            probes["opposed_D_teacher"]["V_gain"] < 0 and score_a == score_b):
        raise AssertionError("Real learning/negative-transfer/identity probes did not hold")

    caches = [PackageCache(config["cache_bytes"], backend.specs) for _ in range(clients)]
    fixed = config["fixed"]
    states = [initial_state(backend, cache, Structure(tuple(fixed["mask"]), fixed["ratio"]),
              FitSettings(fixed["lr"], fixed["residual"]), seed) for cache in caches]
    evaluators = [EvaluationCache() for _ in range(clients)]
    published = [{} for _ in range(clients)]
    source_ledgers = [Ledger(Cost(1e30, 1e30, 1e30, 1e30), Cost(), enforce=()) for _ in range(clients)]
    offline = Ledger(Cost(1e30, 1e30, 1e30, 1e30), Cost(), enforce=())
    journal, rows, runs = TransferJournal(), [], []
    orders = []
    for client in range(clients):
        order = list(range(stages))
        random.Random(seed + client).shuffle(order)
        orders.append(order)
    for stage in range(1, stages + 1):
        for client in range(clients):
            domain = str(orders[client][stage - 1])
            data = StageData(*(teacher_split(role, client, stage, domain,
                seed + 100 * stage + 1000 * client + j)
                for j, role in enumerate(("D", "V", "A"))))
            ledger = Ledger(Cost(**config["budgets"]["search"]), Cost(**config["budgets"]["release"]))
            exchange = ContactExchange(str(client), caches[client], ledger, journal)
            for peer in range(clients):
                if peer != client:
                    exchange.observe(str(peer), published[peer], time.time(), config["contact_quota"], source_ledgers[peer])
            result = run_stage(backend, states[client], data, caches[client], exchange, ledger,
                copy.deepcopy(config), retained=(states[client],), evaluator=evaluators[client], owner=str(client))
            states[client] = result.deployed
            for package in result.published:
                published[client][package.hash] = package.wire
            path = output / "run" / f"client{client}" / f"stage{stage}"
            export_stage(result, path, backend, data, config, evaluators[client], caches[client], exchange)
            info = json.loads((path / "stage.json").read_bytes())
            info["deployed_external_paths"] = len(result.deployed.C)
            ledger_rows = [json.loads(line) for line in (path / "ledger.jsonl").read_bytes().splitlines()]
            for scope in ("search", "release"):
                for dimension in ("F", "E", "B", "W"):
                    actual = sum(row["cost"][dimension] for row in ledger_rows if row["scope"] == scope)
                    if not math.isclose(actual, info["actual"][scope][dimension], rel_tol=1e-12, abs_tol=1e-9):
                        raise AssertionError("Ledger does not reconcile")
            if info["actual"]["search"]["E"] > 1 + len(result.actions):
                raise AssertionError("Unexpected selection evaluation count")
            runs.append({"client": str(client), "stage": stage, "actions": result.actions,
                         "raw": (path / "actions.jsonl").relative_to(output).as_posix(), "info": info})
            for past in orders[client][:stage]:
                test = teacher_split("T", client, 0, str(past), seed + 90000 + 100 * client + past)
                model = backend.assembly(result.deployed, caches[client])
                with work(offline, "independent_teacher_test"):
                    score = backend.score(model, test)
                rows.append({"seed": seed, "client": str(client), "stage": stage,
                    "current_domain": domain, "domain": str(past), "deployed": result.deployed.hash,
                    "test_manifest": test.identity, "first_stage": orders[client].index(past) + 1,
                    "score": score, "metric": backend.metric, "purpose": "SYNTHETIC_BEHAVIOR_ONLY"})
    endpoints = export_matrix(output / "run", rows, {str(i) for i in range(clients)})
    decision_audit = audit_files(sorted((output / "run").glob("client*/stage*/actions.jsonl")))
    audit_stages = decision_audit["stages"]
    observed = sum(row["observed_targets"] for row in audit_stages)
    audit_summary = {"actions": decision_audit["actions"],
        "nonzero_predictions": sum(row["nonzero_predictions"] for row in audit_stages),
        "priority_decisions": decision_audit["priority_decisions"],
        "changed_priority_decisions": decision_audit["changed_priority_decisions"],
        "feature_collision_groups": sum(len(row["feature_collision_groups"]) for row in audit_stages),
        "negative_retained": sum(row["negative_retained"] for row in audit_stages),
        "negative_reused_as_parent": sum(row["negative_reused_as_parent"] for row in audit_stages),
        "all_selected_replays_match": all(row["all_selected_replays_match"] is not False for row in audit_stages)}
    for key in ("prediction_mae", "zero_baseline_mae", "past_mean_baseline_mae"):
        audit_summary[key] = (sum(row[key] * row["observed_targets"] for row in audit_stages
                                  if row[key] is not None) / observed if observed else None)
    decision_audit["summary"] = audit_summary
    (output / "decision_audit.json").write_bytes(canonical(decision_audit))
    actions = [a for r in runs for a in r["actions"]]
    gains = [a["gain"] for a in actions if a["status"] != "failed"]
    summary = {"purpose": "SYNTHETIC_BEHAVIOR_ONLY", "metric": backend.metric,
        "backend": "identity_port_teacher_v1", "formal_performance_result": False,
        "attempts": len(actions), "published_versions": sum(map(len, published)),
        "events": len(runs), "network_tx_bytes": journal.network_bytes,
        "gain_signs": dict(Counter("positive" if g > 0 else "negative" if g < 0 else "zero" for g in gains)),
        "nonzero_predictions": sum(a["prediction"] != 0 for a in actions),
        "decision_audit": audit_summary,
        "kinds": dict(Counter(a["kind"] for a in actions)),
        "endpoints": endpoints, "environment": environment(),
        "source_serving": [asdict(l.used["search"]) for l in source_ledgers],
        "offline_test_cost_partial": asdict(offline.used["search"])}
    if not any(g != 0 for g in gains):
        raise AssertionError("The teacher loop produced no nonzero feedback")
    (output / "run" / "summary.json").write_bytes(canonical(summary))
    (output / "run" / "network_transfers.json").write_bytes(canonical(journal.records))
    edges = {}
    for record in journal.records.values():
        key = record["sender"], record["receiver"]
        edges[key] = edges.get(key, 0) + record["bytes"]
    if sum(edges.values()) != journal.network_bytes:
        raise AssertionError("Transfer byte sum does not reconcile")
    command = (f".\\.venv\\Scripts\\python.exe examples/run_learning_demo.py "
               f"--output examples_output/new_learning_demo --clients {clients} --stages {stages} --seed {seed}")
    data = {"clients": clients, "stages": stages, "summary": summary, "config": config,
        "runs": runs, "command": command,
        "edges": [{"sender": a, "receiver": b, "bytes": n} for (a, b), n in edges.items()]}
    (output / "demo_data.json").write_bytes(canonical(data))
    page = PAGE.replace("TinyBase", "六个恒等端口的教师学生网络").replace(
        "小型网络＋合成数据", "恒等端口网络＋固定教师生成的合成数据").replace(
        "每个D批次有2个合成样本。这里每个epoch只有1个批次", "每个D批次有4个合成样本。这里每个epoch有2个批次")
    page = page.replace('href="README_示例.md"', 'href="README_学习示例.md"')
    # Existing viewer counts fit batches; stage budget is epochs × D batches.
    page = page.replace("cap=D.config.fitting.stage_epochs;", "cap=D.config.fitting.stage_epochs*2;")
    diagnostic_html = ("<section class=\"box\"><h2>学习与负迁移探针</h2>"
        "<p>固定教师与学生目标一致时，真实拟合提高独立V分数；D教师方向反转时，V分数下降。"
        "这是受控教学探针，不代表论文性能或控制器优越性。</p><table><tr><th>探针</th><th>V增益</th><th>参数变化 L2</th></tr>" +
        "".join(f"<tr><td>{name}</td><td>{probes[name]['V_gain']:.8g}</td>"
                f"<td>{probes[name]['local_parameter_delta_l2']:.8g}</td></tr>"
                for name in ("legacy_initial_block", "aligned_teacher", "opposed_D_teacher")) +
        "</table><a href=\"learning_probes.json\">梯度、损失、逐步更新及分数原始记录</a></section>")
    diagnostic_html += ("<section class=\"box\"><h2>预测器是否改变了选择</h2>"
        f"<p>{audit_summary['nonzero_predictions']} 次非零预测；"
        f"{audit_summary['priority_decisions']} 次按优先级决策中，"
        f"同一候选池的Full与No-gain排序有 {audit_summary['changed_priority_decisions']} 次改选。"
        "改选不是性能提升证据；未选候选没有假造结果。</p>"
        "<table><tr><th>已观察子状态增益的预测</th><th>MAE</th></tr>" +
        "".join(f"<tr><td>{label}</td><td>{audit_summary[key]:.8g}</td></tr>"
                for key, label in (("prediction_mae", "Ridge"),
                                   ("zero_baseline_mae", "恒为0"),
                                   ("past_mean_baseline_mae", "阶段内过去增益均值"))) +
        "</table>"
        f"<p>池内特征碰撞组：{audit_summary['feature_collision_groups']}；"
        f"负增益完成状态保留 {audit_summary['negative_retained']}，"
        f"其中后续用作父状态 {audit_summary['negative_reused_as_parent']}。"
        "特征碰撞表示不同提案有相同编码，仍需原生数据检验其影响。</p>"
        "<a href=\"decision_audit.json\">逐阶段预测与同池排序审计</a></section>")
    page = page.replace("<footer>", diagnostic_html + "<footer>")
    (output / "打开学习示例.html").write_text(page.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/")), encoding="utf-8")
    with (output / "逐步动作.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ("client", "stage", "attempt", "kind", "decision", "fit_steps", "score", "gain", "prediction")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in runs:
            for action in r["actions"]:
                writer.writerow({k: r[k] if k in ("client", "stage") else action[k] for k in fields})
    verification = {"scope": "CPU_SYNTHETIC_BEHAVIOR_ONLY_NOT_AP", "actions": len(actions),
        "nonzero_feedback_checked": True, "heldout_probe_signs_checked": True,
        "ledger_reconciled": True, "network_bytes_reconciled": True,
        "decision_audit": audit_summary,
        "frozen_base_checked": all(probes[n]["base_unchanged"] for n in
            ("legacy_initial_block", "aligned_teacher", "opposed_D_teacher")),
        "stage_fit_caps_checked": all(a["stage_fit_steps"] <= a["stage_fit_step_limit"] for a in actions),
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), ROOT / "softcas/toy_learning.py")}}
    (output / "verification.json").write_bytes(canonical(verification))
    (output / "README_学习示例.md").write_text(
        "# 可观察学习反馈的CPU示例\n\n"
        "打开 `打开学习示例.html`。旧零增益示例完整保留。本示例用固定教师生成合成目标，"
        "六端口Assembly、Adapter、拟合、档案、ridge和审计均调用已有代码。"
        "分数不是AP，四客户端是单CPU顺序逻辑节点。\n\n"
        "`learning_probes.json`记录实际梯度、参数更新、D损失和V分数。"
        "旧TinyBase初始块探针只是定位分辨率与灵敏度，不是重放所有旧动作。"
        "相反D教师用于独立负迁移探针，不注入在线增益。"
        "教师示例证明优化和反馈管线可响应，不证明CIRA优于其他选择策略。\n\n"
        f"```powershell\n{command}\n```\n\n"
        "请使用新输出目录；每阶段12epoch，每epoch两个D批次，共24步，非论文训练规模。"
        "训练和前向float32，教师示例V的平方误差用float64累加。"
        "CPU时间和部分FLOPs不是Jetson标定。\n", encoding="utf-8")
    print(json.dumps({"entry": str(output / "打开学习示例.html"), **summary}, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1001)
    args = parser.parse_args()
    run(args.output, args.clients, args.stages, args.seed)
