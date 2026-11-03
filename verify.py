"""Reproducible local verification and provenance capture; never edits sources."""
from __future__ import annotations
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
import torch
from softcas.cli import environment
from softcas.controller import Ridge
from softcas.state import State, Package, canonical
from softcas.tables import export_tables

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_suite(path):
    summaries = json.loads((path / "suite.json").read_bytes())
    assert len(summaries) == 12
    attempts, failures, stages, flags = 0, 0, 0, []
    for stage_file in sorted(path.glob("*/client*/stage*/stage.json")):
        directory = stage_file.parent
        info = json.loads(stage_file.read_bytes())
        assert info["purpose"] == "SYNTHETIC_BEHAVIOR_ONLY"
        assert info["metric"] == "synthetic_quality_NOT_AP"
        config = info["config"]
        rows = [json.loads(line) for line in (directory / "actions.jsonl").read_bytes().splitlines()]
        assert len(rows) <= config["attempts"]
        assert info["actual"]["search"]["E"] <= 1 + len(rows)
        assert info["audit"]["deployed"] == info["deployed"]
        assert State.load(directory / "checkpoints" / (info["deployed"] + ".json")).hash == info["deployed"]
        ridge = Ridge(len(config["features"]["fields"]), config["ridge_lambda"])
        for row in rows:
            assert len(row["features"]) == len(config["features"]["fields"])
            mu, s = ridge.predict(row["features"])
            assert math.isclose(mu, row["prediction"], abs_tol=1e-10, rel_tol=1e-8)
            assert math.isclose(s, row["coverage"], abs_tol=1e-10, rel_tol=1e-8)
            assert row["predictor_n_before"] == ridge.n
            if row["status"] == "failed":
                assert row["gain"] == 0 and row["child"] == row["parent"]
                failures += 1
            else:
                assert math.isclose(row["gain"], row["child_utility"] - row["parent_utility"], abs_tol=1e-12)
            ridge.update(row["features"], row["gain"])
        events = [json.loads(line) for line in (directory / "ledger.jsonl").read_bytes().splitlines()]
        for scope in ("search", "release"):
            for r in ("F", "E", "B", "W"):
                total = sum(e["cost"][r] for e in events if e["scope"] == scope)
                assert math.isclose(total, info["actual"][scope][r], rel_tol=1e-12, abs_tol=1e-9)
        for package_file in list((directory / "packages").glob("*.json")) + list((directory / "published").glob("*.json")):
            package = Package.read(package_file.read_bytes())
            assert package.hash == package_file.stem == sha(package_file)
            if package_file.parent.name == "published":
                assert package.owner == directory.parent.name.removeprefix("client")
        flags.extend(info["violations"][scope] for scope in ("search", "release") if info["violations"][scope])
        stages += 1
        attempts += len(rows)
    assert stages == 48
    return {"variants": 12, "client_stages": stages, "attempts": attempts, "observed_failures": failures,
            "ledger_reconciled": True, "ridge_replayed": True, "package_hashes_verified": True,
            "budget_violation_lists": flags, "suite_path": str(path)}


def provenance():
    """Hash the current implementation; do not overwrite historical snapshots."""
    paper = ROOT.parent / "output" / "CIRA_当前唯一版本_20261101" / "paper"
    references = [{"path": str(p), "sha256": sha(p)} for p in sorted(paper.rglob("*.tex"))] if paper.is_dir() else []
    code = {str(p.relative_to(ROOT)): sha(p) for pattern in ("softcas/*.py", "tests/*.py", "configs/*.json") for p in ROOT.glob(pattern)}
    for name in ("pyproject.toml", "requirements-lock.txt", "verify.py", "README.md", "PAPER_MAPPING.md"):
        code[name] = sha(ROOT / name)
    return {"references": references, "reference_available": bool(references), "implementation_sha256": code,
            "source_manuscript_edit_performed": False, "historical_interface_reports_used": False}


def main(argv=None):
    import argparse
    import re
    parser = argparse.ArgumentParser(description="Current CPU tests and optional explicit fixture replay; no training or historical report reuse")
    parser.add_argument("--output", required=True, help="A new or empty verification directory")
    parser.add_argument("--suite", help="Optional explicit SYNTHETIC_BEHAVIOR_ONLY fixture suite to replay")
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Choose a new/empty verification output; old evidence is preserved")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    process = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                             cwd=ROOT, capture_output=True, text=True)
    (output / "unit_tests.txt").write_text(process.stdout + process.stderr, encoding="utf-8")
    if process.returncode:
        print(process.stderr)
        raise SystemExit(process.returncode)
    match = re.search(r"Ran (\d+) tests", process.stdout + process.stderr)
    if not match:
        raise RuntimeError("Missing current test count")
    summary = {"scope": "CPU_SOFTWARE_BEHAVIOR_NOT_DETECTOR_OR_DEVICE_RESULTS",
               "unit_tests": {"count": int(match[1]), "passed": True},
               "suite": check_suite(Path(args.suite)) if args.suite else None,
               "environment": environment(), "formal_detection_result": False,
               "historical_interface_reports_used": False, "provenance": provenance()}
    (output / "verification.json").write_bytes(canonical(summary))
    print(json.dumps({"unit_tests": summary["unit_tests"], "explicit_suite_checked": bool(args.suite),
                      "formal_detection_result": False, "output": str(output)}))


if __name__ == "__main__":
    main()
