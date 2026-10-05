"""The trace pipeline is runnable without GPU libraries."""

import json
import subprocess
import sys
from pathlib import Path

from test_agentrl_trace import envelope

ROOT = Path(__file__).resolve().parents[1]


def invoke(*args):
    return subprocess.run(
        [sys.executable, str(ROOT / "agent-rl-trace.py"), *map(str, args)], capture_output=True, text=True, timeout=15
    )


def test_profile_and_compare_cli(tmp_path):
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    reference.write_text(json.dumps(envelope()))
    data = envelope()
    data["provenance"]["run_id"] = "holdout"
    candidate.write_text(json.dumps(data))
    profile, report = tmp_path / "profile.json", tmp_path / "comparison.json"
    result = invoke("profile", "--input", reference, "--output", profile)
    assert result.returncode == 0, result.stderr
    assert json.loads(profile.read_text())["records"]
    result = invoke("compare", "--reference", reference, "--candidate", candidate, "--output", report)
    assert result.returncode == 0, result.stderr
    assert json.loads(report.read_text())["status"] == "within_tolerances"
    reference.write_text("{}")
    result = invoke("profile", "--input", reference, "--output", profile)
    assert result.returncode != 0 and "trace" in result.stderr


def test_runner_accepts_profile_flag(tmp_path):
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps(envelope()))
    profile = tmp_path / "profile.json"
    result = invoke("profile", "--input", reference, "--output", profile)
    assert result.returncode == 0, result.stderr
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"iterations": 1, "requests_per_rank": 1, "kv_offload_fraction": 1.0}))
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "agent-rl.py"),
            "--config",
            str(config),
            "--profile",
            str(profile),
            "--storage-root",
            str(tmp_path / "data"),
            "--results-dir",
            str(tmp_path / "results"),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    trace = json.loads(next((tmp_path / "results").glob("*/trace-rank-0.json")).read_text())
    assert trace["requests"][0]["trajectory_bytes"] == 257
