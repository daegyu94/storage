"""Exercise the real entry point and safely reject unsupported async launches."""

import json
from dataclasses import asdict, replace

import pytest
from test_agentrl_async import async_config
from test_agentrl_cli import invoke
from test_agentrl_trace import envelope, profile_record

from kv_cache.agentrl_trace import analyze_trace, load_trace, make_profile


@pytest.mark.parametrize("mode", ["colocate_async", "separate_async"])
def test_cli_executes_local_async_and_rejects_inflight_recovery(tmp_path, mode):
    config = asdict(replace(async_config(mode), iterations=2))
    process = invoke(tmp_path, config, extra=("--mpi",))
    assert process.returncode == 0, process.stderr
    path = next((tmp_path / "results").glob("*/summary.json"))
    summary = json.loads(path.read_text())
    assert summary["trainer_mode"] == mode and summary["async"]["sampled_groups"] == 2
    assert summary["topology"]["distributed_async"] is False
    assert not analyze_trace(load_trace(path.parent))["violations"]
    manifest = next((tmp_path / "data").glob("*/checkpoints/step-1/manifest.json"))
    resumed = invoke(tmp_path / "resume", config, extra=("--resume", str(manifest)))
    assert resumed.returncode != 0 and "async resume" in resumed.stderr
    assert not (tmp_path / "resume/data").exists()


def test_mpi_async_rejects_role_protocol_before_creating_files(tmp_path):
    process = invoke(tmp_path, asdict(async_config("separate_async")), ranks=2)
    assert process.returncode != 0 and "multi-rank async" in process.stderr
    assert not (tmp_path / "data").exists()


def test_cli_async_accepts_group_aligned_joint_profile_without_validation_claim(tmp_path):
    profile = make_profile(envelope([profile_record(), profile_record()], mode="separate_async"))
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    config = asdict(replace(async_config("separate_async"), iterations=2))
    config.pop("request_profile")
    process = invoke(tmp_path, config, extra=("--profile", str(path)))
    assert process.returncode == 0, process.stderr
    summary_path = next((tmp_path / "results").glob("*/summary.json"))
    summary = json.loads(summary_path.read_text())
    assert summary["fidelity"] == "uncalibrated"
    observed = load_trace(summary_path.parent)
    assert observed["provenance"]["calibration_sha256"] == profile["provenance"]["source_sha256"]
    assert {r["trajectory_bytes"] for r in observed["requests"]} == {257}
