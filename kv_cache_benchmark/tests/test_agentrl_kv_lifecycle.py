"""Causal lifecycle integration of working-set capacity with real storage."""

import json
import time
from dataclasses import asdict, replace

import pytest
from test_agentrl import tiny
from test_agentrl_async import async_config
from test_agentrl_cli import invoke

from kv_cache.agentrl import AgentRLConfig, create_lifecycle
from kv_cache.agentrl_trace import analyze_trace, load_trace
from kv_cache.backends import NVMeBackend


def configured(mode="sync", capacity=768, **fields):
    base = tiny() if mode == "sync" else async_config(mode, kv_gc_delay_s=0.01)
    changes = {
        "iterations": 2,
        "requests_per_rank": 4,
        "concurrency": 4,
        "response_tokens": [6, 6],
        "turns": 2,
        "tool_delay_s": 0.035,
        "observation_tokens": 2,
        "kv_offload_fraction": 0,
        "kv_cache_model": {"capacity_bytes": capacity, "block_tokens": 2},
        **fields,
    }
    return replace(base, **changes)


def execute(tmp_path, config, **kwargs):
    runner = create_lifecycle(config, tmp_path / "data", tmp_path / "results", **kwargs)
    summary = runner.run()
    return summary, load_trace(runner.result_dir), runner


@pytest.mark.parametrize("mode", ["sync", "colocate_async", "separate_async"])
def test_capacity_creates_actual_offload_and_resume_reads_in_each_mode(tmp_path, mode):
    summary, trace, runner = execute(tmp_path, configured(mode))
    writes = [e for e in trace["events"] if e["event"] == "io_end" and e["kind"] == "kv" and e["op"] == "write"]
    reads = [e for e in trace["events"] if e["event"] == "io_end" and e["kind"] == "kv" and e["op"] == "read"]
    assert writes and reads
    assert all(e["bytes"] <= 2 * runner.config.bytes_per_token for e in writes + reads)
    cache = summary["ranks"][0]["kv_cache_model"]["owners"]
    assert all(c["resident_peak_bytes"] <= c["capacity_bytes"] for c in cache.values())
    assert all(c["resident_bytes"] == c["pinned_blocks"] == 0 for c in cache.values())
    assert summary["fidelity"] == "uncalibrated"
    assert not analyze_trace(trace)["violations"]
    assert not list((tmp_path / "data").glob("**/kv/*.npy"))
    assert any(
        tool["request"] == read["request"] and tool["t_s"] < read["t_s"]
        for tool in trace["events"]
        if tool["event"] == "tool_end"
        for read in reads
    )


@pytest.mark.parametrize("mode", ["sync", "colocate_async", "separate_async"])
def test_large_resident_capacity_avoids_fraction_style_disk_kv(tmp_path, mode):
    summary, _, _ = execute(tmp_path, configured(mode, capacity=64 * 1024))
    assert "kv_write" not in summary["io_totals"] and "kv_read" not in summary["io_totals"]
    assert summary["io_totals"]["trajectory_write"]["ops"] > 0
    assert summary["final_policy_version"] == 2


def test_storage_service_delay_flows_back_to_generation_and_checkpoint(tmp_path):
    class Slow(NVMeBackend):
        def write(self, key, data):
            if self.base_path.name == "kv":
                time.sleep(0.025)
            return super().write(key, data)

    config = configured(iterations=1)
    _, base, _ = execute(tmp_path / "base", config)
    _, slow, _ = execute(tmp_path / "slow", config, backend_factory=Slow)

    def checkpoint(trace):
        return next(e["t_s"] for e in trace["events"] if e["event"] == "checkpoint_begin")

    assert checkpoint(slow) - checkpoint(base) > 0.05


def test_partial_policy_transition_invalidates_logical_entries_without_old_reload(tmp_path):
    config = replace(configured("colocate_async", capacity=4096), iterations=5, response_tokens=[2, 2, 30, 30])
    summary, trace, _ = execute(tmp_path, config)
    assert any(e["event"] == "re_prefill_end" for e in trace["events"])
    for invalid in (e for e in trace["events"] if e["event"] == "cache_logical_invalidate"):
        assert not any(
            e["event"] == "io_begin" and e["op"] == "read" and e["key"] in invalid["keys"] and e["t_s"] > invalid["t_s"]
            for e in trace["events"]
        )
    assert all(c["resident_bytes"] == 0 for c in summary["ranks"][0]["kv_cache_model"]["owners"].values())


@pytest.mark.parametrize(
    "field,value", [("capacity_bytes", 0), ("capacity_bytes", True), ("block_tokens", -1), ("unknown", 1)]
)
def test_invalid_cache_settings_are_rejected(field, value):
    config = asdict(configured())
    config["kv_cache_model"][field] = value
    with pytest.raises(ValueError, match="cache|kv_cache"):
        AgentRLConfig.from_dict(config)


def test_capacity_requires_full_attention_fit_and_explicit_offload_choice():
    with pytest.raises(ValueError, match="working set"):
        AgentRLConfig.from_dict(asdict(configured(capacity=128)))
    with pytest.raises(ValueError, match="fraction"):
        AgentRLConfig.from_dict(asdict(configured(kv_offload_fraction=0.5)))


def test_cache_block_layout_allows_large_prompt_footprint_without_large_payload(tmp_path):
    config = configured(capacity=6912, prompt_tokens=100, max_payload_bytes=2048)
    # Full prompt KV is 6400 bytes, but each actual file block is at most 128.
    config.validate()
    summary, _, _ = execute(tmp_path, config)
    assert summary["final_policy_version"] == 2
    assert summary["io_totals"]["kv_write"]["ops"] > 0


def test_sync_mpi_preserves_each_rank_cache_and_recovery_boundary(tmp_path):
    config = configured()
    process = invoke(tmp_path, asdict(config), ranks=2)
    assert process.returncode == 0, process.stderr
    summary = json.loads(next((tmp_path / "results").glob("*/summary.json")).read_text())
    for rank in (0, 1):
        pools = summary["ranks"][rank]["kv_cache_model"]["owners"]
        assert set(pools) == {str(rank)}
        assert pools[str(rank)]["resident_bytes"] == 0
    manifest = next((tmp_path / "data").glob("*/checkpoints/step-1/manifest.json"))
    restored = invoke(tmp_path / "resume", asdict(config), ranks=2, extra=("--resume", str(manifest)))
    assert restored.returncode == 0, restored.stderr


def test_separated_mpi_uses_only_worker_cache_pools(tmp_path):
    base = configured("separate_async")
    config = replace(base, async_workload={**base.async_workload, "execution": "mpi_shared", "rollout_owners": 2})
    process = invoke(tmp_path, asdict(config), ranks=3)
    assert process.returncode == 0, process.stderr
    summary = json.loads(next((tmp_path / "results").glob("*/summary.json")).read_text())
    assert summary["ranks"][0]["kv_cache_model"]["owners"] == {}
    for rank in (1, 2):
        pools = summary["ranks"][rank]["kv_cache_model"]["owners"]
        assert set(pools) == {str(rank - 1)}
        assert pools[str(rank - 1)]["resident_bytes"] == 0


def test_trace_declares_cache_assumption_for_future_calibration(tmp_path):
    config = configured(iterations=1)
    _, trace, _ = execute(tmp_path, config)
    assert trace["kv_cache_model"]["settings"] == config.kv_cache_model
    assert trace["kv_cache_model"]["fidelity"] == "uncalibrated"
    assert trace["kv_cache_model"]["residency_model"] == "chunk_safe_working_set"


@pytest.mark.parametrize("mode", ["sync", "colocate_async", "separate_async"])
def test_eviction_failure_suppresses_success_summary(tmp_path, mode):
    class Broken(NVMeBackend):
        def write(self, key, data):
            if self.base_path.name == "kv":
                raise OSError("injected eviction failure")
            return super().write(key, data)

    with pytest.raises(Exception, match="injected|sub-exception"):
        execute(tmp_path, configured(mode), backend_factory=Broken)
    assert not list((tmp_path / "results").glob("*/summary.json"))
