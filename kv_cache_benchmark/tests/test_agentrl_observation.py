"""Byte, policy and clock boundaries needed for reference-trace comparison."""

import time

import pytest
from test_agentrl import run, tiny

from kv_cache.agentrl import AgentRLConfig, SyncLifecycle


def test_io_arrivals_have_sizes_identity_and_distinct_actual_completion(tmp_path):
    runner, summary = run(tmp_path, tiny(iterations=1, network_mode="estimate", nic_rtt_us=2000))
    events = runner.trace.events
    begins = [e for e in events if e["event"] == "io_begin"]
    assert all(e["bytes"] > 0 for e in begins if e["op"] == "read")
    assert len({e["io_id"] for e in begins}) == len(begins)
    for begin in begins:
        matched = [e for e in events if e.get("io_id") == begin["io_id"]]
        assert [e["event"] for e in matched] == ["io_begin", "io_actual_end", "io_end"]
        actual, end = matched[1:]
        assert actual["bytes"] == begin["bytes"] == end["bytes"]
        assert end["t_s"] - actual["t_s"] >= 0.0018
        if end["op"] != "delete":
            assert end["physical_file_bytes"] > end["bytes"]
            assert end["backend_s"] <= end["actual_s"]
    assert summary["ranks"][0]["io_active_peak"] >= 1


def test_checkpoint_io_is_owned_by_saved_or_loaded_policy(tmp_path):
    runner, _ = run(tmp_path)
    assert [e["policy"] for e in runner.trace.events if e["event"] == "io_begin" and e["kind"] == "checkpoint"] == [
        1,
        2,
    ]
    manifest = runner.storage_dir / "checkpoints" / "step-1" / "manifest.json"
    resumed, _ = run(tmp_path / "resume", resume=manifest)
    load = next(e for e in resumed.trace.events if e["event"] == "io_begin" and e["op"] == "read")
    assert load["kind"] == "checkpoint" and load["policy"] == 1 and load["bytes"] == 128


def test_kv_lifetime_accounts_shared_prefix_once_and_excludes_startup(tmp_path):
    class SlowSetup(SyncLifecycle):
        def setup(self):
            time.sleep(0.04)
            super().setup()

    runner = SlowSetup(tiny(iterations=1), tmp_path / "data", tmp_path / "results")
    summary = runner.run()
    rank = summary["ranks"][0]
    assert rank["kv_live_payload_bytes"] == 0
    assert rank["kv_peak_payload_bytes"] == (4 + 4 + 8 + 4) * runner.config.bytes_per_token
    assert rank["prefix_hits"] == 2 and rank["prefix_misses"] == 1
    assert rank["elapsed_s"] - rank["iteration_elapsed_s"] > 0.04
    assert rank["generated_tokens"] == 16
    begin = next(e["t_s"] for e in runner.trace.events if e["event"] == "iteration_begin")
    end = next(e["t_s"] for e in runner.trace.events if e["event"] == "iteration_end")
    assert rank["iteration_elapsed_s"] == pytest.approx(end - begin)


def test_default_config_retains_v01_checkpoint_fingerprint():
    # Captured directly from fork revision 1573ea6, before profile/capacity fields.
    assert AgentRLConfig().fingerprint == "730cbd4d5532f15f6b4d44d76c8966c0e97e7565cf0b9b36cea0fea7c0b5c7de"
