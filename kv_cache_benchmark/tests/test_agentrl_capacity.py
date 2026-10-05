"""Concurrency hides storage waits without creating unlimited decode capacity."""

import asyncio

import pytest
from test_agentrl import run, tiny

from kv_cache.agentrl import SyncLifecycle
from kv_cache.backends import NVMeBackend


def test_admission_allocates_only_bounded_worker_tasks(tmp_path):
    class ObservedTasks(SyncLifecycle):
        tasks_peak = 0

        async def rollout(self, request, semaphore):
            self.tasks_peak = max(self.tasks_peak, len(asyncio.all_tasks()))
            await super().rollout(request, semaphore)

    config = tiny(iterations=1, requests_per_rank=40, concurrency=3, kv_offload_fraction=0, tokens_per_second=40000)
    runner = ObservedTasks(config, tmp_path / "data", tmp_path / "results")
    runner.run()
    assert runner.tasks_peak <= config.concurrency + 1
    starts = [e for e in runner.trace.events if e["event"] == "rollout_start"]
    assert len(starts) == 40
    assert starts[-1]["queue_wait_s"] > starts[0]["queue_wait_s"]


def test_owner_budget_caps_total_rate_but_request_mode_is_compatible(tmp_path):
    base = {
        "iterations": 1,
        "requests_per_rank": 4,
        "concurrency": 4,
        "prompt_tokens": 1,
        "response_tokens": [8],
        "chunk_tokens": 2,
        "tokens_per_second": 80,
        "kv_offload_fraction": 0,
        "persist_trajectories": False,
        "checkpoint_every": 0,
    }
    runner, summary = run(tmp_path / "owner", tiny(**base, generation_rate_scope="owner"))
    rank = summary["ranks"][0]
    assert rank["achieved_decode_tokens_per_s"] <= 81
    assert rank["compute_queue_wait_s"] > 0.15
    events = [e for e in runner.trace.events if e["event"] in ("generation_begin", "generation_end")]
    assert [e["event"] for e in events] == ["generation_begin", "generation_end"] * 16
    _, legacy = run(tmp_path / "request", tiny(**base))
    assert legacy["ranks"][0]["achieved_decode_tokens_per_s"] > 160


def test_owner_lock_is_recreated_across_iteration_and_recovery(tmp_path):
    runner, summary = run(tmp_path, tiny(generation_rate_scope="owner"))
    assert summary["final_policy_version"] == 2
    manifest = runner.storage_dir / "checkpoints" / "step-1" / "manifest.json"
    _, resumed = run(tmp_path / "resume", tiny(generation_rate_scope="owner"), resume=manifest)
    assert resumed["final_policy_version"] == 2


def test_cancelled_rollout_drains_active_counters_and_never_reports_success(tmp_path):
    class BrokenKV(NVMeBackend):
        def write(self, key, data):
            if self.base_path.name == "kv":
                raise OSError("injected KV failure")
            return super().write(key, data)

    runner = SyncLifecycle(tiny(), tmp_path / "data", tmp_path / "results", backend_factory=BrokenKV)
    with pytest.raises(RuntimeError, match="rollout"):
        runner.run()
    assert runner.active == runner.io_active == 0
    assert not list((tmp_path / "results").glob("*/summary.json"))


def test_invalid_generation_scope():
    with pytest.raises(ValueError, match="generation_rate_scope"):
        tiny(generation_rate_scope="unbounded")
