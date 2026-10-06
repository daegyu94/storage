"""CPU primary staging and unbounded filesystem secondary contracts."""

import asyncio
import json
from dataclasses import asdict

import numpy as np
import pytest
from test_agentrl import tiny
from test_agentrl_cli import invoke
from test_agentrl_kv_lifecycle import configured, execute
from test_agentrl_reference import reference

from kv_cache.agentrl import AgentRLConfig
from kv_cache.backends import NVMeBackend


def tier(tmp_path, *, pages=1, fs=True, hold=None, fail=None, background=False, workers=16):
    from kv_cache.agentrl_tiering import CPUPrimary, TierSettings

    backend = NVMeBackend(str(tmp_path))
    operations, events = [], []

    async def io(op, key, size, **context):
        if hold:
            await hold(op, key)
        if op == fail:
            raise OSError("injected secondary failure")
        if op == "write":
            backend.write(key, np.arange(size, dtype=np.uint8))
        else:
            data, _ = backend.read(key)
            assert data.nbytes == size
        operations.append((op, key, size))

    values = {"cpu_capacity_bytes": pages * 32, "fs_enabled": fs}
    if background:
        values.update(fs_execution="background", fs_write_workers=workers)
    settings = TierSettings.parse(values)
    cpu = CPUPrimary(settings, 32, io, lambda name, **data: events.append({"event": name, **data}))
    return cpu, backend, operations, events


def test_completed_store_cascades_before_cpu_eviction_and_cpu_hit_avoids_fs(tmp_path):
    cpu, _, io, _ = tier(tmp_path)

    async def scenario():
        await cpu.store("a")
        assert io == [("write", "a", 32)]
        assert await cpu.load("a")
        assert io == [("write", "a", 32)]
        assert cpu.summary()["cpu_hits"] == 1

    asyncio.run(scenario())


def test_cpu_miss_promotes_from_fs_and_eviction_keeps_unbounded_secondary(tmp_path):
    cpu, _, io, _ = tier(tmp_path)

    async def scenario():
        for key in ("a", "b", "c"):
            await cpu.store(key)
        assert await cpu.load("a")
        assert io == [("write", k, 32) for k in ("a", "b", "c")] + [("read", "a", 32)]
        stats = cpu.summary()
        assert stats["cpu_reserved_bytes"] == stats["cpu_peak_bytes"] == 32
        assert stats["fs_valid_payload_bytes"] == 96
        assert not stats["fs_capacity_bounded"] and stats["fs_capacity_evictions"] == 0
        assert stats["cpu_promotions"] == 1

    asyncio.run(scenario())


def test_cpu_only_has_no_filesystem_io_and_missing_copy_returns_miss(tmp_path):
    cpu, _, io, _ = tier(tmp_path, fs=False)

    async def scenario():
        await cpu.store("a")
        await cpu.store("b")
        assert not await cpu.load("a")
        assert await cpu.load("b")
        assert io == []
        assert cpu.summary()["fs_valid_payload_bytes"] == 0

    asyncio.run(scenario())


def test_small_cpu_changes_reads_but_completed_store_volume_is_write_through(tmp_path):
    async def scenario(pages):
        cpu, _, io, _ = tier(tmp_path / str(pages), pages=pages)
        for key in ("a", "b"):
            await cpu.store(key)
        for key in ("a", "b"):
            assert await cpu.load(key)
        return io

    small, large = asyncio.run(scenario(1)), asyncio.run(scenario(2))
    assert sum(op == "write" for op, _, _ in small) == sum(op == "write" for op, _, _ in large) == 2
    assert sum(op == "read" for op, _, _ in small) == 2
    assert sum(op == "read" for op, _, _ in large) == 0


def test_cpu_pin_stalls_new_store_until_secondary_finishes(tmp_path):
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()

        async def hold(op, key):
            if op == "write" and key == "a":
                started.set()
                await release.wait()

        cpu, _, io, _ = tier(tmp_path, hold=hold)
        a = asyncio.create_task(cpu.store("a"))
        await started.wait()
        b = asyncio.create_task(cpu.store("b"))
        await asyncio.sleep(0.005)
        assert not b.done() and cpu.summary()["cpu_pinned_pages"] == 1
        assert cpu.summary()["cpu_reserved_bytes"] == 32 and not io
        release.set()
        await asyncio.wait_for(asyncio.gather(a, b), 1)
        assert cpu.summary()["cpu_wait_events"] > 0
        assert cpu.summary()["cpu_pinned_pages"] == 0

    asyncio.run(scenario())


def test_duplicate_store_and_promotion_share_one_transfer(tmp_path):
    async def scenario():
        cpu, _, io, _ = tier(tmp_path)
        await asyncio.gather(cpu.store("a"), cpu.store("a"))
        await cpu.store("b")
        await asyncio.gather(cpu.load("a"), cpu.load("a"))
        assert io.count(("write", "a", 32)) == io.count(("read", "a", 32)) == 1

    asyncio.run(scenario())


def test_policy_invalidation_forbids_old_fs_lookup_before_physical_gc(tmp_path):
    cpu, backend, io, _ = tier(tmp_path)

    async def scenario():
        await cpu.store("v0-a")
        cpu.invalidate()
        assert backend._get_path("v0-a").exists()
        assert not await cpu.load("v0-a")
        assert io == [("write", "v0-a", 32)]
        assert cpu.summary()["cpu_reserved_bytes"] == cpu.summary()["fs_valid_payload_bytes"] == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("op", ["write", "read"])
def test_secondary_failure_releases_cpu_pins_and_does_not_promote_false_data(tmp_path, op):
    async def scenario():
        cpu, _, _, _ = tier(tmp_path, fail="write" if op == "write" else None)
        if op == "read":
            await cpu.store("a")
            await cpu.store("b")

            async def broken(*args, **kwargs):
                raise OSError("injected secondary failure")

            cpu.io = broken
        with pytest.raises(OSError, match="secondary"):
            await (cpu.store("a") if op == "write" else cpu.load("a"))
        assert cpu.summary()["cpu_pinned_pages"] == 0
        if op == "read":
            assert "a" not in cpu.entries
            with pytest.raises(OSError, match="secondary"):
                await cpu.load("a")
            assert cpu.summary()["cpu_promotions"] == 0
        else:
            assert "a" not in cpu.fs_keys

    asyncio.run(scenario())


def test_cancelled_cascade_drains_io_before_unpinning_cpu(tmp_path):
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()

        async def hold(op, key):
            started.set()
            await release.wait()

        cpu, _, io, _ = tier(tmp_path, hold=hold)
        task = asyncio.create_task(cpu.store("a"))
        await started.wait()
        task.cancel()
        await asyncio.sleep(0.005)
        assert not task.done() and cpu.summary()["cpu_pinned_pages"] == 1
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert io == [("write", "a", 32)]
        assert cpu.summary()["cpu_pinned_pages"] == 0

    asyncio.run(scenario())


def test_policy_invalidation_rejects_active_cpu_transfer(tmp_path):
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()

        async def hold(op, key):
            started.set()
            await release.wait()

        cpu, _, _, _ = tier(tmp_path, hold=hold)
        task = asyncio.create_task(cpu.store("a"))
        await started.wait()
        with pytest.raises(ValueError, match="active CPU"):
            cpu.invalidate()
        release.set()
        await task
        cpu.invalidate()
        assert not cpu.fs_keys and not cpu.entries

    asyncio.run(scenario())


def test_completed_block_publish_requires_pinned_gpu_lease(tmp_path):
    from kv_cache.agentrl import create_lifecycle

    async def scenario():
        runner = create_lifecycle(config(), tmp_path / "data", tmp_path / "results")
        pool = runner.make_cache_pool(0, NVMeBackend(str(tmp_path / "kv")))
        async with pool.lease(0, "r", "p", 4, 4, 0, iteration=0) as lease:
            pool.materialize(lease, 4)
        with pytest.raises(ValueError, match="pinned lease"):
            await pool.publish(lease, 4, prompt_tokens=4)

    asyncio.run(scenario())


def test_store_cursor_does_not_recopy_old_prompt_each_decode_step(tmp_path):
    from kv_cache.agentrl import create_lifecycle

    async def scenario():
        runner = create_lifecycle(config(cpu_capacity_bytes=256), tmp_path / "data", tmp_path / "results")
        pool = runner.make_cache_pool(0, NVMeBackend(str(tmp_path / "kv")))
        async with pool.lease(0, "r", "p", 4, 4, 2, iteration=0) as lease:
            pool.materialize(lease, 4)
            await pool.publish(lease, 4, prompt_tokens=4)
            assert pool.tiers.stores == 2
            pool.materialize(lease, 6)
            await pool.publish(lease, 6, prompt_tokens=4)
            assert pool.tiers.stores == 2
            # New tool call reconsiders the longer prompt, matching a new
            # native generation request rather than repeated decode stores.
            await pool.publish(lease, 6, prompt_tokens=6)
            assert pool.tiers.stores == 5
            assert pool.tiers.fs_writes == 3
        await pool.release_request("r")
        assert not pool.published

    asyncio.run(scenario())


def config(mode="sync", **tier_fields):
    raw = asdict(configured(mode, iterations=2))
    raw["rollout_reference"] = reference(2, kv_budget_bytes_per_gpu=768)
    raw["kv_cache_model"].pop("capacity_bytes")
    raw["kv_offload_tiers"] = {"cpu_capacity_bytes": 512, **tier_fields}
    return AgentRLConfig.from_dict(raw)


@pytest.mark.parametrize(
    "fields",
    [
        {"cpu_capacity_bytes": 0},
        {"cpu_capacity_bytes": True},
        {"cpu_capacity_bytes": 255},
        {"fs_capacity_bytes": 1024},
        {"fs_enabled": "yes"},
        {"offload_prompt_only": 1},
        {"typo": 1},
    ],
)
def test_invalid_tier_config_fails_before_io(fields):
    with pytest.raises(ValueError, match="tier|CPU|cpu"):
        config(**fields)


def test_tiers_require_reference_and_gpu_capacity_model():
    raw = asdict(tiny())
    raw["kv_offload_tiers"] = {"cpu_capacity_bytes": 1024}
    with pytest.raises(ValueError, match="tier.*reference|tier.*capacity"):
        AgentRLConfig.from_dict(raw)


@pytest.mark.parametrize("mode", ["sync", "colocate_async", "separate_async"])
def test_completed_store_runs_without_gpu_pressure_and_cpu_only_control_is_zero_fs(tmp_path, mode):
    for fs in (True, False):
        raw = asdict(config(mode, fs_enabled=fs))
        raw["rollout_reference"]["kv_budget_bytes_per_gpu"] = 64 * 1024
        raw["kv_cache_model"].pop("capacity_bytes")
        summary, trace, _ = execute(tmp_path / str(fs), AgentRLConfig.from_dict(raw))
        assert (summary["io_totals"].get("kv_write", {}).get("payload_bytes", 0) > 0) == fs
        if not fs:
            assert not summary["io_totals"].get("kv_read")
        assert trace["kv_offload_tiers"]["fidelity"] == "uncalibrated"
        assert trace["kv_offload_tiers"]["settings"]["fs_capacity_bytes"] is None
        assert summary["fidelity"] == "uncalibrated"


def test_prompt_only_excludes_decode_blocks_and_incomplete_tail(tmp_path):
    counts = []
    for prompt_only in (True, False):
        raw = asdict(config(offload_prompt_only=prompt_only))
        raw.update(
            iterations=1,
            requests_per_rank=1,
            concurrency=1,
            prompt_tokens=3,
            response_tokens=[6],
            turns=1,
            observation_tokens=0,
            prefix_reuse_probability=0,
        )
        raw["rollout_reference"]["kv_budget_bytes_per_gpu"] = 64 * 1024
        raw["kv_cache_model"].pop("capacity_bytes")
        summary, _, _ = execute(tmp_path / str(prompt_only), AgentRLConfig.from_dict(raw))
        counts.append(summary["io_totals"]["kv_write"]["payload_bytes"])
    # One complete 2-token prompt block versus four complete history blocks.
    assert counts == [256, 1024]


@pytest.mark.parametrize("mode", ["sync", "colocate_async", "separate_async"])
def test_cpu_fs_capacity_readback_recompute_and_policy_gc_are_connected(tmp_path, mode):
    summary, trace, runner = execute(tmp_path, config(mode))
    assert summary["io_totals"]["kv_read"]["payload_bytes"] > 0
    assert any(e["event"] == "cpu_promote_end" for e in trace["events"])
    assert any(e["event"] == "gpu_recompute_end" for e in trace["events"])
    for pool in summary["ranks"][0]["kv_cache_model"]["owners"].values():
        tiers = pool["offload_tiers"]
        assert tiers["cpu_peak_bytes"] <= 512
        assert tiers["cpu_reserved_bytes"] == tiers["cpu_pinned_pages"] == 0
        assert tiers["fs_valid_payload_bytes"] == tiers["fs_capacity_evictions"] == 0
    assert not list(runner.storage_dir.glob("**/kv/*.npy"))


@pytest.mark.parametrize("mode,ranks", [("sync", 2), ("separate_async", 3)])
def test_native_mpi_roles_preserve_cpu_budget_and_fs_payloads(tmp_path, mode, ranks):
    raw = asdict(config(mode))
    if mode != "sync":
        raw["async_workload"].update(execution="mpi_shared", rollout_owners=2)
    process = invoke(tmp_path, raw, ranks=ranks)
    assert process.returncode == 0, process.stderr
    summary = json.loads(next((tmp_path / "results").glob("*/summary.json")).read_text())
    assert summary["io_totals"]["kv_write"]["payload_bytes"] > 0
    for rank in summary["ranks"]:
        for pool in rank["kv_cache_model"]["owners"].values():
            assert pool["offload_tiers"]["cpu_peak_bytes"] <= 512


@pytest.mark.parametrize("mode", ["sync", "colocate_async", "separate_async"])
def test_larger_cpu_retains_completed_blocks_and_avoids_fs_promotion_reads(tmp_path, mode):
    summary, trace, _ = execute(tmp_path, config(mode, cpu_capacity_bytes=65536))
    assert any(e["event"] == "cpu_cache_hit" for e in trace["events"])
    assert summary["io_totals"]["kv_write"]["payload_bytes"] > 0
    assert not summary["io_totals"].get("kv_read")


@pytest.mark.parametrize("mode", ["sync", "colocate_async", "separate_async"])
def test_cascade_failure_prevents_successful_lifecycle_result(tmp_path, mode):
    class Broken(NVMeBackend):
        def write(self, key, data):
            if self.base_path.name == "kv":
                raise OSError("injected FS cascade failure")
            return super().write(key, data)

    with pytest.raises(Exception, match="FS cascade|TaskGroup"):
        execute(tmp_path, config(mode), backend_factory=Broken)
    assert not list((tmp_path / "results").glob("*/summary.json"))


def test_tool_resume_makes_previous_decode_and_observation_history_prompt_eligible(tmp_path):
    volumes = []
    for turns in (1, 2):
        raw = asdict(config())
        raw.update(
            iterations=1,
            requests_per_rank=1,
            concurrency=1,
            prompt_tokens=4,
            response_tokens=[6],
            turns=turns,
            observation_tokens=2,
            prefix_reuse_probability=0,
        )
        raw["rollout_reference"]["kv_budget_bytes_per_gpu"] = 65536
        raw["kv_cache_model"].pop("capacity_bytes")
        summary, trace, _ = execute(tmp_path / str(turns), AgentRLConfig.from_dict(raw))
        volumes.append(summary["io_totals"]["kv_write"]["payload_bytes"])
        if turns == 2:
            tool = next(e["t_s"] for e in trace["events"] if e["event"] == "tool_end")
            assert any(e["event"] == "fs_cascade_end" and e["t_s"] > tool for e in trace["events"])
    assert volumes == [512, 1024]


def test_partial_policy_resume_refills_new_namespace_and_never_reads_invalidated_fs(tmp_path):
    raw = asdict(config("colocate_async"))
    raw.update(iterations=4, response_tokens=[2, 2, 30, 30])
    raw["rollout_reference"]["kv_budget_bytes_per_gpu"] = 4096
    raw["kv_cache_model"].pop("capacity_bytes")
    _, trace, _ = execute(tmp_path, AgentRLConfig.from_dict(raw))
    assert any(e["event"] == "re_prefill_end" and e["history_tokens"] > 4 for e in trace["events"])
    for invalid in (e for e in trace["events"] if e["event"] == "cpu_fs_logical_invalidate"):
        assert not any(
            e["event"] == "io_begin"
            and e["op"] == "read"
            and e["key"] in invalid["fs_keys"]
            and e["t_s"] > invalid["t_s"]
            for e in trace["events"]
        )


def test_background_store_returns_with_cpu_ready_and_fs_pending(tmp_path):
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()

        async def hold(op, key):
            started.set()
            await release.wait()

        cpu, backend, operations, events = tier(tmp_path, background=True, hold=hold)
        assert await cpu.store("a") is True
        await asyncio.wait_for(started.wait(), 1)
        assert not operations and cpu.summary()["cpu_pinned_pages"] == 1
        assert await cpu.load("a") and not cpu.fs_keys
        with pytest.raises(ValueError, match="active"):
            cpu.invalidate()
        release.set()
        await cpu.drain()
        assert cpu.fs_keys == {"a"} and backend._get_path("a").exists()
        assert cpu.summary()["cpu_pinned_pages"] == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("execution", ["caller_await", "background"])
def test_trace_transfer_pipeline_matches_store_execution(tmp_path, execution):
    summary, trace, _ = execute(tmp_path, config(fs_execution=execution))
    expected = "background_cpu_pinned" if execution == "background" else "caller_awaits_completion"
    assert summary["kv_offload_tiers"]["transfer_pipeline"] == expected
    assert trace["kv_offload_tiers"]["transfer_pipeline"] == expected


def test_background_full_staging_drops_store_without_blocking_generation(tmp_path):
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()

        async def hold(op, key):
            started.set()
            await release.wait()

        cpu, _, operations, _ = tier(tmp_path, background=True, hold=hold)
        await cpu.store("a")
        await started.wait()
        assert await asyncio.wait_for(cpu.store("b"), 0.1) is False
        assert not await cpu.load("b")
        assert cpu.summary()["cpu_store_skipped"] == 1 and not operations
        release.set()
        await cpu.drain()
        assert cpu.fs_keys == {"a"}

    asyncio.run(scenario())


def test_background_backlog_is_bounded_by_cpu_pages_and_workers(tmp_path):
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()

        async def hold(op, key):
            started.set()
            await release.wait()

        cpu, _, operations, _ = tier(tmp_path, pages=2, background=True, workers=1, hold=hold)
        for index in range(10):
            await cpu.store(str(index))
        await started.wait()
        stats = cpu.summary()
        assert stats["fs_pending_stores"] == stats["fs_pending_peak"] == 2
        assert stats["fs_write_active_peak"] == 1 and stats["cpu_store_skipped"] == 8
        assert stats["cpu_reserved_bytes"] == 64
        release.set()
        await cpu.drain()
        assert len(operations) == 2 and cpu.summary()["fs_pending_stores"] == 0

    asyncio.run(scenario())


def test_background_duplicate_store_reuses_one_pending_copy(tmp_path):
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()

        async def hold(op, key):
            started.set()
            await release.wait()

        cpu, _, operations, _ = tier(tmp_path, background=True, hold=hold)
        await cpu.store("a")
        await started.wait()
        await cpu.store("a")
        assert cpu.summary()["fs_pending_stores"] == 1
        release.set()
        await cpu.drain()
        assert operations == [("write", "a", 32)]

    asyncio.run(scenario())


def test_background_failure_is_observed_at_drain_without_false_fs_hit(tmp_path):
    async def scenario():
        cpu, _, _, events = tier(tmp_path, background=True, fail="write")
        await cpu.store("a")
        with pytest.raises(OSError, match="secondary"):
            await cpu.drain()
        assert not cpu.fs_keys and cpu.summary()["cpu_pinned_pages"] == 0
        assert cpu.summary()["fs_pending_stores"] == 0
        assert [e["success"] for e in events if e["event"] == "fs_store_end"] == [False]

    asyncio.run(scenario())


def test_background_cancelled_drain_waits_for_real_io_and_unpin(tmp_path):
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()

        async def hold(op, key):
            started.set()
            await release.wait()

        cpu, _, operations, _ = tier(tmp_path, background=True, hold=hold)
        await cpu.store("a")
        await started.wait()
        drain = asyncio.create_task(cpu.drain())
        await asyncio.sleep(0)
        drain.cancel()
        await asyncio.sleep(0.005)
        assert not drain.done() and cpu.summary()["cpu_pinned_pages"] == 1
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await drain
        assert operations == [("write", "a", 32)] and cpu.summary()["cpu_pinned_pages"] == 0

    asyncio.run(scenario())


def test_background_cpu_only_has_no_tasks_or_files(tmp_path):
    async def scenario():
        cpu, _, operations, _ = tier(tmp_path, background=True, fs=False)
        await cpu.store("a")
        await cpu.store("b")
        await cpu.drain()
        assert not operations and not cpu.fs_keys
        assert cpu.summary()["fs_pending_peak"] == 0

    asyncio.run(scenario())


def test_background_batch_larger_than_cpu_is_rejected_without_partial_store(tmp_path):
    async def scenario():
        cpu, _, operations, _ = tier(tmp_path, background=True)
        assert await cpu.store_batch(["a", "b"]) is False
        assert not cpu.entries and not operations and not cpu.pending
        assert cpu.summary()["cpu_store_skipped"] == 2

    asyncio.run(scenario())


def test_background_batch_protects_reused_keys_and_rejects_without_eviction(tmp_path):
    async def scenario():
        cpu, _, operations, _ = tier(tmp_path, pages=2, background=True)
        await cpu.store("a")
        await cpu.store("b")
        await cpu.drain()
        assert await cpu.store_batch(["a", "c", "d"]) is False
        assert list(cpu.entries) == ["a", "b"] and cpu.evictions == 0
        assert await cpu.store_batch(["a", "c"]) is True
        assert set(cpu.entries) == {"a", "c"}
        await cpu.drain()
        assert operations == [("write", key, 32) for key in ("a", "b", "c")]

    asyncio.run(scenario())


def test_background_batch_full_pins_keep_ready_keys_and_reject_new_copy(tmp_path):
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()

        async def hold(op, key):
            started.set()
            await release.wait()

        cpu, _, _, _ = tier(tmp_path, pages=2, background=True, workers=1, hold=hold)
        assert await cpu.store_batch(["a", "b"]) is True
        await started.wait()
        assert await cpu.store_batch(["a", "c"]) is False
        assert set(cpu.entries) == {"a", "b"} and await cpu.load("a")
        assert cpu.summary()["fs_pending_stores"] == 2
        release.set()
        await cpu.drain()

    asyncio.run(scenario())


def test_background_rejected_publish_retries_after_cpu_pin_release(tmp_path):
    from kv_cache.agentrl import create_lifecycle

    async def scenario():
        runner = create_lifecycle(
            config(cpu_capacity_bytes=512, fs_execution="background", offload_prompt_only=False),
            tmp_path / "data",
            tmp_path / "results",
        )
        pool = runner.make_cache_pool(0, NVMeBackend(str(tmp_path / "kv")))
        release = asyncio.Event()
        original = pool.tiers.io

        async def hold(*args, **kwargs):
            await release.wait()
            return await original(*args, **kwargs)

        pool.tiers.io = hold
        async with pool.lease(0, "r", None, 0, 2, 4, iteration=0) as lease:
            for history in (2, 4, 6):
                pool.materialize(lease, history)
                await pool.publish(lease, history, prompt_tokens=2)
            assert len(pool.published["r"][1]) == 2
            assert pool.tiers.summary()["cpu_store_skipped"] == 1
            release.set()
            await pool.tiers.drain()
            await pool.publish(lease, 6, prompt_tokens=2)
            await pool.tiers.drain()
            assert len(pool.published["r"][1]) == 3
            assert pool.tiers.summary()["fs_write_ops"] == 3

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "field,value", [("fs_execution", "unknown"), ("fs_write_workers", 0), ("fs_write_workers", True)]
)
def test_background_settings_reject_invalid_scheduler_configuration(field, value):
    from kv_cache.agentrl_tiering import TierSettings

    with pytest.raises(ValueError):
        TierSettings.parse({"cpu_capacity_bytes": 32, field: value})


@pytest.mark.parametrize("mode", ["sync", "colocate_async", "separate_async"])
def test_background_fs_outlives_request_but_is_drained_before_next_policy(tmp_path, mode):
    import time

    from kv_cache.agentrl_trace import analyze_trace

    class Slow(NVMeBackend):
        def write(self, key, data):
            if self.base_path.name == "kv":
                time.sleep(0.02)
            return super().write(key, data)

    raw = asdict(config(mode, fs_execution="background", fs_write_workers=1, cpu_capacity_bytes=64 * 1024))
    raw.update(
        iterations=1,
        requests_per_rank=2,
        concurrency=2,
        prompt_tokens=4,
        response_tokens=[2],
        turns=1,
        observation_tokens=0,
        prefix_reuse_probability=0,
    )
    if mode != "sync":
        raw["async_workload"].update(group_size=1, batch_groups=1, outstanding_groups=2, queue_capacity=1)
    summary, trace, runner = execute(tmp_path, AgentRLConfig.from_dict(raw), backend_factory=Slow)
    completed = {e["request"]: e["t_s"] for e in trace["events"] if e["event"] == "rollout_complete"}
    writes = [e for e in trace["events"] if e["event"] == "io_end" and e["kind"] == "kv" and e["op"] == "write"]
    assert any(e["request"] in completed and e["t_s"] > completed[e["request"]] for e in writes)
    assert any(e["event"] == "offload_drain_end" for e in trace["events"])
    installed = next(e["t_s"] for e in trace["events"] if e["event"] == "policy_install" and e["policy"] == 1)
    assert all(e["t_s"] < installed for e in writes if e["policy"] == 0)
    assert not analyze_trace(trace)["violations"]
    for pool in runner.cache_pools.values():
        assert pool.tiers.summary()["fs_pending_stores"] == pool.tiers.summary()["cpu_pinned_pages"] == 0
    assert summary["fidelity"] == "uncalibrated"


def test_background_secondary_failure_aborts_sync_before_policy_install(tmp_path):
    class Broken(NVMeBackend):
        def write(self, key, data):
            if self.base_path.name == "kv":
                raise OSError("background secondary failure")
            return super().write(key, data)

    from kv_cache.agentrl import create_lifecycle

    runner = create_lifecycle(
        config(fs_execution="background"), tmp_path / "data", tmp_path / "results", backend_factory=Broken
    )
    with pytest.raises(RuntimeError, match="secondary"):
        runner.run()
    assert not any(e["event"] == "policy_install" and e["policy"] > 0 for e in runner.trace.events)


def test_background_sync_checkpoint_recovery_and_corruption(tmp_path):
    from kv_cache.agentrl_trace import analyze_trace

    settings = config(fs_execution="background", cpu_capacity_bytes=4096)
    _, _, original = execute(tmp_path / "original", settings)
    manifest = original.storage_dir / "checkpoints/step-1/manifest.json"
    summary, trace, restored = execute(tmp_path / "resumed", settings, resume=manifest)
    assert summary["final_policy_version"] == 2
    assert min(e["iteration"] for e in trace["events"] if e["event"] == "rollout_start") == 1
    assert summary["io_totals"]["checkpoint_read"]["payload_bytes"] == settings.checkpoint_bytes_per_rank
    assert summary["kv_offload_tiers"]["transfer_pipeline"] == "background_cpu_pinned"
    assert not analyze_trace(trace)["violations"]
    assert all(pool.tiers.summary()["fs_pending_stores"] == 0 for pool in restored.cache_pools.values())
    metadata = json.loads(manifest.read_text())
    (manifest.parent / metadata["shards"][0]["path"]).write_bytes(b"broken")
    with pytest.raises(RuntimeError, match="checkpoint|checksum"):
        execute(tmp_path / "corrupt", settings, resume=manifest)


def test_background_fs_hit_refuses_pinned_cpu_then_retries_after_drain(tmp_path):
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()

        async def hold(op, key):
            if op == "write" and key == "b":
                started.set()
                await release.wait()

        cpu, backend, operations, events = tier(tmp_path, background=True, hold=hold)
        await cpu.store("a")
        await cpu.drain()
        await cpu.store("b")
        await started.wait()
        try:
            assert await cpu.load("b") and operations == [("write", "a", 32)]
            assert await asyncio.wait_for(cpu.load("a", request=0, policy=0), 0.1) is False
            assert backend._get_path("a").is_file() and "a" in cpu.fs_keys
            assert list(cpu.entries) == ["b"] and cpu.entries["b"].pins == 1
            assert operations == [("write", "a", 32)]
            assert cpu.summary()["cpu_promotion_attempts"] == cpu.summary()["cpu_promotion_refusals"] == 1
            assert cpu.summary()["cpu_wait_events"] == 0
            assert next(e for e in events if e["event"] == "cpu_promote_reject")["request"] == 0
        finally:
            release.set()
            await cpu.drain()
        assert await cpu.load("a")
        assert operations[-1] == ("read", "a", 32)
        assert cpu.summary()["cpu_promotion_attempts"] == 2 and cpu.summary()["cpu_promotion_refusals"] == 1
        assert cpu.summary()["cpu_pinned_pages"] == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("outcome", ["cancel", "failure"])
def test_promotion_read_cancel_and_failure_preserve_transfer_and_pin_lifetime(tmp_path, background, outcome):
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()

        async def hold(op, key):
            if op == "read":
                started.set()
                await release.wait()

        cpu, _, operations, _ = tier(
            tmp_path, background=background, hold=hold, fail="read" if outcome == "failure" else None
        )
        await cpu.store("a")
        await cpu.drain()
        await cpu.store("b")
        await cpu.drain()
        task = asyncio.create_task(cpu.load("a"))
        await started.wait()
        if outcome == "cancel":
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        assert cpu.summary()["cpu_pinned_pages"] == 1
        release.set()
        with pytest.raises(asyncio.CancelledError if outcome == "cancel" else OSError):
            await task
        stats = cpu.summary()
        assert stats["cpu_pinned_pages"] == stats["cpu_promotion_refusals"] == 0
        assert stats["cpu_promotion_attempts"] == 1
        if outcome == "cancel":
            assert operations[-1] == ("read", "a", 32) and stats["fs_read_ops"] == 1
            assert cpu.entries["a"].ready and await cpu.load("a")
        else:
            assert "a" not in cpu.entries and stats["fs_read_ops"] == 0
            assert all(op == "write" for op, _, _ in operations)

    asyncio.run(scenario())


def test_caller_await_fs_hit_still_waits_for_pinned_cpu(tmp_path):
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()

        async def hold(op, key):
            if op == "write" and key == "b":
                started.set()
                await release.wait()

        cpu, _, operations, _ = tier(tmp_path, hold=hold)
        await cpu.store("a")
        store = asyncio.create_task(cpu.store("b"))
        await started.wait()
        load = asyncio.create_task(cpu.load("a"))
        try:
            await asyncio.sleep(0)
            assert not load.done() and operations == [("write", "a", 32)]
        finally:
            release.set()
            await store
        assert await load
        assert cpu.summary()["cpu_wait_events"] == 1 and cpu.summary()["cpu_promotion_refusals"] == 0

    asyncio.run(scenario())


@pytest.mark.parametrize("background", [False, True])
def test_promotion_pending_same_key_shares_one_read(tmp_path, background):
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()

        async def hold(op, key):
            if op == "read":
                started.set()
                await release.wait()

        cpu, _, operations, _ = tier(tmp_path, background=background, hold=hold)
        await cpu.store("a")
        await cpu.drain()
        await cpu.store("b")
        await cpu.drain()
        first = asyncio.create_task(cpu.load("a"))
        await started.wait()
        second = asyncio.create_task(cpu.load("a"))
        try:
            await asyncio.sleep(0)
            assert not second.done() and cpu.summary()["cpu_pinned_pages"] == 1
        finally:
            release.set()
        assert await first and await second
        assert operations.count(("read", "a", 32)) == 1
        assert cpu.summary()["cpu_promotion_attempts"] == 1 and cpu.summary()["cpu_promotion_refusals"] == 0
        assert cpu.summary()["cpu_pinned_pages"] == 0

    asyncio.run(scenario())


def test_background_promotion_refusal_uses_existing_kv_recompute(tmp_path):
    from kv_cache.agentrl_kv import CacheSettings, KVPool

    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()
        first_key = "v0-o0-r0-block-0"
        second_key = "v0-o0-r1-block-0"

        async def hold(op, key):
            if op == "write" and key == second_key:
                started.set()
                await release.wait()

        cpu, _, operations, events = tier(tmp_path, background=True, hold=hold)
        recomputed = []

        async def recompute(key, size, **context):
            recomputed.append((key, size, context))

        pool = KVPool(CacheSettings(32, 16), 2, 0, cpu.io, cpu.emit, tiers=cpu, recompute=recompute)
        for request in (0, 1):
            async with pool.lease(0, request, None, 0, 16, 0, iteration=0) as lease:
                pool.materialize(lease, 16)
                await pool.publish(lease, 16, prompt_tokens=16)
            if request == 0:
                await cpu.drain()
        await started.wait()

        async def reload():
            async with pool.lease(0, 0, None, 0, 16, 0, iteration=0):
                assert recomputed == [(first_key, 32, {"request": 0, "policy": 0, "iteration": 0})]
                assert pool.entries[first_key].resident

        reload_task = asyncio.create_task(reload())
        try:
            done, _ = await asyncio.wait([reload_task], timeout=0.1)
            assert reload_task in done
            await reload_task
            assert all(op == "write" for op, _, _ in operations)
            assert any(e["event"] == "cpu_promote_reject" and e["key"] == first_key for e in events)
        finally:
            release.set()
            await cpu.drain()
            await asyncio.gather(reload_task, return_exceptions=True)
        assert pool.summary()["pinned_blocks"] == cpu.summary()["cpu_pinned_pages"] == 0

    asyncio.run(scenario())
