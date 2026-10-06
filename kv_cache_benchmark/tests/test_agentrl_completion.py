"""Real FS service completion must not imply primary slot availability."""

import asyncio
import time
from dataclasses import asdict

import numpy as np
import pytest
from test_agentrl_cli import invoke
from test_agentrl_kv_lifecycle import execute
from test_agentrl_tiering import config
from test_agentrl_tiering import tier as legacy_primary

from kv_cache.agentrl import AgentRLConfig
from kv_cache.agentrl_tiering import CPUPrimary, TierSettings
from kv_cache.agentrl_trace import analyze_trace
from kv_cache.backends import NVMeBackend


def polled_config(mode="sync", **fields):
    return config(
        mode,
        **{
            "fs_execution": "background",
            "fs_completion_processing": "polled",
            "fs_completion_poll_intervals_s": [0.01],
            **fields,
        },
    )


def primary(tmp_path, *, pages=1, intervals=(10,), hold=None, fail=None):
    backend, events, ready = NVMeBackend(str(tmp_path)), [], asyncio.Queue()

    def emit(name, **context):
        events.append({"event": name, "t_s": time.monotonic(), **context})
        if name == "fs_primary_completion_ready":
            ready.put_nowait(context)

    async def io(op, key, size, **context):
        if hold:
            await hold(op, key)
        if op == fail:
            raise OSError("injected completion service failure")
        if op == "write":
            backend.write(key, np.arange(size, dtype=np.uint8))
        else:
            data, _ = backend.read(key)
            assert data.nbytes == size

    cpu = CPUPrimary(
        TierSettings.parse(
            {
                "cpu_capacity_bytes": pages * 32,
                "fs_execution": "background",
                "fs_write_workers": 1,
                "fs_completion_processing": "polled",
                "fs_completion_poll_intervals_s": list(intervals),
            }
        ),
        32,
        io,
        emit,
    )
    return cpu, backend, events, ready


@pytest.mark.parametrize("value", [None, [], [0], [-0.1], [True], [float("inf")], [float("nan")], 0.01, "fast"])
def test_polled_schedule_requires_explicit_positive_finite_intervals(value):
    with pytest.raises(ValueError, match="poll|completion"):
        polled_config(fs_completion_poll_intervals_s=value)


@pytest.mark.parametrize(
    "fields",
    [
        {"fs_completion_processing": "unknown"},
        {"fs_execution": "caller_await"},
        {"fs_enabled": False},
        {"fs_completion_processing": "task"},
    ],
)
def test_completion_config_rejects_incompatible_modes_before_io(fields):
    with pytest.raises(ValueError, match="poll|completion"):
        polled_config(**fields)


def test_legacy_completion_default_keeps_checkpoint_fingerprint():
    settings = TierSettings.parse({"cpu_capacity_bytes": 32})
    assert settings.fs_completion_processing == "task"
    assert settings.fs_completion_poll_intervals_s is None
    assert AgentRLConfig.from_dict({}).fingerprint == "730cbd4d5532f15f6b4d44d76c8966c0e97e7565cf0b9b36cea0fea7c0b5c7de"


def test_finished_files_keep_pins_but_free_workers_until_batch_poll(tmp_path):
    async def scenario():
        cpu, backend, events, ready = primary(tmp_path, pages=2)
        await cpu.store_batch(["a", "b"])
        await asyncio.wait_for(ready.get(), 1)
        await asyncio.wait_for(ready.get(), 1)
        assert backend._get_path("a").exists() and backend._get_path("b").exists()
        assert cpu.summary()["fs_write_active_peak"] == 1
        assert cpu.write_active == 0 and cpu.summary()["cpu_pinned_pages"] == 2
        assert not await cpu.store("c")
        # CPU-ready data is usable although its FS cascade still protects it.
        assert await cpu.load("a")
        await cpu.process_completions("scheduler")
        assert cpu.summary()["cpu_pinned_pages"] == 0
        assert await cpu.store("c")
        await cpu.drain()
        applied = [e for e in events if e["event"] == "fs_primary_completion_apply"]
        assert {e["key"] for e in applied} == {"a", "b", "c"}
        assert all(e["service_to_primary_apply_s"] >= 0 for e in applied)
        assert cpu.summary()["fs_completion_pending"] == 0 and cpu.poll_task is None

    asyncio.run(scenario())


def test_promotion_and_same_key_waiters_stay_pending_until_poll(tmp_path):
    async def scenario():
        cpu, backend, _, ready = primary(tmp_path)
        for key in ("a", "b"):
            await cpu.store(key)
            await cpu.drain()
        while not ready.empty():
            ready.get_nowait()
        first = asyncio.create_task(cpu.load("a"))
        await asyncio.wait_for(ready.get(), 1)
        second = asyncio.create_task(cpu.load("a"))
        await asyncio.sleep(0)
        assert backend._get_path("a").exists()
        assert not cpu.entries["a"].ready and cpu.entries["a"].pins == 1
        assert not first.done() and not second.done()
        await cpu.process_completions("scheduler")
        assert await first and await second
        assert cpu.entries["a"].ready and cpu.entries["a"].pins == 0
        assert cpu.summary()["cpu_promotions"] == cpu.summary()["fs_read_ops"] == 1
        await cpu.drain()

    asyncio.run(scenario())


def test_idle_poll_releases_completed_store_without_generation(tmp_path):
    async def scenario():
        cpu, _, events, ready = primary(tmp_path, intervals=(0.005, 0.01))
        await cpu.store("a")
        await asyncio.wait_for(ready.get(), 1)
        await asyncio.wait_for(next(iter(cpu.pending.values())), 1)
        assert cpu.summary()["cpu_pinned_pages"] == 0
        assert any(e["event"] == "fs_completion_poll_begin" and e["origin"] == "scheduler" for e in events)
        await cpu.drain()
        assert cpu.poll_task is None

    asyncio.run(scenario())


def test_empty_polls_and_repeating_schedule_do_not_invent_completions(tmp_path):
    async def scenario():
        release, polled = asyncio.Event(), asyncio.Event()

        async def hold(op, key):
            await release.wait()

        cpu, _, events, _ = primary(tmp_path, intervals=(0.002, 0.005), hold=hold)
        original = cpu.emit

        def emit(name, **data):
            original(name, **data)
            if len([e for e in events if e["event"] == "fs_completion_poll_end"]) == 3:
                polled.set()

        cpu.emit = emit
        await cpu.store("a")
        await asyncio.wait_for(polled.wait(), 1)
        polls = [e for e in events if e["event"] == "fs_completion_poll_begin"]
        assert [e["modeled_interval_s"] for e in polls[:3]] == [0.002, 0.005, 0.002]
        assert not any(e["event"] == "fs_primary_completion_apply" for e in events)
        assert cpu.summary()["cpu_pinned_pages"] == 1
        release.set()
        await cpu.drain()

    asyncio.run(scenario())


def test_drain_forces_application_without_waiting_for_declared_period(tmp_path):
    async def scenario():
        cpu, _, events, _ = primary(tmp_path, intervals=(10,))
        await cpu.store("a")
        await asyncio.wait_for(cpu.drain(), 1)
        assert cpu.summary()["cpu_pinned_pages"] == cpu.summary()["fs_completion_pending"] == 0
        assert cpu.poll_task is None
        assert {e["origin"] for e in events if e["event"] == "fs_primary_completion_apply"} == {"reset"}
        cpu.invalidate()

    asyncio.run(scenario())


@pytest.mark.parametrize("op", ["write", "read"])
def test_failed_service_does_not_publish_ready_data_and_drain_releases_pins(tmp_path, op):
    async def scenario():
        cpu, _, _, _ = primary(tmp_path, fail="write" if op == "write" else None)
        if op == "write":
            await cpu.store("a")
            with pytest.raises(OSError, match="completion service"):
                await asyncio.wait_for(cpu.drain(), 1)
            assert not cpu.fs_keys
        else:
            for key in ("a", "b"):
                await cpu.store(key)
                await cpu.drain()

            async def fail_read(*args, **kwargs):
                raise OSError("injected completion service failure")

            cpu.io = fail_read
            task = asyncio.create_task(cpu.load("a"))
            await asyncio.sleep(0)
            with pytest.raises(OSError, match="completion service"):
                await asyncio.wait_for(cpu.drain(), 1)
            with pytest.raises(OSError, match="completion service"):
                await task
            assert "a" not in cpu.entries
        assert cpu.summary()["cpu_pinned_pages"] == cpu.summary()["fs_completion_pending"] == 0
        assert cpu.poll_task is None

    asyncio.run(scenario())


def test_cancelled_drain_waits_for_service_and_forces_primary_cleanup(tmp_path):
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()

        async def hold(op, key):
            started.set()
            await release.wait()

        cpu, backend, _, _ = primary(tmp_path, hold=hold)
        await cpu.store("a")
        await started.wait()
        task = asyncio.create_task(cpu.drain())
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and cpu.summary()["cpu_pinned_pages"] == 1
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert backend._get_path("a").exists()
        assert cpu.summary()["cpu_pinned_pages"] == cpu.summary()["fs_completion_pending"] == 0
        assert cpu.poll_task is None

    asyncio.run(scenario())


def test_cancelled_store_before_coroutine_start_does_not_strand_completion_drain(tmp_path):
    async def scenario():
        cpu, _, _, _ = primary(tmp_path)
        await cpu.store("a")
        next(iter(cpu.pending.values())).cancel()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert all(job.service_ready.done() for job in cpu.completion_jobs.values())
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(cpu.drain(), 1)
        assert cpu.summary()["cpu_pinned_pages"] == cpu.summary()["fs_completion_pending"] == 0
        assert cpu.poll_task is None

    asyncio.run(scenario())


def test_default_background_prestart_cancellation_also_retires_cpu_reservation(tmp_path):
    async def scenario():
        cpu, _, operations, _ = legacy_primary(tmp_path, background=True)
        await cpu.store("a")
        next(iter(cpu.pending.values())).cancel()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        with pytest.raises(asyncio.CancelledError):
            await cpu.drain()
        assert not operations and not cpu.pending and cpu.summary()["cpu_pinned_pages"] == 0

    asyncio.run(scenario())


def test_failed_promotion_wakes_coalesced_waiter_without_new_service(tmp_path):
    async def scenario():
        cpu, _, _, ready = primary(tmp_path, intervals=(0.02,))
        for key in ("a", "b"):
            await cpu.store(key)
            await cpu.drain()
        while not ready.empty():
            ready.get_nowait()
        calls = []
        started, release = asyncio.Event(), asyncio.Event()

        async def failed(*args, **kwargs):
            calls.append(args)
            started.set()
            await release.wait()
            raise OSError("injected coalesced failure")

        cpu.io = failed
        first = asyncio.create_task(cpu.load("a"))
        await started.wait()
        second = asyncio.create_task(cpu.load("a"))
        await asyncio.sleep(0)
        assert not second.done()
        release.set()
        await asyncio.wait_for(ready.get(), 1)
        await cpu.process_completions("scheduler")
        with pytest.raises(OSError, match="coalesced failure"):
            await second
        with pytest.raises(OSError, match="coalesced failure"):
            await cpu.drain()
        with pytest.raises(OSError, match="coalesced failure"):
            await first
        assert len(calls) == 1 and not cpu.completion_jobs

    asyncio.run(scenario())


@pytest.mark.parametrize("op", ["write", "read"])
@pytest.mark.parametrize("phase", ["during_service", "after_service"])
def test_cancelled_transfer_keeps_slot_protected_until_service_and_application(tmp_path, op, phase):
    async def scenario():
        cpu, backend, _, ready = primary(tmp_path)
        if op == "read":
            for key in ("a", "b"):
                await cpu.store(key)
                await cpu.drain()
        while not ready.empty():
            ready.get_nowait()
        start, release = asyncio.Event(), asyncio.Event()
        io = cpu.io

        async def held(*args, **kwargs):
            start.set()
            await release.wait()
            return await io(*args, **kwargs)

        cpu.io = held
        if op == "write":
            await cpu.store("c")
            task = cpu.pending["c"]
        else:
            task = asyncio.create_task(cpu.load("a"))
        await start.wait()
        if phase == "after_service":
            release.set()
            await asyncio.wait_for(ready.get(), 1)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done() and cpu.summary()["cpu_pinned_pages"] == 1
        if phase == "during_service":
            release.set()
            await asyncio.wait_for(ready.get(), 1)
        await cpu.process_completions("scheduler")
        with pytest.raises(asyncio.CancelledError):
            await task
        assert backend._get_path("c" if op == "write" else "a").exists()
        try:
            await cpu.drain()
        except asyncio.CancelledError:
            pass
        assert cpu.summary()["cpu_pinned_pages"] == cpu.summary()["fs_completion_pending"] == 0
        assert cpu.poll_task is None

    asyncio.run(scenario())


@pytest.mark.parametrize("mode,ranks", [("sync", 2), ("separate_async", 3)])
def test_mpi_owners_apply_and_drain_completions_in_their_own_processes(tmp_path, mode, ranks):
    import json

    from kv_cache.agentrl_trace import load_trace

    raw = asdict(polled_config(mode, fs_retention="persistent"))
    if mode == "separate_async":
        raw["async_workload"].update(execution="mpi_shared", rollout_owners=2)
    process = invoke(tmp_path, raw, ranks=ranks)
    assert process.returncode == 0, process.stderr
    path = next((tmp_path / "results").glob("*/summary.json"))
    summary, trace = json.loads(path.read_text()), load_trace(path.parent)
    assert not analyze_trace(trace)["violations"]
    applications = [e for e in trace["events"] if e["event"] == "fs_primary_completion_apply"]
    assert {e["rank"] for e in applications} == ({0, 1} if mode == "sync" else {1, 2})
    for rank in summary["ranks"]:
        for pool in rank["kv_cache_model"]["owners"].values():
            tier = pool["offload_tiers"]
            assert tier["fs_completion_pending"] == tier["cpu_pinned_pages"] == 0


@pytest.mark.parametrize("target", ["worker_kv", "worker_gc", "trainer_checkpoint", "trainer_trajectory"])
def test_mpi_polled_failures_use_existing_coordinated_abort(tmp_path, target):
    from test_agentrl_mpi_async import test_mpi_role_failure_never_hangs_or_publishes_complete_summary as check_failure

    check_failure(tmp_path, target, background=True, completion="polled")


def test_forced_drain_blocks_new_admissions_until_slot_cleanup(tmp_path):
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()

        async def hold(op, key):
            started.set()
            await release.wait()

        cpu, _, _, _ = primary(tmp_path, hold=hold)
        await cpu.store("a")
        await started.wait()
        drain = asyncio.create_task(cpu.drain())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert cpu.draining
        with pytest.raises(ValueError, match="drain"):
            await cpu.store("b")
        with pytest.raises(ValueError, match="drain"):
            await cpu.load("a")
        release.set()
        await drain
        assert not cpu.draining and not cpu.completion_jobs

    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["sync", "colocate_async", "separate_async"])
def test_mode_boundaries_drain_applied_completions_before_new_policy(tmp_path, mode):
    summary, trace, runner = execute(tmp_path, polled_config(mode, fs_retention="persistent"))
    assert summary["fidelity"] == "uncalibrated" and not analyze_trace(trace)["violations"]
    assert any(e["event"] == "fs_primary_completion_apply" for e in trace["events"])
    for pool in runner.cache_pools.values():
        assert pool.tiers.summary()["fs_completion_pending"] == pool.tiers.summary()["cpu_pinned_pages"] == 0
        assert pool.tiers.poll_task is None
    for installed in (e for e in trace["events"] if e["event"] == "policy_install" and not e["initial"]):
        assert not any(
            e["event"] == "fs_primary_completion_apply"
            and e["policy"] < installed["policy"]
            and e["t_s"] > installed["t_s"]
            for e in trace["events"]
        )


def test_scheduler_poll_progresses_during_tool_pause(tmp_path):
    raw = asdict(polled_config(cpu_capacity_bytes=65536, fs_completion_poll_intervals_s=[0.03]))
    raw.update(iterations=1, requests_per_rank=1, concurrency=1, response_tokens=[2], tool_delay_s=0.1)
    _, trace, _ = execute(tmp_path, AgentRLConfig.from_dict(raw))
    events = trace["events"]
    begin = next(e["t_s"] for e in events if e["event"] == "tool_begin")
    end = next(e["t_s"] for e in events if e["event"] == "tool_end")
    assert any(
        e["event"] == "fs_primary_completion_apply" and e["origin"] == "scheduler" and begin < e["t_s"] < end
        for e in events
    )
