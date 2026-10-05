"""Capacity, pin/reference and real-file contracts for logical KV residency."""

import asyncio

import numpy as np
import pytest

from kv_cache.agentrl_kv import CacheSettings, KVPool
from kv_cache.backends import NVMeBackend


def pool(tmp_path, pages=2, *, delay=0, fail=None):
    backend = NVMeBackend(str(tmp_path))
    operations, events = [], []

    async def io(op, key, size, **context):
        await asyncio.sleep(delay)
        if op == fail:
            raise OSError("injected cache transfer failure")
        if op == "write":
            backend.write(key, np.zeros(size, dtype=np.uint8))
        else:
            data, _ = backend.read(key)
            assert data.nbytes == size
        operations.append({"op": op, "key": key, "bytes": size, **context})

    cache = KVPool(
        CacheSettings(capacity_bytes=pages * 32, block_tokens=2),
        16,
        0,
        io,
        lambda n, **v: events.append({"event": n, **v}),
    )
    return cache, backend, operations, events


def lease(cache, request, history=2, extra=0, *, prefix=0, policy=0):
    return cache.lease(policy, request, "shared", prefix, history, extra, iteration=0)


def test_capacity_drives_dirty_offload_and_exact_tail_reload(tmp_path):
    cache, backend, operations, _ = pool(tmp_path, pages=1)

    async def scenario():
        async with lease(cache, 0, history=1) as current:
            cache.materialize(current, 1)
        async with lease(cache, 1) as current:
            cache.materialize(current, 2)
        assert operations == [
            {"op": "write", "key": "v0-o0-r0-block-0", "bytes": 16, "request": 1, "policy": 0, "iteration": 0}
        ]
        async with lease(cache, 0, history=1):
            pass
        assert operations[-1]["op"] == "read" and operations[-1]["bytes"] == 16
        assert backend.metadata["v0-o0-r0-block-0"]["size"] == 16
        assert cache.summary()["resident_peak_bytes"] == 32

    asyncio.run(scenario())


def test_clean_repeated_eviction_does_not_rewrite_persisted_copy(tmp_path):
    cache, _, operations, _ = pool(tmp_path, pages=1)

    async def scenario():
        for request in (0, 1, 0, 1, 0):
            async with lease(cache, request) as current:
                cache.materialize(current, 2)
        assert sum(e["op"] == "write" and e["key"] == "v0-o0-r0-block-0" for e in operations) == 1
        assert cache.summary()["reload_ops"] == 3

    asyncio.run(scenario())


def test_tail_growth_marks_copy_dirty_and_offloads_new_valid_size(tmp_path):
    cache, _, operations, _ = pool(tmp_path, pages=1)

    async def scenario():
        async with lease(cache, 0, history=1) as current:
            cache.materialize(current, 1)
        async with lease(cache, 1):
            pass
        async with lease(cache, 0, history=1, extra=1) as current:
            cache.materialize(current, 2)
        async with lease(cache, 1):
            pass
        assert [e["bytes"] for e in operations if e["op"] == "write"] == [16, 32]

    asyncio.run(scenario())


def test_pinned_working_set_causes_capacity_wait_without_eviction(tmp_path):
    cache, _, operations, events = pool(tmp_path, pages=1)

    async def scenario():
        started = asyncio.Event()

        async def competing():
            started.set()
            async with lease(cache, 1) as current:
                cache.materialize(current, 2)

        async with lease(cache, 0) as current:
            cache.materialize(current, 2)
            task = asyncio.create_task(competing())
            await started.wait()
            await asyncio.sleep(0.005)
            assert not operations and not task.done()
        await asyncio.wait_for(task, 1)
        assert operations[0]["op"] == "write"
        assert any(e["event"] == "cache_capacity_wait" for e in events)
        assert cache.summary()["pinned_blocks"] == 0

    asyncio.run(scenario())


def test_shared_full_prefix_is_reference_counted_and_private_blocks_are_distinct(tmp_path):
    cache, _, operations, _ = pool(tmp_path, pages=3)

    async def scenario():
        async with lease(cache, 0, history=4, prefix=3) as first:
            cache.materialize(first, 4)
            async with lease(cache, 1, history=4, prefix=3) as second:
                assert cache.prefix_ready(second)
                cache.materialize(second, 4)
                assert cache.summary()["pinned_blocks"] == 3
                assert max(e.pins for e in cache.entries.values()) == 2
        assert not operations
        assert cache.summary()["resident_bytes"] == 96

    asyncio.run(scenario())


def test_release_and_policy_invalidation_retire_logical_data_before_file_delete(tmp_path):
    cache, backend, operations, events = pool(tmp_path, pages=1)

    async def scenario():
        for request in (0, 1):
            async with lease(cache, request) as current:
                cache.materialize(current, 2)
        retired = await cache.release_request(0)
        assert retired == ["v0-o0-r0-block-0"] and retired[0] in backend.metadata
        assert all(e.request != 0 for e in cache.entries.values())
        cache.invalidate()
        assert not cache.entries and cache.summary()["resident_bytes"] == 0
        async with lease(cache, 0, policy=1) as current:
            assert not cache.prefix_ready(current)
            cache.materialize(current, 2)
        assert not any(e["op"] == "read" for e in operations)
        assert any(e["event"] == "cache_logical_invalidate" for e in events)

    asyncio.run(scenario())


@pytest.mark.parametrize("op", ["write", "read"])
def test_transfer_failure_is_propagated_and_does_not_pin_failed_lease(tmp_path, op):
    cache, _, _, _ = pool(tmp_path, pages=1)

    async def scenario():
        async with lease(cache, 0) as current:
            cache.materialize(current, 2)
        if op == "read":
            async with lease(cache, 1):
                pass
        original = cache.io

        async def broken(operation, *args, **kwargs):
            if operation == op:
                raise OSError("injected cache transfer failure")
            return await original(operation, *args, **kwargs)

        cache.io = broken
        with pytest.raises(OSError, match="injected cache"):
            async with lease(cache, 1 if op == "write" else 0):
                pass
        assert cache.summary()["pinned_blocks"] == 0
        assert cache.summary()["resident_bytes"] <= 32

    asyncio.run(scenario())


def test_cancelled_transfer_drains_and_releases_acquired_pins(tmp_path):
    cache, _, operations, _ = pool(tmp_path, pages=1, delay=0.02)

    async def scenario():
        async with lease(cache, 0) as current:
            cache.materialize(current, 2)

        async def competing():
            async with lease(cache, 1) as current:
                cache.materialize(current, 2)

        task = asyncio.create_task(competing())
        await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(operations) == 1 and operations[0]["op"] == "write"
        assert cache.summary()["pinned_blocks"] == 0
        assert cache.summary()["resident_bytes"] <= 32

    asyncio.run(scenario())


def test_larger_than_capacity_working_set_fails_instead_of_silent_partial_attention(tmp_path):
    cache, _, _, _ = pool(tmp_path, pages=1)

    async def scenario():
        with pytest.raises(ValueError, match="working set"):
            async with lease(cache, 0, history=3):
                pass
        assert not cache.entries

    asyncio.run(scenario())


def test_pinned_data_cannot_be_retired_and_unleased_data_cannot_be_mutated(tmp_path):
    cache, _, _, _ = pool(tmp_path)

    async def scenario():
        async with lease(cache, 0) as current:
            cache.materialize(current, 2)
            with pytest.raises(ValueError, match="pinned"):
                await cache.release_request(0)
            with pytest.raises(ValueError, match="pinned"):
                cache.invalidate()
        with pytest.raises(ValueError, match="pinned"):
            cache.materialize(current, 2)

    asyncio.run(scenario())
