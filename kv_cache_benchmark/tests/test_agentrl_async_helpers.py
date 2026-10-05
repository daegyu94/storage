"""The shared I/O/checkpoint helpers must be safe inside a persistent loop."""

import asyncio
import time

from test_agentrl import tiny

from kv_cache.agentrl import SyncLifecycle
from kv_cache.backends import NVMeBackend


def test_io_keeps_request_origin_and_storage_owner_during_controller_advance(tmp_path):
    runner = SyncLifecycle(tiny(), tmp_path / "data", tmp_path / "results")
    runner.setup()

    async def observe():
        task = asyncio.create_task(
            runner.io(
                runner.kv, "write", "owned", 16, kind="kv", policy=3, iteration=2, owner="rollout-0", role="rollout"
            )
        )
        await asyncio.sleep(0)
        runner.trace.iteration = runner.version = 7
        await task

    asyncio.run(observe())
    selected = [e for e in runner.trace.events if e.get("key") == "owned"]
    assert all(e["iteration"] == 2 and e["policy"] == 3 and e["owner"] == "rollout-0" for e in selected)
    assert all(e["role"] == "rollout" for e in selected)


def test_checkpoint_allows_other_owner_progress_in_same_loop(tmp_path):
    class SlowCheckpoint(NVMeBackend):
        def write(self, key, data):
            if key == "state":
                time.sleep(0.025)
            return super().write(key, data)

    runner = SyncLifecycle(tiny(), tmp_path / "data", tmp_path / "results", backend_factory=SlowCheckpoint)
    runner.setup()

    async def observe():
        async def peer():
            await asyncio.sleep(0.005)
            runner.trace.emit("peer_progress")

        async with asyncio.TaskGroup() as jobs:
            jobs.create_task(peer())
            jobs.create_task(runner.async_checkpoint(1))

    asyncio.run(observe())
    events = [e["event"] for e in runner.trace.events]
    assert events.index("checkpoint_begin") < events.index("peer_progress") < events.index("checkpoint_commit")
    assert (runner.storage_dir / "checkpoints/step-1/manifest.json").exists()
