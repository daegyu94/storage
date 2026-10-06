"""Completed-block CPU staging with real, logically unbounded FS cascade.

CPU reservation is metadata only. This does not emulate DMA or vLLM's
scheduler-step transfer pipeline. Background cascade is an opt-in model;
CPU slots remain pinned and full staging skips new stores without blocking.
"""

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class TierSettings:
    cpu_capacity_bytes: int
    fs_enabled: bool = True
    fs_capacity_bytes: None = None
    offload_prompt_only: bool = True
    fs_execution: str = "caller_await"
    fs_write_workers: int = 16

    @classmethod
    def parse(cls, values):
        if not isinstance(values, dict) or set(values) - {f.name for f in fields(cls)}:
            raise ValueError("kv_offload_tiers requires known tier settings")
        try:
            result = cls(**values)
        except TypeError as exc:
            raise ValueError("offload tiers require cpu_capacity_bytes") from exc
        if type(result.cpu_capacity_bytes) is not int or result.cpu_capacity_bytes <= 0:
            raise ValueError("tier CPU capacity must be a positive integer")
        if type(result.fs_enabled) is not bool or type(result.offload_prompt_only) is not bool:
            raise ValueError("tier fs_enabled/offload_prompt_only must be boolean")
        if result.fs_capacity_bytes is not None:
            raise ValueError("tier FS capacity must be null (logically unbounded)")
        if result.fs_execution not in ("caller_await", "background"):
            raise ValueError("tier FS execution must be caller_await or background")
        if type(result.fs_write_workers) is not int or result.fs_write_workers <= 0:
            raise ValueError("tier FS write workers must be a positive integer")
        return result

    def validate(self, config):
        if config.rollout_reference is None or config.kv_cache_model is None:
            raise ValueError("offload tiers require reference geometry and GPU-like capacity")
        page = config.kv_cache_model.get("block_tokens", 16) * config.bytes_per_token
        if self.cpu_capacity_bytes < page:
            raise ValueError("tier CPU capacity must fit a complete staging block")


@dataclass
class CPUBlock:
    ready: bool
    pins: int = 1


class CPUPrimary:
    def __init__(self, settings, page_bytes, io, emit):
        self.settings, self.page_bytes, self.io, self.emit = settings, page_bytes, io, emit
        self.pages = settings.cpu_capacity_bytes // page_bytes
        if not self.pages:
            raise ValueError("tier CPU capacity must fit one block")
        self.entries = OrderedDict()
        self.fs_keys = set()
        self.condition = asyncio.Condition()
        self.peak = self.fs_peak = 0
        self.hits = self.misses = self.evictions = self.promotions = self.stores = 0
        self.fs_reads = self.fs_writes = self.waits = 0
        self.pending = {}
        self.failure = None
        self.write_slots = asyncio.Semaphore(settings.fs_write_workers)
        self.pending_peak = self.write_active = self.write_active_peak = 0
        self.store_skipped = self.failed_stores = 0
        self.queue_wait_s = 0.0
        self.store_sequence = 0

    async def reserve(self, key, *, ready, context):
        async with self.condition:
            waiting = False
            while True:
                entry = self.entries.get(key)
                if entry and entry.ready:
                    self.entries.move_to_end(key)
                    return entry, False
                if entry is None:
                    if len(self.entries) < self.pages:
                        entry = CPUBlock(ready)
                        self.entries[key] = entry
                        self.peak = max(self.peak, len(self.entries) * self.page_bytes)
                        return entry, True
                    victim = next((k for k, e in self.entries.items() if not e.pins), None)
                    if victim is not None:
                        self.entries.pop(victim)
                        self.evictions += 1
                        self.emit("cpu_cache_evict", key=victim, bytes=self.page_bytes, **context)
                        continue
                if not waiting:
                    self.waits += 1
                    waiting = True
                    self.emit("cpu_capacity_wait", key=key, **context)
                await self.condition.wait()

    async def transfer(self, op, key, context):
        task = asyncio.create_task(self.io(op, key, self.page_bytes, tier="fs", **context))
        cancelled = False
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            cancelled = True
        return cancelled

    async def unpin(self, entry):
        async with self.condition:
            entry.pins -= 1
            self.condition.notify_all()

    async def store(self, key, **context):
        if self.settings.fs_execution == "background":
            return await self.background_store(key, context)
        entry, created = await self.reserve(key, ready=True, context=context)
        if not created:
            return
        cancelled = False
        try:
            self.stores += 1
            self.emit("cpu_store_ready", key=key, bytes=self.page_bytes, **context)
            if self.settings.fs_enabled and key not in self.fs_keys:
                cancelled = await self.transfer("write", key, context)
                self.fs_keys.add(key)
                self.fs_writes += 1
                self.fs_peak = max(self.fs_peak, len(self.fs_keys) * self.page_bytes)
                self.emit("fs_cascade_end", key=key, bytes=self.page_bytes, **context)
        finally:
            await self.unpin(entry)
        if cancelled:
            raise asyncio.CancelledError

    def check_failure(self):
        if self.failure is not None:
            raise self.failure

    async def background_store(self, key, context):
        return await self.store_batch([key], **context)

    async def store_batch(self, keys, **context):
        if self.settings.fs_execution != "background":
            for key in keys:
                await self.store(key, **context)
            return True
        for completed_key, completed in list(self.pending.items()):
            if completed.done():
                self.pending.pop(completed_key)
        self.check_failure()
        keys = list(dict.fromkeys(keys))
        async with self.condition:
            new_keys = [key for key in keys if key not in self.entries]
            needed = max(0, len(self.entries) + len(new_keys) - self.pages)
            victims = [key for key, entry in self.entries.items() if not entry.pins and key not in keys]
            if needed > len(victims):
                reason = "batch_capacity" if len(new_keys) > self.pages else "pinned_or_protected_capacity"
                for key in new_keys:
                    self.store_skipped += 1
                    self.emit("cpu_store_skip", key=key, bytes=self.page_bytes, reason=reason, **context)
                return False
            for key in victims[:needed]:
                self.entries.pop(key)
                self.evictions += 1
                self.emit("cpu_cache_evict", key=key, bytes=self.page_bytes, **context)
            created = []
            for key in keys:
                if key not in self.entries:
                    entry = self.entries[key] = CPUBlock(ready=True)
                    created.append((key, entry))
                self.entries.move_to_end(key)
            self.peak = max(self.peak, len(self.entries) * self.page_bytes)
        for key, entry in created:
            self.stores += 1
            self.emit("cpu_store_ready", key=key, bytes=self.page_bytes, **context)
            if not self.settings.fs_enabled or key in self.fs_keys:
                await self.unpin(entry)
            else:
                self.submit_store(key, entry, context)
        return True

    def submit_store(self, key, entry, context):
        sequence = self.store_sequence
        self.store_sequence += 1
        submitted = time.monotonic()
        self.emit("fs_store_submit", key=key, store_id=sequence, bytes=self.page_bytes, **context)
        task = asyncio.create_task(self.cascade(key, entry, sequence, submitted, context))
        self.pending[key] = task
        self.pending_peak = max(self.pending_peak, len(self.pending))

        def settled(task):
            if self.pending.get(key) is task:
                self.pending.pop(key)
            if not task.cancelled():
                task.exception()  # The failure is latched for store/load/drain.

        task.add_done_callback(settled)

    async def cascade(self, key, entry, sequence, submitted, context):
        success = False
        error_type = None
        try:
            async with self.write_slots:
                waited = time.monotonic() - submitted
                self.queue_wait_s += waited
                self.write_active += 1
                self.write_active_peak = max(self.write_active_peak, self.write_active)
                self.emit("fs_store_begin", key=key, store_id=sequence, executor_wait_s=waited, **context)
                try:
                    cancelled = await self.transfer("write", key, context)
                    self.fs_keys.add(key)
                    self.fs_writes += 1
                    self.fs_peak = max(self.fs_peak, len(self.fs_keys) * self.page_bytes)
                    self.emit("fs_cascade_end", key=key, bytes=self.page_bytes, **context)
                    success = True
                    if cancelled:
                        raise asyncio.CancelledError
                finally:
                    self.write_active -= 1
        except BaseException as exc:
            error_type = type(exc).__name__
            self.failed_stores += not success
            if self.failure is None:
                self.failure = exc
            raise
        finally:
            try:
                self.emit("fs_store_end", key=key, store_id=sequence, success=success, error_type=error_type, **context)
            finally:
                await self.unpin(entry)

    async def drain(self):
        if self.pending:
            jobs = asyncio.gather(*list(self.pending.values()), return_exceptions=True)
            try:
                await asyncio.shield(jobs)
            except asyncio.CancelledError:
                await jobs
                raise
        self.check_failure()

    async def load(self, key, **context):
        self.check_failure()
        async with self.condition:
            while key in self.entries and not self.entries[key].ready:
                await self.condition.wait()
            entry = self.entries.get(key)
            if entry:
                self.entries.move_to_end(key)
                self.hits += 1
                self.emit("cpu_cache_hit", key=key, bytes=self.page_bytes, **context)
                return True
            self.misses += 1
            self.emit("cpu_cache_miss", key=key, **context)
            if key not in self.fs_keys:
                self.emit("fs_cache_miss", key=key, **context)
                return False
        entry, created = await self.reserve(key, ready=False, context=context)
        if not created:
            self.hits += 1
            return True
        cancelled = False
        try:
            self.emit("cpu_promote_begin", key=key, bytes=self.page_bytes, **context)
            cancelled = await self.transfer("read", key, context)
            entry.ready = True
            self.promotions += 1
            self.fs_reads += 1
            self.emit("cpu_promote_end", key=key, bytes=self.page_bytes, **context)
        except BaseException:
            self.entries.pop(key)
            raise
        finally:
            await self.unpin(entry)
        if cancelled:
            raise asyncio.CancelledError
        return True

    def invalidate(self):
        if self.pending or self.condition.locked() or any(e.pins for e in self.entries.values()):
            raise ValueError("cannot invalidate active CPU transfers")
        self.emit("cpu_fs_logical_invalidate", cpu_keys=list(self.entries), fs_keys=sorted(self.fs_keys))
        self.entries.clear()
        self.fs_keys.clear()
        self.condition = asyncio.Condition()
        self.write_slots = asyncio.Semaphore(self.settings.fs_write_workers)

    def summary(self):
        return {
            "cpu_capacity_bytes": self.settings.cpu_capacity_bytes,
            "cpu_reserved_bytes": len(self.entries) * self.page_bytes,
            "cpu_peak_bytes": self.peak,
            "cpu_pinned_pages": sum(e.pins for e in self.entries.values()),
            "cpu_hits": self.hits,
            "cpu_misses": self.misses,
            "cpu_evictions": self.evictions,
            "cpu_promotions": self.promotions,
            "cpu_store_ops": self.stores,
            "cpu_wait_events": self.waits,
            "cpu_store_skipped": self.store_skipped,
            "cpu_store_skipped_payload_bytes": self.store_skipped * self.page_bytes,
            "fs_execution": self.settings.fs_execution,
            "fs_write_workers": self.settings.fs_write_workers,
            "fs_pending_stores": len(self.pending),
            "fs_pending_peak": self.pending_peak,
            "fs_write_active_peak": self.write_active_peak,
            "fs_executor_wait_s": self.queue_wait_s,
            "fs_failed_stores": self.failed_stores,
            "fs_write_ops": self.fs_writes,
            "fs_write_payload_bytes": self.fs_writes * self.page_bytes,
            "fs_read_ops": self.fs_reads,
            "fs_read_payload_bytes": self.fs_reads * self.page_bytes,
            "fs_valid_payload_bytes": len(self.fs_keys) * self.page_bytes,
            "fs_valid_peak_bytes": self.fs_peak,
            "fs_capacity_bounded": False,
            "fs_capacity_evictions": 0,
            "cpu_residency_model": "metadata_only",
            "fidelity": "uncalibrated",
        }
