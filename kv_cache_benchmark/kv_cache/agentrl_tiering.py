"""Completed-block CPU staging with real, logically unbounded FS cascade.

CPU reservation is metadata only. This does not emulate DMA or vLLM's
scheduler-step transfer pipeline; callers await cascade/promotion completion.
"""

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class TierSettings:
    cpu_capacity_bytes: int
    fs_enabled: bool = True
    fs_capacity_bytes: None = None
    offload_prompt_only: bool = True

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

    async def load(self, key, **context):
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
        if self.condition.locked() or any(e.pins for e in self.entries.values()):
            raise ValueError("cannot invalidate active CPU transfers")
        self.emit("cpu_fs_logical_invalidate", cpu_keys=list(self.entries), fs_keys=sorted(self.fs_keys))
        self.entries.clear()
        self.fs_keys.clear()
        self.condition = asyncio.Condition()

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
