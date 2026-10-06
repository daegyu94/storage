"""Metadata-only resident pages with real storage-enabled write-back swapping.

This is a chunk-safe working-set assumption, not a CUDA/vLLM cache emulator.
No resident payload arrays are allocated; only offload/reload use file I/O.
"""

import asyncio
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class CacheSettings:
    capacity_bytes: int
    block_tokens: int = 16

    @classmethod
    def parse(cls, values):
        if not isinstance(values, dict) or set(values) - {f.name for f in fields(cls)}:
            raise ValueError("kv_cache_model must contain known settings")
        try:
            settings = cls(**values)
        except TypeError as exc:
            raise ValueError("kv_cache_model requires capacity_bytes") from exc
        for name in ("capacity_bytes", "block_tokens"):
            if type(getattr(settings, name)) is not int or getattr(settings, name) <= 0:
                raise ValueError(f"cache {name} must be a positive integer")
        return settings

    def validate(self, config):
        if config.kv_offload_fraction != 0:
            raise ValueError("kv_cache_model requires kv_offload_fraction=0; capacity drives offload")
        if self.block_tokens * config.bytes_per_token > config.max_payload_bytes:
            raise ValueError("cache block payload exceeds max_payload_bytes")
        lengths = [config.prompt_tokens + max(config.response_tokens) + (config.turns - 1) * config.observation_tokens]
        if config.request_profile:
            lengths = [
                record["prompt_tokens"]
                + sum(turn["generated_tokens"] + turn["observation_tokens"] for turn in record["turns"])
                for record in config.request_profile["records"]
            ]
        if (
            max((n + self.block_tokens - 1) // self.block_tokens for n in lengths)
            * self.block_tokens
            * config.bytes_per_token
            > self.capacity_bytes
        ):
            raise ValueError("cache capacity must fit each full request working set")


@dataclass
class Block:
    request: int | None
    size: int = 0
    resident: bool = False
    dirty: bool = False
    stored_size: int | None = None
    pins: int = 0


@dataclass
class Lease:
    policy: int
    request: int
    prefix_id: str | None
    prefix_tokens: int
    iteration: int
    planned: dict
    active: bool = False

    @property
    def context(self):
        return {"request": self.request, "policy": self.policy, "iteration": self.iteration}


class KVPool:
    def __init__(self, settings, bytes_per_token, owner, io, emit, *, tiers=None, recompute=None):
        self.settings, self.bytes_per_token, self.owner = settings, bytes_per_token, owner
        self.io, self.emit = io, emit
        self.tiers, self.recompute = tiers, recompute
        self.page_bytes = settings.block_tokens * bytes_per_token
        self.entries = OrderedDict()
        self.condition = asyncio.Condition()
        self.prefix_locks = {}
        self.published = {}
        self.used = self.peak = 0
        self.offload_ops = self.offload_bytes = self.reload_ops = self.reload_bytes = 0
        self.wait_events, self.wait_s = 0, 0.0

    def plan(self, policy, request, prefix_id, prefix_tokens, history):
        block = self.settings.block_tokens
        prefix = prefix_tokens // block * block
        planned = {}
        for offset in range(0, prefix, block):
            planned[f"v{policy}-o{self.owner}-prefix-{prefix_id}-block-{offset // block}"] = (None, self.page_bytes)
        for offset in range(0, history - prefix, block):
            size = min(block, history - prefix - offset) * self.bytes_per_token
            planned[f"v{policy}-o{self.owner}-r{request}-block-{offset // block}"] = (request, size)
        return planned

    @asynccontextmanager
    async def lease(self, policy, request, prefix_id, prefix_tokens, history, extra, *, iteration):
        planned = self.plan(policy, request, prefix_id, prefix_tokens, history + extra)
        if len(planned) * self.page_bytes > self.settings.capacity_bytes:
            raise ValueError("request working set exceeds cache capacity")
        current = Lease(
            policy,
            request,
            prefix_id,
            prefix_tokens // self.settings.block_tokens * self.settings.block_tokens,
            iteration,
            planned,
        )
        task, acquired = asyncio.create_task(self.acquire(current)), False
        try:
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                # Commit/drain any file transfer before releasing metadata/pins.
                await task
                acquired = True
                raise
            acquired = True
            current.active = True
            yield current
        finally:
            current.active = False
            if acquired:
                async with self.condition:
                    for key in planned:
                        entry = self.entries[key]
                        entry.pins -= 1
                        if not entry.pins and not entry.size:
                            self.entries.pop(key)
                            self.used -= self.page_bytes
                    self.condition.notify_all()

    async def acquire(self, lease):
        wanted = lease.planned
        async with self.condition:
            started = None
            while True:
                other_pinned = sum(e.resident and e.pins > 0 for k, e in self.entries.items() if k not in wanted)
                if (len(wanted) + other_pinned) * self.page_bytes <= self.settings.capacity_bytes:
                    break
                if started is None:
                    started = time.monotonic()
                    self.wait_events += 1
                    self.emit("cache_capacity_wait", **lease.context, required_page_bytes=len(wanted) * self.page_bytes)
                await self.condition.wait()
            if started is not None:
                duration = time.monotonic() - started
                self.wait_s += duration
                self.emit("cache_capacity_ready", **lease.context, wait_s=duration)
            missing = sum(k not in self.entries or not self.entries[k].resident for k in wanted)
            for key, entry in list(self.entries.items()):
                if self.used + missing * self.page_bytes <= self.settings.capacity_bytes:
                    break
                if key in wanted or not entry.resident or entry.pins:
                    continue
                if entry.size and entry.dirty and self.tiers is None:
                    await self.io("write", key, entry.size, **lease.context)
                    entry.stored_size, entry.dirty = entry.size, False
                    self.offload_ops += 1
                    self.offload_bytes += entry.size
                    self.emit("cache_offload", key=key, bytes=entry.size, victim_request=entry.request, **lease.context)
                entry.resident = False
                self.used -= self.page_bytes
            for key, (request, _) in wanted.items():
                entry = self.entries.setdefault(key, Block(request))
                if not entry.resident:
                    if self.tiers is not None and entry.size:
                        loaded = await self.tiers.load(key, **lease.context)
                        if not loaded:
                            await self.recompute(key, entry.size, **lease.context)
                    elif entry.stored_size is not None:
                        await self.io("read", key, entry.stored_size, **lease.context)
                        self.reload_ops += 1
                        self.reload_bytes += entry.stored_size
                        self.emit("cache_reload", key=key, bytes=entry.stored_size, **lease.context)
                    entry.resident = True
                    self.used += self.page_bytes
                self.entries.move_to_end(key)
            for key in wanted:
                self.entries[key].pins += 1
            self.peak = max(self.peak, self.used)
            self.emit(
                "cache_lease",
                **lease.context,
                resident_bytes=self.used,
                pinned_blocks=sum(e.pins > 0 for e in self.entries.values()),
            )

    def prefix_ready(self, lease):
        keys = [k for k, (request, _) in lease.planned.items() if request is None]
        return bool(keys) and all(self.entries[k].size == self.page_bytes for k in keys)

    def materialize(self, lease, history, *, prefix_only=False):
        if not lease.active:
            raise ValueError("materialization requires a pinned lease")
        planned = self.plan(lease.policy, lease.request, lease.prefix_id, lease.prefix_tokens, history)
        for key, (request, size) in planned.items():
            if prefix_only and request is not None:
                continue
            if key not in lease.planned or size > lease.planned[key][1]:
                raise ValueError("materialization exceeds pinned working set")
            entry = self.entries[key]
            if not entry.resident or not entry.pins:
                raise ValueError("materialization requires pinned resident KV")
            if entry.size != size:
                entry.size, entry.dirty = size, True

    async def publish(self, lease, history, *, prompt_tokens):
        if self.tiers is None:
            return
        if not lease.active:
            raise ValueError("completed block publish requires a pinned lease")
        call = (lease.policy, prompt_tokens)
        previous = self.published.get(lease.request)
        if previous is None or previous[0] != call:
            previous = self.published[lease.request] = (call, set())
        stored = previous[1]
        limit = prompt_tokens if self.tiers.settings.offload_prompt_only else history
        eligible = self.plan(lease.policy, lease.request, lease.prefix_id, lease.prefix_tokens, limit)
        for key, (_, size) in eligible.items():
            if key not in stored and size == self.page_bytes and self.entries[key].size == self.page_bytes:
                await self.tiers.store(key, **lease.context)
                stored.add(key)

    async def release_request(self, request):
        async with self.condition:
            keys = [k for k, e in self.entries.items() if e.request == request]
            if any(self.entries[k].pins for k in keys):
                raise ValueError("cannot retire pinned request KV")
            self.published.pop(request, None)
            physical = []
            for key in keys:
                entry = self.entries.pop(key)
                self.used -= self.page_bytes if entry.resident else 0
                if entry.stored_size is not None and self.tiers is None:
                    physical.append(key)
            self.emit("cache_request_release", request=request, keys=keys, resident_bytes=self.used)
            self.condition.notify_all()
            return physical

    def invalidate(self):
        if self.condition.locked() or any(e.pins for e in self.entries.values()):
            raise ValueError("cannot invalidate pinned KV")
        if self.tiers is not None:
            self.tiers.invalidate()
        self.emit("cache_logical_invalidate", keys=list(self.entries), resident_bytes=self.used)
        self.entries.clear()
        self.prefix_locks.clear()
        self.published.clear()
        # Sync creates a fresh event loop for each collective phase. Reset
        # loop-bound primitives only after the complete owner has drained.
        self.condition = asyncio.Condition()
        self.used = 0

    def summary(self):
        return {
            "capacity_bytes": self.settings.capacity_bytes,
            "block_tokens": self.settings.block_tokens,
            "page_bytes": self.page_bytes,
            "resident_bytes": self.used,
            "resident_peak_bytes": self.peak,
            "resident_valid_payload_bytes": sum(e.size for e in self.entries.values() if e.resident),
            "pinned_blocks": sum(e.pins > 0 for e in self.entries.values()),
            "offload_ops": self.offload_ops,
            "offload_payload_bytes": self.offload_bytes,
            "reload_ops": self.reload_ops,
            "reload_payload_bytes": self.reload_bytes,
            "capacity_wait_events": self.wait_events,
            "capacity_wait_s": self.wait_s,
            "residency_model": "chunk_safe_working_set",
            "storage_capacity_bounded": False,
            **({"offload_tiers": self.tiers.summary()} if self.tiers is not None else {}),
        }
