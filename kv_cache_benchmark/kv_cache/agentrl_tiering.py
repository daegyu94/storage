"""Completed-block CPU staging with real, logically unbounded FS cascade.

CPU reservation is metadata only. This does not emulate DMA or vLLM's
scheduler-step transfer pipeline. Background cascade is an opt-in model;
CPU slots remain pinned; full staging refuses new stores and promotions.
"""

import asyncio
import math
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
    fs_retention: str = "policy_gc"
    fs_completion_processing: str = "task"
    fs_completion_poll_intervals_s: list | None = None

    @property
    def promotion_admission(self):
        return "refuse_pinned_capacity" if self.fs_execution == "background" else "await_slot"

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
        if result.fs_retention not in ("policy_gc", "persistent"):
            raise ValueError("tier FS retention must be policy_gc or persistent")
        if result.fs_retention == "persistent" and not result.fs_enabled:
            raise ValueError("persistent FS retention requires fs_enabled")
        if result.fs_completion_processing not in ("task", "polled"):
            raise ValueError("FS completion processing must be task or polled")
        intervals = result.fs_completion_poll_intervals_s
        if result.fs_completion_processing == "task":
            if intervals is not None:
                raise ValueError("FS completion poll intervals require polled processing")
        elif result.fs_execution != "background" or not result.fs_enabled:
            raise ValueError("polled completion requires background FS execution")
        elif (
            not isinstance(intervals, list)
            or not intervals
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in intervals)
        ):
            raise ValueError("FS completion poll intervals must be a nonempty positive finite list")
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


@dataclass
class PrimaryCompletion:
    identity: int
    op: str
    key: str
    entry: CPUBlock
    context: dict
    service_ready: asyncio.Future
    applied: asyncio.Future
    success: bool = False
    finished_s: float = 0


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
        self.promotion_attempts = self.promotion_refusals = 0
        self.completion_jobs = {}
        self.completion_sequence = self.completion_pending_peak = self.completion_ready_peak = 0
        self.completion_polls = self.completion_empty_polls = self.completion_reset_polls = 0
        self.completion_applied = 0
        self.completion_lag_sum_s = self.completion_lag_max_s = 0.0
        self.poll_task = None
        self.poll_index = 0
        self.draining = False

    def completion_ticket(self, op, key, entry, context):
        if self.settings.fs_completion_processing == "task":
            return None
        loop = asyncio.get_running_loop()
        ticket = PrimaryCompletion(
            self.completion_sequence, op, key, entry, context, loop.create_future(), loop.create_future()
        )
        self.completion_sequence += 1
        self.completion_jobs[ticket.identity] = ticket
        self.completion_pending_peak = max(self.completion_pending_peak, len(self.completion_jobs))
        if self.poll_task is None or self.poll_task.done():
            self.poll_task = asyncio.create_task(self.poll_completions())

            def settled(task):
                if not task.cancelled() and task.exception() is not None and self.failure is None:
                    self.failure = task.exception()

            self.poll_task.add_done_callback(settled)
        return ticket

    async def finish_primary(self, ticket, entry, *, success):
        if ticket is None:
            await self.unpin(entry)
            return
        self.mark_service_ready(ticket, success=success)
        try:
            await asyncio.shield(ticket.applied)
        except asyncio.CancelledError:
            await ticket.applied
            raise

    def mark_service_ready(self, ticket, *, success):
        ticket.success, ticket.finished_s = success, time.monotonic()
        try:
            self.emit(
                "fs_primary_completion_ready",
                completion_id=ticket.identity,
                op=ticket.op,
                key=ticket.key,
                success=success,
                **ticket.context,
            )
        finally:
            ticket.service_ready.set_result(None)
        self.completion_ready_peak = max(
            self.completion_ready_peak, sum(job.service_ready.done() for job in self.completion_jobs.values())
        )

    async def process_completions(self, origin, modeled_interval_s=None):
        """One host poll applies a snapshot; new completions wait for another tick."""
        ready = [job for job in self.completion_jobs.values() if job.service_ready.done()]
        poll = self.completion_polls
        self.completion_polls += 1
        self.completion_empty_polls += not ready
        self.completion_reset_polls += origin == "reset"
        self.emit(
            "fs_completion_poll_begin",
            poll_id=poll,
            origin=origin,
            pending_completions=len(self.completion_jobs),
            ready_completions=len(ready),
            modeled_interval_s=modeled_interval_s,
        )
        for job in ready:
            async with self.condition:
                if job.op == "read":
                    if job.success:
                        job.entry.ready = True
                        self.promotions += 1
                        self.fs_reads += 1
                        self.emit("cpu_promote_end", key=job.key, bytes=self.page_bytes, **job.context)
                    elif self.entries.get(job.key) is job.entry:
                        self.entries.pop(job.key)
                job.entry.pins -= 1
                self.condition.notify_all()
            lag = time.monotonic() - job.finished_s
            self.completion_applied += 1
            self.completion_lag_sum_s += lag
            self.completion_lag_max_s = max(self.completion_lag_max_s, lag)
            self.completion_jobs.pop(job.identity)
            job.applied.set_result(None)
            self.emit(
                "fs_primary_completion_apply",
                completion_id=job.identity,
                poll_id=poll,
                origin=origin,
                op=job.op,
                key=job.key,
                success=job.success,
                service_to_primary_apply_s=lag,
                **job.context,
            )
        self.emit(
            "fs_completion_poll_end",
            poll_id=poll,
            origin=origin,
            applied_completions=len(ready),
            pending_completions=len(self.completion_jobs),
        )

    async def poll_completions(self):
        try:
            while self.completion_jobs:
                intervals = self.settings.fs_completion_poll_intervals_s
                interval = intervals[self.poll_index % len(intervals)]
                self.poll_index += 1
                await asyncio.sleep(interval)
                await self.process_completions("scheduler", interval)
        finally:
            if self.poll_task is asyncio.current_task():
                self.poll_task = None

    async def drain_polled(self):
        self.draining = True
        try:
            task = self.poll_task
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            if self.completion_jobs:
                await asyncio.gather(*(job.service_ready for job in self.completion_jobs.values()))
            await self.process_completions("reset")
            if self.pending:
                await asyncio.gather(*list(self.pending.values()), return_exceptions=True)
        finally:
            self.draining = False

    async def reserve(self, key, *, ready, context, wait=True):
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
                    if not wait:
                        return None, False
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
        if self.draining:
            raise ValueError("cannot admit stores during primary completion drain")
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
        completion = self.completion_ticket("write", key, entry, context)
        started = False

        async def launch():
            nonlocal started
            started = True
            await self.cascade(key, entry, sequence, submitted, context, completion)

        task = asyncio.create_task(launch())
        self.pending[key] = task
        self.pending_peak = max(self.pending_peak, len(self.pending))

        def settled(task):
            if self.pending.get(key) is task:
                self.pending.pop(key)
            if task.cancelled() and not started:
                # Cancellation before the coroutine starts has no finally block.
                # No I/O began, but reset must still consume its reserved slot.
                self.failed_stores += 1
                if self.failure is None:
                    self.failure = asyncio.CancelledError()
                self.emit(
                    "fs_store_end", key=key, store_id=sequence, success=False, error_type="CancelledError", **context
                )
                if completion is not None:
                    self.mark_service_ready(completion, success=False)
                else:
                    cleanup = asyncio.create_task(self.unpin(entry))
                    self.pending[key] = cleanup

                    def released(result):
                        if self.pending.get(key) is result:
                            self.pending.pop(key)
                        if not result.cancelled():
                            result.exception()

                    cleanup.add_done_callback(released)
            elif not task.cancelled():
                task.exception()  # The failure is latched for store/load/drain.

        task.add_done_callback(settled)

    async def cascade(self, key, entry, sequence, submitted, context, completion=None):
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
                await self.finish_primary(completion, entry, success=success)

    async def drain(self):
        if self.settings.fs_completion_processing == "polled":
            cleanup = asyncio.create_task(self.drain_polled())
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise
            self.check_failure()
            return
        cancelled = False
        while self.pending:
            jobs = asyncio.gather(*list(self.pending.values()), return_exceptions=True)
            try:
                await asyncio.shield(jobs)
            except asyncio.CancelledError:
                await jobs
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError
        self.check_failure()

    async def load(self, key, **context):
        if self.draining:
            raise ValueError("cannot admit loads during primary completion drain")
        self.check_failure()
        async with self.condition:
            while key in self.entries and not self.entries[key].ready:
                await self.condition.wait()
            self.check_failure()
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
        self.promotion_attempts += 1
        entry, created = await self.reserve(
            key, ready=False, context=context, wait=self.settings.fs_execution == "caller_await"
        )
        if entry is None:
            self.promotion_refusals += 1
            self.emit(
                "cpu_promote_reject",
                key=key,
                bytes=self.page_bytes,
                reason="pinned_capacity",
                cpu_pinned_pages=sum(e.pins for e in self.entries.values()),
                completed_pending_keys=[j.key for j in self.completion_jobs.values() if j.service_ready.done()],
                **context,
            )
            return False
        if not created:
            self.hits += 1
            return True
        cancelled = False
        success = False
        completion = self.completion_ticket("read", key, entry, context)
        try:
            self.emit("cpu_promote_begin", key=key, bytes=self.page_bytes, **context)
            cancelled = await self.transfer("read", key, context)
            success = True
            if completion is None:
                entry.ready = True
                self.promotions += 1
                self.fs_reads += 1
                self.emit("cpu_promote_end", key=key, bytes=self.page_bytes, **context)
        except BaseException as exc:
            if completion is None:
                self.entries.pop(key)
            elif self.failure is None:
                self.failure = exc
            raise
        finally:
            await self.finish_primary(completion, entry, success=success)
        if cancelled:
            raise asyncio.CancelledError
        return True

    def invalidate(self):
        if (
            self.pending
            or self.completion_jobs
            or self.poll_task is not None
            or self.draining
            or self.condition.locked()
            or any(e.pins for e in self.entries.values())
        ):
            raise ValueError("cannot invalidate active CPU transfers")
        self.emit("cpu_fs_logical_invalidate", cpu_keys=list(self.entries), fs_keys=sorted(self.fs_keys))
        self.entries.clear()
        self.fs_keys.clear()
        self.condition = asyncio.Condition()
        self.write_slots = asyncio.Semaphore(self.settings.fs_write_workers)
        self.poll_index = 0

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
            "cpu_promotion_attempts": self.promotion_attempts,
            "cpu_promotion_refusals": self.promotion_refusals,
            "promotion_admission": self.settings.promotion_admission,
            "cpu_store_ops": self.stores,
            "cpu_wait_events": self.waits,
            "cpu_store_skipped": self.store_skipped,
            "cpu_store_skipped_payload_bytes": self.store_skipped * self.page_bytes,
            "fs_execution": self.settings.fs_execution,
            "fs_completion_processing": self.settings.fs_completion_processing,
            "fs_completion_poll_intervals_s": self.settings.fs_completion_poll_intervals_s,
            "fs_completion_pending": len(self.completion_jobs),
            "fs_completion_pending_peak": self.completion_pending_peak,
            "fs_completion_ready_peak": self.completion_ready_peak,
            "fs_completion_poll_count": self.completion_polls,
            "fs_completion_empty_polls": self.completion_empty_polls,
            "fs_completion_reset_polls": self.completion_reset_polls,
            "fs_completion_applied": self.completion_applied,
            "fs_service_to_primary_apply_sum_s": self.completion_lag_sum_s,
            "fs_service_to_primary_apply_max_s": self.completion_lag_max_s,
            "fs_retention": self.settings.fs_retention,
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
