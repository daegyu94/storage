"""Bounded local async control flow over the existing experimental file I/O.

Logical owners share one process; this is not a distributed/GPU scheduler.
"""

import asyncio
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, fields
from functools import partial

from kv_cache.agentrl import SyncLifecycle, kv_delta_bytes, network_delay


@dataclass(frozen=True)
class AsyncSettings:
    execution: str = "local"
    rpc_timeout_s: float = 30.0
    group_size: int = 2
    batch_groups: int = 1
    outstanding_groups: int = 4
    queue_capacity: int = 2
    rollout_owners: int = 1
    parameter_sync_step: int = 1
    max_prompt_age: int | None = None
    staleness_strategy: str = "drop"
    kv_gc_delay_s: float = 0.0

    @classmethod
    def parse(cls, values, mode):
        if not isinstance(values, dict) or set(values) - {f.name for f in fields(cls)}:
            raise ValueError("async_workload must be a mapping of known settings")
        result = cls(**values)
        if result.execution not in ("local", "mpi_shared"):
            raise ValueError("async execution must be local or mpi_shared")
        for name in (
            "group_size",
            "batch_groups",
            "outstanding_groups",
            "queue_capacity",
            "rollout_owners",
            "parameter_sync_step",
        ):
            if type(getattr(result, name)) is not int or getattr(result, name) <= 0:
                raise ValueError(f"async {name} must be a positive integer")
        if result.batch_groups > min(result.outstanding_groups, result.queue_capacity):
            raise ValueError("batch_groups exceeds outstanding/queue capacity")
        if result.max_prompt_age is not None and (type(result.max_prompt_age) is not int or result.max_prompt_age <= 0):
            raise ValueError("max_prompt_age must be null or a positive integer")
        if result.staleness_strategy not in ("drop", "wait"):
            raise ValueError("staleness_strategy must be drop or wait")
        from kv_cache.agentrl_trace import number

        number(result.kv_gc_delay_s, "kv_gc_delay_s")
        number(result.rpc_timeout_s, "rpc_timeout_s")
        if result.rpc_timeout_s <= 0:
            raise ValueError("rpc_timeout_s must be positive")
        if mode == "colocate_async" and result.parameter_sync_step != 1:
            raise ValueError("colocate_async uses a full batch, parameter_sync_step must be 1")
        return result

    def validate_profile_groups(self, profile, requests_per_rank):
        if profile["schema_version"] == 2:
            from kv_cache.agentrl_trace import profile_groups

            groups = profile_groups(profile)
            if any(len(g) != self.group_size for g in groups):
                raise ValueError("grouped profile fanout must match group_size")
            if any(r["owner"] >= self.rollout_owners for r in profile["records"]):
                raise ValueError("grouped profile owner exceeds rollout_owners")
            return
        records = profile["records"]
        if len(records) % self.group_size or requests_per_rank % self.group_size:
            raise ValueError("async profile/catalog must contain complete prompt groups")
        for offset in range(0, len(records), self.group_size):
            siblings = records[offset : offset + self.group_size]
            shapes = {tuple(r[k] for k in ("prompt_tokens", "prefix_tokens", "prefix_id")) for r in siblings}
            if len(shapes) != 1:
                raise ValueError("async profile group siblings must share prompt/prefix shape")


def age_blocks(age, threshold, strategy, *, terminal):
    if threshold is None:
        return False
    return (strategy == "drop" and terminal and age > threshold) or (
        strategy == "wait" and not terminal and age >= threshold
    )


class PauseGate:
    """Drain engine sections at chunk/I/O boundaries without discarding locals."""

    def __init__(self):
        self.opened, self.idle = asyncio.Event(), asyncio.Event()
        self.opened.set()
        self.idle.set()
        self.busy = 0

    @asynccontextmanager
    async def section(self):
        while True:
            await self.opened.wait()
            if self.opened.is_set():
                self.busy += 1
                self.idle.clear()
                break
        try:
            yield
        finally:
            self.busy -= 1
            if not self.busy:
                self.idle.set()

    async def pause(self):
        self.opened.clear()
        await self.idle.wait()

    def resume(self):
        self.opened.set()


@dataclass
class RequestState:
    request: int
    group: int
    owner: int
    origin: int
    source_request: int
    shape: dict
    history: int
    generated: int = 0
    kv_version: int | None = None
    first_version: int | None = None
    last_version: int | None = None
    record: dict | None = None
    calibration: dict | None = None


class AsyncLifecycle(SyncLifecycle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.trainer_mode not in ("colocate_async", "separate_async"):
            raise ValueError("AsyncLifecycle requires an async trainer_mode")
        if self.world != 1:
            raise ValueError("multi-rank async is unsupported; distributed role protocol is not implemented")
        if self.resume:
            raise ValueError("async resume is unsupported; pending/finished groups are not checkpointed")
        self.settings = AsyncSettings.parse(self.config.async_workload, self.config.trainer_mode)
        self.groups, self.states = {}, {}
        self.ready = deque()
        self.next_group = self.queue_peak = self.outstanding_peak = 0
        self.retired = set()
        self.retirement_sequence = 0
        self.gc_jobs = set()
        from kv_cache.agentrl_trace import profile_groups

        self.profile_catalog = (
            profile_groups(self.config.request_profile)
            if self.config.request_profile and self.config.request_profile["schema_version"] == 2
            else None
        )

    def owner_root(self, owner):
        return self.storage_dir / f"rank-{self.rank}" / f"rollout-{owner}"

    def setup(self):
        super().setup()
        self.owners = []
        for owner in range(self.settings.rollout_owners):
            root = self.owner_root(owner)
            self.owners.append(
                {
                    "kv": self._backend(root / "kv"),
                    "trajectory": self._backend(root / "trajectories"),
                    "prefixes": set(),
                    "prefix_locks": {},
                }
            )
            if self.config.kv_cache_model is not None and self.cache_owner_active(owner):
                self.cache_pools[owner] = self.make_cache_pool(owner, self.owners[owner]["kv"])

    def cache_owner_active(self, owner):
        return True

    async def cache_retire_files(self, state, keys):
        self.queue_retirement([(state.owner, key) for key in keys])

    def event(self, name, state=None, **data):
        if state:
            data = {
                "request": state.request,
                "group": state.group,
                "iteration": state.origin,
                "owner": f"rollout-{state.owner}",
                "policy": self.version,
                **(state.calibration or {}),
                **data,
            }
        self.trace.emit(name, **data)

    async def io(self, *args, **kwargs):
        # A cancelled to_thread future cannot stop the filesystem operation.
        # Drain its complete trace/metadata update before retiring the owner.
        pending = asyncio.create_task(super().io(*args, **kwargs))
        try:
            return await asyncio.shield(pending)
        except asyncio.CancelledError:
            await pending
            raise

    async def request_io(self, state, op, key, size=None, *, kind="kv", role="rollout", policy=None):
        backend = self.owners[state.owner]["kv" if kind == "kv" else "trajectory"]
        return await self.io(
            backend,
            op,
            key,
            size,
            kind=kind,
            request=state.request,
            policy=self.version if policy is None else policy,
            iteration=state.origin,
            owner=f"rollout-{state.owner}",
            role=role,
        )

    def factor(self, owner):
        return self.config.rank_rate_factors[owner % len(self.config.rank_rate_factors)]

    async def write_tokens(self, state, previous, count, label):
        for offset in range(0, count, self.config.chunk_tokens):
            tokens = min(self.config.chunk_tokens, count - offset)
            size = kv_delta_bytes(
                previous + offset, tokens, self.config.bytes_per_token, self.config.kv_offload_fraction
            )
            if size:
                key = f"v{self.version}-o{state.owner}-r{state.request}-{label}-{offset}"
                await self.request_io(state, "write", key, size)

    async def prepare_kv(self, state):
        if state.kv_version == self.version:
            return
        if state.kv_version is not None:
            self.event("rollout_resume", state, generated_tokens=state.generated, previous_policy=state.kv_version)
            self.event("re_prefill_begin", state, history_tokens=state.history)
            await asyncio.sleep(state.history / (self.config.prefill_tokens_per_second * self.factor(state.owner)))
            await self.write_tokens(state, 0, state.history, "re-prefill")
            self.event("re_prefill_end", state, history_tokens=state.history)
        else:
            prefix = state.shape["prefix_tokens"]
            owner = self.owners[state.owner]
            if prefix:
                key = f"v{self.version}-o{state.owner}-prefix-{state.shape['prefix_id']}"
                async with owner["prefix_locks"].setdefault(key, asyncio.Lock()):
                    size = kv_delta_bytes(0, prefix, self.config.bytes_per_token, self.config.kv_offload_fraction)
                    if key in owner["prefixes"]:
                        self.event("prefix_hit", state, key=key)
                        if size:
                            await self.request_io(state, "read", key)
                    else:
                        self.event("prefix_miss", state, key=key)
                        await asyncio.sleep(prefix / (self.config.prefill_tokens_per_second * self.factor(state.owner)))
                        if size:
                            await self.request_io(state, "write", key, size)
                        owner["prefixes"].add(key)
            suffix = state.shape["prompt_tokens"] - prefix
            if suffix:
                await asyncio.sleep(suffix / (self.config.prefill_tokens_per_second * self.factor(state.owner)))
                await self.write_tokens(state, prefix, suffix, "suffix")
        state.kv_version = self.version

    async def decode(self, state, tokens, turn, planned):
        queued = time.monotonic()
        self.event("generation_queued", state, tokens=tokens, turn=turn)
        async with self.decode_locks[state.owner]:
            target = tokens / (self.config.tokens_per_second * self.factor(state.owner))
            if self.config.request_profile:
                target = max(target, planned["decode_active_s"] * tokens / planned["generated_tokens"])
            started = time.monotonic()
            self.event(
                "generation_begin",
                state,
                tokens=tokens,
                turn=turn,
                compute_queue_wait_s=started - queued,
                modeled_compute_s=target,
            )
            await asyncio.sleep(target)
            duration = time.monotonic() - started
            self.event("generation_end", state, tokens=tokens, turn=turn)
            state.first_version = self.version if state.first_version is None else state.first_version
            state.last_version = self.version
        previous, position = state.history, state.generated
        # Completed decode remains observed progress even when its offload
        # write is cancelled; io() drains that real file operation.
        state.history += tokens
        state.generated += tokens
        if self.config.kv_cache_model is None:
            await self.write_tokens(state, previous, tokens, f"decode-{position}")
        return duration

    async def request(self, state):
        queued = time.monotonic()
        async with self.request_slots:
            self.active += 1
            self.active_peak = max(self.active_peak, self.active)
            started = time.monotonic()
            self.event("rollout_start", state, active=self.active, queue_wait_s=started - queued)
            observed = []
            try:
                await self.io(
                    self.prompts,
                    "read",
                    self.prompt_key(state.source_request, state.origin),
                    kind="prompt",
                    request=state.request,
                    policy=state.origin,
                    iteration=state.origin,
                    owner="dataset",
                    role="rollout",
                )
                for turn, planned in enumerate(state.shape["turns"]):
                    remaining, active_s = planned["generated_tokens"], 0.0
                    while remaining:
                        tokens = min(remaining, self.config.chunk_tokens)
                        async with self.gate.section():
                            if self.config.kv_cache_model is not None:
                                active_s += await self.cached_step(
                                    state, tokens, partial(self.decode, state, tokens, turn, planned)
                                )
                            else:
                                await self.prepare_kv(state)
                                active_s += await self.decode(state, tokens, turn, planned)
                        remaining -= tokens
                    tool_s = 0.0
                    if turn + 1 < len(state.shape["turns"]):
                        self.event("tool_begin", state, turn=turn)
                        tool_start = time.monotonic()
                        try:
                            await asyncio.sleep(planned["tool_delay_s"])
                        except asyncio.CancelledError:
                            self.event("tool_end", state, turn=turn, cancelled=True)
                            raise
                        tool_s = time.monotonic() - tool_start
                        self.event("tool_end", state, turn=turn)
                        async with self.gate.section():
                            if self.config.kv_cache_model is not None:
                                await self.cached_step(state, planned["observation_tokens"])
                            else:
                                await self.prepare_kv(state)
                                await self.write_tokens(
                                    state, state.history, planned["observation_tokens"], f"obs-{turn}"
                                )
                            state.history += planned["observation_tokens"]
                    observed.append({**planned, "decode_active_s": active_s, "tool_delay_s": tool_s})
                key = f"trajectory-g{state.group}-r{state.request}"
                if self.config.persist_trajectories:
                    await self.request_io(
                        state,
                        "write",
                        key,
                        state.shape["trajectory_bytes"],
                        kind="trajectory",
                        policy=state.last_version,
                    )
                ended = time.monotonic()
                state.record = {
                    **state.shape,
                    "rank": self.rank,
                    **(state.calibration or {}),
                    "turns": observed,
                    "request": state.request,
                    "group": state.group,
                    "owner": f"rollout-{state.owner}",
                    "iteration": state.origin,
                    "start_s": started - self.trace.start,
                    "end_s": ended - self.trace.start,
                    "policy_start": state.first_version,
                    "policy_end": state.last_version,
                    "prompt_policy": state.origin,
                    "trainer_policy_at_accept": None,
                    "disposition": "queued",
                    "trajectory_bytes": state.shape["trajectory_bytes"] if self.config.persist_trajectories else 0,
                }
                self.completed_requests.append(state.record)
                self.event(
                    "rollout_complete",
                    state,
                    generated_tokens=state.generated,
                    duration_s=ended - started,
                    policy=state.last_version,
                    active=self.active - 1,
                )
            except asyncio.CancelledError:
                self.event(
                    "request_interrupted",
                    state,
                    reason="cancelled",
                    generated_tokens=state.generated,
                    planned_generated_tokens=sum(t["generated_tokens"] for t in state.shape["turns"]),
                    history_tokens=state.history,
                    planned_shape=state.shape,
                    policy_start=state.first_version,
                    policy_end=state.last_version,
                )
                raise
            finally:
                if self.config.kv_cache_model is not None:
                    await self.retire_cache_request(state)
                self.active -= 1

    def build_group(self, group, origin, owner):
        members = []
        profile = self.config.request_profile
        if profile and profile["schema_version"] == 2:
            from kv_cache.agentrl_trace import GROUP_FIELDS, RECORD_FIELDS

            catalogs = self.profile_catalog
            catalog = catalogs[group % len(catalogs)]
            source = (group % len(catalogs)) * self.settings.group_size
            for record in catalog:
                identifier = group * self.settings.group_size + record["sibling"]
                shape = {k: record[k] for k in RECORD_FIELDS}
                state = RequestState(
                    identifier,
                    group,
                    record["owner"],
                    origin,
                    source,
                    shape,
                    shape["prompt_tokens"],
                    calibration={
                        "sibling": record["sibling"],
                        "calibration_source_rank": record["rank"],
                        **{f"calibration_{k}": record[k] for k in GROUP_FIELDS if k != "sibling"},
                    },
                )
                self.states[identifier] = state
                members.append(state)
            return members
        prompt_source = (group * self.settings.group_size) % self.config.requests_per_rank
        prompt_shape = self.request_shape(prompt_source, origin)
        prefix_id = prompt_shape["prefix_id"]
        if not self.config.request_profile and str(prefix_id).startswith("cold-"):
            prefix_id = f"cold-group-{group}"
        for sibling in range(self.settings.group_size):
            identifier = group * self.settings.group_size + sibling
            source = identifier % self.config.requests_per_rank
            shape = self.request_shape(source, origin)
            if not self.config.request_profile:
                shape = {**shape, "prefix_id": prefix_id}
            state = RequestState(identifier, group, owner, origin, prompt_source, shape, shape["prompt_tokens"])
            self.states[identifier] = state
            members.append(state)
        return members

    async def admit_group(self):
        pass

    async def execute_group(self, members):
        async with asyncio.TaskGroup() as children:
            for state in members:
                children.create_task(self.request(state))

    async def enqueue_terminal(self, group):
        self.groups[group]["status"] = "terminal"
        async with self.condition:
            self.condition.notify_all()
            if len(self.ready) == self.settings.queue_capacity:
                self.event("queue_backpressure", group=group, queue_depth=len(self.ready))
            await self.condition.wait_for(lambda: len(self.ready) < self.settings.queue_capacity)
            self.ready.append(group)
            self.queue_peak = max(self.queue_peak, len(self.ready))
            self.event(
                "group_ready", group=group, prompt_policy=self.groups[group]["origin"], queue_depth=len(self.ready)
            )
            self.condition.notify_all()

    async def producer(self):
        while True:
            await self.credits.acquire()
            await self.admit_group()
            group = self.next_group
            self.next_group += 1
            origin, owner = self.version, group % self.settings.rollout_owners
            self.groups[group] = {"origin": origin, "status": "running"}
            self.outstanding_peak = max(self.outstanding_peak, len(self.groups))
            members = self.build_group(group, origin, owner)
            self.event(
                "group_dispatch",
                group=group,
                prompt_policy=origin,
                owner=f"rollout-{members[0].owner}" if len({s.owner for s in members}) == 1 else None,
                members=[s.request for s in members],
                member_owners=[f"rollout-{s.owner}" for s in members],
            )
            await self.execute_group(members)
            await self.enqueue_terminal(group)

    def stale(self, group, *, terminal):
        return age_blocks(
            self.version - self.groups[group]["origin"] + 1,
            self.settings.max_prompt_age,
            self.settings.staleness_strategy,
            terminal=terminal,
        )

    async def dispose(self, group, disposition):
        for state in [s for s in self.states.values() if s.group == group]:
            if state.record:
                state.record["disposition"] = disposition
                state.record["trainer_policy_at_accept"] = self.version if disposition == "accepted" else None
            if self.config.persist_trajectories:
                key = f"trajectory-g{group}-r{state.request}"
                role = "trainer" if disposition == "accepted" else "gc"
                if disposition == "accepted" and state.record:
                    await self.request_io(state, "read", key, kind="trajectory", role=role, policy=state.last_version)
                if key in self.owners[state.owner]["trajectory"].metadata:
                    await self.request_io(state, "delete", key, kind="trajectory", role=role, policy=state.last_version)
            self.states.pop(state.request)

    async def sample(self):
        quota = self.settings.batch_groups
        while True:
            selected, dropped = [], []
            async with self.condition:
                dropped = [g for g in self.ready if self.stale(g, terminal=True)]
                for group in dropped:
                    self.ready.remove(group)
                    self.event("group_drop", group=group, prompt_age=self.version - self.groups[group]["origin"] + 1)
                    self.groups.pop(group)
                    self.credits.release()
                blocked = any(
                    v["status"] == "running" and self.stale(g, terminal=False) for g, v in self.groups.items()
                )
                if not dropped and not blocked and len(self.ready) >= quota:
                    selected = sorted(self.ready, key=lambda g: (self.groups[g]["origin"], g))[:quota]
                    for group in selected:
                        self.ready.remove(group)
                        self.event(
                            "group_sample", group=group, prompt_age=self.version - self.groups[group]["origin"] + 1
                        )
                        self.groups.pop(group)
                        self.credits.release()
                if dropped or selected:
                    self.condition.notify_all()
                else:
                    if blocked:
                        self.event("staleness_wait", policy=self.version)
                    await self.condition.wait()
            for group in dropped:
                await self.dispose(group, "dropped")
            if selected:
                return selected

    async def pause(self, reason):
        self.event("rollout_pause_begin", reason=reason, policy=self.version)
        await self.gate.pause()
        for state in self.states.values():
            if state.record is None and state.kv_version is not None:
                self.event("rollout_abort", state, generated_tokens=state.generated, reason=reason)
        self.event("rollout_pause_end", reason=reason, policy=self.version)

    def retire_kv(self):
        for pool in self.cache_pools.values():
            pool.invalidate()
        snapshot = []
        for owner, data in enumerate(self.owners):
            for key in list(data["kv"].metadata):
                identity = (owner, key)
                if identity not in self.retired:
                    self.retired.add(identity)
                    snapshot.append(identity)
            data["prefixes"].clear()
            data["prefix_locks"].clear()
        if not snapshot:
            return
        self.queue_retirement(snapshot)

    def queue_retirement(self, snapshot):
        if not snapshot:
            return
        self.retired.update(snapshot)
        retirement = self.retirement_sequence
        self.retirement_sequence += 1
        self.event(
            "kv_logical_invalidate",
            retirement_id=retirement,
            policy=self.version,
            keys=[k for _, k in snapshot],
            modeled_gc_delay_s=self.settings.kv_gc_delay_s,
        )
        task = self.jobs.create_task(self.gc(snapshot, retirement, self.version, self.trace.iteration))
        self.gc_jobs.add(task)
        task.add_done_callback(self.gc_jobs.discard)

    async def gc(self, snapshot, retirement, version, iteration):
        await asyncio.sleep(self.settings.kv_gc_delay_s)
        self.event("kv_gc_begin", retirement_id=retirement, policy=version, iteration=iteration)
        for owner, key in snapshot:
            await self.io(
                self.owners[owner]["kv"],
                "delete",
                key,
                kind="kv",
                policy=version,
                iteration=iteration,
                owner=f"rollout-{owner}",
                role="gc",
            )
            self.retired.discard((owner, key))
        self.event("kv_gc_end", retirement_id=retirement, policy=version, iteration=iteration)

    async def transition(self, version):
        self.event("weight_sync_begin", policy=version, initial=False)
        modeled = network_delay(self.config, self.config.weight_bytes)
        await asyncio.sleep(self.config.weight_sync_delay_s + modeled)
        self.version = version
        self.event(
            "weight_sync_end",
            policy=version,
            initial=False,
            modeled_network_s=modeled,
            modeled_delay_s=self.config.weight_sync_delay_s + modeled,
        )
        self.event("policy_install", policy=version, initial=False)
        self.gate.resume()

    async def controller(self):
        self.gate = PauseGate()
        self.condition = asyncio.Condition()
        self.credits = asyncio.Semaphore(self.settings.outstanding_groups)
        self.request_slots = asyncio.Semaphore(self.config.concurrency)
        self.decode_locks = [asyncio.Lock() for _ in self.owners]
        self.event("rollout_phase_begin", policy=self.version)
        async with asyncio.TaskGroup() as jobs:
            self.jobs = jobs
            producers = [jobs.create_task(self.producer()) for _ in range(self.settings.outstanding_groups)]
            try:
                for iteration in range(self.config.iterations):
                    self.trace.iteration = iteration
                    self.event("iteration_begin", policy=self.version)
                    for mini in range(self.settings.parameter_sync_step):
                        selected = await self.sample()
                        if self.config.trainer_mode == "colocate_async":
                            await self.pause("training")
                            self.retire_kv()
                        for group in selected:
                            await self.dispose(group, "accepted")
                        self.event("train_begin", policy=self.version, mini_batch=mini, owner="trainer")
                        await asyncio.sleep(self.config.train_delay_s / self.settings.parameter_sync_step)
                        self.event("train_end", policy=self.version + 1, mini_batch=mini, owner="trainer")
                    if self.config.checkpoint_every and (iteration + 1) % self.config.checkpoint_every == 0:
                        await self.async_checkpoint(iteration + 1)
                    if self.config.trainer_mode == "separate_async":
                        await self.pause("weight_sync")
                        self.retire_kv()
                    await self.transition(iteration + 1)
                    self.event("iteration_end", policy=self.version)
                await self.pause("run_end")
            finally:
                for task in producers:
                    task.cancel()
                await asyncio.gather(*producers, return_exceptions=True)
            for group in list(self.groups):
                await self.dispose(group, "run_end")
            self.groups.clear()
            self.ready.clear()
            self.retire_kv()
        self.event("rollout_phase_end", policy=self.version)

    def checkpoint_metadata(self):
        return {"lifecycle_mode": self.config.trainer_mode, "inflight_recoverable": False}

    def rank_summary(self):
        summary = super().rank_summary()
        decoded = sum(e["tokens"] for e in self.trace.events if e["event"] == "generation_end")
        begin = next(e["t_s"] for e in self.trace.events if e["event"] == "rollout_phase_begin")
        end = next(e["t_s"] for e in self.trace.events if e["event"] == "rollout_phase_end")
        return {
            **summary,
            "admission_worker_limit": self.config.concurrency,
            "decoded_tokens": decoded,
            "unfinished_generated_tokens": decoded - summary["generated_tokens"],
            "achieved_decode_tokens_per_s": decoded / (end - begin),
        }

    def summary_fields(self):
        return {
            "async": {
                "settings": asdict(self.settings),
                "profile_mapping": (
                    "group_catalog_owner"
                    if self.config.request_profile and self.config.request_profile["schema_version"] == 2
                    else "shared_rank_0_catalog"
                ),
                "terminal_queue_peak": self.queue_peak,
                "outstanding_groups_peak": self.outstanding_peak,
                "sampled_groups": sum(e["event"] == "group_sample" for e in self.trace.events),
                "dropped_groups": sum(e["event"] == "group_drop" for e in self.trace.events),
                "accepted_requests": sum(r["disposition"] == "accepted" for r in self.completed_requests),
                "partial_resume_events": sum(e["event"] == "rollout_resume" for e in self.trace.events),
                "re_prefill_tokens": sum(
                    e["history_tokens"] for e in self.trace.events if e["event"] == "re_prefill_end"
                ),
            },
            "topology": {
                "execution": "single_process_logical_owners",
                "rollout_owners": len(self.owners),
                "trainer_pools": 1,
                "storage_access": "shared_filesystem",
                "distributed_async": False,
            },
        }

    def run(self):
        self.phase("setup", self.setup)
        self.install_weights(0, initial=True)
        self.phase("async workload", lambda: asyncio.run(self.controller()))
        return self.publish_results()
