"""Role-aware MPI control for experimental shared-POSIX separated rollouts.

MPI carries control/trajectory descriptors, never emulated KV or weights.
Each process retains its own observation clock and real filesystem I/O.
"""

import asyncio
import os
import socket
import sys
import uuid
from dataclasses import asdict
from pathlib import Path

from kv_cache.agentrl import Trace
from kv_cache.agentrl_async import AsyncLifecycle, AsyncSettings, PauseGate, RequestState
from kv_cache.backends import NVMeBackend


class Endpoint:
    """One event-loop MPI owner; no blocking collective while actors run."""

    TAG = 17301

    def __init__(self, comm, jobs, handler, progress, timeout):
        self.comm, self.jobs, self.handler = comm, jobs, handler
        self.on_progress, self.timeout = progress, timeout
        self.sequence, self.pending, self.sends = 0, {}, []
        self.handlers = set()
        self.closed = False

    def post(self, peer, message):
        self.sends.append(self.comm.isend(message, dest=peer, tag=self.TAG))

    async def call(self, peer, command, payload=None, *, bounded=True):
        sequence = self.sequence
        self.sequence += 1
        future = asyncio.get_running_loop().create_future()
        self.pending[sequence] = (peer, future)
        self.post(peer, {"type": "call", "id": sequence, "command": command, "payload": payload})
        try:
            if bounded:
                return await asyncio.wait_for(asyncio.shield(future), timeout=self.timeout)
            return await future
        except TimeoutError as exc:
            raise RuntimeError(f"rank {peer} control RPC timeout: {command}") from exc
        finally:
            self.pending.pop(sequence, None)
            if not future.done():
                future.cancel()

    async def respond(self, peer, message):
        try:
            result = await self.handler(message["command"], message["payload"])
        except asyncio.CancelledError:
            self.post(peer, {"type": "reply", "id": message["id"], "error": "cancelled dispatch"})
            raise
        except Exception as exc:
            self.post(peer, {"type": "reply", "id": message["id"], "error": repr(exc)})
            raise
        else:
            self.post(peer, {"type": "reply", "id": message["id"], "value": result})

    async def pump(self):
        from mpi4py import MPI

        status = MPI.Status()
        while not self.closed or self.sends:
            self.sends = [request for request in self.sends if not request.Test()]
            for _ in range(64):
                if not self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=self.TAG, status=status):
                    break
                peer = status.Get_source()
                message = self.comm.recv(source=peer, tag=self.TAG)
                if message["type"] == "call":
                    task = self.jobs.create_task(self.respond(peer, message))
                    self.handlers.add(task)
                    task.add_done_callback(self.handlers.discard)
                elif message["type"] == "reply":
                    pending = self.pending.get(message["id"])
                    if pending:
                        expected, future = pending
                        if expected != peer or future.done():
                            raise RuntimeError("mismatched/duplicate MPI reply")
                        if "error" in message:
                            future.set_exception(RuntimeError(f"rank {peer}: {message['error']}"))
                        else:
                            future.set_result(message["value"])
                elif message["type"] == "progress":
                    self.on_progress(peer, message["payload"])
                else:
                    raise RuntimeError("unknown MPI control message")
            # This is MPI progress polling, not a modeled storage/service delay.
            await asyncio.sleep(0.001)

    async def finish(self):
        if self.handlers:
            await asyncio.gather(*list(self.handlers))
        self.closed = True


class RoleEngine(AsyncLifecycle):
    def cache_owner_active(self, owner):
        return self.rank == owner + 1

    def configure_role(self, rank, run_id, roles):
        self.rank, self.trace = rank, Trace(rank)
        self.shared_run_id, self.roles = run_id, roles
        self.execution_role = "trainer" if rank == 0 else "rollout"

    def broadcast(self, value):
        return self.shared_run_id

    def owner_root(self, owner):
        return self.storage_dir / f"rank-{owner + 1}" / f"rollout-{owner}"

    def request_shape(self, request, iteration):
        # The existing calibration contract is a shared rank-0 catalog.
        if self.config.request_profile and self.config.request_profile["schema_version"] == 1:
            records = self.config.request_profile["records"]
            return records[(iteration * self.config.requests_per_rank + request) % len(records)]
        return super().request_shape(request, iteration)

    def normalized_trace_fields(self):
        return {**super().normalized_trace_fields(), "execution_role": self.execution_role}

    def export_trace(self):
        return True

    def checkpoint_metadata(self):
        return {
            **super().checkpoint_metadata(),
            "execution_world_size": len(self.roles),
            "trainer_ranks": [0],
        }


class RolloutEngine(RoleEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retired_policy = None

    def io_context(self):
        return {"phase_token": self.phase_token}

    def io_completed(self, identity):
        if identity["phase_token"] is not None:
            self.endpoint.post(
                0,
                {
                    "type": "progress",
                    "payload": {
                        "token": identity["phase_token"],
                        **{k: identity[k] for k in ("io_id", "kind", "op", "owner", "role", "policy")},
                    },
                },
            )

    async def dispatch(self, descriptors):
        group = descriptors[0]["group"]
        if group in self.groups:
            raise ValueError("duplicate remote group dispatch")
        self.groups[group] = {"origin": descriptors[0]["origin"], "status": "running"}
        members = []
        for descriptor in descriptors:
            if descriptor["owner"] != self.rank - 1 or descriptor["origin"] > self.version:
                raise ValueError("wrong remote owner/future prompt policy")
            state = RequestState(**descriptor)
            state.shape = {**state.shape, "rank": self.rank}
            self.states[state.request] = state
            members.append(state)
        task = asyncio.current_task()
        self.dispatches.add(task)
        try:
            await super().execute_group(members)
        finally:
            self.dispatches.discard(task)
        self.groups[group]["status"] = "terminal"
        return [state.record for state in members]

    async def command(self, name, payload):
        if name == "dispatch":
            return await self.dispatch(payload)
        if name == "ready":
            return self.version
        if name == "observe":
            self.phase_token = payload
            self.event("phase_token_received", token=payload)
        elif name == "pause":
            await self.pause(payload)
        elif name == "retire":
            if self.gate.opened.is_set() or self.gate.busy:
                raise ValueError("KV retire requires a paused, drained owner")
            self.retire_kv()
            self.retired_policy = self.version
        elif name == "install":
            if payload != self.version + 1 or self.gate.opened.is_set() or self.retired_policy != self.version:
                raise ValueError("policy install requires paused, retired next-version owner")
            self.version = payload
            self.retired_policy = None
            self.event("policy_install", policy=payload, initial=False)
            self.gate.resume()
            return {"owner": self.rank - 1, "policy": self.version}
        elif name == "release":
            group, disposition, accepted = payload
            if self.groups[group]["status"] != "terminal":
                raise ValueError("cannot release incomplete prompt group")
            for state in [s for s in self.states.values() if s.group == group]:
                state.record["disposition"] = disposition
                state.record["trainer_policy_at_accept"] = accepted
                key = f"trajectory-g{group}-r{state.request}"
                # Trainer already read/deleted the shared file before this ack.
                self.owners[state.owner]["trajectory"].metadata.pop(key, None)
                self.states.pop(state.request)
            self.groups.pop(group)
        elif name == "stop":
            await self.pause("run_end")
            for task in list(self.dispatches):
                task.cancel()
            await asyncio.gather(*list(self.dispatches), return_exceptions=True)
            for group in list(self.groups):
                await self.dispose(group, "run_end")
            self.groups.clear()
            self.retire_kv()
            if self.gc_jobs:
                await asyncio.gather(*list(self.gc_jobs))
            self.stopped.set()
        else:
            raise ValueError(f"unknown rollout command: {name}")
        return self.version

    async def execute(self, comm):
        self.gate, self.phase_token = PauseGate(), None
        self.request_slots = asyncio.Semaphore(self.config.concurrency)
        self.decode_locks = [asyncio.Lock() for _ in self.owners]
        self.dispatches, self.stopped = set(), asyncio.Event()
        self.event("rollout_phase_begin", policy=self.version)
        try:
            async with asyncio.TaskGroup() as jobs:
                self.jobs = jobs
                self.endpoint = Endpoint(
                    comm, jobs, self.command, lambda peer, payload: None, self.settings.rpc_timeout_s
                )
                jobs.create_task(self.endpoint.pump())
                await self.stopped.wait()
                await self.endpoint.finish()
        finally:
            await self.drain_offloads("run_cleanup")
        self.event("rollout_phase_end", policy=self.version)


class TrainerEngine(RoleEngine):
    async def command(self, name, payload):
        raise ValueError("trainer cannot receive rollout commands")

    async def broadcast_command(self, name, payload=None):
        return await asyncio.gather(*(self.endpoint.call(rank, name, payload) for rank in range(1, len(self.roles))))

    async def admit_group(self):
        await self.admission.wait()

    async def execute_group(self, members):
        owners = {state.owner for state in members}
        subsets = await asyncio.gather(
            *(
                self.endpoint.call(
                    owner + 1, "dispatch", [asdict(s) for s in members if s.owner == owner], bounded=False
                )
                for owner in sorted(owners)
            )
        )
        records = [record for subset in subsets for record in subset]
        if len(records) != self.settings.group_size:
            raise ValueError("remote terminal group has incomplete siblings")
        by_request = {r["request"]: r for r in records}
        if len(by_request) != len(members) or set(by_request) != {s.request for s in members}:
            raise ValueError("remote group/request identity mismatch")
        for state in members:
            record = by_request[state.request]
            if record["request"] != state.request or record["group"] != state.group:
                raise ValueError("remote group/request identity mismatch")
            state.record, state.last_version = record, record["policy_end"]
            if self.config.persist_trajectories:
                key = f"trajectory-g{state.group}-r{state.request}"
                self.owners[state.owner]["trajectory"].metadata[key] = {"size": record["trajectory_bytes"]}
        self.event("remote_group_complete", group=members[0].group, worker_ranks=[o + 1 for o in sorted(owners)])

    async def dispose(self, group, disposition):
        owners = {s.owner for s in self.states.values() if s.group == group}
        await super().dispose(group, disposition)
        await asyncio.gather(
            *(
                self.endpoint.call(
                    owner + 1, "release", [group, disposition, self.version if disposition == "accepted" else None]
                )
                for owner in sorted(owners)
            )
        )

    async def pause(self, reason):
        self.admission.clear()
        self.event("rollout_pause_begin", reason=reason, policy=self.version)
        await self.broadcast_command("pause", reason)
        self.event("rollout_pause_end", reason=reason, policy=self.version)

    def progress(self, peer, payload):
        stage = self.observations.get(payload["token"])
        eligible = stage is not None
        self.event("remote_io_completion", worker_rank=peer, causal_overlap=eligible, **payload)
        if eligible:
            self.overlap[stage] += 1
            kind_op = f"{payload['kind']}_{payload['op']}"
            self.overlap_by_kind[stage][kind_op] = self.overlap_by_kind[stage].get(kind_op, 0) + 1

    def observe_begin(self, stage, token):
        self.observations[token] = stage
        return self.jobs.create_task(self.broadcast_command("observe", token))

    async def observe_end(self, token, acknowledgments):
        self.observations.pop(token)
        await acknowledgments
        await self.broadcast_command("observe", None)

    async def checkpoint_observation(self, begin):
        if begin:
            self.cp_token = f"checkpoint-{self.trace.iteration}"
            self.cp_observation = self.observe_begin("checkpoint", self.cp_token)
        else:
            await self.observe_end(self.cp_token, self.cp_observation)

    async def transition(self, version):
        self.event("weight_sync_begin", policy=version, initial=False)
        await asyncio.sleep(self.config.weight_sync_delay_s)
        responses = await self.broadcast_command("install", version)
        if {r["owner"] for r in responses} != set(range(self.settings.rollout_owners)) or any(
            r["policy"] != version for r in responses
        ):
            raise ValueError("incomplete policy install acknowledgments")
        self.install_acks += len(responses)
        for response in responses:
            self.event("policy_ack", worker_rank=response["owner"] + 1, policy=version)
        self.version = version
        self.event(
            "weight_sync_end",
            policy=version,
            initial=False,
            modeled_delay_s=self.config.weight_sync_delay_s,
            modeled_network_s=0.0,
            installed_owners=len(responses),
        )
        self.event("policy_install", policy=version, initial=False)
        if version < self.config.iterations:
            self.admission.set()

    async def execute(self, comm):
        self.condition = asyncio.Condition()
        self.credits = asyncio.Semaphore(self.settings.outstanding_groups)
        self.admission = asyncio.Event()
        self.admission.set()
        self.observations, self.overlap = {}, {"train": 0, "checkpoint": 0}
        self.overlap_by_kind = {"train": {}, "checkpoint": {}}
        self.install_acks = 0
        self.event("rollout_phase_begin", policy=self.version)
        async with asyncio.TaskGroup() as jobs:
            self.jobs = jobs
            self.endpoint = Endpoint(comm, jobs, self.command, self.progress, self.settings.rpc_timeout_s)
            jobs.create_task(self.endpoint.pump())
            versions = await self.broadcast_command("ready")
            if any(v != 0 for v in versions):
                raise ValueError("rollout owners are not initially ready")
            producers = [jobs.create_task(self.producer()) for _ in range(self.settings.outstanding_groups)]
            try:
                for iteration in range(self.config.iterations):
                    self.trace.iteration = iteration
                    self.event("iteration_begin", policy=self.version)
                    for mini in range(self.settings.parameter_sync_step):
                        selected = await self.sample()
                        for group in selected:
                            await self.dispose(group, "accepted")
                        self.event("train_begin", policy=self.version, mini_batch=mini, owner="trainer")
                        token = f"train-{iteration}-{mini}"
                        observation = self.observe_begin("train", token)
                        await asyncio.sleep(self.config.train_delay_s / self.settings.parameter_sync_step)
                        self.event("train_end", policy=self.version + 1, mini_batch=mini, owner="trainer")
                        await self.observe_end(token, observation)
                    if self.config.checkpoint_every and (iteration + 1) % self.config.checkpoint_every == 0:
                        await self.async_checkpoint(iteration + 1)
                    await self.pause("weight_sync")
                    await self.broadcast_command("retire")
                    await self.transition(iteration + 1)
                    self.event("iteration_end", policy=self.version)
                await self.pause("run_end")
            finally:
                for task in producers:
                    task.cancel()
                await asyncio.gather(*producers, return_exceptions=True)
            await self.broadcast_command("stop")
            self.groups.clear()
            self.states.clear()
            self.ready.clear()
            await self.endpoint.finish()
        self.event("rollout_phase_end", policy=self.version)

    def rank_summary(self):
        return {**super().rank_summary(), "admission_worker_limit": 0}

    def summary_fields(self):
        result = super().summary_fields()
        result["async"]["sampled_groups"] = sum(e["event"] == "group_sample" for e in self.trace.events)
        result["async"]["accepted_requests"] = result["async"]["sampled_groups"] * self.settings.group_size
        result["topology"] = {
            "execution": "mpi_shared",
            "roles": self.roles,
            "observed_host_domains": len({r["node"] for r in self.roles}),
            "physical_node_count_verified": False,
            "rollout_owners": self.settings.rollout_owners,
            "trainer_pools": 1,
            "storage_access": "shared_filesystem",
            "distributed_async": True,
            "global_clock_aligned": False,
            "per_rollout_process_request_limit": self.config.concurrency,
            "profile_mapping": result["async"]["profile_mapping"],
        }
        result["distributed"] = {
            "causal_io_overlap": self.overlap,
            "installed_policy_acks": self.install_acks,
            "causal_io_overlap_by_kind_op": self.overlap_by_kind,
        }
        return result


class MPILifecycle:
    """Collective setup/reporting around independently progressing roles."""

    def __init__(self, config, storage_root, results_dir, *, comm=None, resume=None, backend_factory=NVMeBackend):
        config.validate()
        self.settings = AsyncSettings.parse(config.async_workload, config.trainer_mode)
        if config.trainer_mode != "separate_async" or comm is None or comm.Get_size() < 2:
            raise ValueError("mpi_shared requires separate_async and at least two MPI ranks")
        if self.settings.rollout_owners != comm.Get_size() - 1:
            raise ValueError("mpi_shared rollout_owners must equal world_size - 1")
        if resume:
            raise ValueError("async resume is unsupported; pending/finished groups are not checkpointed")
        if config.network_mode != "none":
            raise ValueError("mpi_shared requires network_mode=none to avoid double-counting filesystem transport")
        self.comm, self.rank = comm, comm.Get_rank()
        self.config, self.storage_root, self.results_root = config, Path(storage_root), Path(results_dir)
        self.backend_factory = backend_factory

    def collectively(self, name, action):
        result, error = None, None
        try:
            result = action()
        except Exception as exc:
            error = f"rank {self.rank} {name}: {exc!r}"
        errors = [e for e in self.comm.allgather(error) if e]
        if errors:
            raise RuntimeError("; ".join(errors))
        return result

    def run(self):
        if len(set(self.comm.allgather(self.config.fingerprint))) != 1:
            raise ValueError("all MPI roles must use the same configuration")
        run_id = self.comm.bcast(f"agentrl-{uuid.uuid4().hex}" if self.rank == 0 else None, root=0)
        membership = self.comm.allgather({"rank": self.rank, "host": socket.gethostname(), "pid": os.getpid()})
        nodes = list(dict.fromkeys(r["host"] for r in membership))
        roles = [
            {
                "rank": r["rank"],
                "node": nodes.index(r["host"]),
                "pid": r["pid"],
                "role": "trainer" if r["rank"] == 0 else "rollout",
            }
            for r in membership
        ]
        cls = TrainerEngine if self.rank == 0 else RolloutEngine
        engine = cls(self.config, self.storage_root, self.results_root, backend_factory=self.backend_factory)
        engine.configure_role(self.rank, run_id, roles)
        self.collectively("role setup", engine.setup)
        self.result_dir = engine.result_dir

        def markers():
            for directory in (engine.storage_dir, engine.result_dir):
                (directory / f".mpi-probe-{self.rank}").write_text(f"{run_id}:{self.rank}")

        self.collectively("shared markers", markers)

        def visible():
            for directory in (engine.storage_dir, engine.result_dir):
                for rank in range(len(roles)):
                    if (directory / f".mpi-probe-{rank}").read_text() != f"{run_id}:{rank}":
                        raise ValueError("MPI roles do not share the same filesystem namespace")

        self.collectively("shared visibility", visible)
        self.collectively("initial weights", lambda: engine.install_weights(0, initial=True))
        try:
            asyncio.run(engine.execute(self.comm))
        except Exception as exc:
            print(f"agentrl MPI rank {self.rank}: {exc!r}", file=sys.stderr, flush=True)
            self.comm.Abort(1)
            raise
        # No role tasks remain. The existing reporter may now use collectives.
        engine.comm, engine.world = self.comm, len(roles)
        if self.rank != 0:
            engine.summary_fields = lambda: {}
        return engine.publish_results()
