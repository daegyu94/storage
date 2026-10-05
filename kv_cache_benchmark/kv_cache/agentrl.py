"""Experimental sync Agent RL lifecycle using the existing KV file backend.

No model execution occurs. Compute delays gate real I/O and collective phases.
This extension is not an MLPerf submission benchmark.
"""

import asyncio
import hashlib
import json
import math
import os
import random
import socket
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from kv_cache.agentrl_trace import validate_profile
from kv_cache.backends import NVMeBackend
from kv_cache.models import ModelConfig


@dataclass(frozen=True)
class AgentRLConfig:
    trainer_mode: str = "sync"
    request_profile: dict | None = None
    generation_rate_scope: str = "request"
    iterations: int = 2
    requests_per_rank: int = 4
    concurrency: int = 2
    seed: int = 42
    prompt_tokens: int = 16
    response_tokens: tuple = (16, 32)
    chunk_tokens: int = 8
    tokens_per_second: float = 100.0
    prefill_tokens_per_second: float = 1000.0
    rank_rate_factors: tuple = (1.0,)
    kv_offload_fraction: float = 0.0
    prefix_reuse_probability: float = 0.5
    hot_prefixes: int = 4
    prefix_skew: float = 1.0
    turns: int = 1
    tool_delay_s: float = 0.0
    observation_tokens: int = 0
    persist_trajectories: bool = True
    train_delay_s: float = 0.01
    checkpoint_every: int = 1
    checkpoint_bytes_per_rank: int = 4096
    weight_sync_delay_s: float = 0.0
    weight_bytes: int = 0
    network_mode: str = "none"
    nic_gbps: float = 100.0
    nic_efficiency: float = 0.8
    nic_rtt_us: float = 100.0
    nic_sharers: int = 1
    max_payload_bytes: int = 64 * 1024 * 1024
    model: dict = field(
        default_factory=lambda: {
            "name": "synthetic-smoke",
            "num_layers": 2,
            "hidden_dim": 16,
            "num_heads": 2,
            "kv_heads": 1,
            "dtype": "float16",
        }
    )

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise ValueError("configuration must be a mapping")
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
        config = cls(**values)
        config.validate()
        return config

    def validate(self):
        if self.trainer_mode != "sync":
            raise ValueError("v0.1 supports trainer_mode=sync only")
        if self.generation_rate_scope not in ("request", "owner"):
            raise ValueError("generation_rate_scope must be request or owner")
        if self.request_profile is not None:
            validate_profile(self.request_profile)
            if self.request_profile["mode"] != "sync":
                raise ValueError("only sync profiles are replayable; async profiles are analysis-only")
            if self.request_profile["provenance"].get("observation_boundary") != "host_file_api":
                raise ValueError("profile requires host_file_api observation boundary")
        if self.network_mode not in ("none", "estimate"):
            raise ValueError("network_mode must be none or estimate")
        if type(self.persist_trajectories) is not bool:
            raise ValueError("persist_trajectories must be a boolean")
        positive_ints = (
            "iterations",
            "requests_per_rank",
            "concurrency",
            "prompt_tokens",
            "chunk_tokens",
            "hot_prefixes",
            "turns",
            "nic_sharers",
            "max_payload_bytes",
        )
        nonnegative_ints = (
            "seed",
            "observation_tokens",
            "checkpoint_every",
            "checkpoint_bytes_per_rank",
            "weight_bytes",
        )
        for name in positive_ints + nonnegative_ints:
            value = getattr(self, name)
            minimum = 1 if name in positive_ints else 0
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in (
            "tokens_per_second",
            "prefill_tokens_per_second",
            "kv_offload_fraction",
            "prefix_reuse_probability",
            "prefix_skew",
            "tool_delay_s",
            "train_delay_s",
            "weight_sync_delay_s",
            "nic_gbps",
            "nic_efficiency",
            "nic_rtt_us",
        ):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
        for name in ("tokens_per_second", "prefill_tokens_per_second", "nic_gbps", "nic_efficiency"):
            if getattr(self, name) == 0:
                raise ValueError(f"{name} must be positive")
        for name in ("kv_offload_fraction", "prefix_reuse_probability", "nic_efficiency"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must be <= 1")
        for name in ("response_tokens", "rank_rate_factors"):
            values = getattr(self, name)
            if not isinstance(values, (list, tuple)) or not values:
                raise ValueError(f"{name} must be a nonempty sequence")
            for value in values:
                if name == "response_tokens":
                    valid = type(value) is int and value > 0
                else:
                    valid = type(value) in (float, int) and math.isfinite(value) and value > 0
                if not valid:
                    raise ValueError(f"invalid value in {name}")
        for factor in self.rank_rate_factors:
            for rate in (self.tokens_per_second, self.prefill_tokens_per_second):
                effective = rate * factor
                if effective == 0 or not math.isfinite(effective):
                    raise ValueError("effective generation/prefill rate must be finite and positive")
        if self.network_mode == "estimate":
            try:
                bandwidth = self.nic_gbps * 1e9 / 8 * self.nic_efficiency / self.nic_sharers
                valid = (
                    bandwidth > 0
                    and math.isfinite(bandwidth)
                    and math.isfinite(max(self.max_payload_bytes, self.weight_bytes) / bandwidth)
                )
            except (OverflowError, ZeroDivisionError):
                valid = False
            if not valid:
                raise ValueError("effective network bandwidth/delay must be finite and positive")
        if not isinstance(self.model, dict):
            raise ValueError("model must be a mapping")
        try:
            model = ModelConfig(**self.model)
        except TypeError as exc:
            raise ValueError(f"invalid model fields: {exc}") from exc
        for name in ("num_layers", "hidden_dim", "num_heads", "kv_heads"):
            if type(getattr(model, name)) is not int or getattr(model, name) <= 0:
                raise ValueError(f"invalid model {name}")
        if model.hidden_dim % model.num_heads or model.num_heads % model.kv_heads:
            raise ValueError("model heads must divide hidden_dim and num_heads")
        if model.dtype not in ("float32", "float16", "bfloat16", "int8"):
            raise ValueError("unsupported KV dtype")
        if model.attention_type not in ("mha", "gqa", "mla"):
            raise ValueError("unsupported attention_type")
        for name in ("_kv_dim_override", "kv_lora_rank", "qk_rope_head_dim"):
            if type(getattr(model, name)) is not int or getattr(model, name) < 0:
                raise ValueError(f"invalid model {name}")
        if model.attention_type == "mla" and model.kv_lora_rank == 0:
            raise ValueError("MLA requires a positive kv_lora_rank")
        largest = max(
            self.checkpoint_bytes_per_rank,
            self.prompt_tokens * 4,
            int(
                max(self.prompt_tokens, self.chunk_tokens, self.observation_tokens)
                * self.bytes_per_token
                * self.kv_offload_fraction
            ),
            (self.prompt_tokens + max(self.response_tokens) + (self.turns - 1) * self.observation_tokens) * 12,
        )
        if self.request_profile:
            for record in self.request_profile["records"]:
                largest = max(
                    largest,
                    record["prompt_tokens"] * 4,
                    record["trajectory_bytes"],
                    int(
                        max(
                            record["prefix_tokens"],
                            self.chunk_tokens,
                            max(t["observation_tokens"] for t in record["turns"]),
                        )
                        * self.bytes_per_token
                        * self.kv_offload_fraction
                    ),
                )
        if largest > self.max_payload_bytes:
            raise ValueError("payload exceeds max_payload_bytes; use smaller chunks or an explicit memory budget")

    @property
    def bytes_per_token(self):
        return ModelConfig(**self.model).kv_cache_size_per_token

    @property
    def fingerprint(self):
        values = asdict(self)
        if self.request_profile is None:
            values.pop("request_profile")  # Keep v0.1 checkpoint fingerprints compatible.
        if self.generation_rate_scope == "request":
            values.pop("generation_rate_scope")
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def kv_delta_bytes(previous_tokens, new_tokens, bytes_per_token, fraction):
    """Conserve bytes across chunk boundaries, including fractional offload."""
    return int((previous_tokens + new_tokens) * bytes_per_token * fraction) - int(
        previous_tokens * bytes_per_token * fraction
    )


def network_delay(config, size_bytes):
    """Optional serialization+RTT estimate; never a measured NIC latency."""
    if config.network_mode == "none":
        return 0.0
    bandwidth = config.nic_gbps * 1e9 / 8 * config.nic_efficiency / config.nic_sharers
    return config.nic_rtt_us * 1e-6 + size_bytes / bandwidth


class Trace:
    def __init__(self, rank):
        self.rank = rank
        self.start = time.monotonic()
        self.iteration = -1
        self.events = []

    def emit(self, event, **data):
        self.events.append(
            {
                "seq": len(self.events),
                "rank": self.rank,
                "iteration": self.iteration,
                "t_s": time.monotonic() - self.start,
                "event": event,
                **data,
            }
        )


class SyncLifecycle:
    def __init__(self, config, storage_root, results_dir, *, comm=None, resume=None, backend_factory=NVMeBackend):
        config.validate()
        self.config = config
        self.comm = comm
        self.rank = comm.Get_rank() if comm else 0
        self.world = comm.Get_size() if comm else 1
        self.trace = Trace(self.rank)
        self.resume = Path(resume) if resume else None
        self.storage_root = Path(storage_root)
        self.results_root = Path(results_dir)
        self.backend_factory = backend_factory
        self.active = self.active_peak = 0
        self.version = 0
        self.start_iteration = 0
        self.prefixes = set()
        self.prefix_locks = {}
        self.io_sequence = self.io_active = self.io_active_peak = 0
        self.kv_sizes = {}
        self.kv_peak_bytes = 0
        self.completed_requests = []

    def gather_all(self, value):
        return self.comm.allgather(value) if self.comm else [value]

    def broadcast(self, value):
        return self.comm.bcast(value, root=0) if self.comm else value

    def phase(self, name, action):
        """All ranks exchange errors before entering the next phase."""
        result, error = None, None
        try:
            result = action()
        except Exception as exc:
            error = f"rank {self.rank} {name}: {exc!r}"
        errors = [e for e in self.gather_all(error) if e]
        if errors:
            raise RuntimeError("; ".join(errors))
        return result

    def _backend(self, path):
        return self.backend_factory(str(path), preserve_existing=True)

    def setup(self):
        if len(set(self.gather_all(self.config.fingerprint))) != 1:
            raise RuntimeError("all ranks must use the same configuration")
        run_id = self.broadcast(f"agentrl-{uuid.uuid4().hex}" if self.rank == 0 else None)
        self.storage_dir = self.storage_root / run_id
        self.result_dir = self.results_root / run_id
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        rank_dir = self.storage_dir / f"rank-{self.rank}"
        self.kv = self._backend(rank_dir / "kv")
        self.prompts = self._backend(rank_dir / "prompts")
        self.trajectories = self._backend(rank_dir / "trajectories")
        steps = range(self.config.iterations) if self.config.request_profile else (0,)
        for iteration in steps:
            for request in range(self.config.requests_per_rank):
                key = self.prompt_key(request, iteration)
                shape = self.request_shape(request, iteration)
                self.prompts.write(key, self.payload(key, shape["prompt_tokens"] * 4))
        if self.rank == 0:
            self.write_json(self.result_dir / "config.json", asdict(self.config))

    def payload(self, key, size):
        if size > self.config.max_payload_bytes:
            raise ValueError("payload exceeds max_payload_bytes")
        seed = int.from_bytes(
            hashlib.sha256(f"{self.config.seed}:{self.rank}:{self.trace.iteration}:{key}".encode()).digest()[:8],
            "little",
        )
        return np.random.default_rng(seed).integers(0, 256, size=size, dtype=np.uint8)

    async def io(
        self, backend, op, key, size=0, *, kind, request=None, policy=None, iteration=None, owner=None, role=None
    ):
        if op == "read" and not size:
            size = backend.metadata[key]["size"]
        data = self.payload(key, size) if op == "write" else None
        identity = {
            "io_id": self.io_sequence,
            "op": op,
            "key": key,
            "kind": kind,
            "request": request,
            "policy": self.version if policy is None else policy,
            "iteration": self.trace.iteration if iteration is None else iteration,
            "owner": owner,
            "role": role,
        }
        self.io_sequence += 1
        self.io_active += 1
        self.io_active_peak = max(self.io_active_peak, self.io_active)
        self.trace.emit("io_begin", **identity, bytes=size, io_active=self.io_active)
        started = time.monotonic()
        try:
            if op == "write":
                timing = await asyncio.to_thread(backend.write, key, data)
            elif op == "read":
                data, timing = await asyncio.to_thread(backend.read, key)
                if data.nbytes != size:
                    raise ValueError("read payload size differs from arrival size")
            elif op == "delete":
                await asyncio.to_thread(backend.delete, key)
                timing = None
            else:
                raise ValueError(f"unsupported I/O operation: {op}")
            actual_s = time.monotonic() - started
            if kind == "kv":
                if op == "write":
                    self.kv_sizes[key] = size
                elif op == "delete":
                    self.kv_sizes.pop(key, None)
                self.kv_peak_bytes = max(self.kv_peak_bytes, sum(self.kv_sizes.values()))
            # File length includes the NumPy header, not device or wire traffic.
            physical = backend._get_path(key).stat().st_size if op != "delete" else 0
            measured = {
                "bytes": size,
                "actual_s": actual_s,
                "backend_s": timing.total if timing else None,
                "physical_file_bytes": physical,
                "kv_live_payload_bytes": sum(self.kv_sizes.values()),
            }
            self.trace.emit("io_actual_end", **identity, **measured)
            estimated_s = network_delay(self.config, size)
            await asyncio.sleep(estimated_s)
            self.trace.emit(
                "io_end",
                **identity,
                **measured,
                modeled_network_s=estimated_s,
                duration_s=time.monotonic() - started,
            )
        finally:
            self.io_active -= 1
        return data

    def prompt_key(self, request, iteration):
        return f"prompt-i{iteration}-r{request}" if self.config.request_profile else f"prompt-{request}"

    def request_shape(self, request, iteration):
        config = self.config
        if config.request_profile:
            records = [r for r in config.request_profile["records"] if r["rank"] == self.rank]
            if not records:
                raise ValueError(f"profile has no records for rank {self.rank}")
            return records[(iteration * config.requests_per_rank + request) % len(records)]
        rng = random.Random(config.seed + self.rank * 1_000_003 + iteration * 1009 + request)
        response = config.response_tokens[request % len(config.response_tokens)]
        factor = config.rank_rate_factors[self.rank % len(config.rank_rate_factors)]
        hot = rng.random() < config.prefix_reuse_probability
        prefix = (
            str(
                rng.choices(
                    range(config.hot_prefixes),
                    weights=[(i + 1) ** -config.prefix_skew for i in range(config.hot_prefixes)],
                )[0]
            )
            if hot
            else f"cold-{iteration}-{request}"
        )
        turns = []
        for turn in range(config.turns):
            tokens = response // config.turns + (turn < response % config.turns)
            last = turn + 1 == config.turns
            turns.append(
                {
                    "generated_tokens": tokens,
                    "decode_active_s": tokens / (config.tokens_per_second * factor),
                    "tool_delay_s": 0 if last else config.tool_delay_s,
                    "observation_tokens": 0 if last else config.observation_tokens,
                }
            )
        return {
            "rank": self.rank,
            "prompt_tokens": config.prompt_tokens,
            "prefix_tokens": config.prompt_tokens,
            "prefix_id": prefix,
            "turns": turns,
            "trajectory_bytes": (config.prompt_tokens + response + (config.turns - 1) * config.observation_tokens) * 12,
        }

    async def rollout(self, request, semaphore):
        queued = self.rollout_offered_at
        async with semaphore:
            self.active += 1
            self.active_peak = max(self.active_peak, self.active)
            try:
                await self.rollout_body(request, queued)
            finally:
                self.active -= 1

    async def rollout_body(self, request, queued):
        config = self.config
        request_start = time.monotonic()
        shape = self.request_shape(request, self.trace.iteration)
        factor = config.rank_rate_factors[self.rank % len(config.rank_rate_factors)]
        self.trace.emit(
            "rollout_start",
            request=request,
            policy=self.version,
            active=self.active,
            prompt_tokens=shape["prompt_tokens"],
            prefix_tokens=shape["prefix_tokens"],
            queue_wait_s=request_start - queued,
        )
        await self.io(
            self.prompts, "read", self.prompt_key(request, self.trace.iteration), kind="prompt", request=request
        )
        prefix_tokens = shape["prefix_tokens"]
        if prefix_tokens:
            prefix_key = f"v{self.version}-prefix-{shape['prefix_id']}"
            lock = self.prefix_locks.setdefault(prefix_key, asyncio.Lock())
            prefix_bytes = kv_delta_bytes(0, prefix_tokens, config.bytes_per_token, config.kv_offload_fraction)
            async with lock:
                if prefix_key in self.prefixes:
                    self.trace.emit("prefix_hit", request=request, key=prefix_key, policy=self.version)
                    if prefix_bytes:
                        await self.io(self.kv, "read", prefix_key, kind="kv", request=request)
                else:
                    self.trace.emit("prefix_miss", request=request, key=prefix_key, policy=self.version)
                    await asyncio.sleep(prefix_tokens / (config.prefill_tokens_per_second * factor))
                    if prefix_bytes:
                        await self.io(self.kv, "write", prefix_key, prefix_bytes, kind="kv", request=request)
                    self.prefixes.add(prefix_key)
        suffix = shape["prompt_tokens"] - prefix_tokens
        if suffix:
            await asyncio.sleep(suffix / (config.prefill_tokens_per_second * factor))
            size = kv_delta_bytes(prefix_tokens, suffix, config.bytes_per_token, config.kv_offload_fraction)
            if size:
                await self.io(self.kv, "write", f"v{self.version}-r{request}-suffix", size, kind="kv", request=request)
        history = shape["prompt_tokens"]
        generated = chunk_id = 0
        observed_turns = []
        for turn, planned in enumerate(shape["turns"]):
            remaining = planned["generated_tokens"]
            active_s = 0.0
            while remaining:
                tokens = min(config.chunk_tokens, remaining)
                delay = planned["decode_active_s"] * tokens / planned["generated_tokens"]
                active_s += await self.generate(request, tokens, turn, delay)
                size = kv_delta_bytes(history, tokens, config.bytes_per_token, config.kv_offload_fraction)
                if size:
                    await self.io(
                        self.kv,
                        "write",
                        f"v{self.version}-r{request}-chunk-{chunk_id}",
                        size,
                        kind="kv",
                        request=request,
                    )
                history += tokens
                generated += tokens
                chunk_id += 1
                remaining -= tokens
            tool_s = 0.0
            if turn + 1 < len(shape["turns"]):
                self.trace.emit("tool_begin", request=request, turn=turn)
                started = time.monotonic()
                await asyncio.sleep(planned["tool_delay_s"])
                tool_s = time.monotonic() - started
                self.trace.emit("tool_end", request=request, turn=turn)
                size = kv_delta_bytes(
                    history, planned["observation_tokens"], config.bytes_per_token, config.kv_offload_fraction
                )
                if size:
                    await self.io(
                        self.kv, "write", f"v{self.version}-r{request}-obs-{turn}", size, kind="kv", request=request
                    )
                history += planned["observation_tokens"]
            observed_turns.append({**planned, "decode_active_s": active_s, "tool_delay_s": tool_s})
        if config.persist_trajectories:
            await self.io(
                self.trajectories,
                "write",
                f"v{self.version}-r{request}",
                shape["trajectory_bytes"],
                kind="trajectory",
                request=request,
            )
        ended = time.monotonic()
        self.completed_requests.append(
            {
                **shape,
                "turns": observed_turns,
                "trajectory_bytes": shape["trajectory_bytes"] if config.persist_trajectories else 0,
                "iteration": self.trace.iteration,
                "request": request,
                "start_s": request_start - self.trace.start,
                "end_s": ended - self.trace.start,
                "policy_start": self.version,
                "policy_end": self.version,
                "prompt_policy": self.version,
                "trainer_policy_at_accept": self.version,
            }
        )
        self.trace.emit(
            "rollout_complete",
            request=request,
            policy=self.version,
            generated_tokens=generated,
            duration_s=ended - request_start,
            active=self.active - 1,
        )

    async def generate(self, request, tokens, turn, delay):
        queued = time.monotonic()
        self.trace.emit("generation_queued", request=request, tokens=tokens, turn=turn, policy=self.version)
        if self.config.generation_rate_scope == "owner":
            factor = self.config.rank_rate_factors[self.rank % len(self.config.rank_rate_factors)]
            delay = max(delay, tokens / (self.config.tokens_per_second * factor))

        async def service():
            started = time.monotonic()
            self.trace.emit(
                "generation_begin",
                request=request,
                tokens=tokens,
                turn=turn,
                policy=self.version,
                modeled_compute_s=delay,
                compute_queue_wait_s=started - queued,
            )
            await asyncio.sleep(delay)
            active_s = time.monotonic() - started
            self.trace.emit("generation_end", request=request, tokens=tokens, turn=turn, policy=self.version)
            return active_s

        if self.config.generation_rate_scope == "owner":
            async with self.decode_lock:
                return await service()
        return await service()

    async def rollouts(self):
        self.rollout_offered_at = time.monotonic()
        self.decode_lock = asyncio.Lock()  # Each iteration uses a fresh asyncio.run loop.
        self.trace.emit("rollout_phase_begin", policy=self.version)
        semaphore = asyncio.Semaphore(self.config.concurrency)
        requests = iter(range(self.config.requests_per_rank))

        async def worker():
            for request in requests:
                await self.rollout(request, semaphore)

        # TaskGroup drains/cancels peers on failure before the rank exchanges errors.
        async with asyncio.TaskGroup() as group:
            for _ in range(min(self.config.concurrency, self.config.requests_per_rank)):
                group.create_task(worker())
        self.trace.emit("rollout_phase_end", policy=self.version)

    @staticmethod
    def write_json(path, data):
        temporary = path.with_suffix(".tmp")
        with temporary.open("w") as stream:
            json.dump(data, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    async def async_phase(self, name, action):
        result, error = None, None
        try:
            result = await action()
        except Exception as exc:
            error = f"rank {self.rank} {name}: {exc!r}"
        errors = [e for e in self.gather_all(error) if e]
        if errors:
            raise RuntimeError("; ".join(errors))
        return result

    def checkpoint(self, next_iteration):
        return asyncio.run(self.async_checkpoint(next_iteration))

    def checkpoint_metadata(self):
        return {}

    async def async_checkpoint(self, next_iteration):
        self.trace.emit("checkpoint_begin", policy=next_iteration)
        step_dir = self.storage_dir / "checkpoints" / f"step-{next_iteration}"

        async def save_shard():
            backend = self._backend(step_dir / f"rank-{self.rank}")
            data = await self.io(
                backend,
                "write",
                "state",
                self.config.checkpoint_bytes_per_rank,
                kind="checkpoint",
                policy=next_iteration,
                owner="trainer",
                role="trainer",
            )
            return {
                "rank": self.rank,
                "path": f"rank-{self.rank}/state.npy",
                "bytes": data.nbytes,
                "sha256": hashlib.sha256(data.tobytes()).hexdigest(),
            }

        shard = await self.async_phase("checkpoint shard", save_shard)
        manifest = {
            "schema_version": 1,
            "policy_version": next_iteration,
            "next_iteration": next_iteration,
            "world_size": self.world,
            "config_sha256": self.config.fingerprint,
            "shards": self.gather_all(shard),
            **self.checkpoint_metadata(),
        }

        async def commit():
            if self.rank == 0:
                await asyncio.to_thread(self.write_json, step_dir / "manifest.json", manifest)

        await self.async_phase("checkpoint commit", commit)
        self.trace.emit("checkpoint_commit", policy=next_iteration)
        self.trace.emit("checkpoint_end", policy=next_iteration)

    def recover(self):
        self.trace.emit("recovery_begin")
        data = self.phase(
            "checkpoint manifest", lambda: json.loads(self.resume.read_text()) if self.rank == 0 else None
        )
        data = self.broadcast(data)

        def validate_load():
            if data.get("schema_version") != 1 or data.get("world_size") != self.world:
                raise ValueError("checkpoint schema/world size mismatch")
            if data.get("config_sha256") != self.config.fingerprint:
                raise ValueError("checkpoint config mismatch")
            next_step = data.get("next_iteration")
            if (
                type(next_step) is not int
                or not 0 <= next_step <= self.config.iterations
                or data.get("policy_version") != next_step
            ):
                raise ValueError("checkpoint iteration/version mismatch")
            if len(data.get("shards", [])) != self.world:
                raise ValueError("checkpoint shard count mismatch")
            shard = data["shards"][self.rank]
            expected_path = f"rank-{self.rank}/state.npy"
            if shard.get("rank") != self.rank or shard.get("path") != expected_path:
                raise ValueError("checkpoint shard ownership/path mismatch")
            path = self.resume.parent / expected_path
            # Validate allocation geometry before np.load materializes the array.
            with path.open("rb") as stream:
                header_version = np.lib.format.read_magic(stream)
                if header_version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                elif header_version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("unsupported checkpoint header version")
                expected_bytes = self.config.checkpoint_bytes_per_rank
                if (
                    shape != (expected_bytes,)
                    or dtype != np.dtype("uint8")
                    or fortran
                    or path.stat().st_size != stream.tell() + expected_bytes
                ):
                    raise ValueError("checkpoint header/physical size mismatch")
            backend = self._backend(self.resume.parent / f"rank-{self.rank}")
            payload = asyncio.run(
                self.io(
                    backend,
                    "read",
                    "state",
                    expected_bytes,
                    kind="checkpoint",
                    policy=next_step,
                )
            )
            if payload.nbytes != self.config.checkpoint_bytes_per_rank or payload.nbytes != shard.get("bytes"):
                raise ValueError("checkpoint byte mismatch")
            if hashlib.sha256(payload.tobytes()).hexdigest() != shard.get("sha256"):
                raise ValueError("checkpoint checksum mismatch")
            self.start_iteration = self.version = next_step

        self.phase("checkpoint load", validate_load)
        self.trace.emit("recovery_end", policy=self.version)

    async def invalidate(self):
        for key in list(self.kv.metadata):
            await self.io(self.kv, "delete", key, kind="kv")
        self.prefixes.clear()
        self.prefix_locks.clear()
        self.trace.emit("kv_invalidate", policy=self.version)

    def rank_summary(self):
        totals = {}
        for event in self.trace.events:
            if event["event"] == "io_end":
                name = f"{event['kind']}_{event['op']}"
                value = totals.setdefault(name, {"ops": 0, "payload_bytes": 0})
                value["ops"] += 1
                value["payload_bytes"] += event["bytes"]
        generated_tokens = sum(e["generated_tokens"] for e in self.trace.events if e["event"] == "rollout_complete")
        rollout_s = sum(
            end["t_s"] - begin["t_s"]
            for begin, end in zip(
                [e for e in self.trace.events if e["event"] == "rollout_phase_begin"],
                [e for e in self.trace.events if e["event"] == "rollout_phase_end"],
                strict=True,
            )
        )
        return {
            "rank": self.rank,
            "host": socket.gethostname(),
            "local_rank": os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", os.environ.get("MPI_LOCALRANKID", "0")),
            "active_peak": self.active_peak,
            "io_active_peak": self.io_active_peak,
            "admission_worker_limit": min(self.config.concurrency, self.config.requests_per_rank),
            "achieved_decode_tokens_per_s": generated_tokens / rollout_s if rollout_s else 0,
            "compute_queue_wait_s": sum(e.get("compute_queue_wait_s", 0) for e in self.trace.events),
            "modeled_decode_service_s": sum(e.get("modeled_compute_s", 0) for e in self.trace.events),
            "kv_live_payload_bytes": sum(self.kv_sizes.values()),
            "kv_peak_payload_bytes": self.kv_peak_bytes,
            "prefix_hits": sum(e["event"] == "prefix_hit" for e in self.trace.events),
            "prefix_misses": sum(e["event"] == "prefix_miss" for e in self.trace.events),
            "generated_tokens": sum(
                e.get("generated_tokens", 0) for e in self.trace.events if e["event"] == "rollout_complete"
            ),
            "iteration_elapsed_s": sum(
                end["t_s"] - begin["t_s"]
                for begin, end in zip(
                    [e for e in self.trace.events if e["event"] == "iteration_begin"],
                    [e for e in self.trace.events if e["event"] == "iteration_end"],
                    strict=True,
                )
            ),
            "io": totals,
            "elapsed_s": time.monotonic() - self.trace.start,
            "modeled_network_s": sum(e.get("modeled_network_s", 0) for e in self.trace.events),
            "barrier_wait_s": sum(e.get("duration_s", 0) for e in self.trace.events if e["event"] == "rollout_barrier"),
        }

    def install_weights(self, version, *, initial=False):
        self.trace.emit("weight_sync_begin", policy=version, initial=initial)
        modeled_s = network_delay(self.config, self.config.weight_bytes)
        delay = self.config.weight_sync_delay_s + modeled_s
        self.phase("weight sync", lambda: time.sleep(delay))
        self.version = version
        self.trace.emit(
            "weight_sync_end",
            policy=version,
            initial=initial,
            modeled_delay_s=delay,
            base_delay_s=self.config.weight_sync_delay_s,
            modeled_network_s=modeled_s,
        )
        self.trace.emit("policy_install", policy=version, initial=initial)

    def run(self):
        self.phase("setup", self.setup)
        if self.resume:
            self.recover()
        self.install_weights(self.version, initial=True)
        for iteration in range(self.start_iteration, self.config.iterations):
            self.trace.iteration = iteration
            self.trace.emit("iteration_begin", policy=self.version)
            started = time.monotonic()
            self.phase("rollout", lambda: asyncio.run(self.rollouts()))
            self.trace.emit(
                "rollout_barrier",
                duration_s=time.monotonic()
                - started
                - (
                    next(e["t_s"] for e in reversed(self.trace.events) if e["event"] == "rollout_phase_end")
                    - next(e["t_s"] for e in reversed(self.trace.events) if e["event"] == "rollout_phase_begin")
                ),
            )
            if self.config.persist_trajectories:

                async def consume_trajectories():
                    for request in range(self.config.requests_per_rank):
                        await self.io(
                            self.trajectories, "read", f"v{self.version}-r{request}", kind="trajectory", request=request
                        )

                self.phase("trajectory consume", lambda: asyncio.run(consume_trajectories()))
            self.trace.emit("train_begin", policy=self.version)
            self.phase("training", lambda: time.sleep(self.config.train_delay_s))
            self.trace.emit("train_end", policy=self.version + 1)
            if self.config.checkpoint_every and (iteration + 1) % self.config.checkpoint_every == 0:
                self.checkpoint(iteration + 1)
            self.install_weights(iteration + 1)
            self.phase("KV invalidation", lambda: asyncio.run(self.invalidate()))
            self.trace.emit("iteration_end", policy=self.version)
        return self.publish_results()

    def summary_fields(self):
        return {}

    def publish_results(self):
        ranks = self.gather_all(self.rank_summary())
        latencies = self.gather_all(
            [(f"{e['kind']}_{e['op']}", e["actual_s"]) for e in self.trace.events if e["event"] == "io_end"]
        )
        flat_latencies = [v for values in latencies for _, v in values]
        by_kind = {}
        for values in latencies:
            for kind, value in values:
                by_kind.setdefault(kind, []).append(value)

        def quantiles(values):
            return dict(zip(("p50", "p95", "p99"), np.percentile(values, (50, 95, 99)).tolist(), strict=True))

        totals = {}
        for rank in ranks:
            for kind, value in rank["io"].items():
                total = totals.setdefault(kind, {"ops": 0, "payload_bytes": 0})
                total["ops"] += value["ops"]
                total["payload_bytes"] += value["payload_bytes"]
        summary = {
            "schema_version": 1,
            "status": "complete",
            "benchmark": "experimental-agentrl",
            "fidelity": "uncalibrated",
            "trainer_mode": self.config.trainer_mode,
            "world_size": self.world,
            "final_policy_version": self.version,
            "config_sha256": self.config.fingerprint,
            "network_mode": self.config.network_mode,
            "ranks": ranks,
            "wall_clock_s": max(rank["elapsed_s"] for rank in ranks),
            "io_totals": totals,
            "latency_by_kind_op_s": {k: quantiles(v) for k, v in by_kind.items()},
            "io_latency_s": quantiles(flat_latencies) if flat_latencies else {},
            **self.summary_fields(),
        }
        self.phase("results", lambda: self.write_json(self.result_dir / f"rank-{self.rank}.json", self.trace.events))
        if self.completed_requests:
            provenance = {
                "kind": "poc",
                "run_id": self.result_dir.name,
                "revision": "source-sha256:"
                + hashlib.sha256(
                    b"".join(
                        (Path(__file__).parent / name).read_bytes()
                        for name in ("agentrl.py", "agentrl_async.py", "agentrl_trace.py", "backends.py", "models.py")
                        if (Path(__file__).parent / name).exists()
                    )
                ).hexdigest(),
                "observation_boundary": "host_file_api",
                "clock": "rank_local",
            }
            if self.config.request_profile:
                provenance["calibration_sha256"] = self.config.request_profile["provenance"]["source_sha256"]
            normalized = {
                "schema_version": 1,
                "mode": self.config.trainer_mode,
                "provenance": provenance,
                "requests": self.completed_requests,
                "events": self.trace.events,
            }
        else:
            normalized = None
        self.phase(
            "normalized trace",
            lambda: (
                self.write_json(self.result_dir / f"trace-rank-{self.rank}.json", normalized) if normalized else None
            ),
        )
        self.phase(
            "summary", lambda: self.write_json(self.result_dir / "summary.json", summary) if self.rank == 0 else None
        )
        return summary
