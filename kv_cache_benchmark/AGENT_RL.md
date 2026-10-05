# Experimental GPU-less Agent RL lifecycle

`agent-rl.py` runs a finite synchronous rollout/training lifecycle using the existing `ModelConfig` and `NVMeBackend`.
Compute is replaced by delays; the workload performs real POSIX prompt, KV, trajectory and synthetic checkpoint I/O.
This is an experimental extension, not an official MLPerf submission workload.

## Run

Install `numpy`, `PyYAML` and optionally `mpi4py` in a Python 3.12 environment.
Torch, CuPy and GPU hardware are unnecessary.

```yaml
# agentrl.yaml: small correctness example, not a calibrated performance profile
trainer_mode: sync
iterations: 2
requests_per_rank: 4
concurrency: 2
prompt_tokens: 16
response_tokens: [16, 32]
chunk_tokens: 8
tokens_per_second: 100
kv_offload_fraction: 1.0
persist_trajectories: true
checkpoint_every: 1
checkpoint_bytes_per_rank: 4096
```

```bash
python kv_cache_benchmark/agent-rl.py --config agentrl.yaml \
  --storage-root data --results-dir results
mpiexec -n 2 python kv_cache_benchmark/agent-rl.py --mpi --config agentrl.yaml \
  --storage-root data --results-dir results
```

Each rank owns its prompt/KV/trajectory namespace and a checkpoint shard.
All rollout owners finish before training; optional checkpoints commit before weight installation and the next rollout.
Slower generation or I/O delays that lifecycle rather than running on an independent schedule.
Multiple admitted requests overlap their compute delays and threaded I/O; a request waits for its own chunk write before generating the next chunk.

## Configuration

`AgentRLConfig` in `kv_cache/agentrl.py` is the canonical schema and rejects unknown fields.
JSON and YAML configurations are supported; omitted fields use dataclass defaults recorded in `config.json`.
Only `trainer_mode=sync` is implemented.

| Fields | Meaning |
| --- | --- |
| `iterations`, `requests_per_rank`, `concurrency`, `seed` | Finite workload, request admission and deterministic profiles |
| `model` | Existing `ModelConfig` constructor fields; default is a tiny synthetic model |
| `prompt_tokens`, `response_tokens`, `chunk_tokens` | Initial context, deterministic length mixture and decode chunk size |
| `tokens_per_second`, `prefill_tokens_per_second`, `rank_rate_factors` | Per-request rates; factors repeat over logical ranks |
| `kv_offload_fraction` | Fraction of newly created KV bytes written to storage, default 0 |
| `prefix_reuse_probability`, `hot_prefixes`, `prefix_skew` | Requests choose a hot catalog or unique cold prefix; actual hits depend on policy/owner key existence |
| `turns`, `tool_delay_s`, `observation_tokens` | Multi-turn timing pause and history growth; no sandbox files |
| `persist_trajectories` | Synthetic payload save and training input read; set false for an in-memory queue, default true |
| `train_delay_s` | Training compute delay on the same logical pool |
| `checkpoint_every`, `checkpoint_bytes_per_rank` | Save cadence and synthetic state bytes; every=0 disables saves |
| `weight_sync_delay_s`, `weight_bytes` | Base transfer delay and optional NIC estimate; no real weight traffic |
| `network_mode` | `none` uses actual backend latency; `estimate` adds modeled transport sleep |
| `nic_gbps`, `nic_efficiency`, `nic_rtt_us`, `nic_sharers` | Decimal link rate, efficiency, RTT and fixed concurrent-flow sharing |
| `max_payload_bytes` | Per-object allocation guard, default 64 MiB; large streaming checkpoints require a later adapter |

The hot catalog treats each entry as an entire reusable prompt prefix; no token text matching occurs.
KV bytes use the existing model geometry and cumulative fractional rounding.
A `kv_offload_fraction=0` deployment produces no storage KV I/O; nonzero offload must correspond to a storage-enabled or explicitly hypothetical scenario.
Trajectory payload size is 12 bytes per history token as a configurable-profile approximation encoded by this version, not veRL's exact serialized layout.
Synthetic data is deterministic pseudorandom bytes with independent rank/iteration keys.

The NIC estimate is `RTT + bytes / (Gbps * 1e9 / 8 * efficiency / sharers)`.
It models latency only and does not reproduce real network congestion, transfers or collectives.
Use `network_mode=none` on an actual remote filesystem to avoid counting its network time twice.

## Checkpoint recovery

A checkpoint consists of per-rank `.npy` payloads and a manifest published after all shard writes succeed.
The manifest records policy version, next iteration, configuration fingerprint, world size, rank ownership, payload sizes and checksums.
Use the same configuration and owner mapping on resume.

```bash
python kv_cache_benchmark/agent-rl.py --config agentrl.yaml \
  --storage-root data --results-dir results \
  --resume data/RUN_ID/checkpoints/step-1/manifest.json
```

Recovery reopens files with `NVMeBackend(preserve_existing=True)` without deleting checkpoint payloads.
It verifies every local shard before resuming at a synchronous iteration boundary; The restored policy is installed with the configured weight-sync delay before new rollout; KV is reconstructed in the new run namespace.
Missing/incomplete/corrupt checkpoints and configuration/world-size mismatches fail.
Node-local storage requires the same rank-to-node placement and paths on resume.
External storage KV is removed after weight installation; this differs from the earlier GPU-cache release at the real sync trainer's sample boundary.
This is a synthetic byte checkpoint, not DLIO or FSDP/Megatron state serialization, and does not recover in-flight async trajectories.

## Results and interpretation

`config.json`, per-rank event arrays and `summary.json` live under a fresh run ID.
Summary fields distinguish payload bytes, host-side actual I/O latency and modeled network delay; events record queue wait, tool gaps, policy boundaries and barrier wait.
Percentiles aggregate operation samples, while summed operation duration is not wall-clock elapsed under concurrency.
Rank-relative monotonic clocks are not aligned across hosts.

Each operation has a rank-local `io_id` shared by `io_begin`, `io_actual_end` and `io_end`.
Read arrivals carry the expected payload size; mismatches fail rather than silently changing the workload.
`actual_s` includes executor scheduling and the backend call, while `backend_s` measures the existing backend's own timer.
`io_actual_end` precedes the optional modeled NIC delay; `io_end` is the point at which the request may advance.
`physical_file_bytes` is the observed `.npy` file length, not measured disk or network traffic.
Checkpoint I/O carries the saved/loaded policy version even before that policy is installed for generation.
Rank summaries include KV live/peak payload, prefix hit/miss counts, I/O concurrency, generated tokens and `iteration_elapsed_s`, which excludes setup, recovery and initial weight installation.

The existing backend uses `.npy` serialization and fsync plus best-effort read fadvise, not guaranteed O_DIRECT or device-level latency tracing.
The backend's constructor preserves its original destructive-reset default; this runner always selects preservation and a unique namespace.
Payload byte counters exclude serialization headers and device/wire amplification.
The profile remains `uncalibrated` until compared with real Agent RL traces at the same observation boundary.

## Tests

```bash
PYTHONPATH=kv_cache_benchmark python -m pytest kv_cache_benchmark/tests \
  -q -o addopts='' -m 'not slow'
```

New tests cover token/byte conservation, version boundaries, actual delay feedback, queue caps, tool pauses, checkpoint recovery and real two-rank MPI failure propagation.
MPI subprocess tests have bounded timeouts and skip only when the MPI tools are unavailable.
