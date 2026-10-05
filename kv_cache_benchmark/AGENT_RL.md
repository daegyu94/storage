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

## Decode capacity and bounded admission

`generation_rate_scope: request` retains the original independent per-request rate.
`generation_rate_scope: owner` caps aggregate decode service per rank at `tokens_per_second * rank_rate_factor`.
A FIFO asyncio lock schedules token chunks; storage and tool waits release the compute slot so other requests can advance.
With a joint profile the service delay is the greater of the observed request's decode delay and the owner budget delay.
This is a conservative chunk service model, not calibrated continuous GPU batching or prefill/decode interference.
Prefill remains a separate configured delay; one rank represents one configured owner, not automatically one physical GPU or node.
`concurrency` bounds active requests and the number of worker tasks; the full sync batch is logically offered at phase start.
Queue wait includes admission delay from that common offered time.
Summary fields include compute queue wait, modeled decode service and achieved generated tokens divided by rollout phase wall time.
Summed compute wait across requests is not elapsed time.

## Joint request calibration and reference comparison

`agent-rl.py --profile profile.json` accepts a joint request profile generated by:

```bash
python kv_cache_benchmark/agent-rl-trace.py profile \
  --input reference-calibration.json --output profile.json
python kv_cache_benchmark/agent-rl.py --config agentrl.yaml --profile profile.json \
  --storage-root data --results-dir results
python kv_cache_benchmark/agent-rl-trace.py compare \
  --reference reference-holdout.json --candidate results/RUN_ID \
  --output comparison.json --relative-tolerance 0.25 --ks-tolerance 0.3
```

The input is an explicit normalized observation contract, not arbitrary veRL console logs.
The envelope has `schema_version: 1`, `mode`, `provenance`, `requests` and `events`.
Provenance requires `kind` (`verl`, `synthetic_fixture`, or `poc`), `run_id`, `revision`, `observation_boundary` and `clock: rank_local`.
Each request contains `rank`, `iteration`, `request`, `start_s`, `end_s`, `policy_start`, `policy_end`, `prompt_policy`, `trainer_policy_at_accept`, `prompt_tokens`, `prefix_tokens`, sanitized `prefix_id`, `trajectory_bytes` and `turns`.
Each turn has `generated_tokens`, `decode_active_s`, `tool_delay_s` and `observation_tokens`; the last turn has no trailing tool interaction.
Decode active time excludes tool, I/O and compute queue waits so these delays are not counted twice during replay.
The adapter must measure decode service at a declared host-visible boundary, not infer it from end-to-end request latency.

Request profiles preserve these fields together and retain a canonical source SHA256.
Profiles are sampled deterministically in input order per rank, cycling at exhaustion; source rank identities must match the launch.
Profile prompt/turn/trajectory fields replace their scalar config counterparts; prefill rate, chunk size, offload fraction and trainer/checkpoint timing remain configured.
Partial-prefix reuse stores the shared prefix once, then prefills/writes the request-specific suffix.
Prefix reuse is rank-local and scoped to the installed policy version.
Config fingerprints include the full profile; absent profiles retain v0.1 checkpoint fingerprint compatibility.

Runs export `trace-rank-N.json` alongside the original event arrays.
They contain observed request active times, event boundaries and a source-code hash, with fidelity still `uncalibrated`.
The comparison reports per-rank arrival gaps, request/tool/rate/byte distributions, operation totals, phase lags, prompt age and generated policy span separately.
It checks I/O pairing, tool pauses, sync completion ordering and same-rank compute overlap across iteration boundaries.
The `colocate_async` and `separate_async` modes are accepted for reference analysis only; the runner rejects async replay profiles.
For colocated overlap checks, producers must use a common owner observation clock; unaligned process clocks cannot be relabeled as one rank.
No cross-rank concurrency is inferred.

The default thresholds are illustrative and must be registered before an independent holdout run.
Reusing the calibration source or run ID is reported as inconclusive; fixture comparisons never establish real Agent RL fidelity.
Reports always set `real_validation: false`: distribution agreement is evidence to review with coverage, instrumentation and controlled interventions, not certification.
Checkpoint-only recovery at the final iteration produces no completed requests and therefore no normalized request trace.

## Tests

```bash
PYTHONPATH=kv_cache_benchmark python -m pytest kv_cache_benchmark/tests \
  -q -o addopts='' -m 'not slow'
```

New tests cover token/byte conservation, version boundaries, actual delay feedback, queue caps, tool pauses, checkpoint recovery and real two-rank MPI failure propagation.
MPI subprocess tests have bounded timeouts and skip only when the MPI tools are unavailable.
