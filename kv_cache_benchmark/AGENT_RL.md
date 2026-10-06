# Experimental GPU-less Agent RL lifecycle

`agent-rl.py` runs finite sync or local async rollout/training lifecycles using the existing `ModelConfig` and `NVMeBackend`.
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
`trainer_mode` accepts `sync`, `colocate_async` and `separate_async`.
Sync supports MPI; async defaults to one process with explicit logical owners.
Standalone separated async also supports opt-in shared-filesystem MPI execution below.

| Fields | Meaning |
| --- | --- |
| `iterations`, `requests_per_rank`, `concurrency`, `seed` | Finite workload, request admission and deterministic profiles |
| `model` | Existing `ModelConfig` constructor fields; default is a tiny synthetic model |
| `prompt_tokens`, `response_tokens`, `chunk_tokens` | Initial context, deterministic length mixture and decode chunk size |
| `tokens_per_second`, `prefill_tokens_per_second`, `rank_rate_factors` | Configured rates; factors repeat over sync ranks or async logical rollout owners |
| `kv_offload_fraction` | Fraction of newly created KV bytes written to storage, default 0 |
| `prefix_reuse_probability`, `hot_prefixes`, `prefix_skew` | Requests choose a hot catalog or unique cold prefix; actual hits depend on policy/owner key existence |
| `turns`, `tool_delay_s`, `observation_tokens` | Multi-turn timing pause and history growth; no sandbox files |
| `persist_trajectories` | Synthetic payload save and training input read; set false for an in-memory queue, default true |
| `train_delay_s` | Compute delay per global update, split over separated mini-batches |
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

## Local asynchronous workload

Add the following to the small configuration above and use the same entry point:

```yaml
trainer_mode: separate_async   # or colocate_async
generation_rate_scope: owner
async_workload:
  group_size: 2
  batch_groups: 1
  outstanding_groups: 4
  queue_capacity: 2
  rollout_owners: 2
  parameter_sync_step: 2       # must be 1 for colocate_async
  max_prompt_age: null
  staleness_strategy: drop     # or wait
  kv_gc_delay_s: 0.01
```

`iterations` counts global policy transitions, not fixed rollout batches.
Each prompt group has `group_size` siblings sharing one prompt file and owner, with individual configured response/turn budgets.
Only groups whose siblings all finish can be sampled; `batch_groups` is the number of groups per trainer mini-batch.
`outstanding_groups` bounds dispatch credits and `queue_capacity` bounds terminal groups waiting for the trainer.
Both must be at least `batch_groups`; selected/dropped groups release one dispatch credit each.
`concurrency` separately bounds executing requests, not the entire set of queued sibling tasks.
This constant-window refill is a bounded approximation of veRL prompt refill, not an exact replay of its dispatch distribution.

Colocated mode pauses engine sections after sampling and retires old KV before training.
Standalone separated mode allows generation, KV writes and trajectory production during trajectory consumption, training and checkpoint writes.
After `parameter_sync_step` mini-batches it pauses rollout for weight synchronization and installs the next policy.
Modeled weight-sync latency influences I/O timing without transferring real weights.
The implementation does not reproduce veRL hybrid idle-pool lending.

Pauses drain the current token chunk and related KV I/O, retaining generated tokens and the original remaining budget.
On a new policy, unfinished requests re-prefill their retained prompt/generated/observation history before decoding resumes.
CPU tool delays can continue while generation is paused; they still produce no sandbox filesystem traffic.
Policy retirement makes old keys inaccessible immediately; `kv_gc_delay_s` delays physical delete of a frozen key snapshot.
New policy keys remain independent of old GC, and run shutdown drains file operations before cleanup and reporting.
Physical KV peak includes retired bytes until GC.
By default the legacy fraction model has no finite cache capacity; opt in to the working-set model below.

Prompt age is `trainer_policy - prompt_policy + 1`.
With `drop`, only terminal groups with age **greater than** the threshold are discarded.
With `wait`, sampling waits for running groups whose age is **at least** the threshold.
Null disables this bound; generated policy span remains a separate observation.
Unconsumed completed requests have disposition `queued`, `dropped` or `run_end` and a null `trainer_policy_at_accept`.
Accepted requests have disposition `accepted` and the policy at actual sampling.

Every rollout owner has a real KV/trajectory namespace on the same filesystem; trainer and GC I/O have explicit role tags.
Owner count is a synthetic compute/storage concurrency parameter, not measured physical node or GPU count.
Local async multi-rank launch is rejected before I/O; use the explicit separated MPI topology below.
Sync MPI behavior is preserved.
Async checkpoints contain real shard/manifest I/O but mark `inflight_recoverable: false`; async `--resume` is rejected.
Source hashes, effective settings, queue/window peaks and completion dispositions support reproduction while fidelity remains `uncalibrated`.
Async `requests_per_rank` is a shape catalog size, not a fixed offered batch or admission cap.
Async `decoded_tokens` counts every completed decode chunk, including unfinished requests discarded at shutdown.
`generated_tokens` retains the completed-request total, and `unfinished_generated_tokens` records the difference.
Async achieved decode rate uses all decoded tokens over the full rollout phase, including training/transition pauses and final GC.
It is a workload wall-time rate, not measured GPU service throughput.

## Capacity-driven KV storage

The optional model below replaces unconditional fractional writes with logical resident pages and real write-back storage I/O:

```yaml
kv_offload_fraction: 0
kv_cache_model:
  capacity_bytes: 768
  block_tokens: 2
```

These tiny values suit the smoke model, not a real accelerator specification.
The model works in sync, both local async modes and separated shared-filesystem MPI.
Each rollout owner has its own logical capacity; trainer-only MPI rank 0 has no KV pool.
Use `kv_offload_fraction: 0` to make the offload choice explicit and keep the original fraction model when this mapping is absent.
Legacy checkpoint fingerprints omit an absent mapping.

Resident pages contain metadata only; no GPU/CPU tensor arrays are allocated.
Page charge is `block_tokens * bytes_per_token`; storage payload is the exact valid token extent, including a partial tail.
Blocks must fit `max_payload_bytes`, and every full request history must fit one owner's page budget.
A too-long request is rejected before files are created rather than pretending partial history is sufficient for full attention.

The working set for the current chunk/observation ingestion is pinned until its safe point completes.
Admission waits when other pinned sections occupy the needed capacity; unpinned LRU blocks are evicted first.
Dirty eviction performs a real backend write, offloaded demand performs a real read, and a clean persisted copy avoids repeated writes.
Resident hits need no file I/O.
Tool waits unpin history so pressure from other requests may offload it; reload must finish before generation resumes.
Already admitted compute may overlap cache-transfer I/O, while metadata/placement transactions serialize per owner.

Only full-block prefixes share identity within a policy/owner; the partial prefix tail and generated history remain request-private.
Completed/cancelled trajectories release private KV; reusable prefix pages remain until policy retirement.
Async file retirement uses the existing delayed GC and sync deletes after logical release.
New policy invalidates logical residency and retained requests re-prefill under the new version without reading old keys.
Logical `resident_bytes`/page peak, offload/reload payload/ops, capacity wait and pin count are reported per rank/owner.
Physical KV payload/file occupancy remains a separate metric and may include retired or clean persisted copies.
The normalized observation records these cache settings and `uncalibrated` status.

This is a storage-enabled chunk-safe working-set swapping assumption, not stock veRL/vLLM GPU-cache behavior.
It does not emulate CUDA/CPU staging, continuous batching, radix-tree sharing, full-lifetime pinning or finite storage-tier capacity.
Actual disk KV in a real reference must be traced separately from logical GPU-cache events before calibration.

## Separated MPI execution

Set `trainer_mode: separate_async`, `generation_rate_scope: owner`, and `network_mode: none`.
The settings below use one trainer/coordinator rank and two rollout ranks:

```yaml
async_workload:
  execution: mpi_shared
  rollout_owners: 2
  group_size: 2
  batch_groups: 1
  outstanding_groups: 4
  queue_capacity: 2
  parameter_sync_step: 2
  max_prompt_age: null
  staleness_strategy: drop
  kv_gc_delay_s: 0.01
  rpc_timeout_s: 30
```

```bash
mpiexec -n 3 python kv_cache_benchmark/agent-rl.py --mpi \
  --config agentrl.yaml --storage-root shared/data --results-dir shared/results
```

Rank 0 trains; ranks 1 through N each execute one rollout owner, with `rollout_owners = world_size - 1`.
Rank is a process identity: topology records rank roles, process IDs and observed hostname aliases separately.
Observed hostname domains do not certify physical node identity, container placement or accelerator counts.
All ranks need the same code/configuration and mutually visible POSIX storage/results directories; setup checks fresh run markers in both directories.
Actual trajectory files cross the producer/trainer boundary; no tensor payload is substituted with an MPI message.
KV remains owner-scoped, while trainer reads/deletes the shared trajectory namespace.
Unconsumed worker trajectories are deleted with role `gc`; accepted trajectory reads are trainer I/O.

Coordinator credits and the terminal queue are global; `concurrency` limits requests per rollout process.
Per-owner compute rates and global group fanout/window must be held explicit when comparing pool ratios.
Checkpoint payloads belong to trainer rank 0 only; the manifest's shard `world_size` is 1, with `execution_world_size` and `trainer_ranks` recording the separate execution topology.
Pause and KV retirement acknowledgments precede each owner policy install, and all install acknowledgments precede coordinator dispatch under the new version.
No new group is dispatched under the final policy after the last configured iteration.

MPI collectives occur only during setup and after role tasks stop for reporting.
A single event-loop progress task multiplexes nonblocking metadata/control messages; its polling/instrumentation overhead is part of observed host time, not a calibrated NIC delay.
The RPC timeout bounds control commands, not legitimate full trajectory duration.
Worker/GC/trainer storage failures fail the MPI job; this is not fault-tolerant MPI or asynchronous replay recovery.

`distributed.causal_io_overlap` and its per-kind/op breakdown count operations whose phase token was frozen at `io_begin` and whose completion was received before trainer phase end.
The trainer phase start causes token delivery before that I/O begins, so the chain establishes overlap without comparing cross-rank timestamps.
Operations received after phase end are not counted even if their token matches.
Raw traces retain independent rank-local clocks and I/O IDs; no global concurrency peak is inferred by summing rank peaks.
Trainer-only and unfinished-rollout trace shards may have no completed requests but preserve their role and actual events.

This first distributed mode requires shared storage and one trainer rank.
Colocated MPI, multiple trainer ranks, node-local trajectory movement, dynamic NIC emulation and async resume are unsupported.
V1 rank-0 joint profiles are replicated as a shared catalog.
V2 grouped profiles preserve dispatch order and sibling-level owner placement, including groups spanning workers.
The coordinator waits for all owner subsets before terminal enqueue and releases files on every participating owner.
Arrival-time calibration and dynamic routing remain outside this catalog model.
Use actual remote-filesystem timing with `network_mode: none`; modeled weight-sync sleep remains separate from real MPI control traffic.

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
V1 profiles are sampled deterministically in input order per rank, cycling at exhaustion.
V1 sync source ranks must match the launch; V1 async uses a shared rank-0 catalog.
Profile prompt/turn/trajectory fields replace their scalar config counterparts; prefill rate, chunk size, offload fraction and trainer/checkpoint timing remain configured.
Partial-prefix reuse stores the shared prefix once, then prefills/writes the request-specific suffix.
Prefix reuse is rank-local and scoped to the installed policy version.
Config fingerprints include the full profile; absent profiles retain v0.1 checkpoint fingerprint compatibility.

Runs export `trace-rank-N.json` alongside the original event arrays.
They contain observed request active times, event boundaries and a source-code hash, with fidelity still `uncalibrated`.
The comparison reports per-rank arrival gaps, request/tool/rate/byte distributions, operation totals, phase lags, prompt age and generated policy span separately.
It checks I/O pairing, tool pauses, sync completion ordering and same-rank compute overlap across iteration boundaries.
The local and separated MPI async runners accept mode-matched V1 rank-0 profiles with complete contiguous groups of `group_size` records sharing prompt/prefix shape.
`requests_per_rank` must also be a multiple of `group_size` for V1 profile replay; sibling turn/trajectory fields remain joint and unchanged.
V1 record ordering supplies synthetic group assignment; real enqueue timestamps, dispatch groups and arrival correlations are not replayed.
An exported run ending with only some siblings complete may require an explicit group-aware calibration selection before conversion.
The generic converter does not silently invent missing siblings or filter discarded samples.
For colocated overlap checks, producers must use a common owner observation clock; unaligned process clocks cannot be relabeled as one rank.
No cross-rank concurrency is inferred.

The default thresholds are illustrative and must be registered before an independent holdout run.
Reusing the calibration source or run ID is reported as inconclusive; fixture comparisons never establish real Agent RL fidelity.
Declared real traces without generation/completion/training/policy boundaries are also inconclusive; reports list observed and missing boundaries.
Arrival, actual completion and request-visible completion must retain the same operation/key/request/policy identity.
Reports always set `real_validation: false`: distribution agreement is evidence to review with coverage, instrumentation and controlled interventions, not certification.
Checkpoint-only recovery at the final iteration produces no completed requests and therefore no normalized request trace.

## Tests

```bash
PYTHONPATH=kv_cache_benchmark python -m pytest kv_cache_benchmark/tests \
  -q -o addopts='' -m 'not slow'
```

New tests cover token/byte conservation, version boundaries, actual delay feedback, queue caps, tool pauses, checkpoint recovery and real two-rank MPI failure propagation.
MPI subprocess tests have bounded timeouts and skip only when the MPI tools are unavailable.

## Grouped async calibration (V2)

```bash
python kv_cache_benchmark/agent-rl-trace.py profile --grouped \
  --input reference-calibration --output grouped-profile.json
python kv_cache_benchmark/agent-rl.py --config separate-async.yaml \
  --profile grouped-profile.json --storage-root data --results-dir results --mpi
```

Use an MPI launcher for `execution: mpi_shared` or omit `--mpi` for local async.
The input directory must include the coordinator shard and all worker shards; independent rank clocks are retained.
This converter requires `group_dispatch` events on one coordinator clock, globally unique group/request IDs, ordered `members` and parallel `member_owners` arrays such as `["rollout-0", "rollout-1"]`.
`member_owners` is authoritative when present.
A legacy scalar `owner` is the fallback when the array is absent and all members share one owner.
Each complete request must have its matching `group` and `owner` identity.
Source collectors must join prompt uid, session identity, actual replica acquisition and backend observations explicitly; this is not a veRL native log parser.

The profile envelope remains `mode`, `provenance`, `records`, with `schema_version: 2`.
Each record has the six V1 shape fields plus sanitized `group_id`, contiguous `group_sequence`, zero-based `sibling` and integer `owner`.
Records are in coordinator dispatch/sibling order, have fixed fanout matching `group_size`, and each source owner maps consistently to one source rank.
Siblings share prompt/prefix geometry but may belong to different source ranks/owners.
Every owner must fit `rollout_owners`; source rank numbers need not match the new launch.
Runtime records use actual process rank and include `calibration_source_rank`, `calibration_group_id`, `calibration_group_sequence`, `calibration_owner` and `sibling`.

Group catalogs cycle by dispatch count, independently of source iteration/policy or completion ordering.
For V2, the catalog determines request shapes and prompt catalog length; scalar `requests_per_rank` does not select or regroup records.
Existing outstanding credits, whole-group terminal queue, tool/compute/I/O delays, staleness and policy cadence generate new timing.
V2 does not replay observed enqueue intervals, completion timestamps, drop/retry or GPU scheduler behavior.
Prefill/chunk/trainer/checkpoint settings and owner generation budget remain configured.

Incomplete groups are excluded as a whole and `provenance.selection` reports their IDs/reasons, included/dispatched group counts and interrupted request/token counts.
At least one complete group is required; unsupported variable fanout or ambiguous ownership fails rather than being silently collapsed.
This complete-group selection is subject to survivorship bias and cannot reconstruct unobserved EOS or service demand.
A drained calibration interval and separate holdout with long-tail/censoring analysis are needed.
`request_interrupted` events distinguish planned shape/token budget from observed generated/history tokens, including a request cancelled after generation but before publication.
They do not become completed records or trainer samples.

Comparison adds completed-owner distribution (total variation distance), complete-group fanout and same-rank completion gap.
`cross_rank_groups` counts complete groups whose sibling clocks differ; their raw timestamps are never subtracted to create straggler latency.
Group coverage and partial/censoring evidence must be inspected alongside marginal byte/rate agreement.
`real_validation` remains false and all current runs remain `uncalibrated`.

## veRL/vLLM reference geometry

The optional `rollout_reference` describes veRL + Ray with its pinned vLLM 0.29.0 dense cache geometry.
It does not start Ray, vLLM or a GPU; other Agent RL frameworks remain TBD.
Existing configs keep their byte calculations and checkpoint fingerprints.

```yaml
rollout_reference:
  gpu_name: target GPU label (unmeasured)
  framework: verl
  orchestrator: ray
  engine: vllm
  engine_version: 0.29.0
  tensor_parallel_size: 4
  storage_payload_scope: replica_aggregate_unpadded
  kv_budget_bytes_per_gpu: 2949120
kv_cache_model:
  block_tokens: 16
# Dense Qwen3-8B cache geometry; dtype is KV storage dtype.
model:
  name: Qwen/Qwen3-8B
  num_layers: 36
  hidden_dim: 4096
  num_heads: 32
  kv_heads: 8
  _kv_dim_override: 128
  attention_type: gqa
  dtype: bfloat16
kv_offload_fraction: 0
```

This deliberately constrained budget is a correctness scenario, not a GPU memory/performance measurement.
Keep the remaining lifecycle settings in your config and ensure each full request fits the resolved cache budget.
An optional `gpu_memory_bytes` declares a ceiling; no GPU catalog or throughput inference is performed.
If a KV budget is given, capacity mode is required and missing `capacity_bytes` is resolved from whole worker blocks.
Explicit capacity conflicting with the GPU budget fails before I/O.
Without a KV budget, geometry can also be used with the existing fractional offload model.

Uniform dense MHA/GQA with equal K/V head dimensions and float32/float16/bfloat16 is supported.
PP/DCP, MLA, hybrid/sliding-window attention and packed/quantized KV layouts require a different observed cache specification.
Query heads must divide across TP; KV heads are sharded when TP is smaller and replicated when TP exceeds their count.
Thus TP 16 for 8 KV heads stores two copies of the unique KV payload across the replica.
`bytes_per_token` uses the aggregate TP worker payload, preserving actual chunk/cache I/O accounting.
Each synthetic owner remains a complete rollout replica; MPI ranks are not TP GPU workers.

All TP worker payloads are combined into one owner file; per-TP files, padded layouts and TP I/O concurrency are not emulated.
The GPU label does not change generation or prefill service budgets.
Summary and normalized trace report worker/unique/aggregate geometry, GPU budget and the explicit `uncalibrated` service-time and connector-behavior states.
Native vLLM CPU-only offload does not imply filesystem KV I/O.
Its filesystem secondary tier, completed-block/prompt eligibility and CPU staging differ from this extension's fractional or chunk-safe dirty-eviction model.
Specifying reference geometry does not certify native connector fidelity.

## Bounded CPU primary and unbounded FS secondary

Opt in with `kv_offload_tiers` alongside reference geometry and `kv_cache_model`:

```yaml
kv_offload_tiers:
  cpu_capacity_bytes: 2359296
  fs_enabled: true
  fs_capacity_bytes: null
  offload_prompt_only: true
```

The CPU budget covers aggregate TP payload per rollout owner/replica, unlike the per-GPU KV budget.
It must fit one complete block and is rounded down to whole staging pages.
CPU residency is metadata only, not allocated host RAM or measured RSS.
Completed eligible blocks enter CPU and cascade immediately to actual FS writes, including when GPU capacity is sufficient.
CPU eviction does not trigger cascade; CPU hits avoid FS reads and misses promote valid FS copies through pinned CPU pages.
Transfers cannot evict pinned staging pages; cancellation drains actual I/O before releasing pins.

Prompt-only is the default.
Incomplete tails and generated blocks outside prompt eligibility have no offloaded copy and use configured prefill recompute when GPU-like residency is lost.
A per-generation-call store cursor prevents recopying the same prompt every decode step.
Tool observations and policy re-prefill start a new longer prompt call.
Setting `offload_prompt_only: false` also stores completed generated blocks.
CPU-only (`fs_enabled: false`) produces zero KV filesystem reads/writes.
Absent tier settings preserve the existing dirty-eviction behavior and fingerprints.

FS has no logical capacity eviction; actual ENOSPC and I/O errors fail the run.
Request completion releases GPU working sets, retaining CPU/FS copies until policy invalidation.
Logical invalidation removes lookup eligibility before existing physical GC.
Summary owner `offload_tiers` reports CPU reserved/peak bytes, hits/misses, stores/promotions/evictions/waits and successful FS transfer totals.
Legacy cache `offload_ops/reload_ops` count only the dirty-eviction path; use tier counters or `io_totals.kv_*` for this model.
FS valid bytes exclude retired copies pending GC; existing physical KV metrics track those files.

Normalized traces include resolved `kv_offload_tiers` and `tier: fs` on actual KV I/O.
By default callers await FS completion while other requests can overlap compute/I/O.
The optional background path below changes store admission and request overlap.
Native vLLM asynchronous scheduler-step DMA/cascade, per-TP files, padded layouts and connector reset behavior remain uncalibrated.
This extension does not start Ray/vLLM or certify native fidelity.

### Background FS admission and drain

These fields extend an otherwise valid `kv_offload_tiers` config:

```yaml
kv_offload_tiers:
  cpu_capacity_bytes: 67108864
  fs_enabled: true
  fs_capacity_bytes: null
  offload_prompt_only: true
  fs_execution: background
  fs_write_workers: 4
```

The default `fs_execution: caller_await` preserves the existing path.
Opt-in `background` returns after CPU staging admission and performs real FS writes concurrently, with at most `fs_write_workers` active writes per owner.
CPU slots stay pinned until completion, and full pinned staging skips new stores instead of forcing every generation to wait.
Backlog is bounded by CPU page count, and a CPU hit may serve a block while its FS write is pending.
Missing copies follow the existing cache miss/re-prefill path; reads/promotions still await storage.

Sync drains at rollout end; colocated async drains after pausing for training; separated async allows writes to overlap training/checkpoint and drains before policy installation.
All modes drain on cleanup before invalidating old-policy metadata and doing physical GC.
Failed writes cannot become valid FS hits, and background errors propagate through local or MPI failure handling.
Trace events include `cpu_store_skip`, `fs_store_submit/begin/end` and `offload_drain_begin/end`.
Owner summaries expose skipped payloads, pending/active peaks, executor wait and failed-store counts.

This admission/pinning model follows the vLLM v0.29 CPU manager contract but does not reproduce DMA, scheduler-step batch admission, native priority pools or newer latency-based proportional throttling.
Policy invalidation and physical GC remain conservative: native secondary FS retention/cache namespaces require further trace validation.
All results remain `uncalibrated`.
