# Cascade KV cache placement

Select `cascade` to study CPU-primary offloading, eager secondary replication, and read-through reuse.
The default `waterfall` policy and the MLPerf CLOSED invocation are unchanged.
Cascade is an experimental workload extension; these results do not establish MLPerf submission eligibility.

The names describe placement: **eviction-driven waterfall** moves the only copy downward under pressure, while **replicated offload cascade** copies a completed CPU store to secondary storage.
Both policies use LRU eviction; cascade does not rename the eviction algorithm or imply a chain of exclusive tiers.
The existing name is retained for CLI compatibility, and the new name follows vLLM's explicit cascade/store terminology.

## Run the policy

From the repository root, run a standalone experiment:

```bash
python kv_cache_benchmark/kv-cache.py \
  --model tiny-1b --num-users 1 --duration 5 --max-requests 1 \
  --gpu-mem-gb 0 --cpu-mem-gb 0.25 --storage-capacity-gb 1 \
  --generation-mode none --disable-prefix-caching --disable-multi-turn \
  --tiering-policy cascade --cache-dir ./cascade-cache --output cascade.json
```

Add `--io-trace-log cascade.csv` to record logical I/O without allocating cache payloads or performing backend I/O.
For suite experiments, `mlpstorage open kvcache run` and `whatif` accept `--tiering-policy cascade`; supply enough CPU capacity for the selected model and objects.
CLOSED rejects the policy flag, does not forward an override, and keeps its default configuration and output schema.
The default cache constructor and benchmark API select waterfall.

Standalone `--config` can select the policy:

```yaml
tiering:
  policy: cascade
```

An explicit CLI policy overrides YAML.
There is no implicit policy change in the shipped configuration.
`--tiering-policy waterfall` selects the existing implementation, including its eviction thresholds and terminal-tier admission behavior.

## What cascade means

The policy follows completed operations in vLLM's `TieringOffloadingManager` and the common LMCache CPU/local-disk configuration.
Each new immutable object is admitted to CPU primary, replicated to NVMe secondary, and optionally retained in the GPU cache.
CPU is also the staging source for secondary writes and the destination for secondary reads.
An existing resident key is idempotent; allocation does not rewrite data or repair evicted secondary replicas.

Each tier uses its own LRU order and capacity.
Eviction deletes that replica and produces no read/write spill traffic; other replicas survive.
The NVMe capacity is strict, including oversized-object rejection, while waterfall retains its existing terminal-tier behavior.
Cascade requires a positive finite CPU capacity and rejects an object that cannot fit in CPU; secondary admission/write failure still permits CPU/GPU reuse.

Lookup uses GPU, CPU, then NVMe, skipping disabled tiers.
An NVMe hit reads once into CPU, then populates GPU if enabled and large enough; a CPU hit may also populate GPU.
Promotion retains the source replica and does not write NVMe again.
The API returns the tier that supplied the original hit, while `check_cache_exists` subsequently reports the fastest current replica without I/O or an LRU update.

Operations run synchronously under a policy lock, so a returned hit is ready for reuse and concurrent stores cannot publish duplicate replicas.
GPU backend eviction callbacks remove only GPU metadata, preserving offloaded copies.
The lock represents a completed transfer boundary, not asynchronous production throughput.
The implementation reuses the existing backends and does not add CUDA, vLLM, or LMCache dependencies.

## Read the metrics

Existing logical request totals describe application payload, and per-tier byte totals include successful backend transfers.
Cascade exports the additional counters below in `summary.cache_stats` and labels its result with `tiering_policy=cascade` and `tiering_io_model=synchronous-completion`.
Tier hit attribution follows the original source, and tier misses count ordered probes that reach that tier, not every global miss.
A failed backend read or required staging write counts as an application miss and has a separate failure counter.

| Counter | Meaning |
|---|---|
| `tier_{gpu,cpu,storage}_kv_bytes_{read,written}` | Integer successful backend payload bytes; existing GiB fields remain |
| `tier_{gpu,cpu,storage}_{read,write}_operations` | Successful backend operation counts |
| `tier_{gpu,cpu,nvme}_{hits,misses}` | Original-source hits and ordered-probe misses |
| `tier_{gpu,cpu,nvme}_{evictions,evicted_bytes}` | Replica removal count and bytes; not spill I/O |
| `promotion_{nvme_cpu,cpu_gpu}_{count,bytes}` | Successful upward copies, one count per object/edge |
| `replications`, `replication_bytes` | Successful CPU-to-NVMe fan-out copies |
| `duplicate_stores`, `lost_entries` | Resident allocation suppression and last-replica eviction |
| `tier_{gpu,cpu,nvme}_{admission,read,write,eviction}_failures` | Failed capacity admission or backend operation |
| `eviction_io_bytes` | Zero in this replica-drop model |

GPU/CPU/storage resident entry counts count tier replicas; their sum can exceed the number of unique keys.
Resetting counters preserves placement and LRU order, including after preconditioning.
CSV columns stay the same; `Replicate` and `Promote` distinguish the new transfer phases from `Prefill`, `Decode`, and waterfall `Evict`.
Deletion is represented by eviction counters, not an invented payload I/O event.

## Validate storage traffic

Run the deterministic comparison with the same objects, capacities, and ordered operations for both policies:

```bash
python kv_cache_benchmark/utils/compare_tiering.py --output comparison.json
PYTHONPATH=kv_cache_benchmark python -m pytest kv_cache_benchmark/tests/test_cascade.py
```

The comparison includes hot-fit, reused-cold, CPU-thrashing, and disk-pressure cases.
It asserts exact expected storage bytes and hit outcomes and reports eviction, replication, and promotion traffic.
The [checked-in validation report](../vllm_lmcache_validate/cascade_validation.md) includes source-method execution and backward-compatibility evidence.
The [design](../../plans/kvcache-cascade-model.md) contains pinned production source references and the full behavior comparison.

## Limits and production trace validation

Cascade captures one bounded local secondary and completed immutable objects.
It does not reproduce every vLLM/LMCache backend, GPU block lifetime, prefix boundary, pinning state, allocator batch eviction, remote serialization, request-level admission, failure retry, or speculative prefetch policy.
Source-tier LRU updates are a deterministic approximation: vLLM can touch multiple tiers, while LMCache lookup and `touch_cache` can defer recency updates.
The production filesystem secondary may also be unbounded; use an ample explicit storage capacity to study that case.

This simulator still uses its existing workload keys and object sizes, including synthetic decode accesses and metadata-only multi-turn checks.
It does not automatically turn those into vLLM block hashes or LMCache token-prefix chunks.
The GPU backend also uses existing NumPy-generated payloads and backend transfers, so the measured timing is not an inference-engine DMA timing model.
NVMe byte counters are payload volumes: they exclude NumPy headers, filesystem metadata, page-cache effects, compression, and device write amplification.

For runtime fidelity, capture completed per-key store/load/evict events from an instrumented vLLM or LMCache deployment with source revision, active backends, chunk size, TP geometry, and store/retrieve options.
Drain asynchronous work at comparison boundaries, replay the same admitted immutable keys and accesses, and compare per-operation replica sets, source-tier hits/misses, storage bytes, and promotion edges.
Separate speculative prefetch from demanded loads and GPU-resident decode from external reuse; replaying every decode token as a disk load would overstate storage reads.
A source-method harness validates orchestration only; a deployment trace is required before claiming inference timing, asynchronous overlap, or full serving fidelity.
