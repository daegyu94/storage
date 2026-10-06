# Optional cascade KV cache model

## Scope and source evidence

This design adds an experimental placement policy to the existing KV cache benchmark without changing its default waterfall behavior or the CLOSED workload.
The word **cascade** follows vLLM's source: a completed GPU-to-CPU store fans out to secondary tiers; it does not mean eviction-driven movement down a chain.
The simulator models completed immutable KV objects, ordered lookup, independent replica eviction, and demand promotion.
It does not model inference computation or asynchronous scheduling latency.

Source revisions inspected on 2026-10-06:

- Benchmark fork `main`: `01273ba885273e70d1ab1af03a0fc18f27c6ed01`.
- vLLM `main`: `b2863c0de953204b64e4e3d184f96eac26bd5b46`.
- LMCache `dev`: `8c77a6f77b4029269199de39c6f509944294c968`.

The PR branch was subsequently moved onto the fork's original `main`, `ab2b2e9d8dcb2b2cb499e66c163bfba572153e16`, when the fork's fast-forwarded experimental work was restored to individual branches.
The benchmark cache, CLI, and wrapper files used by this change are identical between that base and the initially investigated `01273ba` revision.
The cascade branch therefore contains only this design and implementation, without the separate Agent RL extensions.

### Existing benchmark

`kv_cache/cache.py:MultiTierCache` stores one `location` per key.
Allocation tries GPU, CPU, then NVMe; `_ensure_space_in_tier` reserves capacity and demotes LRU victims recursively.
`_demote_entry` reads the source, writes the destination, and deletes the source copy.
Intermediate tiers target 80% utilization and reject large objects; the terminal NVMe tier can admit an oversized object after evicting existing objects.
`access_cache` reads the recorded location and never promotes it, so repeated decode reads can repeatedly read the same NVMe object.

Backends implement `write/read/delete/clear` and return total, device, and host timing.
NVMe uses NumPy files, fsync, and page-cache advice; trace mode uses `NullBackend` and CSV `IOTracer`.
Logical request bytes and per-tier backend bytes already differ: eviction traffic appears in per-tier counters, while request totals describe requested payload.
Tests in `tests/test_kv_cache.py` cover configuration, backends, allocation, waterfall capacity/eviction, preconditioning, tracing, and TP accounting; GPU tests are conditional.
`IntegratedBenchmark` and its workload managers use the cache's public API, including metadata-only `check_cache_exists` for some multi-turn checks.

`vllm_lmcache_validate/validation_results.md` is the only checked-in validation artifact in that directory.
It reports historical Mistral/ShareGPT storage throughput and an environment with vLLM 0.13.0, but contains no raw per-key production trace or executable placement comparison.
Its numbers and claims about decode storage reads are not evidence that the current waterfall placement matches current production offloading.
The proposal document also explicitly records the absence of NVMe-to-GPU promotion.

### Current production semantics

[vLLM TieringOffloadingManager](https://github.com/vllm-project/vllm/blob/b2863c0de953204b64e4e3d184f96eac26bd5b46/vllm/v1/kv_offload/tiering/manager.py) completes the primary CPU store, then `complete_store` creates a CPU read job for each admitted secondary tier.
CPU references protect data during transfers; completion releases them.
`lookup` searches primary first and secondaries in order, reserves primary space for promotion, and returns a pending result until the transfer completes.
GPU transfers access only CPU primary; promotion preserves the secondary copy and does not trigger another fan-out store.
Backpressure, request-level policies, asynchronous lookup, and transfer failures can change admission and readiness.
[Filesystem secondary](https://github.com/vllm-project/vllm/blob/b2863c0de953204b64e4e3d184f96eac26bd5b46/vllm/v1/kv_offload/tiering/fs/manager.py) submits asynchronous file stores and loads; its policy is not a universal bounded LRU secondary cache.

[LMCache StorageManager](https://github.com/LMCache/LMCache/blob/8c77a6f77b4029269199de39c6f509944294c968/lmcache/v1/storage_backend/storage_manager.py) `batched_put` submits to all active backends by default, subject to explicit location selection and bypass controls.
Blocking `get` searches in backend order and registers a disk/remote hit in local CPU cache.
This is replication on store and read-through CPU population, rather than CPU-eviction-triggered disk writes.
[LocalCPUBackend](https://github.com/LMCache/LMCache/blob/8c77a6f77b4029269199de39c6f509944294c968/lmcache/v1/storage_backend/local_cpu_backend.py) admits references, suppresses resident duplicate keys, and frees evictable hot-cache objects under allocator pressure without writing them to disk.
[LocalDiskBackend](https://github.com/LMCache/LMCache/blob/8c77a6f77b4029269199de39c6f509944294c968/lmcache/v1/storage_backend/local_disk_backend.py) suppresses resident/in-flight duplicates, bounds capacity, removes local victims, and loads into CPU staging allocations.
Async prefetch has pinning, priority, prefix-boundary handling, and different callback behavior; it is not interchangeable with every blocking retrieval path.
[RemoteBackend](https://github.com/LMCache/LMCache/blob/8c77a6f77b4029269199de39c6f509944294c968/lmcache/v1/storage_backend/remote_backend.py) serializes and asynchronously submits connector puts; in-flight suppression and batching depend on the path, and eviction/durability belong to the remote service.
Resident duplicate suppression is therefore not a universal remote guarantee.
[LMCache engine](https://github.com/LMCache/LMCache/blob/8c77a6f77b4029269199de39c6f509944294c968/lmcache/v1/cache_engine.py) performs token/chunk lookup and GPU retrieval; prefix reuse, explicit store/retrieve locations, and optional remove-after-retrieve modes constrain which objects are actually stored or reused.

## Model definition

| Dimension | Existing waterfall | Optional cascade projection |
|---|---|---|
| Placement | One copy in first admissible tier | CPU primary plus NVMe secondary, optional GPU resident copy |
| Store | Storage write only on direct placement/demotion | Every admitted new object is replicated to NVMe immediately |
| Eviction | Read/write/delete to next tier | Delete only the selected tier's replica; no spill write |
| Replication | Exclusive | Inclusive but not permanently nested: tiers evict independently |
| Promotion | None | NVMe to CPU, then optional GPU; CPU to optional GPU |
| Read path | Read recorded tier on every access | Fastest resident copy; lower copies retained after promotion |
| Async I/O | Blocking backend calls | Blocking completion-boundary projection of production transfers |
| Storage traffic | Cold placement plus repeated lower-tier reads | Earlier/more writes for hot objects, fewer reads after reuse |

CPU primary is required in cascade and must hold one complete object.
An object larger than CPU capacity fails admission rather than inventing a direct GPU-to-secondary production path.
NVMe capacity is strict for cascade; a rejected/failed secondary write leaves valid CPU/GPU replicas available and records the failure.
GPU retention/promotion is optional and best effort; eviction drops only its GPU copy, including backend OOM callbacks.
Keys are immutable, as in the benchmark today; allocating an already resident key is idempotent and does not repair missing lower-tier copies.
An NVMe read is counted once for promotion, CPU/GPU transfers are counted separately, and subsequent resident accesses avoid NVMe.
If staging fails, the request is a miss; there is no claim that a secondary-only read bypasses CPU primary.

## Minimal implementation and compatibility

Keep the existing waterfall methods intact and dispatch the public cache operations to a small `CascadePolicy` object only when selected.
Reuse existing backends, generation, tracer, request tuple signatures, and latency export.
For cascade, metadata retains `location` as the fastest replica for callers, adds replica membership, and maintains an ordered LRU map per tier with deterministic insertion/access order.
Use one policy lock to serialize completed operations and prevent same-key duplicate writes or eviction during synchronous transfer.
Avoid a generic scheduler, pluggable topology framework, new remote connectors, or production dependencies.

Add `--tiering-policy {waterfall,cascade}` to the standalone CLI and OPEN/whatif suite CLI.
The constructor defaults to `waterfall`; standalone YAML may specify `tiering.policy`, with explicit CLI selection taking precedence.
Leave default `config.yaml` unchanged, and reject invalid policy values.
CLOSED does not expose the flag and forwards no policy override; OPEN metadata records an explicitly selected policy.
Default waterfall output fields and counters remain unchanged; cascade output identifies the experimental policy and its synchronous I/O model.

## Metrics

Preserve logical request byte totals and existing per-tier read/write counters.
Cascade adds integer backend read/write bytes and operations by tier, source-tier hits and ordered-probe misses, per-tier evicted replica counts/bytes, lost-last-copy counts, promotion counts/bytes by edge, fan-out replication counts/bytes, duplicate store suppression, and admission/read/write failure counters.
Resident entry counts count replicas rather than assigning each key to one tier.
Emit `Replicate` and `Promote` CSV phases without changing the CSV columns.
Payload bytes exclude NumPy headers, filesystem metadata, device write amplification, and compressed or serialized remote representations.

## Deterministic verification and fidelity

1. Use tiny fixed-size KV objects and trace/in-memory backends to test exact store fan-out, independent capacity/LRU eviction, source retention, repeated NVMe-hit promotion, CPU/GPU reuse, duplicate keys, TP sizing, resets, and metadata-only checks.
2. Exercise real temporary-file NVMe reads/writes and verify payload equality and file presence after promotion/eviction.
3. Inject admission and backend failures; verify replica metadata/capacity is published only after successful writes, failed reads count misses, and surviving replicas remain usable.
4. Run identical ordered workloads through waterfall and cascade, derive hit tiers from returned values, and assert exact NVMe read/write volume and promotion/eviction traffic rather than timing thresholds.
5. Compare default and explicit waterfall traces after removing timestamps, and test suite CLI forwarding plus CLOSED rejection/default invariance.
6. Provide an offline validation utility that optionally extracts and executes the pinned LMCache `StorageManager.batched_put/get` methods with bounded deterministic CPU/disk backend doubles.
   Replay the same workload through those actual source methods and the simulator, comparing ordered backend I/O and hit tiers at completed-operation boundaries.
   Record file SHA-256 and supplied revision; distinguish real upstream method execution from full vLLM/LMCache runtime validation.
7. For deployment fidelity, capture per-key completed put/get/evict events with actual chunk bytes and store/retrieve configuration, drain async jobs at comparison boundaries, map chunk keys to simulator keys, and compare backend byte totals, placement, misses, and promotions.
   Compare GPU-enabled inference only with the same chunking/TP geometry and active GPU lifetimes; decode-token simulator reads are not automatically production disk reads.

## Upstream PR boundaries

The policy is an experimental workload extension, not a change to MLPerf CLOSED rules or a claim of submission eligibility.
The first PR models one bounded local secondary and completed operations; remote fan-out, speculative prefetch, pinning, retries, async overlap, GPU block scheduling, and production latency remain documented limitations.
Source links and revisions make the semantics reviewable without installing vLLM, LMCache, CUDA, or a model.
Commit the design before implementation, keep policy code separate, run existing regression tests, and publish a reproducible comparison report with the observed I/O differences and validation limits.
