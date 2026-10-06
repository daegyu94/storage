# Add an optional replicated cascade KV cache placement policy

The existing KV cache benchmark moves the only copy of each object down GPU/CPU/NVMe tiers on eviction and repeatedly reads lower-tier objects without promotion.
Current vLLM CPU-primary tiering and common LMCache CPU/local-disk offloading instead replicate stores to secondary storage and reuse secondary hits through CPU staging.
This change adds `--tiering-policy cascade` to model those completed-operation placement and I/O semantics while retaining the default waterfall implementation.
The branch is based on the fork's original `ab2b2e9` main and contains only cascade changes; the fork's other experimental techniques remain on their own branches.

## Design

- Dispatch existing cache APIs to a separate `CascadePolicy` only when selected.
- Replicate CPU-admitted immutable objects to NVMe, evict replicas independently, and promote secondary hits without deleting or rewriting their secondary copy.
- Preserve optional GPU residency with CPU-staged secondary transfers and replica-local GPU eviction callbacks.
- Require CPU primary admission and enforce strict bounded secondary capacity, with failure counters and surviving replicas retained after secondary failures.
- Export replica hits/misses, eviction bytes, replication/promotion traffic, backend payload bytes/operation counts, and a synchronous-completion model label.

`waterfall` retains the repository's existing terminology for sequential eviction-driven demotion.
`cascade` follows vLLM's explicit name for primary-store fan-out; the documentation calls it replicated offload cascade to distinguish it from a demotion chain.
The implementation deliberately uses synchronous backend completion and one policy lock rather than adding a scheduler, new storage connectors, or serving-framework dependencies.

## Backward compatibility

Default constructors, CLI behavior, shipped YAML, waterfall eviction thresholds, metrics, CSV columns, and CLOSED workloads remain unchanged.
Only OPEN/whatif exposes the suite policy flag; CLOSED rejects overrides and forwards no policy option.
Standalone YAML can select a policy, with explicit CLI selection taking precedence.
Original `main` and current default waterfall produced identical exported statistics, hit tiers, and timestamp-normalized traces in NVMe-only, CPU-only, and three-tier replay checks.

## Validation

The [validation report](../kv_cache_benchmark/vllm_lmcache_validate/cascade_validation.md) and [JSON results](../kv_cache_benchmark/vllm_lmcache_validate/cascade_comparison.json) contain reproducible workload comparisons and source-method evidence.
For three stores followed by three reads of an evicted object, waterfall writes 64 bytes and reads 192 bytes from NVMe, while cascade writes 192 bytes and reads 64 bytes.
CPU-thrashing and bounded-disk tests also demonstrate higher cascade read traffic and replica loss where expected, rather than assuming cascade always improves I/O.
Real-file worker integration tests reproduce the same traffic difference and verify payload integrity.

Actual pinned LMCache `StorageManager.batched_put/get` method bodies match simulator placement, hit tiers, and disk events in four workloads using deterministic backend doubles.
The pinned vLLM `complete_store` method confirms all-secondary fan-out, failed-primary suppression, and secondary admission filtering using transfer/request doubles.
Additional tests cover concurrent duplicate allocation, source retention, independent LRU eviction, TP sizing, statistics reset, CLI/config precedence, and backend failure paths.

## Limits

This is an experimental workload extension and does not claim official MLPerf submission eligibility.
It models one bounded local secondary and completed immutable objects; it does not simulate remote serialization/eviction, speculative prefetch, async overlap, allocator pinning/batching, request-level admission, retries, or active GPU block scheduling.
The existing workload's synthetic decode reads and keys are preserved, so full serving fidelity still requires a matching production trace and chunk/TP mapping.
Storage volumes are logical payload bytes, and the source-method harness is not a full GPU inference experiment.

See the [source-grounded design](kvcache-cascade-model.md) and [usage/metrics guide](../kv_cache_benchmark/docs/cascade_tiering.md) for source revisions, assumptions, and runtime validation procedures.
