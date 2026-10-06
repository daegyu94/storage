# Cascade placement and I/O validation

The optional cascade model produces the expected eager-write and read-through-reuse traffic, while the default waterfall matches the original `main` implementation.
This report validates completed-operation semantics and logical payload bytes, not asynchronous serving throughput or device-level NVMe traffic.
The [usage guide](../docs/cascade_tiering.md) defines metrics and limitations, and the [design](../../plans/kvcache-cascade-model.md) pins the source revisions and production semantics.

## Identical workload comparison

Each object is 64 bytes (four tokens in the tiny comparison model).
Both policies receive the same ordered operations and capacities: CPU 160 bytes, GPU disabled, NVMe 512 bytes except disk-pressure at 160 bytes.
The 160-byte CPU capacity admits two objects in both models despite waterfall's existing 80% target.

| Workload | Policy | NVMe write bytes | NVMe read bytes | CPU / NVMe hits | Global misses | Eviction payload I/O | NVMe-to-CPU promotion bytes |
|---|---|---:|---:|---|---:|---:|---:|
| Put A/B; read A three times | waterfall | 0 | 0 | 3 / 0 | 0 | 0 | 0 |
| Same hot-fit workload | cascade | 128 | 0 | 3 / 0 | 0 | 0 | 0 |
| Put A/B/C; read A three times | waterfall | 64 | 192 | 0 / 3 | 0 | 128 | 0 |
| Same reused-cold workload | cascade | 192 | 64 | 2 / 1 | 0 | 0 | 64 |
| Put A/B/C; read A/B/C twice | waterfall | 64 | 128 | 4 / 2 | 0 | 128 | 0 |
| Same CPU-thrashing workload | cascade | 192 | 384 | 0 / 6 | 0 | 0 | 384 |
| Put A/B/C; read A/B/C, bounded disk | waterfall | 64 | 64 | 2 / 1 | 0 | 128 | 0 |
| Same disk-pressure workload | cascade | 192 | 0 | 2 / 0 | 1 | 0 | 0 |

The reused-cold case has three times the initial NVMe write volume and one third of the NVMe read volume under cascade.
One promotion serves the first cold read and CPU serves the next two; secondary eviction is not required for promotion.
The waterfall demotion adds one CPU read and one NVMe write, accounting for 128 bytes of eviction payload I/O.

Cascade does not guarantee lower reads for every workload.
When A/B/C repeatedly compete for two CPU slots, every cascade lookup misses CPU and re-promotes from disk; waterfall keeps B/C in CPU and repeatedly reads only A from disk.
Under bounded disk capacity, cascade's replicas consume CPU and disk slots simultaneously, so A can lose both copies and miss even though waterfall retains its single NVMe copy.
The [machine-readable result](cascade_comparison.json) includes ordered hit tiers, tier misses/evictions, source-method matches, and the exact disk operations.

## Production source-method experiment

The comparison executed the actual LMCache `StorageManager.batched_put` and blocking `get` function bodies from revision `8c77a6f77b4029269199de39c6f509944294c968`.
The harness supplies deterministic bounded CPU/disk backend doubles and a shared CPU allocator; these capacity, recency, and completion assumptions are harness inputs, not claims that the full production backends were executed.
In all four workloads, per-operation replica placement, ordered source hits, and exact NVMe read/write events matched cascade.
CPU reference operations in LMCache are not equivalent to the benchmark's CPU memcpy counters, so those counters were not asserted as production CPU traffic.

The actual vLLM `TieringOffloadingManager.complete_store` body from revision `b2863c0de953204b64e4e3d184f96eac26bd5b46` was also executed with transfer/request doubles.
It submitted to both admitted secondary tiers on successful primary completion, submitted nothing on primary failure, and skipped a secondary rejected by admission control.
This validates the fan-out naming/contract; vLLM lookup, pinning, worker transfer, and GPU inference were not run in that harness.
The result records both source-file SHA-256 values so method changes are detectable.

To reproduce with source snapshots downloaded from the pinned official repositories:

```bash
mkdir -p source-snapshots
curl -fsSL https://raw.githubusercontent.com/LMCache/LMCache/8c77a6f77b4029269199de39c6f509944294c968/lmcache/v1/storage_backend/storage_manager.py \
  -o source-snapshots/lmcache-storage-manager.py
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm/b2863c0de953204b64e4e3d184f96eac26bd5b46/vllm/v1/kv_offload/tiering/manager.py \
  -o source-snapshots/vllm-tiering-manager.py
python kv_cache_benchmark/utils/compare_tiering.py \
  --lmcache-storage-manager source-snapshots/lmcache-storage-manager.py \
  --source-revision 8c77a6f77b4029269199de39c6f509944294c968 \
  --vllm-tiering-manager source-snapshots/vllm-tiering-manager.py \
  --output comparison.json
```

Without the source arguments, the utility runs the same simulator comparisons and exact expected-outcome assertions offline.
Source extraction avoids installing serving frameworks and executes only the selected method bodies; the source input should be the intended reviewed snapshot.

## Compatibility and backend validation

An additional baseline experiment loaded the original `MultiTierCache` directly from `01273ba:kv_cache_benchmark/kv_cache/cache.py` and replayed six stores and eight accesses.
CPU-only, NVMe-only, and GPU/CPU/NVMe trace configurations produced identical hit tiers, complete exported statistics, and CSV rows after removing timestamps when compared with the new default waterfall.
The unit suite also compares default versus explicit waterfall and checks standalone YAML/CLI precedence, OPEN/whatif forwarding, metadata, and CLOSED flag rejection.

Real temporary-file `NVMeBackend` tests verify NumPy payload equality after promotion and confirm the secondary file remains available.
An integration test drives the unchanged benchmark worker through three prefill requests followed by a batched decode request; it reproduces the 64/192 versus 192/64 write/read-byte outcomes with real CPU/file backends.
Tests exercise replica-local disk/GPU eviction, concurrent duplicate allocation, TP byte sizing, metadata-only probes, statistics reset, strict secondary capacity, and injected read/write/staging failures.
The standalone cascade command was also run with real CPU/NVMe backends and successfully exported labeled JSON metrics with matching NVMe write and replication bytes.
These tests verify backend calls and payload integrity; they do not measure flash-controller read/write volume.

Executed checks (Python 3.12, CPU-only serving dependencies):

| Check | Result |
|---|---|
| Initial investigation tree's KV cache tests, including separate Agent RL extensions, with MPI available | 532 passed, 22 skipped, 2 deselected |
| Final independent fork-base branch: KV tests plus CLI/wrapper/core-config tests | 513 passed, 22 skipped, 2 deselected |
| CLI, benchmark wrapper, and core-config compatibility tests | 275 passed |
| Final cascade tests, including the two subsequently added worker integration cases | 22 passed |
| Ruff correctness/import/modern-Python checks on new Python files | Passed |
| Original-main versus default-waterfall replay | Identical in three configurations |
| Source-method experiments | Four LMCache workloads and three vLLM fan-out cases passed |

The full suite emitted one existing NumPy reduction warning in the waterfall payload-preservation test.
GPU-dependent tests were skipped; the optional GPU cascade path was verified with trace backends and eviction callbacks, rather than CUDA execution.

## Remaining runtime fidelity work

No raw modern production deployment trace exists in the checked-in historical validation directory, and no full model-serving run was performed here.
The historical report's throughput values are preserved and are not reused as cascade fidelity evidence.
Before claiming full vLLM/LMCache fidelity, capture completed per-key transfers and reuse under matching chunk geometry and backend configuration, then follow the runtime replay procedure in the usage guide.
Async overlap, speculative prefetch, remote behavior, allocation pinning/batching, and active GPU block lifetimes remain outside this first model.
