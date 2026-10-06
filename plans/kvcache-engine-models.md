# Representative KV-cache engine projections

## Scope and naming

The goal is to model storage traffic caused by representative KV-cache engines without implementing an inference scheduler.
This PR builds on `feature/kvcache-cascade-model`; both branches remain independent of `main` until review.
Existing `waterfall`, `cascade`, CLI defaults, and CLOSED workloads retain their behavior.

Use behavioral names rather than engine names: `gpu-write-through`, `gpu-selective`, and `gpu-write-back`.
The names specify the GPU-to-host backup decision, rather than assigning ambiguous meanings to cascade or waterfall.
They project SGLang HiCache admission behavior with a local NVMe secondary; they do not claim to emulate every engine configuration.
The existing CPU-primary `cascade` remains the projection for eager multi-backend LMCache storage and vLLM primary/secondary offload completion.

## Source evidence

Research was performed on 2026-10-06 and pins these source revisions:

| Engine | Revision | Relevant behavior |
| --- | --- | --- |
| [SGLang](https://github.com/sgl-project/sglang/tree/36b216e2ee7bae046a77a329e2fcda0e33beade5/python/sglang/srt/mem_cache) | `36b216e2ee7bae046a77a329e2fcda0e33beade5` | `UnifiedTreeCore._inc_hit_count_and_check` gates backup by hit count; `evict_device_leaf` backs up unbacked write-back leaves but deletes unbacked write-through leaves. |
| [SGLang HiCache](https://github.com/sgl-project/sglang/blob/36b216e2ee7bae046a77a329e2fcda0e33beade5/python/sglang/srt/mem_cache/unified_radix_cache.py) | Same | Thresholds are 1 for write-through and 2 for selective; `_finish_write_through_ack` schedules external storage backup after host backup completion. |
| [vLLM](https://github.com/vllm-project/vllm/blob/97bd8c74ebf12c3d847fb59f24966a83bf6f6920/vllm/v1/core/kv_cache_utils.py) | `97bd8c74ebf12c3d847fb59f24966a83bf6f6920` | `hash_block_tokens` includes parent hash, current tokens, and extra identity keys; complete blocks form a prefix chain. |
| [LMCache](https://github.com/LMCache/LMCache/blob/8c77a6f77b4029269199de39c6f509944294c968/lmcache/v1/token_database.py) | `8c77a6f77b4029269199de39c6f509944294c968` | Chunked token database chains prefix hashes; partial-chunk storage is configurable. |

LMCache has multiple storage-manager implementations and deployment modes.
The prior cascade evidence concerns the in-process storage manager, not a universal LMCache architecture.
SGLang has radix-tree dependencies, component pools, transfer queues, host locks, and failure fallbacks; a flat-entry simulator must document those differences.

## Model semantics

All new policies admit newly computed entries to GPU first and require positive GPU and CPU capacities and an available GPU backend (trace mode supplies virtual GPU storage).
GPU and host/secondary replicas have independent capacity-bounded LRU replacement.
Backing up an entry copies GPU to CPU, then CPU to NVMe when configured; successful replicas remain independently reusable.
Secondary failure does not invalidate a successful GPU/CPU copy.

| Policy | Backup trigger | Unbacked GPU eviction | Expected storage effect |
| --- | --- | --- | --- |
| `gpu-write-through` | First insertion | Normally already backed up | Eager NVMe writes, including one-use entries. |
| `gpu-selective` | Second observed use, counting first insertion as use one | Discard | Fewer writes for one-use entries, potentially more recomputation/misses. |
| `gpu-write-back` | GPU capacity eviction | Complete backup before removal | No writes while the working set fits GPU; eviction causes writes. |

Demand retrieval checks GPU, CPU, then NVMe and retains the source replica when promoting.
An existing secondary replica prevents redundant eviction backup.
Write-back backup failure preserves the GPU victim and rejects the incoming allocation; this conservative choice differs from SGLang's subtree-drop fallback and is counted explicitly.
Unexpected physical GPU eviction cannot run a reliable pre-eviction transfer and is counted separately.
No policy promises lower I/O for every workload.

## Minimal implementation

Add one `GPUBackupPolicy` subclass using the existing cascade transfer, placement, locking, and accounting primitives.
Keep waterfall dispatch and cascade behavior unchanged; extend the existing `--tiering-policy` and `tiering.policy` enum for OPEN/standalone usage only.
Keep the selective threshold fixed at two for this initial source-grounded projection, avoiding configuration combinations without evidence.
Expose backup count/bytes by insertion, reuse, and eviction trigger, backup failures, unexpected GPU eviction, discarded unbacked entries, and existing replica/hit/promotion metrics.

Add a separate sequential prefix-request replay utility, rather than changing MLPerf request generation or decode processing.
It accepts integer token IDs and an identity namespace, derives parent-dependent full-chunk keys, retrieves only the longest contiguous cached prefix once per request, and stores missing complete chunks after simulated computation.
Partial tails are computed but not stored in this projection, matching full-block caching rather than LMCache's optional partial-chunk mode.
Decode stays resident and contributes no external cache lookup; the utility reports reused/computed tokens and exact payload I/O separately.
Requests are sequential completion boundaries, not concurrent active sequences; no pin/refcount model is introduced.

## Validation before PR

Use deterministic 64-byte entries with tiny GPU/CPU/NVMe capacities to test all admission triggers, discard behavior, independent eviction, demand promotion, duplicate stores, oversized entries, reset, and failed backups.
Run the same operation stream through all five policies and assert NVMe reads/writes, original hit tiers, backup and promotion volume, and misses.
Use a small real local-NVMe experiment to check payload integrity and agreement with trace-mode placement accounting.
Test prefix-chain isolation, shared-prefix reuse, first-gap termination, and partial-tail recomputation independently of the existing workload.
Extract and execute the pinned SGLang backup-decision method against bounded doubles, record source SHA256, and compare its admission decisions with simulator traces.
Retain source links and reproducible commands with the resulting comparison artifact.
Run the existing KV and root CLI tests to establish compatibility, including the absence of new flags in CLOSED.

## Limits and future work

This models completed payload transfers, not asynchronous overlap, queue latency, cancellation, speculative prefetch, batched transfer coalescing, or physical-device write amplification.
Flat LRU entries do not reproduce radix parent/child eviction constraints, distributed rank consensus, active-sequence pinning, SWA/Mamba pools, or remote transport.
The replay namespace represents model/adapter/cache-salt identity but does not reproduce engine hash bytes or multimodal identity encoding.
Runtime GPU-engine validation remains distinct from source-method validation and a local-file experiment.
Do not add generic policy plug-ins, a remote backend, scheduler simulation, or timing predictions in this PR.
