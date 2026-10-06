# Representative KV-cache engine models

Choose a placement policy by its backup decision rather than by an engine brand.
The GPU-first policies project the completed-operation behavior of SGLang HiCache; the existing CPU-primary cascade covers the earlier LMCache/vLLM offload projection.
They share transfer accounting and source-retaining promotion, while `waterfall` remains the default.
The [design plan](../../plans/kvcache-engine-models.md) pins the production sources and explains their mapping.

| Policy | Initial placement | Backup trigger | Unbacked GPU eviction |
| --- | --- | --- | --- |
| `waterfall` | Fastest available tier | Capacity demotion | Move down one tier |
| `cascade` | CPU, secondary, optional GPU replicas | Insertion | Delete GPU copy |
| `gpu-write-through` | GPU | Insertion | Discard if backup failed |
| `gpu-selective` | GPU | Second observed use | Discard |
| `gpu-write-back` | GPU | Capacity eviction | Backup before removal |

GPU-first policies require positive GPU and CPU capacities.
Each entry must fit both tiers; NVMe admission is independent and a failed secondary write leaves successful memory replicas usable.
With an actual GPU backend they perform real payload transfers; trace mode uses virtual tiers and requires no GPU.
Write-back eviction only transfers an entry if no host or secondary copy survives, so promotion does not automatically rewrite existing secondary data.

## Running placement experiments

Use the existing standalone or OPEN/whatif `--tiering-policy` argument with any policy in the table.
CLOSED does not expose this override, and its default workload and CLI remain unchanged.
The standalone YAML alternative is:

```yaml
tiering:
  policy: gpu-selective
```

The legacy benchmark still processes its existing decode reads.
In `gpu-selective`, initial insertion counts as one use and a successful GPU access supplies the second use; duplicate stores and metadata probes do not count.
Consequently, legacy decode accesses can trigger backup; use request replay below when evaluating request-boundary prefix reuse rather than the MLPerf access pattern.

Run all five policies on identical deterministic operations from the repository root:

```bash
python kv_cache_benchmark/utils/compare_engine_models.py --output comparison.json
```

The comparison includes GPU-fit, reuse, cold pressure, selected hot data, and thrashing scenarios.
It records exact NVMe payload bytes, original hit tiers, tier probe misses/evictions, backup triggers, replica placement, and promotion traffic.
Unavailable waterfall probe counters are represented as `null`, rather than fabricated from aggregate misses.

## Request-boundary prefix replay

Provide one JSON array of integer token IDs per line, as in [the deterministic fixture](../vllm_lmcache_validate/prefix_requests.jsonl).
Every row represents a completed sequential request; the replay retrieves the longest contiguous cached prefix once, computes the remaining tokens, and stores missing complete chunks.
The namespace must identify the model, adapter, and cache-salt combination when comparing separate workloads.

```bash
python kv_cache_benchmark/utils/compare_engine_models.py \
  --requests kv_cache_benchmark/vllm_lmcache_validate/prefix_requests.jsonl \
  --chunk-size 4 --namespace model/default --output prefix-comparison.json
```

Full chunks derive their identity from the parent prefix and current tokens, so an identical suffix under another prefix does not falsely hit.
Replay stops retrieval at the first missing chunk, even if later chunks happen to be resident.
Partial tails are always recomputed and never stored; LMCache's optional partial-chunk behavior is outside this projection.
Decode contributes no external-cache lookup and no generated-token storage in this prompt-only replay; this tool does not change MLPerf generation.
Capacities in this small comparison are fixed at GPU/CPU 160 bytes and NVMe 512 bytes with 16 bytes/token, so chunk sizes above ten tokens cannot be admitted.

## Metrics and fidelity

GPU-first results add `backup_{insertion,reuse,eviction}_{count,bytes}`, `backup_failures`, `discarded_unbacked_entries`, and `unexpected_gpu_evictions`.
Backup bytes measure successful GPU-to-CPU copies; `replication_bytes` measures successful CPU-to-NVMe copies separately.
`eviction_io_bytes` sums all successful tier payload reads and writes performed inside eviction backup, including transfers preceding a failed backup.
Existing promotion metrics distinguish NVMe-to-CPU and CPU-to-GPU population, and CSV adds a `Backup` phase without changing columns.
These are payload bytes, not physical NVMe media traffic or filesystem write amplification.

For source-method validation, supply the pinned SGLang `unified_cache/unified_tree_core.py` file:

```bash
python kv_cache_benchmark/utils/compare_engine_models.py \
  --requests kv_cache_benchmark/vllm_lmcache_validate/prefix_requests.jsonl \
  --sglang-tree-core path/to/unified_tree_core.py --output comparison.json
```

The harness executes the actual backup-decision and unbacked-leaf eviction methods with bounded doubles.
It checks simulator admission decisions at matching use boundaries and records the source SHA256; the caller must supply the revision pinned in the design plan.
The [validation report](../vllm_lmcache_validate/engine_validation.md) distinguishes source-method evidence, local-file payload validation, and missing GPU-runtime validation.

Flat entries do not reproduce radix-tree dependencies or parent-before-child backup ordering.
The model excludes active-sequence pins, distributed ranks, remote transport, async overlap, speculative prefetch, SWA/Mamba pools, and timing prediction.
Backup failure preserves a write-back GPU victim and rejects the incoming allocation, while production SGLang can drop an unbacked subtree under host pressure.
Unexpected physical GPU eviction is counted and cannot perform a pre-eviction backup after the source has disappeared.
