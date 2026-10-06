# GPU backup and prefix reuse validation

This report validates completed-operation storage behavior rather than engine throughput.
The [machine-readable comparison](engine_comparison.json) includes identical operation streams for every policy, exact payload bytes, hit/miss attribution, evictions, backup and promotion counters, final placement, and source-method evidence.
The production source mappings and pinned revisions are in [the design plan](../../plans/kvcache-engine-models.md).

## Identical operation-stream comparison

Each object has four tokens and 64 bytes of KV payload.
All policies use the same GPU/CPU capacities of 160 bytes and NVMe capacity of 512 bytes.
The retained waterfall's existing 80% eviction target and the replica policies' strict capacity behavior remain unchanged.

| Scenario | Policy | NVMe write bytes | NVMe read bytes | Original hit tiers |
| --- | --- | ---: | ---: | --- |
| Insert a,b; read a three times | waterfall | 0 | 0 | GPU: 3 |
| Same | cascade | 128 | 0 | GPU: 3 |
| Same | gpu-write-through | 128 | 0 | GPU: 3 |
| Same | gpu-selective | 64 | 0 | GPU: 3 |
| Same | gpu-write-back | 0 | 0 | GPU: 3 |
| Insert a,b,c,d,e; read a three times | waterfall | 64 | 192 | NVMe: 3 |
| Same | cascade | 320 | 64 | NVMe: 1, GPU: 2 |
| Same | gpu-write-through | 320 | 64 | NVMe: 1, GPU: 2 |
| Same | gpu-selective | 0 | 0 | Miss: 3 |
| Same | gpu-write-back | 256 | 64 | NVMe: 1, GPU: 2 |
| Insert a,b,c,d,e; read a,b,c,d,e twice | waterfall | 64 | 128 | NVMe: 2, CPU: 4, GPU: 4 |
| Same | cascade | 320 | 640 | NVMe: 10 |
| Same | gpu-write-through | 320 | 640 | NVMe: 10 |
| Same | gpu-selective | 128 | 0 | Miss: 6, GPU: 4 |
| Same | gpu-write-back | 320 | 640 | NVMe: 10 |

In the cold-pressure case, write-back backs up a,b,c during allocation pressure and d when promoting a to GPU.
That produces 256 bytes of successful GPU-to-CPU backup, 256 bytes of NVMe replication, and 64 bytes each of NVMe-to-CPU and CPU-to-GPU promotion.
Total eviction-triggered payload traffic across GPU/CPU/NVMe is 1,024 bytes, counting both read and write sides of each copy.
Selective discards three never-reused GPU entries instead, so its zero NVMe traffic comes with three application misses and does not imply better overall performance.

When a is reused before pressure, selective preserves its CPU/NVMe backup and later hits CPU once then GPU twice without any NVMe read.
Its 64 bytes of NVMe writes contrast with 320 bytes for eager replication under the same input.
Thrashing demonstrates the opposite tradeoff: promotion can produce more repeated storage reads than waterfall's unchanged placement.

## Prefix-request experiment

The six requests in [prefix_requests.jsonl](prefix_requests.jsonl) contain 48 prompt tokens, a shared prefix, a divergent suffix, and later reuse after GPU pressure.
All five policies reuse `[0, 8, 4, 0, 8, 8]` tokens per request, for 28 reused and 20 computed tokens.
The result measures storage traffic for that same reuse outcome:

| Policy | NVMe write bytes | NVMe read bytes |
| --- | ---: | ---: |
| waterfall | 64 | 128 |
| cascade | 320 | 128 |
| gpu-write-through | 320 | 128 |
| gpu-selective | 128 | 0 |
| gpu-write-back | 320 | 64 |

This is a deterministic synthetic token-ID experiment, not a captured production inference trace.
Replay checks the longest contiguous prefix once per completed request; decode does not repeat external retrieval.
Tests separately check namespace isolation, parent-dependent suffix identity, first-gap termination, and uncached partial-tail recomputation.

## Fidelity evidence

The harness executes SGLang's actual `_inc_hit_count_and_check` and `evict_device_leaf` methods from revision `36b216e2ee7bae046a77a329e2fcda0e33beade5` with minimal tree/node doubles.
Its source file SHA256 is `0dec99f54263e21cc75814fc648dc22f911c5ba81ed890bb27907c6d7f0cf6fb`.
First/second/third-use backup decisions match simulator insertion/reuse boundaries: through `[true,false,false]`, selective `[false,true,false]`, and write-back `[false,false,false]`.
The source eviction method requests backup for an unbacked write-back leaf and discards an unbacked write-through/selective leaf; deterministic simulator pressure tests exercise these outcomes.
This does not equate every legacy decode access with a production radix insertion; request replay provides the intended boundary for reuse experiments.

A real `NVMeBackend` experiment uses deterministic NumPy payloads and host-memory doubles for GPU transfer.
After inserting three entries into one-entry GPU/CPU tiers and retrieving the oldest, it verifies byte-for-byte GPU payload integrity and exact NVMe writes/reads of 192/64 bytes, matching the trace-mode projection.
Failure injection checks preservation of a GPU victim after failed host backup, rejection of the incoming allocation, secondary-write failure isolation, and unexpected physical GPU eviction accounting.

The previous repository vLLM/LMCache validation report provides context but no raw production trace for this comparison.
Real GPU engine execution, asynchronous overlap, radix dependency fidelity, and remote backends remain unvalidated.
No engine performance or physical NVMe media-write prediction follows from these payload counters.

## Reproduction and compatibility

From the repository root with the benchmark dependencies installed:

```bash
python kv_cache_benchmark/utils/compare_engine_models.py \
  --requests kv_cache_benchmark/vllm_lmcache_validate/prefix_requests.jsonl \
  --sglang-tree-core path/to/pinned/unified_tree_core.py \
  --output engine_comparison.json

PYTHONPATH="$PWD/kv_cache_benchmark:$PWD" python -m pytest -q \
  -m 'not slow' -o addopts='' kv_cache_benchmark/tests \
  tests/unit/test_cli_kvcache.py tests/unit/test_benchmarks_kvcache.py \
  tests/unit/test_kvcache_vdb_allowlists.py
```

Existing default/explicit-waterfall equality tests, cascade tests, standalone config/CLI precedence, OPEN/whatif parsing, CLOSED override rejection, and wrapper allowlists remain part of validation.
No changes are made to the default configuration, MLPerf workload generation, CSV columns, CLOSED flags, or waterfall transfer logic.

The combined command above completed with **545 passed, 22 skipped, and 2 deselected**.
Skipped cases require optional platform/dependency capabilities, and the two deselected tests carry the existing `slow` marker.
One existing NumPy reduction warning remains in the waterfall demotion payload test.
Ruff checks on the four added Python files (`E,F,I,UP,B,C4`, excluding `E501`) and the whitespace check passed.
