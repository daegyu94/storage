# Add GPU-first backup admission and prefix reuse experiments

CPU-primary replication alone cannot represent the storage traffic of GPU-first KV caches that back up entries only after reuse or capacity pressure.
Add optional `gpu-write-through`, `gpu-selective`, and `gpu-write-back` policies, plus a sequential prefix-request replay utility, to distinguish eager writes, selective retention, deferred backups, and demand promotion.

This is the second of two dependent PRs.
Its review base is `feature/kvcache-cascade-model`, which supplies replica placement and transfer accounting; neither branch is merged into local or remote `main`.
After the first PR is accepted, retarget this branch to upstream `main` and retain only the second PR's commits.

The new policies reuse existing transfer, locking, eviction, and metrics primitives rather than adding a scheduler or backend framework.
Behavioral names identify backup triggers without claiming full SGLang/vLLM/LMCache emulation.
Pinned SGLang source methods ground the admission/eviction decisions; prefix-chain identity follows vLLM/LMCache's parent-dependent key structure without copying their hash encoding.

Default waterfall, existing cascade, CLOSED workloads/flags, and CSV columns retain their behavior.
Only standalone/OPEN/whatif policy choices expand, and the new CSV `Backup` phase is opt-in.
The separate replay tool leaves the existing MLPerf decode access pattern unchanged.

Identical cold-pressure input produces NVMe write/read bytes of 64/192 for waterfall, 320/64 for GPU write-through, 256/64 for GPU write-back, and 0/0 with three misses for selective admission.
Thrashing also shows cases where promotion increases reads relative to waterfall.
Source-method comparison, real local-file payload integrity, failure injection, prefix reuse tests, and existing compatibility checks are documented in [the validation report](../kv_cache_benchmark/vllm_lmcache_validate/engine_validation.md).
The combined benchmark/CLI suite passes with 545 passed, 22 skipped, and two slow tests deselected; added Python files pass Ruff checks.

Limitations are explicit: flat LRU entries, sequential request boundaries, synchronous completed transfers, prompt-only full chunks, and conservative write-back failure handling.
Runtime GPU-engine validation, active-sequence pinning, radix dependencies, async timing, speculative prefetch, remote storage, and physical write amplification are outside this PR.
