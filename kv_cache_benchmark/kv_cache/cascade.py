"""Completed-operation projection of CPU-primary offload and replica reuse.

Stores replicate to the local secondary before returning. Each tier evicts
independently; demand reads populate CPU/GPU without deleting lower replicas.
This intentionally does not simulate production asynchronous scheduling.
"""

import logging
import math
import threading
import time
from collections import Counter, OrderedDict

from kv_cache.models import InferencePhase

logger = logging.getLogger(__name__)
TIERS = ('gpu', 'cpu', 'nvme')


class CascadePolicy:
    def __init__(self, cache):
        if not math.isfinite(cache.cpu_memory_limit) or cache.cpu_memory_limit <= 0:
            raise ValueError('cascade requires a positive finite CPU primary capacity')
        self.cache = cache
        # Serialize synchronous transfers, admission and metadata publication.
        # Reentrant because the GPU backend can call back during a write.
        self.lock = threading.RLock()
        self.lru = {tier: OrderedDict() for tier in TIERS}
        self.metrics = Counter()

    def _enabled(self, tier):
        return tier in self.cache.backends and self.cache._get_tier_limit(tier) > 0

    def _touch(self, tier, key):
        self.lru[tier].move_to_end(key)

    def _drop(self, tier, key, backend_deleted=False):
        cache = self.cache
        entry = cache.cache_entries.get(key)
        if entry is None or tier not in entry['replicas']:
            return
        if not backend_deleted:
            cache.backends[tier].delete(key)
        self.lru[tier].pop(key, None)
        entry['replicas'].remove(tier)
        cache._update_tier_usage(tier, -entry['size'])
        self.metrics[f'tier_{tier}_evictions'] += 1
        self.metrics[f'tier_{tier}_evicted_bytes'] += entry['size']
        cache.stats['evictions'] += 1
        if entry['replicas']:
            entry['location'] = next(t for t in TIERS if t in entry['replicas'])
        else:
            del cache.cache_entries[key]
            self.metrics['lost_entries'] += 1

    def gpu_evicted(self, key):
        with self.lock:
            self._drop('gpu', key, backend_deleted=True)

    def _ensure_space(self, tier, size):
        cache = self.cache
        limit = cache._get_tier_limit(tier)
        if not self._enabled(tier) or size > limit:
            self.metrics[f'tier_{tier}_admission_failures'] += 1
            return False
        while cache._get_tier_usage(tier) + size > limit:
            victim = next(iter(self.lru[tier]))
            try:
                self._drop(tier, victim)
            except Exception:
                logger.exception('cascade eviction failed: %s %s', tier, victim)
                self.metrics[f'tier_{tier}_eviction_failures'] += 1
                return False
        return True

    def _record_io(self, tier, operation, key, size, timing, phase):
        cache = self.cache
        label = 'storage' if tier == 'nvme' else tier
        suffix = 'read' if operation == 'Read' else 'written'
        cache.stats[f'tier_{label}_kv_bytes_{suffix}'] += size
        op = operation.lower()
        cache.stats[f'{label}_{op}_latencies'].append(timing.total)
        self.metrics[f'tier_{label}_{op}_operations'] += 1
        if tier == 'nvme':
            cache.stats[f'storage_{op}_device_latencies'].append(timing.device)
            cache.stats[f'storage_{op}_host_latencies'].append(timing.host)
            bytes_per_token = cache.model_config.kv_cache_size_per_token / cache.tensor_parallel
            cache.stats['storage_tokens_processed'] += size / bytes_per_token
        if cache.io_tracer is not None:
            cache.io_tracer.log(operation, size, tier, key=key, phase=phase)

    def _read(self, tier, key, size, phase):
        cache = self.cache
        try:
            if cache.io_tracer is not None:
                # NullBackend.read materializes a payload-sized dummy; the
                # placement-only path must not allocate that staging buffer.
                timing = cache.backends[tier]._ZERO_TIMING
                data = None
            else:
                data, timing = cache.backends[tier].read(key)
        except Exception:
            logger.exception('cascade read failed: %s %s', tier, key)
            self.metrics[f'tier_{tier}_read_failures'] += 1
            return False, None, 0.0
        self._record_io(tier, 'Read', key, size, timing, phase)
        self._touch(tier, key)
        return True, data, timing.total

    def _write(self, tier, key, size, data, phase):
        cache = self.cache
        if not self._ensure_space(tier, size):
            return False, 0.0
        try:
            if cache.io_tracer is not None:
                timing = cache.backends[tier].write_size(key, size)
            else:
                timing = cache.backends[tier].write(key, data)
        except Exception:
            logger.exception('cascade write failed: %s %s', tier, key)
            self.metrics[f'tier_{tier}_write_failures'] += 1
            # A backend can create a partial file before failing. It is not
            # published as a replica and must not survive as stale data.
            try:
                cache.backends[tier].delete(key)
            except Exception:
                logger.exception('cascade failed-write cleanup: %s %s', tier, key)
            return False, 0.0
        entry = cache.cache_entries.setdefault(key, {
            'size': size, 'replicas': set(), 'last_access': time.time(), 'access_count': 0,
        })
        entry['replicas'].add(tier)
        entry['location'] = next(t for t in TIERS if t in entry['replicas'])
        self.lru[tier][key] = None
        cache._update_tier_usage(tier, size)
        self._record_io(tier, 'Write', key, size, timing, phase)
        if tier == 'cpu':
            cache.stats['offloads_cpu'] += 1
        elif tier == 'nvme':
            cache.stats['offloads_storage'] += 1
        return True, timing.total

    def allocate(self, key, num_tokens, phase):
        with self.lock:
            cache = self.cache
            if key in cache.cache_entries:
                self.metrics['duplicate_stores'] += 1
                return True, cache.cache_entries[key]['location'], 0.0
            size = cache.model_config.kv_cache_size_per_token * num_tokens // cache.tensor_parallel
            if size <= 0 or size > cache.cpu_memory_limit:
                self.metrics['tier_cpu_admission_failures'] += 1
                return False, 'none', 0.0
            data = None
            if cache.io_tracer is None:
                try:
                    data = cache.generator.generate(sequence_length=num_tokens, key=key)
                    if cache.tensor_parallel > 1:
                        data = data.ravel()[:data.size // cache.tensor_parallel]
                    size = data.nbytes
                except Exception:
                    logger.exception('cascade generation failed: %s', key)
                    self.metrics['generation_failures'] += 1
                    return False, 'none', 0.0
            ok, latency = self._write('cpu', key, size, data, phase.value.capitalize())
            if not ok:
                return False, 'none', latency
            # CPU is the source for secondary transfers. The GPU copy is
            # retained as a separate resident cache, never an eviction source.
            if size <= cache.nvme_memory_limit:
                ok, cpu_data, elapsed = self._read('cpu', key, size, 'Replicate')
                latency += elapsed
                if ok:
                    copied, elapsed = self._write('nvme', key, size, cpu_data, 'Replicate')
                    latency += elapsed
                    if copied:
                        self.metrics['replications'] += 1
                        self.metrics['replication_bytes'] += size
            else:
                self.metrics['tier_nvme_admission_failures'] += 1
            if self._enabled('gpu'):
                _, elapsed = self._write('gpu', key, size, data, phase.value.capitalize())
                latency += elapsed
            cache.stats['write_operations'] += 1
            cache.stats['total_write_bytes'] += size
            if phase == InferencePhase.PREFILL:
                cache.stats['prefill_writes'] += 1
            cache.cache_entries[key]['access_count'] = 1
            return True, cache.cache_entries[key]['location'], latency

    def exists(self, key):
        with self.lock:
            entry = self.cache.cache_entries.get(key)
            return (entry['location'], entry['size']) if entry else (None, 0)

    def access(self, key, phase, cache_type):
        with self.lock:
            cache = self.cache
            entry = cache.cache_entries.get(key)
            source = None
            for tier in TIERS:
                if not self._enabled(tier):
                    continue
                if entry and tier in entry['replicas']:
                    source = tier
                    break
                self.metrics[f'tier_{tier}_misses'] += 1
            if source is None:
                cache.stats['cache_misses'] += 1
                return None, 0.0
            size = entry['size']
            # Make CPU staging capacity available before reading secondary.
            if source == 'nvme' and not self._ensure_space('cpu', size):
                cache.stats['cache_misses'] += 1
                return None, 0.0
            promoting = source == 'nvme' or (source == 'cpu' and self._enabled('gpu')
                                             and size <= cache.gpu_memory_limit)
            ok, data, latency = self._read(source, key, size,
                                          'Promote' if promoting else phase.value.capitalize())
            if not ok:
                cache.stats['cache_misses'] += 1
                return None, 0.0
            if source == 'nvme':
                ok, elapsed = self._write('cpu', key, size, data, 'Promote')
                latency += elapsed
                if not ok:
                    cache.stats['cache_misses'] += 1
                    return None, latency
                self.metrics['promotion_nvme_cpu_count'] += 1
                self.metrics['promotion_nvme_cpu_bytes'] += size
            if source != 'gpu' and self._enabled('gpu') and size <= cache.gpu_memory_limit:
                if source == 'nvme':
                    ok, data, elapsed = self._read('cpu', key, size, 'Promote')
                    latency += elapsed
                if ok:
                    copied, elapsed = self._write('gpu', key, size, data, 'Promote')
                    latency += elapsed
                    if copied:
                        self.metrics['promotion_cpu_gpu_count'] += 1
                        self.metrics['promotion_cpu_gpu_bytes'] += size
            self.metrics[f'tier_{source}_hits'] += 1
            cache.stats['cache_hits'] += 1
            cache.stats['read_operations'] += 1
            cache.stats['total_read_bytes'] += size
            if phase == InferencePhase.DECODE:
                cache.stats['decode_reads'] += 1
            hit_counter = {'system': 'system_prompt_hits', 'common': 'common_phrase_hits',
                           'multi_turn': 'multi_turn_hits'}.get(cache_type, 'user_cache_hits')
            cache.stats[hit_counter] += 1
            entry['last_access'] = time.time()
            entry['access_count'] += 1
            # Return the original hit tier so workload/validation attribution
            # remains useful even when the fastest replica changes on access.
            return source, latency

    def get_stats(self, duration):
        with self.lock:
            stats = self.cache._get_stats_inner(duration)
            stats.update(tiering_policy='cascade', tiering_io_model='synchronous-completion',
                         eviction_io_bytes=0)
            for tier in TIERS:
                label = 'storage' if tier == 'nvme' else tier
                stats[f'{label}_entries'] = len(self.lru[tier])
                for name in ('hits', 'misses', 'evictions', 'evicted_bytes',
                             'admission_failures', 'eviction_failures', 'read_failures', 'write_failures'):
                    stats[f'tier_{tier}_{name}'] = self.metrics[f'tier_{tier}_{name}']
                for op, suffix in (('read', 'read'), ('write', 'written')):
                    stats[f'tier_{label}_{op}_operations'] = self.metrics[f'tier_{label}_{op}_operations']
                    stats[f'tier_{label}_kv_bytes_{suffix}'] = self.cache.stats[f'tier_{label}_kv_bytes_{suffix}']
            for name in ('replications', 'replication_bytes', 'duplicate_stores', 'lost_entries',
                         'generation_failures', 'promotion_nvme_cpu_count', 'promotion_nvme_cpu_bytes',
                         'promotion_cpu_gpu_count', 'promotion_cpu_gpu_bytes'):
                stats[name] = self.metrics[name]
            stats['storage_memory_used_gb'] = self.cache.nvme_memory_used / 1024**3
            return stats

    def reset_stats(self):
        with self.lock:
            self.cache._reset_stats_inner()
            self.metrics.clear()
