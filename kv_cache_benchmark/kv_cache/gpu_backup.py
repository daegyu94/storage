"""Flat-entry projection of GPU-first HiCache backup admission decisions."""

from kv_cache.cascade import CascadePolicy
from kv_cache.models import InferencePhase

GPU_POLICIES = ('gpu-write-through', 'gpu-selective', 'gpu-write-back')


class GPUBackupPolicy(CascadePolicy):
    def __init__(self, cache):
        super().__init__(cache)
        self._eviction_latency = 0.0
        if not self._enabled('gpu'):
            raise ValueError(f'{cache.tiering_policy} requires an available positive GPU tier')

    def _backup(self, key, trigger):
        entry = self.cache.cache_entries[key]
        if 'cpu' in entry['replicas'] or 'nvme' in entry['replicas']:
            return True, 0.0
        size = entry['size']
        ok, data, latency = self._read('gpu', key, size, 'Backup')
        if ok:
            ok, elapsed = self._write('cpu', key, size, data, 'Backup')
            latency += elapsed
        if not ok:
            self.metrics['backup_failures'] += 1
            return False, latency
        self.metrics[f'backup_{trigger}_count'] += 1
        self.metrics[f'backup_{trigger}_bytes'] += size
        if self._enabled('nvme'):
            ok, data, elapsed = self._read('cpu', key, size, 'Replicate')
            latency += elapsed
            if ok:
                copied, elapsed = self._write('nvme', key, size, data, 'Replicate')
                latency += elapsed
                if copied:
                    self.metrics['replications'] += 1
                    self.metrics['replication_bytes'] += size
        return True, latency

    def _drop(self, tier, key, backend_deleted=False):
        entry = self.cache.cache_entries.get(key)
        if tier == 'gpu' and entry and entry['replicas'] == {'gpu'}:
            if self.cache.tiering_policy == 'gpu-write-back' and not backend_deleted:
                before = self._payload_bytes()
                ok, elapsed = self._backup(key, 'eviction')
                self.metrics['eviction_io_bytes'] += self._payload_bytes() - before
                self._eviction_latency += elapsed
                if not ok:
                    raise RuntimeError('GPU write-back backup failed; victim retained')
            else:
                self.metrics['discarded_unbacked_entries'] += 1
        super()._drop(tier, key, backend_deleted)

    def _payload_bytes(self):
        return sum(self.cache.stats[f'tier_{tier}_kv_bytes_{suffix}']
                   for tier in ('gpu', 'cpu', 'storage') for suffix in ('read', 'written'))

    def gpu_evicted(self, key):
        with self.lock:
            self.metrics['unexpected_gpu_evictions'] += 1
            self._drop('gpu', key, backend_deleted=True)

    def allocate(self, key, num_tokens, phase):
        with self.lock:
            cache = self.cache
            if key in cache.cache_entries:
                self.metrics['duplicate_stores'] += 1
                return True, cache.cache_entries[key]['location'], 0.0
            size = cache.model_config.kv_cache_size_per_token * num_tokens // cache.tensor_parallel
            if size <= 0 or size > min(cache.gpu_memory_limit, cache.cpu_memory_limit):
                self.metrics['tier_gpu_admission_failures'] += 1
                return False, 'none', 0.0
            data = None
            if cache.io_tracer is None:
                try:
                    data = cache.generator.generate(sequence_length=num_tokens, key=key)
                    if cache.tensor_parallel > 1:
                        data = data.ravel()[:data.size // cache.tensor_parallel]
                    size = data.nbytes
                except Exception:
                    self.metrics['generation_failures'] += 1
                    return False, 'none', 0.0
            self._eviction_latency = 0.0
            ok, latency = self._write('gpu', key, size, data, phase.value.capitalize())
            latency += self._eviction_latency
            if not ok:
                return False, 'none', latency
            cache.cache_entries[key]['access_count'] = 1
            if cache.tiering_policy == 'gpu-write-through':
                _, elapsed = self._backup(key, 'insertion')
                latency += elapsed
            cache.stats['write_operations'] += 1
            cache.stats['total_write_bytes'] += size
            if phase == InferencePhase.PREFILL:
                cache.stats['prefill_writes'] += 1
            return True, cache.cache_entries[key]['location'], latency

    def access(self, key, phase, cache_type):
        with self.lock:
            self._eviction_latency = 0.0
            source, latency = super().access(key, phase, cache_type)
            latency += self._eviction_latency
            if source == 'gpu' and self.cache.tiering_policy == 'gpu-selective':
                entry = self.cache.cache_entries[key]
                if entry['access_count'] >= 2:
                    _, elapsed = self._backup(key, 'reuse')
                    latency += elapsed
            return source, latency

    def get_stats(self, duration):
        with self.lock:
            stats = super().get_stats(duration)
            stats['tiering_policy'] = self.cache.tiering_policy
            stats['eviction_io_bytes'] = self.metrics['eviction_io_bytes']
            for trigger in ('insertion', 'reuse', 'eviction'):
                for suffix in ('count', 'bytes'):
                    name = f'backup_{trigger}_{suffix}'
                    stats[name] = self.metrics[name]
            for name in ('backup_failures', 'discarded_unbacked_entries', 'unexpected_gpu_evictions'):
                stats[name] = self.metrics[name]
            return stats
