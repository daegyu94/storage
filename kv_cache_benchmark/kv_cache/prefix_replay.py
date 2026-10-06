"""Sequential request-boundary reuse projection, separate from MLPerf workloads."""

import hashlib
import json
from collections import Counter

from kv_cache.models import InferencePhase


def prefix_keys(tokens, chunk_size, namespace):
    """Stable simulator identity; complete chunks include their parent prefix."""
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError('chunk_size must be a positive integer')
    if not isinstance(namespace, str) or not namespace:
        raise ValueError('namespace must be a nonempty model/adapter/cache-salt identity')
    if not isinstance(tokens, list) or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError('tokens must be a list of nonnegative integer token IDs')
    parent = namespace
    keys = []
    for start in range(0, len(tokens) - chunk_size + 1, chunk_size):
        parent = hashlib.sha256(json.dumps(
            [parent, tokens[start:start + chunk_size]], separators=(',', ':')
        ).encode()).hexdigest()
        keys.append(parent)
    return keys


class PrefixReplay:
    """Replay completed sequential requests; decode has no external-cache reads."""

    def __init__(self, cache, chunk_size=256, namespace='model/default'):
        prefix_keys([], chunk_size, namespace)
        self.cache = cache
        self.chunk_size = chunk_size
        self.namespace = namespace
        self.metrics = Counter()

    def request(self, tokens):
        keys = prefix_keys(tokens, self.chunk_size, self.namespace)
        reused = 0
        for key in keys:
            # Metadata lookup has no payload I/O and stops at the first gap.
            if self.cache.check_cache_exists(key)[0] is None:
                break
            source, _ = self.cache.access_cache(key, phase=InferencePhase.PREFILL)
            if source is None:
                break
            self.metrics[f'prefix_{source}_hits'] += 1
            reused += self.chunk_size
        self.metrics['requests'] += 1
        self.metrics['prompt_tokens'] += len(tokens)
        self.metrics['reused_tokens'] += reused
        self.metrics['computed_tokens'] += len(tokens) - reused
        self.metrics['uncached_tail_tokens'] += len(tokens) % self.chunk_size
        for key in keys[reused // self.chunk_size:]:
            if self.cache.check_cache_exists(key)[0] is None:
                ok, _, _ = self.cache.allocate_cache(key, self.chunk_size)
                if not ok:
                    self.metrics['store_failures'] += 1
        return reused

    def get_stats(self):
        return {name: self.metrics[name] for name in (
            'requests', 'prompt_tokens', 'reused_tokens', 'computed_tokens',
            'uncached_tail_tokens', 'store_failures', 'prefix_gpu_hits',
            'prefix_cpu_hits', 'prefix_nvme_hits',
        )}
