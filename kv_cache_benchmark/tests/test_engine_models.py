"""Admission and request reuse tests with deterministic tiny KV payloads."""

import numpy as np
import pytest

from kv_cache.cache import MultiTierCache
from kv_cache.config import set_config
from kv_cache.models import ModelConfig
from kv_cache.prefix_replay import PrefixReplay, prefix_keys
from kv_cache.tracer import IOTracer

MODEL = ModelConfig('fixture', 1, 4, 1, 1)
SIZE = 64
POLICIES = ('gpu-write-through', 'gpu-selective', 'gpu-write-back')


@pytest.fixture(autouse=True)
def generator(monkeypatch):
    class Generator:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, sequence_length, key):
            return np.arange(sequence_length * 8, dtype=np.float16)

    monkeypatch.setattr('kv_cache.cache.KVCacheGenerator', Generator)
    set_config(None)
    yield
    set_config(None)


def make_cache(tmp_path, policy='gpu-selective', gpu=2, cpu=2, disk=8):
    return MultiTierCache(MODEL, gpu * SIZE / 1024**3, cpu * SIZE / 1024**3,
                          cache_dir=str(tmp_path / policy),
                          storage_capacity_gb=disk * SIZE / 1024**3,
                          io_tracer=IOTracer(str(tmp_path / f'{policy}.csv')),
                          tiering_policy=policy)


def put(cache, key):
    return cache.allocate_cache(key, 4)


@pytest.mark.parametrize('policy', POLICIES)
def test_fit_then_reuse(tmp_path, policy):
    cache = make_cache(tmp_path, policy)
    put(cache, 'a')
    put(cache, 'b')
    expected = 2 * SIZE if policy == 'gpu-write-through' else 0
    assert cache.stats['tier_storage_kv_bytes_written'] == expected
    for _ in range(3):
        assert cache.access_cache('a')[0] == 'gpu'
    assert cache.stats['tier_storage_kv_bytes_written'] == (
        2 * SIZE if policy == 'gpu-write-through' else SIZE if policy == 'gpu-selective' else 0)
    assert cache.stats['tier_storage_kv_bytes_read'] == 0
    assert cache.get_stats(1)['backup_reuse_count'] == int(policy == 'gpu-selective')


@pytest.mark.parametrize('policy', POLICIES)
def test_cold_eviction_and_promotion(tmp_path, policy):
    cache = make_cache(tmp_path, policy, gpu=1, cpu=1)
    for key in 'abc':
        assert put(cache, key)[0]
    source, _ = cache.access_cache('a')
    stats = cache.get_stats(1)
    if policy == 'gpu-selective':
        assert source is None
        assert stats['discarded_unbacked_entries'] == 2
        assert stats['tier_storage_kv_bytes_written'] == 0
    else:
        assert source == 'nvme'
        assert cache.access_cache('a')[0] == 'gpu'
        assert stats['tier_storage_kv_bytes_read'] == SIZE
        assert stats['promotion_nvme_cpu_bytes'] == SIZE
        assert stats['promotion_cpu_gpu_bytes'] == SIZE
        assert stats['tier_storage_kv_bytes_written'] == 3 * SIZE
        assert 'nvme' in cache.cache_entries['a']['replicas']
    if policy == 'gpu-write-back':
        # Two allocation evictions plus promotion eviction of c.
        assert stats['backup_eviction_bytes'] == 3 * SIZE
        assert stats['eviction_io_bytes'] == 12 * SIZE


def test_selected_entry_survives_gpu_eviction(tmp_path):
    cache = make_cache(tmp_path, gpu=1)
    put(cache, 'a')
    assert cache.check_cache_exists('a')[0] == 'gpu'
    assert cache.get_stats(1)['backup_reuse_count'] == 0
    cache.access_cache('a')
    put(cache, 'b')
    assert cache.access_cache('a')[0] == 'cpu'
    assert cache.stats['tier_storage_kv_bytes_written'] == SIZE


def test_write_back_failure_retains_victim(tmp_path, monkeypatch):
    cache = make_cache(tmp_path, 'gpu-write-back', gpu=1)
    put(cache, 'a')

    def fail(*args):
        raise OSError('injected host transfer failure')

    monkeypatch.setattr(cache.backends['cpu'], 'write_size', fail)
    assert not put(cache, 'b')[0]
    assert cache.cache_entries['a']['replicas'] == {'gpu'}
    assert 'b' not in cache.cache_entries
    stats = cache.get_stats(1)
    assert stats['backup_failures'] == 1
    assert stats['tier_gpu_eviction_failures'] == 1
    assert cache.gpu_memory_used == SIZE


def test_secondary_failure_preserves_primary_and_unexpected_eviction(tmp_path, monkeypatch):
    cache = make_cache(tmp_path, 'gpu-write-through', gpu=1)

    def fail(*args):
        raise OSError('injected secondary failure')

    monkeypatch.setattr(cache.backends['nvme'], 'write_size', fail)
    assert put(cache, 'a')[0]
    assert cache.cache_entries['a']['replicas'] == {'gpu', 'cpu'}
    assert cache.get_stats(1)['tier_nvme_write_failures'] == 1
    cache = make_cache(tmp_path, 'gpu-write-back', gpu=1)
    put(cache, 'a')
    cache.backends['gpu'].delete('a')
    cache._handle_gpu_eviction('a', 'gpu', SIZE)
    assert cache.check_cache_exists('a') == (None, 0)
    assert cache.get_stats(1)['unexpected_gpu_evictions'] == 1
    assert cache.get_stats(1)['discarded_unbacked_entries'] == 1


@pytest.mark.parametrize('policy', POLICIES)
def test_duplicate_oversized_reset_and_capacity(tmp_path, policy):
    cache = make_cache(tmp_path, policy, gpu=1, cpu=1, disk=1)
    put(cache, 'a')
    before = cache.stats['tier_storage_kv_bytes_written']
    put(cache, 'a')
    assert cache.stats['tier_storage_kv_bytes_written'] == before
    assert not cache.allocate_cache('oversized', 8)[0]
    for key in 'bcdef':
        assert put(cache, key)[0]
        assert cache.gpu_memory_used <= SIZE
        assert cache.cpu_memory_used <= SIZE
        assert cache.nvme_memory_used <= SIZE
    entries = set(cache.cache_entries)
    cache.reset_stats()
    assert set(cache.cache_entries) == entries
    assert cache.get_stats(1)['backup_eviction_count'] == 0


@pytest.mark.parametrize('policy', POLICIES)
def test_requires_gpu(tmp_path, policy):
    with pytest.raises(ValueError, match='GPU'):
        make_cache(tmp_path, policy, gpu=0)


def test_prefix_identity_and_partial_tail(tmp_path):
    a = prefix_keys([1, 2, 3, 4], 2, 'model')
    b = prefix_keys([9, 2, 3, 4], 2, 'model')
    assert a[0] != b[0] and a[1] != b[1]
    assert a != prefix_keys([1, 2, 3, 4], 2, 'adapter')
    cache = make_cache(tmp_path, 'gpu-write-through', gpu=8)
    replay = PrefixReplay(cache, chunk_size=2)
    assert replay.request([1, 2, 3, 4, 5]) == 0
    assert replay.request([1, 2, 3, 4, 5]) == 4
    assert replay.get_stats()['computed_tokens'] == 6
    assert replay.get_stats()['uncached_tail_tokens'] == 2
    assert cache.stats['tier_storage_kv_bytes_written'] == SIZE
    assert cache.stats['tier_storage_kv_bytes_read'] == 0
    assert cache.stats['decode_reads'] == 0


def test_prefix_stops_at_first_gap(tmp_path):
    cache = make_cache(tmp_path, 'gpu-write-through', gpu=8)
    replay = PrefixReplay(cache, chunk_size=4)
    tokens = list(range(12))
    keys = prefix_keys(tokens, 4, replay.namespace)
    put(cache, keys[0])
    put(cache, keys[2])
    assert replay.request(tokens) == 4
    assert cache.stats['read_operations'] == 1
    assert replay.get_stats()['computed_tokens'] == 8


@pytest.mark.parametrize('tokens,chunk,namespace', [([True], 1, 'm'), ([1], 0, 'm'),
                                                  ([1], 1, ''), ([-1], 1, 'm')])
def test_invalid_prefix_input(tokens, chunk, namespace):
    with pytest.raises(ValueError):
        prefix_keys(tokens, chunk, namespace)


def test_real_nvme_backup_and_payload(tmp_path):
    # Host-backed GPU double exercises real payload transfer/local files,
    # rather than requiring a GPU to validate completed-operation semantics.
    from kv_cache.backends import CPUMemoryBackend

    cache = make_cache(tmp_path, 'gpu-write-back', gpu=1, cpu=1)
    cache.io_tracer.close()
    cache.io_tracer = None
    from kv_cache.backends import NVMeBackend
    cache.backends = {'gpu': CPUMemoryBackend(), 'cpu': CPUMemoryBackend(),
                      'nvme': NVMeBackend(base_path=str(tmp_path / 'files'))}
    cache.generator = type('Generator', (), {
        'generate': lambda self, sequence_length, key: np.arange(sequence_length * 8, dtype=np.float16)
    })()
    for key in 'abc':
        assert put(cache, key)[0]
    assert cache.access_cache('a')[0] == 'nvme'
    actual, _ = cache.backends['gpu'].read('a')
    np.testing.assert_array_equal(actual, np.arange(32, dtype=np.float16))
    assert cache.stats['tier_storage_kv_bytes_written'] == 3 * SIZE
    assert cache.stats['tier_storage_kv_bytes_read'] == SIZE


def test_same_workload_comparison():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / 'utils' / 'compare_engine_models.py'
    spec = importlib.util.spec_from_file_location('engine_comparison', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.compare([[1, 2, 3, 4], [1, 2, 3, 4]])
    cold = result['scenarios']['cold_pressure']
    assert cold['waterfall']['nvme_read_bytes'] == 3 * SIZE
    assert cold['gpu-write-back']['backup_bytes']['eviction'] == 4 * SIZE
    assert cold['gpu-selective']['discarded_unbacked_entries'] == 3
    thrash = result['scenarios']['thrashing']
    assert thrash['gpu-write-back']['nvme_read_bytes'] > thrash['waterfall']['nvme_read_bytes']
    for row in result['prefix_replay']['policies'].values():
        assert row['reused_tokens_per_request'] == [0, 4]
