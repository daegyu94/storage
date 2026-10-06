"""Deterministic completed-operation placement and payload-I/O tests."""

import csv
import importlib.util
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kv_cache.cache import MultiTierCache
from kv_cache.config import ConfigLoader, set_config
from kv_cache.models import GenerationMode, InferencePhase, InferenceRequest, ModelConfig
from kv_cache.tracer import IOTracer

MODEL = ModelConfig('fixture', 1, 4, 1, 1)  # 16 bytes/token
SIZE = 64


@pytest.fixture(autouse=True)
def small_generator(monkeypatch):
    class Generator:
        def __init__(self, *args, **kwargs):
            pass

        def generate(self, sequence_length, key):
            return np.arange(sequence_length * 8, dtype=np.float16)

    monkeypatch.setattr('kv_cache.cache.KVCacheGenerator', Generator)
    set_config(None)
    yield
    set_config(None)


def make_cache(tmp_path, policy='cascade', cpu=2, disk=8, gpu=0, trace=True):
    tracer = IOTracer(str(tmp_path / f'{policy}.csv')) if trace else None
    cache = MultiTierCache(MODEL, gpu * SIZE / 1024**3, cpu * SIZE / 1024**3,
                           cache_dir=str(tmp_path / policy), storage_capacity_gb=disk * SIZE / 1024**3,
                           io_tracer=tracer, tiering_policy=policy)
    return cache


def put(cache, key):
    return cache.allocate_cache(key, 4)


def test_replicate_before_eviction_and_reuse(tmp_path):
    cache = make_cache(tmp_path)
    assert put(cache, 'a')[:2] == (True, 'cpu')
    assert cache.stats['tier_storage_kv_bytes_written'] == SIZE
    put(cache, 'b')
    put(cache, 'c')
    assert cache.cache_entries['a']['replicas'] == {'nvme'}
    assert cache.access_cache('a')[0] == 'nvme'
    assert cache.access_cache('a')[0] == 'cpu'
    assert cache.cache_entries['a']['replicas'] == {'cpu', 'nvme'}
    stats = cache.get_stats(1)
    assert stats['tier_storage_kv_bytes_read'] == SIZE
    assert stats['replication_bytes'] == 3 * SIZE
    assert stats['promotion_nvme_cpu_bytes'] == SIZE
    assert stats['tier_cpu_evictions'] == 2
    assert stats['tier_cpu_misses'] == 1
    assert stats['tier_nvme_hits'] == stats['tier_cpu_hits'] == 1
    assert stats['eviction_io_bytes'] == 0
    assert stats['cpu_entries'] == 2 and stats['storage_entries'] == 3


def test_gpu_staging_and_source_retention(tmp_path):
    cache = make_cache(tmp_path, gpu=1)
    put(cache, 'a')
    put(cache, 'b')
    put(cache, 'c')
    assert cache.cache_entries['a']['replicas'] == {'nvme'}
    assert cache.access_cache('a')[0] == 'nvme'
    assert cache.access_cache('a')[0] == 'gpu'
    assert cache.cache_entries['a']['replicas'] == {'gpu', 'cpu', 'nvme'}
    stats = cache.get_stats(1)
    assert stats['promotion_nvme_cpu_bytes'] == SIZE
    assert stats['promotion_cpu_gpu_bytes'] == SIZE
    assert stats['tier_storage_kv_bytes_written'] == 3 * SIZE
    assert stats['tier_storage_kv_bytes_read'] == SIZE


def test_independent_disk_eviction_and_metadata_probe(tmp_path):
    cache = make_cache(tmp_path, cpu=2, disk=1)
    put(cache, 'a')
    put(cache, 'b')
    assert cache.cache_entries['a']['replicas'] == {'cpu'}
    before = cache.stats.copy()
    assert cache.check_cache_exists('a') == ('cpu', SIZE)
    assert cache.stats == before
    assert cache.access_cache('a')[0] == 'cpu'
    put(cache, 'c')
    assert cache.check_cache_exists('b') == (None, 0)
    assert cache.get_stats(1)['lost_entries'] == 1
    assert cache.access_cache('b')[0] is None


def test_duplicates_are_serialized(tmp_path):
    cache = make_cache(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: put(cache, 'a'), range(20)))
    assert all(r[0] for r in results)
    stats = cache.get_stats(1)
    assert stats['tier_storage_write_operations'] == 1
    assert stats['duplicate_stores'] == 19
    assert cache.cpu_memory_used == cache.nvme_memory_used == SIZE


def test_capacity_and_failed_writes(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='CPU'):
        make_cache(tmp_path, cpu=0)
    cache = make_cache(tmp_path, cpu=1)
    assert cache.allocate_cache('large', 8)[0] is False
    assert not cache.cache_entries

    def fail(*args):
        raise OSError('injected write failure')

    monkeypatch.setattr(cache.backends['nvme'], 'write_size', fail)
    assert put(cache, 'a')[0]
    assert cache.cache_entries['a']['replicas'] == {'cpu'}
    assert cache.nvme_memory_used == 0
    assert cache.get_stats(1)['tier_nvme_write_failures'] == 1
    assert cache.access_cache('a')[0] == 'cpu'


def test_read_failure_is_a_miss_and_keeps_metadata(tmp_path, monkeypatch):
    cache = make_cache(tmp_path, cpu=1, trace=False)
    put(cache, 'a')
    put(cache, 'b')

    def fail(*args):
        raise OSError('injected read failure')

    monkeypatch.setattr(cache.backends['nvme'], 'read', fail)
    assert cache.access_cache('a') == (None, 0.0)
    assert cache.cache_entries['a']['replicas'] == {'nvme'}
    assert cache.stats['cache_hits'] == 0 and cache.stats['cache_misses'] == 1
    assert cache.get_stats(1)['tier_nvme_read_failures'] == 1


def test_real_disk_payload_and_promotion(tmp_path):
    cache = make_cache(tmp_path, cpu=1, trace=False)
    put(cache, 'a')
    put(cache, 'b')
    expected = np.arange(32, dtype=np.float16)
    assert cache.access_cache('a')[0] == 'nvme'
    np.testing.assert_array_equal(cache.backends['cpu'].read('a')[0], expected)
    np.testing.assert_array_equal(cache.backends['nvme'].read('a')[0], expected)
    assert cache.access_cache('a')[0] == 'cpu'
    assert len(list((tmp_path / 'cascade').glob('*.npy'))) == 2


def test_gpu_backend_eviction_drops_only_gpu_replica(tmp_path):
    cache = make_cache(tmp_path, gpu=1)
    put(cache, 'a')
    cache.backends['gpu'].delete('a')
    cache._handle_gpu_eviction('a', 'gpu', SIZE)
    assert cache.cache_entries['a']['replicas'] == {'cpu', 'nvme'}
    assert cache.gpu_memory_used == 0
    assert cache.access_cache('a')[0] == 'cpu'


def test_tp_and_reset_preserve_placement(tmp_path):
    cache = make_cache(tmp_path)
    cache.tensor_parallel = 2
    put(cache, 'a')
    assert cache.cpu_memory_used == cache.nvme_memory_used == SIZE // 2
    cache.reset_stats()
    assert cache.get_stats(1)['replication_bytes'] == 0
    assert cache.check_cache_exists('a') == ('cpu', SIZE // 2)
    assert cache.access_cache('a')[0] == 'cpu'


def test_config_policy_validation(tmp_path):
    path = tmp_path / 'config.yaml'
    path.write_text('tiering:\n  policy: cascade\n')
    assert ConfigLoader(str(path)).get('tiering', 'policy') == 'cascade'
    path.write_text('tiering:\n  policy: invalid\n')
    with pytest.raises(ValueError, match='policy'):
        ConfigLoader(str(path))


def test_default_and_explicit_waterfall_match(tmp_path):
    traces, stats = [], []
    for name, kwargs in [('default', {}), ('explicit', {'tiering_policy': 'waterfall'})]:
        with IOTracer(str(tmp_path / f'{name}.csv')) as tracer:
            cache = MultiTierCache(MODEL, 0, 2.5 * SIZE / 1024**3,
                                   io_tracer=tracer, storage_capacity_gb=8 * SIZE / 1024**3, **kwargs)
            for key in 'abc':
                put(cache, key)
            for key in 'aaba':
                cache.access_cache(key)
            stats.append(cache.get_stats(1))
        with (tmp_path / f'{name}.csv').open() as handle:
            traces.append([{k: v for k, v in row.items() if k != 'Timestamp'}
                           for row in csv.DictReader(handle)])
    assert stats[0] == stats[1]
    assert traces[0] == traces[1]
    assert 'tiering_policy' not in stats[0]


def test_secondary_rejection_keeps_cpu_and_no_spill_io(tmp_path):
    cache = make_cache(tmp_path, disk=0.5)
    assert put(cache, 'a')[0]
    assert cache.cache_entries['a']['replicas'] == {'cpu'}
    assert cache.get_stats(1)['tier_nvme_admission_failures'] == 1
    assert cache.stats['tier_storage_kv_bytes_written'] == 0


def test_staging_write_failure_keeps_secondary_replica(tmp_path, monkeypatch):
    cache = make_cache(tmp_path, cpu=1)
    put(cache, 'a')
    put(cache, 'b')

    def fail(*args):
        raise OSError('injected staging write failure')

    monkeypatch.setattr(cache.backends['cpu'], 'write_size', fail)
    assert cache.access_cache('a')[0] is None
    assert cache.cache_entries['a']['replicas'] == {'nvme'}
    assert cache.cpu_memory_used == 0
    assert cache.get_stats(1)['tier_cpu_write_failures'] == 1


def test_trace_promotion_does_not_materialize_payload(tmp_path, monkeypatch):
    cache = make_cache(tmp_path, cpu=1)
    put(cache, 'a')
    put(cache, 'b')

    def fail(*args):
        raise AssertionError('Trace mode must not allocate a dummy buffer')

    monkeypatch.setattr(cache.backends['nvme'], 'read', fail)
    assert cache.access_cache('a')[0] == 'nvme'


@pytest.mark.parametrize('configured,explicit,expected', [
    (None, None, 'waterfall'), ('cascade', None, 'cascade'),
    ('cascade', 'waterfall', 'waterfall'), ('waterfall', 'cascade', 'cascade'),
    ('gpu-selective', None, 'gpu-selective'),
    ('waterfall', 'gpu-write-back', 'gpu-write-back'),
    (None, 'gpu-write-through', 'gpu-write-through'),
])
def test_standalone_cli_policy_precedence(tmp_path, monkeypatch, configured, explicit, expected):
    import kv_cache.cli as cli

    captured = {}

    class Benchmark:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            return {}

    monkeypatch.setattr(cli, 'IntegratedBenchmark', Benchmark)
    argv = ['kv-cache', '--model', 'tiny-1b', '--output', str(tmp_path / 'result.json')]
    if configured:
        path = tmp_path / 'cli.yaml'
        path.write_text(f'tiering:\n  policy: {configured}\n')
        argv += ['--config', str(path)]
    if explicit:
        argv += ['--tiering-policy', explicit]
    monkeypatch.setattr(sys, 'argv', argv)
    cli.main()
    assert captured['tiering_policy'] == expected


def test_comparison_utility_known_outcomes():
    path = Path(__file__).resolve().parents[1] / 'utils' / 'compare_tiering.py'
    spec = importlib.util.spec_from_file_location('compare_tiering', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.compare()
    assert set(result['scenarios']) == {'hot_fit', 'reused_cold', 'thrashing', 'disk_pressure'}


@pytest.mark.parametrize('policy,writes,reads', [('waterfall', SIZE, 3 * SIZE),
                                               ('cascade', 3 * SIZE, SIZE)])
def test_benchmark_worker_replays_same_prefill_decode_workload(tmp_path, policy, writes, reads):
    from kv_cache.benchmark import IntegratedBenchmark

    benchmark = IntegratedBenchmark(
        model_config=MODEL, num_users=1, gpu_memory_gb=0,
        cpu_memory_gb=2.5 * SIZE / 1024**3, duration_seconds=1,
        storage_capacity_gb=8 * SIZE / 1024**3, cache_dir=str(tmp_path / policy),
        generation_mode=GenerationMode.NONE, enable_prefix_caching=False,
        enable_multi_turn=False, max_requests=4, tiering_policy=policy,
    )
    benchmark.stop_event = threading.Event()
    for index, (key, phase) in enumerate([('a', InferencePhase.PREFILL),
                                         ('b', InferencePhase.PREFILL),
                                         ('c', InferencePhase.PREFILL),
                                         ('a', InferencePhase.DECODE)]):
        request = InferenceRequest('fixture', str(index), datetime(2026, 1, 1),
                                   4, 64 if phase == InferencePhase.DECODE else 0,
                                   0, phase=phase, cache_key=key)
        benchmark.request_queue.put(((0, index), request))
    benchmark.process_requests(benchmark.stop_event)
    assert benchmark.results['requests_completed'] == 4
    stats = benchmark.cache.get_stats(1)
    assert benchmark.cache.stats['tier_storage_kv_bytes_written'] == writes
    assert benchmark.cache.stats['tier_storage_kv_bytes_read'] == reads
    assert stats['cache_hits'] == 3 and stats['cache_misses'] == 0
    assert stats['total_write_bytes'] == 3 * SIZE


def test_same_workload_has_expected_storage_io_difference(tmp_path):
    results = {}
    for policy in ('waterfall', 'cascade'):
        # Waterfall's existing 80% target admits two objects at this capacity.
        cache = make_cache(tmp_path, policy, cpu=2.5)
        for key in ('a', 'b', 'c'):
            put(cache, key)
        hits = [cache.access_cache('a')[0] for _ in range(3)]
        cache.io_tracer.close()
        rows = list(csv.DictReader((tmp_path / f'{policy}.csv').open()))
        results[policy] = (cache, hits, rows)
    waterfall, hits, _ = results['waterfall']
    cascade, cascade_hits, rows = results['cascade']
    assert hits == ['nvme'] * 3
    assert cascade_hits == ['nvme', 'cpu', 'cpu']
    assert waterfall.stats['tier_storage_kv_bytes_written'] == SIZE
    assert cascade.stats['tier_storage_kv_bytes_written'] == 3 * SIZE
    assert waterfall.stats['tier_storage_kv_bytes_read'] == 3 * SIZE
    assert cascade.stats['tier_storage_kv_bytes_read'] == SIZE
    assert any(r['Phase'] == 'Promote' for r in rows)
    assert any(r['Phase'] == 'Replicate' for r in rows)
