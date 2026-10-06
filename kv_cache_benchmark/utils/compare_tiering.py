#!/usr/bin/env python3
"""Replay identical KV operations and optionally execute upstream LMCache methods.

The source harness executes actual StorageManager methods with deterministic
bounded backend doubles. It validates orchestration at completion boundaries,
not GPU inference, production allocator behavior, or asynchronous performance.
"""

import argparse
import ast
import csv
import hashlib
import json
import sys
import tempfile
import threading
from collections import Counter, OrderedDict
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kv_cache.cache import MultiTierCache
from kv_cache.models import ModelConfig
from kv_cache.tracer import IOTracer

MODEL = ModelConfig('comparison', 1, 4, 1, 1)
TOKENS = 4
SIZE = MODEL.kv_cache_size_per_token * TOKENS
PINNED_LMCACHE_REVISION = '8c77a6f77b4029269199de39c6f509944294c968'
SCENARIOS = {
    'hot_fit': [('put', 'a'), ('put', 'b')] + [('get', 'a')] * 3,
    'reused_cold': [('put', k) for k in 'abc'] + [('get', 'a')] * 3,
    'thrashing': [('put', k) for k in 'abc'] + [('get', k) for k in 'abcabc'],
    'disk_pressure': [('put', k) for k in 'abc'] + [('get', k) for k in 'abc'],
}


def run_simulator(directory, policy, operations, disk_entries):
    path = directory / f'{policy}.csv'
    with IOTracer(str(path)) as tracer:
        cache = MultiTierCache(MODEL, 0, 2.5 * SIZE / 1024**3,
                               storage_capacity_gb=disk_entries * SIZE / 1024**3,
                               io_tracer=tracer, tiering_policy=policy)
        hits = []
        placements = []
        for operation, key in operations:
            if operation == 'put':
                assert cache.allocate_cache(key, TOKENS)[0]
            else:
                hits.append(cache.access_cache(key)[0])
            placements.append({k: sorted(e.get('replicas', {e['location']}))
                               for k, e in sorted(cache.cache_entries.items())})
        stats = cache.get_stats(1)
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    disk_io = [(r['Operation'], r['Key'], int(r['Object_Size_Bytes']))
               for r in rows if r['Tier'] == 'Tier-2']
    summary = {
        'nvme_read_bytes': cache.stats['tier_storage_kv_bytes_read'],
        'nvme_write_bytes': cache.stats['tier_storage_kv_bytes_written'],
        'hit_tiers': hits,
        'tier_hits': dict(Counter(t or 'miss' for t in hits)),
        'tier_misses': {'cpu': sum(t != 'cpu' for t in hits), 'nvme': hits.count(None)},
        'eviction_io_bytes': sum(int(r['Object_Size_Bytes']) for r in rows if r['Phase'] == 'Evict'),
        'promotion_bytes': stats.get('promotion_nvme_cpu_bytes', 0),
        'replication_bytes': stats.get('replication_bytes', 0),
        'tier_evictions': ({t: stats[f'tier_{t}_evictions'] for t in ('cpu', 'nvme')}
                           if policy == 'cascade' else {
                               'cpu': sum(r['Operation'] == 'Read' and r['Tier'] == 'Tier-1'
                                          and r['Phase'] == 'Evict' for r in rows),
                               'nvme': cache.stats['evictions'] - sum(
                                   r['Operation'] == 'Read' and r['Phase'] == 'Evict' for r in rows),
                           }),
        'cache_misses': cache.stats['cache_misses'],
    }
    return summary, disk_io, placements


class MemoryObject:
    def ref_count_down(self):
        pass


class CPUBackend:
    """Backend double; capacity, LRU, and I/O are assumptions of this harness."""
    name = 'LocalCPUBackend'
    tier = 'cpu'

    def __init__(self, manager, capacity=2):
        self.manager = manager
        self.capacity = capacity
        self.objects = OrderedDict()

    def get_allocator_backend(self):
        return self.manager.allocator_backend

    def submit_put_task(self, key, obj):
        if key in self.objects:
            return
        while len(self.objects) >= self.capacity:
            self.objects.popitem(last=False)
        self.objects[key] = obj
        if self.tier == 'nvme':
            self.manager.disk_io.append(('Write', key, SIZE))

    def batched_submit_put_task(self, keys, objs, transfer_spec=None):
        for key, obj in zip(keys, objs, strict=True):
            self.submit_put_task(key, obj)

    def get_blocking(self, key):
        if key not in self.objects:
            return None
        self.objects.move_to_end(key)
        self.manager.last_source = self.tier
        if self.tier == 'nvme':
            self.manager.disk_io.append(('Read', key, SIZE))
        return self.objects[key]


class DiskBackend(CPUBackend):
    name = 'LocalDiskBackend'
    tier = 'nvme'


def load_lmcache_methods(path):
    source = path.read_text()
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'StorageManager')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in ('batched_put', 'get')]
    if len(methods) != 2:
        raise ValueError('Expected StorageManager.batched_put and get in supplied source')
    for method in methods:
        method.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                              *methods], type_ignores=[])
    namespace = {'get_backend_cname': lambda b: b.name, 'LocalCPUBackend': CPUBackend}
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    return namespace, hashlib.sha256(source.encode()).hexdigest()


def validate_vllm_fanout(path):
    source = path.read_text()
    cls = next(n for n in ast.parse(source).body
               if isinstance(n, ast.ClassDef) and n.name == 'TieringOffloadingManager')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'complete_store')
    method.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                              method], type_ignores=[])
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    outcomes = []
    for success, admitted, expected in [(True, (True, True), [0, 1]),
                                        (False, (True, True), []),
                                        (True, (True, False), [0])]:
        calls = []
        context = SimpleNamespace(req_id='fixture')
        state = SimpleNamespace(pending_primary_stores=1)
        tiers = [SimpleNamespace(admitted=a, submit_store=lambda job, captured=calls: captured.append(job))
                 for a in admitted]
        manager = SimpleNamespace(
            primary_tier=SimpleNamespace(complete_store=lambda *args: None), secondary_tiers=tiers,
            _should_store_to_tier=lambda tier, count: tier.admitted,
            create_store_job=lambda keys, context, index: index,
            _req_state={'fixture': state}, _maybe_finalize_request=lambda req_id: None,
        )
        namespace['complete_store'](manager, ['a'], context, success)
        assert calls == expected and state.pending_primary_stores == 0
        outcomes.append({'primary_success': success, 'admitted': admitted, 'secondary_jobs': calls})
    return {'method_file_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'scope': 'actual vLLM complete_store method; transfer/request doubles; fan-out contract only',
            'cases': outcomes}


def run_lmcache(methods, operations, disk_entries):
    manager = SimpleNamespace(_bypass_lock=threading.Lock(), _bypassed_backends=set(),
                              internal_copy_stream=None, disk_io=[], last_source=None)
    cpu = CPUBackend(manager)
    disk = DiskBackend(manager, capacity=int(disk_entries))
    manager.allocator_backend = cpu
    manager.storage_backends = {cpu.name: cpu, disk.name: disk}
    manager.get_active_storage_backends = lambda location: list(manager.storage_backends.items())
    hits, placements = [], []
    for operation, key in operations:
        if operation == 'put':
            methods['batched_put'](manager, [key], [MemoryObject()])
        else:
            manager.last_source = None
            methods['get'](manager, key)
            hits.append(manager.last_source)
        keys = sorted(set(cpu.objects) | set(disk.objects))
        placements.append({k: [b.tier for b in (cpu, disk) if k in b.objects] for k in keys})
    return hits, manager.disk_io, placements


def compare(lmcache_path=None, source_revision=None):
    result = {
        'object_bytes': SIZE,
        'cpu_capacity_bytes': int(2.5 * SIZE),
        'scope': 'logical payload I/O; synchronous completed operations; no inference timing',
        'scenarios': {},
    }
    methods = None
    if lmcache_path:
        methods, digest = load_lmcache_methods(lmcache_path)
        result['source_harness'] = {
            'method_file_sha256': digest, 'revision_label': source_revision,
            'pinned_design_revision': PINNED_LMCACHE_REVISION,
            'scope': 'actual LMCache batched_put/get methods; CPU/disk backend doubles',
        }
    with tempfile.TemporaryDirectory() as root:
        for name, operations in SCENARIOS.items():
            directory = Path(root) / name
            directory.mkdir()
            disk_entries = 2.5 if name == 'disk_pressure' else 8
            waterfall, _, _ = run_simulator(directory, 'waterfall', operations, disk_entries)
            cascade, disk_io, placements = run_simulator(directory, 'cascade', operations, disk_entries)
            item = {'operations': operations, 'disk_capacity_bytes': int(disk_entries * SIZE),
                    'waterfall': waterfall, 'cascade': cascade}
            if methods:
                hits, upstream_io, upstream_placements = run_lmcache(methods, operations, disk_entries)
                # Compare each operation boundary, not only aggregate totals.
                assert hits == cascade['hit_tiers'], (name, hits, cascade['hit_tiers'])
                assert upstream_io == disk_io, (name, upstream_io, disk_io)
                assert upstream_placements == placements, (name, upstream_placements, placements)
                item['lmcache_source_match'] = True
                item['lmcache_disk_io'] = upstream_io
            result['scenarios'][name] = item
    # Explicit known outcomes prevent matching two implementations that make
    # the same mistake from being mistaken for validation.
    reused = result['scenarios']['reused_cold']
    assert reused['waterfall']['nvme_write_bytes'] == SIZE
    assert reused['waterfall']['nvme_read_bytes'] == 3 * SIZE
    assert reused['cascade']['nvme_write_bytes'] == 3 * SIZE
    assert reused['cascade']['nvme_read_bytes'] == SIZE
    assert result['scenarios']['hot_fit']['waterfall']['nvme_write_bytes'] == 0
    assert result['scenarios']['hot_fit']['cascade']['nvme_write_bytes'] == 2 * SIZE
    assert result['scenarios']['thrashing']['cascade']['nvme_read_bytes'] == 6 * SIZE
    assert result['scenarios']['disk_pressure']['cascade']['cache_misses'] == 1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lmcache-storage-manager', type=Path,
                        help='Optional upstream lmcache/v1/storage_backend/storage_manager.py')
    parser.add_argument('--source-revision', help='Revision label for the supplied method file')
    parser.add_argument('--vllm-tiering-manager', type=Path,
                        help='Optional upstream vllm/v1/kv_offload/tiering/manager.py')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = compare(args.lmcache_storage_manager, args.source_revision)
    if args.vllm_tiering_manager:
        result['vllm_source_harness'] = validate_vllm_fanout(args.vllm_tiering_manager)
    output = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    else:
        print(output, end='')


if __name__ == '__main__':
    main()
