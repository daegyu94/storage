#!/usr/bin/env python3
"""Compare completed KV I/O and optionally replay request token IDs/source methods."""

import argparse
import ast
import hashlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kv_cache.cache import MultiTierCache
from kv_cache.gpu_backup import GPU_POLICIES
from kv_cache.models import ModelConfig
from kv_cache.prefix_replay import PrefixReplay
from kv_cache.tracer import IOTracer

POLICIES = ('waterfall', 'cascade', *GPU_POLICIES)
MODEL = ModelConfig('comparison', 1, 4, 1, 1)
SIZE = 64
SGLANG_REVISION = '36b216e2ee7bae046a77a329e2fcda0e33beade5'
SCENARIOS = {
    'fit_once': [('put', 'a'), ('put', 'b')],
    'fit_reuse': [('put', 'a'), ('put', 'b')] + [('get', 'a')] * 3,
    'cold_pressure': [('put', key) for key in 'abcde'] + [('get', 'a')] * 3,
    'selected_survives': [('put', 'a'), ('get', 'a')] + [('put', key) for key in 'bcde']
                         + [('get', 'a')] * 3,
    'thrashing': [('put', key) for key in 'abcde'] + [('get', key) for key in 'abcdeabcde'],
}


def summarize(cache, hits):
    stats = cache.get_stats(1)
    return {
        'nvme_read_bytes': cache.stats['tier_storage_kv_bytes_read'],
        'nvme_write_bytes': cache.stats['tier_storage_kv_bytes_written'],
        'hit_tiers': hits, 'tier_hits': dict(Counter(hit or 'miss' for hit in hits)),
        'cache_misses': cache.stats['cache_misses'],
        'tier_probe_misses': {tier: stats.get(f'tier_{tier}_misses')
                              for tier in ('gpu', 'cpu', 'nvme')},
        'eviction_io_bytes': stats.get('eviction_io_bytes'),
        'replication_bytes': stats.get('replication_bytes', 0),
        'promotion_nvme_cpu_bytes': stats.get('promotion_nvme_cpu_bytes', 0),
        'promotion_cpu_gpu_bytes': stats.get('promotion_cpu_gpu_bytes', 0),
        'backup_bytes': {trigger: stats.get(f'backup_{trigger}_bytes', 0)
                         for trigger in ('insertion', 'reuse', 'eviction')},
        'tier_evictions': {tier: stats.get(f'tier_{tier}_evictions')
                           for tier in ('gpu', 'cpu', 'nvme')},
        'discarded_unbacked_entries': stats.get('discarded_unbacked_entries', 0),
        'replicas': {key: sorted(entry.get('replicas', {entry['location']}))
                     for key, entry in sorted(cache.cache_entries.items())},
    }


def cache_for(directory, policy):
    return MultiTierCache(MODEL, 2.5 * SIZE / 1024**3, 2.5 * SIZE / 1024**3,
                          storage_capacity_gb=8 * SIZE / 1024**3,
                          io_tracer=IOTracer(str(directory / f'{policy}.csv')),
                          tiering_policy=policy)


def compare(requests=None, chunk_size=4, namespace='model/default'):
    result = {'scope': 'completed payload operations; flat entries; sequential requests',
              'entry_bytes': SIZE, 'gpu_capacity_bytes': 160, 'cpu_capacity_bytes': 160,
              'nvme_capacity_bytes': 512, 'scenarios': {}}
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        for name, operations in SCENARIOS.items():
            result['scenarios'][name] = {}
            for policy in POLICIES:
                cache = cache_for(directory, policy)
                hits = []
                for operation, key in operations:
                    if operation == 'put':
                        assert cache.allocate_cache(key, 4)[0]
                    else:
                        hits.append(cache.access_cache(key)[0])
                result['scenarios'][name][policy] = summarize(cache, hits)
                cache.io_tracer.close()
        if requests is not None:
            result['prefix_replay'] = {'chunk_size': chunk_size, 'namespace': namespace,
                                       'policies': {}}
            for policy in POLICIES:
                cache = cache_for(directory, policy)
                replay = PrefixReplay(cache, chunk_size, namespace)
                reused = [replay.request(tokens) for tokens in requests]
                result['prefix_replay']['policies'][policy] = {
                    'reused_tokens_per_request': reused, **replay.get_stats(),
                    **summarize(cache, []),
                }
                cache.io_tracer.close()
    fit = result['scenarios']['fit_once']
    assert fit['gpu-write-back']['nvme_write_bytes'] == 0
    assert fit['gpu-selective']['nvme_write_bytes'] == 0
    assert fit['gpu-write-through']['nvme_write_bytes'] == 2 * SIZE
    reuse = result['scenarios']['fit_reuse']
    assert reuse['gpu-selective']['nvme_write_bytes'] == SIZE
    assert reuse['gpu-selective']['backup_bytes']['reuse'] == SIZE
    cold = result['scenarios']['cold_pressure']
    assert cold['gpu-selective']['cache_misses'] == 3
    assert cold['gpu-write-back']['nvme_read_bytes'] == SIZE
    assert cold['gpu-write-through']['nvme_read_bytes'] == SIZE
    return result


def load_method(source, name, globals_dict):
    tree = ast.parse(source)
    methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(methods) != 1:
        raise ValueError(f'Expected exactly one {name} method')
    method = methods[0]
    method.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[
        ast.alias(name='annotations')], level=0), method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), '<upstream-source>', 'exec'), globals_dict)
    return globals_dict[name]


def validate_sglang(path):
    source = path.read_text()
    globals_dict = {'EvictDeviceLeafResult': lambda: SimpleNamespace(tracker=None, backup_kv=None)}
    count = load_method(source, '_inc_hit_count_and_check', globals_dict)
    evict = load_method(source, 'evict_device_leaf', globals_dict)
    decisions = {}
    for policy in GPU_POLICIES:
        node = SimpleNamespace(id=1, evicted=False, backuped=False, hit_count=0)
        state = SimpleNamespace(is_write_back=policy == 'gpu-write-back',
                                enable_external_cache_linker=False, enable_hicache=True,
                                write_through_threshold=1 if policy == 'gpu-write-through' else 2)
        backups = []
        for _ in range(3):
            backup = count(state, node)
            backups.append(backup)
            if backup:
                node.backuped = True
        assert backups == ([True, False, False] if policy == 'gpu-write-through'
                           else [False, True, False] if policy == 'gpu-selective'
                           else [False, False, False])
        # Independently check eviction of a never-reused, unbacked leaf.
        node.backuped = False
        events = []
        state.node_by_id = lambda _, node=node: node
        state._is_device_leaf = lambda _: True
        state._begin_tracking_unbacked_tokens = lambda: None
        state._finish_tracking_unbacked_tokens = lambda: 4
        state._build_backup_kv_action = lambda *args, **kwargs: 'backup'
        state._delete_unbacked_device_leaf = lambda *args, events=events, **kwargs: events.append('discard')
        state._demote = lambda *args, events=events: events.append('demote')
        # Fields used by the source result's tracker plumbing.
        globals_dict['EvictDeviceLeafResult'] = lambda: SimpleNamespace(
            tracker=None, backup_kv=None, device_frees=[], host_frees=[])
        action = evict(state, 1, state.is_write_back)
        assert (action.backup_kv == 'backup') == (policy == 'gpu-write-back')
        assert events == ([] if policy == 'gpu-write-back' else ['discard'])
        decisions[policy] = {'backup_on_uses': backups,
                             'unbacked_eviction': 'backup' if action.backup_kv else 'discard'}
    # Check simulator decisions on identical use boundaries, rather than only
    # checking the extracted source against handwritten expected outcomes.
    with tempfile.TemporaryDirectory() as temporary:
        for policy, expected in decisions.items():
            cache = cache_for(Path(temporary), policy)
            observed = []
            cache.allocate_cache('a', 4)
            observed.append('cpu' in cache.cache_entries['a']['replicas'])
            for _ in range(2):
                before = cache.get_stats(1)
                cache.access_cache('a')
                after = cache.get_stats(1)
                observed.append(after['backup_reuse_count'] > before['backup_reuse_count'])
            assert observed == expected['backup_on_uses']
            cache.io_tracer.close()
    return {'revision': SGLANG_REVISION, 'sha256': hashlib.sha256(source.encode()).hexdigest(),
            'methods': ['_inc_hit_count_and_check', 'evict_device_leaf'],
            'backup_decisions_match': True, 'decisions': decisions,
            'limits': 'Source methods with doubles; no radix dependencies, runtime GPU or async timing.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', type=Path, help='JSONL integer-token arrays, one completed request per line')
    parser.add_argument('--chunk-size', type=int, default=4)
    parser.add_argument('--namespace', default='model/default')
    parser.add_argument('--sglang-tree-core', type=Path, help='Pinned unified_cache/unified_tree_core.py')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    requests = ([json.loads(line) for line in args.requests.read_text().splitlines() if line.strip()]
                if args.requests else None)
    result = compare(requests, args.chunk_size, args.namespace)
    if args.sglang_tree_core:
        result['sglang_source_harness'] = validate_sglang(args.sglang_tree_core)
    output = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end='')


if __name__ == '__main__':
    main()
