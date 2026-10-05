"""Source-grounded dense KV geometry and real lifecycle payload contracts."""

import json
from dataclasses import asdict

import numpy as np
import pytest
from test_agentrl import tiny
from test_agentrl_cli import invoke
from test_agentrl_kv_lifecycle import configured, execute

from kv_cache.agentrl import AgentRLConfig, create_lifecycle
from kv_cache.agentrl_trace import analyze_trace, load_trace
from kv_cache.backends import NVMeBackend


def reference(tp=1, **overrides):
    return {"gpu_name": "declared-target-gpu", "tensor_parallel_size": tp, **overrides}


def qwen(tp=1, *, large=False, dtype="bfloat16", **overrides):
    # Independently read from pinned Qwen3-8B / Qwen3-32B HF config.json.
    return AgentRLConfig.from_dict(
        {
            **asdict(tiny()),
            "model": {
                "name": "Qwen/Qwen3-32B" if large else "Qwen/Qwen3-8B",
                "num_layers": 64 if large else 36,
                "hidden_dim": 5120 if large else 4096,
                "num_heads": 64 if large else 32,
                "kv_heads": 8,
                "_kv_dim_override": 128,
                "attention_type": "gqa",
                "dtype": dtype,
            },
            "rollout_reference": reference(tp),
            **overrides,
        }
    )


@pytest.mark.parametrize("tp", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("large", [False, True])
def test_pinned_qwen_geometry_includes_tp_replication(tp, large):
    config = qwen(tp, large=large)
    report = config.reference_summary
    g = report["kv_geometry"]
    unique = 262144 if large else 147456
    factor = max(1, tp // 8)
    assert g["unique_bytes_per_token"] == unique
    assert g["worker_bytes_per_token"] == unique * factor // tp
    assert g["worker_kv_heads"] == max(1, 8 // tp)
    assert g["kv_replication_factor"] == factor
    assert config.bytes_per_token == g["replica_storage_bytes_per_token"] == unique * factor
    assert report["storage_payload_scope"] == "replica_aggregate_unpadded"
    assert report["framework"] == "verl" and report["orchestrator"] == "ray"
    assert report["engine"] == "vllm" and report["engine_version"] == "0.29.0"
    assert report["service_time_fidelity"] == report["connector_behavior_fidelity"] == "uncalibrated"


@pytest.mark.parametrize("dtype,multiplier", [("float16", 1), ("bfloat16", 1), ("float32", 2)])
def test_kv_dtype_and_explicit_head_dim_are_not_model_weight_dtype(dtype, multiplier):
    config = qwen(4, large=True, dtype=dtype)
    assert config.bytes_per_token == 262144 * multiplier
    # hidden_size / q_heads is 80, while this model's actual head_dim is 128.
    assert config.reference_summary["kv_geometry"]["head_dim"] == 128


@pytest.mark.parametrize(
    "overrides",
    [
        {"tensor_parallel_size": 0},
        {"tensor_parallel_size": True},
        {"tensor_parallel_size": 3},
        {"tensor_parallel_size": 64},
        {"gpu_name": ""},
        {"gpu_name": 10},
        {"framework": "other"},
        {"orchestrator": "other"},
        {"engine": "sglang"},
        {"engine_version": "0.31.0"},
        {"storage_payload_scope": "tp_rank_files"},
        {"gpu_memory_bytes": 0},
        {"kv_budget_bytes_per_gpu": float("inf")},
        {"kv_budget_bytes_per_gpu": True},
        {"gpu_memory_bytes": 100, "kv_budget_bytes_per_gpu": 101},
        {"typo": 1},
    ],
)
def test_invalid_reference_is_rejected_before_io(overrides):
    with pytest.raises(ValueError, match="reference|TP|budget|gpu"):
        qwen(rollout_reference=reference(**overrides))


@pytest.mark.parametrize("attention,dtype", [("mla", "float16"), ("gqa", "int8")])
def test_unsupported_reference_geometry_does_not_invent_packing(attention, dtype):
    model = {**asdict(tiny())["model"], "attention_type": attention, "dtype": dtype, "kv_lora_rank": 512}
    with pytest.raises(ValueError, match="reference.*dense|reference.*dtype"):
        AgentRLConfig.from_dict({**asdict(tiny()), "model": model, "rollout_reference": reference()})


def test_legacy_bytes_and_checkpoint_fingerprint_remain_compatible():
    legacy = AgentRLConfig.from_dict({})
    assert legacy.fingerprint == "730cbd4d5532f15f6b4d44d76c8966c0e97e7565cf0b9b36cea0fea7c0b5c7de"
    assert legacy.bytes_per_token == 64
    assert legacy.reference_summary is None


def test_gpu_budget_resolves_whole_worker_blocks_without_mutating_input():
    raw = {**asdict(tiny()), "kv_offload_fraction": 0, "kv_cache_model": {"block_tokens": 2}}
    raw["rollout_reference"] = reference(2, kv_budget_bytes_per_gpu=769, gpu_memory_bytes=1024)
    config = AgentRLConfig.from_dict(raw)
    assert raw["kv_cache_model"] == {"block_tokens": 2}
    assert config.kv_cache_model == {"capacity_bytes": 1536, "block_tokens": 2}
    assert config.reference_summary["gpu_kv_budget"] == {
        "declared_bytes_per_gpu": 769,
        "pages_per_worker": 6,
        "tokens_per_worker": 12,
        "unused_bytes_per_gpu": 1,
        "replica_capacity_bytes": 1536,
    }
    # The GPU label does not manufacture a different token service rate.
    assert config.tokens_per_second == tiny().tokens_per_second
    assert config.prefill_tokens_per_second == tiny().prefill_tokens_per_second
    assert AgentRLConfig.from_dict(asdict(config)).fingerprint == config.fingerprint


@pytest.mark.parametrize("budget,capacity", [(127, None), (256, None), (768, 768)])
def test_impossible_or_conflicting_gpu_kv_capacity_is_rejected(budget, capacity):
    cache = {"block_tokens": 2}
    if capacity is not None:
        cache["capacity_bytes"] = capacity
    with pytest.raises(ValueError, match="block|working set|capacity"):
        AgentRLConfig.from_dict(
            {
                **asdict(tiny()),
                "kv_offload_fraction": 0,
                "kv_cache_model": cache,
                "rollout_reference": reference(2, kv_budget_bytes_per_gpu=budget),
            }
        )


def test_gpu_budget_requires_capacity_mode_and_declared_reference():
    with pytest.raises(ValueError, match="capacity|kv_cache_model"):
        qwen(rollout_reference=reference(kv_budget_bytes_per_gpu=1024))
    with pytest.raises(ValueError, match="capacity"):
        AgentRLConfig.from_dict({**asdict(tiny()), "kv_cache_model": {"block_tokens": 2}})


@pytest.mark.parametrize("tp,expected_kv_bytes", [(1, 1024), (2, 2048)])
def test_real_files_use_tp_aggregate_without_scaling_other_lifecycle_io(tmp_path, tp, expected_kv_bytes):
    config = AgentRLConfig.from_dict(
        {
            **asdict(tiny()),
            "iterations": 1,
            "requests_per_rank": 2,
            "response_tokens": [4],
            "prefix_reuse_probability": 0,
            "rollout_reference": reference(tp),
        }
    )
    writes = []

    class Observed(NVMeBackend):
        def write(self, key, data):
            result = super().write(key, data)
            if self.base_path.name == "kv":
                file = self.base_path / f"{key}.npy"
                assert np.load(file, allow_pickle=False).nbytes == data.nbytes
                assert file.stat().st_size > data.nbytes
                writes.append(data.nbytes)
            return result

    runner = create_lifecycle(config, tmp_path / "data", tmp_path / "results", backend_factory=Observed)
    summary = runner.run()
    assert sum(writes) == summary["io_totals"]["kv_write"]["payload_bytes"] == expected_kv_bytes
    assert summary["io_totals"]["prompt_read"]["payload_bytes"] == 32
    assert summary["io_totals"]["trajectory_write"]["payload_bytes"] == 192
    assert summary["io_totals"]["checkpoint_write"]["payload_bytes"] == 128
    trace = load_trace(runner.result_dir)
    assert trace["rollout_reference"] == summary["rollout_reference"] == config.reference_summary
    assert not analyze_trace(trace)["violations"]
    assert summary["fidelity"] == "uncalibrated"


@pytest.mark.parametrize("mode", ["sync", "colocate_async", "separate_async"])
def test_reference_capacity_drives_real_offload_and_tool_resume_reads(tmp_path, mode):
    raw = asdict(configured(mode, iterations=1))
    raw["kv_cache_model"].pop("capacity_bytes")
    raw["rollout_reference"] = reference(2, kv_budget_bytes_per_gpu=768)
    config = AgentRLConfig.from_dict(raw)
    summary, trace, _ = execute(tmp_path, config)
    transfers = [e for e in trace["events"] if e["event"] == "io_end" and e["kind"] == "kv"]
    assert {e["op"] for e in transfers} >= {"write", "read"}
    assert all(e["bytes"] <= 256 for e in transfers)
    assert all(e["bytes"] % 128 == 0 for e in transfers)
    pools = summary["ranks"][0]["kv_cache_model"]["owners"]
    assert all(p["capacity_bytes"] == 1536 and p["resident_peak_bytes"] <= 1536 for p in pools.values())
    assert summary["rollout_reference"] == trace["rollout_reference"]
    assert not analyze_trace(trace)["violations"]


@pytest.mark.parametrize("mode,ranks", [("sync", 2), ("separate_async", 3)])
def test_native_mpi_keeps_rollout_owner_distinct_from_tp_worker(tmp_path, mode, ranks):
    raw = asdict(configured(mode, iterations=1))
    raw["kv_cache_model"].pop("capacity_bytes")
    raw["rollout_reference"] = reference(2, kv_budget_bytes_per_gpu=768)
    if mode != "sync":
        raw["async_workload"].update(execution="mpi_shared", rollout_owners=2)
    process = invoke(tmp_path, raw, ranks=ranks)
    assert process.returncode == 0, process.stderr
    path = next((tmp_path / "results").glob("*/summary.json"))
    summary = json.loads(path.read_text())
    assert summary["world_size"] == ranks
    assert summary["rollout_reference"]["tensor_parallel_size"] == 2
    for shard in path.parent.glob("trace-rank-*.json"):
        assert json.loads(shard.read_text())["rollout_reference"] == summary["rollout_reference"]


def test_explicit_head_dimension_allows_independent_hidden_projection_width():
    model = {**asdict(tiny())["model"], "hidden_dim": 17, "_kv_dim_override": 8}
    config = AgentRLConfig.from_dict({**asdict(tiny()), "model": model, "rollout_reference": reference(2)})
    assert config.bytes_per_token == 128


def test_kv_heads_and_tp_require_exact_sharding_even_when_query_heads_divide():
    model = {**asdict(tiny())["model"], "hidden_dim": 96, "num_heads": 12, "kv_heads": 6}
    with pytest.raises(ValueError, match="reference TP"):
        AgentRLConfig.from_dict({**asdict(tiny()), "model": model, "rollout_reference": reference(4)})
