"""Observable contracts for the GPU-less Agent RL extension."""

import json
from dataclasses import replace

import numpy as np
import pytest

from kv_cache.agentrl import AgentRLConfig, SyncLifecycle, kv_delta_bytes, network_delay
from kv_cache.backends import NVMeBackend


def tiny(**overrides):
    return AgentRLConfig.from_dict(
        {
            "iterations": 2,
            "requests_per_rank": 3,
            "concurrency": 2,
            "prompt_tokens": 4,
            "response_tokens": [4, 8],
            "chunk_tokens": 2,
            "tokens_per_second": 400,
            "prefill_tokens_per_second": 4000,
            "kv_offload_fraction": 1.0,
            "prefix_reuse_probability": 1.0,
            "hot_prefixes": 1,
            "train_delay_s": 0.001,
            "checkpoint_bytes_per_rank": 128,
            "model": {
                "name": "smoke",
                "num_layers": 2,
                "hidden_dim": 16,
                "num_heads": 2,
                "kv_heads": 1,
                "dtype": "float16",
            },
            **overrides,
        }
    )


def run(tmp_path, config=None, **kwargs):
    runner = SyncLifecycle(config or tiny(), tmp_path / "data", tmp_path / "results", **kwargs)
    return runner, runner.run()


@pytest.mark.parametrize(
    "field,value",
    [
        ("tokens_per_second", 0),
        ("concurrency", 0),
        ("iterations", -1),
        ("tokens_per_second", float("nan")),
        ("train_delay_s", float("inf")),
        ("kv_offload_fraction", 1.1),
        ("turns", True),
        ("response_tokens", []),
        ("rank_rate_factors", [0]),
        ("hot_prefixes", 0),
        ("nic_gbps", 0),
        ("network_mode", "magic"),
        ("trainer_mode", "colocate_async"),
        ("checkpoint_bytes_per_rank", 1.5),
        ("requests_per_rank", "3"),
    ],
)
def test_invalid_config(field, value):
    with pytest.raises(ValueError):
        tiny(**{field: value})


def test_unknown_field_and_model_shape():
    with pytest.raises(ValueError, match="unknown"):
        tiny(typo_rate=1)
    with pytest.raises(ValueError):
        tiny(model={"name": "bad", "num_layers": 1, "hidden_dim": 15, "num_heads": 2, "kv_heads": 1})


@pytest.mark.parametrize("fraction", [0, 0.13, 0.5, 1])
def test_chunk_byte_conservation(fraction):
    total = sum(kv_delta_bytes(i, min(3, 13 - i), 128, fraction) for i in range(0, 13, 3))
    assert total == int(13 * 128 * fraction)


def test_network_decimal_units_and_sharing():
    config = tiny(network_mode="estimate", nic_gbps=10, nic_efficiency=0.8, nic_rtt_us=100)
    assert network_delay(config, 1_000_000_000) == pytest.approx(1.0001)
    assert network_delay(replace(config, nic_sharers=4), 1_000_000_000) == pytest.approx(4.0001)
    assert network_delay(replace(config, network_mode="none"), 1_000_000_000) == 0
    assert network_delay(config, 0) == pytest.approx(0.0001)


def test_backend_reopen_preserves_checkpoint_and_default_resets(tmp_path):
    backend = NVMeBackend(str(tmp_path))
    backend.write("shard", np.arange(32, dtype=np.uint8))
    reopened = NVMeBackend(str(tmp_path), preserve_existing=True)
    assert np.array_equal(reopened.read("shard")[0], np.arange(32, dtype=np.uint8))
    NVMeBackend(str(tmp_path))
    assert not (tmp_path / "shard.npy").exists()


def test_lifecycle_order_prefix_and_policy_boundary(tmp_path):
    runner, summary = run(tmp_path)
    events = runner.trace.events
    assert summary["status"] == "complete"
    assert summary["final_policy_version"] == 2
    for step in range(2):
        selected = [e for e in events if e["iteration"] == step]
        names = [e["event"] for e in selected]
        last_rollout = max(i for i, e in enumerate(selected) if e["event"] == "rollout_complete")
        assert last_rollout < names.index("train_begin") < names.index("checkpoint_commit")
        assert names.index("checkpoint_commit") < names.index("policy_install") < names.index("kv_invalidate")
        assert sum(e["event"] == "prefix_miss" for e in selected) == 1
        assert sum(e["event"] == "prefix_hit" for e in selected) == 2
    assert summary["ranks"][0]["active_peak"] <= 2
    assert not list((runner.storage_dir / "rank-0" / "kv").glob("*.npy"))
    assert (runner.result_dir / "summary.json").exists()


def test_zero_offload_has_no_storage_kv(tmp_path):
    runner, _ = run(tmp_path, tiny(kv_offload_fraction=0))
    assert not [e for e in runner.trace.events if e["event"] == "io_end" and e["kind"] == "kv"]


def test_generation_slowdown_delays_checkpoint(tmp_path):
    fast, _ = run(tmp_path / "fast", tiny(iterations=1, tokens_per_second=400))
    slow, _ = run(tmp_path / "slow", tiny(iterations=1, tokens_per_second=40))

    def time_of(r):
        return next(e["t_s"] for e in r.trace.events if e["event"] == "checkpoint_begin")

    assert time_of(slow) > time_of(fast) + 0.12


def test_tool_pause_and_byte_conservation(tmp_path):
    runner, _ = run(
        tmp_path,
        tiny(iterations=1, requests_per_rank=1, turns=2, response_tokens=[5], observation_tokens=3, tool_delay_s=0.02),
    )
    events = runner.trace.events
    start = next(e for e in events if e["event"] == "tool_begin")
    end = next(e for e in events if e["event"] == "tool_end")
    assert end["t_s"] - start["t_s"] >= 0.019
    assert not [e for e in events if e["event"] == "generation_end" and start["t_s"] < e["t_s"] < end["t_s"]]
    writes = [e["bytes"] for e in events if e["event"] == "io_end" and e["kind"] == "kv" and e["op"] == "write"]
    assert sum(writes) == (4 + 5 + 3) * runner.config.bytes_per_token


def test_checkpoint_resume_and_corruption(tmp_path):
    runner, _ = run(tmp_path)
    manifest = runner.storage_dir / "checkpoints" / "step-1" / "manifest.json"
    resumed, summary = run(tmp_path / "resumed", resume=manifest)
    assert summary["final_policy_version"] == 2
    assert min(e["iteration"] for e in resumed.trace.events if e["event"] == "rollout_start") == 1
    data = json.loads(manifest.read_text())
    shard = manifest.parent / data["shards"][0]["path"]
    shard.write_bytes(b"broken")
    with pytest.raises(RuntimeError, match="checkpoint|checksum"):
        run(tmp_path / "bad", resume=manifest)


def test_resume_rejects_config_mismatch_and_missing_commit(tmp_path):
    runner, _ = run(tmp_path)
    manifest = runner.storage_dir / "checkpoints" / "step-1" / "manifest.json"
    with pytest.raises(RuntimeError, match="config"):
        run(tmp_path / "bad-config", tiny(seed=99), resume=manifest)
    with pytest.raises(RuntimeError, match="checkpoint"):
        run(tmp_path / "missing", resume=manifest.parent / "missing.json")


def test_slow_io_backpressures_generation(tmp_path):
    import time

    class SlowKV(NVMeBackend):
        def write(self, key, data):
            if self.base_path.name == "kv":
                time.sleep(0.015)
            return super().write(key, data)

    config = tiny(iterations=1, requests_per_rank=1, response_tokens=[8], concurrency=1)
    fast, _ = run(tmp_path / "fast", config)
    slow, _ = run(tmp_path / "slow", config, backend_factory=SlowKV)

    def checkpoint(r):
        return next(e["t_s"] for e in r.trace.events if e["event"] == "checkpoint_begin")

    assert checkpoint(slow) > checkpoint(fast) + 0.055
    events = slow.trace.events
    generations = [e for e in events if e["event"] == "generation_begin"]
    assert all(b["t_s"] - a["t_s"] > 0.018 for a, b in zip(generations[:-1], generations[1:], strict=True))


def test_compute_overlap_and_queue_admission(tmp_path):
    runner, _ = run(tmp_path, tiny(iterations=1, tokens_per_second=100))
    events = runner.trace.events
    first_complete = next(e["t_s"] for e in events if e["event"] == "rollout_complete")
    assert sum(e["event"] == "rollout_start" and e["t_s"] < first_complete for e in events) == 2
    assert max(e["active"] for e in events if e["event"] == "rollout_start") == 2
    assert next(e["queue_wait_s"] for e in events if e["event"] == "rollout_start" and e["request"] == 2) > 0.035


def test_io_failure_and_incomplete_checkpoint_never_commit(tmp_path):
    class FailingCheckpoint(NVMeBackend):
        def write(self, key, data):
            if key == "state":
                raise OSError("injected shard failure")
            return super().write(key, data)

    with pytest.raises(RuntimeError, match="checkpoint shard"):
        run(tmp_path, backend_factory=FailingCheckpoint)
    assert not list((tmp_path / "data").glob("*/checkpoints/*/manifest.json"))
    assert not list((tmp_path / "results").glob("*/summary.json"))


def test_checksum_mismatch_on_valid_numpy_file(tmp_path):
    runner, _ = run(tmp_path)
    manifest = runner.storage_dir / "checkpoints" / "step-1" / "manifest.json"
    shard = manifest.parent / "rank-0" / "state.npy"
    np.save(shard, np.zeros(128, dtype=np.uint8), allow_pickle=False)
    with pytest.raises(RuntimeError, match="checksum"):
        run(tmp_path / "tampered", resume=manifest)


@pytest.mark.parametrize("seed", [1, 17, 99])
@pytest.mark.parametrize("fraction,turns", [(0, 1), (0.13, 3), (1, 2)])
def test_small_profile_matrix(tmp_path, seed, fraction, turns):
    runner, summary = run(
        tmp_path,
        tiny(
            seed=seed,
            iterations=1,
            requests_per_rank=1,
            prefix_reuse_probability=0,
            kv_offload_fraction=fraction,
            turns=turns,
            observation_tokens=2,
            response_tokens=[7],
        ),
    )
    events = runner.trace.events
    kv_bytes = sum(e["bytes"] for e in events if e["event"] == "io_end" and e["kind"] == "kv" and e["op"] == "write")
    assert kv_bytes == int((4 + 7 + (turns - 1) * 2) * runner.config.bytes_per_token * fraction)
    assert sum(e.get("generated_tokens", 0) for e in events) == 7
    assert summary["final_policy_version"] == 1


def test_payload_budget_dtype_and_mla_validation():
    with pytest.raises(ValueError, match="payload"):
        tiny(max_payload_bytes=1)
    with pytest.raises(ValueError, match="dtype"):
        tiny(model={**tiny().model, "dtype": "mistyped"})
    config = tiny(model={**tiny().model, "attention_type": "mla", "kv_lora_rank": 8, "qk_rope_head_dim": 4})
    assert config.bytes_per_token == 2 * (8 + 4) * 2


def test_finished_checkpoint_resume_runs_no_new_trajectories(tmp_path):
    runner, _ = run(tmp_path)
    manifest = runner.storage_dir / "checkpoints" / "step-2" / "manifest.json"
    resumed, summary = run(tmp_path / "finished", resume=manifest)
    assert summary["final_policy_version"] == 2
    assert not [e for e in resumed.trace.events if e["event"] == "rollout_start"]


def test_aggregate_metrics_and_opt_out_trajectory_persistence(tmp_path):
    runner, summary = run(tmp_path)
    assert summary["io_totals"]["trajectory_read"]["ops"] == 6
    assert summary["io_totals"]["checkpoint_write"]["payload_bytes"] == 256
    assert "kv_write" in summary["latency_by_kind_op_s"]
    baseline, summary = run(tmp_path / "memory-queue", tiny(persist_trajectories=False))
    assert not [e for e in baseline.trace.events if e.get("kind") == "trajectory"]
    assert "trajectory_write" not in summary["io_totals"]


def test_network_slowdown_delays_policy_install(tmp_path):
    config = tiny(
        iterations=1,
        requests_per_rank=1,
        response_tokens=[2],
        network_mode="estimate",
        nic_gbps=0.001,
        nic_rtt_us=0,
        weight_bytes=1000,
    )
    fast, _ = run(tmp_path / "fast", config)
    slow, _ = run(tmp_path / "slow", replace(config, nic_gbps=0.0001))

    def install(r):
        return next(e["t_s"] for e in r.trace.events if e["event"] == "policy_install")

    assert install(slow) > install(fast) + 0.12


def test_extreme_finite_rates_rejected_before_division():
    with pytest.raises(ValueError, match="effective"):
        tiny(tokens_per_second=1e-300, rank_rate_factors=[1e-300])
