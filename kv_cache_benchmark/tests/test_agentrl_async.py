"""Source-grounded async control flow with real files, without GPU claims."""

import json
import time
from dataclasses import replace

import pytest
from test_agentrl import tiny
from test_agentrl_trace import envelope, profile_record

from kv_cache.agentrl_async import AsyncLifecycle, age_blocks
from kv_cache.agentrl_trace import analyze_trace, load_trace, make_profile
from kv_cache.backends import NVMeBackend


def async_config(mode="colocate_async", **settings):
    return tiny(
        trainer_mode=mode,
        generation_rate_scope="owner",
        iterations=6,
        requests_per_rank=4,
        concurrency=4,
        response_tokens=[2, 2, 12, 12],
        tokens_per_second=400,
        train_delay_s=0.003,
        async_workload={
            "group_size": 2,
            "batch_groups": 1,
            "outstanding_groups": 3,
            "queue_capacity": 2,
            "rollout_owners": 1,
            **settings,
        },
    )


def execute(tmp_path, config=None, **kwargs):
    runner = AsyncLifecycle(config or async_config(), tmp_path / "data", tmp_path / "results", **kwargs)
    return runner, runner.run()


def test_colocate_samples_whole_groups_and_preserves_partial_budget(tmp_path):
    runner, summary = execute(tmp_path)
    assert summary["trainer_mode"] == "colocate_async" and summary["fidelity"] == "uncalibrated"
    assert summary["final_policy_version"] == 6
    events = runner.trace.events
    samples = [e for e in events if e["event"] == "group_sample"]
    assert len(samples) == 6
    for sample in samples:
        completed = [e for e in events if e["event"] == "rollout_complete" and e["group"] == sample["group"]]
        assert len(completed) == 2 and max(e["t_s"] for e in completed) < sample["t_s"]
    assert any(e["generated_tokens"] > 0 for e in events if e["event"] == "rollout_abort")
    assert any(e["event"] == "re_prefill_end" and e["history_tokens"] > runner.config.prompt_tokens for e in events)
    records = load_trace(runner.result_dir)["requests"]
    assert any(r["policy_end"] > r["policy_start"] for r in records)
    assert all(sum(t["generated_tokens"] for t in r["turns"]) in (2, 12) for r in records)
    assert not analyze_trace(load_trace(runner.result_dir))["violations"]
    assert runner.active == runner.io_active == 0


def test_colocate_decode_stops_during_train_but_physical_gc_is_deferred(tmp_path):
    runner, summary = execute(tmp_path, async_config(kv_gc_delay_s=0.03))
    events = runner.trace.events
    begins = [e for e in events if e["event"] == "train_begin"]
    ends = [e for e in events if e["event"] == "train_end"]
    for begin, end in zip(begins, ends, strict=True):
        assert not any(e["event"] == "generation_begin" and begin["t_s"] < e["t_s"] < end["t_s"] for e in events)
    retirements = [e for e in events if e["event"] == "kv_logical_invalidate"]
    assert retirements
    for retired in retirements:
        keys = set(retired["keys"])
        assert not any(
            e["event"] == "io_begin" and e["op"] == "read" and e["key"] in keys and e["t_s"] > retired["t_s"]
            for e in events
        )
        gc = next(e for e in events if e["event"] == "kv_gc_begin" and e["retirement_id"] == retired["retirement_id"])
        assert gc["t_s"] - retired["t_s"] >= 0.029
    assert summary["ranks"][0]["kv_live_payload_bytes"] == 0
    assert not list(runner.storage_dir.glob("**/kv/*.npy"))


@pytest.mark.parametrize(
    "settings",
    [
        {"group_size": 0},
        {"queue_capacity": 0},
        {"batch_groups": 4},
        {"max_prompt_age": 0},
        {"staleness_strategy": "invalid"},
        {"kv_gc_delay_s": -1},
    ],
)
def test_async_rejects_invalid_bounds(settings):
    with pytest.raises(ValueError):
        async_config(**settings)


def test_async_rejects_inflight_resume_and_multi_rank_before_io(tmp_path):
    class TwoRanks:
        def Get_rank(self):  # noqa: N802 - MPI interface
            return 0

        def Get_size(self):  # noqa: N802 - MPI interface
            return 2

    with pytest.raises(ValueError, match="multi-rank"):
        execute(tmp_path, comm=TwoRanks())
    with pytest.raises(ValueError, match="resume"):
        execute(tmp_path, resume=tmp_path / "manifest.json")
    assert not (tmp_path / "data").exists()


def test_separate_overlaps_real_io_and_syncs_after_mini_batch_cycle(tmp_path):
    config = async_config("separate_async", rollout_owners=2, parameter_sync_step=2)
    config = replace(config, iterations=2, train_delay_s=0.06)
    runner, summary = execute(tmp_path, config)
    events = runner.trace.events
    begins = [e for e in events if e["event"] == "train_begin"]
    ends = [e for e in events if e["event"] == "train_end"]
    assert len(begins) == len(ends) == 4
    assert any(
        e["event"] == "generation_end" and begin["t_s"] < e["t_s"] < end["t_s"]
        for begin, end in zip(begins, ends, strict=True)
        for e in events
    )
    installs = [e for e in events if e["event"] == "policy_install" and not e["initial"]]
    assert len(installs) == 2
    assert summary["async"]["terminal_queue_peak"] <= 2
    assert summary["async"]["outstanding_groups_peak"] <= 3
    assert {e["owner"] for e in events if e["event"] == "io_begin" and e["kind"] == "kv"} == {"rollout-0", "rollout-1"}
    assert all(
        e["role"] == "trainer"
        for e in events
        if e["event"] == "io_begin" and e["kind"] == "trajectory" and e["op"] == "read"
    )
    assert not analyze_trace(load_trace(runner.result_dir))["violations"]


@pytest.mark.parametrize(
    "strategy,terminal,age,expected",
    [
        ("drop", True, 2, False),
        ("drop", True, 3, True),
        ("drop", False, 3, False),
        ("wait", False, 1, False),
        ("wait", False, 2, True),
        ("wait", True, 3, False),
    ],
)
def test_prompt_age_matches_verl_inequalities(strategy, terminal, age, expected):
    assert age_blocks(age, 2, strategy, terminal=terminal) is expected
    assert age_blocks(age, None, strategy, terminal=terminal) is False


def test_wait_and_small_full_queue_do_not_deadlock(tmp_path):
    config = async_config(
        "separate_async", queue_capacity=1, max_prompt_age=1, staleness_strategy="wait", parameter_sync_step=2
    )
    runner, summary = execute(tmp_path, config)
    assert summary["final_policy_version"] == 6
    assert not [e for e in runner.trace.events if e["event"] == "group_drop"]


def test_shutdown_during_tool_or_trajectory_io_keeps_trace_and_files_consistent(tmp_path):
    config = replace(async_config("separate_async"), iterations=1, turns=2, tool_delay_s=0.04, observation_tokens=2)
    runner, _ = execute(tmp_path, config)
    assert not analyze_trace(load_trace(runner.result_dir))["violations"]
    assert runner.active == runner.io_active == 0
    assert not list(runner.storage_dir.glob("**/trajectories/*.npy"))


@pytest.mark.parametrize("target", ["kv", "state", "gc"])
def test_async_failure_drains_workers_and_never_publishes_success(tmp_path, target):
    class BrokenStorage(NVMeBackend):
        def write(self, key, data):
            if (target == "kv" and self.base_path.name == "kv") or (target == "state" and key == "state"):
                raise OSError("injected async storage failure")
            return super().write(key, data)

        def delete(self, key):
            if target == "gc" and self.base_path.name == "kv":
                raise OSError("injected async GC failure")
            return super().delete(key)

    runner = AsyncLifecycle(async_config(), tmp_path / "data", tmp_path / "results", backend_factory=BrokenStorage)
    with pytest.raises(RuntimeError, match="async workload"):
        runner.run()
    assert runner.active == runner.io_active == 0
    assert not list((tmp_path / "results").glob("*/summary.json"))


def test_siblings_share_prompt_and_owner_but_keep_individual_response_budgets(tmp_path):
    runner, _ = execute(tmp_path, replace(async_config(), prefix_reuse_probability=0))
    records = load_trace(runner.result_dir)["requests"]
    accepted = [r for r in records if r["disposition"] == "accepted"]
    for group in {r["group"] for r in accepted}:
        siblings = [r for r in accepted if r["group"] == group]
        assert len({r["prefix_id"] for r in siblings}) == 1
        assert len({r["owner"] for r in siblings}) == 1
        reads = [
            e
            for e in runner.trace.events
            if e["event"] == "io_begin" and e["kind"] == "prompt" and e["request"] in {r["request"] for r in siblings}
        ]
        assert len({e["key"] for e in reads}) == 1


def test_separate_drops_only_whole_terminal_groups_and_refills_once(tmp_path):
    config = replace(async_config("separate_async", max_prompt_age=1), train_delay_s=0.05)
    runner, summary = execute(tmp_path, config)
    dropped = [e for e in runner.trace.events if e["event"] == "group_drop"]
    assert dropped and summary["async"]["dropped_groups"] == len(dropped)
    records = load_trace(runner.result_dir)["requests"]
    for event in dropped:
        siblings = [r for r in records if r["group"] == event["group"]]
        assert len(siblings) == 2 and all(r["disposition"] == "dropped" for r in siblings)
        assert max(r["end_s"] for r in siblings) < event["t_s"]
        assert event["prompt_age"] > 1
    assert all(
        r["trainer_policy_at_accept"] - r["prompt_policy"] + 1 <= 1 for r in records if r["disposition"] == "accepted"
    )
    dispatch = [e["group"] for e in runner.trace.events if e["event"] == "group_dispatch"]
    assert len(dispatch) == len(set(dispatch))
    assert not list(runner.storage_dir.glob("**/trajectories/*.npy"))


def test_separate_rollout_io_continues_during_real_checkpoint(tmp_path):
    class SlowCheckpoint(NVMeBackend):
        def write(self, key, data):
            if key == "state":
                time.sleep(0.06)
            return super().write(key, data)

    config = replace(async_config("separate_async"), iterations=2, response_tokens=[2, 2, 40, 40])
    runner, _ = execute(tmp_path, config, backend_factory=SlowCheckpoint)
    events = runner.trace.events
    intervals = [
        (b["t_s"], e["t_s"])
        for b, e in zip(
            [x for x in events if x["event"] == "checkpoint_begin"],
            [x for x in events if x["event"] == "checkpoint_commit"],
            strict=True,
        )
    ]
    assert any(
        b < e["t_s"] < end
        for b, end in intervals
        for e in events
        if e["event"] == "io_end" and e["kind"] == "kv" and e["op"] == "write"
    )
    manifest = json.loads(next(runner.storage_dir.glob("checkpoints/step-1/manifest.json")).read_text())
    assert manifest["inflight_recoverable"] is False
    assert manifest["lifecycle_mode"] == "separate_async"


@pytest.mark.parametrize(
    "records",
    [
        [profile_record()],
        [profile_record(), profile_record(prompt_tokens=15)],
        [profile_record(), profile_record(prefix_id="different")],
    ],
)
def test_async_profiles_require_complete_groups_with_shared_prompt_shape(records):
    profile = make_profile(envelope(records, mode="colocate_async"))
    with pytest.raises(ValueError, match="group"):
        replace(async_config(), request_profile=profile).validate()


@pytest.mark.parametrize("mode", ["colocate_async", "separate_async"])
@pytest.mark.parametrize("fraction", [0, 0.13, 1])
def test_joint_profile_and_offload_work_in_both_local_async_modes(tmp_path, mode, fraction):
    profile = make_profile(envelope([profile_record(), profile_record()], mode=mode))
    config = replace(
        async_config(mode, rollout_owners=2),
        iterations=2,
        request_profile=profile,
        kv_offload_fraction=fraction,
        turns=2,
    )
    runner, summary = execute(tmp_path, config)
    assert not analyze_trace(load_trace(runner.result_dir))["violations"]
    records = load_trace(runner.result_dir)["requests"]
    assert all(r["prompt_tokens"] == 7 and r["trajectory_bytes"] == 257 for r in records)
    assert all(sum(t["generated_tokens"] for t in r["turns"]) == 8 for r in records)
    assert summary["ranks"][0]["kv_live_payload_bytes"] == 0
    kv = [e for e in runner.trace.events if e["event"] == "io_begin" and e["kind"] == "kv"]
    assert bool(kv) is (fraction > 0)


def test_async_metrics_count_partial_decode_and_distinguish_catalog_from_admission(tmp_path):
    config = replace(async_config(), iterations=1, concurrency=6, response_tokens=[2, 2, 40, 40])
    runner, summary = execute(tmp_path, config)
    rank = summary["ranks"][0]
    assert rank["admission_worker_limit"] == 6  # requests_per_rank=4 is a shape catalog here.
    decoded = sum(e["tokens"] for e in runner.trace.events if e["event"] == "generation_end")
    assert rank["decoded_tokens"] == decoded
    assert rank["unfinished_generated_tokens"] == decoded - rank["generated_tokens"] > 0
    phase = next(e["t_s"] for e in runner.trace.events if e["event"] == "rollout_phase_begin")
    end = next(e["t_s"] for e in runner.trace.events if e["event"] == "rollout_phase_end")
    assert rank["achieved_decode_tokens_per_s"] == pytest.approx(decoded / (end - phase))
