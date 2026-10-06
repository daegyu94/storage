"""Group-aware calibration preserves routing without replaying completion times."""

import copy
import json
import time
from dataclasses import replace

import pytest
from test_agentrl_async import async_config, execute
from test_agentrl_mpi_async import run_mpi
from test_agentrl_trace import envelope, profile_record
from test_agentrl_trace_cli import invoke

from kv_cache.agentrl_trace import analyze_trace, compare_traces, load_trace, make_profile, validate_profile
from kv_cache.backends import NVMeBackend


def grouped_reference(mode="separate_async"):
    trace = envelope(mode=mode)
    trace["requests"] = []
    trace["events"] = []
    for group, owner, rank, first, prompt in ((9, 1, 8, 190, 7), (4, 0, 7, 80, 11)):
        trace["events"].append(
            {
                "event": "group_dispatch",
                "rank": 0,
                "iteration": 0,
                "t_s": len(trace["events"]),
                "group": group,
                "owner": f"rollout-{owner}",
                "members": [first, first + 1],
            }
        )
        for sibling in range(2):
            record = envelope([profile_record(rank=rank, prompt_tokens=prompt)])["requests"][0]
            record.update(
                request=first + sibling,
                group=group,
                owner=f"rollout-{owner}",
                start_s=rank * 1000,
                end_s=rank * 1000 + 0.5 + sibling * 0.25,
            )
            record["turns"][-1]["generated_tokens"] += sibling
            trace["requests"].append(record)
    return trace


def group_profile(mode="separate_async"):
    return make_profile(grouped_reference(mode), grouped=True)


def test_group_profile_uses_dispatch_order_not_rank_completion_or_request_sort():
    trace = grouped_reference()
    trace["requests"].reverse()
    profile = make_profile(trace, grouped=True)
    assert profile["schema_version"] == 2
    assert [r["group_sequence"] for r in profile["records"]] == [0, 0, 1, 1]
    assert [r["owner"] for r in profile["records"]] == [1, 1, 0, 0]
    assert [r["rank"] for r in profile["records"]] == [8, 8, 7, 7]
    assert [r["sibling"] for r in profile["records"]] == [0, 1, 0, 1]
    assert profile["provenance"]["selection"]["included_groups"] == 2
    for request in trace["requests"]:
        request["start_s"] += 9000 * request["rank"]
        request["end_s"] += 9000 * request["rank"]
    assert make_profile(trace, grouped=True)["records"] == profile["records"]


def test_group_selection_censors_whole_group_and_reports_partial_progress():
    trace = grouped_reference()
    trace["requests"].pop(1)
    trace["events"].append(
        {
            "event": "request_interrupted",
            "rank": 8,
            "iteration": 0,
            "t_s": 9000,
            "request": 191,
            "group": 9,
            "owner": "rollout-1",
            "generated_tokens": 2,
            "planned_generated_tokens": 9,
            "history_tokens": 9,
            "reason": "cancelled",
        }
    )
    profile = make_profile(trace, grouped=True)
    assert len(profile["records"]) == 2
    selection = profile["provenance"]["selection"]
    assert selection["dispatched_groups"] == 2 and selection["included_groups"] == 1
    assert selection["excluded_groups"] == [{"group_id": "group-9", "reason": "incomplete_siblings"}]
    assert selection["interrupted_requests"] == 1 and selection["interrupted_generated_tokens"] == 2
    assert all(r["owner"] == 0 for r in profile["records"])


@pytest.mark.parametrize("fault", ["members", "owner", "duplicate", "missing_dispatch", "sync"])
def test_group_extraction_rejects_ambiguous_source_evidence(fault):
    trace = grouped_reference()
    if fault == "members":
        trace["events"][0].pop("members")
    elif fault == "owner":
        trace["requests"][0]["owner"] = "rollout-0"
    elif fault == "duplicate":
        trace["events"].append(copy.deepcopy(trace["events"][0]))
    elif fault == "missing_dispatch":
        trace["events"] = []
    else:
        trace["mode"] = "sync"
    with pytest.raises(ValueError):
        make_profile(trace, grouped=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("owner", True),
        ("owner", -1),
        ("sibling", 2),
        ("group_sequence", 9),
        ("group_id", "../raw-prompt"),
        ("prompt_tokens", 12),
        ("rank", 99),
    ],
)
def test_v2_rejects_invalid_incomplete_or_cross_owner_geometry(field, value):
    profile = group_profile()
    profile["records"][0][field] = value
    with pytest.raises(ValueError):
        validate_profile(profile)


@pytest.mark.parametrize("mode", ["colocate_async", "separate_async"])
def test_local_replay_routes_joint_groups_and_source_rank_explicitly(tmp_path, mode):
    profile = group_profile(mode)
    config = replace(async_config(mode, rollout_owners=2), request_profile=profile, requests_per_rank=2)
    config.validate()
    runner, summary = execute(tmp_path, config)
    dispatch = [e for e in runner.trace.events if e["event"] == "group_dispatch"]
    assert [e["owner"] for e in dispatch[:4]] == ["rollout-1", "rollout-0"] * 2
    assert all(len(e["members"]) == 2 for e in dispatch)
    for record in runner.completed_requests:
        sequence = record["group"] % 2
        assert record["rank"] == 0
        assert record["calibration_source_rank"] == (8 if sequence == 0 else 7)
        assert record["prompt_tokens"] == (7 if sequence == 0 else 11)
        assert record["owner"] == f"rollout-{1 - sequence}"
        assert sum(t["generated_tokens"] for t in record["turns"]) == 8 + record["sibling"]
    assert summary["async"]["profile_mapping"] == "group_catalog_owner"
    assert runner.config.request_profile["records"] == profile["records"]
    replay = make_profile(load_trace(runner.result_dir), grouped=True)
    assert replay["provenance"]["kind"] == "poc"


def test_owner_count_and_fanout_must_match_before_any_io(tmp_path):
    profile = group_profile()
    for settings in ({"rollout_owners": 1}, {"rollout_owners": 2, "group_size": 1}):
        config = replace(async_config("separate_async", **settings), request_profile=profile)
        with pytest.raises(ValueError):
            config.validate()
    assert not list(tmp_path.iterdir())


def test_mpi_replay_preserves_physical_rank_and_can_be_calibrated_again(tmp_path):
    profile = group_profile()
    config = replace(
        async_config("separate_async", rollout_owners=2, execution="mpi_shared"),
        request_profile=profile,
        iterations=3,
        requests_per_rank=2,
    )
    summary, trace, _ = run_mpi(tmp_path, config)
    assert summary["topology"]["profile_mapping"] == "group_catalog_owner"
    assert trace["requests"]
    assert all(r["rank"] == int(r["owner"].split("-")[1]) + 1 for r in trace["requests"])
    second = make_profile(trace, grouped=True)
    config = replace(config, request_profile=second)
    config.validate()
    _, next_trace, _ = run_mpi(tmp_path / "again", config)
    assert next_trace["requests"]


def test_interrupted_request_keeps_budget_separate_from_observed_progress(tmp_path):
    config = replace(async_config(), response_tokens=[2, 2, 80, 80], iterations=1)
    runner, _ = execute(tmp_path, config)
    interrupted = [e for e in runner.trace.events if e["event"] == "request_interrupted"]
    assert interrupted
    completed = {r["request"] for r in runner.completed_requests}
    assert all(e["request"] not in completed for e in interrupted)
    assert any(0 < e["generated_tokens"] < e["planned_generated_tokens"] for e in interrupted)
    assert all(e["planned_generated_tokens"] == config.response_tokens[e["request"] % 4] for e in interrupted)
    assert runner.active == runner.io_active == 0


def test_group_and_owner_metrics_keep_same_rank_clock_and_detect_routing_drift():
    trace = grouped_reference()
    metrics = analyze_trace(trace)
    assert metrics["group_fanout"] == [2, 2]
    assert metrics["group_straggler_s"] == pytest.approx([0.25, 0.25])
    assert metrics["completed_requests_by_owner"] == {"rollout-1": 2, "rollout-0": 2}


def test_same_bytes_with_different_owner_locality_is_not_agreement():
    reference, candidate = grouped_reference(), grouped_reference()
    candidate["provenance"]["run_id"] = "independent"
    candidate["events"][0]["owner"] = "rollout-0"
    for request in candidate["requests"]:
        request["owner"] = "rollout-0"
    report = compare_traces(reference, candidate)
    assert report["status"] == "outside_tolerances"
    assert report["metrics"]["completed_requests_by_owner"]["total_variation_distance"] == 0.5
    assert report["real_validation"] is False


def test_grouped_cli_and_holdout_identity_guard(tmp_path):
    trace = grouped_reference()
    source, output = tmp_path / "source.json", tmp_path / "profile.json"
    source.write_text(json.dumps(trace))
    process = invoke("profile", "--grouped", "--input", source, "--output", output)
    assert process.returncode == 0, process.stderr
    profile = json.loads(output.read_text())
    assert profile["schema_version"] == 2
    candidate = copy.deepcopy(trace)
    candidate["provenance"].update(run_id="candidate", calibration_sha256=profile["provenance"]["source_sha256"])
    assert compare_traces(trace, candidate)["status"] == "inconclusive_reused_reference"


def test_legacy_mpi_catalog_records_use_real_worker_rank(tmp_path):
    profile = make_profile(envelope([profile_record()] * 4, mode="separate_async"))
    config = replace(
        async_config("separate_async", rollout_owners=2, execution="mpi_shared"), request_profile=profile, iterations=1
    )
    _, trace, _ = run_mpi(tmp_path, config)
    assert all(r["rank"] in (1, 2) and r["rank"] == int(r["owner"].split("-")[1]) + 1 for r in trace["requests"])


@pytest.mark.parametrize("fault", ["negative", "over_budget", "duplicate_sibling", "empty", "extra"])
def test_v2_boundary_failures_are_rejected(fault):
    profile = group_profile()
    if fault == "negative":
        profile["records"][0]["group_sequence"] = -1
    elif fault == "over_budget":
        trace = grouped_reference()
        trace["events"].append(
            {
                "event": "request_interrupted",
                "rank": 8,
                "iteration": 0,
                "t_s": 9999,
                "request": 191,
                "generated_tokens": 10,
                "planned_generated_tokens": 9,
                "history_tokens": 20,
            }
        )
        with pytest.raises(ValueError):
            make_profile(trace, grouped=True)
        return
    elif fault == "duplicate_sibling":
        profile["records"][1]["sibling"] = 0
    elif fault == "empty":
        profile["records"] = []
    else:
        profile["records"][0]["unknown"] = 1
    with pytest.raises(ValueError):
        validate_profile(profile)


def test_grouped_profile_storage_delay_changes_checkpoint_timing(tmp_path):
    class DelayedKV(NVMeBackend):
        def write(self, key, data):
            if self.base_path.name == "kv":
                time.sleep(0.015)
            return super().write(key, data)

    config = replace(
        async_config("colocate_async", rollout_owners=2),
        request_profile=group_profile("colocate_async"),
        iterations=1,
        requests_per_rank=2,
    )
    baseline, _ = execute(tmp_path / "base", config)
    delayed, _ = execute(tmp_path / "slow", config, backend_factory=DelayedKV)

    def checkpoint(runner):
        return next(e["t_s"] for e in runner.trace.events if e["event"] == "checkpoint_begin")

    assert checkpoint(delayed) > checkpoint(baseline) + 0.04


def split_reference(mode="separate_async"):
    trace = grouped_reference(mode)
    for event in trace["events"]:
        event.pop("owner")
        event["member_owners"] = ["rollout-0", "rollout-1"]
    for index, request in enumerate(trace["requests"]):
        owner = index % 2
        request.update(owner=f"rollout-{owner}", rank=7 + owner)
    return trace


@pytest.mark.parametrize(
    "mode,execution", [("colocate_async", "local"), ("separate_async", "local"), ("separate_async", "mpi_shared")]
)
def test_siblings_on_different_owners_settle_as_one_group_and_release_files(tmp_path, mode, execution):
    profile = make_profile(split_reference(mode), grouped=True)
    config = replace(
        async_config(mode, rollout_owners=2, execution=execution),
        request_profile=profile,
        iterations=3,
        requests_per_rank=2,
    )
    if execution == "local":
        runner, summary = execute(tmp_path, config)
        trace = load_trace(runner.result_dir)
    else:
        summary, trace, _ = run_mpi(tmp_path, config)
    for event in [e for e in trace["events"] if e["event"] == "group_sample"]:
        siblings = [r for r in trace["requests"] if r["group"] == event["group"]]
        assert len(siblings) == 2
        assert {r["owner"] for r in siblings} == {"rollout-0", "rollout-1"}
        assert all(r["disposition"] == "accepted" for r in siblings)
    assert summary["async"]["sampled_groups"] == 3
    assert not list((tmp_path / "data").glob("**/trajectories/*.npy"))
    assert not analyze_trace(trace)["violations"]
    second = make_profile(trace, grouped=True)
    assert [r["owner"] for r in second["records"][:2]] == [0, 1]
    if execution == "mpi_shared":
        metrics = analyze_trace(trace)
        assert "group_straggler_s" not in metrics
        assert metrics["cross_rank_groups"][0] > 0


def test_cross_clock_siblings_do_not_produce_fake_straggler():
    trace = split_reference()
    metrics = analyze_trace(trace)
    assert metrics["group_fanout"] == [2, 2]
    assert metrics["cross_rank_groups"] == [2]
    assert "group_straggler_s" not in metrics


@pytest.mark.parametrize("fault", ["owner_length", "owner_format", "fanout", "orphan_partial", "duplicate_partial"])
def test_grouped_source_cannot_hide_unsupported_or_ambiguous_members(fault):
    trace = split_reference()
    if fault == "owner_length":
        trace["events"][0]["member_owners"] = ["rollout-0"]
    elif fault == "owner_format":
        trace["events"][0]["member_owners"][0] = "raw-server-address"
    elif fault == "fanout":
        trace["events"][0]["members"].append(192)
        trace["events"][0]["member_owners"].append("rollout-0")
    else:
        trace["requests"].pop(1)
        partial = {
            "event": "request_interrupted",
            "rank": 8,
            "iteration": 0,
            "t_s": 9000,
            "group": 9,
            "request": 999 if fault == "orphan_partial" else 191,
            "owner": "rollout-1",
            "generated_tokens": 2,
            "planned_generated_tokens": 9,
            "history_tokens": 9,
        }
        trace["events"].append(partial)
        if fault == "duplicate_partial":
            trace["events"].append(copy.deepcopy(partial))
    with pytest.raises(ValueError):
        make_profile(trace, grouped=True)


@pytest.mark.parametrize("target", ["worker_kv", "trainer_trajectory", "worker_gc"])
def test_split_owner_failure_propagates_without_complete_summary(tmp_path, monkeypatch, target):
    import test_agentrl_mpi_async as mpi_tests

    profile = make_profile(split_reference(), grouped=True)
    config = replace(mpi_tests.config_for_mpi(), request_profile=profile)
    monkeypatch.setattr(mpi_tests, "config_for_mpi", lambda: config)
    mpi_tests.test_mpi_role_failure_never_hangs_or_publishes_complete_summary(tmp_path, target, background=False)


def test_cancel_during_actual_kv_write_conserves_completed_generation(tmp_path):
    import asyncio

    from kv_cache.agentrl_async import AsyncLifecycle, RequestState

    async def attempt():
        writing = asyncio.Event()
        loop = asyncio.get_running_loop()

        class SlowKV(NVMeBackend):
            def write(self, key, data):
                if self.base_path.name == "kv":
                    loop.call_soon_threadsafe(writing.set)
                    time.sleep(0.03)
                return super().write(key, data)

        runner = AsyncLifecycle(async_config(), tmp_path / "data", tmp_path / "results", backend_factory=SlowKV)
        runner.setup()
        runner.decode_locks = [asyncio.Lock()]
        shape = runner.request_shape(0, 0)
        state = RequestState(0, 0, 0, 0, 0, shape, shape["prompt_tokens"])
        task = asyncio.create_task(runner.decode(state, 2, 0, shape["turns"][0]))
        await asyncio.wait_for(writing.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        generated = sum(e["tokens"] for e in runner.trace.events if e["event"] == "generation_end")
        assert state.generated == generated == 2
        assert state.history == shape["prompt_tokens"] + 2
        assert runner.io_active == 0
        assert len(runner.owners[0]["kv"].metadata) == 1
        assert any(e["event"] == "io_actual_end" for e in runner.trace.events)

    asyncio.run(attempt())
