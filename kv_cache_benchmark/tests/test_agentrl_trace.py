"""Reference ingestion preserves joint request shape and rejects false validation."""

import copy
import json

import pytest
from test_agentrl import run, tiny

from kv_cache.agentrl_trace import compare_traces, load_trace, make_profile, validate_profile


def profile_record(**kwargs):
    return {
        "rank": 0,
        "prompt_tokens": 7,
        "prefix_tokens": 3,
        "prefix_id": "shared",
        "trajectory_bytes": 257,
        "turns": [
            {"generated_tokens": 3, "decode_active_s": 0.015, "tool_delay_s": 0.012, "observation_tokens": 2},
            {"generated_tokens": 5, "decode_active_s": 0.025, "tool_delay_s": 0, "observation_tokens": 0},
        ],
        **kwargs,
    }


def envelope(records=None, **kwargs):
    records = records or [profile_record()]
    requests = [
        {
            **r,
            "iteration": 0,
            "request": i,
            "start_s": i,
            "end_s": i + 0.5,
            "policy_start": 0,
            "policy_end": 0,
            "prompt_policy": 0,
            "trainer_policy_at_accept": 0,
        }
        for i, r in enumerate(records)
    ]
    return {
        "schema_version": 1,
        "mode": "sync",
        "provenance": {
            "kind": "synthetic_fixture",
            "run_id": "fixture",
            "revision": "fixture-v1",
            "observation_boundary": "host_file_api",
            "clock": "rank_local",
        },
        "requests": requests,
        "events": [],
        **kwargs,
    }


def test_calibration_keeps_joint_turn_shape_and_source_identity(tmp_path):
    reference = envelope([profile_record(), profile_record(prompt_tokens=15, prefix_tokens=0, prefix_id=None)])
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference))
    loaded = load_trace(path)
    profile = make_profile(loaded)
    assert profile["records"][1]["prompt_tokens"] == 15
    assert profile["records"][0]["turns"][0]["tool_delay_s"] == 0.012
    assert profile["provenance"]["source_sha256"]
    assert profile["provenance"]["kind"] == "synthetic_fixture"


@pytest.mark.parametrize(
    "change",
    [
        {"prompt_tokens": True},
        {"prefix_tokens": 8},
        {"trajectory_bytes": -1},
        {
            "turns": [
                {"generated_tokens": 2, "decode_active_s": float("nan"), "tool_delay_s": 0, "observation_tokens": 0}
            ]
        },
        {"turns": [{"generated_tokens": 2, "decode_active_s": 0.1, "tool_delay_s": 0.1, "observation_tokens": 0}]},
        {"prefix_id": "../unsafe"},
    ],
)
def test_profile_validation_rejects_invalid_or_unreplayable_shape(change):
    profile = make_profile(envelope())
    profile["records"][0].update(change)
    with pytest.raises(ValueError):
        validate_profile(profile)


def test_joint_profile_drives_prompt_kv_tool_and_trajectory_io(tmp_path):
    profile = make_profile(
        envelope([profile_record(), profile_record(prompt_tokens=15, prefix_tokens=0, prefix_id=None)])
    )
    config = tiny(iterations=1, requests_per_rank=2, request_profile=profile)
    runner, _ = run(tmp_path, config)
    events = runner.trace.events
    assert {e["request"]: e["bytes"] for e in events if e["event"] == "io_end" and e["kind"] == "prompt"} == {
        0: 28,
        1: 60,
    }
    assert [
        e["bytes"] for e in events if e["event"] == "io_end" and e["kind"] == "trajectory" and e["op"] == "write"
    ] == [257, 257]
    assert sum(e["tokens"] for e in events if e["event"] == "generation_end") == 16
    assert (
        sum(e["bytes"] for e in events if e["event"] == "io_end" and e["kind"] == "kv" and e["op"] == "write")
        == (7 + 15 + 16 + 4) * config.bytes_per_token
    )
    normalized = load_trace(runner.result_dir)
    assert normalized["requests"][1]["prompt_tokens"] in (7, 15)
    assert normalized["provenance"]["calibration_sha256"] == profile["provenance"]["source_sha256"]


def test_comparison_detects_joint_and_timing_drift_without_byte_change():
    reference = envelope()
    candidate = copy.deepcopy(reference)
    candidate["provenance"]["run_id"] = "different"
    candidate["requests"][0]["turns"][0]["tool_delay_s"] = 0.12
    report = compare_traces(reference, candidate)
    assert report["status"] == "outside_tolerances"
    assert report["metrics"]["tool_delay_s"]["relative_quantile_error"] > 1
    assert report["real_validation"] is False


@pytest.mark.parametrize("mode", ["sync", "colocate_async", "separate_async"])
def test_mode_analysis_distinguishes_prompt_age_and_generated_version_span(mode):
    reference = envelope(mode=mode)
    reference["requests"][0].update(policy_start=2, policy_end=4, prompt_policy=1, trainer_policy_at_accept=5)
    report = compare_traces(reference, reference)
    assert report["candidate"]["prompt_age_steps"] == [5]
    assert report["candidate"]["generated_policy_span"] == [2]
    assert report["status"] == "inconclusive_reused_reference"
    assert report["real_validation"] is False


def test_mode_clock_and_boundary_mismatch_are_not_comparable():
    reference, candidate = envelope(), envelope()
    candidate["mode"] = "separate_async"
    with pytest.raises(ValueError, match="mode"):
        compare_traces(reference, candidate)
    candidate["mode"] = "sync"
    candidate["provenance"]["observation_boundary"] = "device"
    with pytest.raises(ValueError, match="boundary"):
        compare_traces(reference, candidate)


def test_async_profile_is_analysis_only_not_silently_sync(tmp_path):
    profile = make_profile(envelope(mode="separate_async"))
    with pytest.raises(ValueError, match="sync"):
        tiny(request_profile=profile)


def test_same_totals_with_training_before_completion_fails_causal_check():
    reference, candidate = envelope(), envelope()
    candidate["provenance"]["run_id"] = "bad-order"
    candidate["events"] = [
        {"rank": 0, "iteration": 0, "t_s": 0.1, "event": "train_begin"},
        {"rank": 0, "iteration": 0, "t_s": 0.2, "event": "rollout_complete", "request": 0},
    ]
    report = compare_traces(reference, candidate)
    assert report["status"] == "causal_violation"
    assert report["candidate"]["violations"]


@pytest.mark.parametrize(
    "mode,expected", [("colocate_async", "causal_violation"), ("separate_async", "within_tolerances")]
)
def test_overlap_across_iterations_checks_same_owner_clock(mode, expected):
    reference = envelope(mode=mode)
    reference["events"] = [
        {"rank": 0, "iteration": 0, "t_s": 0.1, "event": "train_begin"},
        {"rank": 0, "iteration": 1, "t_s": 0.2, "event": "generation_begin", "request": 0},
        {"rank": 0, "iteration": 1, "t_s": 0.3, "event": "generation_end", "request": 0},
        {"rank": 0, "iteration": 0, "t_s": 0.4, "event": "train_end"},
    ]
    candidate = copy.deepcopy(reference)
    candidate["provenance"]["run_id"] = "other"
    report = compare_traces(reference, candidate)
    assert report["status"] == expected
    assert report["candidate"]["same_rank_train_decode_overlap_s"] == pytest.approx([0.1])


@pytest.mark.parametrize("field,value", [("policy", 1), ("op", "read"), ("key", "different")])
def test_io_identity_cannot_change_at_actual_completion(field, value):
    reference = envelope()
    base = {
        "rank": 0,
        "iteration": 0,
        "io_id": 0,
        "op": "write",
        "kind": "kv",
        "key": "prefix",
        "request": 0,
        "policy": 0,
        "bytes": 128,
        "actual_s": 0.1,
    }
    reference["events"] = [
        {**base, "event": name, "t_s": t} for name, t in (("io_begin", 0.1), ("io_actual_end", 0.2), ("io_end", 0.2))
    ]
    candidate = copy.deepcopy(reference)
    candidate["provenance"]["run_id"] = "corrupt"
    candidate["events"][1][field] = value
    assert compare_traces(reference, candidate)["status"] == "causal_violation"


def test_declared_real_trace_without_lifecycle_coverage_is_inconclusive():
    reference = envelope([profile_record() for _ in range(20)])
    reference["provenance"]["kind"] = "verl"
    candidate = copy.deepcopy(reference)
    candidate["provenance"].update(run_id="holdout", kind="poc")
    assert compare_traces(reference, candidate)["status"] == "inconclusive_coverage"


@pytest.mark.parametrize(
    "change",
    [
        {"clock": "unaligned_wall_clock"},
        {"kind": "unknown"},
        {"run_id": ""},
    ],
)
def test_invalid_trace_provenance_is_rejected(change):
    data = envelope()
    data["provenance"].update(change)
    with pytest.raises(ValueError, match="provenance"):
        make_profile(data)


def test_profile_rejects_prefix_identity_with_different_lengths():
    with pytest.raises(ValueError, match="geometry"):
        make_profile(envelope([profile_record(), profile_record(prefix_tokens=4)]))


def test_duplicate_request_identity_is_rejected():
    data = envelope()
    data["requests"].append(data["requests"][0])
    with pytest.raises(ValueError, match="duplicate"):
        make_profile(data)
