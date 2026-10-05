"""Explicit host-observation contract, joint calibration and holdout comparison.

This is not a parser for arbitrary veRL logs. Producers must instrument and
normalize the documented boundaries; rank-local clocks are never merged.
"""

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

MODES = ("sync", "colocate_async", "separate_async")
KINDS = ("verl", "synthetic_fixture", "poc")


def digest(trace):
    clean = {**trace, "provenance": {k: v for k, v in trace["provenance"].items() if k != "source_sha256"}}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, allow_nan=False).encode()).hexdigest()


def number(value, name, *, integer=False, minimum=0):
    types = (int,) if integer else (int, float)
    try:
        valid = type(value) in types and math.isfinite(value) and value >= minimum
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"invalid {name}")


def validate_record(record):
    required = {"rank", "prompt_tokens", "prefix_tokens", "prefix_id", "trajectory_bytes", "turns"}
    if not isinstance(record, dict) or set(record) != required:
        raise ValueError("profile record fields mismatch")
    for name in ("rank", "prompt_tokens", "prefix_tokens", "trajectory_bytes"):
        number(record[name], name, integer=True, minimum=1 if name == "prompt_tokens" else 0)
    if record["prefix_tokens"] > record["prompt_tokens"]:
        raise ValueError("prefix exceeds prompt")
    prefix = record["prefix_id"]
    if record["prefix_tokens"]:
        if not isinstance(prefix, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", prefix):
            raise ValueError("prefix_id must be a sanitized stable identifier")
    elif prefix is not None:
        raise ValueError("zero prefix requires prefix_id=null")
    turns = record["turns"]
    if not isinstance(turns, list) or not turns:
        raise ValueError("profile turns must be a nonempty list")
    for turn in turns:
        if not isinstance(turn, dict) or set(turn) != {
            "generated_tokens",
            "decode_active_s",
            "tool_delay_s",
            "observation_tokens",
        }:
            raise ValueError("turn fields mismatch")
        for name in ("generated_tokens", "observation_tokens"):
            number(turn[name], name, integer=True)
        for name in ("decode_active_s", "tool_delay_s"):
            number(turn[name], name)
        if bool(turn["generated_tokens"]) != bool(turn["decode_active_s"]):
            raise ValueError("decode active time must match nonzero tokens")
        if turn["generated_tokens"] and not math.isfinite(turn["generated_tokens"] / turn["decode_active_s"]):
            raise ValueError("decode rate is not finite")
    if not sum(t["generated_tokens"] for t in turns):
        raise ValueError("profile must generate tokens")
    if turns[-1]["tool_delay_s"] or turns[-1]["observation_tokens"]:
        raise ValueError("last turn cannot leave an unmodeled tool interaction")


def validate_profile(profile):
    if not isinstance(profile, dict) or set(profile) != {"schema_version", "mode", "provenance", "records"}:
        raise ValueError("profile envelope fields mismatch")
    if type(profile["schema_version"]) is not int or profile["schema_version"] != 1 or profile["mode"] not in MODES:
        raise ValueError("unsupported profile schema/mode")
    provenance = profile["provenance"]
    if not isinstance(provenance, dict) or provenance.get("kind") not in KINDS:
        raise ValueError("profile source kind missing")
    if not re.fullmatch(r"[a-f0-9]{64}", provenance.get("source_sha256", "")):
        raise ValueError("profile source SHA256 missing")
    records = profile["records"]
    if not isinstance(records, list) or not records:
        raise ValueError("profile must contain joint records")
    prefixes = {}
    for record in records:
        validate_record(record)
        identity = (record["rank"], record["prefix_id"])
        if record["prefix_tokens"]:
            if identity in prefixes and prefixes[identity] != record["prefix_tokens"]:
                raise ValueError("prefix identity has inconsistent geometry")
            prefixes[identity] = record["prefix_tokens"]
    return profile


def validate_trace(trace):
    if (
        not isinstance(trace, dict)
        or type(trace.get("schema_version")) is not int
        or trace.get("schema_version") != 1
        or trace.get("mode") not in MODES
    ):
        raise ValueError("unsupported trace schema/mode")
    provenance = trace.get("provenance", {})
    if (
        provenance.get("kind") not in KINDS
        or provenance.get("clock") != "rank_local"
        or not all(
            isinstance(provenance.get(k), str) and provenance[k] for k in ("run_id", "revision", "observation_boundary")
        )
    ):
        raise ValueError("trace provenance or rank_local clock missing")
    if (
        not isinstance(trace.get("requests"), list)
        or not trace["requests"]
        or not isinstance(trace.get("events"), list)
    ):
        raise ValueError("trace requires requests and events lists")
    seen = set()
    record_fields = ("rank", "prompt_tokens", "prefix_tokens", "prefix_id", "trajectory_bytes", "turns")
    for request in trace["requests"]:
        validate_record({k: request.get(k) for k in record_fields})
        for name in ("iteration", "request", "policy_start", "policy_end", "prompt_policy", "trainer_policy_at_accept"):
            number(request.get(name), name, integer=True)
        for name in ("start_s", "end_s"):
            number(request.get(name), name)
        if request["end_s"] < request["start_s"] or request["policy_end"] < request["policy_start"]:
            raise ValueError("request time/version reversed")
        if request["trainer_policy_at_accept"] < request["prompt_policy"]:
            raise ValueError("prompt is from a future policy")
        if request["trainer_policy_at_accept"] < request["policy_end"]:
            raise ValueError("generated tokens are from a future policy")
        identity = tuple(request[k] for k in ("rank", "iteration", "request"))
        if identity in seen:
            raise ValueError("duplicate request identity")
        seen.add(identity)
    for event in trace["events"]:
        if not isinstance(event, dict) or not isinstance(event.get("event"), str):
            raise ValueError("invalid event")
        number(event.get("rank"), "event rank", integer=True)
        number(event.get("iteration"), "event iteration", integer=True, minimum=-1)
        number(event.get("t_s"), "event time")
        if event["event"] in ("io_begin", "io_actual_end", "io_end"):
            number(event.get("io_id"), "I/O identity", integer=True)
            number(event.get("bytes"), "I/O bytes", integer=True)
            if event.get("op") not in ("read", "write", "delete") or not isinstance(event.get("kind"), str):
                raise ValueError("invalid I/O operation")
            if not isinstance(event.get("key"), str):
                raise ValueError("I/O key missing")
            number(event.get("policy"), "I/O policy", integer=True)
            if event.get("request") is not None:
                number(event["request"], "I/O request", integer=True)
            if event["event"] == "io_end":
                number(event.get("actual_s"), "actual I/O latency")
        if event["event"] == "rollout_start":
            number(event.get("queue_wait_s"), "admission wait")
    return trace


def load_trace(path):
    path = Path(path)
    if path.is_dir():
        paths = sorted(path.glob("trace-rank-*.json"))
        if not paths:
            raise ValueError("no normalized per-rank traces found")
        traces = [load_trace(p) for p in paths]
        first = traces[0]
        if any(t["mode"] != first["mode"] or t["provenance"] != first["provenance"] for t in traces):
            raise ValueError("per-rank provenance mismatch")
        return validate_trace(
            {
                **first,
                "requests": [r for t in traces for r in t["requests"]],
                "events": [e for t in traces for e in t["events"]],
            }
        )
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("trace shard exceeds 64 MiB; split by rank/run")
    return validate_trace(json.loads(path.read_text()))


def make_profile(trace):
    validate_trace(trace)
    names = ("rank", "prompt_tokens", "prefix_tokens", "prefix_id", "trajectory_bytes", "turns")
    records = [
        {k: r[k] for k in names}
        for r in sorted(trace["requests"], key=lambda r: (r["rank"], r["iteration"], r["request"]))
    ]
    return validate_profile(
        {
            "schema_version": 1,
            "mode": trace["mode"],
            "provenance": {**trace["provenance"], "source_sha256": digest(trace)},
            "records": records,
        }
    )


def analyze_trace(trace):
    validate_trace(trace)
    metrics, violations = defaultdict(list), []
    for request in trace["requests"]:
        turns = request["turns"]
        tokens = sum(t["generated_tokens"] for t in turns)
        active = sum(t["decode_active_s"] for t in turns)
        metrics["prompt_tokens"].append(request["prompt_tokens"])
        metrics["response_tokens"].append(tokens)
        metrics["prefix_fraction"].append(request["prefix_tokens"] / request["prompt_tokens"])
        metrics["trajectory_bytes"].append(request["trajectory_bytes"])
        metrics["request_duration_s"].append(request["end_s"] - request["start_s"])
        metrics["decode_tokens_per_active_s"].append(tokens / active)
        metrics["tool_delay_s"].extend(t["tool_delay_s"] for t in turns[:-1])
        metrics["prompt_age_steps"].append(request["trainer_policy_at_accept"] - request["prompt_policy"] + 1)
        metrics["generated_policy_span"].append(request["policy_end"] - request["policy_start"])
        metrics["generated_policy_lag"].append(request["trainer_policy_at_accept"] - request["policy_end"])
    grouped = defaultdict(list)
    owner_events = defaultdict(list)
    ios = defaultdict(dict)
    for event in trace["events"]:
        grouped[(event["rank"], event["iteration"])].append(event)
        owner_events[event["rank"]].append(event)
        name = event["event"]
        if name.startswith("io_") and "io_id" in event:
            identity = (event["rank"], event["io_id"])
            if name in ios[identity]:
                violations.append(f"duplicate {name} for {identity}")
            ios[identity][name] = event
        if name == "io_begin":
            metrics[f"{event['kind']}_{event['op']}_bytes"].append(event["bytes"])
        if name == "io_end":
            metrics[f"{event['kind']}_{event['op']}_latency_s"].append(event["actual_s"])
        if name == "rollout_start":
            metrics["admission_wait_s"].append(event["queue_wait_s"])
        if name == "generation_begin" and "compute_queue_wait_s" in event:
            metrics["compute_queue_wait_s"].append(event["compute_queue_wait_s"])
        if name == "io_actual_end" and "kv_live_payload_bytes" in event:
            metrics["kv_live_payload_bytes"].append(event["kv_live_payload_bytes"])
    for identity, pairs in ios.items():
        if set(pairs) != {"io_begin", "io_actual_end", "io_end"}:
            violations.append(f"incomplete I/O {identity}")
            continue
        begin, actual, end = (pairs[k] for k in ("io_begin", "io_actual_end", "io_end"))
        if not begin["t_s"] <= actual["t_s"] <= end["t_s"] or len({e["bytes"] for e in pairs.values()}) != 1:
            violations.append(f"I/O boundary mismatch {identity}")
        if any(
            len({e.get(k) for e in pairs.values()}) != 1
            for k in ("op", "kind", "key", "request", "policy", "iteration")
        ):
            violations.append(f"I/O identity mismatch {identity}")
    for identity, events in grouped.items():
        events.sort(key=lambda e: e["t_s"])

        def times(name, events=events):
            return [e["t_s"] for e in events if e["event"] == name]

        completed, trained = times("rollout_complete"), times("train_begin")
        if trace["mode"] == "sync" and completed and trained and min(trained) < max(completed):
            violations.append(f"sync training precedes completion {identity}")
        if completed and trained:
            metrics["completion_to_train_s"].append(min(trained) - max(completed))
        start, checkpoint = times("iteration_begin"), times("checkpoint_begin")
        if start and checkpoint:
            metrics["iteration_to_checkpoint_s"].append(min(checkpoint) - min(start))
        arrivals = [e["t_s"] for e in events if e["event"] == "io_begin" and e["kind"] == "kv" and e["op"] == "write"]
        metrics["kv_write_interarrival_s"].extend(b - a for a, b in zip(arrivals, arrivals[1:], strict=False))
    for identity, events in owner_events.items():
        events.sort(key=lambda e: e["t_s"])
        opened, spans = {}, defaultdict(list)
        for event in events:
            name = event["event"]
            stage = name.removesuffix("_begin").removesuffix("_end")
            if stage not in ("generation", "tool", "train"):
                continue
            request_identity = (event["iteration"], event.get("request"))
            key = (stage, request_identity)
            if name.endswith("_begin"):
                if key in opened:
                    violations.append(f"duplicate stage begin {identity} {key}")
                opened[key] = event["t_s"]
            elif key in opened:
                spans[stage].append((opened.pop(key), event["t_s"], request_identity))
            else:
                violations.append(f"unpaired stage end {identity} {key}")
        if opened:
            violations.append(f"unclosed stages {identity}")
        overlap = 0.0
        for a, b, request in spans["generation"]:
            for c, d, tool_request in spans["tool"]:
                if request == tool_request and min(b, d) > max(a, c):
                    violations.append(f"generation during tool pause {identity} {request}")
            overlap += sum(max(0, min(b, d) - max(a, c)) for c, d, _ in spans["train"])
        if overlap and trace["mode"] in ("sync", "colocate_async"):
            violations.append(f"same owner compute overlap {identity}")
        metrics["same_rank_train_decode_overlap_s"].append(overlap)
    for key in list(metrics):
        if key.endswith("_bytes") and key != "trajectory_bytes" and key != "kv_live_payload_bytes":
            metrics[f"{key}_total"] = [sum(metrics[key])]
            metrics[f"{key}_op_count"] = [len(metrics[key])]
    metrics["request_count"] = [len(trace["requests"])]
    return {**metrics, "violations": violations}


def compare_traces(reference, candidate, *, relative_tolerance=0.25, ks_tolerance=0.3):
    validate_trace(reference)
    validate_trace(candidate)
    for value in (relative_tolerance, ks_tolerance):
        number(value, "comparison tolerance")
    if reference["mode"] != candidate["mode"]:
        raise ValueError("mode mismatch")
    if reference["provenance"]["observation_boundary"] != candidate["provenance"]["observation_boundary"]:
        raise ValueError("observation boundary mismatch")
    left, right = analyze_trace(reference), analyze_trace(candidate)
    comparisons, passed = {}, True
    for key in sorted((set(left) | set(right)) - {"violations"}):
        a, b = left.get(key, []), right.get(key, [])
        if not a and not b:
            continue
        if not a or not b:
            comparisons[key] = {"coverage_match": False, "pass": False}
            passed = False
            continue
        qa, qb = np.percentile(a, (50, 95, 99)), np.percentile(b, (50, 95, 99))
        relative = float(np.max(np.abs(qb - qa) / np.maximum(np.abs(qa), 1e-9)))
        points = np.unique(a + b)
        ks = float(
            np.max(
                np.abs(
                    np.searchsorted(np.sort(a), points, side="right") / len(a)
                    - np.searchsorted(np.sort(b), points, side="right") / len(b)
                )
            )
        )
        accepted = relative <= relative_tolerance and ks <= ks_tolerance
        comparisons[key] = {
            "coverage_match": True,
            "reference_count": len(a),
            "candidate_count": len(b),
            "reference_p50_p95_p99": qa.tolist(),
            "candidate_p50_p95_p99": qb.tolist(),
            "relative_quantile_error": relative,
            "ks_distance": ks,
            "pass": accepted,
        }
        passed = passed and accepted
    source_sha = digest(reference)
    reused = (
        source_sha == digest(candidate)
        or reference["provenance"]["run_id"] == candidate["provenance"]["run_id"]
        or source_sha == candidate["provenance"].get("calibration_sha256")
    )
    status = "within_tolerances" if passed else "outside_tolerances"
    required = {"generation_begin", "generation_end", "rollout_complete", "train_begin", "train_end", "policy_install"}
    coverage = {
        name: {
            "observed_events": sorted({e["event"] for e in trace["events"]}),
            "missing_lifecycle_boundaries": sorted(required - {e["event"] for e in trace["events"]}),
        }
        for name, trace in (("reference", reference), ("candidate", candidate))
    }
    if reference["provenance"]["kind"] == "verl" and any(v["missing_lifecycle_boundaries"] for v in coverage.values()):
        status = "inconclusive_coverage"
    if reference["provenance"]["kind"] == "verl" and min(len(reference["requests"]), len(candidate["requests"])) < 20:
        status = "inconclusive_sample_size"
    if reused:
        status = "inconclusive_reused_reference"
    if left["violations"] or right["violations"]:
        status = "causal_violation"
    return {
        "schema_version": 1,
        "status": status,
        "real_validation": False,
        "reference_sha256": source_sha,
        "candidate_sha256": digest(candidate),
        "reference_kind": reference["provenance"]["kind"],
        "mode": reference["mode"],
        "tolerances": {"relative_quantile": relative_tolerance, "ks": ks_tolerance},
        "reference": left,
        "candidate": right,
        "metrics": comparisons,
        "coverage": coverage,
        "limitations": [
            "Thresholds must be registered before an independent holdout run.",
            "No aligned cross-rank concurrency or device/wire traffic is inferred.",
            "Observed agreement alone is not a full Agent RL fidelity certification.",
        ],
    }
