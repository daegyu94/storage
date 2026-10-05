"""Real process/role/storage tests for the shared-filesystem MPI adapter."""

import asyncio
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, replace

import pytest
from test_agentrl_async import async_config
from test_agentrl_cli import invoke

from kv_cache.agentrl_async import PauseGate
from kv_cache.agentrl_mpi import Endpoint, RolloutEngine
from kv_cache.agentrl_trace import analyze_trace, load_trace


def config_for_mpi(owners=2, **settings):
    settings = {"parameter_sync_step": 2, **settings}
    return replace(
        async_config("separate_async", rollout_owners=owners, execution="mpi_shared", **settings),
        iterations=3,
        train_delay_s=0.08,
        response_tokens=[2, 2, 30, 30],
    )


def run_mpi(tmp_path, config=None, ranks=3):
    config = config or config_for_mpi()
    process = invoke(tmp_path, asdict(config), ranks=ranks)
    assert process.returncode == 0, process.stderr
    path = next((tmp_path / "results").glob("*/summary.json"))
    return json.loads(path.read_text()), load_trace(path.parent), path


def test_separate_mpi_has_real_roles_and_cross_process_trajectory_io(tmp_path):
    summary, trace, _ = run_mpi(tmp_path)
    assert summary["world_size"] == 3 and summary["fidelity"] == "uncalibrated"
    assert summary["topology"]["execution"] == "mpi_shared"
    roles = summary["topology"]["roles"]
    assert [r["role"] for r in roles] == ["trainer", "rollout", "rollout"]
    assert len({r["pid"] for r in roles}) == 3
    events = trace["events"]
    assert {e["rank"] for e in events if e["event"] == "train_begin"} == {0}
    assert {e["rank"] for e in events if e["event"] == "generation_begin"} == {1, 2}
    writes = [e for e in events if e["event"] == "io_end" and e["kind"] == "trajectory" and e["op"] == "write"]
    reads = [e for e in events if e["event"] == "io_begin" and e["kind"] == "trajectory" and e["op"] == "read"]
    assert reads and all(e["rank"] == 0 and e["role"] == "trainer" for e in reads)
    assert all(
        any(
            w["key"] == r["key"] and w["owner"] == r["owner"] and w["bytes"] == r["bytes"] and w["rank"] in (1, 2)
            for w in writes
        )
        for r in reads
    )
    assert summary["async"]["terminal_queue_peak"] <= 2
    assert len([e for e in events if e["event"] == "train_begin"]) == 6
    assert summary["async"]["outstanding_groups_peak"] <= 3
    assert all(r["kv_live_payload_bytes"] == 0 for r in summary["ranks"])
    assert all(
        e["role"] == "gc"
        for e in events
        if e["event"] == "io_begin" and e["rank"] != 0 and e["kind"] == "trajectory" and e["op"] == "delete"
    )
    assert not list((tmp_path / "data").glob("**/trajectories/*.npy"))
    assert not analyze_trace(trace)["violations"]
    assert summary["topology"]["global_clock_aligned"] is False


def test_mpi_partial_versions_and_causal_overlap_do_not_depend_on_clock_alignment(tmp_path):
    config = replace(config_for_mpi(), iterations=6, train_delay_s=0.06, response_tokens=[2, 2, 24, 24])
    summary, trace, _ = run_mpi(tmp_path, config)
    assert any(e["event"] == "re_prefill_end" for e in trace["events"])
    assert any(r["policy_end"] > r["policy_start"] for r in trace["requests"])
    overlap = summary["distributed"]["causal_io_overlap"]
    assert overlap["train"] > 0
    assert summary["distributed"]["installed_policy_acks"] == 2 * config.iterations
    assert summary["distributed"]["causal_io_overlap_by_kind_op"]["train"]["kv_write"] > 0
    for progress in (e for e in trace["events"] if e["event"] == "remote_io_completion" and e["causal_overlap"]):
        operation = [
            e
            for e in trace["events"]
            if e["rank"] == progress["worker_rank"] and e.get("io_id") == progress["io_id"] and e["event"] == "io_begin"
        ]
        assert len(operation) == 1 and operation[0]["phase_token"] == progress["token"]
    for rank in (1, 2):
        events = [e for e in trace["events"] if e["rank"] == rank]
        for retired in (e for e in events if e["event"] == "kv_logical_invalidate"):
            assert not any(
                e["event"] == "io_begin"
                and e["op"] == "read"
                and e["key"] in retired["keys"]
                and e["t_s"] > retired["t_s"]
                for e in events
            )
    # Deliberate independent clock shifts preserve I/O/stage pairing locally.
    for e in trace["events"]:
        e["t_s"] += e["rank"] * 1000
    for r in trace["requests"]:
        r["start_s"] += r["rank"] * 1000
        r["end_s"] += r["rank"] * 1000
    assert not analyze_trace(trace)["violations"]


@pytest.mark.parametrize("strategy", ["drop", "wait"])
def test_mpi_staleness_and_small_queue_finish_without_deadlock(tmp_path, strategy):
    config = config_for_mpi(queue_capacity=1, max_prompt_age=1, staleness_strategy=strategy)
    summary, trace, _ = run_mpi(tmp_path, config)
    assert summary["final_policy_version"] == 3
    accepted = [r for r in trace["requests"] if r["disposition"] == "accepted"]
    if strategy == "drop":
        assert all(r["trainer_policy_at_accept"] - r["prompt_policy"] + 1 <= 1 for r in accepted)
    else:
        assert summary["async"]["dropped_groups"] == 0


@pytest.mark.parametrize(
    "ranks,owners,mode", [(1, 1, "separate_async"), (2, 2, "separate_async"), (3, 2, "colocate_async")]
)
def test_invalid_mpi_role_topology_fails_before_io(tmp_path, ranks, owners, mode):
    config = replace(config_for_mpi(owners), trainer_mode=mode)
    process = invoke(tmp_path, asdict(config), ranks=ranks)
    assert process.returncode != 0
    assert not (tmp_path / "data").exists()


def test_trainer_only_normalized_shard_is_preserved(tmp_path):
    _, _, path = run_mpi(tmp_path)
    trainer = json.loads((path.parent / "trace-rank-0.json").read_text())
    assert trainer["execution_role"] == "trainer" and trainer["requests"] == []
    assert any(e["event"] == "train_begin" for e in trainer["events"])


@pytest.mark.parametrize("fraction", [0, 1])
def test_mpi_tool_pauses_and_memory_trajectory_mode(tmp_path, fraction):
    config = replace(
        config_for_mpi(),
        turns=2,
        tool_delay_s=0.025,
        observation_tokens=2,
        persist_trajectories=False,
        kv_offload_fraction=fraction,
    )
    summary, trace, _ = run_mpi(tmp_path, config)
    assert "trajectory_write" not in summary["io_totals"]
    assert all(r["trajectory_bytes"] == 0 for r in trace["requests"])
    assert bool([e for e in trace["events"] if e["event"] == "io_begin" and e["kind"] == "kv"]) is (fraction > 0)
    assert any(e["event"] == "tool_begin" for e in trace["events"])
    assert not analyze_trace(trace)["violations"]


@pytest.mark.parametrize("command", ["retire", "install"])
def test_owner_requires_safe_retirement_before_policy_install(tmp_path, command):
    engine = RolloutEngine(config_for_mpi(), tmp_path / "data", tmp_path / "results")
    engine.gate = PauseGate()

    async def attempt():
        if command == "install":
            await engine.gate.pause()
        with pytest.raises(ValueError, match="retire|paused"):
            await engine.command(command, 1)

    asyncio.run(attempt())


def test_control_rpc_timeout_clears_pending_future():
    class SilentPeer:
        def isend(self, message, dest, tag):
            return object()

    endpoint = Endpoint(SilentPeer(), None, None, None, 0.005)

    async def request():
        with pytest.raises(RuntimeError, match="control RPC timeout"):
            await endpoint.call(1, "pause")
        assert not endpoint.pending

    asyncio.run(request())


@pytest.mark.parametrize("field", ["owner", "role", "phase_token"])
def test_trace_detects_changed_storage_context_between_arrival_and_completion(tmp_path, field):
    from test_agentrl import run, tiny

    runner, _ = run(tmp_path, tiny(iterations=1, requests_per_rank=1))
    trace = load_trace(runner.result_dir)
    next(e for e in trace["events"] if e["event"] == "io_end")[field] = "changed-context"
    assert any("I/O identity mismatch" in v for v in analyze_trace(trace)["violations"])


@pytest.mark.parametrize("target", ["worker_kv", "worker_gc", "trainer_checkpoint", "trainer_trajectory", "unshared"])
def test_mpi_role_failure_never_hangs_or_publishes_complete_summary(tmp_path, target):
    if not shutil.which("mpiexec"):
        pytest.skip("mpiexec unavailable")
    pytest.importorskip("mpi4py")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(asdict(config_for_mpi())))
    helper = tmp_path / "role_failure.py"
    helper.write_text("""
import json, sys
from pathlib import Path
from mpi4py import MPI
from kv_cache.agentrl import AgentRLConfig, create_lifecycle
from kv_cache.backends import NVMeBackend
rank = MPI.COMM_WORLD.Get_rank()
target = sys.argv[4]
class Broken(NVMeBackend):
    def write(self, key, data):
        if (target == "worker_kv" and rank == 1 and self.base_path.name == "kv") or (
            target == "trainer_checkpoint" and rank == 0 and key == "state"):
            raise OSError("injected role storage failure")
        return super().write(key, data)
    def read(self, key):
        if target == "trainer_trajectory" and rank == 0 and self.base_path.name == "trajectories":
            raise OSError("injected trainer read failure")
        return super().read(key)
    def delete(self, key):
        if target == "worker_gc" and rank == 2 and self.base_path.name == "kv":
            raise OSError("injected worker GC failure")
        return super().delete(key)
try:
    root = Path(sys.argv[2])
    if target == "unshared": root = root / f"isolated-rank-{rank}"
    create_lifecycle(AgentRLConfig.from_dict(json.loads(Path(sys.argv[1]).read_text())),
                     root, sys.argv[3], comm=MPI.COMM_WORLD, backend_factory=Broken).run()
except Exception as exc:
    print(str(exc), file=sys.stderr, flush=True)
    MPI.COMM_WORLD.Abort(1)
""")
    from test_agentrl_cli import SCRIPT

    process = subprocess.run(
        [
            "mpiexec",
            "--oversubscribe",
            "-n",
            "3",
            sys.executable,
            str(helper),
            str(config_path),
            str(tmp_path / "data"),
            str(tmp_path / "results"),
            target,
        ],
        capture_output=True,
        text=True,
        timeout=20,
        env={
            **os.environ,
            "PYTHONPATH": str(SCRIPT.parent),
            "OMPI_ALLOW_RUN_AS_ROOT": "1",
            "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
        },
    )
    assert process.returncode != 0
    assert "injected" in process.stderr if target != "unshared" else "shared visibility" in process.stderr
    assert not list((tmp_path / "results").glob("*/summary.json"))
