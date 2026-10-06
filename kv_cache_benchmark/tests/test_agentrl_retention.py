"""Physical FS retention must not extend logical policy/cache eligibility."""

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from builtins import ExceptionGroup
from dataclasses import asdict

import pytest
from test_agentrl_cli import SCRIPT, invoke
from test_agentrl_kv_lifecycle import execute
from test_agentrl_tiering import config

from kv_cache.agentrl import AgentRLConfig, create_lifecycle
from kv_cache.agentrl_tiering import TierSettings
from kv_cache.agentrl_trace import analyze_trace, load_trace
from kv_cache.backends import NVMeBackend

MODES = ("sync", "colocate_async", "separate_async")


def retained_config(mode="sync", **fields):
    return config(mode, **{"fs_retention": "persistent", **fields})


def completed_kv(events, op):
    return [e for e in events if e["event"] == "io_end" and e["kind"] == "kv" and e["op"] == op]


def assert_policy_isolation(events):
    for retired in (e for e in events if e["event"] in ("cache_logical_invalidate", "kv_fs_retained")):
        assert not any(
            e["event"] == "io_begin"
            and e["kind"] == "kv"
            and e["op"] == "read"
            and e["key"] in retired["keys"]
            and e["t_s"] > retired["t_s"]
            for e in events
        )
    for e in completed_kv(events, "read") + completed_kv(events, "write"):
        assert e["key"].startswith(f"v{e['policy']}-")


@pytest.mark.parametrize("value", [None, True, "keep", 1, []])
def test_unknown_retention_is_rejected_before_creating_storage(value):
    with pytest.raises(ValueError, match="retention"):
        retained_config(fs_retention=value)


def test_default_retention_and_cpu_only_constraint():
    assert TierSettings.parse({"cpu_capacity_bytes": 512}).fs_retention == "policy_gc"
    with pytest.raises(ValueError, match="retention|fs_enabled"):
        retained_config(fs_enabled=False)
    assert AgentRLConfig().fingerprint == "730cbd4d5532f15f6b4d44d76c8966c0e97e7565cf0b9b36cea0fea7c0b5c7de"


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("execution", ["caller_await", "background"])
def test_retained_files_survive_policy_and_run_but_are_never_reloaded(tmp_path, mode, execution):
    raw = asdict(retained_config(mode, fs_execution=execution))
    if mode != "sync":
        # A queued completed cohort alone need not decode under a new policy.
        # Stragglers force the abort/re-prefill path this test must exercise.
        raw.update(iterations=5, response_tokens=[2, 2, 30, 30])
        raw["rollout_reference"]["kv_budget_bytes_per_gpu"] = 4096
        raw["kv_cache_model"].pop("capacity_bytes")
    cfg = AgentRLConfig.from_dict(raw)
    summary, trace, runner = execute(tmp_path, cfg)
    events = trace["events"]
    assert summary["fidelity"] == "uncalibrated"
    assert not analyze_trace(trace)["violations"]
    writes = completed_kv(events, "write")
    assert writes and not completed_kv(events, "delete")
    policies = {e["policy"] for e in writes}
    assert 0 in policies and any(version > 0 for version in policies)
    files = list(runner.storage_dir.glob("**/kv/*.npy"))
    assert len(files) == len({e["key"] for e in writes})
    rank = summary["ranks"][0]
    physical = sum(p.stat().st_size for p in files)
    payload = sum(runner.kv_sizes.values())
    assert rank["kv_retained_file_count"] == len(files)
    assert rank["kv_retained_payload_bytes"] == rank["kv_live_payload_bytes"] == payload > 0
    assert rank["kv_retained_physical_file_bytes"] == rank["kv_live_physical_file_bytes"] == physical > payload
    assert rank["kv_peak_physical_file_bytes"] == physical
    for pool in runner.cache_pools.values():
        assert not pool.tiers.fs_keys and not pool.entries
        assert pool.tiers.summary()["cpu_reserved_bytes"] == 0
        assert pool.tiers.summary()["cpu_peak_bytes"] <= cfg.kv_offload_tiers["cpu_capacity_bytes"]
    retired = [e for e in events if e["event"] == "kv_fs_retained"]
    assert retired and all(e["file_count"] == len(e["keys"]) for e in retired)
    assert sum(e["payload_bytes"] for e in retired) == payload
    assert sum(e["physical_file_bytes"] for e in retired) == physical
    assert_policy_isolation(events)
    if mode == "colocate_async":
        assert any(e["event"] == "re_prefill_end" for e in events)
    if mode != "sync":
        assert not runner.retired and not runner.gc_jobs
        assert not any(e["event"] == "kv_gc_begin" for e in events)


@pytest.mark.parametrize("mode", MODES)
def test_omitted_retention_keeps_policy_gc_and_zero_final_footprint(tmp_path, mode):
    summary, trace, runner = execute(tmp_path, config(mode))
    assert completed_kv(trace["events"], "delete")
    assert not list(runner.storage_dir.glob("**/kv/*.npy"))
    rank = summary["ranks"][0]
    assert rank["kv_live_physical_file_bytes"] == rank["kv_retained_file_count"] == 0
    assert rank["kv_retained_payload_bytes"] == rank["kv_retained_physical_file_bytes"] == 0
    assert rank["kv_peak_physical_file_bytes"] > rank["kv_peak_payload_bytes"] > 0


def test_actual_file_accounting_handles_overwrite_read_and_delete(tmp_path):
    runner = create_lifecycle(retained_config(), tmp_path / "data", tmp_path / "results")
    runner.setup()

    async def operations():
        await runner.io(runner.kv, "write", "a", 16, kind="kv")
        await runner.io(runner.kv, "write", "b", 32, kind="kv")
        peak = runner.kv._get_path("a").stat().st_size + runner.kv._get_path("b").stat().st_size
        await runner.io(runner.kv, "write", "a", 8, kind="kv")
        await runner.io(runner.kv, "read", "b", kind="kv")
        rank = runner.rank_summary()
        assert rank["kv_live_payload_bytes"] == 40
        assert rank["kv_live_physical_file_bytes"] == peak - 8
        assert rank["kv_peak_payload_bytes"] == 48
        assert rank["kv_peak_physical_file_bytes"] == peak
        await runner.io(runner.kv, "delete", "b", kind="kv")
        assert runner.rank_summary()["kv_live_payload_bytes"] == 8
        assert runner.rank_summary()["kv_live_physical_file_bytes"] == runner.kv._get_path("a").stat().st_size
        await runner.io(runner.kv, "delete", "a", kind="kv")
        assert runner.rank_summary()["kv_live_physical_file_bytes"] == 0

    asyncio.run(operations())
    assert all("kv_live_physical_file_bytes" in e for e in runner.trace.events if e["event"] == "io_actual_end")


def test_sync_recovery_is_cold_kv_in_new_namespace_and_keeps_old_files(tmp_path):
    cfg = retained_config()
    _, _, original = execute(tmp_path / "original", cfg)
    files = list(original.storage_dir.glob("**/kv/*.npy"))
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest = original.storage_dir / "checkpoints/step-1/manifest.json"
    summary, trace, resumed = execute(tmp_path / "resume", cfg, resume=manifest)
    assert summary["final_policy_version"] == 2
    assert original.storage_dir.name != resumed.storage_dir.name
    assert hashes == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    writes = completed_kv(trace["events"], "write")
    assert writes and {e["policy"] for e in writes} == {1}
    assert not any(e["key"].startswith("v0-") for e in completed_kv(trace["events"], "read"))
    assert_policy_isolation(trace["events"])


def test_sync_delete_service_is_removed_from_next_rollout_causal_path(tmp_path):
    class SlowDelete(NVMeBackend):
        def delete(self, key):
            if self.base_path.name == "kv":
                time.sleep(0.003)
            return super().delete(key)

    for retention in ("policy_gc", "persistent"):
        _, trace, _ = execute(tmp_path / retention, config(fs_retention=retention), backend_factory=SlowDelete)
        events = trace["events"]
        first_end = next(e for e in events if e["event"] == "iteration_end" and e["iteration"] == 0)
        next_rollout = next(e for e in events if e["event"] == "rollout_phase_begin" and e["iteration"] == 1)
        deletes = [e for e in completed_kv(events, "delete") if e["iteration"] == 0]
        assert next_rollout["t_s"] > first_end["t_s"]
        if retention == "policy_gc":
            assert deletes and all(e["actual_s"] >= 0.003 for e in deletes)
            assert max(e["t_s"] for e in deletes) < first_end["t_s"]
        else:
            assert not deletes
            assert any(e["event"] == "kv_fs_retained" and e["t_s"] < first_end["t_s"] for e in events)


@pytest.mark.parametrize("mode,ranks", [("sync", 2), ("separate_async", 3)])
def test_mpi_retention_preserves_rank_ownership_without_old_policy_reads(tmp_path, mode, ranks):
    raw = asdict(retained_config(mode, fs_execution="background"))
    if mode == "separate_async":
        raw["async_workload"].update(execution="mpi_shared", rollout_owners=2)
    process = invoke(tmp_path, raw, ranks=ranks)
    assert process.returncode == 0, process.stderr
    path = next((tmp_path / "results").glob("*/summary.json"))
    summary, trace = json.loads(path.read_text()), load_trace(path.parent)
    assert not analyze_trace(trace)["violations"]
    assert not completed_kv(trace["events"], "delete")
    for rank in summary["ranks"]:
        events = [e for e in trace["events"] if e["rank"] == rank["rank"]]
        assert_policy_isolation(events)
        files = list((tmp_path / "data").glob(f"*/rank-{rank['rank']}/**/kv/*.npy"))
        assert rank["kv_retained_file_count"] == len(files)
        assert rank["kv_live_physical_file_bytes"] == sum(p.stat().st_size for p in files)
        if mode == "separate_async" and rank["rank"] == 0:
            assert not files and not completed_kv(events, "write")
        else:
            assert files and completed_kv(events, "write")
    if mode == "sync":
        files = list((tmp_path / "data").glob("**/kv/*.npy"))
        hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        manifest = next((tmp_path / "data").glob("*/checkpoints/step-1/manifest.json"))
        process = invoke(tmp_path / "resume", raw, ranks=ranks, extra=("--resume", str(manifest)))
        assert process.returncode == 0, process.stderr
        assert hashes == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        path = next((tmp_path / "resume/results").glob("*/summary.json"))
        trace = load_trace(path.parent)
        assert {e["policy"] for e in completed_kv(trace["events"], "write")} == {1}


@pytest.mark.parametrize("mode", MODES)
def test_persistent_write_failure_still_prevents_success_and_drains(tmp_path, mode):
    class Broken(NVMeBackend):
        def write(self, key, data):
            if self.base_path.name == "kv":
                raise OSError("injected FS full")
            return super().write(key, data)

    runner = create_lifecycle(
        retained_config(mode, fs_execution="background"),
        tmp_path / "data",
        tmp_path / "results",
        backend_factory=Broken,
    )
    with pytest.raises((ExceptionGroup, RuntimeError, OSError)):
        runner.run()
    assert not list((tmp_path / "results").glob("*/summary.json"))
    assert not runner.io_active
    assert not runner.kv_sizes


@pytest.mark.parametrize("mode,ranks", [("sync", 2), ("separate_async", 3)])
@pytest.mark.parametrize("target", ["kv", "checkpoint"])
def test_mpi_persistent_storage_failure_propagates_without_success(tmp_path, mode, ranks, target):
    if not shutil.which("mpiexec"):
        pytest.skip("mpiexec unavailable")
    pytest.importorskip("mpi4py")
    raw = asdict(retained_config(mode, fs_execution="background"))
    if mode == "separate_async":
        raw["async_workload"].update(execution="mpi_shared", rollout_owners=2)
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(raw))
    helper = tmp_path / "fail_retained.py"
    helper.write_text("""
import json, sys
from pathlib import Path
from mpi4py import MPI
from kv_cache.agentrl import AgentRLConfig, create_lifecycle
from kv_cache.backends import NVMeBackend
rank = MPI.COMM_WORLD.Get_rank()
class Broken(NVMeBackend):
    def write(self, key, data):
        if (sys.argv[4] == "kv" and rank == 1 and self.base_path.name == "kv") or (
            sys.argv[4] == "checkpoint" and rank == 0 and key == "state"):
            raise OSError("injected persistent storage failure")
        return super().write(key, data)
try:
    config = AgentRLConfig.from_dict(json.loads(Path(sys.argv[1]).read_text()))
    create_lifecycle(config, sys.argv[2], sys.argv[3], comm=MPI.COMM_WORLD, backend_factory=Broken).run()
except Exception as exc:
    print(str(exc), file=sys.stderr, flush=True)
    MPI.COMM_WORLD.Abort(1)
""")
    process = subprocess.run(
        [
            "mpiexec",
            "--oversubscribe",
            "-n",
            str(ranks),
            sys.executable,
            str(helper),
            str(config_file),
            str(tmp_path / "data"),
            str(tmp_path / "results"),
            target,
        ],
        capture_output=True,
        text=True,
        timeout=25,
        env={
            **os.environ,
            "PYTHONPATH": str(SCRIPT.parent),
            "OMPI_ALLOW_RUN_AS_ROOT": "1",
            "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
        },
    )
    assert process.returncode != 0 and "injected persistent storage failure" in process.stderr
    assert not list((tmp_path / "results").glob("*/summary.json"))
