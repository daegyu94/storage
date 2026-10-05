"""Actual CLI/MPI correctness tests; subprocess timeouts catch hangs."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "agent-rl.py"


def invoke(tmp_path, config=None, ranks=1, extra=(), env=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps(
            config
            or {
                "iterations": 2,
                "requests_per_rank": 2,
                "prompt_tokens": 2,
                "response_tokens": [2, 8],
                "tokens_per_second": 100,
                "checkpoint_bytes_per_rank": 64,
                "kv_offload_fraction": 1,
                "rank_rate_factors": [1, 0.5],
            }
        )
    )
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--config",
        str(config_file),
        "--storage-root",
        str(tmp_path / "data"),
        "--results-dir",
        str(tmp_path / "results"),
        *extra,
    ]
    if ranks > 1:
        if not shutil.which("mpiexec"):
            pytest.skip("mpiexec unavailable")
        pytest.importorskip("mpi4py")
        cmd = ["mpiexec", "--oversubscribe", "-n", str(ranks), *cmd, "--mpi"]
    process = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=25,
        env={**os.environ, **(env or {}), "OMPI_ALLOW_RUN_AS_ROOT": "1", "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1"},
    )
    return process


@pytest.mark.parametrize("ranks", [1, 2])
def test_cli_run_and_resume(tmp_path, ranks):
    process = invoke(tmp_path, ranks=ranks)
    assert process.returncode == 0, process.stderr
    summary_file = next((tmp_path / "results").glob("*/summary.json"))
    summary = json.loads(summary_file.read_text())
    assert summary["world_size"] == ranks
    assert summary["final_policy_version"] == 2
    events = json.loads((summary_file.parent / "rank-0.json").read_text())
    assert events
    manifest = next((tmp_path / "data").glob("*/checkpoints/step-1/manifest.json"))
    resumed = invoke(tmp_path / "resume", ranks=ranks, extra=("--resume", str(manifest)))
    assert resumed.returncode == 0, resumed.stderr
    if ranks == 2:
        # The faster rank waits for the slower rank before any trainer phase.
        assert summary["ranks"][0]["barrier_wait_s"] > 0.06


def test_mpi_rank_failure_does_not_hang_or_publish_success(tmp_path):
    # A blocked filesystem path forces a coordinated setup failure.
    (tmp_path / "data").write_text("not a directory")
    process = invoke(tmp_path, ranks=2)
    assert process.returncode != 0
    assert "setup" in process.stderr
    assert not list((tmp_path / "results").glob("*/summary.json"))


@pytest.mark.parametrize("config", [{"trainer_mode": "separate_async"}, {"tokens_per_second": 0}, {"typo": 1}])
def test_cli_rejects_bad_configuration(tmp_path, config):
    process = invoke(tmp_path, config)
    assert process.returncode != 0
    assert not (tmp_path / "data").exists()


def test_launcher_without_mpi_flag_is_rejected(tmp_path):
    process = invoke(tmp_path, env={"OMPI_COMM_WORLD_SIZE": "2"})
    assert process.returncode != 0
    assert "--mpi" in process.stderr


def test_resume_wrong_world_size_is_rejected(tmp_path):
    process = invoke(tmp_path, ranks=1)
    assert process.returncode == 0
    manifest = next((tmp_path / "data").glob("*/checkpoints/step-1/manifest.json"))
    resumed = invoke(tmp_path / "wrong-world", ranks=2, extra=("--resume", str(manifest)))
    assert resumed.returncode != 0
    assert "world size" in resumed.stderr


def test_single_rank_rollout_io_failure_reaches_all_peers(tmp_path):
    if not shutil.which("mpiexec"):
        pytest.skip("mpiexec unavailable")
    pytest.importorskip("mpi4py")
    helper = tmp_path / "fail_rank.py"
    helper.write_text("""
from mpi4py import MPI
from kv_cache.agentrl import AgentRLConfig, SyncLifecycle
from kv_cache.backends import NVMeBackend
import sys
class Broken(NVMeBackend):
    def write(self, key, data):
        if MPI.COMM_WORLD.Get_rank() == 1 and self.base_path.name == "kv":
            raise OSError("one-rank failure")
        return super().write(key, data)
try:
    SyncLifecycle(AgentRLConfig.from_dict({"kv_offload_fraction": 1}),
                  sys.argv[1], sys.argv[2], comm=MPI.COMM_WORLD, backend_factory=Broken).run()
except RuntimeError as exc:
    print(str(exc), file=sys.stderr, flush=True)
    sys.exit(7)
""")
    process = subprocess.run(
        [
            "mpiexec",
            "--oversubscribe",
            "-n",
            "2",
            sys.executable,
            str(helper),
            str(tmp_path / "data"),
            str(tmp_path / "results"),
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
    assert process.returncode != 0
    assert process.stderr.count("rank 1 rollout") == 2
    assert not list((tmp_path / "results").glob("*/summary.json"))
