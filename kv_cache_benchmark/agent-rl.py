#!/usr/bin/env python3
"""GPU-less experimental Agent RL workload entry point."""

import argparse
import json
import os
import sys
from pathlib import Path

from kv_cache.agentrl import AgentRLConfig, create_lifecycle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--storage-root", required=True, type=Path)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--profile", type=Path, help="Joint normalized request profile matching trainer_mode")
    parser.add_argument("--mpi", action="store_true", help="Use mpi4py collectives under an MPI launcher")
    args = parser.parse_args()
    comm = None
    try:
        if args.mpi:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
        elif any(
            int(os.environ.get(name, "1")) > 1
            for name in (
                "OMPI_COMM_WORLD_SIZE",
                "PMI_SIZE",
                "MV2_COMM_WORLD_SIZE",
            )
        ):
            raise ValueError("MPI launcher detected: pass --mpi to connect lifecycle collectives")
        raw = args.config.read_text()
        if args.config.suffix.lower() == ".json":
            values = json.loads(raw)
        else:
            import yaml

            values = yaml.safe_load(raw)
        if args.profile:
            if not isinstance(values, dict) or "request_profile" in values:
                raise ValueError("--profile requires a mapping without an embedded request_profile")
            values["request_profile"] = json.loads(args.profile.read_text())
        config = AgentRLConfig.from_dict(values)
        runner = create_lifecycle(config, args.storage_root, args.results_dir, comm=comm, resume=args.resume)
        summary = runner.run()
        if runner.rank == 0:
            print(
                json.dumps(
                    {
                        "summary": str(runner.result_dir / "summary.json"),
                        "world_size": summary["world_size"],
                        "status": summary["status"],
                    }
                )
            )
        return 0
    except Exception as exc:
        print(f"agentrl: {exc}", file=sys.stderr, flush=True)
        if comm is not None and comm.Get_size() > 1:
            # Also handles failures before the first collective, e.g. config parse.
            comm.Abort(1)
        return 1


if __name__ == "__main__":
    sys.exit(main())
