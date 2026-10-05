#!/usr/bin/env python3
"""Normalize joint calibration profiles or compare a declared holdout trace."""

import argparse
import json
import sys
from pathlib import Path

from kv_cache.agentrl import SyncLifecycle
from kv_cache.agentrl_trace import compare_traces, load_trace, make_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    profile = commands.add_parser("profile")
    profile.add_argument("--input", type=Path, required=True)
    profile.add_argument(
        "--grouped", action="store_true", help="Preserve async coordinator groups and rollout owners (V2)"
    )
    compare = commands.add_parser("compare")
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--relative-tolerance", type=float, default=0.25)
    compare.add_argument("--ks-tolerance", type=float, default=0.3)
    for command in (profile, compare):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "profile":
            data = make_profile(load_trace(args.input), grouped=args.grouped)
        else:
            data = compare_traces(
                load_trace(args.reference),
                load_trace(args.candidate),
                relative_tolerance=args.relative_tolerance,
                ks_tolerance=args.ks_tolerance,
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        SyncLifecycle.write_json(args.output, data)
        print(json.dumps({"output": str(args.output), "status": data.get("status", "profile_created")}))
        return 0
    except Exception as exc:
        print(f"agentrl-trace: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
