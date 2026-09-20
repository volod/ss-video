"""Shared Python helpers for repo shell scripts.

This module provides a small CLI so bash scripts can delegate JSON parsing and
numeric calculations to named Python functions instead of inline heredocs.
`max-jobs` and `compute-flash-attn-jobs` re-export `ss_kit.hw`.
"""

import argparse
import json
import sys
from pathlib import Path

from ss_kit import hw as kit_hw


def pretty_json(stdin_text: str) -> str:
    """Format JSON from stdin with indentation."""
    payload = json.loads(stdin_text)
    return json.dumps(payload, indent=2)


def json_field(stdin_text: str, field: str, default: str = "") -> str:
    """Extract a top-level JSON field from stdin."""
    payload = json.loads(stdin_text)
    value = payload.get(field, default)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def cuda_version_from_json(path: str) -> str:
    """Read CUDA version from `/usr/local/cuda/version.json`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = payload.get("cuda", {}).get("version", "")
    return version.rsplit(".", 1)[0] if version else ""


def compute_flash_attn_jobs(
    total_kb: int,
    avail_kb: int,
    cpu_cores: int,
    ram_per_job_gb: float = 12.0,
    reserve_frac: float = 0.20,
) -> str:
    """Compute a conservative flash-attn parallel build budget."""
    return kit_hw.build_budget(
        total_kb, avail_kb, cpu_cores, ram_per_job_gb, reserve_frac
    ).summary()


def max_jobs(ram_per_job_gb: float = 12.0, reserve_frac: float = 0.20) -> int:
    """Return safe parallel C++/CUDA compilation job count for this machine.

    Canonical heavy-build budget for the main project (xformers, flash-attn).
    See AGENTS.md; the implementation is `ss_kit.hw.max_jobs` (`ss-kit max-jobs`).
    """
    return kit_hw.max_jobs(ram_per_job_gb, reserve_frac)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shell helper utilities for selfsuvis scripts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("pretty-json", help="Pretty-print JSON from stdin")

    json_field_parser = subparsers.add_parser("json-field", help="Read a JSON field from stdin")
    json_field_parser.add_argument("--field", required=True, help="Top-level JSON field name")
    json_field_parser.add_argument(
        "--default", default="", help="Fallback value when field is absent"
    )

    cuda_parser = subparsers.add_parser(
        "cuda-version-from-json",
        help="Read CUDA version from a version.json file",
    )
    cuda_parser.add_argument("--path", required=True, help="Path to CUDA version.json")

    subparsers.add_parser(
        "max-jobs",
        help="Print safe C++/CUDA parallel build job count for this machine",
    )

    flash_parser = subparsers.add_parser(
        "compute-flash-attn-jobs",
        help="Compute flash-attn parallel build job budget",
    )
    flash_parser.add_argument(
        "--total-kb", required=True, type=int, help="MemTotal from /proc/meminfo"
    )
    flash_parser.add_argument(
        "--avail-kb",
        required=True,
        type=int,
        help="MemAvailable from /proc/meminfo",
    )
    flash_parser.add_argument("--cpu-cores", required=True, type=int, help="Available CPU cores")
    flash_parser.add_argument(
        "--ram-per-job-gb",
        default=12.0,
        type=float,
        help="Estimated peak RAM per compilation job in GiB",
    )
    flash_parser.add_argument(
        "--reserve-frac",
        default=0.20,
        type=float,
        help="Fraction of total RAM to keep free",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "pretty-json":
        sys.stdout.write(pretty_json(sys.stdin.read()))
    elif args.command == "json-field":
        sys.stdout.write(json_field(sys.stdin.read(), field=args.field, default=args.default))
    elif args.command == "cuda-version-from-json":
        sys.stdout.write(cuda_version_from_json(args.path))
    elif args.command == "max-jobs":
        sys.stdout.write(f"{max_jobs()}\n")
    else:
        sys.stdout.write(
            compute_flash_attn_jobs(
                total_kb=args.total_kb,
                avail_kb=args.avail_kb,
                cpu_cores=args.cpu_cores,
                ram_per_job_gb=args.ram_per_job_gb,
                reserve_frac=args.reserve_frac,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
