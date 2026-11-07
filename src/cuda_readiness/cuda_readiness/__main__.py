"""Command line for bookkeeping. The default command does not load Qwen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .estimate import build_estimate
from .hashes import hash_files
from .manifest import build_manifest
from .preflight import probe, require_cuda


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="CUDA readiness bookkeeping. Does not run an attack and does not load Qwen."
    )
    sub = parser.add_subparsers(dest="command")

    preflight = sub.add_parser("preflight", help="Report CUDA visibility without scoring.")
    preflight.add_argument("--require-cuda", action="store_true")
    preflight.add_argument("--output", type=Path)

    manifest = sub.add_parser("manifest", help="Write the development manifest and science hashes.")
    manifest.add_argument("--output", type=Path)

    estimate = sub.add_parser("estimate", help="Print the planning estimate. This is not a CUDA measurement.")
    estimate.add_argument("--output", type=Path)

    hashed = sub.add_parser("hash", help="SHA-256 the given files. Held-out paths are refused.")
    hashed.add_argument("paths", nargs="+", type=Path)

    args = parser.parse_args(argv)
    command = args.command or "preflight"
    root = _repo_root(Path(__file__).resolve())
    if command == "preflight":
        report = probe()
        _emit(report, getattr(args, "output", None))
        if getattr(args, "require_cuda", False):
            require_cuda(report)
        return
    if command == "manifest":
        _emit(build_manifest(root), args.output)
        return
    if command == "estimate":
        _emit(build_estimate(root), args.output)
        return
    if command == "hash":
        _emit(hash_files(args.paths, root=root), None)
        return
    parser.error(f"unknown command {command}")


def _emit(payload: dict[str, Any], output: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    sys.stdout.write(text)


def _repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "docs" / "checkpoint-1.9" / "ATTACK_PROTOCOL.md").is_file():
            return candidate
    raise RuntimeError(f"Checkpoint 1.9 protocol is not above {start}")


if __name__ == "__main__":
    main()
