"""Command line. The default command prints the plan and does not load Qwen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _bootstrap() -> None:
    here = Path(__file__).resolve()
    root = None
    for candidate in (here, *here.parents):
        if (candidate / "docs" / "checkpoint-1.9" / "ATTACK_PROTOCOL.md").is_file():
            root = candidate
            break
    if root is None:
        return
    for name in (
        "cuda_validation",
        "cuda_readiness",
        "attack_feasibility",
        "qwen_rgb",
        "qwen_rgb_loss",
        "choice_eval",
        "vlm_iface",
        "pilot",
        "dev_pilot",
    ):
        entry = str(root / "src" / name)
        if entry not in sys.path:
            sys.path.insert(0, entry)


_bootstrap()

from .compare import IdenticalDevicesError, compare_records
from .measure import measure
from .protocol import build_protocol, repo_root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="One-scene MPS versus CUDA validation. Does not launch a RunPod pod."
    )
    sub = parser.add_subparsers(dest="command")
    protocol = sub.add_parser("protocol", help="Print the plan. Does not load Qwen.")
    protocol.add_argument("--output", type=Path)
    compared = sub.add_parser("compare", help="Record differences between one MPS sidecar and one CUDA sidecar.")
    compared.add_argument("mps_json", type=Path)
    compared.add_argument("cuda_json", type=Path)
    compared.add_argument("--output", type=Path)
    measured = sub.add_parser("measure", help="Run the frozen 3-step attack on dev4-ref-00-a.")
    measured.add_argument("--device", required=True, choices=["mps", "cuda"])
    measured.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    command = args.command or "protocol"
    root = repo_root(Path(__file__).resolve())
    if command == "protocol":
        _emit(build_protocol(root), getattr(args, "output", None))
        return
    if command == "compare":
        _compare(args.mps_json, args.cuda_json, args.output)
        return
    if command == "measure":
        payload = measure(root, device_name=args.device, output=args.output)
        if payload.get("status") == "error":
            raise SystemExit(1)
        return
    parser.error(f"unknown command {command}")


def _compare(mps_path: Path, cuda_path: Path, output: Path | None) -> None:
    try:
        mps = json.loads(mps_path.read_text(encoding="utf-8"))
        cuda = json.loads(cuda_path.read_text(encoding="utf-8"))
        report = compare_records(mps, cuda, mps_dir=mps_path.parent, cuda_dir=cuda_path.parent)
    except (IdenticalDevicesError, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(2) from exc
    _emit(report, output)
    if report["status"] != "differences_recorded":
        raise SystemExit(3)


def _emit(payload: dict[str, Any], output: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
