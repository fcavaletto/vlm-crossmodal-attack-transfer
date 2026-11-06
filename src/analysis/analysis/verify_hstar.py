"""Rebuild the fabricated H* summary and compare it to the committed file.

Exit status 0 means the worked packet reproduces the summary. The packet is
fabricated. This command does not load a model, does not run an attack, and
does not read the Checkpoint 1 development scenes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .hstar import fit_hstar, packet_sha256
from .hstar_packet import PACKET_FILENAME, build_worked_packet, fixture_dir, load_packet
from .toy_summary import canonical_json
from .verify_toy import compare_summaries, load_published_summary

SUMMARY_FILENAME = "hstar_summary.json"
WORKED_BOOT = 200
WORKED_SEED = 20261107
WORKED_ALPHA = 0.05


def build_hstar_summary(rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    packet = build_worked_packet() if rows is None else rows
    fitted = fit_hstar(packet, n_boot=WORKED_BOOT, seed=WORKED_SEED, alpha=WORKED_ALPHA)
    fitted["packet_sha256"] = packet_sha256(packet)
    fitted["packet_name"] = PACKET_FILENAME
    fitted["worked_example"] = {
        "included_pgd_alignment_coefficient": 0.5,
        "included_pgd_intercept": 0.5,
        "included_task_coefficients": 0.0,
        "noise_alignment_coefficient": -0.5,
        "note": (
            "The included PGD rows satisfy strict_error = 1/2 + (1/2) A(s, t). "
            "Task indicators have coefficient 0. The noise rows flip the "
            "outcome, so their alignment coefficient is -1/2. Residuals are 0, "
            "and the wild-bootstrap interval collapses to that point."
        ),
    }
    return fitted


def verify_hstar_summary(directory: Path | None = None) -> dict[str, Any]:
    root = Path(directory) if directory is not None else fixture_dir()
    published_path = root / SUMMARY_FILENAME
    packet_path = root / PACKET_FILENAME
    if not published_path.is_file() or not packet_path.is_file():
        missing = [str(path) for path in (packet_path, published_path) if not path.is_file()]
        return {
            "ok": False,
            "n_mismatches": len(missing),
            "mismatches": [f"missing {path}" for path in missing],
            "numpy_version": np.__version__,
        }
    try:
        loaded = load_packet(root)
        expected = build_worked_packet()
        mismatches = compare_summaries(loaded, expected, prefix="packet")
        rebuilt = build_hstar_summary(loaded)
        published = load_published_summary(published_path)
        mismatches.extend(compare_summaries(rebuilt, published))
    except (OSError, TypeError, ValueError) as exc:
        return {
            "ok": False,
            "n_mismatches": 1,
            "mismatches": [str(exc)],
            "numpy_version": np.__version__,
        }
    return {
        "ok": len(mismatches) == 0,
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:30],
        "numpy_version": np.__version__,
        "summary_path": str(published_path),
    }


def write_hstar_summary(directory: Path | None = None) -> Path:
    """Maintainer helper. Verification itself does not rewrite the file."""
    root = Path(directory) if directory is not None else fixture_dir()
    root.mkdir(parents=True, exist_ok=True)
    payload = build_hstar_summary(load_packet(root))
    path = root / SUMMARY_FILENAME
    path.write_text(canonical_json(payload), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, default=None)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Rewrite hstar_summary.json from the packet. Not used for verification.",
    )
    args = parser.parse_args()
    if args.write:
        written = write_hstar_summary(args.fixture_dir)
        print(json.dumps({"wrote": str(written)}, indent=2))
    report = verify_hstar_summary(args.fixture_dir)
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
