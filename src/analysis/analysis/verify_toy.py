"""Rebuild the committed toy summary and compare it to the file on disk.

Exit status 0 means the compact fixtures reproduce the summary. This command
does not load a model and does not run an attack.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .schemas import SchemaError
from .toy_summary import SUMMARY_FILENAME, build_toy_summary, canonical_json, fixture_dir

ABSOLUTE_TOLERANCE = 1e-9


def compare_summaries(rebuilt: Any, published: Any, prefix: str = "$") -> list[str]:
    """Structural differences. Floats may differ by at most 1e-9."""
    if type(rebuilt) is bool or type(published) is bool:
        if type(rebuilt) is bool and type(published) is bool and rebuilt == published:
            return []
        return [f"{prefix}: {rebuilt!r} != {published!r}"]
    if isinstance(rebuilt, dict) or isinstance(published, dict):
        if not isinstance(rebuilt, dict) or not isinstance(published, dict):
            return [f"{prefix}: type mismatch"]
        mismatches = []
        for key in sorted(set(rebuilt) | set(published)):
            path = f"{prefix}.{key}"
            if key not in rebuilt:
                mismatches.append(f"{path}: missing from reconstruction")
            elif key not in published:
                mismatches.append(f"{path}: unexpected key in reconstruction")
            else:
                mismatches.extend(compare_summaries(rebuilt[key], published[key], path))
        return mismatches
    if isinstance(rebuilt, list) or isinstance(published, list):
        if not isinstance(rebuilt, list) or not isinstance(published, list):
            return [f"{prefix}: type mismatch"]
        if len(rebuilt) != len(published):
            return [f"{prefix}: length {len(rebuilt)} != {len(published)}"]
        mismatches = []
        for index, (got, want) in enumerate(zip(rebuilt, published)):
            mismatches.extend(compare_summaries(got, want, f"{prefix}[{index}]"))
        return mismatches
    if isinstance(rebuilt, (int, float)) and isinstance(published, (int, float)):
        if isinstance(rebuilt, int) and isinstance(published, int):
            if rebuilt != published:
                return [f"{prefix}: {rebuilt!r} != {published!r}"]
            return []
        if abs(float(rebuilt) - float(published)) <= ABSOLUTE_TOLERANCE:
            return []
        return [f"{prefix}: {rebuilt!r} != {published!r}"]
    if rebuilt != published:
        return [f"{prefix}: {rebuilt!r} != {published!r}"]
    return []


def load_published_summary(path: Path) -> Any:
    def reject(token: str) -> None:
        raise ValueError(f"non-finite JSON token {token}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def verify_toy_summary(directory: Path | None = None) -> dict[str, Any]:
    root = Path(directory) if directory is not None else fixture_dir()
    published_path = root / SUMMARY_FILENAME
    if not published_path.is_file():
        return {
            "ok": False,
            "n_mismatches": 1,
            "mismatches": [f"missing {published_path}"],
            "numpy_version": np.__version__,
        }
    try:
        rebuilt = build_toy_summary(root)
        published = load_published_summary(published_path)
    except (SchemaError, ValueError, OSError) as exc:
        return {
            "ok": False,
            "n_mismatches": 1,
            "mismatches": [str(exc)],
            "numpy_version": np.__version__,
        }
    mismatches = compare_summaries(rebuilt, published)
    return {
        "ok": len(mismatches) == 0,
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:30],
        "numpy_version": np.__version__,
        "summary_path": str(published_path),
    }


def write_toy_summary(directory: Path | None = None) -> Path:
    """Maintainer helper. Verification itself does not rewrite the file."""
    root = Path(directory) if directory is not None else fixture_dir()
    payload = build_toy_summary(root)
    path = root / SUMMARY_FILENAME
    path.write_text(canonical_json(payload), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, default=None)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Rewrite toy_summary.json from the fixtures. Not used for verification.",
    )
    args = parser.parse_args()
    if args.write:
        written = write_toy_summary(args.fixture_dir)
        print(json.dumps({"wrote": str(written)}, indent=2))
    report = verify_toy_summary(args.fixture_dir)
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
