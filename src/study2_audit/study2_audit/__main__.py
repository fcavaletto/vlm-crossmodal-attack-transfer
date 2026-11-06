"""Print the audit JSON. Does not write the Qwen measurement."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .audit import build_audit


def _repository_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "docs" / "checkpoint-1.6" / "qwen_feasibility.json").is_file():
            return candidate
    raise SystemExit("Could not find the Checkpoint 1.6 Qwen measurement from this file.")


def main() -> int:
    root = _repository_root()
    report = build_audit(root)
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    if report["original_qwen_json_rewritten"] or report["h_star_fitted"] or report["model_inference"]:
        return 1
    if not report["qwen_records_match_pins"] or not report["reproducibility_ok"]:
        return 1
    if report["decision"]["decision"] != "no-go" or report["decision"]["supports_h_star"] is not False:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
