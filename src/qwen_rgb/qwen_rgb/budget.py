"""Pre-load memory budget. The numbers live in the Checkpoint 1.5 script.

This module refuses to run if that script's spare line, projection floor, or
the 6 GiB OS floor has been edited. It does not define a lower line.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from vlm_iface.pins import MIN_AVAILABLE_BYTES

SPARE_REQUIRED_BYTES = 5 * 1024**3
PROJECTED_REMAINING_FLOOR_BYTES = 2 * 1024**3
OS_FLOOR_BYTES = 6 * 1024**3


def repo_root_from(start: Path) -> Path:
    """Find the worktree that contains the Checkpoint 1.5 budget script."""
    for candidate in (start, *start.parents):
        budget = candidate / "docs" / "checkpoint-1.5" / "run_model_feasibility.py"
        if budget.is_file():
            return candidate
    raise RuntimeError(f"Checkpoint 1.5 budget script is not above {start}")


def load_budget_module(repo_root: Path):
    """Import the Checkpoint 1.5 budget without running its ``main``."""
    path = repo_root / "docs" / "checkpoint-1.5" / "run_model_feasibility.py"
    if not path.is_file():
        raise RuntimeError(f"Missing budget script at {path}")
    spec = importlib.util.spec_from_file_location("checkpoint15_model_feasibility", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load the budget module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.SPARE_REQUIRED_BYTES != SPARE_REQUIRED_BYTES:
        raise RuntimeError("Checkpoint 1.5 spare line is not 5 GiB. Refusing to replace it.")
    if module.PROJECTED_REMAINING_FLOOR_BYTES != PROJECTED_REMAINING_FLOOR_BYTES:
        raise RuntimeError("Checkpoint 1.5 projection floor is not 2 GiB. Refusing to replace it.")
    if MIN_AVAILABLE_BYTES != OS_FLOOR_BYTES:
        raise RuntimeError("The 6 GiB OS floor changed. Refusing to adopt a lower floor.")
    module.assert_budget_fixtures(MIN_AVAILABLE_BYTES)
    return module


def dtype_decisions(budget, *, parameter_count: int, float32_weight_bytes: int, available_bytes: int, mps_available: bool) -> dict[str, Any]:
    """Float32 and BF16 decisions from the imported budget. Available bytes are the caller's reading."""
    if parameter_count < 0 or float32_weight_bytes < 0 or available_bytes < 0:
        raise ValueError("byte counts must be non-negative")
    decisions = {
        "float32": budget.budget_decision(
            resident_bytes=int(float32_weight_bytes),
            available_bytes=int(available_bytes),
            ratio=budget.C1_DROP_PER_CURRENT,
            min_available_bytes=MIN_AVAILABLE_BYTES,
            mps_available=mps_available,
        ),
        "bfloat16": budget.budget_decision(
            resident_bytes=int(parameter_count) * 2,
            available_bytes=int(available_bytes),
            ratio=budget.C1_DROP_PER_CURRENT,
            min_available_bytes=MIN_AVAILABLE_BYTES,
            mps_available=mps_available,
        ),
    }
    decisions["float32"]["dtype"] = "float32"
    decisions["bfloat16"]["dtype"] = "bfloat16"
    return decisions


def select_dtype(decisions: dict[str, Any], *, bf16_probe_ok: bool) -> dict[str, Any]:
    """Float32 when its budget passes, otherwise BF16 when its budget and the MPS probe pass."""
    if decisions["float32"].get("permitted"):
        return {"dtype": "float32", "permitted": True, "reason": "The float32 budget passed."}
    bf16 = decisions["bfloat16"]
    if bf16.get("permitted") and bf16_probe_ok:
        return {
            "dtype": "bfloat16",
            "permitted": True,
            "reason": "Float32 failed. The BF16 budget passed and the MPS BF16 probe ran.",
        }
    if bf16.get("permitted") and not bf16_probe_ok:
        return {
            "dtype": None,
            "permitted": False,
            "reason": "The BF16 budget passed, but the MPS BF16 probe did not. from_pretrained was not called.",
        }
    return {
        "dtype": None,
        "permitted": False,
        "reason": "Neither dtype cleared the Checkpoint 1.5 pre-load budget.",
    }
