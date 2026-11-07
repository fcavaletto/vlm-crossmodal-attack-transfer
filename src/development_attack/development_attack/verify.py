"""Checksums, bounds, decisions, and the untouched memory lines.

This module does not load a model and does not take a step.
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any

from .protocol import (
    BF16_REQUIRED_AVAILABLE_BYTES,
    BOUND_ABSOLUTE_SLACK,
    COLORS,
    EPSILON_KEY,
    FLOAT32_REQUIRED_AVAILABLE_BYTES,
    OS_FLOOR_BYTES,
    PGD_STEPS,
    PRESERVED_PATHS,
    PRIMARY_STEPS,
    SEED_PREFIX,
    SOURCE_CONTEXT,
    STARTED_FROM_COMMIT,
    DevelopmentStop,
    epsilon,
)


def assert_memory_lines(root: Path) -> dict[str, Any]:
    """Refuse to start when a Checkpoint 1.5 line is missing or smaller."""
    from qwen_rgb.budget import OS_FLOOR_BYTES as BUDGET_FLOOR
    from qwen_rgb.budget import dtype_decisions, load_budget_module
    from qwen_rgb_loss.protocol import assert_unchanged_lines

    try:
        assert_unchanged_lines()
    except RuntimeError as exc:
        raise DevelopmentStop("stopped_memory_guard", str(exc)) from exc
    if BUDGET_FLOOR != OS_FLOOR_BYTES or OS_FLOOR_BYTES != 6 * 1024**3:
        raise DevelopmentStop(
            "stopped_memory_guard",
            "The 6 GiB floor is not 6,442,450,944. The guard was not lowered, and no perturbation was run.",
        )
    budget = load_budget_module(root)
    decisions = dtype_decisions(
        budget,
        parameter_count=budget.QWEN_C0_PARAMETERS,
        float32_weight_bytes=budget.QWEN_C0_FLOAT32_BYTES,
        available_bytes=BF16_REQUIRED_AVAILABLE_BYTES,
        mps_available=True,
    )
    bf16 = int(decisions["bfloat16"]["required_available_bytes"])
    fp32 = int(decisions["float32"]["required_available_bytes"])
    floor = int(decisions["bfloat16"]["lines_bytes"]["os_floor"])
    if bf16 != BF16_REQUIRED_AVAILABLE_BYTES or fp32 != FLOAT32_REQUIRED_AVAILABLE_BYTES or floor != OS_FLOOR_BYTES:
        raise DevelopmentStop(
            "stopped_memory_guard",
            "The pre-load lines do not match the analysis protocol "
            f"(bfloat16 {bf16}, float32 {fp32}, floor {floor}). No perturbation was run.",
        )
    if decisions["float32"].get("permitted") is True:
        raise DevelopmentStop(
            "stopped_memory_guard",
            "Float32 was permitted at the bfloat16 line. No perturbation was run.",
        )
    return decisions


def preserved_changes(root: Path) -> list[str]:
    """Paths from earlier checkpoints that differ from the protocol commit."""
    diff = subprocess.check_output(
        ["git", "diff", "--name-only", STARTED_FROM_COMMIT, "--", *PRESERVED_PATHS],
        cwd=root,
        text=True,
    )
    return [line for line in diff.splitlines() if line.strip()]


def assert_preserved(root: Path) -> None:
    changes = preserved_changes(root)
    if changes:
        preview = ", ".join(changes[:8])
        raise DevelopmentStop(
            "stopped_protocol",
            "Earlier checkpoint files changed before the attack: " + preview,
        )


def bound_errors(linf_before: object, linf_after: object, epsilon_rgb: float) -> list[str]:
    """The projected image stays in the ball. The official image may move by half a level."""
    before = _finite(linf_before)
    after = _finite(linf_after)
    if before is None or after is None:
        return ["a recorded image is missing a finite L-infinity distance"]
    errors: list[str] = []
    if before < -BOUND_ABSOLUTE_SLACK or after < -BOUND_ABSOLUTE_SLACK:
        errors.append("an L-infinity distance is negative")
    if before > float(epsilon_rgb) + BOUND_ABSOLUTE_SLACK:
        errors.append(
            f"projected image leaves the L-infinity ball: {before} > {float(epsilon_rgb)}"
        )
    round_trip_limit = float(epsilon_rgb) + (0.5 / 255.0) + BOUND_ABSOLUTE_SLACK
    if after > round_trip_limit:
        errors.append(
            f"official image leaves the ball by more than half a level: {after} > {round_trip_limit}"
        )
    return errors


def decision_errors(decision: object) -> list[str]:
    """A missing decision is missing. Any other string is illegal."""
    if decision is None:
        return []
    if isinstance(decision, str) and decision in COLORS:
        return []
    return [f"decision {decision!r} is not one of red, blue, green, yellow"]


def attack_constant_errors(progress: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if progress.get("source_context") != SOURCE_CONTEXT:
        errors.append("source context is not neutral")
    if progress.get("epsilon") != EPSILON_KEY:
        errors.append("epsilon is not 1/255")
    if progress.get("pgd_steps") != PGD_STEPS:
        errors.append("the trajectory is not 7 steps")
    if progress.get("primary_steps") != PRIMARY_STEPS:
        errors.append("the primary step count is not 3")
    if progress.get("seed_prefix") != SEED_PREFIX:
        errors.append("the random-sign seed prefix changed")
    if progress.get("epsilon_raised") is True or progress.get("steps_added") is True:
        errors.append("the frozen attack was changed")
    if progress.get("objective_changed") is True:
        errors.append("the objective changed")
    if progress.get("h_star_fitted") is True:
        errors.append("the attack run fitted H*")
    if progress.get("held_out_scenes_opened") is True:
        errors.append("a held-out scene was opened")
    if progress.get("float32_loaded") is True:
        errors.append("float32 weights were loaded")
    return errors


def scored_blocks(scene: dict[str, Any]) -> list[dict[str, Any]]:
    epsilon_block = (scene.get("epsilons") or {}).get(EPSILON_KEY) or {}
    blocks: list[dict[str, Any]] = []
    pgd = epsilon_block.get("pgd") or {}
    if isinstance(pgd, dict):
        blocks.extend(block for block in pgd.values() if isinstance(block, dict))
    for name in ("fgsm", "random"):
        block = epsilon_block.get(name)
        if isinstance(block, dict):
            blocks.append(block)
    return blocks


def record_bound_errors(scenes: dict[str, Any], epsilon_rgb: float) -> list[str]:
    errors: list[str] = []
    for scene_id, scene in scenes.items():
        if not isinstance(scene, dict) or scene.get("finished") is not True:
            continue
        for block in scored_blocks(scene):
            for item in bound_errors(
                block.get("linf_before_round_trip"),
                block.get("linf_after_round_trip"),
                epsilon_rgb,
            ):
                errors.append(f"{scene_id}: {item}")
    return errors


def record_decision_errors(scenes: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for scene_id, scene in scenes.items():
        if not isinstance(scene, dict):
            continue
        clean = scene.get("clean") or {}
        if isinstance(clean, dict):
            for context, view in clean.items():
                if isinstance(view, dict):
                    for item in decision_errors(view.get("decision")):
                        errors.append(f"{scene_id} clean {context}: {item}")
        for block in scored_blocks(scene):
            perturbed = block.get("perturbed") or {}
            if not isinstance(perturbed, dict):
                continue
            for context, view in perturbed.items():
                if isinstance(view, dict):
                    for item in decision_errors(view.get("decision")):
                        errors.append(f"{scene_id} {context}: {item}")
    return errors


def memory_line_errors(progress: dict[str, Any]) -> list[str]:
    decision = progress.get("budget_recheck_immediately_before_load") or {}
    bf16 = (decision.get("bfloat16") or {}) if isinstance(decision, dict) else {}
    fp32 = (decision.get("float32") or {}) if isinstance(decision, dict) else {}
    errors: list[str] = []
    if int(bf16.get("required_available_bytes") or 0) != BF16_REQUIRED_AVAILABLE_BYTES:
        errors.append("the recorded bfloat16 line is not 11,509,447,200")
    if int(fp32.get("required_available_bytes") or 0) != FLOAT32_REQUIRED_AVAILABLE_BYTES:
        errors.append("the recorded float32 line is not 20,871,410,752")
    floor = ((bf16.get("lines_bytes") or {}) if isinstance(bf16, dict) else {}).get("os_floor")
    if int(floor or 0) != OS_FLOOR_BYTES:
        errors.append("the recorded 6 GiB floor is not 6,442,450,944")
    for entry in progress.get("budget_decision_history") or []:
        if isinstance(entry, dict) and entry.get("guard_lowered") is True:
            errors.append("a budget history entry lowered the guard")
    return errors


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def expected_epsilon_rgb() -> float:
    return epsilon()
