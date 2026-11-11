"""Fabricated H* packet used to check the estimator arithmetic.

The rows are constructed. They are not model outputs. The included PGD
block is an exact linear function of alignment, so the wild-bootstrap
interval collapses to the OLS point. That collapse is intentional: a
reviewer can check β_A = 1/2 by hand. The precision simulation is the
check that uses residual noise.

Cell labels ``epsilon=0.01`` and ``step_budget=2`` are bookkeeping. They
are not a chosen attack budget.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PACKET_FILENAME = "packet.jsonl"

# Six groups, two of each task family. Two ordered directions per scene.
# Alignment is -1 on neutral → helpful and +1 on helpful → neutral.
# Strict error equals 1 only on the +1 direction for pgd and fgsm.
# Noise flips that pattern, so its alignment coefficient is -1/2.
GROUP_FAMILIES = (
    ("fab-g0", "attribute"),
    ("fab-g1", "attribute"),
    ("fab-g2", "counting"),
    ("fab-g3", "counting"),
    ("fab-g4", "spatial"),
    ("fab-g5", "spatial"),
)
DIRECTIONS = (
    ("neutral", "helpful_nonrevealing", -1.0, False),
    ("helpful_nonrevealing", "neutral", 1.0, True),
)
METHODS = ("pgd", "fgsm", "noise")


def fixture_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "hstar"


def build_worked_packet() -> list[dict[str, Any]]:
    """Return the fabricated rows in a stable order."""
    rows: list[dict[str, Any]] = []
    for group_id, family in GROUP_FAMILIES:
        scene_id = f"{group_id}-s0"
        for source, target, alignment, strict_error in DIRECTIONS:
            for method in METHODS:
                flipped = not strict_error if method == "noise" else strict_error
                rows.append(
                    _row(
                        scene_id=scene_id,
                        group_id=group_id,
                        family=family,
                        source=source,
                        target=target,
                        method=method,
                        alignment=alignment,
                        strict_error=flipped,
                    )
                )
    rows.extend(_excluded_rows())
    return rows


def write_packet(directory: Path | None = None) -> Path:
    root = Path(directory) if directory is not None else fixture_dir()
    root.mkdir(parents=True, exist_ok=True)
    path = root / PACKET_FILENAME
    lines = [
        json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":"))
        for row in build_worked_packet()
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def load_packet(directory: Path | None = None) -> list[dict[str, Any]]:
    root = Path(directory) if directory is not None else fixture_dir()
    path = root / PACKET_FILENAME
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _row(
    *,
    scene_id: str,
    group_id: str,
    family: str,
    source: str,
    target: str,
    method: str,
    alignment: float | None,
    strict_error: bool,
    **overrides: Any,
) -> dict[str, Any]:
    invalid = False
    valid_wrong = strict_error
    row: dict[str, Any] = {
        "data_origin": "fabricated",
        "scene_id": scene_id,
        "counterfactual_group_id": group_id,
        "task_family": family,
        "source_context": source,
        "target_context": target,
        "method": method,
        "epsilon": 0.01,
        "step_budget": 2,
        "alignment_a": alignment,
        "clean_correct_source": True,
        "clean_correct_target": True,
        "source_reveals_answer": False,
        "target_reveals_answer": False,
        "source_contains_answer": False,
        "target_contains_answer": False,
        "primary_contexts_invariant": True,
        "ambiguity_flagged": False,
        "gradients_finite": True,
        "gradient_source_l1": 4.0,
        "gradient_target_l1": 4.0,
        "clean_confidence": 0.6,
        "strict_error": strict_error,
        "invalid_target": invalid,
        "valid_wrong_target": valid_wrong,
        "source_strict_error": True,
    }
    row.update(overrides)
    if row["strict_error"]:
        if row["invalid_target"] == row["valid_wrong_target"]:
            row["invalid_target"] = False
            row["valid_wrong_target"] = True
    else:
        row["invalid_target"] = False
        row["valid_wrong_target"] = False
    return row


def _excluded_rows() -> list[dict[str, Any]]:
    """One primary-context row for each inclusion gate.

    Outcomes on these rows are schema-valid and are not used by inclusion.
    """
    specs: list[tuple[str, dict[str, Any]]] = [
        ("fab-ex-source-clean", {"clean_correct_source": False}),
        ("fab-ex-target-clean", {"clean_correct_target": False}),
        ("fab-ex-contains", {"source_contains_answer": True}),
        ("fab-ex-reveals", {"target_reveals_answer": True}),
        ("fab-ex-invariant", {"primary_contexts_invariant": False}),
        ("fab-ex-ambiguity", {"ambiguity_flagged": True}),
        ("fab-ex-zero-gradient", {"gradient_target_l1": 0.0, "alignment_a": None}),
        ("fab-ex-nonfinite", {"gradients_finite": False}),
        ("fab-ex-undefined-alignment", {"alignment_a": None}),
        (
            "fab-ex-context",
            {"source_context": "contradictory", "target_context": "neutral"},
        ),
    ]
    rows = []
    for scene_id, overrides in specs:
        rows.append(
            _row(
                scene_id=scene_id,
                group_id=f"group-{scene_id}",
                family="attribute",
                source="neutral",
                target="helpful_nonrevealing",
                method="pgd",
                alignment=-1.0,
                strict_error=False,
                **overrides,
            )
        )
    return rows
