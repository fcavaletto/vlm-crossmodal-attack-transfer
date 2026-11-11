"""Rules fixed in CALIBRATION_PROTOCOL.md before any perturbation.

Nothing here reads a live gradient, a margin, or an alignment.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

CHECKPOINT = "checkpoint-2.0"
LABEL = "DEVELOPMENT ONLY"
STARTED_FROM_COMMIT = "0cd2f2a8e566b6530dcfd4dd7021961b2fc17990"
PROTOCOL_RELATIVE = Path("docs/checkpoint-2.0/CALIBRATION_PROTOCOL.md")
OUTPUT_RELATIVE = Path("docs/checkpoint-2.0/calibration")
REPORT_RELATIVE = Path("docs/checkpoint-2.0/CALIBRATION_REPORT.md")
PRIOR_PROGRESS_RELATIVE = Path("docs/checkpoint-1.8/qwen_rgb_loss/progress.json")

RULE_ID = "counterfactual-group-stride-3"
SAMPLE_LABEL = "exploratory"
EXPECTED_GROUP_COUNT = 48
GROUP_INDEXES = tuple(range(0, 46, 3))
SOURCE_CONTEXT = "neutral"
TARGET_CONTEXT = "helpful_nonrevealing"
EVAL_CONTEXTS = (SOURCE_CONTEXT, TARGET_CONTEXT)
BUDGETS = (
    ("1/255", 1.0),
    ("2/255", 2.0),
)
PGD_STEPS = 7
RECORDED_STEPS = (2, 3, 4, 5, 7)
SEED_PREFIX = "checkpoint-2.0-calibration"
NEAR_EDGE = 1

# group id, in the stride order
FROZEN_GROUP_IDS = (
    "cf4-ref-00",
    "cf4-ref-03",
    "cf4-ref-06",
    "cf4-ref-09",
    "cf4-ref-12",
    "cf4-ref-15",
    "cf4-ref-18",
    "cf4-ref-21",
    "cf4-ref-24",
    "cf4-ref-27",
    "cf4-ref-30",
    "cf4-ref-33",
    "cf4-ref-36",
    "cf4-ref-39",
    "cf4-ref-42",
    "cf4-ref-45",
)

# scene id, group id, labeled color, random-sign seed
FROZEN_SCENES = (
    ("dev4-ref-00-a", "cf4-ref-00", "red", 1342106899),
    ("dev4-ref-00-b", "cf4-ref-00", "blue", 974576453),
    ("dev4-ref-03-a", "cf4-ref-03", "red", 131700877),
    ("dev4-ref-03-b", "cf4-ref-03", "blue", 79946813),
    ("dev4-ref-06-a", "cf4-ref-06", "blue", 1250892028),
    ("dev4-ref-06-b", "cf4-ref-06", "red", 2020182026),
    ("dev4-ref-09-a", "cf4-ref-09", "blue", 473034008),
    ("dev4-ref-09-b", "cf4-ref-09", "red", 771346531),
    ("dev4-ref-12-a", "cf4-ref-12", "blue", 2074605998),
    ("dev4-ref-12-b", "cf4-ref-12", "green", 1845135049),
    ("dev4-ref-15-a", "cf4-ref-15", "blue", 701364960),
    ("dev4-ref-15-b", "cf4-ref-15", "green", 2037154491),
    ("dev4-ref-18-a", "cf4-ref-18", "green", 708532238),
    ("dev4-ref-18-b", "cf4-ref-18", "blue", 1440290221),
    ("dev4-ref-21-a", "cf4-ref-21", "green", 666959696),
    ("dev4-ref-21-b", "cf4-ref-21", "blue", 578873171),
    ("dev4-ref-24-a", "cf4-ref-24", "green", 1598181378),
    ("dev4-ref-24-b", "cf4-ref-24", "yellow", 1679499347),
    ("dev4-ref-27-a", "cf4-ref-27", "green", 167202381),
    ("dev4-ref-27-b", "cf4-ref-27", "yellow", 1733500241),
    ("dev4-ref-30-a", "cf4-ref-30", "yellow", 2054639494),
    ("dev4-ref-30-b", "cf4-ref-30", "green", 2103363729),
    ("dev4-ref-33-a", "cf4-ref-33", "yellow", 1664993314),
    ("dev4-ref-33-b", "cf4-ref-33", "green", 1652986261),
    ("dev4-ref-36-a", "cf4-ref-36", "yellow", 1445696036),
    ("dev4-ref-36-b", "cf4-ref-36", "red", 1114441199),
    ("dev4-ref-39-a", "cf4-ref-39", "yellow", 755236030),
    ("dev4-ref-39-b", "cf4-ref-39", "red", 463988851),
    ("dev4-ref-42-a", "cf4-ref-42", "red", 1529302010),
    ("dev4-ref-42-b", "cf4-ref-42", "yellow", 659569850),
    ("dev4-ref-45-a", "cf4-ref-45", "red", 512862953),
    ("dev4-ref-45-b", "cf4-ref-45", "yellow", 234080788),
)

BLOCKING_STATUSES = frozenset(
    {
        "stopped_oom",
        "stopped_protocol",
        "stopped_loss_identity",
        "stopped_revision",
        "stopped_tokenization",
        "stopped_prompt",
        "stopped_scene_rule",
        "stopped_grid",
        "stopped_group_count",
        "stopped_missing_weights",
        "stopped_unexpected_header",
        "stopped_unexpected_load",
        "completed",
    }
)


class CalibrationStop(RuntimeError):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status


def epsilon(numerator: float) -> float:
    """RGB radius on the [0, 1] scale. The numerator is in units of 1/255."""
    return float(numerator) / 255.0


def configurations() -> tuple[dict[str, object], ...]:
    """The recorded grid. PGD steps 2, 3, 4, 5, and 7, then the two controls."""
    rows: list[dict[str, object]] = []
    for key, numerator in BUDGETS:
        rgb = epsilon(numerator)
        for step in RECORDED_STEPS:
            rows.append({"method": "pgd", "epsilon": key, "epsilon_rgb": rgb, "steps": step})
        for method in ("fgsm", "random"):
            rows.append({"method": method, "epsilon": key, "epsilon_rgb": rgb, "steps": 1})
    return tuple(rows)


def output_dir(root: Path) -> Path:
    return root / OUTPUT_RELATIVE


def assert_protocol_committed(root: Path) -> str:
    """Refuse a perturbation when the protocol file is absent or dirty."""
    relative = str(PROTOCOL_RELATIVE)
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain", "--", relative],
        cwd=root,
        text=True,
    )
    if porcelain.strip():
        raise CalibrationStop(
            "stopped_protocol",
            "The calibration protocol is not a clean committed file. No perturbation was run.",
        )
    commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", relative],
        cwd=root,
        text=True,
    ).strip()
    if not commit:
        raise CalibrationStop(
            "stopped_protocol",
            "The calibration protocol has no commit. No perturbation was run.",
        )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root,
        check=False,
    )
    if ancestor.returncode != 0:
        raise CalibrationStop(
            "stopped_protocol",
            "The calibration protocol commit is not in HEAD. No perturbation was run.",
        )
    return commit
