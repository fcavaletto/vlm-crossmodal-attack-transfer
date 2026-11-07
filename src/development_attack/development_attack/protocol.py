"""Rules fixed in the Checkpoint 2.1 analysis protocol before this attack.

Nothing here reads a live gradient, a margin, or an alignment.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

CHECKPOINT = "checkpoint-2.1"
LABEL = "DEVELOPMENT ONLY"
STARTED_FROM_COMMIT = "481d7d9b9a18e832aad65c78a120b215e50010d0"
PROTOCOL_RELATIVE = Path("docs/checkpoint-2.1/ANALYSIS_PROTOCOL.md")
OUTPUT_RELATIVE = Path("docs/checkpoint-2.1/development")
REPORT_RELATIVE = Path("docs/checkpoint-2.1/DEVELOPMENT_REPORT.md")
PRIOR_PROGRESS_RELATIVE = Path("docs/checkpoint-1.8/qwen_rgb_loss/progress.json")
CALIBRATION_RECORDS_RELATIVE = Path("docs/checkpoint-2.0/calibration/records.jsonl")

SAMPLE_RULE = "all-development-v4-scenes"
SOURCE_CONTEXT = "neutral"
TARGET_CONTEXT = "helpful_nonrevealing"
EVAL_CONTEXTS = (SOURCE_CONTEXT, TARGET_CONTEXT)
EPSILON_KEY = "1/255"
EPSILON_NUMERATOR = 1.0
PGD_STEPS = 7
PRIMARY_STEPS = 3
COMPARISON_STEPS = 7
RECORDED_STEPS = (PRIMARY_STEPS, COMPARISON_STEPS)
STEP_DENOMINATOR = 4
SEED_PREFIX = "checkpoint-2.0-calibration"
N_SCENES = 96
N_GROUPS = 48
COLORS = ("red", "blue", "green", "yellow")

# Memory lines named by the analysis protocol. A later run does not lower them.
BF16_REQUIRED_AVAILABLE_BYTES = 11_509_447_200
FLOAT32_REQUIRED_AVAILABLE_BYTES = 20_871_410_752
OS_FLOOR_BYTES = 6_442_450_944

# Effective-sample floors from the analysis protocol.
N_REGRESSORS = 4
MIN_EVENTS = 40
N_PARAMETERS = 5
MIN_GROUPS = 6
BOOTSTRAP_SEED = 20261109
WILD_DRAWS = 999
RATE_DRAWS = 4000
RATE_ALPHA = 0.05
HSTAR_UNESTIMATED = "H* cannot be meaningfully estimated."

# Float32 noise around one 8-bit level. Half a level is the round-trip slack.
BOUND_ABSOLUTE_SLACK = 1e-5

PRESERVED_PATHS = (
    "docs/checkpoint-1.5/run_model_feasibility.py",
    "docs/checkpoint-1.7",
    "docs/checkpoint-1.8",
    "docs/checkpoint-1.9",
    "docs/checkpoint-2.0",
    "docs/checkpoint-2.1/ANALYSIS_PROTOCOL.md",
    "src/attack_calibration",
    "src/attack_feasibility",
    "src/qwen_rgb",
    "src/qwen_rgb_loss",
    "src/choice_eval",
    "src/dev_pilot",
    "src/analysis",
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
        "stopped_checksums",
        "stopped_memory_guard",
        "stopped_bounds",
        "stopped_decision",
        "completed",
    }
)

REPRODUCTION_COMMAND = (
    "HF_HOME=/Users/fedecava/Development/models/huggingface "
    "PYTHONPATH=src/development_attack:src/attack_calibration:src/attack_feasibility:"
    "src/qwen_rgb_loss:src/qwen_rgb:src/choice_eval:src/vlm_iface:src/pilot:"
    "src/dev_pilot:src/analysis:src/scene_v4 "
    "/Users/fedecava/Development/_c0-work/venv-vlm/bin/python -m development_attack"
)


class DevelopmentStop(RuntimeError):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status


def epsilon(numerator: float = EPSILON_NUMERATOR) -> float:
    """RGB radius on the [0, 1] scale. The numerator is in units of 1/255."""
    return float(numerator) / 255.0


def step_size() -> float:
    return epsilon() / float(STEP_DENOMINATOR)


def configurations() -> tuple[dict[str, object], ...]:
    """Primary PGD at step 3, then the three pre-specified comparisons."""
    rgb = epsilon()
    return (
        {
            "method": "pgd",
            "epsilon": EPSILON_KEY,
            "epsilon_rgb": rgb,
            "steps": PRIMARY_STEPS,
            "role": "primary",
        },
        {
            "method": "fgsm",
            "epsilon": EPSILON_KEY,
            "epsilon_rgb": rgb,
            "steps": 1,
            "role": "comparison",
        },
        {
            "method": "random",
            "epsilon": EPSILON_KEY,
            "epsilon_rgb": rgb,
            "steps": 1,
            "role": "comparison",
        },
        {
            "method": "pgd",
            "epsilon": EPSILON_KEY,
            "epsilon_rgb": rgb,
            "steps": COMPARISON_STEPS,
            "role": "comparison",
        },
    )


def output_dir(root: Path) -> Path:
    return root / OUTPUT_RELATIVE


def assert_protocol_committed(root: Path) -> str:
    """Refuse a perturbation when the analysis protocol is absent or dirty."""
    relative = str(PROTOCOL_RELATIVE)
    porcelain = subprocess.check_output(
        ["git", "status", "--porcelain", "--", relative],
        cwd=root,
        text=True,
    )
    if porcelain.strip():
        raise DevelopmentStop(
            "stopped_protocol",
            "The analysis protocol is not a clean committed file. No perturbation was run.",
        )
    commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", relative],
        cwd=root,
        text=True,
    ).strip()
    if commit != STARTED_FROM_COMMIT:
        raise DevelopmentStop(
            "stopped_protocol",
            "The analysis protocol commit is not "
            f"{STARTED_FROM_COMMIT}. No perturbation was run.",
        )
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root,
        check=False,
    )
    if ancestor.returncode != 0:
        raise DevelopmentStop(
            "stopped_protocol",
            "The analysis protocol commit is not in HEAD. No perturbation was run.",
        )
    return commit
