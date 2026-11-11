"""Rules fixed in ATTACK_PROTOCOL.md before any perturbation.

Nothing here reads a live gradient or a model weight.
"""

from __future__ import annotations

from pathlib import Path

CHECKPOINT = "checkpoint-1.9"
LABEL = "DEVELOPMENT ONLY"
STARTED_FROM_COMMIT = "ebe3db8ec300303affdaebadeb280ccba439e97a"
PROTOCOL_RELATIVE = Path("docs/checkpoint-1.9/ATTACK_PROTOCOL.md")
OUTPUT_RELATIVE = Path("docs/checkpoint-1.9/attack_feasibility")
FIGURE_RELATIVE = Path("docs/checkpoint-1.9/figures/illustration.png")
PRIOR_PROGRESS_RELATIVE = Path("docs/checkpoint-1.8/qwen_rgb_loss/progress.json")
PREDICTION_RELATIVE = Path("docs/checkpoint-1.7/development_v4/qwen3-vl-2b/predictions.jsonl")

STAGE_A_RULE_ID = "per-color-extreme-alignment-exploratory"
STAGE_B_RULE_ID = "scene-id-stride-6"
STAGE_A_SAMPLE_LABEL = "exploratory"
STAGE_A_CONTEXT = "neutral"
STAGE_B_SOURCE_CONTEXT = "neutral"
STAGE_B_EVAL_CONTEXTS = ("neutral", "helpful_nonrevealing")

# (scene_id, labeled color, stratum, stored A(neutral, helpful) on the 1.8 record)
STAGE_A_SCENES = (
    ("dev4-ref-01-a", "red", "lowest_alignment_within_color", 0.2738961986673083),
    ("dev4-ref-09-b", "red", "highest_alignment_within_color", 0.9345366800860538),
    ("dev4-ref-25-a", "blue", "lowest_alignment_within_color", 0.4940414745458451),
    ("dev4-ref-21-b", "blue", "highest_alignment_within_color", 0.9522294672411216),
    ("dev4-ref-30-b", "green", "lowest_alignment_within_color", 0.5875956154399549),
    ("dev4-ref-07-a", "green", "highest_alignment_within_color", 0.9775420055497965),
    ("dev4-ref-17-b", "yellow", "lowest_alignment_within_color", 0.19669284166797163),
    ("dev4-ref-45-b", "yellow", "highest_alignment_within_color", 0.9753625290227933),
)

STAGE_B_SCENE_IDS = (
    "dev4-ref-00-a",
    "dev4-ref-03-a",
    "dev4-ref-06-a",
    "dev4-ref-09-a",
    "dev4-ref-12-a",
    "dev4-ref-15-a",
    "dev4-ref-18-a",
    "dev4-ref-21-a",
    "dev4-ref-24-a",
    "dev4-ref-27-a",
    "dev4-ref-30-a",
    "dev4-ref-33-a",
    "dev4-ref-36-a",
    "dev4-ref-39-a",
    "dev4-ref-42-a",
    "dev4-ref-45-a",
)

STAGE_A_BUDGETS = (
    ("0.5/255", 0.5),
    ("1/255", 1.0),
    ("2/255", 2.0),
    ("4/255", 4.0),
)
STAGE_B_BUDGETS = (
    ("1/255", 1.0),
    ("2/255", 2.0),
    ("4/255", 4.0),
)
STAGE_A_MIN_SCENES = 6
STAGE_A_SCENE_COUNT = 8
PGD_STEPS = 7
DISPLAY_GAIN = 32
STAGE_A_ILLUSTRATION = ("dev4-ref-01-a", "4/255", "positive")
STAGE_B_ILLUSTRATION = ("dev4-ref-00-a", "4/255", "pgd")
STAGE_A_SEED_PREFIX = "checkpoint-1.9-stage-a"
STAGE_B_SEED_PREFIX = "checkpoint-1.9-stage-b"
COLOR_ORDER = ("red", "blue", "green", "yellow")

TERMINAL_STATUSES = frozenset(
    {
        "stopped_budget",
        "stopped_oom",
        "stopped_protocol",
        "stopped_resident",
        "stopped_revision",
        "stopped_tokenization",
        "stopped_prompt",
        "stopped_loss_identity",
        "stopped_scene_rule",
        "stopped_grid",
        "stopped_stage_a",
        "stopped_missing_weights",
        "stopped_unexpected_header",
        "stopped_unexpected_load",
        "completed",
        "error",
    }
)


def epsilon(numerator: float) -> float:
    """RGB step on the [0, 1] scale. The numerator is in units of 1/255."""
    return float(numerator) / 255.0


def output_dir(root: Path) -> Path:
    return root / OUTPUT_RELATIVE


def figure_path(root: Path) -> Path:
    return root / FIGURE_RELATIVE
