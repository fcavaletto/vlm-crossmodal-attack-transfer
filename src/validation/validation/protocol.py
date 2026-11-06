"""Locked Checkpoint 1.5 validation choices.

These constants are the code form of ``docs/checkpoint-1.5/VALIDATION_PROTOCOL.md``.
They were fixed before any model call on ``fixtures/development_v2``. They do
not select a scene, a family, a parser, or a prompt from a v2 accuracy.
"""

from __future__ import annotations

from pathlib import Path

LABEL = "DEVELOPMENT ONLY"
MEASUREMENT_KIND = "checkpoint1_5_v2_validation"
GATE_SCENE_ID = "dev2-attribute-00-a"
MANIFEST_RELATIVE = Path("fixtures/development_v2/manifest.jsonl")
MANIFEST_SHA256 = "b62ab1171a0aec7b2f264615a2961b6d384d00a1a4b17977d619ea27481154e1"
PROTOCOL_RELATIVE = Path("docs/checkpoint-1.5/VALIDATION_PROTOCOL.md")
OUTPUT_RELATIVE = Path("docs/checkpoint-1.5/development_v2")
CHECKPOINT1_RELATIVE = Path("docs/checkpoint-1")

N_SCENES = 72
N_GROUPS = 36
N_PER_FAMILY = 24
N_CONTEXTS = 4

CONTEXTS: tuple[str, ...] = (
    "neutral",
    "helpful_nonrevealing",
    "contradictory",
    "answer_revealing",
)
PRIMARY_CONTEXTS: tuple[str, ...] = ("neutral", "helpful_nonrevealing")
TASK_FAMILIES: tuple[str, ...] = ("attribute", "counting", "spatial")

# "About 70%" as a descriptive band. Not a deletion rule.
DESIGN_TARGET_LOW = 0.60
DESIGN_TARGET_HIGH = 0.80
# Below one half in either primary context: recommend drop or replace.
# The rows stay in the table.
UNSUITABLE_BELOW = 0.50

ALGEBRAIC_MIN_GROUPS = 5
COVERAGE_DESCRIPTION_GROUPS = 24

MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
MODEL_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
GRADIENT_LOSS = "canonical_word"
OFFICIAL_PARSER = "checkpoint1_strict_d4"
FORMAT_LINE = "Reply using only the answer. Do not write a sentence."
HELPFUL_TEXT_ROLE = "task guidance, not additional factual scene evidence"

IMAGE_HEIGHT = 384
IMAGE_WIDTH = 512


def repo_root() -> Path:
    """Repository root. This file lives at ``src/validation/validation``."""
    return Path(__file__).resolve().parents[3]


def assert_not_checkpoint1(path: Path, root: Path | None = None) -> None:
    """Refuse any output path that would replace Checkpoint 1 files."""
    root = repo_root() if root is None else root
    resolved = path.resolve()
    forbidden = (root / CHECKPOINT1_RELATIVE).resolve()
    if resolved == forbidden or forbidden in resolved.parents or resolved.is_relative_to(forbidden):
        raise RuntimeError(f"Refusing to write under {forbidden}.")
    if "docs/checkpoint-1" in resolved.as_posix() and "docs/checkpoint-1.5" not in resolved.as_posix():
        raise RuntimeError(f"Refusing output path {resolved}.")
