"""Rules fixed before any development_v4 gradient exists.

Nothing here reads a gradient, a prediction score, or a model weight.
The numeric lines are the ones already used by Checkpoints 1.5 and 1.7.
"""

from __future__ import annotations

from pathlib import Path

from dev_pilot.lock import (
    EXPECTED_TOKEN_IDS,
    MANIFEST_SHA256,
    N_GROUPS,
    N_SCENES,
    PRIMARY_CONTEXTS,
    QWEN_ANSWER_PREFIX,
)
from qwen_rgb.finite_difference import BF16_RELATIVE_ERROR_CEILING, BF16_ULP_AT_ONE
from qwen_rgb.pins import MODEL_ID, PINNED_REVISION

MODEL_KEY = "qwen3-vl-2b"
LABEL = "DEVELOPMENT ONLY"
CHECKPOINT = "checkpoint-1.8"
STARTED_FROM_COMMIT = "b9a31a80533bfa52c973a92c06364a4d74b4e537"
OUTPUT_RELATIVE = Path("docs/checkpoint-1.8/qwen_rgb_loss")
PROTOCOL_RELATIVE = Path("docs/checkpoint-1.8/MEASUREMENT_PROTOCOL.md")
PREDICTION_RELATIVE = Path("docs/checkpoint-1.7/development_v4/qwen3-vl-2b/predictions.jsonl")
GIB_BYTES = 6 * 1024**3
SAME_ROW_LOSS_MAX_ABS = 1e-5

# The repeat and the appended-forward check use the documented BF16 ceiling.
# It was not chosen after a gradient existed.
STABILITY_RELATIVE_L2_CEILING = BF16_RELATIVE_ERROR_CEILING
APPENDED_FORWARD_RELATIVE_CEILING = BF16_RELATIVE_ERROR_CEILING

GATE_RULE_ID = "lexicographic-first-scene-per-labeled-color"
GATE_CONTEXT_COUNT = len(PRIMARY_CONTEXTS)
ALIGNMENT_SOURCE_FOR_ELIGIBILITY = "neutral"
ALIGNMENT_TARGET_FOR_ELIGIBILITY = "helpful_nonrevealing"
HISTOGRAM_EDGES = tuple(index / 5 for index in range(-5, 6))

TOKEN_IDS = EXPECTED_TOKEN_IDS[MODEL_KEY]
ANSWER_PREFIX = QWEN_ANSWER_PREFIX

REPRODUCTION_COMMAND = (
    "HF_HOME=/Users/fedecava/Development/models/huggingface "
    "PYTHONPATH=src/qwen_rgb_loss:src/qwen_rgb:src/choice_eval:src/vlm_iface:"
    "src/pilot:src/dev_pilot:src/analysis:src/scene_v4 "
    "/Users/fedecava/Development/_c0-work/venv-vlm/bin/python -m qwen_rgb_loss"
)


def output_dir(root: Path) -> Path:
    return root / OUTPUT_RELATIVE


def assert_unchanged_lines() -> None:
    """Refuse to start if a locked id or the 6 GiB floor moved."""
    if GIB_BYTES != 6 * 1024**3:
        raise RuntimeError("The 6 GiB floor is not 6 * 1024**3.")
    if TOKEN_IDS != {"red": (1151,), "blue": (12203,), "green": (13250,), "yellow": (27869,)}:
        raise RuntimeError("Qwen bare color ids do not match the Checkpoint 1.7 table.")
    if ANSWER_PREFIX != "<|im_end|>\n<|im_start|>assistant\n":
        raise RuntimeError("The Qwen answer prefix changed.")
    if PINNED_REVISION != "89644892e4d85e24eaac8bacfd4f463576704203":
        raise RuntimeError("The Qwen revision changed.")
    if MODEL_ID != "Qwen/Qwen3-VL-2B-Instruct":
        raise RuntimeError("The Qwen model id changed.")
    if MANIFEST_SHA256 != "0d9d592b6700acf610453e4f4adf639fc778b4efb24530f60deb01045440d597":
        raise RuntimeError("The development_v4 manifest hash changed.")
    if N_SCENES != 96 or N_GROUPS != 48:
        raise RuntimeError("The development_v4 scene count changed.")
    if PRIMARY_CONTEXTS != ("neutral", "helpful_nonrevealing"):
        raise RuntimeError("The primary contexts changed.")
    if BF16_ULP_AT_ONE != 2.0**-7 or BF16_RELATIVE_ERROR_CEILING != 0.5:
        raise RuntimeError("The documented BF16 finite-difference tolerance changed.")
    if not ANSWER_PREFIX.endswith("\n"):
        raise RuntimeError("The Qwen cue no longer ends in whitespace, so the bare-color rule would change.")
