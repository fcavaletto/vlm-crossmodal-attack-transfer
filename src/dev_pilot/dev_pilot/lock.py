"""Constants fixed by docs/checkpoint-1.7/DEVELOPMENT_PROTOCOL.md.

This module does not load a model and does not define a memory floor. The
floors stay in ``vlm_iface.pins`` and the Checkpoint 1.5 budget script.
"""

from __future__ import annotations

MANIFEST_RELATIVE = "fixtures/development_v4/manifest.jsonl"
CHECKSUMS_RELATIVE = "fixtures/development_v4/checksums.json"
PROTOCOL_RELATIVE = "docs/checkpoint-1.7/DEVELOPMENT_PROTOCOL.md"
OUTPUT_RELATIVE = "docs/checkpoint-1.7/development_v4"
MANIFEST_SHA256 = "0d9d592b6700acf610453e4f4adf639fc778b4efb24530f60deb01045440d597"

PRIMARY_CONTEXTS = ("neutral", "helpful_nonrevealing")
EXTRA_CONTEXTS = ("contradictory", "answer_revealing")
N_SCENES = 96
N_GROUPS = 48
COMPETENCE_MIN_NUMERATOR = 13
COMPETENCE_MIN_DENOMINATOR = 20
CEILING_NUMERATOR = 4
CEILING_DENOMINATOR = 5
MAX_NEW_TOKENS = 16
DO_IMAGE_SPLITTING = False
LABEL = "DEVELOPMENT ONLY"

INSTRUCTION = "Reply with one of these colors: red, blue, green, yellow."
SMOL_ANSWER_PREFIX = "<end_of_utterance>\nAssistant:"
QWEN_ANSWER_PREFIX = "<|im_end|>\n<|im_start|>assistant\n"
ANSWER_PREFIX = {
    "smolvlm2": SMOL_ANSWER_PREFIX,
    "qwen3-vl-2b": QWEN_ANSWER_PREFIX,
}
EXPECTED_TOKEN_IDS = {
    "smolvlm2": {"red": (2382,), "blue": (4461,), "green": (2654,), "yellow": (5724,)},
    "qwen3-vl-2b": {"red": (1151,), "blue": (12203,), "green": (13250,), "yellow": (27869,)},
}
PROBE_CONTEXT = "This sentence exists only to read the generation cue."
PROBE_QUESTION = "What color is the marker?"
REQUIRED_HF_HOME = "/Users/fedecava/Development/models/huggingface"
BASE_COMMIT = "004783b5604c34aa066befbcd5356bfd7bd6e7b2"

PREDICTION_BLOBS = (
    "docs/checkpoint-1.7/development_v4/smolvlm2/predictions.jsonl",
    "docs/checkpoint-1.7/development_v4/qwen3-vl-2b/predictions.jsonl",
)

FORBIDDEN_ROW_KEYS = frozenset(
    {
        "logits",
        "pixel_values",
        "pixel_attention_mask",
        "image_grid_thw",
        "inputs_embeds",
        "state_dict",
        "weights",
    }
)
