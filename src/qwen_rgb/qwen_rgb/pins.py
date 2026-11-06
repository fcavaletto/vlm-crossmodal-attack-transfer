"""Pinned Qwen3-VL-2B processor facts for the RGB gradient check.

The revision is the Checkpoint 1.6 candidate. This package does not adopt the
model. The pixel budget below is the Checkpoint 1.5 function; nothing here
replaces those byte lines.
"""

from __future__ import annotations

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
PINNED_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"

# One development scene. The fixture answer is a color word.
SCENE_ID = "dev2-attribute-00-a"
CONTEXT_KEY = "neutral"
REQUIRED_HF_HOME = "/Users/fedecava/Development/models/huggingface"

# preprocessor_config.json at PINNED_REVISION.
PATCH_SIZE = 16
TEMPORAL_PATCH_SIZE = 2
MERGE_SIZE = 2
IMAGE_MEAN = (0.5, 0.5, 0.5)
IMAGE_STD = (0.5, 0.5, 0.5)
MIN_PIXELS = 65536
MAX_PIXELS = 16_777_216
RESAMPLE_BICUBIC = 3

# A larger gap means the differentiable tensor is not the official model input.
# This is a correctness stop, not a memory line. 1e-3 is above float32 noise
# and below the uint8-resize gap measured on images that actually change size.
OFFICIAL_MATCH_MAX_ABS = 1e-3
