"""Pinned InternVL3.5-1B facts. Read from the snapshot on 2026-11-10."""

from __future__ import annotations

MODEL_ID = "OpenGVLab/InternVL3_5-1B-HF"  # native transformers port; no remote code
PINNED_REVISION = "9191dbccf312b537016f041b25d61c72e7c5c9f3"
REQUIRED_HF_HOME = "/Users/fedecava/Development/models/huggingface"  # the study's model cache (dev_pilot.lock)

# preprocessor_config.json and processing_internvl.py at this revision.
TILE = 448
MIN_PATCHES = 1
MAX_PATCHES = 12
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
TOKENS_PER_TILE = 256
IMAGE_TOKEN = "<IMG_CONTEXT>"

# The processor's default (InternVLProcessor sets crop_to_patches=True): a 4:3 image becomes
# 12 tiles of 448 px plus a 448 px thumbnail. "single" is the image processor's own default
# (crop_to_patches=False): one 448 px resize. Both are recorded; neither replaces the other.
TILINGS = ("dynamic", "single")

ANSWER_PREFIX = "<|im_end|>\n<|im_start|>assistant\n"
