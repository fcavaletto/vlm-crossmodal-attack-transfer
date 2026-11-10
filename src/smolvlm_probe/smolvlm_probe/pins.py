"""Pinned SmolVLM2-2.2B facts. Read from the repository at the pinned revision on 2026-11-10."""

from __future__ import annotations

MODEL_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
PINNED_REVISION = "482adb537c021c86670beed01cd58990d01e72e4"
REQUIRED_HF_HOME = "/Users/fedecava/Development/models/huggingface"  # the study's model cache (dev_pilot.lock)
WEIGHT_SHA256 = {  # LFS object ids of the two safetensors shards at the pinned revision
    "model-00001-of-00002.safetensors": "7c88a90e78f567460b2369a0369cc4dafbfda5607c9cb090b30fa4437b0282f2",
    "model-00002-of-00002.safetensors": "cf3d0958a8240f71054a6812038f2a6bfbe2e5124573a3b363f6b120c9a52657",
}

# preprocessor_config.json at this revision, and SmolVLMImageProcessor (torchvision backend) in
# transformers 5.22.1: uint8 Lanczos resize of the longest edge to 1536 (512x384 -> 1536x1152),
# then either 4x3 tiles of 384 plus a 384x384 global image (do_image_splitting=True, the default),
# or one 384x384 resize of the 1536x1152 image (do_image_splitting=False). Each resize is uint8 in,
# uint8 out. Normalisation: (x/255 - 0.5) / 0.5.
LONGEST_EDGE = 1536
TILE = 384
IMAGE_MEAN = (0.5, 0.5, 0.5)
IMAGE_STD = (0.5, 0.5, 0.5)
TOKENS_PER_TILE = 81
IMAGE_TOKEN = "<image>"

# "split" is the processor default; "single" is do_image_splitting=False (the setting the study
# used for SmolVLM2-500M in Checkpoints 1.6-1.7). Both are measured; neither replaces the other.
TILINGS = ("split", "single")

ANSWER_PREFIX = "<end_of_utterance>\nAssistant:"
EXPECTED_TOKEN_IDS = {"red": 2382, "blue": 4461, "green": 2654, "yellow": 5724}  # dev_pilot.lock, space-prefixed
