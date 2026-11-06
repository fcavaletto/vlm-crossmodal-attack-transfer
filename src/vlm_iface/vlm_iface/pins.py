"""Pinned checkpoint for the Checkpoint 0 model interface.

Study 1 recorded the same Hugging Face revision in ``docs/freeze-config.md``.
This package loads that commit on purpose and checks the Hub sha at load time.
It does not follow the moving ``main`` branch.
"""

from __future__ import annotations

MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
PINNED_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"

# Candidate only. The loader refuses this id. See ``candidate.py``.
QWEN_CANDIDATE_ID = "Qwen/Qwen3-VL-2B-Instruct"

# Weight-only ceiling for a from_pretrained call on a 24 GB unified-memory Mac.
# Backward activations sit on top of this number, and the OS shares the same memory.
# 6 GiB of float32 weights is the abort line used at this checkpoint.
FLOAT32_WEIGHT_ABORT_BYTES = 6 * 1024**3

# Do not start a load when the OS reports less free memory than this.
MIN_AVAILABLE_BYTES = 6 * 1024**3
