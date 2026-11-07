"""Path setup. These tests do not load Qwen."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for relative in (
    "src/cuda_validation",
    "src/cuda_readiness",
    "src/attack_feasibility",
    "src/qwen_rgb",
    "src/qwen_rgb_loss",
    "src/choice_eval",
    "src/vlm_iface",
    "src/pilot",
    "src/dev_pilot",
):
    entry = str(ROOT / relative)
    if entry not in sys.path:
        sys.path.insert(0, entry)
