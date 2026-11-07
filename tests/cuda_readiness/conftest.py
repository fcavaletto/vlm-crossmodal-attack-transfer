"""Path setup for the readiness tests. These tests do not load Qwen."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for relative in ("src/cuda_readiness", "src/dev_pilot", "src/qwen_rgb"):
    entry = str(ROOT / relative)
    if entry not in sys.path:
        sys.path.insert(0, entry)
