"""Make ``study2_audit`` importable from a checkout without an editable install."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "src" / "study2_audit"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
