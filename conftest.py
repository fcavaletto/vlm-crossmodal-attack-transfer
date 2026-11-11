"""Repository-level pytest setup. No model, no network.

Every package lives in its own ``src/<name>/`` directory. This puts each of them on
``sys.path`` and on ``PYTHONPATH`` (some tests start ``python -m <package>`` in a
subprocess), so ``python -m pytest`` from the repository root runs every suite in
one call without a hand-built ``PYTHONPATH``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SRC = sorted(str(path) for path in (Path(__file__).resolve().parent / "src").iterdir() if path.is_dir())

for _entry in reversed(_SRC):
    if _entry not in sys.path:
        sys.path.insert(0, _entry)

_existing = [entry for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep) if entry]
os.environ["PYTHONPATH"] = os.pathsep.join(_SRC + [entry for entry in _existing if entry not in _SRC])
