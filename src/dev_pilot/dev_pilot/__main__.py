"""``python -m dev_pilot --model smolvlm2`` scores one model."""

from __future__ import annotations

import sys

from .run import main

if __name__ == "__main__":
    sys.exit(main())
