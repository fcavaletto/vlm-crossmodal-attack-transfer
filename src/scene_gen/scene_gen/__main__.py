"""``python -m scene_gen`` writes the development sample."""

from __future__ import annotations

import sys

from .generate import main

if __name__ == "__main__":
    sys.exit(main())
