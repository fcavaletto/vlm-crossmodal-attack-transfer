"""python -m study_cache <module> [args...]  runs  python -m <module> [args...]."""

from __future__ import annotations

import os
import runpy
import sys

from .resolve import OVERRIDE_ENV, apply_override, resolve_override


def main(argv: list[str]) -> None:
    if not argv or argv[0] in {"-h", "--help"}:
        raise SystemExit(
            f"usage: [{OVERRIDE_ENV}=/abs/hf-cache] python -m study_cache <module> [args...]\n"
            f"Without {OVERRIDE_ENV} the module runs unchanged, with the original cache guard."
        )
    override = resolve_override(os.environ)
    if override is not None:
        apply_override(override)
    sys.argv = [argv[0], *argv[1:]]
    runpy.run_module(argv[0], run_name="__main__", alter_sys=True)


main(sys.argv[1:])
