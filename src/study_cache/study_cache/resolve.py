"""Validate the opt-in cache path and point the existing guards at it. No model import here."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Mapping

OVERRIDE_ENV = "VLM_STUDY_MODEL_CACHE"
GUARD_NAME = "REQUIRED_HF_HOME"

# Every module that holds the study cache path as a module attribute and compares HF_HOME to it.
GUARDED_MODULES = ("qwen_rgb.pins", "qwen_rgb.measure", "dev_pilot.lock", "dev_pilot.run")


def resolve_override(environ: Mapping[str, str]) -> Path | None:
    """The validated override directory, or None when the override was not asked for."""
    value = environ.get(OVERRIDE_ENV)
    if value is None:
        return None
    if not value.strip():
        raise SystemExit(f"{OVERRIDE_ENV} is set but empty.")
    path = Path(value)
    if not path.is_absolute():
        raise SystemExit(f"{OVERRIDE_ENV} must be an absolute path, got {value}.")
    if not path.is_dir():
        raise SystemExit(f"{OVERRIDE_ENV} is {value}, which is not an existing directory.")
    home = environ.get("HF_HOME")
    if home is not None and Path(home).resolve() != path.resolve():
        raise SystemExit(f"HF_HOME is {home} but {OVERRIDE_ENV} is {value}. Refusing an ambiguous cache.")
    return path


def apply_override(path: Path, modules: tuple[str, ...] = GUARDED_MODULES) -> dict[str, Any]:
    """Point each guard at ``path`` for this process. Returns what was replaced."""
    # Import everything first: a module imported after a patch would copy the patched value.
    loaded = [importlib.import_module(name) for name in modules]
    replaced: dict[str, str] = {}
    for name, module in zip(modules, loaded):
        if not hasattr(module, GUARD_NAME):
            raise SystemExit(f"{name} has no {GUARD_NAME}; the cache guard moved and this launcher is out of date.")
        replaced[name] = getattr(module, GUARD_NAME)
    defaults = sorted(set(replaced.values()))
    if len(defaults) != 1:
        raise SystemExit(f"The guarded modules disagree on the study cache: {defaults}.")
    for module in loaded:
        setattr(module, GUARD_NAME, str(path))
    os.environ["HF_HOME"] = str(path)
    record = {"model_cache_override": str(path), "model_cache_default": defaults[0], "guards_repointed": list(modules)}
    print(
        f"[study_cache] {OVERRIDE_ENV} is set: reading the pinned snapshot from {path} "
        f"instead of {defaults[0]}. Model id and revision are unchanged.",
        file=sys.stderr,
    )
    return record
