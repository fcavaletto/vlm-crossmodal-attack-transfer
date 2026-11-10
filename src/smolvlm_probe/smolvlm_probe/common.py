"""Shared setup: cache path, development scenes, output store. Refuses non-development data."""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

from .pins import MODEL_ID, PINNED_REVISION, REQUIRED_HF_HOME

OUT_RELATIVE = Path("docs/checkpoint-6-smolvlm2/stage0")


class Stop(RuntimeError):
    pass


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for c in here.parents:
        if (c / "fixtures" / "development_v4").is_dir() and (c / "src").is_dir():
            return c
    raise Stop("repository root not found")


def hf_home() -> str:
    """The study cache, or an explicit opt-in override (same variable as ``study_cache``)."""
    return os.environ.get("VLM_STUDY_MODEL_CACHE", REQUIRED_HF_HOME)


def snapshot_dir() -> Path:
    path = Path(hf_home()) / "hub" / ("models--" + MODEL_ID.replace("/", "--")) / "snapshots" / PINNED_REVISION
    if not path.is_dir():
        raise Stop(f"pinned snapshot not found at {path}")
    return path


def load_processor():
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(str(snapshot_dir()))


def development_scenes(root: Path) -> list[dict[str, Any]]:
    """The 96 hash-checked development_v4 rows. The confirmation split is never read."""
    from dev_pilot.fixtures import load_manifest

    rows = load_manifest(root)
    if any(r.get("split") != "development" for r in rows):
        raise Stop("a non-development row was loaded")
    return rows


def s16(rows: list[dict[str, Any]]) -> list[str]:
    """The Checkpoint 2.3 / 3 subset of 16 development scenes (same rule as path_probe.run)."""
    from path_probe.run import select_scenes

    return select_scenes(rows)[0]


def scene_image(root: Path, row: dict[str, Any]):
    from attack_calibration.scene import _development_image

    return _development_image(root, row)


def git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def json_ready(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): json_ready(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [json_ready(x) for x in v]
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def write_json(root: Path, name: str, obj: Any) -> Path:
    out = root / OUT_RELATIVE
    out.mkdir(parents=True, exist_ok=True)
    gi = out / ".gitignore"
    if not gi.is_file():
        gi.write_text("tensors/\n*.log\n")
    path = out / name
    path.write_text(json.dumps(json_ready(obj), indent=1, sort_keys=True, allow_nan=False) + "\n")
    return path


def verify_weights() -> dict[str, str]:
    """sha256 of each pinned safetensors shard, checked against the repository's LFS ids."""
    import hashlib

    from .pins import WEIGHT_SHA256

    got = {}
    for name, want in WEIGHT_SHA256.items():
        h = hashlib.sha256()
        with (snapshot_dir() / name).open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b""):
                h.update(chunk)
        got[name] = h.hexdigest()
        if got[name] != want:
            raise Stop(f"{name}: sha256 {got[name]} != pinned {want}")
    return got
