"""Load the committed development manifest and resolve its images."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scene_gen.paths import resolve_image_path

from .constants import EXPECTED_SCENE_IDS, MANIFEST_SHA256, N_SCENES


def repo_root() -> Path:
    """Repository root. This file lives at ``src/pilot/pilot``."""
    return Path(__file__).resolve().parents[3]


def load_committed_manifest(root: Path | None = None) -> tuple[Path, list[dict]]:
    """Read the 24 development scenes. Any other manifest is refused."""
    repo = repo_root() if root is None else Path(root)
    path = repo / "fixtures" / "development" / "manifest.jsonl"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != MANIFEST_SHA256:
        raise RuntimeError(
            "Refusing to score a manifest whose sha256 is not the committed development sample. "
            f"Got {digest}."
        )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row["scene_id"] for row in rows]
    if len(rows) != N_SCENES or set(ids) != set(EXPECTED_SCENE_IDS) or len(ids) != len(set(ids)):
        raise RuntimeError(f"Expected the 24 committed development scene ids, found {len(rows)} rows.")
    for row in rows:
        if row.get("split") != "development":
            raise RuntimeError(f"{row.get('scene_id')}: refusing a scene that is not in the development split.")
        image = resolve_image_path(path, row["image_path"])
        if not image.is_file():
            raise RuntimeError(f"{row['scene_id']}: image is missing at {image}")
    rows.sort(key=lambda row: row["scene_id"])
    return path, rows
