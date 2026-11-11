"""D3: image paths resolve from the manifest directory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scene_gen.paths import resolve_image_path


def test_committed_images_resolve_beside_the_manifest(repo_root: Path) -> None:
    manifest = repo_root / "fixtures" / "development" / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 24
    for record in rows:
        resolved = resolve_image_path(manifest, record["image_path"])
        assert resolved.is_file()
        assert resolved.parent.name == "images"
        assert resolved.parent.parent == manifest.parent.resolve()
        wrong_root = repo_root / "fixtures" / record["image_path"]
        assert not wrong_root.is_file()


def test_absolute_paths_and_parent_segments_are_rejected(repo_root: Path) -> None:
    manifest = repo_root / "fixtures" / "development" / "manifest.jsonl"
    with pytest.raises(ValueError, match="relative"):
        resolve_image_path(manifest, "/tmp/dev-attribute-00-a.png")
    with pytest.raises(ValueError, match="\\.\\."):
        resolve_image_path(manifest, "../development/images/dev-attribute-00-a.png")
    with pytest.raises(ValueError, match="manifest file"):
        resolve_image_path(manifest.parent, "images/dev-attribute-00-a.png")
