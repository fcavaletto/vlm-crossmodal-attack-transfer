"""Resolve a manifest ``image_path`` from the manifest directory.

The sample stores ``images/<scene_id>.png`` beside
``fixtures/development_v4/manifest.jsonl``.
"""

from __future__ import annotations

from pathlib import Path


def resolve_image_path(manifest_path: Path | str, image_path: str) -> Path:
    """Return the image file for one manifest row.

    Absolute paths, empty paths, and any ``.`` or ``..`` segment are rejected.
    The resolved file must stay inside the manifest directory.
    """
    if not isinstance(image_path, str) or image_path == "":
        raise ValueError("image_path must be a non-empty relative path")
    manifest = Path(manifest_path)
    if manifest.is_dir():
        raise ValueError("manifest_path must be the manifest file, not its directory")
    root = manifest.resolve().parent
    relative = Path(image_path)
    if relative.is_absolute():
        raise ValueError("image_path must be relative to the directory that contains the manifest")
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("image_path must not contain '.' or '..'")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("image_path escapes the directory that contains the manifest")
    return resolved
