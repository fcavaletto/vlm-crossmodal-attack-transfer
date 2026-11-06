"""Read the locked development_v4 manifest. No model and no regeneration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .lock import CHECKSUMS_RELATIVE, MANIFEST_RELATIVE, MANIFEST_SHA256, N_GROUPS, N_SCENES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(root: Path) -> list[dict]:
    """Return the 96 development rows after the protocol hash matches."""
    path = root / MANIFEST_RELATIVE
    digest = sha256_file(path)
    if digest != MANIFEST_SHA256:
        raise RuntimeError(
            f"development_v4 manifest hash {digest} does not match the protocol hash {MANIFEST_SHA256}"
        )
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != N_SCENES:
        raise RuntimeError(f"expected {N_SCENES} scenes, found {len(records)}")
    if any(record.get("split") != "development" for record in records):
        raise RuntimeError("a manifest row is not split development")
    if any(record.get("task_family") != "referent_color" for record in records):
        raise RuntimeError("a manifest row is not referent_color")
    groups = {record["counterfactual_group_id"] for record in records}
    if len(groups) != N_GROUPS:
        raise RuntimeError(f"expected {N_GROUPS} groups, found {len(groups)}")
    scene_ids = [record["scene_id"] for record in records]
    if len(scene_ids) != len(set(scene_ids)):
        raise RuntimeError("duplicate scene id")
    return records


def verify_checksums(root: Path) -> dict[str, int]:
    """Check every committed development_v4 checksum. Raise on the first mismatch set."""
    payload = json.loads((root / CHECKSUMS_RELATIVE).read_text(encoding="utf-8"))
    if payload.get("algorithm") != "sha256":
        raise RuntimeError("checksum algorithm is not sha256")
    files = payload["files"]
    mismatches: list[str] = []
    for relative, info in files.items():
        path = root / "fixtures" / "development_v4" / relative
        if not path.is_file():
            mismatches.append(f"missing {relative}")
            continue
        digest = sha256_file(path)
        size = path.stat().st_size
        if digest != info["sha256"] or size != info["bytes"]:
            mismatches.append(relative)
    if mismatches:
        preview = ", ".join(mismatches[:8])
        raise RuntimeError(f"{len(mismatches)} development_v4 checksum mismatches: {preview}")
    manifest = files.get("manifest.jsonl")
    if manifest is None or manifest["sha256"] != MANIFEST_SHA256:
        raise RuntimeError("checksums.json does not pin the protocol manifest hash")
    return {"files": len(files)}
