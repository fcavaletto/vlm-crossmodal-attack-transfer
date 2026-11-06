"""Layouts are not recolors or reflections, and they do not reuse Checkpoint 1 graphs."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

CHECKPOINT1_MANIFEST_SHA256 = "3aea59c848aaafc76de6c70ebe77d452c5f89db7920b573161d30f9a23695453"


def _signature(record: dict) -> tuple[tuple[str, str, int, int], ...]:
    return tuple(
        sorted(
            (
                obj["category"],
                obj["attributes"]["size"],
                obj["position"]["col"],
                obj["position"]["row"],
            )
            for obj in record["objects"]
        )
    )


def _flip(signature: tuple[tuple[str, str, int, int], ...], col_flip: bool, row_flip: bool):
    flipped = []
    for shape, size, col, row in signature:
        if col_flip:
            col = 3 - col
        if row_flip:
            row = 2 - row
        flipped.append((shape, size, col, row))
    return tuple(sorted(flipped))


def test_groups_do_not_recolor_or_reflect_each_other(records: list[dict]) -> None:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["counterfactual_group_id"]].append(record)
    signatures = {}
    for group_id, members in grouped.items():
        members.sort(key=lambda record: record["scene_id"])
        assert _signature(members[0]) == _signature(members[1])
        signatures[group_id] = _signature(members[0])
    assert len(signatures) == 36
    items = list(signatures.items())
    for index, (group_id, signature) in enumerate(items):
        mirrors = {
            _flip(signature, True, False),
            _flip(signature, False, True),
            _flip(signature, True, True),
        }
        for other_id, other in items[index + 1 :]:
            assert other != signature, (group_id, other_id)
            assert other not in mirrors, (group_id, other_id)


def test_checkpoint1_manifest_and_graphs_are_untouched(repo_root: Path, generated: Path) -> None:
    manifest = repo_root / "fixtures" / "development" / "manifest.jsonl"
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert digest == CHECKPOINT1_MANIFEST_SHA256
    v1_rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(v1_rows) == 24
    assert {row["scene_id"] for row in v1_rows}.isdisjoint({row["scene_id"] for row in _rows(generated)})
    v1_hashes = json.loads(
        (repo_root / "fixtures" / "development" / "scene_graph_hashes.json").read_text(encoding="utf-8")
    )["hashes"]
    v2_hashes = json.loads((generated / "scene_graph_hashes.json").read_text(encoding="utf-8"))["hashes"]
    assert set(v1_hashes.values()).isdisjoint(v2_hashes.values())
    assert len(v1_hashes) == 24
    assert len(v2_hashes) == 72


def _rows(generated: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (generated / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_v2_pixels_are_not_the_checkpoint1_images(repo_root: Path, generated: Path) -> None:
    from PIL import Image

    v1_pixels = {
        Image.open(path).tobytes()
        for path in (repo_root / "fixtures" / "development" / "images").glob("*.png")
    }
    assert len(v1_pixels) == 24
    v2_pixels = [Image.open(path).tobytes() for path in (generated / "images").glob("*.png")]
    assert len(v2_pixels) == 72
    assert len(set(v2_pixels)) == 72
    assert v1_pixels.isdisjoint(v2_pixels)
