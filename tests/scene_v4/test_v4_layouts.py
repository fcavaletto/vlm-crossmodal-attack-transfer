"""Layouts are not recolors or reflections of each other, and not copies of older samples."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

CHECKPOINT1_MANIFEST_SHA256 = "3aea59c848aaafc76de6c70ebe77d452c5f89db7920b573161d30f9a23695453"
CHECKPOINT15_MANIFEST_SHA256 = "b62ab1171a0aec7b2f264615a2961b6d384d00a1a4b17977d619ea27481154e1"
CHECKPOINT16_MANIFEST_SHA256 = "bf21cdcfeca8f5c9a05e9f53465402c48f4a3175bbd363a38ca36dbc443f2367"


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
            row = 1 - row
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
        for shape, _size, col, row in signatures[group_id]:
            assert 0 <= col <= 3
            assert row in {0, 1}
    assert len(signatures) == 48
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


def _rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_older_manifests_and_graphs_are_untouched(repo_root: Path, generated: Path) -> None:
    manifests = {
        "development": (repo_root / "fixtures" / "development", CHECKPOINT1_MANIFEST_SHA256, 24),
        "development_v2": (repo_root / "fixtures" / "development_v2", CHECKPOINT15_MANIFEST_SHA256, 72),
        "development_simple": (
            repo_root / "fixtures" / "development_simple",
            CHECKPOINT16_MANIFEST_SHA256,
            24,
        ),
    }
    new_rows = _rows(generated / "manifest.jsonl")
    new_ids = {row["scene_id"] for row in new_rows}
    new_hashes = set(json.loads((generated / "scene_graph_hashes.json").read_text())["hashes"].values())
    assert len(new_hashes) == 96
    for name, (directory, digest, count) in manifests.items():
        manifest = directory / "manifest.jsonl"
        assert hashlib.sha256(manifest.read_bytes()).hexdigest() == digest, name
        rows = _rows(manifest)
        assert len(rows) == count
        assert new_ids.isdisjoint({row["scene_id"] for row in rows})
        hashes = set(json.loads((directory / "scene_graph_hashes.json").read_text())["hashes"].values())
        assert len(hashes) == count
        assert new_hashes.isdisjoint(hashes)


def test_pixels_are_not_copies_of_the_older_samples(repo_root: Path, generated: Path) -> None:
    from PIL import Image

    def pixels(directory: Path) -> set[bytes]:
        return {Image.open(path).tobytes() for path in (directory / "images").glob("*.png")}

    older = [
        pixels(repo_root / "fixtures" / "development"),
        pixels(repo_root / "fixtures" / "development_v2"),
        pixels(repo_root / "fixtures" / "development_simple"),
    ]
    new = [Image.open(path).tobytes() for path in sorted((generated / "images").glob("*.png"))]
    assert [len(sample) for sample in older] == [24, 72, 24]
    assert len(new) == 96
    assert len(set(new)) == 96
    for sample in older:
        assert sample.isdisjoint(new)
