"""Pixels, contact sheet, checksums, and regeneration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image

from scene_gen.generate import SealedSplitError, generate_development, refuse_non_development_split
from scene_gen.render import CONTACT_SHEET_SIZE
from scene_gen.seeds import DEVELOPMENT_SEED_DOMAIN, development_seed
from scene_gen.vocabulary import BACKGROUND, PALETTE


def _pixels(path: Path) -> tuple[tuple[int, ...], bytes]:
    image = Image.open(path)
    assert image.mode == "RGB"
    return image.size, image.tobytes()


def test_center_pixel_matches_the_object_color(records: list[dict], generated) -> None:
    for record in records:
        image = Image.open(generated / record["image_path"])
        assert image.size == (512, 384)
        assert image.getpixel((0, 0)) == BACKGROUND
        assert image.getpixel((511, 383)) == BACKGROUND
        for obj in record["objects"]:
            cx = obj["position"]["cx"]
            cy = obj["position"]["cy"]
            assert image.getpixel((cx, cy)) == PALETTE[obj["attributes"]["color"]]


def test_contact_sheet_is_small_and_not_a_scene(records: list[dict], generated) -> None:
    sheet = Image.open(generated / "contact_sheet.png")
    # 8 + 3 * (256 + 8) by 8 + 2 * (192 + 28 + 8), computed here on purpose.
    assert sheet.size == (800, 464)
    assert sheet.size == CONTACT_SHEET_SIZE
    assert all(record["image_path"] != "contact_sheet.png" for record in records)
    metadata = json.loads((generated / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["contact_sheet_scene_ids"] == [
        "dev-attribute-00-a",
        "dev-counting-00-a",
        "dev-spatial-00-a",
        "dev-attribute-00-b",
        "dev-counting-00-b",
        "dev-spatial-00-b",
    ]


def test_known_seed_vector() -> None:
    # md5sum of the bytes study2-development-v1, NUL, dev-attribute-00-a.
    digest = "7e6e5a265ef7ba216cdbfbb784aa362f"
    assert int(digest, 16) % 2_147_483_647 == 1_303_635_806
    assert development_seed("dev-attribute-00-a") == 1_303_635_806
    assert DEVELOPMENT_SEED_DOMAIN == "study2-development-v1"


def test_seeds_are_unique_and_domain_separated(records: list[dict]) -> None:
    seeds = [record["seed"] for record in records]
    assert len(seeds) == len(set(seeds))
    for record in records:
        assert record["seed"] == development_seed(record["scene_id"])


def test_two_generations_have_identical_bytes(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_development(first)
    generate_development(second)
    files = sorted(path.relative_to(first).as_posix() for path in first.rglob("*") if path.is_file())
    assert files
    for relative in files:
        assert (first / relative).read_bytes() == (second / relative).read_bytes()


def test_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    generate_development(tmp_path)
    try:
        generate_development(tmp_path)
    except FileExistsError:
        return
    raise AssertionError("second generate should have refused")


def test_sealed_split_is_refused_and_writes_nothing(tmp_path: Path) -> None:
    for split in ("sealed", "test", "heldout"):
        try:
            refuse_non_development_split(split)
        except SealedSplitError:
            pass
        else:
            raise AssertionError(split)
    assert list(tmp_path.iterdir()) == []


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_cli_refuses_sealed_split(repo_root: Path, tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src" / "scene_gen")
    result = subprocess.run(
        [sys.executable, "-m", "scene_gen", "--output", str(tmp_path / "out"), "--split", "sealed"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "development" in result.stderr
    assert not (tmp_path / "out").exists()


DEVELOPMENT_FIXTURES = ["development", "development_simple", "development_v2", "development_v4"]
# The only held-out split allowed in fixtures/. It was committed on purpose at the
# Checkpoint 2.4 freeze (commit c35b02c), and it is allowed only while it is the
# hash-frozen split: its name, its FROZEN.json hashes and its per-file checksums must match.
AUTHORIZED_HELD_OUT = {"trackB_confirm_v1": "docs/checkpoint-2.4-confirm/FROZEN.json"}


def _assert_fixture_directories_are_authorized(repo_root: Path) -> None:
    """Exact allowlist. Any other sealed, test or held-out directory still fails."""
    names = sorted(path.name for path in (repo_root / "fixtures").iterdir())
    assert names == sorted(DEVELOPMENT_FIXTURES + list(AUTHORIZED_HELD_OUT))
    for name, frozen_relative in AUTHORIZED_HELD_OUT.items():
        split = repo_root / "fixtures" / name
        frozen = json.loads((repo_root / frozen_relative).read_text(encoding="utf-8"))["files"]
        frozen_here = {key: value for key, value in frozen.items() if key.startswith(f"fixtures/{name}/")}
        assert sorted(frozen_here) == [f"fixtures/{name}/{leaf}" for leaf in ("checksums.json", "manifest.jsonl", "metadata.json")]
        for relative, digest in frozen_here.items():
            assert hashlib.sha256((repo_root / relative).read_bytes()).hexdigest() == digest
        listed = json.loads((split / "checksums.json").read_text(encoding="utf-8"))["files"]
        for relative, info in listed.items():
            assert hashlib.sha256((split / relative).read_bytes()).hexdigest() == info["sha256"]
        on_disk = sorted(str(path.relative_to(split)) for path in split.rglob("*") if path.is_file())
        assert on_disk == sorted(set(listed) | {"checksums.json"}), "unlisted file in the held-out split"


def test_committed_sample_matches_regeneration_and_checksums(repo_root: Path, generated: Path) -> None:
    committed = repo_root / "fixtures" / "development"
    assert _load_jsonl(committed / "manifest.jsonl") == _load_jsonl(generated / "manifest.jsonl")
    for name in ("contact_sheet.png",):
        assert _pixels(committed / name) == _pixels(generated / name)
    for image in sorted((generated / "images").glob("*.png")):
        assert _pixels(image) == _pixels(committed / "images" / image.name)
    checksums = json.loads((committed / "checksums.json").read_text(encoding="utf-8"))
    assert checksums["pillow"] == "11.3.0"
    for relative, info in checksums["files"].items():
        path = committed / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"]
        assert path.stat().st_size == info["bytes"]
    # development is Checkpoint 1. development_v2 is Checkpoint 1.5.
    # development_simple is Checkpoint 1.6. A sealed, test, or held-out directory still fails.
    # development_v4 is the Checkpoint 1.7 referent-color sample (commit 44aafcf), all rows `development`.
    # trackB_confirm_v1 is the Checkpoint 2.4 held-out split, allowed only as hash-frozen (see helper).
    _assert_fixture_directories_are_authorized(repo_root)


def test_package_does_not_sample_download_or_import_models(repo_root: Path) -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (repo_root / "src" / "scene_gen" / "scene_gen").glob("*.py")
    )
    lowered = source.lower()
    assert "import random" not in source
    assert "from random" not in source
    assert "import torch" not in source
    assert "urllib" not in lowered
    assert "requests" not in lowered
    assert "clevr-dataset" not in lowered
    assert "import clevr" not in lowered
    assert "http://" not in lowered
    assert "https://" not in lowered


def test_dataset_design_states_the_procedure(repo_root: Path) -> None:
    text = (repo_root / "docs" / "data" / "DATASET_DESIGN.md").read_text(encoding="utf-8")
    for phrase in (
        "normalize_for_leakage",
        "helpful_nonrevealing",
        "answer_revealing",
        "study2-development-v1",
        "study2-sealed-v1",
        "CLEVR",
        "Answer leakage",
        "Textual shortcuts",
        "Label imbalance",
        "Scene ambiguity",
        "Counterfactual dependence",
        "Development and test contamination",
        "Clean-accuracy floors",
        "Generalization beyond synthetic images",
    ):
        assert phrase in text, phrase
