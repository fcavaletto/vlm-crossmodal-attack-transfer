"""Rendering, seeds, regeneration, and the refusals."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from scene_simple.generate import SealedSplitError, generate_development, refuse_non_development_split
from scene_simple.render import CONTACT_SHEET_SIZE
from scene_simple.seeds import DEVELOPMENT_SEED_DOMAIN, development_seed
from scene_simple.vocabulary import PALETTE

CHECKPOINT1_MANIFEST_SHA256 = "3aea59c848aaafc76de6c70ebe77d452c5f89db7920b573161d30f9a23695453"
CHECKPOINT15_MANIFEST_SHA256 = "b62ab1171a0aec7b2f264615a2961b6d384d00a1a4b17977d619ea27481154e1"
DEV3_ROW_00_A_SEED = 769796342
DEV3_ROW_00_A_MD5 = "c8674e639c07ade72af52f90249e9f10"


def _pixels(path: Path) -> bytes:
    return Image.open(path).tobytes()


def test_center_pixel_matches_the_object_color(records: list[dict], generated) -> None:
    for record in records:
        image = Image.open(generated / record["image_path"])
        assert image.mode == "RGB"
        assert image.size == (512, 384)
        for obj in record["objects"]:
            cx, cy = obj["position"]["cx"], obj["position"]["cy"]
            assert image.getpixel((cx, cy)) == PALETTE[obj["attributes"]["color"]]


def test_contact_sheet_shows_every_scene_and_is_not_an_input(records: list[dict], generated) -> None:
    sheet = Image.open(generated / "contact_sheet.png")
    assert sheet.size == CONTACT_SHEET_SIZE == (1588, 252)
    metadata = json.loads((generated / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["contact_sheet_is_model_input"] is False
    assert metadata["contact_sheet_scene_ids"] == [
        f"dev3-row-{index:02d}-{arm}" for arm in ("a", "b") for index in range(12)
    ]
    assert {record["scene_id"] for record in records} == set(metadata["contact_sheet_scene_ids"])
    assert all(record["image_path"] != "contact_sheet.png" for record in records)


def test_known_seed_vector() -> None:
    assert development_seed("dev3-row-00-a") == DEV3_ROW_00_A_SEED
    payload = DEVELOPMENT_SEED_DOMAIN + "\0" + "dev3-row-00-a"
    assert hashlib.md5(payload.encode("utf-8"), usedforsecurity=False).hexdigest() == DEV3_ROW_00_A_MD5
    assert DEVELOPMENT_SEED_DOMAIN not in {"study2-development-v1", "study2-development-v2"}


def test_seeds_are_unique(records: list[dict]) -> None:
    seeds = [record["seed"] for record in records]
    assert len(seeds) == len(set(seeds)) == 24


def test_two_generations_have_identical_bytes(tmp_path: Path) -> None:
    first = tmp_path / "one"
    second = tmp_path / "two"
    generate_development(first)
    generate_development(second)
    for relative in ("manifest.jsonl", "metadata.json", "checksums.json", "contact_sheet.png"):
        assert (first / relative).read_bytes() == (second / relative).read_bytes()
    names = sorted(path.name for path in (first / "images").glob("*.png"))
    assert len(names) == 24
    for name in names:
        assert (first / "images" / name).read_bytes() == (second / "images" / name).read_bytes()


def test_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    generate_development(tmp_path)
    with pytest.raises(FileExistsError):
        generate_development(tmp_path)


def test_force_removes_a_stale_image(tmp_path: Path) -> None:
    generate_development(tmp_path)
    stale = tmp_path / "images" / "stale.png"
    stale.write_bytes(b"not a scene")
    generate_development(tmp_path, force=True)
    assert not stale.exists()
    assert len(list((tmp_path / "images").glob("*.png"))) == 24


def test_sealed_split_is_refused_and_writes_nothing(tmp_path: Path) -> None:
    for split in ("sealed", "test", "heldout"):
        target = tmp_path / split
        with pytest.raises(SealedSplitError):
            refuse_non_development_split(split)
        assert not target.exists()


def test_checkpoint_directories_are_refused_even_with_force(tmp_path: Path, repo_root: Path) -> None:
    for name in ("development", "development_v2"):
        target = tmp_path / "fixtures" / name
        with pytest.raises(FileExistsError, match="Refusing to overwrite"):
            generate_development(target, force=True)
        assert not target.exists()
    v1 = repo_root / "fixtures" / "development" / "manifest.jsonl"
    v2 = repo_root / "fixtures" / "development_v2" / "manifest.jsonl"
    before = (
        hashlib.sha256(v1.read_bytes()).hexdigest(),
        hashlib.sha256(v2.read_bytes()).hexdigest(),
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        generate_development(repo_root / "fixtures" / "development", force=True)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        generate_development(repo_root / "fixtures" / "development_v2", force=True)
    after = (
        hashlib.sha256(v1.read_bytes()).hexdigest(),
        hashlib.sha256(v2.read_bytes()).hexdigest(),
    )
    assert before == after
    assert before == (CHECKPOINT1_MANIFEST_SHA256, CHECKPOINT15_MANIFEST_SHA256)


def test_cli_refuses_sealed_split(repo_root: Path, tmp_path: Path) -> None:
    env = dict(**{key: value for key, value in __import__("os").environ.items()})
    env["PYTHONPATH"] = str(repo_root / "src" / "scene_simple")
    target = tmp_path / "should-not-exist"
    result = subprocess.run(
        [sys.executable, "-m", "scene_simple", "--output", str(target), "--split", "sealed"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "sealed" in result.stderr
    assert not target.exists()


def test_cli_refuses_the_older_directories(repo_root: Path, tmp_path: Path) -> None:
    env = dict(**{key: value for key, value in __import__("os").environ.items()})
    env["PYTHONPATH"] = str(repo_root / "src" / "scene_simple")
    for name in ("development", "development_v2"):
        target = tmp_path / "fixtures" / name
        result = subprocess.run(
            [sys.executable, "-m", "scene_simple", "--output", str(target), "--force"],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 1, result.stderr
        assert "Refusing to overwrite" in result.stderr
        assert not target.exists()


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_committed_sample_matches_regeneration_and_checksums(repo_root: Path, generated: Path) -> None:
    committed = repo_root / "fixtures" / "development_simple"
    assert _load_jsonl(committed / "manifest.jsonl") == _load_jsonl(generated / "manifest.jsonl")
    assert _pixels(committed / "contact_sheet.png") == _pixels(generated / "contact_sheet.png")
    for image in sorted((generated / "images").glob("*.png")):
        assert _pixels(image) == _pixels(committed / "images" / image.name)
    checksums = json.loads((committed / "checksums.json").read_text(encoding="utf-8"))
    assert checksums["pillow"] == "11.3.0"
    for relative, info in checksums["files"].items():
        path = committed / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info["sha256"]
        assert path.stat().st_size == info["bytes"]


def test_package_does_not_sample_download_or_import_models(repo_root: Path) -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (repo_root / "src" / "scene_simple" / "scene_simple").glob("*.py")
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


def test_image_paths_stay_inside_the_manifest_directory(generated: Path) -> None:
    from scene_simple.paths import resolve_image_path

    manifest = generated / "manifest.jsonl"
    record = _load_jsonl(manifest)[0]
    resolved = resolve_image_path(manifest, record["image_path"])
    assert resolved.is_file()
    assert resolved.is_relative_to(generated)
    with pytest.raises(ValueError):
        resolve_image_path(manifest, "/tmp/other.png")
    with pytest.raises(ValueError):
        resolve_image_path(manifest, "../development/images/dev-attribute-00-a.png")
