"""Write the Checkpoint 0 development sample. Refuses every other split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import PIL
from PIL import Image

from .design import build_scenes
from .geometry import CANVAS_H, CANVAS_W
from .records import materialize, scene_graph_hash
from .render import render_contact_sheet, render_scene
from .seeds import DEVELOPMENT_SEED_DOMAIN, SEED_MODULUS
from .version import __version__

# First pair of each family, both arms. A review figure, not a model input.
CONTACT_SHEET_IDS: tuple[str, ...] = (
    "dev-attribute-00-a",
    "dev-counting-00-a",
    "dev-spatial-00-a",
    "dev-attribute-00-b",
    "dev-counting-00-b",
    "dev-spatial-00-b",
)


class SealedSplitError(RuntimeError):
    """Raised when a caller asks for anything other than the development split."""


def refuse_non_development_split(split: str) -> None:
    """Checkpoint 0 may describe a sealed set. It may not generate one."""
    if split != "development":
        raise SealedSplitError(
            f"refusing split {split!r}. Checkpoint 0 writes only split 'development'. "
            "The sealed-set procedure is documented in docs/data/DATASET_DESIGN.md "
            "and is not implemented in this package."
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def generate_development(output: Path, *, force: bool = False) -> dict[str, object]:
    """Write the development manifest, images, contact sheet, and checksums.

    ``image_path`` is relative to ``output``, the directory that contains
    ``manifest.jsonl``. That directory is the fixtures root for this sample.
    """
    refuse_non_development_split("development")
    output = Path(output)
    manifest_path = output / "manifest.jsonl"
    if manifest_path.exists() and not force:
        raise FileExistsError(
            f"{manifest_path} already exists. Refusing to overwrite without --force. "
            "Regenerating changes committed fixtures; do it on purpose."
        )
    output.mkdir(parents=True, exist_ok=True)
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records = [materialize(spec) for spec in build_scenes()]
    rendered: dict[str, Image.Image] = {}
    expected_names: set[str] = set()
    for record in records:
        image = render_scene(record, canvas_w=CANVAS_W, canvas_h=CANVAS_H)
        relative = str(record["image_path"])
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, format="PNG", optimize=False, compress_level=9)
        rendered[str(record["scene_id"])] = image
        expected_names.add(target.name)

    for stale in images_dir.glob("*.png"):
        if stale.name not in expected_names:
            stale.unlink()

    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    panels = []
    for scene_id in CONTACT_SHEET_IDS:
        record = next(item for item in records if item["scene_id"] == scene_id)
        caption = f"{record['task_family']} {scene_id[-4:]} {record['answer']}"
        panels.append((caption, rendered[scene_id]))
    sheet = render_contact_sheet(panels)
    sheet_path = output / "contact_sheet.png"
    sheet.save(sheet_path, format="PNG", optimize=False, compress_level=9)

    families: dict[str, int] = {}
    answers: dict[str, dict[str, int]] = {}
    for record in records:
        family = str(record["task_family"])
        families[family] = families.get(family, 0) + 1
        answers.setdefault(family, {})
        answer = str(record["answer"])
        answers[family][answer] = answers[family].get(answer, 0) + 1

    metadata: dict[str, object] = {
        "name": "study2_development_v0",
        "status": "checkpoint_0_development_sample",
        "generator": "scene_gen.generate",
        "generator_version": __version__,
        "scene_count": len(records),
        "split": "development",
        "seed_domain": DEVELOPMENT_SEED_DOMAIN,
        "seed_modulus": SEED_MODULUS,
        "image_size": [CANVAS_W, CANVAS_H],
        "grid": {"cols": 4, "rows": 2, "cell": [128, 192]},
        "pillow": PIL.__version__,
        "task_counts": families,
        "answer_counts": answers,
        "counterfactual_group_count": len({r["counterfactual_group_id"] for r in records}),
        "normalization": "scene_gen.normalize.normalize_for_leakage",
        "image_path_rule": "relative to the directory that contains manifest.jsonl",
        "contact_sheet": "contact_sheet.png",
        "contact_sheet_scene_ids": list(CONTACT_SHEET_IDS),
        "contact_sheet_is_model_input": False,
        "dataset_origin": (
            "Original Pillow drawings from the scene graph. "
            "No photographs, no CLEVR images, no downloaded assets."
        ),
        "caution": (
            "Development sample only. Not a sealed test set. "
            "Not a measurement of clean accuracy or attack success."
        ),
    }
    _write_json(output / "metadata.json", metadata)

    hashes = {
        str(record["scene_id"]): scene_graph_hash(record) for record in records
    }
    _write_json(
        output / "scene_graph_hashes.json",
        {
            "algorithm": "sha256",
            "canonicalization": "json sort_keys separators=(',', ':') of question, answer, objects",
            "purpose": (
                "Exclusion list for a future sealed generator. "
                "This file is not a sealed set."
            ),
            "hashes": hashes,
        },
    )

    checksum_paths = [
        manifest_path,
        sheet_path,
        output / "metadata.json",
        output / "scene_graph_hashes.json",
        *sorted(images_dir.glob("*.png")),
    ]
    files = {}
    for path in checksum_paths:
        files[str(path.relative_to(output))] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    _write_json(
        output / "checksums.json",
        {
            "algorithm": "sha256",
            "pillow": PIL.__version__,
            "note": (
                "Integrity of the committed files. A later machine may encode PNG "
                "bytes differently; compare pixels before replacing these files."
            ),
            "files": files,
        },
    )
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write the Checkpoint 0 development scenes. Will not write a sealed split."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fixtures/development"),
        help="directory that will contain manifest.jsonl (default: fixtures/development)",
    )
    parser.add_argument(
        "--split",
        default="development",
        help="only 'development' is accepted",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing manifest",
    )
    args = parser.parse_args(argv)
    try:
        refuse_non_development_split(args.split)
        metadata = generate_development(args.output, force=args.force)
    except SealedSplitError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
