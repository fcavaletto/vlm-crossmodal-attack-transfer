"""Run manifest for the pinned model and the development scoring path.

The manifest records hashes. It does not load weights and it does not open
an image.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from dev_pilot.lock import (
    EXPECTED_TOKEN_IDS,
    MANIFEST_SHA256,
    N_SCENES,
    QWEN_ANSWER_PREFIX,
)
from qwen_rgb.pins import MODEL_ID, PINNED_REVISION

from .hashes import sha256_file
from .pins import (
    BASE_COMMIT,
    CHECKSUMS_RELATIVE,
    CUDA_WHEEL,
    DEPENDENCY_PINS,
    MANIFEST_RELATIVE,
    MEASURED_ENVIRONMENT_RELATIVE,
    NVIDIA_DRIVER_MINIMUM,
    PROTOCOL_COMMIT,
    PROTOCOL_RELATIVE,
    SCIENCE_FILES,
    TORCH_INDEX_URL,
    parse_measured_environment,
)


def build_manifest(root: Path) -> dict[str, Any]:
    """Identity of the model, the development manifest, and the scoring files."""
    measured = (root / MEASURED_ENVIRONMENT_RELATIVE).read_text(encoding="utf-8")
    recorded = parse_measured_environment(measured)
    if recorded != DEPENDENCY_PINS:
        raise RuntimeError(
            "Dependency pins do not match docs/models/measured-environment.txt. "
            f"Recorded {recorded}."
        )
    manifest_path = root / MANIFEST_RELATIVE
    manifest_hash = sha256_file(manifest_path, root=root)
    if manifest_hash != MANIFEST_SHA256:
        raise RuntimeError("The development_v4 manifest hash changed. Refusing to describe a different set.")
    scene_count = _line_count(manifest_path)
    if scene_count != N_SCENES or scene_count != 96:
        raise RuntimeError(f"development_v4 has {scene_count} manifest rows, not 96.")
    token_ids = EXPECTED_TOKEN_IDS["qwen3-vl-2b"]
    if token_ids != {"red": (1151,), "blue": (12203,), "green": (13250,), "yellow": (27869,)}:
        raise RuntimeError("Qwen bare color ids do not match the Checkpoint 1.7 table.")
    if QWEN_ANSWER_PREFIX != "<|im_end|>\n<|im_start|>assistant\n":
        raise RuntimeError("The Qwen answer prefix changed.")
    if MODEL_ID != "Qwen/Qwen3-VL-2B-Instruct":
        raise RuntimeError("The Qwen model id changed.")
    if PINNED_REVISION != "89644892e4d85e24eaac8bacfd4f463576704203":
        raise RuntimeError("The Qwen revision changed.")
    return {
        "schema": "cuda_readiness.manifest.v1",
        "label": "DEVELOPMENT ONLY",
        "attacks_run": False,
        "model_from_pretrained_called": False,
        "held_out_scenes_opened": False,
        "h_star_fitted": False,
        "cuda_numerical_validation": "not_executed",
        "base_commit": BASE_COMMIT,
        "protocol_commit": PROTOCOL_COMMIT,
        "model_id": MODEL_ID,
        "revision": PINNED_REVISION,
        "attn_implementation": "eager",
        "dtype": "bfloat16",
        "answer_prefix": QWEN_ANSWER_PREFIX,
        "token_ids": {color: list(token_ids[color]) for color in ("red", "blue", "green", "yellow")},
        "development_manifest": MANIFEST_RELATIVE,
        "development_manifest_sha256": manifest_hash,
        "development_scene_count": scene_count,
        "development_checksums_sha256": sha256_file(root / CHECKSUMS_RELATIVE, root=root),
        "protocol_sha256": sha256_file(root / PROTOCOL_RELATIVE, root=root),
        "science_sha256": {
            relative: sha256_file(root / relative, root=root) for relative in SCIENCE_FILES
        },
        "dependency_pins": dict(DEPENDENCY_PINS),
        "torch_index_url": TORCH_INDEX_URL,
        "cuda_wheel": CUDA_WHEEL,
        "nvidia_driver_minimum": NVIDIA_DRIVER_MINIMUM,
        "images_opened": False,
    }


def _line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def load_module_from_path(path: Path, name: str):
    """Import a file by path. Used for the budget script, whose ``main`` is not called.

    Bytecode is not written beside the Checkpoint 1.5 script.
    """
    import sys

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module
