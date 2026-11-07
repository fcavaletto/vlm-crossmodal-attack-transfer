"""The committed estimate and manifest match the measurement files."""

from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

from cuda_readiness.estimate import build_estimate, quoted_rows
from cuda_readiness.manifest import build_manifest
from cuda_readiness.pins import (
    BASE_COMMIT,
    BF16_REQUIRED_AVAILABLE_BYTES,
    DEPENDENCY_PINS,
    GPU_HOURLY_RATE_USD,
    GPU_INSTANCE_TYPE,
    GPU_NAME,
    HELD_OUT_PLANNING_SCENE_COUNT,
)

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "cuda_readiness"


def test_committed_estimate_matches_the_mps_record():
    estimate = build_estimate(ROOT)
    committed = json.loads((ROOT / "docs/checkpoint-2.0/cuda/planning_estimate.json").read_text(encoding="utf-8"))
    assert committed == estimate
    quoted = quoted_rows(estimate)
    assert [row["scene_count"] for row in quoted] == [96, HELD_OUT_PLANNING_SCENE_COUNT]
    assert {row["hourly_rate_usd"] for row in quoted} == {GPU_HOURLY_RATE_USD}
    assert {row["gpu_instance_type"] for row in quoted} == {GPU_INSTANCE_TYPE}
    assert {row["gpu_name"] for row in quoted} == {GPU_NAME}
    assert all(row["cuda_timing_measured"] is False for row in estimate["rows"])
    assert estimate["source"]["stage_b_seconds_sum"] == "7878.226"
    assert estimate["source"]["stage_a_used_for_scaling"] is False
    assert estimate["source"]["finishing_process_wall_used_for_scaling"] is False
    assert estimate["memory"]["unchanged_bf16_lines_bytes"]["calibrated_projection"] == BF16_REQUIRED_AVAILABLE_BYTES
    assert estimate["memory"]["guard_lowered"] is False
    assert estimate["memory"]["local_cuda_allocator_check"] is False
    assert estimate["held_out_scenes_opened"] is False
    development, held_out = quoted
    assert development["estimated_wall_seconds"] == "47272.982"
    assert development["estimated_on_demand_usd"] == "12.84"
    assert held_out["estimated_wall_seconds"] == "126055.242"
    assert held_out["estimated_on_demand_usd"] == "34.23"


def test_planning_writeup_quotes_the_class_and_the_rate():
    text = (ROOT / "docs/checkpoint-2.0/cuda/PLANNING_ESTIMATE.md").read_text(encoding="utf-8")
    for piece in (
        GPU_INSTANCE_TYPE,
        GPU_NAME,
        GPU_HOURLY_RATE_USD,
        "$12.84",
        "$34.23",
        "not a CUDA measurement",
        "11,509,447,200",
        "256",
    ):
        assert piece in text
    guide = (ROOT / "docs/checkpoint-2.0/cuda/RUN_GUIDE.md").read_text(encoding="utf-8")
    validation = (ROOT / "docs/checkpoint-2.0/cuda/VALIDATION.md").read_text(encoding="utf-8")
    assert "--require-cuda" in guide
    assert "were not executed" in validation
    assert "not executed on CUDA" in validation


def test_manifest_locks_the_development_set_and_the_scoring_files():
    manifest = build_manifest(ROOT)
    committed = json.loads((ROOT / "docs/checkpoint-2.0/cuda/manifest.json").read_text(encoding="utf-8"))
    assert committed == manifest
    assert manifest["revision"] == "89644892e4d85e24eaac8bacfd4f463576704203"
    assert manifest["token_ids"]["red"] == [1151]
    assert manifest["development_scene_count"] == 96
    assert manifest["images_opened"] is False
    assert manifest["attacks_run"] is False
    assert manifest["dependency_pins"] == DEPENDENCY_PINS
    assert set(manifest["science_sha256"]) >= {
        "src/qwen_rgb/qwen_rgb/loss.py",
        "src/qwen_rgb/qwen_rgb/rgb_path.py",
        "src/attack_feasibility/attack_feasibility/score.py",
        "src/attack_feasibility/attack_feasibility/steps.py",
    }


def test_readiness_package_does_not_redefine_the_attack():
    text = "\n".join(path.read_text(encoding="utf-8") for path in PACKAGE.rglob("*.py"))
    for banned in (
        ".from_pretrained(",
        "def pgd_update",
        "def project_linf",
        "def backward_at_image",
        "def differentiable_model_pixels",
    ):
        assert banned not in text


def test_image_pins_match_the_measured_environment():
    dockerfile = (ROOT / "docker/cuda/Dockerfile").read_text(encoding="utf-8")
    constraints = (ROOT / "docker/cuda/constraints.txt").read_text(encoding="utf-8")
    assert "python:3.12.14-slim-bookworm" in dockerfile
    assert "https://download.pytorch.org/whl/cu132" in dockerfile
    assert 'CMD ["preflight"]' in dockerfile
    for name, version in DEPENDENCY_PINS.items():
        if name == "python":
            continue
        assert f"{name}=={version}" in constraints


def test_held_out_count_is_the_written_sealed_target_and_the_base_commit_exists():
    proposal = (ROOT / "docs/research/RESEARCH_PROPOSAL.md").read_text(encoding="utf-8")
    assert "256 sealed held-out" in proposal
    subprocess.check_call(["git", "cat-file", "-e", f"{BASE_COMMIT}^{{commit}}"], cwd=ROOT)
    tree = ast.parse((PACKAGE / "cuda_readiness" / "estimate.py").read_text(encoding="utf-8"))
    imported = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    ]
    assert "attack_feasibility" not in " ".join(imported)
