"""The Checkpoint 1.5 budget is imported and not replaced."""

from __future__ import annotations

import ast
from pathlib import Path

from qwen_rgb.budget import dtype_decisions, load_budget_module, select_dtype
from qwen_rgb.pins import SCENE_ID

ROOT = Path(__file__).resolve().parents[2]
PARAMETERS = 2_127_532_032
FLOAT32_BYTES = 8_510_128_128


def test_imported_budget_keeps_the_published_lines():
    budget = load_budget_module(ROOT)
    decisions = dtype_decisions(
        budget,
        parameter_count=PARAMETERS,
        float32_weight_bytes=FLOAT32_BYTES,
        available_bytes=12_188_794_880,
        mps_available=True,
    )
    bf16_lines = decisions["bfloat16"]["lines_bytes"]
    assert bf16_lines["os_floor"] == 6 * 1024**3
    assert bf16_lines["spare_above_weights"] == PARAMETERS * 2 + 5 * 1024**3
    assert bf16_lines["staging_double_copy"] == PARAMETERS * 4
    assert bf16_lines["calibrated_projection"] == 11_509_447_200
    assert decisions["bfloat16"]["required_available_bytes"] == 11_509_447_200
    assert decisions["bfloat16"]["permitted"] is True
    assert decisions["float32"]["permitted"] is False
    below = dtype_decisions(
        budget,
        parameter_count=PARAMETERS,
        float32_weight_bytes=FLOAT32_BYTES,
        available_bytes=11_509_447_200 - 1,
        mps_available=True,
    )
    assert below["bfloat16"]["permitted"] is False
    assert "calibrated_projection" in below["bfloat16"]["failed_lines"]


def test_select_dtype_does_not_prefer_bf16_when_float32_passes():
    decisions = {
        "float32": {"permitted": True},
        "bfloat16": {"permitted": True},
    }
    assert select_dtype(decisions, bf16_probe_ok=True)["dtype"] == "float32"
    refused = select_dtype(
        {"float32": {"permitted": False}, "bfloat16": {"permitted": True}},
        bf16_probe_ok=False,
    )
    assert refused["permitted"] is False


def test_model_from_pretrained_stays_behind_the_recheck():
    source = (ROOT / "src" / "qwen_rgb" / "qwen_rgb" / "measure.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = []
    permit_line = None
    model_call_line = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "from_pretrained":
            calls.append(node.lineno)
        if isinstance(node, ast.FunctionDef) and node.name == "load_model_after_budget":
            for child in ast.walk(node):
                if isinstance(child, ast.Constant) and child.value == "permitted":
                    permit_line = child.lineno
                if (
                    isinstance(child, ast.Call)
                    and isinstance(child.func, ast.Attribute)
                    and child.func.attr == "from_pretrained"
                ):
                    model_call_line = child.lineno
    assert permit_line is not None and model_call_line is not None
    assert permit_line < model_call_line
    assert len(calls) == 2
    assert "fgsm" not in source.lower()
    assert "pgd" not in source.lower()
    assert "dev2-counting" not in source
    assert "dev2-spatial" not in source
    assert SCENE_ID == "dev2-attribute-00-a"
