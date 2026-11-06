"""The written plan and report stay tied to the toy summary and the spec."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "docs" / "statistics" / "ANALYSIS_PLAN.md"
REPORT = ROOT / "docs" / "checkpoint-0" / "AGENT_D_REPORT.md"
SUMMARY = (
    ROOT
    / "src"
    / "analysis"
    / "analysis"
    / "fixtures"
    / "toy"
    / "toy_summary.json"
)

SECTIONS = [
    "# Agent and responsibility",
    "# Executive summary",
    "# Work completed",
    "# Key findings",
    "# Evidence",
    "# Design decisions and rationale",
    "# Known limitations",
    "# Blockers",
    "# Decisions requested from the research lead",
    "# Reproduction instructions",
    "# Git information",
]

# Split so this file does not itself contain the banned published-result tokens.
FORBIDDEN = ("4/" + "32", "phase_e_" + "heldout", "item_" + "004", "mis_vs_" + "sup")


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def test_plan_explains_the_metrics_and_quotes_the_toy_interval():
    text = PLAN.read_text(encoding="utf-8")
    lowered = text.lower()
    for phrase in (
        "clean accuracy",
        "invalid-output rate",
        "attack success rate",
        "additional error",
        "cross-context",
        "a(s, t)",
        "python -m analysis.verify_toy",
        "python -m analysis.simulation",
        "20261106",
        "2000",
        "proposed primary",
        "exploratory",
        "positive control",
        "fabricated",
        "3/5",
        "4/6",
        "-4/6",
        "toy-05",
        "bootstrap",
        "numpy==1.26.4",
    ):
        assert phrase in lowered, phrase
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    illustration = summary["proposed_primary_illustration"]["bootstrap"]
    for key in ("estimate", "ci_low", "ci_high"):
        assert _fmt(illustration[key]) in text
    same = next(
        cell
        for cell in summary["attack_cells"]
        if cell["cell_id"] == "pgd|0.01000000|1|neutral|neutral"
    )
    interval = same["attack_success_bootstrap"]
    for key in ("estimate", "ci_low", "ci_high"):
        assert _fmt(interval[key]) in text
    association = summary["alignment_transfer_association"]["pearson_r"]["value"]
    assert _fmt(association) in text


def test_report_has_the_required_sections_and_a_short_summary():
    text = REPORT.read_text(encoding="utf-8")
    positions = [text.index(heading) for heading in SECTIONS]
    assert positions == sorted(positions)
    start = text.index("# Executive summary")
    body = []
    for line in text[start:].splitlines()[1:]:
        if line.startswith("# "):
            break
        body.append(line)
    words = [word for word in " ".join(body).split() if word]
    assert 1 <= len(words) <= 250
    assert "checkpoint-0/analysis-plan" in text
    assert "python -m pytest -q tests/analysis" in text
    assert "python -m analysis.verify_toy" in text


def test_sources_do_not_copy_study_1_published_results_or_import_a_model():
    roots = [
        ROOT / "src" / "analysis",
        ROOT / "tests" / "analysis",
        ROOT / "docs" / "statistics",
        ROOT / "docs" / "checkpoint-0" / "AGENT_D_REPORT.md",
    ]
    files = []
    for root in roots:
        if root.is_file():
            files.append(root)
        else:
            files.extend(path for path in root.rglob("*") if path.is_file())
    for path in files:
        if path.suffix not in {".py", ".md", ".json", ".jsonl", ".toml"}:
            continue
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN:
            assert token not in text, f"{token} in {path}"
        if path.suffix == ".py":
            assert "import " + "torch" not in text
            assert "vlm_" + "iface" not in text
