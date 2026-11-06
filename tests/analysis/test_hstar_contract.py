"""The H* write-up stays tied to the fabricated summary and does not claim support."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "docs" / "statistics" / "H_STAR_ANALYSIS.md"
REPORT = ROOT / "docs" / "checkpoint-1.5" / "AGENT_D_REPORT.md"
PRECISION = (
    ROOT
    / "src"
    / "analysis"
    / "analysis"
    / "fixtures"
    / "hstar"
    / "precision_simulation.json"
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


def _executive_summary(text: str) -> str:
    start = text.index("# Executive summary")
    end = text.index("\n# ", start + 1)
    return text[start:end].split("\n", 1)[1].strip()


def test_report_sections_and_summary_length():
    text = REPORT.read_text(encoding="utf-8")
    cursor = -1
    for heading in SECTIONS:
        found = text.find(heading)
        assert found > cursor, heading
        cursor = found
    summary = _executive_summary(text)
    assert len(summary.split()) <= 250
    assert "checkpoint-1.5/h-star-estimator" in text
    assert "90c6ffe83d61514da217a20f8dfc2f202afbbe69" in text
    assert "Not merged." in text
    lowered = text.lower()
    assert "0.57" not in text
    assert "183/200" in text
    assert "does not promise" in lowered
    _reject_unnegated_support_claim(text)


def test_analysis_quotes_the_fabricated_planning_sentence_and_the_worked_line():
    text = ANALYSIS.read_text(encoding="utf-8")
    precision = json.loads(PRECISION.read_text(encoding="utf-8"))
    sentence = precision["recommendation"]["sentence"]
    assert sentence in text
    assert precision["not_an_empirical_finding"] is True
    assert precision["recommendation"]["does_not_promise_significance"] is True
    lowered = text.lower()
    for phrase in (
        "fabricated",
        "counterfactual_group_id",
        "wild cluster",
        "webb",
        "logistic",
        "invalid",
        "valid-wrong",
        "too few",
        "flat",
        "clean confidence",
        "gradient",
        "python -m analysis.verify_hstar",
        "python -m analysis.hstar_simulation",
        "numpy==1.26.4",
        "143/197",
        "183/200",
        "1/2",
        "does not promise",
    ):
        assert phrase in lowered, phrase
    assert "0.57" not in text
    assert "confirmatory_supported" in text
    _reject_unnegated_support_claim(text)


def _reject_unnegated_support_claim(text: str) -> None:
    """A sentence may name the claim only in order to withhold it."""
    for line in text.splitlines():
        lowered = line.lower()
        if "h* is supported" not in lowered and "confirmatory_supported` is true" not in lowered:
            continue
        if "true" in lowered and "false" not in lowered:
            raise AssertionError(line)
        if not any(token in lowered for token in ("no ", "not ", "false", "never", "stays false")):
            raise AssertionError(line)
