"""The verification command rebuilds the committed toy summary."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from analysis.schemas import SchemaError
from analysis.toy_summary import TOY_INTERPRETATION, build_toy_summary
from analysis.verify_toy import compare_summaries, verify_toy_summary

ROOT = Path(__file__).resolve().parents[2]
TOY = ROOT / "src" / "analysis" / "analysis" / "fixtures" / "toy"


def test_committed_summary_matches_a_fresh_rebuild():
    report = verify_toy_summary(TOY)
    assert report["ok"], report
    assert report["n_mismatches"] == 0
    summary = json.loads((TOY / "toy_summary.json").read_text(encoding="utf-8"))
    assert summary["data_origin"] == "fabricated_toy_fixtures"
    assert summary["not_an_empirical_finding"] is True
    assert summary["primary_estimand_locked"] is False
    assert summary["labels_are_proposals"] is True
    illustration = summary["proposed_primary_illustration"]
    assert illustration["locked"] is False
    assert illustration["analysis_role"] == "proposed_primary"
    assert illustration["bootstrap"]["estimate"] == pytest.approx(-4 / 6)
    assert illustration["bootstrap"]["seed"] == 20261106
    assert illustration["bootstrap"]["n_boot"] == 2000
    assert illustration["bootstrap"]["unit"] == "scene_id"
    assert illustration["reading"] == TOY_INTERPRETATION
    assert "replicates" not in illustration["bootstrap"]
    noise = next(cell for cell in summary["attack_cells"] if cell["method"] == "noise")
    assert noise["attack_success_rate"]["attack_success_rate"]["value"] is None
    rebuilt = build_toy_summary(TOY)
    assert compare_summaries(rebuilt, summary) == []


def test_a_changed_published_number_is_a_mismatch():
    published = json.loads((TOY / "toy_summary.json").read_text(encoding="utf-8"))
    rebuilt = build_toy_summary(TOY)
    published["n_scenes"] = 999
    mismatches = compare_summaries(rebuilt, published)
    assert any("n_scenes" in line for line in mismatches)


def test_correct_before_must_match_the_clean_target_prediction(tmp_path: Path):
    copied = tmp_path / "toy"
    shutil.copytree(TOY, copied)
    attacks = (copied / "attacks.jsonl").read_text(encoding="utf-8").splitlines()
    first = json.loads(attacks[0])
    first["correct_before"] = not first["correct_before"]
    attacks[0] = json.dumps(first, separators=(", ", ": "))
    (copied / "attacks.jsonl").write_text("\n".join(attacks) + "\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="correct_before"):
        build_toy_summary(copied)
    report = verify_toy_summary(copied)
    assert report["ok"] is False


def test_cli_exits_zero_on_the_committed_packet():
    completed = subprocess.run(
        [sys.executable, "-m", "analysis.verify_toy"],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert payload["n_mismatches"] == 0


def test_cli_exits_one_when_the_summary_is_wrong(tmp_path: Path):
    copied = tmp_path / "toy"
    shutil.copytree(TOY, copied)
    summary_path = copied / "toy_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["n_scenes"] = 1
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "analysis.verify_toy", "--fixture-dir", str(copied)],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    assert payload["ok"] is False
