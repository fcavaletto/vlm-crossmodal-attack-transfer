"""The development protocol and the pure counts. No model is loaded."""

from __future__ import annotations

import json
from pathlib import Path

from dev_pilot.fixtures import load_manifest, verify_checksums
from dev_pilot.lock import (
    EXPECTED_TOKEN_IDS,
    MANIFEST_SHA256,
    QWEN_ANSWER_PREFIX,
    SMOL_ANSWER_PREFIX,
)
from dev_pilot.provenance import protocol_provenance
from dev_pilot.rows import assemble_row, protocol_errors
from dev_pilot.summarize import above_80, at_least_65, clean_both, competence_gate, context_counts, eligible_groups

ROOT = Path(__file__).resolve().parents[2]
COLORS = ("red", "blue", "green", "yellow")


def test_protocol_text_names_the_locked_rule():
    text = (ROOT / "docs/checkpoint-1.7/DEVELOPMENT_PROTOCOL.md").read_text(encoding="utf-8")
    assert MANIFEST_SHA256 in text
    assert "0.65" in text
    assert "80%" in text
    assert json.dumps(SMOL_ANSWER_PREFIX) in text
    assert json.dumps(QWEN_ANSWER_PREFIX) in text
    assert "2382" in text and "1151" in text
    assert "DEVELOPMENT ONLY" in text
    assert "n_groups_eligible" in text


def test_manifest_hash_and_checksums_match_the_protocol():
    records = load_manifest(ROOT)
    assert len(records) == 96
    assert verify_checksums(ROOT)["files"] > 96


def test_protocol_commit_has_no_predictions():
    report = protocol_provenance(ROOT)
    assert report["protocol_commit_is_ancestor"] is True
    assert report["predictions_present_at_protocol_commit"] is False
    assert report["manifest_sha256"] == MANIFEST_SHA256
    assert report["checkpoint_1_6_diff_empty"] is True


def test_integer_competence_boundaries():
    assert at_least_65(62, 96) is False
    assert at_least_65(63, 96) is True
    assert above_80(76, 96) is False
    assert above_80(77, 96) is True


def test_closed_vocabulary_tie_breaks_by_color_name():
    record = _record(0)
    summed = {"red": 0.2, "blue": 0.2, "green": 0.2, "yellow": 0.2}
    row = _row(record, "neutral", summed, generation="blue")
    assert row["chosen"] == "blue"
    assert protocol_errors(row, record) == []


def test_pair_partition_and_clean_both_counts():
    records = [_record(index) for index in range(96)]
    rows = []
    for context in ("neutral", "helpful_nonrevealing"):
        for index, record in enumerate(records):
            correct = index < 70
            answer = record["answer"]
            other = next(color for color in COLORS if color != answer)
            picked = answer if correct else other
            chosen_score = {color: (0.1 if color == picked else 1.0) for color in COLORS}
            rows.append(_row(record, context, chosen_score, generation=picked))
    neutral = context_counts(rows, "neutral")
    assert neutral["closed_vocabulary_correct"] == 70
    assert neutral["n"] == 96
    assert neutral["pairs"]["both_correct"] + neutral["pairs"]["one_correct"] + neutral["pairs"]["neither_correct"] == 48
    gate = competence_gate({"neutral": neutral, "helpful_nonrevealing": context_counts(rows, "helpful_nonrevealing")})
    assert gate["passed"] is True
    assert gate["contexts_above_80"] == []
    closed = clean_both(rows, lambda row: bool(row["closed_vocabulary_correct"]))
    assert closed["n_scenes"] == 70
    eligible = eligible_groups(records, rows, None)
    assert eligible["n_groups_eligible"] is None
    assert eligible["h_star_fitted"] is False


def test_eligible_group_requires_gradients_and_strict_correctness():
    singles = [_record(0), _record(2)]
    rows = []
    for record in singles:
        for context in ("neutral", "helpful_nonrevealing"):
            summed = {color: (0.1 if color == record["answer"] else 1.0) for color in COLORS}
            rows.append(_row(record, context, summed, generation=record["answer"]))
    assert eligible_groups(singles, rows, {})["n_groups_eligible"] == 0
    pair = [_record(0), _record(0)]
    pair[1] = dict(pair[0])
    pair[1]["scene_id"] = "dev4-ref-00-b"
    pair[1]["answer"] = "blue"
    rows = []
    for record in pair:
        for context in ("neutral", "helpful_nonrevealing"):
            summed = {color: (0.1 if color == record["answer"] else 1.0) for color in COLORS}
            rows.append(_row(record, context, summed, generation=record["answer"]))
    gradients = {
        pair[0]["scene_id"]: {
            "neutral": {"finite": True, "l1": 1.0},
            "helpful_nonrevealing": {"finite": True, "l1": 0.0},
            "alignment": 0.25,
        }
    }
    assert eligible_groups(pair, rows, gradients)["n_groups_eligible"] == 0
    gradients[pair[0]["scene_id"]]["helpful_nonrevealing"]["l1"] = 0.5
    assert eligible_groups(pair, rows, gradients)["eligible_group_ids"] == ["cf4-ref-00"]


def test_token_table_matches_the_scoring_proposal():
    assert EXPECTED_TOKEN_IDS["smolvlm2"]["red"] == (2382,)
    assert EXPECTED_TOKEN_IDS["qwen3-vl-2b"]["red"] == (1151,)
    assert SMOL_ANSWER_PREFIX == "<end_of_utterance>\nAssistant:"
    assert QWEN_ANSWER_PREFIX.endswith("\n")
    assert not SMOL_ANSWER_PREFIX[-1].isspace()


def _record(index: int) -> dict:
    arm = "a" if index % 2 == 0 else "b"
    group = index // 2
    answer = COLORS[index % 4]
    other = COLORS[(index + 1) % 4]
    return {
        "scene_id": f"dev4-ref-{group:02d}-{arm}",
        "counterfactual_group_id": f"cf4-ref-{group:02d}",
        "answer": answer if arm == "a" else other,
        "question": "What color is the small circle in the top row?",
        "ambiguity_flags": [],
        "contexts": {
            "neutral": {"text": "Several simple objects appear on a plain background.", "reveals_answer": False},
            "helpful_nonrevealing": {
                "text": "Use the object words in the question together. An object that matches only part of that description is a different object.",
                "reveals_answer": False,
            },
        },
    }


def _row(record: dict, context: str, summed: dict[str, float], *, generation: str) -> dict:
    user = "\n".join(
        (
            record["contexts"][context]["text"],
            record["question"],
            "Reply with one of these colors: red, blue, green, yellow.",
        )
    )
    return assemble_row(
        model_key="smolvlm2",
        model_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        revision="7b375e1b73b11138ff12fe22c8f2822d8fe03467",
        processor_class="SmolVLMProcessor",
        image_processor_class="SmolVLMImageProcessor",
        dtype="float32",
        device="mps",
        scene_id=record["scene_id"],
        counterfactual_group_id=record["counterfactual_group_id"],
        context=context,
        answer=record["answer"],
        summed_nll=summed,
        mean_nll=summed,
        token_ids=EXPECTED_TOKEN_IDS["smolvlm2"],
        continuations={color: " " + color for color in COLORS},
        answer_prefix=SMOL_ANSWER_PREFIX,
        generation_text=generation,
        generation_truncated=False,
        seconds=0.01,
        user_text=user,
    )
