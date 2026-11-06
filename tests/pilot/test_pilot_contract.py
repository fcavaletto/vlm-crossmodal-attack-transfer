"""Integration checks that do not load SmolVLM2."""

from __future__ import annotations

import pytest

from analysis.pilot_metrics import build_pilot_summary
from analysis.pilot_schema import SchemaError, validate_pilot_bundle
from pilot.constants import EXPECTED_SCENE_IDS, MANIFEST_SHA256, N_SCENES
from pilot.manifest import load_committed_manifest, repo_root
from pilot.prompt import user_prompt
from pilot.rows import gradient_stats, judge_parse, make_alignment_row, make_prediction_row
from pilot.run import require_mps
from scene_gen.normalize import leakage_aliases
from scene_gen.vocabulary import COLORS, COUNT_WORDS
from vlm_iface.parsing import COUNT_DIGIT_ALIASES, task_vocabulary


def test_parser_vocabulary_matches_the_scene_generator():
    assert task_vocabulary("attribute") == COLORS
    assert task_vocabulary("spatial") == COLORS
    assert task_vocabulary("counting") == tuple(COUNT_WORDS[count] for count in (1, 2, 3, 4))
    assert COUNT_DIGIT_ALIASES == {"1": "one", "2": "two", "3": "three", "4": "four"}


def test_committed_manifest_is_the_development_sample():
    path, rows = load_committed_manifest(repo_root())
    assert path.name == "manifest.jsonl"
    assert len(rows) == N_SCENES
    assert [row["scene_id"] for row in rows] == sorted(EXPECTED_SCENE_IDS)
    import hashlib

    assert hashlib.sha256(path.read_bytes()).hexdigest() == MANIFEST_SHA256


def test_user_prompt_is_the_approved_join_and_preserves_leakage():
    _, rows = load_committed_manifest(repo_root())
    by_group: dict[str, list[dict]] = {}
    for row in rows:
        by_group.setdefault(row["counterfactual_group_id"], []).append(row)
    assert len(by_group) == 12
    for group in by_group.values():
        assert len(group) == 2
        for context in ("neutral", "helpful_nonrevealing", "contradictory"):
            assert group[0]["contexts"][context]["text"] == group[1]["contexts"][context]["text"]
        assert group[0]["contexts"]["answer_revealing"]["text"] != group[1]["contexts"]["answer_revealing"]["text"]
    for row in rows:
        for context, payload in row["contexts"].items():
            prompt = user_prompt(payload["text"], row["question"])
            assert prompt == payload["text"] + "\n" + row["question"]
            assert "Context:" not in prompt
            assert "Question:" not in prompt
            if context != "answer_revealing":
                assert leakage_aliases(prompt, row["answer_aliases"]) == []
            else:
                assert row["answer"] in leakage_aliases(prompt, row["answer_aliases"])


def test_cpu_device_is_refused_before_a_model_load():
    with pytest.raises(RuntimeError, match="not an MPS result"):
        require_mps("cpu")
    with pytest.raises(RuntimeError, match="not an MPS result"):
        make_prediction_row(
            record=_record("dev-attribute-00-a", "red"),
            context="neutral",
            prompt="Several simple objects appear on a plain background.\nWhat color is the large circle?",
            raw_output="red",
            parsed_answer="red",
            parse_reasons=[],
            invalid=False,
            scores=_scores("red"),
            device="cpu",
        )


def test_judge_splits_correct_valid_wrong_and_invalid():
    assert judge_parse("two", False, "two", ["two", "2"])["correct"] is True
    wrong = judge_parse("three", False, "two", ["two", "2"])
    assert wrong == {"correct": False, "valid_wrong": True, "invalid": False}
    assert judge_parse(None, True, "red", ["red"])["invalid"] is True


def test_summary_keeps_invalid_separate_from_valid_wrong():
    rows = _bundle()
    bundle = validate_pilot_bundle(rows["predictions"], rows["gradients"], rows["alignments"])
    summary = build_pilot_summary(bundle["predictions"], bundle["gradients"], bundle["alignments"])
    neutral = summary["clean_by_context"]["neutral"]
    assert neutral["n_correct"] == 2
    assert neutral["n_valid_wrong"] == 0
    assert neutral["n_invalid"] == 0
    helpful = summary["clean_by_context"]["helpful_nonrevealing"]
    assert helpful["n_valid_wrong"] == 2
    assert helpful["accuracy"]["value"] == 0.0
    contradictory = summary["clean_by_context"]["contradictory"]
    assert contradictory["n_invalid"] == 1
    assert contradictory["n_valid_wrong"] == 1
    assert contradictory["n_correct"] == 0
    visual = summary["counterfactual"]["by_context"]["neutral"]
    assert visual["relevant_to_visual_edit"] is True
    assert visual["parsed_answer_changed"]["numerator"] == 1
    assert visual["correct_on_both_arms"]["numerator"] == 1
    assert visual["changed_to_correct_counterfactual"]["numerator"] == 1
    unchanged = summary["counterfactual"]["by_context"]["helpful_nonrevealing"]
    assert unchanged["unchanged_despite_edit"]["numerator"] == 1
    assert summary["counterfactual"]["definition"].startswith("An answer change")
    assert summary["h_star_fitted"] is False
    assert summary["attacks_run"] is False
    flag_ids = {flag["flag_id"] for flag in summary["quality_flags"]}
    assert "counterfactual_unchanged_answer" in flag_ids
    assert "weak_clean_accuracy" in flag_ids
    assert all(flag["kept"] is True for flag in summary["quality_flags"])
    broken = dict(rows["predictions"][0])
    broken["correct"] = True
    broken["invalid"] = True
    broken["valid_wrong"] = False
    with pytest.raises(SchemaError, match="partition"):
        validate_pilot_bundle([broken], rows["gradients"], rows["alignments"])


def test_gradient_stats_require_the_scene_shape():
    import numpy as np

    gradient = np.zeros((3, 384, 512), dtype=np.float32)
    gradient[0, 0, 0] = 0.5
    stats = gradient_stats(gradient)
    assert stats["finite"] is True
    assert stats["height"] == 384 and stats["width"] == 512
    assert stats["nonzero_fraction"] > 0
    with pytest.raises(ValueError, match="512"):
        gradient_stats(np.zeros((3, 128, 128), dtype=np.float32))


def _record(scene_id: str, answer: str) -> dict:
    other = "blue" if answer == "red" else "red"
    return {
        "scene_id": scene_id,
        "counterfactual_group_id": "cf-attribute-00",
        "task_family": "attribute",
        "answer": answer,
        "answer_aliases": [answer],
        "question": "What color is the large circle?",
        "ambiguity_flags": [],
        "objects": [{}, {}, {}, {}],
        "contexts": {
            "neutral": {"text": "Several simple objects appear on a plain background."},
            "helpful_nonrevealing": {
                "text": "Identify the single object with the size and shape named in the question, then name its color."
            },
            "contradictory": {"text": f"The large circle is {other}."},
            "answer_revealing": {"text": f"The large circle is {answer}."},
        },
    }


def _scores(best: str) -> dict:
    scores = {}
    for word in COLORS:
        scores[word] = {
            "sum_nll": 0.1 if word == best else 1.0,
            "token_count": 1,
            "mean_nll": 0.1 if word == best else 1.0,
            "library_mean_abs_diff": 0.0,
        }
    return scores


def _bundle() -> dict:
    import numpy as np

    left = _record("dev-attribute-00-a", "red")
    right = _record("dev-attribute-00-b", "blue")
    outputs = {
        ("dev-attribute-00-a", "neutral"): ("red", "red", False, []),
        ("dev-attribute-00-b", "neutral"): ("blue", "blue", False, []),
        ("dev-attribute-00-a", "helpful_nonrevealing"): ("green", "green", False, []),
        ("dev-attribute-00-b", "helpful_nonrevealing"): ("green", "green", False, []),
        ("dev-attribute-00-a", "contradictory"): ("", None, True, ["refusal"]),
        ("dev-attribute-00-b", "contradictory"): ("red", "red", False, []),
        ("dev-attribute-00-a", "answer_revealing"): ("red", "red", False, []),
        ("dev-attribute-00-b", "answer_revealing"): ("blue", "blue", False, []),
    }
    predictions = []
    for record in (left, right):
        for context in record["contexts"]:
            raw, parsed, invalid, reasons = outputs[(record["scene_id"], context)]
            best = parsed if isinstance(parsed, str) else record["answer"]
            predictions.append(
                make_prediction_row(
                    record=record,
                    context=context,
                    prompt=user_prompt(record["contexts"][context]["text"], record["question"]),
                    raw_output=raw,
                    parsed_answer=parsed,
                    parse_reasons=reasons,
                    invalid=invalid,
                    scores=_scores(best),
                    device="mps",
                )
            )
    gradient = np.zeros((3, 384, 512), dtype=np.float64)
    gradient[:, :, :256] = 0.25
    other = np.zeros((3, 384, 512), dtype=np.float64)
    other[:, :, 256:] = -0.5
    gradients = []
    alignments = []
    for record, source_array, target_array in (
        (left, gradient, other),
        (right, other, gradient),
    ):
        source_stats = gradient_stats(source_array)
        target_stats = gradient_stats(target_array)
        source_row = _gradient_row(record, "neutral", source_stats)
        target_row = _gradient_row(record, "helpful_nonrevealing", target_stats)
        gradients.extend((source_row, target_row))
        alignments.append(
            make_alignment_row(
                scene_id=record["scene_id"],
                counterfactual_group_id=record["counterfactual_group_id"],
                task_family=record["task_family"],
                source_context="neutral",
                target_context="helpful_nonrevealing",
                source_gradient=source_array,
                target_gradient=target_array,
                source_stats=source_row,
                target_stats=target_row,
                device="mps",
            )
        )
        alignments.append(
            make_alignment_row(
                scene_id=record["scene_id"],
                counterfactual_group_id=record["counterfactual_group_id"],
                task_family=record["task_family"],
                source_context="helpful_nonrevealing",
                target_context="neutral",
                source_gradient=target_array,
                target_gradient=source_array,
                source_stats=target_row,
                target_stats=source_row,
                device="mps",
            )
        )
    return {"predictions": predictions, "gradients": gradients, "alignments": alignments}


def _gradient_row(record: dict, context: str, stats: dict) -> dict:
    from analysis.pilot_schema import LABEL, MODEL_ID, MODEL_REVISION, PREPROCESS

    return {
        "label": LABEL,
        "scene_id": record["scene_id"],
        "counterfactual_group_id": record["counterfactual_group_id"],
        "task_family": record["task_family"],
        "context": context,
        "finite": stats["finite"],
        "nonzero_fraction": stats["nonzero_fraction"],
        "l1": stats["l1"],
        "l2": stats["l2"],
        "linf": stats["linf"],
        "height": stats["height"],
        "width": stats["width"],
        "seconds": 1.0,
        "repeat_max_abs_diff": None,
        "device": "mps",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "preprocess": PREPROCESS,
    }
