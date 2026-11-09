"""Balance, the manifest contract, and the one graded family."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

import PIL

RECORD_KEYS = {
    "ambiguity_flags",
    "answer",
    "answer_aliases",
    "contexts",
    "counterfactual_group_id",
    "image_path",
    "objects",
    "question",
    "scene_id",
    "seed",
    "split",
    "task_family",
}
CONTEXT_KEYS = {"answer_revealing", "contradictory", "helpful_nonrevealing", "neutral"}
OBJECT_KEYS = {"attributes", "category", "object_id", "position"}
SCENE_IDS = [f"dev4-ref-{index:02d}-{arm}" for index in range(48) for arm in ("a", "b")]
_QUESTION = re.compile(
    r"^What color is the (large|small) (circle|square|triangle) in the (top|bottom) row\?$"
)


def test_label_and_factor_balance(records: list[dict]) -> None:
    assert [record["scene_id"] for record in records] == SCENE_IDS
    assert {record["split"] for record in records} == {"development"}
    assert Counter(record["task_family"] for record in records) == {"referent_color": 96}
    assert "counting" not in {record["task_family"] for record in records}
    assert len({record["counterfactual_group_id"] for record in records}) == 48
    assert Counter(record["answer"] for record in records) == {
        "red": 24,
        "blue": 24,
        "green": 24,
        "yellow": 24,
    }
    shapes: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    rows: Counter[str] = Counter()
    by_question: dict[str, Counter[str]] = defaultdict(Counter)
    for record in records:
        match = _QUESTION.fullmatch(record["question"])
        assert match is not None
        sizes[match.group(1)] += 1
        shapes[match.group(2)] += 1
        rows[match.group(3)] += 1
        by_question[record["question"]][record["answer"]] += 1
    assert shapes == {"circle": 32, "square": 32, "triangle": 32}
    assert sizes == {"large": 48, "small": 48}
    assert rows == {"top": 48, "bottom": 48}
    assert len(by_question) == 12
    for question, counts in by_question.items():
        assert dict(counts) == {"red": 2, "blue": 2, "green": 2, "yellow": 2}, (question, counts)
    foils = Counter(
        record["contexts"]["contradictory"]["text"].rsplit(" ", 1)[-1].rstrip(".") for record in records
    )
    assert foils == {"red": 24, "blue": 24, "green": 24, "yellow": 24}


def test_swap_partner_imbalance_is_explicit(records: list[dict]) -> None:
    """The closed form does not balance the partner's shape or its row.

    Answers, queried factors, foils, and color roles do balance. These two
    counts are the remainder.
    """
    shapes: Counter[str] = Counter()
    rows: Counter[str] = Counter()
    for record in records:
        partner = next(obj for obj in record["objects"] if obj["object_id"] == "o4")
        shapes[partner["category"]] += 1
        rows["top" if partner["position"]["row"] == 0 else "bottom"] += 1
    assert shapes == {"triangle": 42, "circle": 32, "square": 22}
    assert rows == {"top": 54, "bottom": 42}


def test_one_family_so_object_count_cannot_name_one(records: list[dict]) -> None:
    families = {record["task_family"] for record in records}
    assert families == {"referent_color"}
    assert Counter(len(record["objects"]) for record in records) == {8: 96}
    for record in records:
        assert not record["question"].casefold().startswith("how many")
        rows = Counter(obj["position"]["row"] for obj in record["objects"])
        assert rows == {0: 4, 1: 4}


def test_manifest_contract(records: list[dict], generated) -> None:
    for record in records:
        assert set(record) == RECORD_KEYS
        assert record["split"] == "development"
        assert set(record["contexts"]) == CONTEXT_KEYS
        for context in record["contexts"].values():
            assert set(context) == {"reveals_answer", "text"}
            assert isinstance(context["text"], str) and context["text"]
            assert isinstance(context["reveals_answer"], bool)
        assert record["image_path"] == f"images/{record['scene_id']}.png"
        assert record["image_path"] != "contact_sheet.png"
        assert (generated / record["image_path"]).is_file()
        assert record["ambiguity_flags"] == []
        assert isinstance(record["seed"], int)
        index = record["scene_id"].split("-")[2]
        assert record["counterfactual_group_id"] == f"cf4-ref-{index}"
        assert record["answer_aliases"] == [record["answer"]]
        assert [obj["object_id"] for obj in record["objects"]] == [f"o{number}" for number in range(8)]
        for obj in record["objects"]:
            assert set(obj) == OBJECT_KEYS
            assert set(obj["attributes"]) == {"color", "size"}
            assert obj["attributes"]["size"] in {"large", "small"}
            assert set(obj["position"]) == {"col", "cx", "cy", "row"}
            col, row = obj["position"]["col"], obj["position"]["row"]
            assert obj["position"]["cx"] == col * 128 + 64
            assert obj["position"]["cy"] == row * 192 + 96


def test_metadata_describes_the_sample(generated, repo_root) -> None:
    metadata = json.loads((generated / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["name"] == "study2_development_v4"
    assert metadata["scene_count"] == 96
    assert metadata["split"] == "development"
    assert metadata["counterfactual_group_count"] == 48
    assert metadata["seed_domain"] == "study2-development-v4"
    assert metadata["image_size"] == [512, 384]
    assert metadata["grid"] == {"cols": 4, "rows": 2, "cell": [128, 192]}
    assert metadata["radii"] == {"large": 60, "small": 28}
    assert metadata["pillow"] == PIL.__version__
    assert metadata["task_families"] == ["referent_color"]
    assert metadata["task_counts"] == {"referent_color": 96}
    assert metadata["answer_counts"]["referent_color"] == {
        "blue": 24,
        "green": 24,
        "red": 24,
        "yellow": 24,
    }
    assert metadata["object_counts_by_family"] == {"referent_color": {"8": 96}}
    assert metadata["queried_shape_counts"] == {"circle": 32, "square": 32, "triangle": 32}
    assert metadata["queried_size_counts"] == {"large": 48, "small": 48}
    assert metadata["queried_row_counts"] == {"bottom": 48, "top": 48}
    assert metadata["foil_color_counts"] == {"blue": 24, "green": 24, "red": 24, "yellow": 24}
    assert metadata["swap_partner_shape_counts"] == {"circle": 32, "square": 22, "triangle": 42}
    assert metadata["swap_partner_row_counts"] == {"bottom": 42, "top": 54}
    assert metadata["color_count_pattern"] == {"1": 1, "2": 2, "3": 1}
    assert metadata["answer_color_count"] == 2
    assert metadata["counting_is_a_graded_family"] is False
    assert metadata["protocol_status"] == "prospective"
    assert metadata["model_outputs_used_to_choose_scenes"] is False
    assert metadata["held_out_scenes_generated"] is False
    assert metadata["model_was_run"] is False
    assert metadata["contact_sheet_is_model_input"] is False
    assert metadata["replaces_checkpoint_1_sample"] is False
    assert metadata["replaces_checkpoint_1_5_sample"] is False
    assert metadata["replaces_checkpoint_1_6_sample"] is False
    assert metadata["generator"] == "scene_v4.generate"
    assert "context text, a newline, then the question" in metadata["prompt_join"]
    assert len(metadata["contact_sheet_scene_ids"]) == 96
    assert metadata["contact_sheet_scene_ids"][0] == "dev4-ref-00-a"
    assert metadata["contact_sheet_scene_ids"][12] == "dev4-ref-00-b"
    assert (repo_root / "fixtures" / "development" / "manifest.jsonl").is_file()
    assert (repo_root / "fixtures" / "development_v2" / "manifest.jsonl").is_file()
    assert (repo_root / "fixtures" / "development_simple" / "manifest.jsonl").is_file()
