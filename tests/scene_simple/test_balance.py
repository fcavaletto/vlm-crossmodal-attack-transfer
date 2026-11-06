"""Balance, the manifest contract, and the one graded family."""

from __future__ import annotations

import json
import re
from collections import Counter

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
SCENE_IDS = [f"dev3-row-{index:02d}-{arm}" for index in range(12) for arm in ("a", "b")]
_QUESTION = re.compile(r"^What color is the (circle|square|triangle) in the (top|bottom) row\?$")


def test_label_and_factor_balance(records: list[dict]) -> None:
    assert [record["scene_id"] for record in records] == SCENE_IDS
    assert {record["split"] for record in records} == {"development"}
    assert Counter(record["task_family"] for record in records) == {"row_color": 24}
    assert "counting" not in {record["task_family"] for record in records}
    assert len({record["counterfactual_group_id"] for record in records}) == 12
    assert Counter(record["answer"] for record in records) == {
        "red": 6,
        "blue": 6,
        "green": 6,
        "yellow": 6,
    }
    shapes = Counter()
    rows = Counter()
    for record in records:
        match = _QUESTION.fullmatch(record["question"])
        assert match is not None
        shapes[match.group(1)] += 1
        rows[match.group(2)] += 1
    assert shapes == {"circle": 8, "square": 8, "triangle": 8}
    assert rows == {"top": 12, "bottom": 12}
    foils = Counter(
        record["contexts"]["contradictory"]["text"].rsplit(" ", 1)[-1].rstrip(".") for record in records
    )
    assert foils == {"red": 6, "blue": 6, "green": 6, "yellow": 6}


def test_one_family_so_object_count_cannot_name_one(records: list[dict]) -> None:
    """Object count named the family when several families had private counts.

    This sample has one family. A count therefore has no family to select.
    Counting is not that family.
    """
    families = {record["task_family"] for record in records}
    assert families == {"row_color"}
    assert Counter(len(record["objects"]) for record in records) == {4: 24}
    for record in records:
        assert not record["question"].casefold().startswith("how many")
        rows = Counter(obj["position"]["row"] for obj in record["objects"])
        assert rows == {0: 2, 1: 2}


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
        assert record["counterfactual_group_id"] == f"cf3-row-{index}"
        assert record["answer_aliases"] == [record["answer"]]
        for obj in record["objects"]:
            assert set(obj) == OBJECT_KEYS
            assert set(obj["attributes"]) == {"color", "size"}
            assert obj["attributes"]["size"] == "large"
            assert set(obj["position"]) == {"col", "cx", "cy", "row"}
            col, row = obj["position"]["col"], obj["position"]["row"]
            assert obj["position"]["cx"] == col * 128 + 64
            assert obj["position"]["cy"] == row * 192 + 96


def test_metadata_describes_the_sample(generated) -> None:
    metadata = json.loads((generated / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["name"] == "study2_development_simple"
    assert metadata["scene_count"] == 24
    assert metadata["split"] == "development"
    assert metadata["counterfactual_group_count"] == 12
    assert metadata["seed_domain"] == "study2-development-simple"
    assert metadata["image_size"] == [512, 384]
    assert metadata["grid"] == {"cols": 4, "rows": 2, "cell": [128, 192]}
    assert metadata["radius"] == 60
    assert metadata["pillow"] == "11.3.0"
    assert metadata["task_families"] == ["row_color"]
    assert metadata["task_counts"] == {"row_color": 24}
    assert metadata["answer_counts"]["row_color"] == {
        "blue": 6,
        "green": 6,
        "red": 6,
        "yellow": 6,
    }
    assert metadata["object_counts_by_family"] == {"row_color": {"4": 24}}
    assert metadata["queried_shape_counts"] == {"circle": 8, "square": 8, "triangle": 8}
    assert metadata["queried_row_counts"] == {"bottom": 12, "top": 12}
    assert metadata["foil_color_counts"] == {"blue": 6, "green": 6, "red": 6, "yellow": 6}
    assert metadata["counting_is_a_graded_family"] is False
    assert metadata["contact_sheet_is_model_input"] is False
    assert metadata["model_was_run"] is False
    assert metadata["replaces_checkpoint_1_sample"] is False
    assert metadata["replaces_checkpoint_1_5_sample"] is False
    assert metadata["generator"] == "scene_simple.generate"
    assert "context text, a newline, then the question" in metadata["prompt_join"]
    assert len(metadata["contact_sheet_scene_ids"]) == 24
    assert metadata["contact_sheet_scene_ids"][0] == "dev3-row-00-a"
    assert metadata["contact_sheet_scene_ids"][12] == "dev3-row-00-b"
