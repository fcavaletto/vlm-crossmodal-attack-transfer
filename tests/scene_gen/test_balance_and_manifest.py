"""Balance, file contract, and the fields other workstreams will read."""

from __future__ import annotations

import json
from collections import Counter

from graph_oracle import graph_answer

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


def test_family_and_answer_balance(records: list[dict]) -> None:
    assert len(records) == 24
    assert Counter(record["task_family"] for record in records) == {
        "attribute": 8,
        "counting": 8,
        "spatial": 8,
    }
    assert Counter(record["answer"] for record in records if record["task_family"] == "attribute") == {
        "red": 2,
        "blue": 2,
        "green": 2,
        "yellow": 2,
    }
    assert Counter(record["answer"] for record in records if record["task_family"] == "spatial") == {
        "red": 2,
        "blue": 2,
        "green": 2,
        "yellow": 2,
    }
    assert Counter(record["answer"] for record in records if record["task_family"] == "counting") == {
        "two": 4,
        "three": 4,
    }
    queried = Counter()
    for record in records:
        if record["task_family"] != "counting":
            continue
        color = record["question"].removeprefix("How many ").split(" ", 1)[0]
        queried[color] += 1
    assert queried == {"red": 2, "blue": 2, "green": 2, "yellow": 2}
    # Object cardinality identifies the task family. That is a limitation, locked
    # here so the write-up cannot claim the families are matched on object count.
    assert {len(record["objects"]) for record in records if record["task_family"] == "attribute"} == {4}
    assert {len(record["objects"]) for record in records if record["task_family"] == "counting"} == {8}
    assert {len(record["objects"]) for record in records if record["task_family"] == "spatial"} == {5}


def test_manifest_contract(records: list[dict], generated) -> None:
    assert len({record["scene_id"] for record in records}) == 24
    assert len({record["counterfactual_group_id"] for record in records}) == 12
    for record in records:
        assert set(record) == RECORD_KEYS
        assert record["split"] == "development"
        assert set(record["contexts"]) == CONTEXT_KEYS
        for context in record["contexts"].values():
            assert set(context) == {"reveals_answer", "text"}
            assert isinstance(context["text"], str) and context["text"]
            assert isinstance(context["reveals_answer"], bool)
        assert record["image_path"] == f"images/{record['scene_id']}.png"
        assert (generated / record["image_path"]).is_file()
        assert record["ambiguity_flags"] == []
        assert isinstance(record["seed"], int)
        if record["task_family"] == "counting":
            digit = {"two": "2", "three": "3"}[record["answer"]]
            assert record["answer_aliases"] == [record["answer"], digit]
        else:
            assert record["answer_aliases"] == [record["answer"]]
        for obj in record["objects"]:
            assert set(obj) == OBJECT_KEYS
            assert set(obj["attributes"]) == {"color", "size"}
            assert set(obj["position"]) == {"col", "cx", "cy", "row"}
            assert obj["position"]["cx"] == obj["position"]["col"] * 128 + 64
            assert obj["position"]["cy"] == obj["position"]["row"] * 192 + 96
        assert graph_answer(record) == record["answer"]


def test_questions_use_only_the_three_templates(records: list[dict]) -> None:
    for record in records:
        question = record["question"]
        if record["task_family"] == "attribute":
            assert question.startswith("What color is the ")
        elif record["task_family"] == "counting":
            assert question.startswith("How many ") and question.endswith(" objects are there?")
        else:
            assert question.startswith("What color is the object immediately left of the ")
        assert question.endswith("?")


def test_metadata_describes_the_sample(records: list[dict], generated) -> None:
    metadata = json.loads((generated / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["scene_count"] == 24
    assert metadata["split"] == "development"
    assert metadata["seed_domain"] == "study2-development-v1"
    assert metadata["pillow"] == "11.3.0"
    assert metadata["contact_sheet_is_model_input"] is False
    assert metadata["image_size"] == [512, 384]
    hashes = json.loads((generated / "scene_graph_hashes.json").read_text(encoding="utf-8"))
    assert set(hashes["hashes"]) == {record["scene_id"] for record in records}
    assert len(set(hashes["hashes"].values())) == 24
