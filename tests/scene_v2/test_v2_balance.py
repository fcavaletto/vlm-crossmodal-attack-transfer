"""Balance, file contract, and the fields a later loader will read."""

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
SCENE_IDS = [
    f"dev2-{family}-{index:02d}-{arm}"
    for family in ("attribute", "counting", "spatial")
    for index in range(12)
    for arm in ("a", "b")
]


def test_family_balance_and_factor_coverage(records: list[dict]) -> None:
    assert [record["scene_id"] for record in records] == SCENE_IDS
    assert Counter(record["task_family"] for record in records) == {
        "attribute": 24,
        "counting": 24,
        "spatial": 24,
    }
    assert len({record["counterfactual_group_id"] for record in records}) == 36
    assert Counter(record["answer"] for record in records if record["task_family"] == "attribute") == {
        "red": 6,
        "blue": 6,
        "green": 6,
        "yellow": 6,
    }
    assert Counter(record["answer"] for record in records if record["task_family"] == "spatial") == {
        "red": 6,
        "blue": 6,
        "green": 6,
        "yellow": 6,
    }
    assert Counter(record["answer"] for record in records if record["task_family"] == "counting") == {
        "one": 4,
        "two": 8,
        "three": 8,
        "four": 4,
    }
    descriptors = Counter()
    for record in records:
        if record["task_family"] == "counting":
            continue
        match = re.search(r"(large|small) (circle|square|triangle)", record["question"])
        assert match is not None
        descriptors[(record["task_family"], match.group(1), match.group(2))] += 1
    for family in ("attribute", "spatial"):
        for size in ("large", "small"):
            for shape in ("circle", "square", "triangle"):
                assert descriptors[(family, size, shape)] == 4
    queried = Counter()
    for record in records:
        if record["task_family"] != "counting":
            continue
        queried[record["question"].split()[2]] += 1
    assert queried == {"red": 6, "blue": 6, "green": 6, "yellow": 6}


def test_object_count_does_not_name_the_family(records: list[dict]) -> None:
    by_count: dict[int, set[str]] = {}
    by_family: dict[str, set[int]] = {}
    for record in records:
        count = len(record["objects"])
        by_count.setdefault(count, set()).add(record["task_family"])
        by_family.setdefault(record["task_family"], set()).add(count)
    assert set(by_count) == {4, 5, 6, 7, 8}
    for families in by_count.values():
        assert len(families) >= 2
    for counts in by_family.values():
        assert len(counts) >= 2
    assert 4 not in by_family["spatial"]
    assert 8 not in by_family["attribute"]


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
        assert (generated / record["image_path"]).is_file()
        assert record["ambiguity_flags"] == []
        assert isinstance(record["seed"], int)
        index = record["scene_id"].split("-")[2]
        family = record["task_family"]
        assert record["counterfactual_group_id"] == f"cf2-{family}-{index}"
        if record["task_family"] == "counting":
            assert record["answer_aliases"] == [
                record["answer"],
                {"one": "1", "two": "2", "three": "3", "four": "4"}[record["answer"]],
            ]
        else:
            assert record["answer_aliases"] == [record["answer"]]
        for obj in record["objects"]:
            assert set(obj) == OBJECT_KEYS
            assert set(obj["attributes"]) == {"color", "size"}
            assert set(obj["position"]) == {"col", "cx", "cy", "row"}
            col, row = obj["position"]["col"], obj["position"]["row"]
            assert obj["position"]["cx"] == col * 128 + 64
            assert obj["position"]["cy"] == row * 128 + 64


def test_metadata_describes_the_sample(generated) -> None:
    metadata = json.loads((generated / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["name"] == "study2_development_v2"
    assert metadata["scene_count"] == 72
    assert metadata["split"] == "development"
    assert metadata["counterfactual_group_count"] == 36
    assert metadata["seed_domain"] == "study2-development-v2"
    assert metadata["image_size"] == [512, 384]
    assert metadata["grid"] == {"cols": 4, "rows": 3, "cell": [128, 128]}
    assert metadata["pillow"] == "11.3.0"
    assert metadata["contact_sheet_is_model_input"] is False
    assert metadata["replaces_checkpoint_1_sample"] is False
    assert metadata["checkpoint_1_fixture_dir"] == "fixtures/development"
    assert metadata["counting_transitions"] == {"1-2": 4, "2-3": 4, "3-4": 4}
    assert metadata["generator"] == "scene_v2.generate"
    assert "No model was run" in metadata["caution"]
