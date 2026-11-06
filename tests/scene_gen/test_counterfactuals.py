"""Matched pairs: same question, one controlled edit, two answers."""

from __future__ import annotations

from collections import defaultdict

from graph_oracle import graph_answer

# Typed by hand from the design table. Not read back from the generator's oracle.
EXPECTED_ANSWERS = {
    "dev-attribute-00-a": "red",
    "dev-attribute-00-b": "blue",
    "dev-attribute-01-a": "green",
    "dev-attribute-01-b": "yellow",
    "dev-attribute-02-a": "red",
    "dev-attribute-02-b": "green",
    "dev-attribute-03-a": "blue",
    "dev-attribute-03-b": "yellow",
    "dev-counting-00-a": "two",
    "dev-counting-00-b": "three",
    "dev-counting-01-a": "two",
    "dev-counting-01-b": "three",
    "dev-counting-02-a": "two",
    "dev-counting-02-b": "three",
    "dev-counting-03-a": "two",
    "dev-counting-03-b": "three",
    "dev-spatial-00-a": "red",
    "dev-spatial-00-b": "blue",
    "dev-spatial-01-a": "green",
    "dev-spatial-01-b": "yellow",
    "dev-spatial-02-a": "red",
    "dev-spatial-02-b": "green",
    "dev-spatial-03-a": "blue",
    "dev-spatial-03-b": "yellow",
}


def test_hand_table_and_independent_reader_agree(records: list[dict]) -> None:
    found = {record["scene_id"]: record["answer"] for record in records}
    assert found == EXPECTED_ANSWERS
    for record in records:
        assert graph_answer(record) == record["answer"]


def _by_id(record: dict) -> dict[str, dict]:
    return {obj["object_id"]: obj for obj in record["objects"]}


def test_attribute_and_counting_pairs_change_one_color(records: list[dict]) -> None:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record["task_family"] == "spatial":
            continue
        grouped[record["counterfactual_group_id"]].append(record)
    assert len(grouped) == 8
    for members in grouped.values():
        members.sort(key=lambda record: record["scene_id"])
        left, right = _by_id(members[0]), _by_id(members[1])
        assert left.keys() == right.keys()
        changed = [object_id for object_id in left if left[object_id] != right[object_id]]
        assert len(changed) == 1
        object_id = changed[0]
        assert left[object_id]["position"] == right[object_id]["position"]
        assert left[object_id]["category"] == right[object_id]["category"]
        assert left[object_id]["attributes"]["size"] == right[object_id]["attributes"]["size"]
        assert left[object_id]["attributes"]["color"] != right[object_id]["attributes"]["color"]


def test_spatial_pairs_swap_two_positions(records: list[dict]) -> None:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        if record["task_family"] != "spatial":
            continue
        grouped[record["counterfactual_group_id"]].append(record)
    assert len(grouped) == 4
    for members in grouped.values():
        members.sort(key=lambda record: record["scene_id"])
        left, right = _by_id(members[0]), _by_id(members[1])
        assert left.keys() == right.keys()
        moved = []
        for object_id in left:
            assert left[object_id]["category"] == right[object_id]["category"]
            assert left[object_id]["attributes"] == right[object_id]["attributes"]
            if left[object_id]["position"] != right[object_id]["position"]:
                moved.append(object_id)
        assert len(moved) == 2
        first, second = moved
        assert left[first]["position"] == right[second]["position"]
        assert left[second]["position"] == right[first]["position"]
