"""The flagger has to fire on broken scenes. Empty flags on the sample are not enough."""

from __future__ import annotations

import copy

from scene_gen.checks import ambiguity_flags
from scene_gen.geometry import MIN_BOX_GAP, bounding_box, boxes_too_close, center


def _attribute(records: list[dict]) -> dict:
    return copy.deepcopy(next(record for record in records if record["task_family"] == "attribute"))


def _spatial(records: list[dict]) -> dict:
    return copy.deepcopy(next(record for record in records if record["scene_id"] == "dev-spatial-00-a"))


def _counting(records: list[dict]) -> dict:
    return copy.deepcopy(
        next(record for record in records if record["question"] == "How many red objects are there?")
    )


def test_development_scenes_have_no_ambiguity_flags(records: list[dict]) -> None:
    for record in records:
        assert ambiguity_flags(record) == []
        assert record["ambiguity_flags"] == []


def test_development_boxes_are_separated(records: list[dict]) -> None:
    for record in records:
        boxes = [
            bounding_box(obj["position"]["col"], obj["position"]["row"], obj["attributes"]["size"])
            for obj in record["objects"]
        ]
        for index, box_a in enumerate(boxes):
            for box_b in boxes[index + 1 :]:
                assert not boxes_too_close(box_a, box_b)


def test_box_gap_helper_sees_overlap_and_clearance() -> None:
    assert boxes_too_close((0, 0, 10, 10), (5, 5, 15, 15))
    assert boxes_too_close((0, 0, 10, 10), (12, 0, 22, 10), min_gap=MIN_BOX_GAP)
    assert not boxes_too_close((0, 0, 10, 10), (40, 0, 50, 10), min_gap=MIN_BOX_GAP)


def test_wrong_answer_is_flagged(records: list[dict]) -> None:
    record = _attribute(records)
    record["answer"] = "yellow" if record["answer"] != "yellow" else "green"
    assert ambiguity_flags(record) == ["answer_disagrees_with_scene"]


def test_duplicate_referent_is_flagged(records: list[dict]) -> None:
    record = _attribute(records)
    target = record["objects"][0]
    other = record["objects"][1]
    other["category"] = target["category"]
    other["attributes"]["size"] = target["attributes"]["size"]
    flags = ambiguity_flags(record)
    assert "non_unique_referent" in flags
    assert "answer_disagrees_with_scene" in flags


def test_shared_cell_and_overlap_are_flagged(records: list[dict]) -> None:
    record = _attribute(records)
    record["objects"][1]["position"] = dict(record["objects"][0]["position"])
    flags = ambiguity_flags(record)
    assert "shared_cell" in flags
    assert "overlapping_boxes" in flags


def test_hand_edited_pixel_center_is_flagged(records: list[dict]) -> None:
    record = _attribute(records)
    record["objects"][0]["position"]["cx"] += 5
    assert "position_not_derived_from_cell" in ambiguity_flags(record)


def test_missing_left_neighbor_is_flagged(records: list[dict]) -> None:
    record = _spatial(records)
    neighbor = next(obj for obj in record["objects"] if obj["object_id"] == "o0")
    neighbor["position"]["col"] = 1
    neighbor["position"]["row"] = 1
    neighbor["position"]["cx"], neighbor["position"]["cy"] = center(1, 1)
    flags = ambiguity_flags(record)
    assert "missing_relation_target" in flags


def test_extra_object_on_the_left_ray_is_flagged(records: list[dict]) -> None:
    record = _spatial(records)
    decoy = next(obj for obj in record["objects"] if obj["object_id"] == "o1")
    decoy["position"]["col"] = 0
    decoy["position"]["row"] = 0
    decoy["position"]["cx"], decoy["position"]["cy"] = center(0, 0)
    flags = ambiguity_flags(record)
    assert "extra_object_on_relation_ray" in flags


def test_zero_count_is_out_of_vocabulary(records: list[dict]) -> None:
    record = _counting(records)
    for obj in record["objects"]:
        if obj["attributes"]["color"] == "red":
            obj["attributes"]["color"] = "blue"
    flags = ambiguity_flags(record)
    assert "count_out_of_vocabulary" in flags


def test_unknown_shape_unparsed_question_and_empty_scene(records: list[dict]) -> None:
    unknown = _attribute(records)
    unknown["objects"][0]["category"] = "star"
    assert "unknown_attribute" in ambiguity_flags(unknown)

    unparsed = _attribute(records)
    unparsed["question"] = "What?"
    assert "unparsed_question" in ambiguity_flags(unparsed)

    empty = _attribute(records)
    empty["objects"] = []
    assert ambiguity_flags(empty) == ["empty_scene"]


def test_duplicate_object_id_is_flagged(records: list[dict]) -> None:
    record = _attribute(records)
    record["objects"][1]["object_id"] = record["objects"][0]["object_id"]
    assert "duplicate_object_id" in ambiguity_flags(record)


def test_cell_outside_the_grid_is_flagged(records: list[dict]) -> None:
    record = _attribute(records)
    record["objects"][0]["position"]["col"] = 9
    assert "object_outside_canvas" in ambiguity_flags(record)
