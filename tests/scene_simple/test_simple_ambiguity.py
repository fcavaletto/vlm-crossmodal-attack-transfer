"""Ambiguity flags fire on broken records and stay empty on the sample."""

from __future__ import annotations

import copy

from scene_simple.checks import ambiguity_flags
from scene_simple.geometry import center


def test_development_scenes_have_no_ambiguity_flags(records: list[dict]) -> None:
    for record in records:
        assert ambiguity_flags(record) == []


def _broken(records: list[dict]) -> dict:
    return copy.deepcopy(records[0])


def test_wrong_answer_is_flagged(records: list[dict]) -> None:
    record = _broken(records)
    record["answer"] = "blue" if record["answer"] != "blue" else "green"
    assert ambiguity_flags(record) == ["answer_disagrees_with_scene"]


def test_duplicate_referent_is_flagged(records: list[dict]) -> None:
    record = _broken(records)
    target = next(obj for obj in record["objects"] if obj["object_id"] == "o0")
    partner = next(obj for obj in record["objects"] if obj["object_id"] == "o2")
    partner["category"] = target["category"]
    flags = ambiguity_flags(record)
    assert "non_unique_referent" in flags
    assert "missing_row_decoy" in flags


def test_missing_shape_decoy_is_flagged(records: list[dict]) -> None:
    record = _broken(records)
    decoy = next(obj for obj in record["objects"] if obj["object_id"] == "o1")
    fourth = next(obj for obj in record["objects"] if obj["object_id"] == "o3")
    decoy["category"] = fourth["category"]
    flags = ambiguity_flags(record)
    assert "missing_shape_decoy" in flags
    assert "answer_disagrees_with_scene" not in flags


def test_shape_decoy_matching_the_answer_is_flagged(records: list[dict]) -> None:
    record = _broken(records)
    decoy = next(obj for obj in record["objects"] if obj["object_id"] == "o1")
    decoy["attributes"]["color"] = record["answer"]
    flags = ambiguity_flags(record)
    assert "shape_decoy_matches_answer" in flags
    assert "answer_disagrees_with_scene" not in flags


def test_row_decoy_matching_the_answer_is_flagged(records: list[dict]) -> None:
    record = _broken(records)
    partner = next(obj for obj in record["objects"] if obj["object_id"] == "o2")
    partner["attributes"]["color"] = record["answer"]
    flags = ambiguity_flags(record)
    assert "row_decoy_matches_answer" in flags
    assert "answer_disagrees_with_scene" not in flags


def test_uneven_rows_are_flagged(records: list[dict]) -> None:
    record = _broken(records)
    record["objects"] = [obj for obj in record["objects"] if obj["object_id"] != "o3"]
    flags = ambiguity_flags(record)
    assert "row_occupancy_names_the_query" in flags


def test_shared_cell_and_cramped_row_are_flagged(records: list[dict]) -> None:
    record = _broken(records)
    target = next(obj for obj in record["objects"] if obj["object_id"] == "o0")
    partner = next(obj for obj in record["objects"] if obj["object_id"] == "o2")
    partner["position"]["col"] = target["position"]["col"]
    partner["position"]["cx"], partner["position"]["cy"] = center(
        partner["position"]["col"], partner["position"]["row"]
    )
    flags = ambiguity_flags(record)
    assert "shared_cell" in flags
    assert "overlapping_boxes" in flags
    assert "cramped_same_row" in flags


def test_hand_edited_pixel_center_is_flagged(records: list[dict]) -> None:
    record = _broken(records)
    record["objects"][0]["position"]["cx"] += 1
    assert "position_not_derived_from_cell" in ambiguity_flags(record)


def test_unknown_shape_unparsed_question_and_empty_scene(records: list[dict]) -> None:
    unknown = _broken(records)
    unknown["objects"][3]["category"] = "diamond"
    assert "unknown_attribute" in ambiguity_flags(unknown)

    unparsed = _broken(records)
    unparsed["question"] = "How many red objects are there?"
    flags = ambiguity_flags(unparsed)
    assert "unparsed_question" in flags
    assert "answer_disagrees_with_scene" in flags

    empty = _broken(records)
    empty["objects"] = []
    assert ambiguity_flags(empty) == ["empty_scene"]


def test_duplicate_object_id_is_flagged(records: list[dict]) -> None:
    record = _broken(records)
    record["objects"][1]["object_id"] = record["objects"][0]["object_id"]
    assert "duplicate_object_id" in ambiguity_flags(record)


def test_cell_outside_the_grid_is_flagged(records: list[dict]) -> None:
    record = _broken(records)
    record["objects"][3]["position"]["col"] = 4
    assert "object_outside_canvas" in ambiguity_flags(record)
