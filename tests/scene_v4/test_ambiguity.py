"""Ambiguity flags fire on broken records and stay empty on the sample."""

from __future__ import annotations

import copy
from collections import Counter

from scene_v4.checks import ambiguity_flags
from scene_v4.geometry import center
from scene_v4.vocabulary import COLORS


def test_development_scenes_have_no_ambiguity_flags(records: list[dict]) -> None:
    for record in records:
        assert ambiguity_flags(record) == []


def _broken(records: list[dict]) -> dict:
    return copy.deepcopy(records[0])


def _target(record: dict) -> dict:
    return next(obj for obj in record["objects"] if obj["object_id"] == "o0")


def test_wrong_answer_is_flagged(records: list[dict]) -> None:
    record = _broken(records)
    counts = Counter(obj["attributes"]["color"] for obj in record["objects"])
    other = next(color for color, count in counts.items() if count == 2 and color != record["answer"])
    record["answer"] = other
    assert ambiguity_flags(record) == ["answer_disagrees_with_scene"]


def test_duplicate_referent_is_flagged(records: list[dict]) -> None:
    record = _broken(records)
    target = _target(record)
    partner = next(obj for obj in record["objects"] if obj["object_id"] == "o3")
    partner["category"] = target["category"]
    partner["attributes"]["size"] = target["attributes"]["size"]
    flags = ambiguity_flags(record)
    assert "non_unique_referent" in flags


def test_missing_cross_row_decoy_is_flagged(records: list[dict]) -> None:
    record = _broken(records)
    target = _target(record)
    foil = next(obj for obj in record["objects"] if obj["object_id"] == "o1")
    foil["category"] = next(shape for shape in ("circle", "square", "triangle") if shape != target["category"])
    flags = ambiguity_flags(record)
    assert "missing_cross_row_decoy" in flags
    assert "answer_disagrees_with_scene" not in flags


def test_second_cross_row_copy_is_flagged(records: list[dict]) -> None:
    record = _broken(records)
    target = _target(record)
    other_row = 1 - target["position"]["row"]
    extra = next(
        obj
        for obj in record["objects"]
        if obj["object_id"] != "o1" and obj["position"]["row"] == other_row
    )
    extra["category"] = target["category"]
    extra["attributes"]["size"] = target["attributes"]["size"]
    flags = ambiguity_flags(record)
    assert "non_unique_cross_row_decoy" in flags


def test_cross_row_decoy_matching_the_answer_is_flagged(records: list[dict]) -> None:
    record = _broken(records)
    foil = next(obj for obj in record["objects"] if obj["object_id"] == "o1")
    foil["attributes"]["color"] = record["answer"]
    flags = ambiguity_flags(record)
    assert "cross_row_decoy_matches_answer" in flags
    assert "answer_disagrees_with_scene" not in flags


def test_size_decoy_matching_the_answer_is_flagged(records: list[dict]) -> None:
    record = _broken(records)
    target = _target(record)
    for obj in record["objects"]:
        if (
            obj["object_id"] != "o0"
            and obj["category"] == target["category"]
            and obj["position"]["row"] == target["position"]["row"]
            and obj["attributes"]["size"] != target["attributes"]["size"]
        ):
            obj["attributes"]["color"] = record["answer"]
    flags = ambiguity_flags(record)
    assert "size_decoy_matches_answer" in flags
    assert "answer_disagrees_with_scene" not in flags


def test_missing_size_decoy_is_flagged(records: list[dict]) -> None:
    record = _broken(records)
    target = _target(record)
    decoy = next(obj for obj in record["objects"] if obj["object_id"] == "o2")
    decoy["category"] = next(shape for shape in ("circle", "square", "triangle") if shape != target["category"])
    flags = ambiguity_flags(record)
    assert "missing_size_decoy" in flags
    assert "answer_disagrees_with_scene" not in flags


def test_row_decoy_matching_the_answer_is_flagged(records: list[dict]) -> None:
    record = _broken(records)
    target = _target(record)
    for obj in record["objects"]:
        if (
            obj["object_id"] != "o0"
            and obj["attributes"]["size"] == target["attributes"]["size"]
            and obj["position"]["row"] == target["position"]["row"]
            and obj["category"] != target["category"]
        ):
            obj["attributes"]["color"] = record["answer"]
    flags = ambiguity_flags(record)
    assert "row_decoy_matches_answer" in flags
    assert "answer_disagrees_with_scene" not in flags


def test_missing_row_decoy_is_flagged(records: list[dict]) -> None:
    record = _broken(records)
    target = _target(record)
    for obj in record["objects"]:
        if obj["object_id"] == "o0":
            continue
        if (
            obj["attributes"]["size"] == target["attributes"]["size"]
            and obj["position"]["row"] == target["position"]["row"]
        ):
            obj["category"] = target["category"]
            obj["attributes"]["size"] = "small" if target["attributes"]["size"] == "large" else "large"
    flags = ambiguity_flags(record)
    assert "missing_row_decoy" in flags


def test_uneven_rows_are_flagged(records: list[dict]) -> None:
    record = _broken(records)
    record["objects"] = [obj for obj in record["objects"] if obj["object_id"] != "o7"]
    flags = ambiguity_flags(record)
    assert "unequal_row_occupancy" in flags


def test_shared_cell_is_flagged(records: list[dict]) -> None:
    record = _broken(records)
    target = _target(record)
    partner = next(obj for obj in record["objects"] if obj["object_id"] == "o5")
    partner["position"]["col"] = target["position"]["col"]
    partner["position"]["row"] = target["position"]["row"]
    partner["position"]["cx"], partner["position"]["cy"] = center(
        partner["position"]["col"], partner["position"]["row"]
    )
    flags = ambiguity_flags(record)
    assert "shared_cell" in flags
    assert "overlapping_boxes" in flags


def test_adjacent_larges_are_flagged(records: list[dict]) -> None:
    record = _broken(records)
    for obj in record["objects"]:
        if obj["position"]["row"] == 0 and obj["position"]["col"] in {0, 1}:
            obj["attributes"]["size"] = "large"
    flags = ambiguity_flags(record)
    assert "adjacent_larges" in flags
    assert "overlapping_boxes" in flags


def test_mode_color_is_flagged_when_it_is_the_referent(records: list[dict]) -> None:
    record = _broken(records)
    answer = record["answer"]
    other = "blue" if answer != "blue" else "green"
    for obj in record["objects"]:
        obj["attributes"]["color"] = answer
    record["objects"][1]["attributes"]["color"] = other
    flags = ambiguity_flags(record)
    assert "mode_color_is_answer" in flags
    assert "singleton_color_is_answer" not in flags


def test_singleton_color_is_flagged_when_it_is_the_referent(records: list[dict]) -> None:
    record = _broken(records)
    answer = record["answer"]
    others = [color for color in COLORS if color != answer]
    palette = [others[0]] * 3 + [others[1]] * 2 + [others[2]] * 2
    index = 0
    for obj in record["objects"]:
        if obj["object_id"] == "o0":
            obj["attributes"]["color"] = answer
        else:
            obj["attributes"]["color"] = palette[index]
            index += 1
    flags = ambiguity_flags(record)
    assert "singleton_color_is_answer" in flags
    assert "mode_color_is_answer" not in flags


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
