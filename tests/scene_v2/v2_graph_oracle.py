"""Scene-graph answers written for the tests, not imported from ``scene_v2.oracle``.

The generator has its own reader. This one exists so a shared bug in that reader
cannot grade itself. Centers are recomputed from the cell. A stored center that
disagrees is treated as no answer.
"""

from __future__ import annotations

import math
import re

_ATTRIBUTE = re.compile(r"^What color is the (large|small) (circle|square|triangle)\?$")
_COUNTING = re.compile(r"^How many (red|blue|green|yellow) objects are there\?$")
_SPATIAL = re.compile(
    r"^What color is the object immediately left of the "
    r"(large|small) (circle|square|triangle) in the same row\?$"
)
_COUNT_WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}
_MIN_LEFT_RATIO = 2.0


def _center(col: int, row: int) -> tuple[int, int] | None:
    if not 0 <= col < 4 or not 0 <= row < 3:
        return None
    return col * 128 + 64, row * 128 + 64


def graph_answer(record: dict) -> str | None:
    question = record["question"]
    objects = record["objects"]
    attribute = _ATTRIBUTE.fullmatch(question)
    if attribute:
        size, shape = attribute.group(1), attribute.group(2)
        hits = [
            obj
            for obj in objects
            if obj["attributes"]["size"] == size and obj["category"] == shape
        ]
        if len(hits) != 1:
            return None
        return hits[0]["attributes"]["color"]
    counting = _COUNTING.fullmatch(question)
    if counting:
        color = counting.group(1)
        count = sum(1 for obj in objects if obj["attributes"]["color"] == color)
        return _COUNT_WORDS.get(count)
    spatial = _SPATIAL.fullmatch(question)
    if spatial is None:
        return None
    size, shape = spatial.group(1), spatial.group(2)
    anchors = [
        obj
        for obj in objects
        if obj["attributes"]["size"] == size and obj["category"] == shape
    ]
    if len(anchors) != 1:
        return None
    anchor = anchors[0]
    expected = _center(anchor["position"]["col"], anchor["position"]["row"])
    if expected != (anchor["position"]["cx"], anchor["position"]["cy"]):
        return None
    row = anchor["position"]["row"]
    col = anchor["position"]["col"]
    adjacent = [
        obj
        for obj in objects
        if obj["position"]["row"] == row and obj["position"]["col"] == col - 1
    ]
    further = [
        obj for obj in objects if obj["position"]["row"] == row and obj["position"]["col"] < col - 1
    ]
    mates = [
        obj
        for obj in objects
        if obj["position"]["col"] == col - 1 and obj["position"]["row"] != row
    ]
    if len(adjacent) != 1 or further or mates:
        return None
    neighbor = adjacent[0]
    if _center(neighbor["position"]["col"], neighbor["position"]["row"]) != (
        neighbor["position"]["cx"],
        neighbor["position"]["cy"],
    ):
        return None
    ax, ay = anchor["position"]["cx"], anchor["position"]["cy"]
    ranked: list[tuple[float, dict]] = []
    for obj in objects:
        if obj["object_id"] == anchor["object_id"]:
            continue
        ox, oy = obj["position"]["cx"], obj["position"]["cy"]
        if ox < ax:
            ranked.append((math.hypot(ax - ox, ay - oy), obj))
    ranked.sort(key=lambda item: (item[0], item[1]["object_id"]))
    if not ranked or ranked[0][1]["object_id"] != neighbor["object_id"]:
        return None
    if any(dist < _MIN_LEFT_RATIO * ranked[0][0] for dist, _obj in ranked[1:]):
        return None
    return neighbor["attributes"]["color"]
