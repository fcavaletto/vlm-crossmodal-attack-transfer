"""Scene-graph answers written for the tests, not imported from ``scene_gen.oracle``.

The generator has its own reader. This one exists so a shared bug in that reader
cannot grade itself. Question strings are matched here on purpose.
"""

from __future__ import annotations

import re

_ATTRIBUTE = re.compile(r"^What color is the (large|small) (circle|square|triangle)\?$")
_COUNTING = re.compile(r"^How many (red|blue|green|yellow) objects are there\?$")
_SPATIAL = re.compile(
    r"^What color is the object immediately left of the (large|small) (circle|square|triangle)\?$"
)
_COUNT_WORDS = {1: "one", 2: "two", 3: "three", 4: "four"}


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
    if spatial:
        size, shape = spatial.group(1), spatial.group(2)
        anchors = [
            obj
            for obj in objects
            if obj["attributes"]["size"] == size and obj["category"] == shape
        ]
        if len(anchors) != 1:
            return None
        anchor = anchors[0]
        row = anchor["position"]["row"]
        col = anchor["position"]["col"]
        adjacent = [
            obj
            for obj in objects
            if obj["position"]["row"] == row and obj["position"]["col"] == col - 1
        ]
        further = [
            obj
            for obj in objects
            if obj["position"]["row"] == row and obj["position"]["col"] < col - 1
        ]
        if len(adjacent) != 1 or further:
            return None
        return adjacent[0]["attributes"]["color"]
    return None
