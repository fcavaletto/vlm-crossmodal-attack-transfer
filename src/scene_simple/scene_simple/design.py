"""Explicit 24-scene development design.

Scenes are not sampled. Each counterfactual pair is written out. The seed
does not choose a cell, a color, or a shape. Layouts are not reflections or
recolors of one another. ``_validate`` rejects a group whose shape-and-position
signature matches another group or a mirror of one.

The graded family is ``row_color`` only. Counting is not a family. The
question names a shape and a row. The same shape sits in the other row in
the foil color, and a different shape sits in the named row in a third color.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .geometry import COLS, MIN_SAME_ROW_COL_GAP, ROWS, bounding_box, boxes_too_close
from .oracle import answer_for
from .vocabulary import COLORS, ROW_INDEX, SHAPES

ROW_WORD = {index: name for name, index in ROW_INDEX.items()}


@dataclass(frozen=True)
class Obj:
    object_id: str
    shape: str
    color: str
    size: str
    col: int
    row: int


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    group_id: str
    task_family: str
    question: str
    intended_answer: str
    foil: str
    objects: tuple[Obj, ...]


@dataclass(frozen=True)
class _Pair:
    """One counterfactual pair. Arm B swaps ``o0`` with the other-row object ``o3``."""

    shape: str
    row: int
    target_col: int
    decoy_col: int
    row_decoy_col: int
    row_decoy_shape: str
    fourth_col: int
    fourth_shape: str
    color_a: str
    color_b: str
    foil: str
    row_decoy_color: str


# shape, row, target col, decoy col, row-decoy col, row-decoy shape,
# fourth col, fourth shape, arm A color, arm B color, foil, row-decoy color.
# Arm A paints the fourth object with the arm B color. Arm B swaps those two.
_PAIRS: tuple[_Pair, ...] = (
    _Pair("circle", 0, 1, 0, 3, "square", 2, "triangle", "red", "blue", "green", "yellow"),
    _Pair("circle", 0, 1, 1, 3, "triangle", 3, "square", "red", "green", "yellow", "blue"),
    _Pair("circle", 0, 1, 2, 3, "square", 0, "triangle", "red", "yellow", "blue", "green"),
    _Pair("circle", 1, 3, 1, 1, "triangle", 3, "square", "blue", "red", "green", "yellow"),
    _Pair("square", 0, 2, 0, 0, "triangle", 3, "circle", "blue", "green", "yellow", "red"),
    _Pair("square", 1, 0, 0, 2, "circle", 2, "triangle", "blue", "yellow", "green", "red"),
    _Pair("square", 1, 0, 1, 2, "triangle", 3, "circle", "green", "red", "blue", "yellow"),
    _Pair("square", 1, 2, 2, 0, "circle", 0, "triangle", "green", "blue", "yellow", "red"),
    _Pair("triangle", 0, 0, 3, 2, "circle", 0, "square", "green", "yellow", "red", "blue"),
    _Pair("triangle", 0, 3, 3, 1, "circle", 1, "square", "yellow", "red", "blue", "green"),
    _Pair("triangle", 1, 2, 2, 0, "square", 0, "circle", "yellow", "blue", "red", "green"),
    _Pair("triangle", 1, 3, 3, 1, "square", 0, "circle", "yellow", "green", "red", "blue"),
)


def _question(shape: str, row: int) -> str:
    return f"What color is the {shape} in the {ROW_WORD[row]} row?"


def _objects(pair: _Pair, color: str, fourth_color: str) -> tuple[Obj, ...]:
    """Arm B swaps the target with the other-row object ``o3``.

    A one-object recolor would introduce a color the other arm does not show,
    and that new color would be the only singleton on that arm. The swap keeps
    all four colors on both arms, each once.
    """
    other = 1 - pair.row
    return (
        Obj("o0", pair.shape, color, "large", pair.target_col, pair.row),
        Obj("o1", pair.shape, pair.foil, "large", pair.decoy_col, other),
        Obj("o2", pair.row_decoy_shape, pair.row_decoy_color, "large", pair.row_decoy_col, pair.row),
        Obj("o3", pair.fourth_shape, fourth_color, "large", pair.fourth_col, other),
    )


def _ids(index: int, arm: str) -> tuple[str, str]:
    return f"dev3-row-{index:02d}-{arm}", f"cf3-row-{index:02d}"


def build_scenes() -> tuple[SceneSpec, ...]:
    """The 24 development scenes, in manifest order."""
    scenes: list[SceneSpec] = []
    for index, pair in enumerate(_PAIRS):
        question = _question(pair.shape, pair.row)
        arms = (
            ("a", pair.color_a, pair.color_b),
            ("b", pair.color_b, pair.color_a),
        )
        for arm, color, fourth_color in arms:
            scene_id, group_id = _ids(index, arm)
            scenes.append(
                SceneSpec(
                    scene_id=scene_id,
                    group_id=group_id,
                    task_family="row_color",
                    question=question,
                    intended_answer=color,
                    foil=pair.foil,
                    objects=_objects(pair, color, fourth_color),
                )
            )
    _validate(scenes)
    return tuple(scenes)


def _signature(objects: tuple[Obj, ...]) -> tuple[tuple[str, str, int, int], ...]:
    return tuple(sorted((obj.shape, obj.size, obj.col, obj.row) for obj in objects))


def _flip(
    signature: tuple[tuple[str, str, int, int], ...],
    *,
    col_flip: bool,
    row_flip: bool,
) -> tuple[tuple[str, str, int, int], ...]:
    flipped = []
    for shape, size, col, row in signature:
        if col_flip:
            col = COLS - 1 - col
        if row_flip:
            row = ROWS - 1 - row
        flipped.append((shape, size, col, row))
    return tuple(sorted(flipped))


def _validate(scenes: list[SceneSpec]) -> None:
    if len(scenes) != 24:
        raise ValueError(f"expected 24 scenes, got {len(scenes)}")
    if len(_PAIRS) != 12:
        raise ValueError(f"expected 12 pairs, got {len(_PAIRS)}")
    grouped: dict[str, list[SceneSpec]] = {}
    for scene in scenes:
        grouped.setdefault(scene.group_id, []).append(scene)
    if len(grouped) != 12:
        raise ValueError("expected 12 counterfactual groups")

    signatures: dict[str, tuple[tuple[str, str, int, int], ...]] = {}
    for group_id, members in grouped.items():
        members.sort(key=lambda scene: scene.scene_id)
        if len(members) != 2:
            raise ValueError(f"{group_id} does not have two arms")
        left, right = members
        if _signature(left.objects) != _signature(right.objects):
            raise ValueError(f"{group_id} arms do not share a layout")
        if left.question != right.question or left.foil != right.foil:
            raise ValueError(f"{group_id} arms do not share the question and the foil")
        if left.intended_answer == right.intended_answer:
            raise ValueError(f"{group_id} arms share an answer")
        changed = [
            before.object_id
            for before, after in zip(left.objects, right.objects, strict=True)
            if before.color != after.color
        ]
        if changed != ["o0", "o3"]:
            raise ValueError(f"{group_id} edits {changed}, not the target and the fourth object")
        by_left = {obj.object_id: obj for obj in left.objects}
        by_right = {obj.object_id: obj for obj in right.objects}
        if by_left["o0"].color != by_right["o3"].color or by_left["o3"].color != by_right["o0"].color:
            raise ValueError(f"{group_id} does not swap the target with the fourth object")
        signatures[group_id] = _signature(left.objects)

    items = list(signatures.items())
    for index, (group_id, signature) in enumerate(items):
        mirrors = {
            _flip(signature, col_flip=True, row_flip=False),
            _flip(signature, col_flip=False, row_flip=True),
            _flip(signature, col_flip=True, row_flip=True),
        }
        for other_id, other in items[index + 1 :]:
            if other == signature:
                raise ValueError(f"{group_id} recolors the layout of {other_id}")
            if other in mirrors:
                raise ValueError(f"{group_id} reflects the layout of {other_id}")

    answers = Counter(scene.intended_answer for scene in scenes)
    if dict(answers) != {color: 6 for color in COLORS}:
        raise ValueError(f"unbalanced answers: {answers}")
    shapes = Counter(pair.shape for pair in _PAIRS)
    rows = Counter(pair.row for pair in _PAIRS)
    target_cols = Counter(pair.target_col for pair in _PAIRS)
    decoy_cols = Counter(pair.decoy_col for pair in _PAIRS)
    row_decoy_cols = Counter(pair.row_decoy_col for pair in _PAIRS)
    row_decoy_shapes = Counter(pair.row_decoy_shape for pair in _PAIRS)
    foils = Counter(pair.foil for pair in _PAIRS)
    if dict(shapes) != {shape: 4 for shape in SHAPES}:
        raise ValueError(f"unbalanced queried shapes: {shapes}")
    if dict(rows) != {0: 6, 1: 6}:
        raise ValueError(f"unbalanced rows: {rows}")
    if any(target_cols[col] != 3 for col in range(COLS)):
        raise ValueError(f"unbalanced target columns: {target_cols}")
    if any(decoy_cols[col] != 3 for col in range(COLS)):
        raise ValueError(f"unbalanced decoy columns: {decoy_cols}")
    if any(row_decoy_cols[col] != 3 for col in range(COLS)):
        raise ValueError(f"unbalanced row-decoy columns: {row_decoy_cols}")
    if dict(row_decoy_shapes) != {shape: 4 for shape in SHAPES}:
        raise ValueError(f"unbalanced row-decoy shapes: {row_decoy_shapes}")
    if dict(foils) != {color: 3 for color in COLORS}:
        raise ValueError(f"unbalanced foils: {foils}")
    same_column = sum(pair.target_col == pair.decoy_col for pair in _PAIRS)
    if same_column != 6:
        raise ValueError(f"expected 6 same-column decoys, got {same_column}")

    for scene in scenes:
        objects = scene.objects
        if len(objects) != 4:
            raise ValueError(f"{scene.scene_id} does not have 4 objects")
        if Counter(obj.row for obj in objects) != {0: 2, 1: 2}:
            raise ValueError(f"{scene.scene_id} rows are not equally occupied")
        cells = [(obj.col, obj.row) for obj in objects]
        if len(cells) != len(set(cells)):
            raise ValueError(f"{scene.scene_id} has a shared cell")
        for obj in objects:
            if not 0 <= obj.col < COLS or not 0 <= obj.row < ROWS:
                raise ValueError(f"{scene.scene_id} places {obj.object_id} outside the grid")
        by_row: dict[int, list[int]] = {0: [], 1: []}
        for obj in objects:
            by_row[obj.row].append(obj.col)
        for cols in by_row.values():
            if abs(cols[0] - cols[1]) < MIN_SAME_ROW_COL_GAP:
                raise ValueError(f"{scene.scene_id} crowds a row")
        boxes = [bounding_box(obj.col, obj.row, obj.size) for obj in objects]
        for index, box_a in enumerate(boxes):
            for box_b in boxes[index + 1 :]:
                if boxes_too_close(box_a, box_b):
                    raise ValueError(f"{scene.scene_id} has overlapping boxes")
        if answer_for(scene.task_family, scene.question, objects) != scene.intended_answer:
            raise ValueError(f"{scene.scene_id} oracle disagrees with the design")
        shape = objects[0].shape
        row = objects[0].row
        mates = [obj for obj in objects if obj.shape == shape and obj.row != row]
        if len(mates) != 1 or mates[0].color != scene.foil or mates[0].color == scene.intended_answer:
            raise ValueError(f"{scene.scene_id} shape decoy is not the foil in the other row")
        partners = [obj for obj in objects if obj.row == row and obj.shape != shape]
        if len(partners) != 1 or partners[0].color == scene.intended_answer:
            raise ValueError(f"{scene.scene_id} row decoy does not force the shape word")
        if partners[0].color in {scene.intended_answer, scene.foil}:
            raise ValueError(f"{scene.scene_id} row-decoy color collides with the answer or the foil")
        colors = [obj.color for obj in objects]
        if sorted(colors) != sorted(COLORS):
            raise ValueError(f"{scene.scene_id} does not show each color once: {colors}")
        present = Counter(obj.shape for obj in objects)
        if present[shape] != 2 or set(present) != set(SHAPES):
            raise ValueError(f"{scene.scene_id} shape counts {present}")
