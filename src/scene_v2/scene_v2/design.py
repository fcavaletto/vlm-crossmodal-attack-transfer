"""Explicit 72-scene development design.

Scenes are not sampled. Each counterfactual pair is written out. The seed does
not choose a cell or a color. Layouts are not reflections or recolors of one
another; ``_validate`` rejects a group whose shape-and-position signature
matches another group or a mirror of one.

Seventy-two is 3 families times 12 pairs times 2 arms. Twelve is both
``4 colors x 3 adjacent count transitions`` and ``6 size-shape descriptors x 2``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace

from .geometry import COLS, MIN_LEFT_RATIO, bounding_box, boxes_too_close, center, distance
from .oracle import answer_for, count_word
from .vocabulary import COLORS, SHAPES, SIZES, WORD_TO_COUNT

Raw = tuple[str, str, str, int, int]  # shape, size, color, col, row


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
class _Attr:
    size: str
    shape: str
    cell: tuple[int, int]
    color_a: str
    color_b: str
    foil: str
    distractors: tuple[Raw, ...]


@dataclass(frozen=True)
class _Count:
    query: str
    low: int
    foil: str
    donor: int
    objects: tuple[Raw, ...]


@dataclass(frozen=True)
class _Spatial:
    size: str
    shape: str
    color_a: str
    color_b: str
    foil: str
    anchor: tuple[int, int]
    neighbor: Raw
    partner: Raw
    extras: tuple[Raw, ...]


def _obj(index: int, raw: Raw) -> Obj:
    shape, size, color, col, row = raw
    return Obj(f"o{index}", shape, color, size, col, row)


def _ids(family: str, index: int, arm: str) -> tuple[str, str]:
    return f"dev2-{family}-{index:02d}-{arm}", f"cf2-{family}-{index:02d}"


def _attribute_rows() -> tuple[_Attr, ...]:
    # Distractors include both arm colors, the foil color, another object of
    # the target shape, and another object of the target size.
    return (
        _Attr("large", "circle", (0, 0), "red", "blue", "green", (
            ("circle", "small", "blue", 3, 1),
            ("square", "large", "red", 2, 2),
            ("triangle", "small", "green", 1, 2),
        )),
        _Attr("large", "circle", (3, 2), "green", "yellow", "red", (
            ("circle", "small", "yellow", 0, 0),
            ("square", "large", "green", 1, 1),
            ("triangle", "small", "red", 2, 0),
            ("triangle", "large", "blue", 0, 2),
        )),
        _Attr("large", "square", (1, 2), "yellow", "red", "blue", (
            ("square", "small", "red", 3, 0),
            ("circle", "large", "yellow", 0, 0),
            ("circle", "small", "blue", 2, 1),
            ("triangle", "large", "green", 3, 2),
            ("triangle", "small", "yellow", 1, 0),
        )),
        _Attr("large", "square", (2, 0), "blue", "green", "yellow", (
            ("square", "small", "green", 0, 2),
            ("circle", "large", "blue", 3, 2),
            ("triangle", "small", "yellow", 1, 1),
        )),
        _Attr("large", "triangle", (0, 1), "red", "green", "yellow", (
            ("triangle", "small", "green", 3, 1),
            ("circle", "large", "red", 2, 1),
            ("circle", "small", "yellow", 1, 0),
            ("square", "large", "blue", 3, 0),
            ("square", "small", "red", 2, 2),
            ("square", "small", "yellow", 0, 2),
        )),
        _Attr("large", "triangle", (3, 0), "blue", "yellow", "red", (
            ("triangle", "small", "yellow", 0, 1),
            ("square", "large", "blue", 1, 2),
            ("circle", "small", "red", 2, 2),
            ("circle", "large", "green", 1, 0),
        )),
        _Attr("small", "circle", (2, 2), "green", "red", "blue", (
            ("circle", "large", "red", 0, 2),
            ("square", "small", "green", 3, 1),
            ("triangle", "large", "blue", 1, 0),
        )),
        _Attr("small", "circle", (1, 0), "yellow", "blue", "green", (
            ("circle", "large", "blue", 2, 2),
            ("square", "small", "yellow", 3, 2),
            ("triangle", "large", "green", 0, 1),
            ("triangle", "small", "red", 2, 0),
            ("square", "large", "red", 0, 2),
        )),
        _Attr("small", "square", (3, 1), "red", "yellow", "blue", (
            ("square", "large", "yellow", 0, 0),
            ("circle", "small", "red", 1, 2),
            ("triangle", "large", "blue", 2, 0),
            ("triangle", "small", "green", 0, 1),
        )),
        _Attr("small", "square", (0, 2), "green", "blue", "red", (
            ("square", "large", "blue", 3, 2),
            ("circle", "small", "green", 2, 0),
            ("circle", "large", "red", 1, 1),
            ("triangle", "small", "yellow", 3, 0),
            ("triangle", "large", "yellow", 2, 1),
            ("triangle", "small", "blue", 1, 0),
        )),
        _Attr("small", "triangle", (2, 1), "yellow", "green", "blue", (
            ("triangle", "large", "green", 0, 0),
            ("circle", "small", "yellow", 3, 2),
            ("square", "large", "blue", 1, 2),
        )),
        _Attr("small", "triangle", (1, 1), "blue", "red", "yellow", (
            ("triangle", "large", "red", 3, 1),
            ("circle", "small", "blue", 0, 0),
            ("square", "large", "yellow", 2, 2),
            ("square", "small", "green", 3, 0),
            ("circle", "large", "green", 0, 2),
        )),
    )


def _counting_rows() -> tuple[_Count, ...]:
    # ``low`` is the queried count on arm A. Arm B recolors ``donor`` onto the
    # queried color, so the count rises by one and the donor color falls by one.
    # The sorted color-count multiset is unchanged. Foil is wrong for both arms.
    return (
        _Count("red", 1, "four", 2, (
            ("circle", "large", "red", 0, 0),
            ("square", "small", "blue", 1, 1),
            ("triangle", "large", "blue", 3, 2),
            ("square", "large", "green", 2, 0),
        )),
        _Count("blue", 1, "three", 2, (
            ("circle", "small", "blue", 1, 2),
            ("square", "large", "yellow", 0, 1),
            ("triangle", "small", "yellow", 3, 0),
            ("circle", "large", "red", 2, 2),
            ("square", "small", "red", 0, 0),
        )),
        _Count("green", 1, "four", 2, (
            ("triangle", "large", "green", 2, 1),
            ("circle", "small", "red", 0, 2),
            ("square", "large", "red", 3, 1),
            ("circle", "large", "yellow", 1, 0),
            ("square", "small", "yellow", 3, 2),
            ("triangle", "small", "blue", 0, 0),
        )),
        _Count("yellow", 1, "three", 2, (
            ("square", "large", "yellow", 3, 0),
            ("circle", "small", "green", 1, 1),
            ("triangle", "large", "green", 2, 2),
            ("circle", "large", "blue", 0, 2),
            ("square", "small", "red", 3, 1),
        )),
        _Count("red", 2, "four", 4, (
            ("circle", "large", "red", 0, 0),
            ("square", "small", "red", 1, 2),
            ("triangle", "large", "blue", 2, 0),
            ("circle", "small", "blue", 3, 0),
            ("square", "large", "blue", 0, 1),
            ("triangle", "small", "green", 1, 1),
            ("circle", "large", "yellow", 3, 2),
        )),
        _Count("blue", 2, "one", 4, (
            ("circle", "large", "blue", 0, 2),
            ("triangle", "small", "blue", 2, 0),
            ("square", "large", "yellow", 1, 2),
            ("circle", "small", "yellow", 3, 2),
            ("triangle", "large", "yellow", 1, 0),
            ("square", "small", "red", 2, 2),
            ("circle", "large", "red", 3, 0),
        )),
        _Count("green", 2, "four", 4, (
            ("square", "small", "green", 0, 0),
            ("circle", "large", "green", 2, 2),
            ("triangle", "large", "red", 1, 1),
            ("square", "large", "red", 3, 2),
            ("circle", "small", "red", 1, 2),
            ("triangle", "small", "yellow", 3, 0),
        )),
        _Count("yellow", 2, "one", 4, (
            ("circle", "small", "yellow", 1, 1),
            ("square", "large", "yellow", 2, 1),
            ("triangle", "small", "blue", 0, 0),
            ("circle", "large", "blue", 3, 1),
            ("square", "small", "blue", 0, 2),
            ("triangle", "large", "green", 2, 0),
            ("circle", "small", "green", 3, 2),
            ("square", "large", "red", 1, 2),
        )),
        _Count("red", 3, "two", 6, (
            ("circle", "large", "red", 0, 0),
            ("square", "small", "red", 1, 2),
            ("triangle", "large", "red", 3, 1),
            ("circle", "small", "blue", 2, 0),
            ("square", "large", "blue", 0, 1),
            ("triangle", "small", "blue", 2, 2),
            ("circle", "large", "blue", 3, 2),
        )),
        _Count("blue", 3, "one", 6, (
            ("square", "large", "blue", 0, 0),
            ("circle", "small", "blue", 1, 0),
            ("triangle", "small", "blue", 2, 1),
            ("circle", "large", "green", 3, 0),
            ("square", "small", "green", 0, 2),
            ("triangle", "large", "green", 1, 1),
            ("square", "large", "green", 3, 2),
            ("circle", "small", "yellow", 2, 2),
        )),
        _Count("green", 3, "two", 6, (
            ("triangle", "small", "green", 0, 1),
            ("circle", "large", "green", 1, 0),
            ("square", "small", "green", 2, 1),
            ("square", "large", "yellow", 3, 1),
            ("circle", "small", "yellow", 0, 0),
            ("triangle", "large", "yellow", 1, 2),
            ("circle", "large", "yellow", 3, 0),
        )),
        _Count("yellow", 3, "one", 6, (
            ("square", "small", "yellow", 1, 0),
            ("circle", "large", "yellow", 2, 2),
            ("triangle", "small", "yellow", 0, 1),
            ("circle", "small", "red", 3, 1),
            ("square", "large", "red", 1, 1),
            ("triangle", "large", "red", 2, 0),
            ("square", "small", "red", 0, 2),
            ("circle", "large", "blue", 3, 2),
        )),
    )


def _spatial_rows() -> tuple[_Spatial, ...]:
    # Neighbor is immediately left of the anchor. Partner is the unique
    # leftmost object, on another row, and swaps color with the neighbor.
    # Nothing else occupies the neighbor's column or the anchor's row to its left.
    return (
        _Spatial("large", "circle", "red", "blue", "green", (3, 0),
                 ("square", "small", "red", 2, 0),
                 ("triangle", "small", "blue", 0, 2), (
                     ("square", "large", "red", 1, 2),
                     ("circle", "small", "blue", 3, 2),
                     ("triangle", "large", "green", 1, 1),
                 )),
        _Spatial("large", "circle", "yellow", "green", "red", (2, 2),
                 ("circle", "small", "yellow", 1, 2),
                 ("square", "large", "green", 0, 0), (
                     ("square", "small", "yellow", 3, 0),
                     ("triangle", "small", "green", 3, 1),
                     ("triangle", "large", "red", 2, 0),
                     ("square", "small", "red", 3, 2),
                     ("circle", "small", "red", 2, 1),
                 )),
        _Spatial("large", "square", "blue", "yellow", "green", (3, 2),
                 ("triangle", "small", "blue", 2, 2),
                 ("circle", "small", "yellow", 0, 1), (
                     ("circle", "large", "blue", 1, 1),
                     ("triangle", "large", "yellow", 3, 0),
                     ("square", "small", "green", 1, 0),
                 )),
        _Spatial("large", "square", "red", "green", "yellow", (2, 0),
                 ("circle", "small", "red", 1, 0),
                 ("triangle", "large", "green", 0, 2), (
                     ("square", "small", "red", 3, 1),
                     ("triangle", "small", "green", 2, 2),
                     ("circle", "large", "yellow", 3, 0),
                     ("square", "small", "yellow", 3, 2),
                 )),
        _Spatial("large", "triangle", "green", "red", "blue", (3, 1),
                 ("square", "small", "green", 2, 1),
                 ("circle", "small", "red", 0, 0), (
                     ("circle", "large", "green", 1, 2),
                     ("square", "large", "red", 3, 2),
                     ("triangle", "small", "blue", 1, 0),
                 )),
        _Spatial("large", "triangle", "yellow", "blue", "red", (2, 1),
                 ("circle", "small", "yellow", 1, 1),
                 ("square", "large", "blue", 0, 2), (
                     ("square", "small", "yellow", 3, 0),
                     ("triangle", "small", "blue", 3, 2),
                     ("circle", "large", "red", 2, 0),
                     ("square", "small", "red", 3, 1),
                 )),
        _Spatial("small", "circle", "blue", "red", "yellow", (3, 1),
                 ("square", "large", "blue", 2, 1),
                 ("triangle", "large", "red", 0, 2), (
                     ("square", "small", "blue", 1, 2),
                     ("circle", "large", "red", 3, 0),
                     ("triangle", "small", "yellow", 1, 0),
                     ("triangle", "small", "yellow", 3, 2),
                 )),
        _Spatial("small", "circle", "green", "yellow", "blue", (2, 0),
                 ("triangle", "large", "green", 1, 0),
                 ("square", "small", "yellow", 0, 1), (
                     ("square", "large", "green", 3, 2),
                     ("circle", "large", "yellow", 2, 2),
                     ("triangle", "small", "blue", 3, 0),
                 )),
        _Spatial("small", "square", "yellow", "red", "green", (3, 0),
                 ("circle", "large", "yellow", 2, 0),
                 ("triangle", "small", "red", 0, 1), (
                     ("triangle", "large", "yellow", 1, 2),
                     ("circle", "small", "red", 3, 2),
                     ("square", "large", "green", 1, 1),
                     ("circle", "small", "green", 3, 1),
                 )),
        _Spatial("small", "square", "blue", "green", "red", (2, 2),
                 ("triangle", "large", "blue", 1, 2),
                 ("circle", "small", "green", 0, 0), (
                     ("circle", "large", "blue", 3, 1),
                     ("square", "large", "green", 2, 0),
                     ("triangle", "small", "red", 3, 0),
                 )),
        _Spatial("small", "triangle", "red", "yellow", "blue", (3, 2),
                 ("square", "large", "red", 2, 2),
                 ("circle", "small", "yellow", 0, 0), (
                     ("circle", "large", "red", 1, 1),
                     ("triangle", "large", "yellow", 3, 0),
                     ("square", "small", "blue", 1, 0),
                     ("square", "small", "yellow", 3, 1),
                 )),
        _Spatial("small", "triangle", "green", "blue", "yellow", (2, 1),
                 ("circle", "large", "green", 1, 1),
                 ("square", "small", "blue", 0, 2), (
                     ("square", "large", "green", 3, 0),
                     ("triangle", "large", "blue", 3, 2),
                     ("circle", "small", "yellow", 2, 0),
                 )),
    )


def _attribute_scenes() -> list[SceneSpec]:
    scenes: list[SceneSpec] = []
    for index, spec in enumerate(_attribute_rows()):
        question = f"What color is the {spec.size} {spec.shape}?"
        col, row = spec.cell
        target_shape = (spec.shape, spec.size)
        for arm, color in (("a", spec.color_a), ("b", spec.color_b)):
            raws = [(target_shape[0], target_shape[1], color, col, row), *spec.distractors]
            scene_id, group_id = _ids("attribute", index, arm)
            scenes.append(
                SceneSpec(
                    scene_id=scene_id,
                    group_id=group_id,
                    task_family="attribute",
                    question=question,
                    intended_answer=color,
                    foil=spec.foil,
                    objects=tuple(_obj(offset, raw) for offset, raw in enumerate(raws)),
                )
            )
    return scenes


def _counting_scenes() -> list[SceneSpec]:
    scenes: list[SceneSpec] = []
    for index, spec in enumerate(_counting_rows()):
        question = f"How many {spec.query} objects are there?"
        arm_a = tuple(_obj(offset, raw) for offset, raw in enumerate(spec.objects))
        donor = arm_a[spec.donor]
        arm_b_list = list(arm_a)
        arm_b_list[spec.donor] = replace(donor, color=spec.query)
        arm_b = tuple(arm_b_list)
        for arm, objects, count in (
            ("a", arm_a, spec.low),
            ("b", arm_b, spec.low + 1),
        ):
            scene_id, group_id = _ids("counting", index, arm)
            scenes.append(
                SceneSpec(
                    scene_id=scene_id,
                    group_id=group_id,
                    task_family="counting",
                    question=question,
                    intended_answer=count_word(count),
                    foil=spec.foil,
                    objects=objects,
                )
            )
    return scenes


def _swap_colors(objects: tuple[Obj, ...]) -> tuple[Obj, ...]:
    by_id = {obj.object_id: obj for obj in objects}
    color_neighbor = by_id["o0"].color
    color_partner = by_id["o1"].color
    swapped = []
    for obj in objects:
        if obj.object_id == "o0":
            swapped.append(replace(obj, color=color_partner))
        elif obj.object_id == "o1":
            swapped.append(replace(obj, color=color_neighbor))
        else:
            swapped.append(obj)
    return tuple(swapped)


def _spatial_scenes() -> list[SceneSpec]:
    scenes: list[SceneSpec] = []
    for index, spec in enumerate(_spatial_rows()):
        question = (
            f"What color is the object immediately left of the "
            f"{spec.size} {spec.shape} in the same row?"
        )
        col, row = spec.anchor
        raws = [
            spec.neighbor,
            spec.partner,
            (spec.shape, spec.size, spec.foil, col, row),
            *spec.extras,
        ]
        objects_a = tuple(_obj(offset, raw) for offset, raw in enumerate(raws))
        objects_b = _swap_colors(objects_a)
        for arm, objects, color in (
            ("a", objects_a, spec.color_a),
            ("b", objects_b, spec.color_b),
        ):
            scene_id, group_id = _ids("spatial", index, arm)
            scenes.append(
                SceneSpec(
                    scene_id=scene_id,
                    group_id=group_id,
                    task_family="spatial",
                    question=question,
                    intended_answer=color,
                    foil=spec.foil,
                    objects=objects,
                )
            )
    return scenes


def build_scenes() -> list[SceneSpec]:
    """Return the 72 development scenes in family, pair, arm order."""
    scenes = [*_attribute_scenes(), *_counting_scenes(), *_spatial_scenes()]
    _validate(scenes)
    return scenes


def _signature(objects: tuple[Obj, ...]) -> tuple[tuple[str, str, int, int], ...]:
    return tuple(sorted((obj.shape, obj.size, obj.col, obj.row) for obj in objects))


def _reflect(signature: tuple[tuple[str, str, int, int], ...], col_flip: bool, row_flip: bool):
    reflected = []
    for shape, size, col, row in signature:
        if col_flip:
            col = COLS - 1 - col
        if row_flip:
            row = 2 - row
        reflected.append((shape, size, col, row))
    return tuple(sorted(reflected))


def _color_counts(objects: tuple[Obj, ...]) -> Counter[str]:
    return Counter(obj.color for obj in objects)


def _validate(scenes: list[SceneSpec]) -> None:
    if len(scenes) != 72:
        raise AssertionError(f"expected 72 development scenes, got {len(scenes)}")
    ids = [scene.scene_id for scene in scenes]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate scene_id")
    groups: dict[str, list[SceneSpec]] = {}
    families: dict[str, list[SceneSpec]] = defaultdict(list)
    for scene in scenes:
        groups.setdefault(scene.group_id, []).append(scene)
        families[scene.task_family].append(scene)
        if answer_for(scene.task_family, scene.question, scene.objects) != scene.intended_answer:
            raise AssertionError(f"{scene.scene_id} oracle disagrees with the intended answer")
        if scene.foil == scene.intended_answer:
            raise AssertionError(f"{scene.scene_id} foil equals the answer")
        _validate_objects(scene)
    if set(families) != {"attribute", "counting", "spatial"}:
        raise AssertionError(f"unexpected families {set(families)}")
    if any(len(members) != 24 for members in families.values()):
        raise AssertionError("families are not balanced at 24 scenes")
    if len(groups) != 36:
        raise AssertionError(f"expected 36 counterfactual groups, got {len(groups)}")
    for group_id, members in groups.items():
        if len(members) != 2:
            raise AssertionError(f"{group_id} does not have two arms")
        if members[0].question != members[1].question:
            raise AssertionError(f"{group_id} questions differ")
        if members[0].intended_answer == members[1].intended_answer:
            raise AssertionError(f"{group_id} answers do not differ")
        if members[0].foil != members[1].foil:
            raise AssertionError(f"{group_id} foils differ")
        if members[0].task_family != members[1].task_family:
            raise AssertionError(f"{group_id} task families differ")
        if _signature(members[0].objects) != _signature(members[1].objects):
            raise AssertionError(f"{group_id} arms do not share a layout")
    _validate_edits(groups)
    _validate_attribute(families["attribute"])
    _validate_counting(families["counting"])
    _validate_spatial(families["spatial"])
    _validate_layouts(groups)
    _validate_object_counts(scenes)


def _validate_objects(scene: SceneSpec) -> None:
    cells = [(obj.col, obj.row) for obj in scene.objects]
    if len(cells) != len(set(cells)):
        raise AssertionError(f"{scene.scene_id} has two objects in one cell")
    ids = [obj.object_id for obj in scene.objects]
    if len(ids) != len(set(ids)):
        raise AssertionError(f"{scene.scene_id} repeats an object id")
    boxes = []
    for obj in scene.objects:
        if obj.shape not in SHAPES or obj.color not in COLORS or obj.size not in SIZES:
            raise AssertionError(f"{scene.scene_id} has an unknown attribute")
        box = bounding_box(obj.col, obj.row, obj.size)
        x0, y0, x1, y1 = box
        if x0 < 0 or y0 < 0 or x1 >= 512 or y1 >= 384:
            raise AssertionError(f"{scene.scene_id} draws {obj.object_id} outside the canvas")
        boxes.append(box)
    for index, box_a in enumerate(boxes):
        for box_b in boxes[index + 1 :]:
            if boxes_too_close(box_a, box_b):
                raise AssertionError(f"{scene.scene_id} has overlapping boxes")


def _changed_colors(left: tuple[Obj, ...], right: tuple[Obj, ...]) -> list[str]:
    by_right = {obj.object_id: obj for obj in right}
    changed = []
    for obj in left:
        other = by_right[obj.object_id]
        if other.col != obj.col or other.row != obj.row or other.shape != obj.shape or other.size != obj.size:
            raise AssertionError("a counterfactual edit moved or reshaped an object")
        if other.color != obj.color:
            changed.append(obj.object_id)
    return changed


def _validate_edits(groups: dict[str, list[SceneSpec]]) -> None:
    for group_id, members in groups.items():
        left, right = members
        changed = _changed_colors(left.objects, right.objects)
        family = left.task_family
        if family == "attribute" and changed != ["o0"]:
            raise AssertionError(f"{group_id} should recolor only the target")
        if family == "counting" and len(changed) != 1:
            raise AssertionError(f"{group_id} should recolor one object")
        if family == "spatial":
            if changed != ["o0", "o1"]:
                raise AssertionError(f"{group_id} should swap the neighbor and partner colors")
            by_left = {obj.object_id: obj for obj in left.objects}
            by_right = {obj.object_id: obj for obj in right.objects}
            if by_right["o0"].color != by_left["o1"].color or by_right["o1"].color != by_left["o0"].color:
                raise AssertionError(f"{group_id} color swap is not an exchange")


def _validate_attribute(scenes: list[SceneSpec]) -> None:
    answers = Counter(scene.intended_answer for scene in scenes)
    if answers != {color: 6 for color in COLORS}:
        raise AssertionError(f"attribute answers are not balanced: {answers}")
    descriptors = Counter()
    for scene in scenes:
        size, shape = scene.question.removeprefix("What color is the ").removesuffix("?").split()
        descriptors[(size, shape)] += 1
        target = next(obj for obj in scene.objects if obj.object_id == "o0")
        if (target.size, target.shape) != (size, shape):
            raise AssertionError(f"{scene.scene_id} target does not match the question")
        same = [obj for obj in scene.objects if obj.size == size and obj.shape == shape]
        if len(same) != 1:
            raise AssertionError(f"{scene.scene_id} referent is not unique")
        if not any(obj.shape == shape and obj.size != size for obj in scene.objects):
            raise AssertionError(f"{scene.scene_id} shape word is redundant")
        if not any(obj.size == size and obj.shape != shape for obj in scene.objects):
            raise AssertionError(f"{scene.scene_id} size word is redundant")
        if _color_counts(scene.objects)[scene.intended_answer] < 2:
            raise AssertionError(f"{scene.scene_id} answer color is unique")
        if scene.foil not in _color_counts(scene.objects):
            raise AssertionError(f"{scene.scene_id} foil color is not drawn")
    if set(descriptors) != {(size, shape) for size in SIZES for shape in SHAPES}:
        raise AssertionError(f"attribute descriptors are missing: {descriptors}")
    if any(count != 4 for count in descriptors.values()):
        raise AssertionError(f"attribute descriptors are unbalanced: {descriptors}")


def _validate_counting(scenes: list[SceneSpec]) -> None:
    answers = Counter(scene.intended_answer for scene in scenes)
    if answers != {"one": 4, "two": 8, "three": 8, "four": 4}:
        raise AssertionError(f"counting answer counts drifted: {answers}")
    queried = Counter()
    transitions = Counter()
    by_group: dict[str, list[SceneSpec]] = defaultdict(list)
    for scene in scenes:
        queried[scene.question.split()[2]] += 1
        by_group[scene.group_id].append(scene)
    if queried != {color: 6 for color in COLORS}:
        raise AssertionError(f"queried colors are not balanced: {queried}")
    for group_id, members in by_group.items():
        low = WORD_TO_COUNT[members[0].intended_answer]
        high = WORD_TO_COUNT[members[1].intended_answer]
        if high - low != 1:
            raise AssertionError(f"{group_id} count edit is not +1")
        transitions[f"{low}-{high}"] += 1
        if WORD_TO_COUNT[members[0].foil] in {low, high}:
            raise AssertionError(f"{group_id} foil is a true count")
        hist_a = tuple(sorted(_color_counts(members[0].objects).values()))
        hist_b = tuple(sorted(_color_counts(members[1].objects).values()))
        if hist_a != hist_b:
            raise AssertionError(f"{group_id} color histogram changes shape")
        query = members[0].question.split()[2]
        got_a = _color_counts(members[0].objects)[query]
        got_b = _color_counts(members[1].objects)[query]
        if (got_a, got_b) != (low, high):
            raise AssertionError(f"{group_id} queried counts are {(got_a, got_b)}")
    if transitions != {"1-2": 4, "2-3": 4, "3-4": 4}:
        raise AssertionError(f"count transitions drifted: {transitions}")
    queries_by_transition: dict[str, list[str]] = defaultdict(list)
    for members in by_group.values():
        low = WORD_TO_COUNT[members[0].intended_answer]
        high = WORD_TO_COUNT[members[1].intended_answer]
        queries_by_transition[f"{low}-{high}"].append(members[0].question.split()[2])
    for name, colors in queries_by_transition.items():
        if sorted(colors) != sorted(COLORS):
            raise AssertionError(f"{name} does not cover every color: {colors}")


def _validate_spatial(scenes: list[SceneSpec]) -> None:
    answers = Counter(scene.intended_answer for scene in scenes)
    if answers != {color: 6 for color in COLORS}:
        raise AssertionError(f"spatial answers are not balanced: {answers}")
    descriptors = Counter()
    anchor_cols = set()
    anchor_rows = set()
    for scene in scenes:
        body = scene.question.removeprefix("What color is the object immediately left of the ")
        body = body.removesuffix(" in the same row?")
        size, shape = body.split()
        descriptors[(size, shape)] += 1
        anchors = [obj for obj in scene.objects if obj.size == size and obj.shape == shape]
        if len(anchors) != 1:
            raise AssertionError(f"{scene.scene_id} anchor is not unique")
        anchor = anchors[0]
        anchor_cols.add(anchor.col)
        anchor_rows.add(anchor.row)
        if anchor.col not in {2, 3}:
            raise AssertionError(f"{scene.scene_id} anchor has no room for a far decoy")
        neighbor = next(obj for obj in scene.objects if obj.object_id == "o0")
        partner = next(obj for obj in scene.objects if obj.object_id == "o1")
        if (neighbor.col, neighbor.row) != (anchor.col - 1, anchor.row):
            raise AssertionError(f"{scene.scene_id} neighbor is not immediately left")
        if neighbor.color != scene.intended_answer:
            raise AssertionError(f"{scene.scene_id} neighbor color is not the answer")
        if partner.col != 0 or partner.row == anchor.row:
            raise AssertionError(f"{scene.scene_id} partner is not a far leftmost decoy")
        leftmost = [obj for obj in scene.objects if obj.col == 0]
        if leftmost != [partner]:
            raise AssertionError(f"{scene.scene_id} leftmost object is not the partner")
        if partner.color == scene.intended_answer:
            raise AssertionError(f"{scene.scene_id} leftmost color is the answer")
        mates = [obj for obj in scene.objects if obj.col == neighbor.col and obj.row != neighbor.row]
        if mates:
            raise AssertionError(f"{scene.scene_id} has a column-mate")
        counts = _color_counts(scene.objects)
        if any(count < 2 for count in counts.values()):
            raise AssertionError(f"{scene.scene_id} has a singleton color")
        if scene.foil not in counts:
            raise AssertionError(f"{scene.scene_id} foil color is not drawn")
        ax, ay = center(anchor.col, anchor.row)
        nx, ny = center(neighbor.col, neighbor.row)
        neighbor_distance = distance(ax, ay, nx, ny)
        others = []
        for obj in scene.objects:
            if obj.object_id == anchor.object_id:
                continue
            ox, oy = center(obj.col, obj.row)
            if ox < ax:
                others.append(distance(ax, ay, ox, oy))
        others.sort()
        if abs(others[0] - neighbor_distance) > 1e-6:
            raise AssertionError(f"{scene.scene_id} nearest left is not the neighbor")
        if others[1] < MIN_LEFT_RATIO * neighbor_distance:
            raise AssertionError(f"{scene.scene_id} left competitor is too close")
    if anchor_cols != {2, 3} or anchor_rows != {0, 1, 2}:
        raise AssertionError(f"anchor support is cols {anchor_cols} rows {anchor_rows}")
    if any(count != 4 for count in descriptors.values()):
        raise AssertionError(f"spatial descriptors are unbalanced: {descriptors}")


def _validate_layouts(groups: dict[str, list[SceneSpec]]) -> None:
    signatures: dict[str, tuple[tuple[str, str, int, int], ...]] = {}
    for group_id, members in groups.items():
        signatures[group_id] = _signature(members[0].objects)
    items = list(signatures.items())
    for index, (group_id, signature) in enumerate(items):
        mirrors = {
            _reflect(signature, True, False),
            _reflect(signature, False, True),
            _reflect(signature, True, True),
        }
        for other_id, other in items[index + 1 :]:
            if other == signature:
                raise AssertionError(f"{group_id} recolors the layout of {other_id}")
            if other in mirrors:
                raise AssertionError(f"{group_id} is a reflection of {other_id}")


def _validate_object_counts(scenes: list[SceneSpec]) -> None:
    by_count: dict[int, set[str]] = defaultdict(set)
    by_family: dict[str, set[int]] = defaultdict(set)
    for scene in scenes:
        count = len(scene.objects)
        by_count[count].add(scene.task_family)
        by_family[scene.task_family].add(count)
    if any(len(families) < 2 for families in by_count.values()):
        raise AssertionError(f"an object count identifies a family: {dict(by_count)}")
    if any(len(counts) < 2 for counts in by_family.values()):
        raise AssertionError(f"a family still has one object count: {dict(by_family)}")
