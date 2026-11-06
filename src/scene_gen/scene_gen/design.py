"""Explicit 24-scene development design.

Scenes are not sampled. Each counterfactual pair is written out so a reviewer
can audit the intervention. ``development_seed`` is attached later; it does not
choose colors or cells.

Counting scenes use all eight cells and the color-count multiset ``{2, 3, 3}``
on both arms. A one-object recoloring moves the queried color from 2 to 3 and
a distractor color from 3 to 2, leaving the multiset unchanged. With only four
objects, a count of three is always the unique majority, which would answer the
question without using the color word. Eight objects avoid that perfect shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .oracle import answer_for


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


def _obj(index: int, shape: str, color: str, size: str, col: int, row: int) -> Obj:
    return Obj(f"o{index}", shape, color, size, col, row)


def _pair_ids(family: str, index: int, arm: str) -> tuple[str, str]:
    return f"dev-{family}-{index:02d}-{arm}", f"cf-{family}-{index:02d}"


def _attribute_pair(
    index: int,
    *,
    size: str,
    shape: str,
    color_a: str,
    color_b: str,
    foil: str,
    target_cell: tuple[int, int],
    distractors: tuple[tuple[str, str, str, int, int], ...],
) -> list[SceneSpec]:
    question = f"What color is the {size} {shape}?"
    col, row = target_cell

    def objects_for(target_color: str) -> tuple[Obj, ...]:
        target = _obj(0, shape, target_color, size, col, row)
        rest = tuple(
            _obj(offset, d_shape, d_color, d_size, d_col, d_row)
            for offset, (d_shape, d_color, d_size, d_col, d_row) in enumerate(distractors, start=1)
        )
        return (target, *rest)

    scenes = []
    for arm, color in (("a", color_a), ("b", color_b)):
        scene_id, group_id = _pair_ids("attribute", index, arm)
        scenes.append(
            SceneSpec(
                scene_id=scene_id,
                group_id=group_id,
                task_family="attribute",
                question=question,
                intended_answer=color,
                foil=foil,
                objects=objects_for(color),
            )
        )
    return scenes


def _grid_objects(cells: tuple[tuple[str, str, str], ...]) -> tuple[Obj, ...]:
    """``cells`` is row-major over the 4 by 2 grid: (shape, color, size)."""
    if len(cells) != 8:
        raise ValueError("counting scenes occupy every cell")
    objects = []
    for index, (shape, color, size) in enumerate(cells):
        objects.append(_obj(index, shape, color, size, index % 4, index // 4))
    return tuple(objects)


def _counting_pair(
    index: int,
    *,
    query_color: str,
    cells: tuple[tuple[str, str, str], ...],
    donor_index: int,
) -> list[SceneSpec]:
    question = f"How many {query_color} objects are there?"
    arm_a = _grid_objects(cells)
    donor = arm_a[donor_index]
    if donor.color == query_color:
        raise ValueError(f"counting pair {index} donor is already {query_color}")
    arm_b_list = list(arm_a)
    arm_b_list[donor_index] = replace(donor, color=query_color)
    arm_b = tuple(arm_b_list)
    scenes = []
    for arm, objects, answer in (
        ("a", arm_a, "two"),
        ("b", arm_b, "three"),
    ):
        scene_id, group_id = _pair_ids("counting", index, arm)
        scenes.append(
            SceneSpec(
                scene_id=scene_id,
                group_id=group_id,
                task_family="counting",
                question=question,
                intended_answer=answer,
                foil="four",
                objects=objects,
            )
        )
    return scenes


def _swap_positions(objects: tuple[Obj, ...], id_a: str, id_b: str) -> tuple[Obj, ...]:
    by_id = {obj.object_id: obj for obj in objects}
    pos_a = (by_id[id_a].col, by_id[id_a].row)
    pos_b = (by_id[id_b].col, by_id[id_b].row)
    swapped = []
    for obj in objects:
        if obj.object_id == id_a:
            swapped.append(replace(obj, col=pos_b[0], row=pos_b[1]))
        elif obj.object_id == id_b:
            swapped.append(replace(obj, col=pos_a[0], row=pos_a[1]))
        else:
            swapped.append(obj)
    return tuple(swapped)


def _spatial_pair(
    index: int,
    *,
    size: str,
    shape: str,
    color_a: str,
    color_b: str,
    foil: str,
    objects_a: tuple[Obj, ...],
    swap_ids: tuple[str, str],
) -> list[SceneSpec]:
    question = f"What color is the object immediately left of the {size} {shape}?"
    objects_b = _swap_positions(objects_a, *swap_ids)
    scenes = []
    for arm, objects, answer in (
        ("a", objects_a, color_a),
        ("b", objects_b, color_b),
    ):
        scene_id, group_id = _pair_ids("spatial", index, arm)
        scenes.append(
            SceneSpec(
                scene_id=scene_id,
                group_id=group_id,
                task_family="spatial",
                question=question,
                intended_answer=answer,
                foil=foil,
                objects=objects,
            )
        )
    return scenes


def _attribute_scenes() -> list[SceneSpec]:
    # Distractors include the arm-A color and the arm-B color, so the target
    # color is never the only object of that color.
    return [
        *_attribute_pair(
            0,
            size="large",
            shape="circle",
            color_a="red",
            color_b="blue",
            foil="green",
            target_cell=(0, 0),
            distractors=(
                ("square", "red", "small", 3, 0),
                ("triangle", "yellow", "small", 1, 1),
                ("square", "blue", "large", 2, 1),
            ),
        ),
        *_attribute_pair(
            1,
            size="large",
            shape="square",
            color_a="green",
            color_b="yellow",
            foil="red",
            target_cell=(2, 0),
            distractors=(
                ("circle", "green", "small", 0, 1),
                ("triangle", "yellow", "large", 3, 1),
                ("triangle", "blue", "small", 1, 0),
            ),
        ),
        *_attribute_pair(
            2,
            size="large",
            shape="triangle",
            color_a="red",
            color_b="green",
            foil="yellow",
            target_cell=(1, 1),
            distractors=(
                ("circle", "red", "small", 3, 0),
                ("square", "green", "small", 0, 0),
                ("circle", "blue", "large", 2, 0),
            ),
        ),
        *_attribute_pair(
            3,
            size="small",
            shape="circle",
            color_a="blue",
            color_b="yellow",
            foil="red",
            target_cell=(3, 1),
            distractors=(
                ("circle", "blue", "large", 1, 0),
                ("square", "yellow", "small", 0, 1),
                ("triangle", "green", "large", 2, 0),
            ),
        ),
    ]


def _counting_scenes() -> list[SceneSpec]:
    # Each grid is (shape, color, size) in row-major order.
    # Arm B recolors the donor onto the queried color: 2 -> 3, and that donor's
    # old color 3 -> 2. The multiset of counts stays {2, 3, 3}.
    return [
        *_counting_pair(
            0,
            query_color="red",
            donor_index=4,
            cells=(
                ("circle", "red", "large"),
                ("square", "red", "small"),
                ("triangle", "blue", "large"),
                ("circle", "blue", "small"),
                ("square", "green", "large"),
                ("triangle", "green", "small"),
                ("circle", "green", "large"),
                ("square", "blue", "small"),
            ),
        ),
        *_counting_pair(
            1,
            query_color="blue",
            donor_index=0,
            cells=(
                ("triangle", "yellow", "small"),
                ("circle", "yellow", "large"),
                ("square", "yellow", "small"),
                ("square", "blue", "large"),
                ("circle", "blue", "small"),
                ("triangle", "red", "large"),
                ("square", "red", "small"),
                ("circle", "red", "large"),
            ),
        ),
        *_counting_pair(
            2,
            query_color="green",
            donor_index=0,
            cells=(
                ("square", "red", "large"),
                ("circle", "yellow", "small"),
                ("triangle", "yellow", "large"),
                ("square", "yellow", "small"),
                ("circle", "green", "large"),
                ("triangle", "green", "small"),
                ("square", "red", "large"),
                ("circle", "red", "small"),
            ),
        ),
        *_counting_pair(
            3,
            query_color="yellow",
            donor_index=5,
            cells=(
                ("circle", "green", "small"),
                ("square", "green", "large"),
                ("triangle", "green", "small"),
                ("circle", "yellow", "large"),
                ("square", "yellow", "small"),
                ("triangle", "blue", "large"),
                ("circle", "blue", "small"),
                ("square", "blue", "large"),
            ),
        ),
    ]


def _spatial_scenes() -> list[SceneSpec]:
    # o0 is the left neighbor on arm A and swaps places with o1 (the decoy).
    # The decoy sits further left on the other row, so "leftmost object" is not
    # the answer. Another object shares the neighbor's column, on the other row.
    return [
        *_spatial_pair(
            0,
            size="large",
            shape="circle",
            color_a="red",
            color_b="blue",
            foil="yellow",
            swap_ids=("o0", "o1"),
            objects_a=(
                _obj(0, "square", "red", "small", 2, 0),
                _obj(1, "triangle", "blue", "small", 0, 1),
                _obj(2, "circle", "green", "large", 3, 0),
                _obj(3, "square", "yellow", "large", 2, 1),
                _obj(4, "circle", "green", "small", 3, 1),
            ),
        ),
        *_spatial_pair(
            1,
            size="large",
            shape="square",
            color_a="green",
            color_b="yellow",
            foil="red",
            swap_ids=("o0", "o1"),
            objects_a=(
                _obj(0, "circle", "green", "small", 2, 1),
                _obj(1, "square", "yellow", "small", 0, 0),
                _obj(2, "square", "blue", "large", 3, 1),
                _obj(3, "triangle", "red", "large", 2, 0),
                _obj(4, "triangle", "blue", "small", 3, 0),
            ),
        ),
        *_spatial_pair(
            2,
            size="large",
            shape="triangle",
            color_a="red",
            color_b="green",
            foil="blue",
            swap_ids=("o0", "o1"),
            objects_a=(
                _obj(0, "square", "red", "small", 1, 0),
                _obj(1, "circle", "green", "small", 0, 1),
                _obj(2, "triangle", "yellow", "large", 2, 0),
                _obj(3, "circle", "blue", "large", 1, 1),
                _obj(4, "triangle", "yellow", "small", 3, 0),
            ),
        ),
        *_spatial_pair(
            3,
            size="small",
            shape="circle",
            color_a="blue",
            color_b="yellow",
            foil="green",
            swap_ids=("o0", "o1"),
            objects_a=(
                _obj(0, "square", "blue", "large", 1, 1),
                _obj(1, "triangle", "yellow", "small", 0, 0),
                _obj(2, "circle", "red", "small", 2, 1),
                _obj(3, "circle", "green", "large", 1, 0),
                _obj(4, "triangle", "red", "large", 3, 1),
            ),
        ),
    ]


def build_scenes() -> list[SceneSpec]:
    """Return the 24 development scenes in family, pair, arm order."""
    scenes = [*_attribute_scenes(), *_counting_scenes(), *_spatial_scenes()]
    _validate(scenes)
    return scenes


def _validate(scenes: list[SceneSpec]) -> None:
    if len(scenes) != 24:
        raise AssertionError(f"expected 24 development scenes, got {len(scenes)}")
    ids = [scene.scene_id for scene in scenes]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate scene_id")
    groups: dict[str, list[SceneSpec]] = {}
    for scene in scenes:
        groups.setdefault(scene.group_id, []).append(scene)
        if answer_for(scene.task_family, scene.question, scene.objects) != scene.intended_answer:
            raise AssertionError(f"{scene.scene_id} oracle disagrees with intended answer")
        cells = [(obj.col, obj.row) for obj in scene.objects]
        if len(cells) != len(set(cells)):
            raise AssertionError(f"{scene.scene_id} has two objects in one cell")
        if scene.foil == scene.intended_answer:
            raise AssertionError(f"{scene.scene_id} foil equals the answer")
    if len(groups) != 12:
        raise AssertionError(f"expected 12 counterfactual groups, got {len(groups)}")
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
