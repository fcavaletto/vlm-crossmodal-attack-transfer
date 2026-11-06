"""Explicit 96-scene development design.

Scenes are not sampled. Index ``i`` in ``0..47`` builds one counterfactual
pair by a closed form. The seed does not choose a cell, a color, or a shape.
``_validate`` raises if a derived pair breaks a constraint. It does not draw
a replacement.

Twelve descriptors are the crossing of two sizes, three shapes, and two rows.
Four replicas put each descriptor in each of the four columns. Color pairs
come from the three 4-cycles on red, blue, green, and yellow. Each descriptor
uses one cycle, so each of its questions has every color twice. Arm order
reverses on half of the descriptors that share a cycle, which balances the
twelve ordered pairs.

Both arms show the same color counts: one color three times, two colors
twice, and one color once. The two answers are the colors that appear twice.
The color that appears three times is never the answer. The color that
appears once is the other-row copy of the referent, and it is never the
answer. Arm B swaps the referent with a different shape, so the count of
every color stays put while the answer changes.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from .geometry import COLS, MIN_LARGE_COL_GAP, ROWS, bounding_box, boxes_too_close
from .oracle import QUESTION_RE, answer_for
from .vocabulary import COLORS, ROW_INDEX, SHAPES

ROW_WORD = {index: name for name, index in ROW_INDEX.items()}
GROUP_COUNT = 48
SCENE_COUNT = 96

# Column pairs that leave two large shapes at least two columns apart.
_LARGE_SETS: tuple[tuple[int, int], ...] = ((0, 2), (0, 3), (1, 3))

# Each tuple is a 4-cycle. Replica r takes edge r. The cycle left out of a
# descriptor is ``d % 3``. Together the three cycles use every unordered pair.
_CYCLES: tuple[tuple[tuple[str, str], ...], ...] = (
    (("red", "blue"), ("blue", "green"), ("green", "yellow"), ("yellow", "red")),
    (("red", "green"), ("green", "blue"), ("blue", "yellow"), ("yellow", "red")),
    (("red", "blue"), ("blue", "yellow"), ("yellow", "green"), ("green", "red")),
)


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


def _question(size: str, shape: str, row: int) -> str:
    return f"What color is the {size} {shape} in the {ROW_WORD[row]} row?"


def _layout(index: int) -> tuple[str, str, str, str, tuple[Obj, ...]]:
    """Question, arm A color, arm B color, foil, and the arm A objects."""
    descriptor = index % 12
    replica = index // 12
    target_size = "large" if descriptor % 2 else "small"
    target_shape = SHAPES[(descriptor // 2) % 3]
    target_row = 0 if descriptor < 6 else 1
    target_col = replica
    other_row = 1 - target_row
    other_size = "small" if target_size == "large" else "large"

    color_a, color_b = _CYCLES[descriptor % 3][replica]
    if descriptor // 3 >= 2:
        color_a, color_b = color_b, color_a
    rest = [color for color in COLORS if color not in (color_a, color_b)]
    if (descriptor + replica) % 2 == 0:
        mode, single = rest[0], rest[1]
    else:
        mode, single = rest[1], rest[0]

    offset = 1 + ((descriptor + 2 * target_row) % 3)
    foil_col = (target_col + offset) % 4
    if target_size == "large":
        target_options = [pair for pair in _LARGE_SETS if target_col in pair]
        target_larges = target_options[(descriptor + 3 * target_row) % len(target_options)]
        other_options = [pair for pair in _LARGE_SETS if foil_col in pair]
    else:
        target_options = [pair for pair in _LARGE_SETS if target_col not in pair]
        target_larges = target_options[(descriptor + target_row) % len(target_options)]
        other_options = [pair for pair in _LARGE_SETS if foil_col not in pair]
    other_larges = other_options[(descriptor + 2 * replica + 3 * target_row) % len(other_options)]

    if target_size == "large":
        small_cols = [col for col in range(COLS) if col not in target_larges]
        size_decoy_col = small_cols[(replica + target_row) % len(small_cols)]
        row_decoy_col = next(col for col in target_larges if col != target_col)
    else:
        size_decoy_col = target_larges[(replica + target_row) % 2]
        small_cols = [col for col in range(COLS) if col not in target_larges]
        row_decoy_col = next(col for col in small_cols if col != target_col)
    row_decoy_shape = SHAPES[
        (SHAPES.index(target_shape) + 1 + ((replica + target_row) % 2)) % 3
    ]

    def size_of(col: int, row: int) -> str:
        larges = target_larges if row == target_row else other_larges
        return "large" if col in larges else "small"

    shape_at: dict[tuple[int, int], str] = {
        (target_col, target_row): target_shape,
        (foil_col, other_row): target_shape,
        (size_decoy_col, target_row): target_shape,
        (row_decoy_col, target_row): row_decoy_shape,
    }

    def allowed(col: int, row: int, shape: str) -> bool:
        size = size_of(col, row)
        if size == target_size and shape == target_shape:
            return False
        if row == target_row and size != target_size and shape == target_shape:
            return False
        return True

    for row in range(ROWS):
        for col in range(COLS):
            if (col, row) in shape_at:
                continue
            placed = None
            base = SHAPES.index(target_shape) + 1 + (col + 1) * (row + 2) + descriptor + replica
            for step in range(3):
                shape = SHAPES[(base + step) % 3]
                if allowed(col, row, shape):
                    placed = shape
                    break
            if placed is None:
                raise ValueError(f"group {index} has no shape for cell {(col, row)}")
            shape_at[(col, row)] = placed

    color_at = {
        (target_col, target_row): color_a,
        (foil_col, other_row): single,
        (size_decoy_col, target_row): mode,
        (row_decoy_col, target_row): mode,
    }
    remaining = [
        (col, row)
        for row in range(ROWS)
        for col in range(COLS)
        if (col, row) not in color_at
    ]
    candidates = [cell for cell in remaining if shape_at[cell] != target_shape]
    if not candidates:
        raise ValueError(f"group {index} has no swap partner of a different shape")
    swap = candidates[(descriptor + replica) % len(candidates)]
    others = [cell for cell in remaining if cell != swap]
    color_at[swap] = color_b
    for cell, color in zip(others, (color_a, color_b, mode), strict=True):
        color_at[cell] = color

    roles = {
        (target_col, target_row): "o0",
        (foil_col, other_row): "o1",
        (size_decoy_col, target_row): "o2",
        (row_decoy_col, target_row): "o3",
        swap: "o4",
    }
    if len(roles) != 5:
        raise ValueError(f"group {index} collapses two roles onto one cell")
    extras = [cell for cell in remaining if cell != swap]
    objects: list[Obj] = []
    for row in range(ROWS):
        for col in range(COLS):
            object_id = roles.get((col, row))
            if object_id is None:
                continue
            objects.append(
                Obj(
                    object_id,
                    shape_at[(col, row)],
                    color_at[(col, row)],
                    size_of(col, row),
                    col,
                    row,
                )
            )
    for number, cell in enumerate(extras):
        col, row = cell
        objects.append(
            Obj(
                f"o{5 + number}",
                shape_at[cell],
                color_at[cell],
                size_of(col, row),
                col,
                row,
            )
        )
    objects.sort(key=lambda obj: obj.object_id)
    if other_size not in {obj.size for obj in objects}:
        raise ValueError(f"group {index} never draws the other size")
    question = _question(target_size, target_shape, target_row)
    return question, color_a, color_b, single, tuple(objects)


def _ids(index: int, arm: str) -> tuple[str, str]:
    return f"dev4-ref-{index:02d}-{arm}", f"cf4-ref-{index:02d}"


def build_scenes() -> tuple[SceneSpec, ...]:
    """The 96 development scenes, in manifest order."""
    scenes: list[SceneSpec] = []
    for index in range(GROUP_COUNT):
        question, color_a, color_b, foil, arm_a = _layout(index)
        for arm, answer, partner in (("a", color_a, color_b), ("b", color_b, color_a)):
            scene_id, group_id = _ids(index, arm)
            objects = tuple(
                Obj(
                    obj.object_id,
                    obj.shape,
                    answer if obj.object_id == "o0" else partner if obj.object_id == "o4" else obj.color,
                    obj.size,
                    obj.col,
                    obj.row,
                )
                for obj in arm_a
            )
            scenes.append(
                SceneSpec(
                    scene_id=scene_id,
                    group_id=group_id,
                    task_family="referent_color",
                    question=question,
                    intended_answer=answer,
                    foil=foil,
                    objects=objects,
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
    if len(scenes) != SCENE_COUNT:
        raise ValueError(f"expected {SCENE_COUNT} scenes, got {len(scenes)}")
    grouped: dict[str, list[SceneSpec]] = {}
    for scene in scenes:
        grouped.setdefault(scene.group_id, []).append(scene)
    if len(grouped) != GROUP_COUNT:
        raise ValueError(f"expected {GROUP_COUNT} counterfactual groups")

    signatures: dict[str, tuple[tuple[str, str, int, int], ...]] = {}
    orders: Counter[tuple[str, str]] = Counter()
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
        if changed != ["o0", "o4"]:
            raise ValueError(f"{group_id} edits {changed}, not the referent and the partner")
        by_left = {obj.object_id: obj for obj in left.objects}
        by_right = {obj.object_id: obj for obj in right.objects}
        if by_left["o0"].color != by_right["o4"].color or by_left["o4"].color != by_right["o0"].color:
            raise ValueError(f"{group_id} does not swap the referent with the partner")
        if by_left["o4"].shape == by_left["o0"].shape and by_left["o4"].size == by_left["o0"].size:
            raise ValueError(f"{group_id} swaps two copies of the referent")
        left_counts = Counter(obj.color for obj in left.objects)
        right_counts = Counter(obj.color for obj in right.objects)
        if left_counts != right_counts:
            raise ValueError(f"{group_id} changes a color count")
        if sorted(left_counts.values()) != [1, 2, 2, 3]:
            raise ValueError(f"{group_id} histogram is {sorted(left_counts.values())}")
        if left_counts[left.intended_answer] != 2 or right_counts[right.intended_answer] != 2:
            raise ValueError(f"{group_id} answer color is not one of the two doubles")
        mode = [color for color, count in left_counts.items() if count == 3]
        single = [color for color, count in left_counts.items() if count == 1]
        if mode == [left.intended_answer] or mode == [right.intended_answer]:
            raise ValueError(f"{group_id} modal color is an answer")
        if single == [left.intended_answer] or single == [right.intended_answer]:
            raise ValueError(f"{group_id} singleton color is an answer")
        if single != [left.foil]:
            raise ValueError(f"{group_id} foil is not the singleton")
        orders[(left.intended_answer, right.intended_answer)] += 1
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
    if dict(answers) != {color: 24 for color in COLORS}:
        raise ValueError(f"unbalanced answers: {answers}")
    if any(count != 4 for count in orders.values()) or len(orders) != 12:
        raise ValueError(f"unbalanced arm order: {orders}")

    by_question: dict[str, Counter[str]] = defaultdict(Counter)
    shapes: Counter[str] = Counter()
    sizes: Counter[str] = Counter()
    rows: Counter[str] = Counter()
    foils: Counter[str] = Counter()
    for scene in scenes:
        by_question[scene.question][scene.intended_answer] += 1
        match = QUESTION_RE.fullmatch(scene.question)
        if match is None:
            raise ValueError(f"unparsed question: {scene.question}")
        size, shape, row = match.group(1), match.group(2), match.group(3)
        shapes[shape] += 1
        sizes[size] += 1
        rows[row] += 1
        foils[scene.foil] += 1
    if len(by_question) != 12:
        raise ValueError(f"expected 12 questions, got {len(by_question)}")
    for question, counts in by_question.items():
        if dict(counts) != {color: 2 for color in COLORS}:
            raise ValueError(f"{question} answers {counts}")
    if dict(shapes) != {shape: 32 for shape in SHAPES}:
        raise ValueError(f"unbalanced shapes: {shapes}")
    if dict(sizes) != {"large": 48, "small": 48}:
        raise ValueError(f"unbalanced sizes: {sizes}")
    if dict(rows) != {"top": 48, "bottom": 48}:
        raise ValueError(f"unbalanced rows: {rows}")
    if dict(foils) != {color: 24 for color in COLORS}:
        raise ValueError(f"unbalanced foils: {foils}")

    for scene in scenes:
        objects = scene.objects
        if [obj.object_id for obj in objects] != [f"o{number}" for number in range(8)]:
            raise ValueError(f"{scene.scene_id} object ids are {[obj.object_id for obj in objects]}")
        if Counter(obj.row for obj in objects) != {0: 4, 1: 4}:
            raise ValueError(f"{scene.scene_id} rows are not equally occupied")
        cells = [(obj.col, obj.row) for obj in objects]
        if len(cells) != len(set(cells)) or len(cells) != 8:
            raise ValueError(f"{scene.scene_id} does not fill the grid")
        for obj in objects:
            if not 0 <= obj.col < COLS or not 0 <= obj.row < ROWS:
                raise ValueError(f"{scene.scene_id} places {obj.object_id} outside the grid")
        for row in (0, 1):
            large_cols = [obj.col for obj in objects if obj.row == row and obj.size == "large"]
            if len(large_cols) != 2 or abs(large_cols[0] - large_cols[1]) < MIN_LARGE_COL_GAP:
                raise ValueError(f"{scene.scene_id} large columns {large_cols}")
        boxes = [bounding_box(obj.col, obj.row, obj.size) for obj in objects]
        for index, box_a in enumerate(boxes):
            for box_b in boxes[index + 1 :]:
                if boxes_too_close(box_a, box_b):
                    raise ValueError(f"{scene.scene_id} has overlapping boxes")
        if answer_for(scene.task_family, scene.question, objects) != scene.intended_answer:
            raise ValueError(f"{scene.scene_id} oracle disagrees with the design")
        target = objects[0]
        if target.color != scene.intended_answer:
            raise ValueError(f"{scene.scene_id} referent is not the answer")
        foil_obj = next(obj for obj in objects if obj.object_id == "o1")
        if foil_obj.col == target.col:
            raise ValueError(f"{scene.scene_id} stacks the referent on its other-row copy")
        _require_decoys(scene, objects, target)


def _require_decoys(scene: SceneSpec, objects: tuple[Obj, ...], target: Obj) -> None:
    answers = {
        obj.color
        for obj in objects
        if obj.object_id in {"o0", "o4"}
    }
    # Both arm answers are o0 and o4 on either arm. On this arm o0 is the
    # answer and o4 is the other answer. A decoy colored with either one
    # would match the answer on one of the two arms.
    cross = [
        obj
        for obj in objects
        if obj.size == target.size and obj.shape == target.shape and obj.row != target.row
    ]
    if len(cross) != 1 or cross[0].color != scene.foil or cross[0].color in answers:
        raise ValueError(f"{scene.scene_id} cross-row decoy is not the foil")
    size_decoys = [
        obj
        for obj in objects
        if obj.shape == target.shape and obj.row == target.row and obj.size != target.size
    ]
    if len(size_decoys) != 1 or size_decoys[0].color in answers:
        raise ValueError(f"{scene.scene_id} size decoy does not force the size word")
    row_decoys = [
        obj
        for obj in objects
        if obj.size == target.size and obj.row == target.row and obj.shape != target.shape
    ]
    if not row_decoys or any(obj.color in answers for obj in row_decoys):
        raise ValueError(f"{scene.scene_id} row decoy does not force the shape word")
    counts = Counter(obj.color for obj in objects)
    if max(counts.values()) == counts[scene.intended_answer]:
        raise ValueError(f"{scene.scene_id} answer is a modal color")
    if counts[scene.intended_answer] == min(counts.values()):
        raise ValueError(f"{scene.scene_id} answer is a least-frequent color")
