"""Seeded, independent counterfactual groups for the Track B confirmation split.

This implements the sampling distribution of
``docs/checkpoint-2.2-draft/V5_GENERATOR_SPEC.md`` section 3, under one split
only: ``study2-trackB-confirm-v1``. The H-dL seed strings of that spec
(``study2-checkpoint-2.2-dev-v5`` and ``study2-checkpoint-2.2-sealed-v1``) are
refused, so this package can never draw a sample reserved for H* or H-dL.

Everything visual is imported from ``scene_v4`` unchanged: canvas, grid,
radii, palette, vocabulary, question template, contexts, oracle, renderer, and
the per-scene ambiguity checks. Only the choice of layout and colors changes:
each v4 closed-form index rule becomes a seeded draw.

Group ``k`` uses its own ``numpy.random.Generator(PCG64(seed))``. The seed is the
first 8 bytes, big-endian, of ``SHA-256("{seed_string}|group|{k}|attempt|{a}")``,
starting at attempt ``a = 0``. A rejected attempt is redrawn with ``a + 1``.
"Uniform over a list" always means ``options[rng.integers(len(options))]`` with
the list in the canonical order written in the code.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass

import numpy as np
from scene_v4.design import Obj, SceneSpec, _flip, _require_decoys, _signature
from scene_v4.geometry import COLS, MIN_LARGE_COL_GAP, ROWS, bounding_box, boxes_too_close
from scene_v4.oracle import answer_for
from scene_v4.vocabulary import COLORS, SHAPES

CONFIRM_SEED_STRING = "study2-trackB-confirm-v1"
REFUSED_SEED_STRINGS = frozenset({
    "study2-checkpoint-2.2-dev-v5",
    "study2-checkpoint-2.2-sealed-v1",
})
MAX_ATTEMPTS = 1000
LARGE_SETS: tuple[tuple[int, int], ...] = ((0, 2), (0, 3), (1, 3))
ROW_WORD = {0: "top", 1: "bottom"}

Signature = tuple[tuple[str, str, int, int], ...]


class Rejected(ValueError):
    """An attempt broke a constraint. The caller redraws with the next attempt."""


@dataclass(frozen=True)
class Group:
    index: int
    attempt: int
    seed: int
    arm_a: SceneSpec
    arm_b: SceneSpec
    rejections: dict[str, int]


def refuse_reserved(seed_string: str) -> None:
    if seed_string in REFUSED_SEED_STRINGS:
        raise PermissionError(f"{seed_string!r} is reserved for H-dL (Checkpoint 2.2). This package never draws it.")
    if seed_string != CONFIRM_SEED_STRING:
        raise PermissionError(f"only {CONFIRM_SEED_STRING!r} is accepted, not {seed_string!r}")


def attempt_seed(seed_string: str, k: int, attempt: int) -> int:
    digest = hashlib.sha256(f"{seed_string}|group|{k}|attempt|{attempt}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _pick(rng: np.random.Generator, options):
    options = list(options)
    if not options:
        raise Rejected("a draw had no valid option")
    return options[int(rng.integers(len(options)))]


def propose(rng: np.random.Generator) -> tuple[str, str, str, str, tuple[Obj, ...]]:
    """One proposal (V5 spec section 3, draws 1-14). Returns the question,
    the arm A and arm B answers, the foil, and the arm A objects."""
    target_size = _pick(rng, ("large", "small"))                      # 1
    target_shape = _pick(rng, SHAPES)                                 # 2
    target_row = _pick(rng, (0, 1))                                   # 3
    target_col = _pick(rng, range(COLS))                              # 4
    pairs = [(a, b) for a in COLORS for b in COLORS if a != b]
    color_a, color_b = _pick(rng, pairs)                              # 5
    rest = [c for c in COLORS if c not in (color_a, color_b)]
    mode, single = (rest[0], rest[1]) if _pick(rng, (0, 1)) == 0 else (rest[1], rest[0])  # 6
    other_row = 1 - target_row
    foil_col = _pick(rng, [c for c in range(COLS) if c != target_col])  # 7

    def sets_for(col: int, size: str):
        return [s for s in LARGE_SETS if (col in s) == (size == "large")]

    target_larges = _pick(rng, sets_for(target_col, target_size))     # 8
    other_larges = _pick(rng, sets_for(foil_col, target_size))        # 9

    def size_of(col: int, row: int) -> str:
        return "large" if col in (target_larges if row == target_row else other_larges) else "small"

    size_decoy_col = _pick(rng, [c for c in range(COLS) if size_of(c, target_row) != target_size])  # 10
    row_decoy_col = _pick(rng, [c for c in range(COLS)
                                if c != target_col and size_of(c, target_row) == target_size])     # 11
    row_decoy_shape = _pick(rng, [s for s in SHAPES if s != target_shape])

    shape_at = {
        (target_col, target_row): target_shape,
        (foil_col, other_row): target_shape,
        (size_decoy_col, target_row): target_shape,
        (row_decoy_col, target_row): row_decoy_shape,
    }
    if len(shape_at) != 4:
        raise Rejected("two roles share a cell")

    def allowed(col: int, row: int, shape: str) -> bool:
        size = size_of(col, row)
        if size == target_size and shape == target_shape:
            return False
        if row == target_row and size != target_size and shape == target_shape:
            return False
        return True

    for row in range(ROWS):                                           # 12
        for col in range(COLS):
            if (col, row) not in shape_at:
                shape_at[(col, row)] = _pick(rng, [s for s in SHAPES if allowed(col, row, s)])

    roles = {(target_col, target_row): "o0", (foil_col, other_row): "o1",
             (size_decoy_col, target_row): "o2", (row_decoy_col, target_row): "o3"}
    remaining = [(c, r) for r in range(ROWS) for c in range(COLS) if (c, r) not in roles]
    swap = _pick(rng, [cell for cell in remaining if shape_at[cell] != target_shape])  # 13
    roles[swap] = "o4"
    extras = [cell for cell in remaining if cell != swap]
    order = rng.permutation(3)                                        # 14
    fill = [(color_a, color_b, mode)[int(i)] for i in order]

    color_at = {(target_col, target_row): color_a, (foil_col, other_row): single,
                (size_decoy_col, target_row): mode, (row_decoy_col, target_row): mode, swap: color_b}
    for cell, color in zip(extras, fill, strict=True):
        color_at[cell] = color
    objects = [Obj(roles[cell], shape_at[cell], color_at[cell], size_of(*cell), cell[0], cell[1])
               for cell in sorted(roles, key=lambda c: (c[1], c[0]))]
    for number, cell in enumerate(extras):
        objects.append(Obj(f"o{5 + number}", shape_at[cell], color_at[cell], size_of(*cell), cell[0], cell[1]))
    objects.sort(key=lambda o: int(o.object_id[1:]))
    question = f"What color is the {target_size} {target_shape} in the {ROW_WORD[target_row]} row?"
    return question, color_a, color_b, single, tuple(objects)


def _arms(k: int, question: str, color_a: str, color_b: str, foil: str, arm_a: tuple[Obj, ...]) -> tuple[SceneSpec, SceneSpec]:
    out = []
    for arm, answer, partner in (("a", color_a, color_b), ("b", color_b, color_a)):
        objects = tuple(Obj(o.object_id, o.shape,
                            answer if o.object_id == "o0" else partner if o.object_id == "o4" else o.color,
                            o.size, o.col, o.row) for o in arm_a)
        out.append(SceneSpec(scene_id=f"tB-confirm-{k:03d}-{arm}", group_id=f"cfB-confirm-{k:03d}",
                             task_family="referent_color", question=question, intended_answer=answer,
                             foil=foil, objects=objects))
    return out[0], out[1]


def check_group(left: SceneSpec, right: SceneSpec) -> None:
    """The per-group and per-scene checks of ``scene_v4.design._validate``.
    The cross-group balance checks are not applied (V5 spec section 2)."""
    if _signature(left.objects) != _signature(right.objects):
        raise Rejected("arms do not share a layout")
    if left.question != right.question or left.foil != right.foil:
        raise Rejected("arms do not share the question and the foil")
    if left.intended_answer == right.intended_answer:
        raise Rejected("arms share an answer")
    changed = [b.object_id for b, a in zip(left.objects, right.objects, strict=True) if b.color != a.color]
    if changed != ["o0", "o4"]:
        raise Rejected(f"arms edit {changed}")
    bl = {o.object_id: o for o in left.objects}
    br = {o.object_id: o for o in right.objects}
    if bl["o0"].color != br["o4"].color or bl["o4"].color != br["o0"].color:
        raise Rejected("not a referent-partner swap")
    if bl["o4"].shape == bl["o0"].shape and bl["o4"].size == bl["o0"].size:
        raise Rejected("swaps two copies of the referent")
    lc = Counter(o.color for o in left.objects)
    if lc != Counter(o.color for o in right.objects):
        raise Rejected("a color count changes")
    if sorted(lc.values()) != [1, 2, 2, 3]:
        raise Rejected(f"histogram {sorted(lc.values())}")
    if lc[left.intended_answer] != 2 or lc[right.intended_answer] != 2:
        raise Rejected("answer is not a double")
    single = [c for c, n in lc.items() if n == 1]
    if single != [left.foil]:
        raise Rejected("foil is not the singleton")
    for scene in (left, right):
        objects = scene.objects
        if [o.object_id for o in objects] != [f"o{n}" for n in range(8)]:
            raise Rejected("object ids")
        if Counter(o.row for o in objects) != {0: 4, 1: 4}:
            raise Rejected("rows not equally occupied")
        cells = [(o.col, o.row) for o in objects]
        if len(set(cells)) != 8:
            raise Rejected("grid not filled")
        for row in (0, 1):
            large = [o.col for o in objects if o.row == row and o.size == "large"]
            if len(large) != 2 or abs(large[0] - large[1]) < MIN_LARGE_COL_GAP:
                raise Rejected(f"large columns {large}")
        boxes = [bounding_box(o.col, o.row, o.size) for o in objects]
        for i, a in enumerate(boxes):
            for b in boxes[i + 1:]:
                if boxes_too_close(a, b):
                    raise Rejected("overlapping boxes")
        if answer_for(scene.task_family, scene.question, objects) != scene.intended_answer:
            raise Rejected("oracle disagrees")
        if objects[0].color != scene.intended_answer:
            raise Rejected("referent is not the answer")
        if next(o for o in objects if o.object_id == "o1").col == objects[0].col:
            raise Rejected("referent stacked on its other-row copy")
        if {o.size for o in objects} != {"large", "small"}:
            raise Rejected("only one size drawn")
        try:
            _require_decoys(scene, objects, objects[0])
        except ValueError as exc:
            raise Rejected(str(exc)) from exc


def mirrors(sig: Signature) -> set[Signature]:
    return {sig, _flip(sig, col_flip=True, row_flip=False), _flip(sig, col_flip=False, row_flip=True),
            _flip(sig, col_flip=True, row_flip=True)}


def draw_group(seed_string: str, k: int, excluded: set[Signature]) -> Group:
    """Group ``k``. ``excluded`` holds every forbidden layout signature, mirrors included."""
    refuse_reserved(seed_string)
    rejections: Counter[str] = Counter()
    for attempt in range(MAX_ATTEMPTS):
        seed = attempt_seed(seed_string, k, attempt)
        rng = np.random.Generator(np.random.PCG64(seed))
        try:
            question, a, b, foil, objs = propose(rng)
            left, right = _arms(k, question, a, b, foil, objs)
            check_group(left, right)
        except Rejected:
            rejections["constraint"] += 1
            continue
        if _signature(left.objects) in excluded:
            rejections["novelty"] += 1
            continue
        return Group(k, attempt, seed, left, right, dict(rejections))
    raise RuntimeError(f"group {k}: no accepted draw in {MAX_ATTEMPTS} attempts")


def draw_split(seed_string: str, n_groups: int, development_signatures: set[Signature]) -> list[Group]:
    """Groups 0..n-1 in order. Novelty: no layout (or mirror) of development_v4 or of an earlier group."""
    refuse_reserved(seed_string)
    excluded: set[Signature] = set()
    for sig in development_signatures:
        excluded |= mirrors(sig)
    groups = []
    for k in range(n_groups):
        g = draw_group(seed_string, k, excluded)
        excluded |= mirrors(_signature(g.arm_a.objects))
        groups.append(g)
    return groups
