"""A 4 by 3 grid on the same 512 by 384 canvas as Checkpoint 1.

Cells are 128 by 128. The extra row is where a far distractor can sit without
occupying the column immediately left of a spatial anchor. Large shapes still
have 36 pixels of gap between neighboring cells, above the 8-pixel minimum.
"""

from __future__ import annotations

import math

from .vocabulary import RADIUS

COLS = 4
ROWS = 3
CELL = 128
CANVAS_W = COLS * CELL  # 512
CANVAS_H = ROWS * CELL  # 384
MIN_BOX_GAP = 8
MIN_LEFT_RATIO = 2.0


def center(col: int, row: int) -> tuple[int, int]:
    """Pixel center of a grid cell. Both coordinates are integers."""
    if not 0 <= col < COLS or not 0 <= row < ROWS:
        raise ValueError(f"cell {(col, row)} is outside the {COLS}x{ROWS} grid")
    return col * CELL + CELL // 2, row * CELL + CELL // 2


def bounding_box(col: int, row: int, size: str) -> tuple[int, int, int, int]:
    """Inclusive square of half-side ``RADIUS[size]`` centered on the cell.

    Circles and squares fill it. Triangles sit inside it. One box for every
    shape makes the overlap test conservative for triangles.
    """
    cx, cy = center(col, row)
    radius = RADIUS[size]
    return cx - radius, cy - radius, cx + radius, cy + radius


def boxes_too_close(
    box_a: tuple[int, int, int, int],
    box_b: tuple[int, int, int, int],
    *,
    min_gap: int = MIN_BOX_GAP,
) -> bool:
    """True when the boxes overlap or the gap is under ``min_gap`` pixels."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    if ax1 < bx0:
        x_gap = bx0 - ax1 - 1
    elif bx1 < ax0:
        x_gap = ax0 - bx1 - 1
    else:
        x_gap = -1
    if ay1 < by0:
        y_gap = by0 - ay1 - 1
    elif by1 < ay0:
        y_gap = ay0 - by1 - 1
    else:
        y_gap = -1
    if x_gap < 0 and y_gap < 0:
        return True
    if x_gap < 0:
        return y_gap < min_gap
    if y_gap < 0:
        return x_gap < min_gap
    return (x_gap * x_gap + y_gap * y_gap) < min_gap * min_gap


def distance(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)
