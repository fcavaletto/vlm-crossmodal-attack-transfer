"""A 4 by 2 grid on the Checkpoint 1 canvas.

Cells are 128 by 192. The center of column ``col`` and row ``row`` is
``(col * 128 + 64, row * 192 + 96)``. Row 0 is the top row. A large shape has
radius 60, so a top-row shape ends at y=156 and a bottom-row shape starts at
y=228. The band between them is empty. A small shape has radius 28 and sits
inside the same band.

Two large shapes in one row need columns at least two apart. A large shape
and a small shape may occupy neighboring columns: the gap is 39 pixels.
"""

from __future__ import annotations

from .vocabulary import RADIUS

COLS = 4
ROWS = 2
CELL_W = 128
CELL_H = 192
CANVAS_W = COLS * CELL_W  # 512
CANVAS_H = ROWS * CELL_H  # 384
MIN_BOX_GAP = 8
MIN_LARGE_COL_GAP = 2


def center(col: int, row: int) -> tuple[int, int]:
    """Pixel center of a grid cell. Both coordinates are integers."""
    if not 0 <= col < COLS or not 0 <= row < ROWS:
        raise ValueError(f"cell {(col, row)} is outside the {COLS}x{ROWS} grid")
    return col * CELL_W + CELL_W // 2, row * CELL_H + CELL_H // 2


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
