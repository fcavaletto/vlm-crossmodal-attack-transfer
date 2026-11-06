"""Draw one scene. No jitter, no speckle, no external image.

The contact sheet is the 96 development scenes, 12 pair columns by 8 rows.
Four bands of two rows cover pairs 00–11, 12–23, 24–35, and 36–47. Within a
band the first row is arm A and the second is arm B. The sheet is a review
figure, not a model input. Captions name the answer so a person can check
the sheet. They are not part of a scene.
"""

from __future__ import annotations

import math

from PIL import Image, ImageDraw

from .vocabulary import BACKGROUND, OUTLINE, PALETTE, RADIUS

THUMB_W = 128
THUMB_H = 96
CAPTION_H = 24
SHEET_PAD = 4
SHEET_COLUMNS = 12
SHEET_ROWS = 8
CONTACT_SHEET_SIZE = (
    SHEET_PAD + SHEET_COLUMNS * (THUMB_W + SHEET_PAD),
    SHEET_PAD + SHEET_ROWS * (THUMB_H + CAPTION_H + SHEET_PAD),
)
OUTLINE_WIDTH = 3


def _triangle(cx: int, cy: int, radius: int) -> list[tuple[float, float]]:
    """Equilateral triangle with its circumcenter at ``(cx, cy)``, point up."""
    points = []
    for step in range(3):
        theta = -math.pi / 2 + step * 2 * math.pi / 3
        points.append((cx + radius * math.cos(theta), cy + radius * math.sin(theta)))
    return points


def render_scene(record: dict[str, object], *, canvas_w: int, canvas_h: int) -> Image.Image:
    image = Image.new("RGB", (canvas_w, canvas_h), BACKGROUND)
    draw = ImageDraw.Draw(image)
    objects = sorted(record["objects"], key=lambda obj: obj["object_id"])  # type: ignore[index]
    for obj in objects:
        color = PALETTE[obj["attributes"]["color"]]
        radius = RADIUS[obj["attributes"]["size"]]
        cx = obj["position"]["cx"]
        cy = obj["position"]["cy"]
        shape = obj["category"]
        if shape == "circle":
            draw.ellipse(
                (cx - radius, cy - radius, cx + radius, cy + radius),
                fill=color,
                outline=OUTLINE,
                width=OUTLINE_WIDTH,
            )
        elif shape == "square":
            draw.rectangle(
                (cx - radius, cy - radius, cx + radius, cy + radius),
                fill=color,
                outline=OUTLINE,
                width=OUTLINE_WIDTH,
            )
        elif shape == "triangle":
            draw.polygon(
                _triangle(cx, cy, radius),
                fill=color,
                outline=OUTLINE,
                width=OUTLINE_WIDTH,
            )
        else:
            raise ValueError(f"cannot draw shape {shape}")
    return image


def contact_sheet_ids() -> tuple[str, ...]:
    """Scene ids in sheet order. Each band is 12 pairs, arm A then arm B."""
    ids: list[str] = []
    for band in range(4):
        for arm in ("a", "b"):
            for column in range(SHEET_COLUMNS):
                index = band * SHEET_COLUMNS + column
                ids.append(f"dev4-ref-{index:02d}-{arm}")
    return tuple(ids)


def render_contact_sheet(panels: list[tuple[str, Image.Image]]) -> Image.Image:
    """All 96 scenes. Fixed size, Pillow's built-in font, no measured text width."""
    from PIL import ImageFont

    if len(panels) != SHEET_COLUMNS * SHEET_ROWS:
        raise ValueError(
            f"contact sheet expects {SHEET_COLUMNS * SHEET_ROWS} panels, got {len(panels)}"
        )
    width, height = CONTACT_SHEET_SIZE
    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    font = ImageFont.load_default()
    for index, (caption, image) in enumerate(panels):
        col = index % SHEET_COLUMNS
        row = index // SHEET_COLUMNS
        left = SHEET_PAD + col * (THUMB_W + SHEET_PAD)
        top = SHEET_PAD + row * (THUMB_H + CAPTION_H + SHEET_PAD)
        thumb = image.resize((THUMB_W, THUMB_H), Image.Resampling.NEAREST)
        sheet.paste(thumb, (left, top))
        caption_bar = Image.new("RGB", (THUMB_W, CAPTION_H), (255, 255, 255))
        bar_draw = ImageDraw.Draw(caption_bar)
        bar_draw.text((2, 6), caption, fill=(0, 0, 0), font=font)
        sheet.paste(caption_bar, (left, top + THUMB_H))
    return sheet
