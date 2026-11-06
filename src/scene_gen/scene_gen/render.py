"""Draw one scene. No jitter, no speckle, no external image.

Study 1 sprinkled faint dots on a one-object canvas. Those dots would be extra
objects for a counting question, so this renderer draws only the scene graph.
"""

from __future__ import annotations

import math

from PIL import Image, ImageDraw

from .vocabulary import BACKGROUND, OUTLINE, PALETTE, RADIUS


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
                width=3,
            )
        elif shape == "square":
            draw.rectangle(
                (cx - radius, cy - radius, cx + radius, cy + radius),
                fill=color,
                outline=OUTLINE,
                width=3,
            )
        elif shape == "triangle":
            draw.polygon(
                _triangle(cx, cy, radius),
                fill=color,
                outline=OUTLINE,
                width=3,
            )
        else:
            raise ValueError(f"cannot draw shape {shape}")
    return image


THUMB_W = 256
THUMB_H = 192
CAPTION_H = 28
SHEET_PAD = 8
SHEET_COLUMNS = 3
SHEET_ROWS = 2
CONTACT_SHEET_SIZE = (
    SHEET_PAD + SHEET_COLUMNS * (THUMB_W + SHEET_PAD),
    SHEET_PAD + SHEET_ROWS * (THUMB_H + CAPTION_H + SHEET_PAD),
)


def render_contact_sheet(
    panels: list[tuple[str, Image.Image]],
) -> Image.Image:
    """Three families by two counterfactual arms. Fixed size, built-in font.

    The font is Pillow's default bitmap font, which ships inside Pillow, so the
    sheet does not depend on a font installed on the machine. Caption bars have
    a fixed height; sheet size does not depend on text measurement.
    """
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
        # Draw at native size, then the bar is tall enough for the default font.
        bar_draw.text((4, 8), caption, fill=(0, 0, 0), font=font)
        sheet.paste(caption_bar, (left, top + THUMB_H))
    return sheet
