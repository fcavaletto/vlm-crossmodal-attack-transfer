"""One precommitted illustration. The display gain is not a second attack."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageDraw
from vlm_iface.images import image_01_to_pil

from .protocol import DISPLAY_GAIN


def write_illustration(
    path: Path,
    clean: torch.Tensor,
    stored: torch.Tensor,
    header: str,
    metadata: dict | None = None,
) -> None:
    """Clean image, official 8-bit image, and the absolute difference times 32."""
    if tuple(clean.shape) != tuple(stored.shape):
        raise ValueError("clean and stored images have different shapes")
    clean_pil = image_01_to_pil(clean)
    stored_pil = image_01_to_pil(stored)
    amplified = (stored.detach() - clean.detach()).abs().mul(DISPLAY_GAIN).clamp(0, 1)
    difference = image_01_to_pil(amplified)
    width = clean_pil.width
    caption = 56
    canvas = Image.new("RGB", (width * 3, clean_pil.height + caption), "white")
    canvas.paste(clean_pil, (0, caption))
    canvas.paste(stored_pil, (width, caption))
    canvas.paste(difference, (width * 2, caption))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), header, fill=(0, 0, 0))
    draw.text((8, 28), "clean", fill=(0, 0, 0))
    draw.text((width + 8, 28), "official 8-bit", fill=(0, 0, 0))
    draw.text((width * 2 + 8, 28), f"abs difference x{DISPLAY_GAIN}, display only", fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")
    import json

    sidecar = path.with_suffix(".json")
    sidecar.write_text(json.dumps({"header": header, "display_gain": DISPLAY_GAIN, **(metadata or {})}, indent=2) + "\n", encoding="utf-8")
