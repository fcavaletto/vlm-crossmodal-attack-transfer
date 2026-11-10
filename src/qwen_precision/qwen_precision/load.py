"""Qwen3-VL-2B load with the frozen arguments, any dtype, any device."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path
from typing import Any

import torch

from .numerics import DTYPES


def snapshot_dir() -> Path:
    from qwen_rgb.pins import MODEL_ID, PINNED_REVISION

    home = os.environ.get("HF_HOME")
    if not home:
        raise RuntimeError("HF_HOME must be set explicitly (local: the study cache; pod: /workspace/hf-cache)")
    path = Path(home) / "hub" / ("models--" + MODEL_ID.replace("/", "--")) / "snapshots" / PINNED_REVISION
    if not path.is_dir():
        raise RuntimeError(f"pinned snapshot not found at {path}")
    return path


def weights_facts(path: Path | None = None) -> dict[str, Any]:
    """sha256 and the stored dtype of every tensor in model.safetensors."""
    f = (path or snapshot_dir()) / "model.safetensors"
    h = hashlib.sha256()
    with f.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    with f.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    counts: dict[str, int] = {}
    for k, v in header.items():
        if k == "__metadata__":
            continue
        m = 1
        for s in v["shape"]:
            m *= s
        counts[v["dtype"]] = counts.get(v["dtype"], 0) + m
    return {"sha256": h.hexdigest(), "stored_parameter_dtypes": counts}


def load_model(dtype: str, device: torch.device):
    """The frozen ``from_pretrained`` call (qwen_rgb.measure.load_model_after_budget) with dtype as a parameter."""
    from transformers import Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(snapshot_dir()), dtype=DTYPES[dtype], attn_implementation="eager")
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dtypes = sorted({str(p.dtype) for p in model.parameters()})
    if dtypes != [str(DTYPES[dtype])]:
        raise RuntimeError(f"parameters are {dtypes}, expected only {DTYPES[dtype]}")
    if model.config._attn_implementation != "eager":
        raise RuntimeError("attention implementation is not eager")
    return model


def load_processor_and_tokens():
    from dev_pilot.scoring import assert_locked_tokenization
    from qwen_rgb.measure import load_processor
    from qwen_rgb_loss.protocol import MODEL_KEY

    processor, image_processor = load_processor()
    identity = assert_locked_tokenization(processor, MODEL_KEY)
    token_ids = {c: tuple(int(t) for t in ids) for c, ids in identity["token_ids"].items()}
    if token_ids != {"red": (1151,), "blue": (12203,), "green": (13250,), "yellow": (27869,)}:
        raise RuntimeError("the colour token ids are not the locked ids")
    return processor, image_processor, token_ids
