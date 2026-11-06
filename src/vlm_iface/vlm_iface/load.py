"""Load the pinned SmolVLM2 checkpoint. No other model id is accepted."""

from __future__ import annotations

import torch

from .devices import available_bytes, resolve_device
from .pins import MIN_AVAILABLE_BYTES, MODEL_ID, PINNED_REVISION


def load_pinned_model(
    device: torch.device | str,
    *,
    model_id: str = MODEL_ID,
    revision: str = PINNED_REVISION,
):
    """Load processor and frozen float32 weights with eager attention.

    The Hub sha must equal ``revision``. Weights are moved with ``.to(device)``
    rather than ``device_map``, so the parameter device is the one requested.
    """
    resolved = device if isinstance(device, torch.device) else resolve_device(device)
    if model_id != MODEL_ID or revision != PINNED_REVISION:
        raise RuntimeError(
            "This checkpoint loads only "
            f"{MODEL_ID}@{PINNED_REVISION}. "
            "Qwen/Qwen3-VL-2B-Instruct is not loaded from this interface."
        )
    free = available_bytes()
    if free < MIN_AVAILABLE_BYTES:
        raise RuntimeError(
            f"Refusing to load {MODEL_ID}: {free} bytes available, "
            f"below the {MIN_AVAILABLE_BYTES} byte guard."
        )

    from huggingface_hub import model_info
    from transformers import AutoModelForImageTextToText, AutoProcessor

    hub = model_info(model_id, revision=revision)
    if str(hub.sha) != revision:
        raise RuntimeError(f"Hub sha {hub.sha} does not match pinned revision {revision}")

    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.float32,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.to(resolved)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _assert_parameter_device(model, resolved)
    _assert_commit_if_present(processor, revision)
    _assert_commit_if_present(model, revision)
    model.config._vlm_iface_revision = revision
    processor._vlm_iface_revision = revision  # type: ignore[attr-defined]
    return processor, model


def _assert_parameter_device(model: torch.nn.Module, device: torch.device) -> None:
    devices = {parameter.device.type for parameter in model.parameters()}
    if devices != {device.type}:
        raise RuntimeError(f"Expected every parameter on {device.type}, found {sorted(devices)}")
    dtypes = {parameter.dtype for parameter in model.parameters()}
    if dtypes != {torch.float32}:
        raise RuntimeError(f"Expected float32 parameters, found {sorted(dtypes, key=str)}")


def _assert_commit_if_present(obj, revision: str) -> None:
    for candidate in (
        getattr(obj, "config", None),
        obj,
        getattr(obj, "tokenizer", None),
        getattr(obj, "image_processor", None),
    ):
        if candidate is None:
            continue
        commit = getattr(candidate, "_commit_hash", None)
        if commit and str(commit) != revision:
            raise RuntimeError(f"Loaded object revision {commit} does not match {revision}")
