"""One labeled-color forward or backward. The caller loads the model."""

from __future__ import annotations

import gc
from typing import Any

import torch
from choice_eval.colors import COLORS
from choice_eval.decision import choose_color
from dev_pilot.scoring import nlls_from_first_logit
from qwen_rgb.measure import encode_chat, move_to_device
from qwen_rgb.rgb_path import differentiable_model_pixels
from qwen_rgb_loss.loss_check import same_row_identity
from qwen_rgb_loss.measure import MemoryStop
from qwen_rgb_loss.records import clean_margin
from vlm_iface.devices import synchronize
from vlm_iface.images import image_01_to_pil

from .steps import directional_agreement, first_order_change, round_trip_metrics


def is_memory_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or ("mps backend" in text and "alloc" in text)


def release_cache(device) -> None:
    gc.collect()
    if getattr(device, "type", None) == "mps":
        torch.mps.empty_cache()


def prepare_prompt(processor, image_processor, image: torch.Tensor, user_text: str) -> dict[str, Any]:
    """CPU prompt tensors. Pixel values are supplied later by the differentiable path."""
    _pixels, grid = differentiable_model_pixels(image.detach(), image_processor)
    encoded = encode_chat(processor, image_01_to_pil(image.detach()), user_text)
    if not torch.equal(grid.cpu(), encoded["image_grid_thw"].cpu()):
        raise RuntimeError("the chat-template grid does not match the differentiable grid")
    encoded.pop("pixel_values", None)
    del _pixels
    return {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in encoded.items()
    }


def _public_score(identity: dict[str, Any], summed: dict[str, float], mean: dict[str, float], label: str) -> dict[str, Any]:
    chosen = choose_color(summed)
    return {
        "loss": identity["loss_value"] if identity["agreed"] else None,
        "loss_identity_agreed": bool(identity["agreed"]),
        "loss_identity_max_abs": identity["max_abs"],
        "summed_nll": {color: float(summed[color]) for color in COLORS},
        "mean_nll": {color: float(mean[color]) for color in COLORS},
        "chosen": chosen,
        "margin": clean_margin(summed, label),
        "reductions_agree": choose_color(mean) == chosen,
        "token_id": identity["token_id"],
    }


def score_differentiable(
    *,
    model,
    image_processor,
    prompt: dict[str, Any],
    image: torch.Tensor,
    token_ids: dict[str, tuple[int, ...]],
    label: str,
    device,
) -> dict[str, Any]:
    """Labeled-color sums on the unclipped differentiable pixel path. No graph is kept."""
    token_id = int(token_ids[label][0])
    with torch.no_grad():
        pixels, grid = differentiable_model_pixels(image.detach(), image_processor)
        if not torch.equal(grid.cpu(), prompt["image_grid_thw"].cpu()):
            raise RuntimeError("a step changed the patch grid")
        moved = move_to_device(prompt, device)
        moved["pixel_values"] = pixels.to(device)
        outputs = model(**moved, use_cache=False)
        identity = same_row_identity(outputs.logits, moved["input_ids"], int(moved["input_ids"].shape[-1]), token_id)
        summed, mean = nlls_from_first_logit(outputs.logits[0, -1], token_ids)
        public = _public_score(
            {key: value for key, value in identity.items() if key != "loss"},
            summed,
            mean,
            label,
        )
        del outputs, moved, pixels, identity
    return public


def score_official(
    *,
    model,
    processor,
    image: torch.Tensor,
    user_text: str,
    token_ids: dict[str, tuple[int, ...]],
    label: str,
    device,
) -> dict[str, Any]:
    """Labeled-color sums of the official 8-bit processor path."""
    from qwen_rgb.constraints import round_trip_uint8

    token_id = int(token_ids[label][0])
    stored = round_trip_uint8(image)
    with torch.no_grad():
        encoded = encode_chat(processor, image_01_to_pil(stored), user_text)
        moved = move_to_device(encoded, device)
        outputs = model(**moved, use_cache=False)
        identity = same_row_identity(outputs.logits, moved["input_ids"], int(moved["input_ids"].shape[-1]), token_id)
        summed, mean = nlls_from_first_logit(outputs.logits[0, -1], token_ids)
        public = _public_score(
            {key: value for key, value in identity.items() if key != "loss"},
            summed,
            mean,
            label,
        )
        del outputs, moved, encoded, identity
    del stored
    return public


def backward_at_image(
    *,
    model,
    image_processor,
    prompt: dict[str, Any],
    image: torch.Tensor,
    token_ids: dict[str, tuple[int, ...]],
    label: str,
    device,
) -> tuple[dict[str, Any], torch.Tensor | None]:
    """RGB gradient of the labeled-color loss. The array is not written by this function."""
    token_id = int(token_ids[label][0])
    leaf = image.detach().to(dtype=torch.float32, device="cpu").requires_grad_(True)
    try:
        pixels, grid = differentiable_model_pixels(leaf, image_processor)
        if not torch.equal(grid.cpu(), prompt["image_grid_thw"].cpu()):
            raise RuntimeError("a step changed the patch grid")
        moved = move_to_device(prompt, device)
        moved["pixel_values"] = pixels.to(device)
        synchronize(device)
        outputs = model(**moved, use_cache=False)
        identity = same_row_identity(outputs.logits, moved["input_ids"], int(moved["input_ids"].shape[-1]), token_id)
        summed, mean = nlls_from_first_logit(outputs.logits[0, -1].detach(), token_ids)
        public = _public_score(
            {key: value for key, value in identity.items() if key != "loss"},
            summed,
            mean,
            label,
        )
        gradient = None
        if identity["agreed"] is True:
            identity["loss"].backward()
            synchronize(device)
            if leaf.grad is not None:
                gradient = leaf.grad.detach().to(dtype=torch.float32, device="cpu").clone()
        del outputs, moved, pixels, identity
        return public, gradient
    except RuntimeError as exc:
        if is_memory_error(exc):
            raise MemoryStop(str(exc)) from exc
        raise
    finally:
        leaf.grad = None
        release_cache(device)


def geometry(clean: torch.Tensor, gradient: torch.Tensor, projected: torch.Tensor) -> tuple[dict[str, Any], torch.Tensor]:
    """L-infinity distances and the stored image. The caller deletes the tensor."""
    metrics = round_trip_metrics(clean, projected)
    stored = metrics.pop("stored")
    return {
        "predicted_change": first_order_change(gradient, clean, projected),
        "linf": metrics["linf_float"],
        "uint8_geometry": {
            "linf_vs_clean": metrics["linf_stored_vs_clean"],
            "linf_vs_projected": metrics["linf_stored_vs_projected"],
            "fraction_pixels_changed": metrics["fraction_pixels_changed"],
        },
    }, stored


def build_step_record(
    *,
    mode: str,
    geometry_record: dict[str, Any],
    float_score: dict[str, Any],
    official_score: dict[str, Any],
    clean_float_loss: float,
    clean_float_chosen: str,
    clean_official_loss: float | None,
    clean_official_chosen: str,
    label: str,
) -> dict[str, Any]:
    """Compact step record. No image and no gradient."""
    predicted = geometry_record["predicted_change"]
    observed = None if float_score.get("loss") is None else float(float_score["loss"]) - float(clean_float_loss)
    official_observed = None
    if official_score.get("loss") is not None and clean_official_loss is not None:
        official_observed = float(official_score["loss"]) - float(clean_official_loss)
    uint8 = dict(geometry_record["uint8_geometry"])
    uint8.update(
        {
            "loss": official_score.get("loss"),
            "margin": official_score.get("margin"),
            "chosen": official_score.get("chosen"),
            "summed_nll": official_score.get("summed_nll"),
            "loss_identity_agreed": official_score.get("loss_identity_agreed"),
            "loss_identity_max_abs": official_score.get("loss_identity_max_abs"),
            "answer_changed": official_score.get("chosen") != clean_official_chosen,
            "left_label": official_score.get("chosen") != label,
            "observed_change": official_observed,
        }
    )
    return {
        "mode": mode,
        "predicted_change": predicted,
        "linf": geometry_record["linf"],
        "loss": float_score.get("loss"),
        "margin": float_score.get("margin"),
        "chosen": float_score.get("chosen"),
        "summed_nll": float_score.get("summed_nll"),
        "loss_identity_agreed": float_score.get("loss_identity_agreed"),
        "loss_identity_max_abs": float_score.get("loss_identity_max_abs"),
        "answer_changed": float_score.get("chosen") != clean_float_chosen,
        "left_label": float_score.get("chosen") != label,
        "observed_change": observed,
        "directional_agreement": None if observed is None else directional_agreement(predicted, observed),
        "uint8": uint8,
    }
