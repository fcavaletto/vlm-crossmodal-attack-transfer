"""One labeled-color backward. This module does not load the model by itself."""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from choice_eval.colors import COLORS
from choice_eval.decision import choose_color
from choice_eval.likelihood import summed_nll_from_causal_logits
from dev_pilot.scoring import nlls_from_first_logit
from qwen_rgb.finite_difference import (
    DIRECTION_SEED,
    PRIMARY_DELTAS,
    analytic_directional,
    ceiling_for_dtype,
    central_difference,
    judge_samples,
    rademacher,
    relative_error,
)
from qwen_rgb.loss import extend_with_answer
from qwen_rgb.measure import encode_chat, move_to_device
from qwen_rgb.rgb_path import differentiable_model_pixels
from vlm_iface.devices import synchronize
from vlm_iface.images import image_01_to_pil, to_image_01

from .loss_check import same_row_identity
from .protocol import APPENDED_FORWARD_RELATIVE_CEILING
from .records import clean_margin, gradient_record


class MemoryStop(RuntimeError):
    """The runtime ran out of memory. The guard is not lowered and the load is not retried."""


def differentiate_scene(
    *,
    model,
    processor,
    image_processor,
    image_path: Path,
    user_text: str,
    token_ids: dict[str, tuple[int, ...]],
    label: str,
    device,
) -> tuple[dict[str, Any], np.ndarray | None]:
    """Backward of the summed NLL of ``label`` with respect to canonical RGB.

    The returned array is for the caller's alignment and finite-difference
    checks. The caller deletes it. It is not written to disk here.
    """
    started = time.perf_counter()
    image = to_image_01(image_path)
    if float(image.min()) < 0.0 or float(image.max()) > 1.0:
        return _missing("canonical RGB is outside [0, 1]", seconds=time.perf_counter() - started), None
    leaf = image.detach().to(dtype=torch.float32, device="cpu").requires_grad_(True)
    try:
        pixels, grid = differentiable_model_pixels(leaf, image_processor)
        encoded = encode_chat(processor, image_01_to_pil(leaf.detach()), user_text)
        if not torch.equal(grid, encoded["image_grid_thw"].cpu()):
            return _missing("the chat-template grid does not match the differentiable grid", seconds=time.perf_counter() - started), None
        prompt_length = int(encoded["input_ids"].shape[-1])
        encoded.pop("pixel_values", None)
        moved = move_to_device(encoded, device)
        moved["pixel_values"] = pixels.to(device)
        if not torch.equal(moved["image_grid_thw"].cpu(), grid):
            return _missing("moving the grid changed its values", seconds=time.perf_counter() - started), None
        synchronize(device)
        outputs = model(**moved, use_cache=False)
        row = outputs.logits[0, -1]
        if not torch.isfinite(row.detach().float()).all():
            return _missing("continuation logits are not finite, so backward was not run", seconds=time.perf_counter() - started), None
        token_id = int(token_ids[label][0])
        identity = same_row_identity(outputs.logits, moved["input_ids"], prompt_length, token_id)
        loss = identity.pop("loss")
        if identity["agreed"] is not True:
            del loss, outputs, moved, pixels
            return (
                _missing(
                    "the summed NLL formulas disagreed on the labeled token, so backward was not run",
                    seconds=time.perf_counter() - started,
                    extra={"loss_identity": _public_identity(identity), "loss_identity_agreed": False},
                ),
                None,
            )
        summed, mean = nlls_from_first_logit(row.detach(), token_ids)
        loss.backward()
        synchronize(device)
        array = None if leaf.grad is None else leaf.grad.detach().to(dtype=torch.float32).cpu().numpy().copy()
        del outputs, loss, moved, pixels
        record = _from_sums(
            array,
            summed=summed,
            mean=mean,
            label=label,
            token_id=token_id,
            identity=_public_identity(identity),
            seconds=time.perf_counter() - started,
            prompt_length=prompt_length,
        )
        return record, array
    except RuntimeError as exc:
        if _is_memory_error(exc):
            raise MemoryStop(str(exc)) from exc
        raise
    finally:
        leaf.grad = None
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()


def finite_difference_check(
    *,
    model,
    image_processor,
    encoded_cpu: dict[str, Any],
    image: torch.Tensor,
    gradient: np.ndarray,
    token_id: int,
    device,
    dtype_name: str,
) -> dict[str, Any]:
    """Directional check at the documented steps and BF16 or float32 ceiling."""
    direction = rademacher(tuple(image.shape), seed=DIRECTION_SEED)
    base = {key: value for key, value in encoded_cpu.items() if key != "pixel_values"}

    def loss_at(candidate: torch.Tensor) -> float:
        with torch.inference_mode():
            pixels, grid = differentiable_model_pixels(candidate, image_processor)
            inputs = move_to_device(base, device)
            if not torch.equal(inputs["image_grid_thw"].cpu(), grid):
                raise RuntimeError("A finite-difference step changed the patch grid.")
            inputs["pixel_values"] = pixels.to(device)
            outputs = model(**inputs, use_cache=False)
            row = outputs.logits[0, -1]
            value = float(summed_nll_value(row, token_id))
            del outputs, inputs, pixels
            return value

    samples: list[tuple[float, float]] = []
    outside: dict[str, float] = {}
    started = time.perf_counter()
    try:
        for delta in PRIMARY_DELTAS:
            estimate = central_difference(loss_at, image, direction, delta)
            samples.append((delta, estimate))
            stepped = image + delta * direction
            outside[str(delta)] = float(((stepped < 0) | (stepped > 1)).float().mean())
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
        synchronize(device)
    except RuntimeError as exc:
        if _is_memory_error(exc):
            raise MemoryStop(str(exc)) from exc
        return {
            "status": "not_run",
            "reason": str(exc),
            "numerical_check": "not_run",
            "gradient_invalid": None,
        }
    analytic = analytic_directional(torch.as_tensor(gradient, dtype=torch.float32), direction)
    judged = judge_samples(analytic, samples, ceiling=ceiling_for_dtype(dtype_name))
    judged["status"] = "completed"
    judged["direction"] = "rademacher_pm_one"
    judged["direction_seed"] = DIRECTION_SEED
    judged["deltas"] = list(PRIMARY_DELTAS)
    judged["fraction_outside_unit_interval"] = outside
    judged["seconds"] = round(time.perf_counter() - started, 3)
    judged["dtype"] = dtype_name
    judged["numerical_check"] = "confirmed" if judged["confirmed"] else "failed"
    judged["gradient_invalid"] = False
    judged["note"] = (
        "A failed numerical check is not by itself an invalid gradient. "
        "The gradient is invalid when it is missing, non-finite, or zero."
    )
    return judged


def appended_forward_gap(
    *,
    model,
    image_processor,
    encoded_cpu: dict[str, Any],
    image: torch.Tensor,
    token_id: int,
    prompt_loss: float,
    device,
) -> dict[str, Any]:
    """Second forward with the bare color id appended. This is a check, not the backward."""
    try:
        with torch.inference_mode():
            pixels, grid = differentiable_model_pixels(image, image_processor)
            extended, prompt_length = extend_with_answer(encoded_cpu, torch.tensor([token_id], dtype=torch.long))
            inputs = move_to_device(extended, device)
            if not torch.equal(inputs["image_grid_thw"].cpu(), grid):
                return {"status": "not_run", "reason": "the appended forward changed the patch grid"}
            inputs["pixel_values"] = pixels.to(device)
            outputs = model(**inputs, use_cache=False)
            causal = summed_nll_from_causal_logits(outputs.logits, inputs["input_ids"], prompt_length)
            value = float(causal.detach().float().cpu())
            del outputs, inputs, pixels
    except RuntimeError as exc:
        if _is_memory_error(exc):
            raise MemoryStop(str(exc)) from exc
        return {"status": "not_run", "reason": str(exc)}
    finally:
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
    error = relative_error(float(prompt_loss), value)
    agreed = error <= APPENDED_FORWARD_RELATIVE_CEILING
    return {
        "status": "completed",
        "appended_loss": value,
        "prompt_only_loss": float(prompt_loss),
        "relative_error": error,
        "ceiling": APPENDED_FORWARD_RELATIVE_CEILING,
        "numerical_check": "confirmed" if agreed else "failed",
        "gradient_invalid": False,
        "reason": None if agreed else "the appended forward disagrees beyond the documented BF16 ceiling",
    }


def summed_nll_value(row: torch.Tensor, token_id: int) -> float:
    """Detached summed NLL of one token. Used by the finite-difference forward."""
    from choice_eval.likelihood import summed_nll

    return float(summed_nll(row.detach().unsqueeze(0), [int(token_id)]).detach().cpu())


def prepare_encoded(
    *,
    processor,
    image_processor,
    image_path: Path,
    user_text: str,
) -> dict[str, Any] | None:
    """CPU prompt tensors for a later finite-difference check. No model forward."""
    image = to_image_01(image_path)
    pixels, grid = differentiable_model_pixels(image, image_processor)
    encoded = encode_chat(processor, image_01_to_pil(image), user_text)
    if not torch.equal(grid, encoded["image_grid_thw"].cpu()):
        return None
    encoded.pop("pixel_values", None)
    del pixels
    return {
        "image": image.detach().to(dtype=torch.float32, device="cpu"),
        "encoded": {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in encoded.items()},
    }


def _from_sums(
    array: np.ndarray | None,
    *,
    summed: dict[str, float],
    mean: dict[str, float],
    label: str,
    token_id: int,
    identity: dict[str, Any],
    seconds: float,
    prompt_length: int,
) -> dict[str, Any]:
    chosen = choose_color(summed)
    gradient = gradient_record(array, missing_reason="backward did not produce an RGB gradient")
    return {
        "loss": identity["loss_value"],
        "loss_finite": True,
        "loss_identity": identity,
        "loss_identity_agreed": True,
        "summed_nll": {color: float(summed[color]) for color in COLORS},
        "mean_nll": {color: float(mean[color]) for color in COLORS},
        "chosen": chosen,
        "margin": clean_margin(summed, label),
        "reductions_agree": choose_color(mean) == chosen,
        "token_id": token_id,
        "prompt_length": prompt_length,
        "scoring_path": "first_continuation_logit",
        "reduction": "summed_nll",
        "gradient": gradient,
        "seconds": round(seconds, 3),
    }


def _missing(reason: str, *, seconds: float, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    record = {
        "loss": None,
        "loss_finite": False,
        "loss_identity_agreed": extra.get("loss_identity_agreed", True) if extra else True,
        "summed_nll": None,
        "mean_nll": None,
        "chosen": None,
        "margin": None,
        "gradient": gradient_record(None, missing_reason=reason),
        "seconds": round(seconds, 3),
        "stopped_reason": reason,
    }
    if extra:
        record.update(extra)
    return record


def _public_identity(identity: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in identity.items() if key != "loss"}


def _is_memory_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or ("mps backend" in text and "alloc" in text)
