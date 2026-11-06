"""Stage A sign steps. One scene at a time. Gradients are not written."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from choice_eval.prompt import build_prompt
from dev_pilot.rows import user_text_sha256
from qwen_rgb_loss.measure import MemoryStop, differentiate_scene
from qwen_rgb_loss.records import gradient_record
from vlm_iface.images import to_image_01

from .figure import write_illustration
from .protocol import (
    STAGE_A_BUDGETS,
    STAGE_A_CONTEXT,
    STAGE_A_ILLUSTRATION,
    STAGE_A_SEED_PREFIX,
    epsilon,
    figure_path,
)
from .score import (
    build_step_record,
    geometry,
    is_memory_error,
    prepare_prompt,
    release_cache,
    score_differentiable,
    score_official,
)
from .steps import matched_rademacher, project_signed_step, scene_seed


class StageStop(RuntimeError):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status


def run_scene(
    *,
    root: Path,
    model,
    processor,
    image_processor,
    record: dict[str, Any],
    selection: dict[str, Any],
    stored_row: dict[str, Any],
    prior_gradient_sha256: str | None,
    token_ids: dict[str, tuple[int, ...]],
    device,
) -> dict[str, Any]:
    """Neutral sign steps at the four budgets. The gradient array is discarded."""
    started = time.perf_counter()
    scene_id = record["scene_id"]
    label = str(record["answer"])
    user_text = build_prompt(record["contexts"][STAGE_A_CONTEXT]["text"], record["question"]).user_text()
    text_hash = user_text_sha256(user_text)
    if stored_row.get("user_text_sha256") != text_hash:
        raise StageStop("stopped_prompt", f"{scene_id} neutral prompt hash does not match the stored row.")
    image_path = root / "fixtures" / "development_v4" / record["image_path"]
    try:
        measured, array = differentiate_scene(
            model=model,
            processor=processor,
            image_processor=image_processor,
            image_path=image_path,
            user_text=user_text,
            token_ids=token_ids,
            label=label,
            device=device,
        )
    except MemoryStop as exc:
        raise StageStop("stopped_oom", "The runtime reported out of memory. The budget was not lowered and the load was not retried.") from exc
    if measured.get("loss_identity_agreed") is not True:
        raise StageStop("stopped_loss_identity", "The labeled-color NLL formulas disagreed, so no step was taken.")
    gradient_info = dict(measured["gradient"])
    gradient_info["checkpoint_1_8_sha256"] = prior_gradient_sha256
    gradient_info["sha256_matches_checkpoint_1_8"] = gradient_info.get("sha256") == prior_gradient_sha256
    clean = to_image_01(image_path)
    official = _official(model, processor, clean, user_text, token_ids, label, device)
    payload: dict[str, Any] = {
        "scene_id": scene_id,
        "answer": label,
        "stratum": selection["stratum"],
        "alignment_neutral_to_helpful": selection["alignment_neutral_to_helpful"],
        "context": STAGE_A_CONTEXT,
        "user_text_sha256": text_hash,
        "finished": False,
        "clean": {
            "loss": measured["loss"],
            "margin": measured["margin"],
            "chosen": measured["chosen"],
            "summed_nll": measured["summed_nll"],
            "loss_identity_agreed": True,
            "max_abs_nll_gap_vs_stored": _gap(measured.get("summed_nll"), stored_row.get("summed_nll")),
            "decision_matches_stored": measured.get("chosen") == stored_row.get("chosen"),
            "official_loss": official.get("loss"),
            "official_margin": official.get("margin"),
            "official_chosen": official.get("chosen"),
            "official_summed_nll": official.get("summed_nll"),
            "official_loss_identity_agreed": official.get("loss_identity_agreed"),
        },
        "gradient": {key: value for key, value in gradient_info.items() if key != "reason" or True},
        "h_star_fitted": False,
    }
    usable = array is not None and gradient_info.get("finite") is True and gradient_info.get("nonzero") is True
    if not usable:
        payload["budgets"] = {
            key: {
                "valid": False,
                "epsilon": epsilon(numerator),
                "reason": "gradient missing, non-finite, or zero",
            }
            for key, numerator in STAGE_A_BUDGETS
        }
        payload["finished"] = True
        payload["seconds"] = round(time.perf_counter() - started, 3)
        return payload
    gradient = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
    del array
    # The hash above used the array inside gradient_record during differentiate_scene.
    # Recompute it here only as a guard that the tensor we step is that array.
    recomputed = gradient_record(gradient.numpy())
    if recomputed["sha256"] != gradient_info["sha256"]:
        raise StageStop("stopped_scene_rule", "The stepped gradient is not the gradient that was hashed.")
    signs = matched_rademacher(gradient, scene_seed(STAGE_A_SEED_PREFIX, scene_id))
    prompt = prepare_prompt(processor, image_processor, clean, user_text)
    budgets: dict[str, Any] = {}
    illustration_scene, illustration_budget, illustration_mode = STAGE_A_ILLUSTRATION
    try:
        for key, numerator in STAGE_A_BUDGETS:
            eps = epsilon(numerator)
            block: dict[str, Any] = {"valid": True, "epsilon": eps, "reason": None}
            for mode in ("positive", "negative", "random"):
                projected = project_signed_step(clean, gradient, eps, mode, signs)
                geo, stored = geometry(clean, gradient, projected)
                float_score = score_differentiable(
                    model=model,
                    image_processor=image_processor,
                    prompt=prompt,
                    image=projected,
                    token_ids=token_ids,
                    label=label,
                    device=device,
                )
                official_score = score_official(
                    model=model,
                    processor=processor,
                    image=stored,
                    user_text=user_text,
                    token_ids=token_ids,
                    label=label,
                    device=device,
                )
                official_loss = official.get("loss")
                official_chosen = official.get("chosen") or ""
                step = build_step_record(
                    mode=mode,
                    geometry_record=geo,
                    float_score=float_score,
                    official_score=official_score,
                    clean_float_loss=float(measured["loss"]),
                    clean_float_chosen=str(measured["chosen"]),
                    clean_official_loss=None if official_loss is None else float(official_loss),
                    clean_official_chosen=str(official_chosen),
                    label=label,
                )
                block[mode] = step
                if (
                    scene_id == illustration_scene
                    and key == illustration_budget
                    and mode == illustration_mode
                ):
                    write_illustration(
                        figure_path(root),
                        clean,
                        stored,
                        f"{scene_id}  positive gradient-sign  {key}",
                        {
                            "stage": "A",
                            "scene_id": scene_id,
                            "budget": key,
                            "mode": mode,
                            "replaced_by_stage_b": False,
                        },
                    )
                del projected, stored
                release_cache(device)
            finite_changes = all(
                isinstance(block[mode].get("observed_change"), (int, float))
                and block[mode]["loss_identity_agreed"] is True
                for mode in ("positive", "negative", "random")
            )
            block["valid"] = bool(finite_changes)
            if not finite_changes:
                block["reason"] = "a float observed change was missing or the loss identity failed"
            budgets[key] = block
    except RuntimeError as exc:
        if is_memory_error(exc):
            raise StageStop(
                "stopped_oom",
                "The runtime reported out of memory. The budget was not lowered and the load was not retried.",
            ) from exc
        if "patch grid" in str(exc):
            raise StageStop("stopped_grid", str(exc)) from exc
        raise
    finally:
        del gradient, signs
        release_cache(device)
    payload["budgets"] = budgets
    payload["finished"] = True
    payload["seconds"] = round(time.perf_counter() - started, 3)
    return payload


def _official(model, processor, image, user_text, token_ids, label, device) -> dict[str, Any]:
    try:
        return score_official(
            model=model,
            processor=processor,
            image=image,
            user_text=user_text,
            token_ids=token_ids,
            label=label,
            device=device,
        )
    except RuntimeError as exc:
        if is_memory_error(exc):
            raise StageStop(
                "stopped_oom",
                "The runtime reported out of memory. The budget was not lowered and the load was not retried.",
            ) from exc
        raise


def _gap(live: dict | None, stored: dict | None) -> float | None:
    if not isinstance(live, dict) or not isinstance(stored, dict):
        return None
    return max(abs(float(live[color]) - float(stored[color])) for color in live)
