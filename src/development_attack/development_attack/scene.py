"""One development scene at epsilon 1/255.

The step, the seed, and the official score are the Checkpoint 2.0 functions.
This module records step 3 and step 7 of that trajectory, plus FGSM and one
matched random-sign step. It does not add an epsilon.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from attack_calibration.protocol import CalibrationStop
from attack_calibration.scene import _clean_official, _development_image, _prompts, _score_image
from attack_feasibility.score import backward_at_image, is_memory_error, release_cache
from attack_feasibility.steps import matched_rademacher, pgd_update, project_signed_step, scene_seed
from qwen_rgb_loss.measure import MemoryStop
from qwen_rgb_loss.records import gradient_record

from .protocol import (
    EPSILON_KEY,
    PGD_STEPS,
    PRIMARY_STEPS,
    RECORDED_STEPS,
    SEED_PREFIX,
    SOURCE_CONTEXT,
    DevelopmentStop,
    epsilon,
)
from .verify import bound_errors, decision_errors


def run_scene(
    *,
    root: Path,
    model,
    processor,
    image_processor,
    record: dict[str, Any],
    stored_rows: dict[tuple[str, str], dict[str, Any]],
    token_ids: dict[str, tuple[int, ...]],
    device,
) -> dict[str, Any]:
    """Neutral PGD at 1/255. The clean image is the only start."""
    started = time.perf_counter()
    scene_id = str(record["scene_id"])
    label = str(record["answer"])
    radius = epsilon()
    result: dict[str, Any] = {
        "scene_id": scene_id,
        "counterfactual_group_id": record["counterfactual_group_id"],
        "answer": label,
        "finished": False,
        "objective": "summed labeled-color NLL under neutral",
        "objective_identity_agreed": None,
        "seed_prefix": SEED_PREFIX,
        "h_star_fitted": False,
        "held_out_opened": False,
        "epsilons": {},
        "recomputed_neutral_gradient_l1": None,
        "error": None,
        "fatal_status": None,
        "fatal_reason": None,
    }
    gradient = None
    try:
        image = _development_image(root, record)
        texts, prompts = _prompts(processor, image_processor, image, record, stored_rows)
        clean_float, gradient = backward_at_image(
            model=model,
            image_processor=image_processor,
            prompt=prompts[SOURCE_CONTEXT],
            image=image,
            token_ids=token_ids,
            label=label,
            device=device,
        )
        if clean_float.get("loss_identity_agreed") is not True or gradient is None:
            result["objective_identity_agreed"] = False
            result["fatal_status"] = "stopped_loss_identity"
            result["fatal_reason"] = "The labeled-color NLL formulas disagreed, so no step was taken."
            return result
        result["objective_identity_agreed"] = True
        if gradient is not None:
            values = gradient.detach().to(dtype=torch.float64).cpu().numpy()
            result["recomputed_neutral_gradient_l1"] = gradient_record(values)["l1"]
            del values
        if not bool(torch.isfinite(gradient).all()) or float(gradient.abs().sum()) == 0.0:
            result["finished"] = True
            result["error"] = "the clean gradient was missing, non-finite, or zero"
            result["clean"] = _clean_official(model, processor, image, texts, token_ids, label, device)
            _assert_clean(result["clean"])
            return result
        attack_seed = scene_seed(SEED_PREFIX, scene_id)
        signs = matched_rademacher(gradient, attack_seed)
        result["seed"] = attack_seed
        result["clean"] = _clean_official(model, processor, image, texts, token_ids, label, device)
        _assert_clean(result["clean"])
        block: dict[str, Any] = {
            "epsilon": radius,
            "step_size": radius / 4.0,
            "pgd_steps": PGD_STEPS,
            "recorded_steps": list(RECORDED_STEPS),
            "primary_steps": PRIMARY_STEPS,
            "init": "clean_image",
            "restarts": 0,
            "momentum": False,
            "pgd": {},
            "error": None,
        }
        result["epsilons"][EPSILON_KEY] = block
        _one_budget(
            block,
            model=model,
            processor=processor,
            image_processor=image_processor,
            image=image,
            gradient=gradient,
            signs=signs,
            texts=texts,
            prompt=prompts[SOURCE_CONTEXT],
            token_ids=token_ids,
            label=label,
            device=device,
            epsilon_value=radius,
            clean_views=result["clean"],
        )
        result["finished"] = True
    except DevelopmentStop as exc:
        result["fatal_status"] = exc.status
        result["fatal_reason"] = str(exc)
    except CalibrationStop as exc:
        result["fatal_status"] = exc.status
        result["fatal_reason"] = str(exc)
    except MemoryStop as exc:
        result["fatal_status"] = "stopped_oom"
        result["fatal_reason"] = (
            "The runtime reported out of memory. The budget was not lowered and the load was not retried."
        )
        result["error"] = str(exc)
    except RuntimeError as exc:
        if is_memory_error(exc):
            result["fatal_status"] = "stopped_oom"
            result["fatal_reason"] = (
                "The runtime reported out of memory. The budget was not lowered and the load was not retried."
            )
            result["error"] = str(exc)
        elif "patch grid" in str(exc) or "chat-template grid" in str(exc):
            result["fatal_status"] = "stopped_grid"
            result["fatal_reason"] = str(exc)
        else:
            raise
    finally:
        gradient = None
        result["seconds"] = round(time.perf_counter() - started, 3)
        release_cache(device)
    return result


def _one_budget(
    block: dict[str, Any],
    *,
    model,
    processor,
    image_processor,
    image,
    gradient,
    signs,
    texts,
    prompt,
    token_ids,
    label,
    device,
    epsilon_value: float,
    clean_views: dict[str, Any],
) -> None:
    fgsm = project_signed_step(image, gradient, epsilon_value, "positive")
    random_image = project_signed_step(image, gradient, epsilon_value, "random", signs)
    current = image.detach().clone()
    step_gradient = gradient
    for step_index in range(PGD_STEPS):
        step_started = time.perf_counter()
        if step_index > 0:
            stepped, step_gradient = backward_at_image(
                model=model,
                image_processor=image_processor,
                prompt=prompt,
                image=current,
                token_ids=token_ids,
                label=label,
                device=device,
            )
            if stepped.get("loss_identity_agreed") is not True or step_gradient is None:
                raise DevelopmentStop(
                    "stopped_loss_identity",
                    "The labeled-color NLL formulas disagreed during PGD, so the objective was not changed.",
                )
            if not bool(torch.isfinite(step_gradient).all()):
                block["error"] = "a PGD gradient was not finite; later steps were not taken"
                break
        current = pgd_update(current, image, step_gradient, epsilon_value / 4.0, epsilon_value)
        done = step_index + 1
        if done in RECORDED_STEPS:
            scored = _score_image(
                model, processor, current, image, texts, token_ids, label, device, clean_views
            )
            scored["seconds"] = round(time.perf_counter() - step_started, 3)
            _assert_scored(scored, epsilon_value)
            block["pgd"][str(done)] = scored
        if step_index > 0:
            del step_gradient
        release_cache(device)
    for name, proposed in (("fgsm", fgsm), ("random", random_image)):
        control_started = time.perf_counter()
        scored = _score_image(
            model, processor, proposed, image, texts, token_ids, label, device, clean_views
        )
        scored["seconds"] = round(time.perf_counter() - control_started, 3)
        _assert_scored(scored, epsilon_value)
        block[name] = scored
        release_cache(device)
    del fgsm, random_image, current


def _assert_clean(clean: dict[str, Any]) -> None:
    for context, view in clean.items():
        errors = decision_errors(view.get("decision") if isinstance(view, dict) else None)
        if errors:
            raise DevelopmentStop("stopped_decision", f"{context}: " + "; ".join(errors))


def _assert_scored(scored: dict[str, Any], epsilon_value: float) -> None:
    bounds = bound_errors(
        scored.get("linf_before_round_trip"),
        scored.get("linf_after_round_trip"),
        epsilon_value,
    )
    if bounds:
        raise DevelopmentStop("stopped_bounds", "; ".join(bounds))
    perturbed = scored.get("perturbed") or {}
    for context, view in perturbed.items():
        errors = decision_errors(view.get("decision") if isinstance(view, dict) else view)
        if errors:
            raise DevelopmentStop("stopped_decision", f"{context}: " + "; ".join(errors))
