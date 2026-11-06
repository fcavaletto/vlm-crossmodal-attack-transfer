"""Stage B projected gradient ascent. Called only after Stage A passes."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from choice_eval.prompt import build_prompt
from dev_pilot.rows import user_text_sha256
from qwen_rgb_loss.measure import MemoryStop
from vlm_iface.images import to_image_01

from .figure import write_illustration
from .protocol import (
    PGD_STEPS,
    STAGE_B_BUDGETS,
    STAGE_B_EVAL_CONTEXTS,
    STAGE_B_ILLUSTRATION,
    STAGE_B_SEED_PREFIX,
    STAGE_B_SOURCE_CONTEXT,
    epsilon,
    figure_path,
)
from .score import (
    backward_at_image,
    geometry,
    is_memory_error,
    prepare_prompt,
    release_cache,
    score_differentiable,
    score_official,
)
from .stage_a import StageStop
from .steps import matched_rademacher, pgd_update, project_signed_step, scene_seed


def run_scene(
    *,
    root: Path,
    model,
    processor,
    image_processor,
    record: dict[str, Any],
    stored_rows: dict[tuple[str, str], dict[str, Any]],
    prior: dict[str, Any],
    token_ids: dict[str, tuple[int, ...]],
    device,
) -> dict[str, Any]:
    """Neutral PGD at three budgets, plus FGSM and matched noise. No extra epsilon."""
    started = time.perf_counter()
    scene_id = record["scene_id"]
    label = str(record["answer"])
    image = to_image_01(root / "fixtures" / "development_v4" / record["image_path"])
    texts: dict[str, str] = {}
    prompts: dict[str, dict[str, Any]] = {}
    for context in STAGE_B_EVAL_CONTEXTS:
        user_text = build_prompt(record["contexts"][context]["text"], record["question"]).user_text()
        stored = stored_rows[(scene_id, context)]
        if stored.get("user_text_sha256") != user_text_sha256(user_text):
            raise StageStop("stopped_prompt", f"{scene_id} {context} prompt hash does not match the stored row.")
        texts[context] = user_text
        prompts[context] = prepare_prompt(processor, image_processor, image, user_text)
    try:
        clean_float, gradient = backward_at_image(
            model=model,
            image_processor=image_processor,
            prompt=prompts[STAGE_B_SOURCE_CONTEXT],
            image=image,
            token_ids=token_ids,
            label=label,
            device=device,
        )
    except MemoryStop as exc:
        raise StageStop(
            "stopped_oom",
            "The runtime reported out of memory. The budget was not lowered and the load was not retried.",
        ) from exc
    if clean_float.get("loss_identity_agreed") is not True or gradient is None:
        raise StageStop("stopped_loss_identity", "The labeled-color NLL formulas disagreed, so PGD was not started.")
    if not bool(torch.isfinite(gradient).all()) or float(gradient.abs().sum()) == 0.0:
        gradient_ok = False
    else:
        gradient_ok = True
    clean_scores = _clean_scores(
        model, processor, image_processor, image, texts, prompts, token_ids, label, device, clean_float
    )
    signs = matched_rademacher(gradient, scene_seed(STAGE_B_SEED_PREFIX, scene_id)) if gradient_ok else None
    epsilons: dict[str, Any] = {}
    illustration_scene, illustration_budget, _mode = STAGE_B_ILLUSTRATION
    try:
        for key, numerator in STAGE_B_BUDGETS:
            eps = epsilon(numerator)
            if not gradient_ok or signs is None:
                epsilons[key] = {"epsilon": eps, "error": "the clean gradient was missing, non-finite, or zero"}
                continue
            fgsm = project_signed_step(image, gradient, eps, "positive")
            random_image = project_signed_step(image, gradient, eps, "random", signs)
            current = image.detach().clone()
            step_gradient = gradient
            trajectory = [clean_float["loss"]]
            for step_index in range(PGD_STEPS):
                if step_index > 0:
                    stepped, step_gradient = backward_at_image(
                        model=model,
                        image_processor=image_processor,
                        prompt=prompts[STAGE_B_SOURCE_CONTEXT],
                        image=current,
                        token_ids=token_ids,
                        label=label,
                        device=device,
                    )
                    if stepped.get("loss_identity_agreed") is not True or step_gradient is None:
                        raise StageStop(
                            "stopped_loss_identity",
                            "The labeled-color NLL formulas disagreed during PGD, so the objective was not changed.",
                        )
                    trajectory.append(stepped["loss"])
                current = pgd_update(current, image, step_gradient, eps / 4.0, eps)
                del step_gradient
                release_cache(device)
            final_float = score_differentiable(
                model=model,
                image_processor=image_processor,
                prompt=prompts[STAGE_B_SOURCE_CONTEXT],
                image=current,
                token_ids=token_ids,
                label=label,
                device=device,
            )
            trajectory.append(final_float.get("loss"))
            block = {
                "epsilon": eps,
                "pgd_steps": PGD_STEPS,
                "step_size": eps / 4.0,
                "init": "clean_image",
                "restarts": 0,
                "pgd_float_loss_trajectory": trajectory,
                "pgd": _evaluate(model, processor, image_processor, current, image, gradient, texts, prompts, token_ids, label, device, clean_scores),
                "fgsm": _evaluate(model, processor, image_processor, fgsm, image, gradient, texts, prompts, token_ids, label, device, clean_scores),
                "random": _evaluate(model, processor, image_processor, random_image, image, gradient, texts, prompts, token_ids, label, device, clean_scores),
            }
            if scene_id == illustration_scene and key == illustration_budget:
                _geo, stored = geometry(image, gradient, current)
                write_illustration(
                    figure_path(root),
                    image,
                    stored,
                    f"{scene_id}  PGD  {key}  7 steps",
                    {
                        "stage": "B",
                        "scene_id": scene_id,
                        "budget": key,
                        "mode": "pgd",
                        "replaced_by_stage_b": True,
                        "steps": PGD_STEPS,
                    },
                )
                del stored
            del fgsm, random_image, current
            release_cache(device)
            epsilons[key] = block
    except MemoryStop as exc:
        raise StageStop(
            "stopped_oom",
            "The runtime reported out of memory. The budget was not lowered and the load was not retried.",
        ) from exc
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
        gradient = None
        release_cache(device)
    return {
        "scene_id": scene_id,
        "answer": label,
        "finished": True,
        "seconds": round(time.perf_counter() - started, 3),
        "checkpoint_1_8_alignment_neutral_to_helpful": prior.get("alignment_neutral_to_helpful"),
        "checkpoint_1_8_neutral_margin": prior.get("neutral", {}).get("margin"),
        "checkpoint_1_8_helpful_margin": prior.get("helpful_nonrevealing", {}).get("margin"),
        "alignment_regressed": False,
        "h_star_fitted": False,
        "clean": clean_scores,
        "epsilons": epsilons,
        "objective": "summed labeled-color NLL under neutral",
    }


def _clean_scores(model, processor, image_processor, image, texts, prompts, token_ids, label, device, neutral_float) -> dict[str, Any]:
    scores: dict[str, Any] = {}
    for context in STAGE_B_EVAL_CONTEXTS:
        if context == STAGE_B_SOURCE_CONTEXT:
            float_score = neutral_float
        else:
            float_score = score_differentiable(
                model=model,
                image_processor=image_processor,
                prompt=prompts[context],
                image=image,
                token_ids=token_ids,
                label=label,
                device=device,
            )
        official = score_official(
            model=model,
            processor=processor,
            image=image,
            user_text=texts[context],
            token_ids=token_ids,
            label=label,
            device=device,
        )
        scores[context] = {
            "loss": float_score.get("loss"),
            "float_margin": float_score.get("margin"),
            "float_chosen": float_score.get("chosen"),
            "summed_nll": float_score.get("summed_nll"),
            "official_loss": official.get("loss"),
            "official_margin": official.get("margin"),
            "official_chosen": official.get("chosen"),
            "official_summed_nll": official.get("summed_nll"),
            "margin": official.get("margin"),
            "chosen": official.get("chosen"),
            "labeled_nll": official.get("loss"),
        }
    return scores


def _view(score: dict[str, Any], baseline_loss: float | None, baseline_chosen: str | None, label: str) -> dict[str, Any]:
    loss = score.get("loss")
    increase = None if loss is None or baseline_loss is None else float(loss) - float(baseline_loss)
    return {
        "labeled_nll": loss,
        "labeled_nll_increase": increase,
        "margin": score.get("margin"),
        "chosen": score.get("chosen"),
        "summed_nll": score.get("summed_nll"),
        "loss_identity_agreed": score.get("loss_identity_agreed"),
        "answer_changed": score.get("chosen") != baseline_chosen,
        "left_label": score.get("chosen") != label,
    }


def _evaluate(model, processor, image_processor, image, clean, gradient, texts, prompts, token_ids, label, device, clean_scores) -> dict[str, Any]:
    geo, stored = geometry(clean, gradient, image)
    float_views = {}
    official_views = {}
    for context in STAGE_B_EVAL_CONTEXTS:
        float_score = score_differentiable(
            model=model,
            image_processor=image_processor,
            prompt=prompts[context],
            image=image,
            token_ids=token_ids,
            label=label,
            device=device,
        )
        official_score = score_official(
            model=model,
            processor=processor,
            image=stored,
            user_text=texts[context],
            token_ids=token_ids,
            label=label,
            device=device,
        )
        baseline = clean_scores[context]
        float_views[context] = _view(float_score, baseline.get("loss"), baseline.get("float_chosen"), label)
        official_views[context] = _view(official_score, baseline.get("official_loss"), baseline.get("chosen"), label)
        release_cache(device)
    result = {
        "float": {"linf": geo["linf"], **float_views},
        "uint8": {
            **geo["uint8_geometry"],
            **official_views,
        },
    }
    del stored
    return result
