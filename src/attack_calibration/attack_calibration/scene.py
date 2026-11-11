"""One development scene. The steps and the official scores come from Checkpoint 1.9."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from attack_feasibility.score import (
    backward_at_image,
    is_memory_error,
    prepare_prompt,
    release_cache,
    score_official,
)
from attack_feasibility.steps import matched_rademacher, pgd_update, project_signed_step
from choice_eval.prompt import build_prompt
from dev_pilot.rows import user_text_sha256
from qwen_rgb.constraints import round_trip_uint8
from qwen_rgb_loss.measure import MemoryStop
from vlm_iface.devices import available_bytes, memory_snapshot, synchronize
from vlm_iface.images import to_image_01

from .counts import scene_flags
from .protocol import (
    BUDGETS,
    EVAL_CONTEXTS,
    PGD_STEPS,
    RECORDED_STEPS,
    SEED_PREFIX,
    SOURCE_CONTEXT,
    CalibrationStop,
    epsilon,
)


def exact_linf(image: torch.Tensor, other: torch.Tensor) -> float:
    """L-infinity distance in float64 of the stored float32 values."""
    delta = image.detach().to(dtype=torch.float64, device="cpu") - other.detach().to(dtype=torch.float64, device="cpu")
    return float(delta.abs().max().cpu())


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
    """Neutral PGD at the listed budgets, plus FGSM and one matched random-sign step."""
    started = time.perf_counter()
    scene_id = str(record["scene_id"])
    label = str(record["answer"])
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
            result["fatal_reason"] = (
                "The labeled-color NLL formulas disagreed, so no step was taken."
            )
            return result
        result["objective_identity_agreed"] = True
        if not bool(torch.isfinite(gradient).all()) or float(gradient.abs().sum()) == 0.0:
            result["finished"] = True
            result["error"] = "the clean gradient was missing, non-finite, or zero"
            result["clean"] = _clean_official(
                model, processor, image, texts, token_ids, label, device
            )
            return result
        attack_seed = _seed(scene_id)
        signs = matched_rademacher(gradient, attack_seed)
        result["seed"] = attack_seed
        result["clean"] = _clean_official(
            model, processor, image, texts, token_ids, label, device
        )
        for key, numerator in BUDGETS:
            block: dict[str, Any] = {
                "epsilon": epsilon(numerator),
                "step_size": epsilon(numerator) / 4.0,
                "pgd_steps": PGD_STEPS,
                "recorded_steps": list(RECORDED_STEPS),
                "init": "clean_image",
                "restarts": 0,
                "momentum": False,
                "pgd": {},
                "error": None,
            }
            result["epsilons"][key] = block
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
                epsilon_value=epsilon(numerator),
                clean_views=result["clean"],
            )
            release_cache(device)
        result["finished"] = True
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


def _seed(scene_id: str) -> int:
    from attack_feasibility.steps import scene_seed

    return scene_seed(SEED_PREFIX, scene_id)


def _development_image(root: Path, record: dict[str, Any]) -> torch.Tensor:
    relative = Path(str(record["image_path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise CalibrationStop("stopped_scene_rule", f"{record['scene_id']} image path leaves development_v4.")
    base = (root / "fixtures" / "development_v4").resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base):
        raise CalibrationStop("stopped_scene_rule", f"{record['scene_id']} image path leaves development_v4.")
    if record.get("split") != "development":
        raise CalibrationStop("stopped_scene_rule", f"{record['scene_id']} is not a development scene.")
    return to_image_01(path)


def _prompts(processor, image_processor, image, record, stored_rows):
    texts: dict[str, str] = {}
    prompts: dict[str, dict[str, Any]] = {}
    scene_id = record["scene_id"]
    for context in EVAL_CONTEXTS:
        user_text = build_prompt(record["contexts"][context]["text"], record["question"]).user_text()
        stored = stored_rows[(scene_id, context)]
        if stored.get("user_text_sha256") != user_text_sha256(user_text):
            raise CalibrationStop(
                "stopped_prompt",
                f"{scene_id} {context} prompt hash does not match the stored row.",
            )
        texts[context] = user_text
        if context == SOURCE_CONTEXT:
            prompts[context] = prepare_prompt(processor, image_processor, image, user_text)
    return texts, prompts


def _clean_official(model, processor, image, texts, token_ids, label, device) -> dict[str, Any]:
    views = {}
    for context in EVAL_CONTEXTS:
        views[context] = _official_view(
            score_official(
                model=model,
                processor=processor,
                image=image,
                user_text=texts[context],
                token_ids=token_ids,
                label=label,
                device=device,
            )
        )
        release_cache(device)
    return views


def _official_view(score: dict[str, Any]) -> dict[str, Any]:
    if score.get("loss_identity_agreed") is not True or score.get("loss") is None:
        return {
            "decision": None,
            "margin": None,
            "nll": None,
            "error": "the labeled-color NLL formulas did not agree",
        }
    decision = score.get("chosen")
    margin = score.get("margin")
    nll = score.get("loss")
    if (
        not isinstance(decision, str)
        or isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or isinstance(nll, bool)
        or not isinstance(nll, (int, float))
    ):
        return {
            "decision": None,
            "margin": None,
            "nll": None,
            "error": "the official score was missing a decision, margin, or NLL",
        }
    if not torch.isfinite(torch.tensor([float(margin), float(nll)])).all():
        return {"decision": None, "margin": None, "nll": None, "error": "the official score was not finite"}
    return {"decision": decision, "margin": float(margin), "nll": float(nll), "error": None}


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
                raise CalibrationStop(
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
        block[name] = scored
        release_cache(device)
    del fgsm, random_image, current


def _score_image(model, processor, projected, clean, texts, token_ids, label, device, clean_views) -> dict[str, Any]:
    stored = round_trip_uint8(projected)
    views: dict[str, Any] = {}
    errors: list[str] = []
    for context in EVAL_CONTEXTS:
        view = _official_view(
            score_official(
                model=model,
                processor=processor,
                image=stored,
                user_text=texts[context],
                token_ids=token_ids,
                label=label,
                device=device,
            )
        )
        if view["error"]:
            errors.append(f"{context}: {view['error']}")
        views[context] = {"decision": view["decision"], "margin": view["margin"], "nll": view["nll"]}
        release_cache(device)
    synchronize(device)
    flags = scene_flags(label, clean_views, views)
    return {
        "perturbed": views,
        "linf_before_round_trip": exact_linf(projected, clean),
        "linf_after_round_trip": exact_linf(stored, clean),
        "linf_stored_vs_projected": exact_linf(stored, projected),
        "source_flip": flags["source_flip"],
        "target_flip": flags["target_flip"],
        "transfer": flags["transfer"],
        "transfer_status": flags["transfer_status"],
        "memory": memory_snapshot(device),
        "os_available_bytes": available_bytes(),
        "error": "; ".join(errors) or None,
    }
