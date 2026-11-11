"""Stop rules from ATTACK_PROTOCOL.md. This module does not load a model."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .protocol import STAGE_A_BUDGETS, STAGE_A_MIN_SCENES, STAGE_A_SCENE_COUNT, STAGE_A_SCENES, STAGE_B_BUDGETS


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def judge_stage_a(scenes: dict[str, Any]) -> dict[str, Any]:
    """Pass only when every listed budget beats both controls on the float loss."""
    budgets: dict[str, Any] = {}
    passed = True
    for key, _numerator in STAGE_A_BUDGETS:
        rows: list[dict[str, Any]] = []
        invalid_ids: list[str] = []
        for scene_id, _answer, _stratum, _alignment in STAGE_A_SCENES:
            block = scenes.get(scene_id, {}).get("budgets", {}).get(key)
            if not isinstance(block, dict) or block.get("valid") is not True:
                invalid_ids.append(scene_id)
                continue
            positive = _finite(block.get("positive", {}).get("observed_change"))
            negative = _finite(block.get("negative", {}).get("observed_change"))
            random = _finite(block.get("random", {}).get("observed_change"))
            if positive is None or negative is None or random is None:
                invalid_ids.append(scene_id)
                continue
            rows.append({"positive": positive, "negative": negative, "random": random, "scene_id": scene_id})
        report: dict[str, Any] = {
            "budget": key,
            "n_scenes": STAGE_A_SCENE_COUNT,
            "n_valid": len(rows),
            "invalid_scene_ids": invalid_ids,
            "min_scenes": STAGE_A_MIN_SCENES,
        }
        if invalid_ids or len(rows) != STAGE_A_SCENE_COUNT:
            report["passed"] = False
            report["reason"] = "a scene is missing a finite observed change and was not imputed as zero"
            passed = False
            budgets[key] = report
            continue
        positive = [row["positive"] for row in rows]
        negative = [row["negative"] for row in rows]
        random = [row["random"] for row in rows]
        raised = sum(value > 0.0 for value in positive)
        beat_negative = sum(left > right for left, right in zip(positive, negative))
        beat_random = sum(left > right for left, right in zip(positive, random))
        median_positive = _median(positive)
        median_negative = _median(negative)
        median_random = _median(random)
        budget_passed = (
            raised >= STAGE_A_MIN_SCENES
            and beat_negative >= STAGE_A_MIN_SCENES
            and beat_random >= STAGE_A_MIN_SCENES
            and median_positive > 0.0
            and median_positive > median_negative
            and median_positive > median_random
        )
        report.update(
            {
                "passed": budget_passed,
                "n_raised": raised,
                "n_beat_negative": beat_negative,
                "n_beat_random": beat_random,
                "median_positive": median_positive,
                "median_negative": median_negative,
                "median_random": median_random,
                "reason": None if budget_passed else "positive-sign steps did not beat both controls at this budget",
            }
        )
        if not budget_passed:
            passed = False
        budgets[key] = report
    return {
        "passed": passed,
        "meaningful_advantage": passed,
        "budgets": budgets,
        "stage_b_allowed": passed,
        "rule": (
            "Every budget needs 6 of 8 strict rises, 6 of 8 strict wins over each control, "
            "and a positive median strictly above both control medians. One invalid cell fails that budget."
        ),
    }


def _eligible_increases(scenes: list[dict[str, Any]], budget: str) -> dict[str, Any]:
    increases: list[float] = []
    fgsm: list[float] = []
    random: list[float] = []
    flips = 0
    eligible = 0
    for scene in scenes:
        answer = scene.get("answer")
        clean = scene.get("clean", {}).get("neutral", {})
        if clean.get("chosen") != answer:
            continue
        eligible += 1
        block = scene.get("epsilons", {}).get(budget, {})
        pgd_chosen = block.get("pgd", {}).get("uint8", {}).get("neutral", {}).get("chosen")
        if pgd_chosen != answer:
            flips += 1
        for name, sink in (("pgd", increases), ("fgsm", fgsm), ("random", random)):
            value = _finite(block.get(name, {}).get("uint8", {}).get("neutral", {}).get("labeled_nll_increase"))
            if value is None:
                return {
                    "eligible": eligible,
                    "valid": False,
                    "reason": f"{scene.get('scene_id')} is missing a finite {name} uint8 increase",
                }
            sink.append(value)
    return {
        "eligible": eligible,
        "valid": True,
        "flips": flips,
        "increases": increases,
        "fgsm": fgsm,
        "random": random,
    }


def judge_stage_b(scenes: list[dict[str, Any]]) -> dict[str, Any]:
    """PGD is ineffective at a budget when it flips nobody and does not beat both medians."""
    budgets: dict[str, Any] = {}
    any_effective = False
    for key, _numerator in STAGE_B_BUDGETS:
        counted = _eligible_increases(scenes, key)
        if counted.get("valid") is not True or counted["eligible"] == 0:
            report = {
                "budget": key,
                "effective": False,
                "source_eligible": counted.get("eligible", 0),
                "source_flips": None,
                "reason": counted.get("reason") or "no source-eligible scene",
            }
            budgets[key] = report
            continue
        median_pgd = _median(counted["increases"])
        median_fgsm = _median(counted["fgsm"])
        median_random = _median(counted["random"])
        beats = median_pgd > median_fgsm and median_pgd > median_random
        effective = counted["flips"] > 0 or beats
        any_effective = any_effective or effective
        budgets[key] = {
            "budget": key,
            "effective": effective,
            "source_eligible": counted["eligible"],
            "source_flips": counted["flips"],
            "source_flip_rate": counted["flips"] / counted["eligible"],
            "median_uint8_nll_increase": median_pgd,
            "median_fgsm": median_fgsm,
            "median_random": median_random,
            "median_beats_both_controls": beats,
            "reason": None if effective else "no source flip and the median increase does not beat both controls",
        }
    return {
        "budgets": budgets,
        "effective_at_any_listed_budget": any_effective,
        "ineffective_at_every_listed_budget": not any_effective,
        "raise_epsilon": False,
        "objective_changed": False,
    }
