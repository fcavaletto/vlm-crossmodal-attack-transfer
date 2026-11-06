"""Compare closed-vocabulary losses with a strict parse of a generation.

Word-only NLL is the Checkpoint 1 canonical token. Class NLL adds the other
single-token surfaces. Official correctness always follows the strict parse.
The embedded-label reading is stored beside it and is not selected.
"""

from __future__ import annotations

from typing import Any

import torch

from pilot.rows import DISAGREEMENT_TOLERANCE, judge_parse, scoring_disagreement

from .class_loss import argmin_name, mixture_nll, word_only_nll
from .reading import side_by_side
from .surfaces import SurfaceSet


def score_logits(
    logits: torch.Tensor,
    surfaces: SurfaceSet,
    *,
    raw_output: str,
    truncated: bool,
    answer: str,
    aliases: list[str],
    extra_logprobs: dict[tuple[int, ...], float] | None = None,
) -> dict[str, Any]:
    """Score one prompt against one generation string.

    Single-token surfaces are read from ``logits``. Longer surfaces, including
    a two-token digit alias, must be supplied in ``extra_logprobs`` as joint
    log-probabilities.
    """
    if surfaces.task_family == "":
        raise ValueError("surface set has no task family")
    supplied = extra_logprobs or {}
    word_scores: dict[str, float] = {}
    class_scores: dict[str, float] = {}
    surface_rows: list[dict[str, Any]] = []
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    for canonical, group in surfaces.by_class.items():
        token_id = surfaces.canonical_token_id(canonical)
        word_scores[canonical] = float(word_only_nll(logits, token_id).detach().cpu())
        collected: list[float] = []
        for surface in group:
            if len(surface.token_ids) == 1:
                logprob = float(log_probs[surface.token_ids[0]].detach().cpu())
            else:
                try:
                    logprob = float(supplied[surface.token_ids])
                except KeyError as exc:
                    raise RuntimeError(
                        f"Missing sequence log-probability for {surface.token_ids}."
                    ) from exc
            collected.append(logprob)
            surface_rows.append({**surface.as_dict(), "logprob": logprob})
        class_scores[canonical] = float(mixture_nll(torch.tensor(collected)).detach().cpu())
    reading = side_by_side(raw_output, truncated=truncated, task_family=surfaces.task_family)
    judgment = judge_parse(
        reading["strict_parsed_answer"],  # type: ignore[arg-type]
        bool(reading["strict_invalid"]),
        answer,
        aliases,
    )
    word_argmin, _word_best, word_flag = scoring_disagreement(
        reading["strict_parsed_answer"],  # type: ignore[arg-type]
        bool(reading["strict_invalid"]),
        {name: {"sum_nll": value} for name, value in word_scores.items()},
    )
    class_argmin, _class_best, class_flag = scoring_disagreement(
        reading["strict_parsed_answer"],  # type: ignore[arg-type]
        bool(reading["strict_invalid"]),
        {name: {"sum_nll": value} for name, value in class_scores.items()},
    )
    if argmin_name(word_scores) != word_argmin or argmin_name(class_scores) != class_argmin:
        raise RuntimeError("argmin helpers disagree")
    alternative = reading["alternative_parsed_answer"]
    return {
        "strict_text": reading["strict_text"],
        "strict_parsed_answer": reading["strict_parsed_answer"],
        "strict_invalid": judgment["invalid"],
        "strict_reasons": reading["strict_reasons"],
        "strict_correct": judgment["correct"],
        "strict_valid_wrong": judgment["valid_wrong"],
        "alternative_name": reading["alternative_name"],
        "alternative_selected": False,
        "alternative_parsed_answer": alternative,
        "alternative_applicable": reading["alternative_applicable"],
        "alternative_matches_answer": alternative == answer if alternative is not None else None,
        "word_only_nll": word_scores,
        "word_only_argmin": word_argmin,
        "word_only_disagreement": word_flag,
        "class_nll": class_scores,
        "class_argmin": class_argmin,
        "class_disagreement": class_flag,
        "class_argmin_matches_strict": (not judgment["invalid"]) and class_argmin == reading["strict_parsed_answer"],
        "word_argmin_matches_strict": (not judgment["invalid"]) and word_argmin == reading["strict_parsed_answer"],
        "class_differs_from_word": class_argmin != word_argmin,
        "true_word_nll": word_scores[answer],
        "true_class_nll": class_scores[answer],
        "surfaces": surface_rows,
        "disagreement_tolerance": DISAGREEMENT_TOLERANCE,
    }


def attach_first_token(row: dict[str, Any], token_ids: list[int] | None) -> dict[str, Any]:
    """Record whether the first generated token is inside the closed surface set."""
    copied = dict(row)
    if not token_ids:
        copied["first_token_id"] = None
        copied["first_token_class"] = None
        copied["first_token_in_surface_set"] = None
        return copied
    first = int(token_ids[0])
    found = None
    prefix = False
    for surface in row["surfaces"]:
        sequence = [int(token) for token in surface["token_ids"]]
        if sequence == [first]:
            found = surface["canonical"]
        elif sequence and sequence[0] == first and len(sequence) > 1:
            prefix = True
    copied["first_token_id"] = first
    copied["first_token_class"] = found
    copied["first_token_in_surface_set"] = found is not None
    copied["first_token_prefixes_multitoken_surface"] = prefix
    return copied
