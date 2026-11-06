"""One closed-vocabulary row, and a summary that does not blend metrics.

``attack_loss`` is the labeled color's summed NLL. The decision is the argmin
of that same reduction. The mean is stored and does not choose the color.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from .colors import COLORS, require_color, require_color_keys
from .decision import choose_color, rank_colors
from .diagnostic import diagnose_free_generation
from .likelihood import mean_nll, summed_nll
from .protocol import (
    ATTACK_LOSS,
    CORRECTS_CHECKPOINT_1_6,
    PRIMARY_METRIC,
    PRIMARY_REDUCTION,
    PROTOCOL_ID,
    PROTOCOL_STATUS,
    TIE_BREAK,
)


def score_example(
    predicting_logits_by_color: Mapping[str, torch.Tensor],
    token_ids_by_color: Mapping[str, Sequence[int] | torch.Tensor],
    *,
    label: str,
    generation_text: str | None = None,
    generation_truncated: bool = False,
) -> dict[str, object]:
    """Score the four full sequences. Generation text does not change the choice."""
    require_color(label)
    require_color_keys(predicting_logits_by_color, what="logits")
    require_color_keys(token_ids_by_color, what="token ids")
    summed = {
        color: summed_nll(predicting_logits_by_color[color], token_ids_by_color[color]) for color in COLORS
    }
    mean = {color: mean_nll(predicting_logits_by_color[color], token_ids_by_color[color]) for color in COLORS}
    chosen = choose_color(summed)
    loss = summed[label]
    row: dict[str, object] = {
        "protocol_status": PROTOCOL_STATUS,
        "protocol_id": PROTOCOL_ID,
        "corrects_checkpoint_1_6": CORRECTS_CHECKPOINT_1_6,
        "primary_metric": PRIMARY_METRIC,
        "primary_reduction": PRIMARY_REDUCTION,
        "attack_loss_name": ATTACK_LOSS,
        "tie_break": TIE_BREAK,
        "label": label,
        "chosen": chosen,
        "closed_vocabulary_correct": chosen == label,
        "target_loss": loss,
        "attack_loss": loss,
        "summed_nll": summed,
        "mean_nll": mean,
        "summed_nll_value": {color: _value(summed[color]) for color in COLORS},
        "mean_nll_value": {color: _value(mean[color]) for color in COLORS},
        "rank_by_sum": rank_colors(summed),
        "rank_by_mean": rank_colors(mean),
        "reductions_agree": choose_color(mean) == chosen,
        "free_generation": None,
    }
    if generation_text is not None:
        row["free_generation"] = diagnose_free_generation(
            generation_text, truncated=generation_truncated, label=label
        )
    return row


def summarize(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Closed-vocabulary accuracy beside the free-generation partition.

    The invalid rate uses every row as its denominator. Invalid strings are
    not removed before that rate, and they are not folded into the
    closed-vocabulary accuracy.
    """
    if not rows:
        raise ValueError("no rows")
    closed = 0
    correct = 0
    valid_but_wrong = 0
    invalid = 0
    for row in rows:
        closed += int(bool(row["closed_vocabulary_correct"]))
        free = row["free_generation"]
        if not isinstance(free, Mapping):
            raise ValueError("summarize requires a free-generation diagnostic on every row")
        if free["is_primary"]:
            raise RuntimeError("free generation is not the primary metric")
        correct += int(bool(free["correct"]))
        valid_but_wrong += int(bool(free["valid_but_wrong"]))
        invalid += int(bool(free["invalid"]))
    n = len(rows)
    if correct + valid_but_wrong + invalid != n:
        raise RuntimeError("free-generation partition does not cover every row")
    return {
        "n": n,
        "closed_vocabulary_correct": closed,
        "closed_vocabulary_accuracy": closed / n,
        "free_generation_correct": correct,
        "free_generation_valid_but_wrong": valid_but_wrong,
        "free_generation_invalid": invalid,
        "free_generation_strict_accuracy": correct / n,
        "free_generation_invalid_rate": invalid / n,
        "metrics_combined": False,
    }


def _value(tensor: torch.Tensor) -> float:
    number = float(tensor.detach().to(dtype=torch.float64).cpu())
    if number != number or number in (float("inf"), float("-inf")):
        raise RuntimeError("score is not finite")
    return number
