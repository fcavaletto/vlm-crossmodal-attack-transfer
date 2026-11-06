"""Unselected sensitivities.

Dropping the digit surfaces changes L on constructed logits. Reading a single
label out of an invalid sentence changes the development counts. Neither
change is an outcome. The development file stores pooled class losses, so the
digit piece cannot be removed from those stored numbers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from .generation import embedded_single_label
from .loss import class_nll_from_surfaces
from .protocol import CONTEXTS, FAMILIES
from .surfaces import DIGIT_KINDS, SurfaceInventory


def losses_without_kinds(
    step_logits: Sequence[torch.Tensor],
    inventory: SurfaceInventory,
    kinds: frozenset[str] | set[str],
) -> dict[str, float]:
    """Class losses after dropping surfaces whose kind is in ``kinds``."""
    reduced = inventory.without_kinds(kinds)
    return {
        name: float(class_nll_from_surfaces(step_logits, surfaces))
        for name, surfaces in reduced.by_class.items()
    }


def digit_kinds() -> frozenset[str]:
    return DIGIT_KINDS


def embedded_read_counterfactual(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Counts that would change if ``not_exact`` sentences were scored.

    ``adopted`` is false. Rows that are already strict-correct stay correct.
    A sentence is credited only when the embedded label equals the answer.
    """
    applicable = 0
    equals_answer = 0
    differs = 0
    cells: dict[str, dict[str, dict[str, int]]] = {}
    for family in (*FAMILIES, "all"):
        cells[family] = {}
        for context in CONTEXTS:
            subset = [
                row
                for row in rows
                if (family == "all" or row["task_family"] == family) and row["context"] == context
            ]
            correct = 0
            for row in subset:
                label = embedded_single_label(
                    str(row["raw_output"]),
                    truncated=bool(row["truncated"]),
                    task_family=str(row["task_family"]),
                )
                if label is None:
                    correct += int(bool(row["correct"]))
                else:
                    correct += int(label == row["answer"])
            cells[family][context] = {"n": len(subset), "correct_if_adopted": correct}
    for row in rows:
        label = embedded_single_label(
            str(row["raw_output"]),
            truncated=bool(row["truncated"]),
            task_family=str(row["task_family"]),
        )
        if label is None:
            continue
        applicable += 1
        if label == row["answer"]:
            equals_answer += 1
        else:
            differs += 1
    return {
        "adopted": False,
        "n_applicable": applicable,
        "n_equals_answer": equals_answer,
        "n_differs_from_answer": differs,
        "by_family": cells,
    }
