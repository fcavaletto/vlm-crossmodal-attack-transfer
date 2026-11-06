"""The kickoff metrics are different questions.

Checkpoint 0 named six summaries. Checkpoint 1.6 scored only the clean
generation table, and it kept the strict parse separate from the class-loss
argmin. H* is a later coefficient on attack rows. A rate that moves because
the parser changes is still a different estimand from that coefficient.
"""

from __future__ import annotations

from typing import Any

# Prospective primary score from the Checkpoint 1.7 protocol. It is not the
# official Checkpoint 1.6 number, and it is not chosen from the row-color table.
PROSPECTIVE_PRIMARY = "teacher_forced_summed_nll"

KICKOFF_METRICS: tuple[dict[str, Any], ...] = (
    {
        "name": "clean_accuracy",
        "question": "Among scenes in one context, what share are strict-correct on the clean image?",
        "denominator": "scenes in that context",
        "invalid_output_counts_as": "incorrect",
        "requires_attack": False,
        "is_h_star": False,
    },
    {
        "name": "invalid_output_rate",
        "question": "Among scenes in one context, what share of generations fail the strict parser?",
        "denominator": "scenes in that context",
        "invalid_output_counts_as": "the event being counted",
        "requires_attack": False,
        "is_h_star": False,
    },
    {
        "name": "attack_success_rate",
        "question": "Among scenes that were strict-correct before the attack, what share become valid and wrong?",
        "denominator": "scenes strict-correct before the attack",
        "invalid_output_counts_as": "stays in the denominator and is not a success",
        "requires_attack": True,
        "is_h_star": False,
    },
    {
        "name": "additional_error",
        "question": "By how much does task error rise from the clean image to the attacked image?",
        "denominator": "scenes in the cell",
        "invalid_output_counts_as": "task error 1",
        "requires_attack": True,
        "is_h_star": False,
    },
    {
        "name": "cross_context_attack_transfer",
        "question": "What are attack success and additional error when the source context and the target context differ?",
        "denominator": "the attack-success or additional-error denominator for that ordered pair",
        "invalid_output_counts_as": "same rule as the rate it summarizes",
        "requires_attack": True,
        "is_h_star": False,
    },
    {
        "name": "gradient_alignment",
        "question": "What share of the target gradient's L1 norm lies along the sign pattern of the source gradient?",
        "denominator": "L1 norm of the target image gradient",
        "invalid_output_counts_as": "not an input",
        "requires_attack": False,
        "is_h_star": False,
    },
)


def metric_catalog() -> dict[str, Any]:
    """Name the estimands the study has to keep apart."""

    names = [item["name"] for item in KICKOFF_METRICS]
    return {
        "kickoff": list(KICKOFF_METRICS),
        "kickoff_names": names,
        "names_are_unique": len(names) == len(set(names)),
        "official_checkpoint_1_6": {
            "name": "strict_free_generation_accuracy",
            "parser": "checkpoint-1 strict parser on the color vocabulary",
            "kept_separate_from": [
                "valid_but_wrong",
                "invalid",
                "class_loss_argmin",
                PROSPECTIVE_PRIMARY,
            ],
        },
        "prospective_primary_score": {
            "name": PROSPECTIVE_PRIMARY,
            "definition": (
                "Teacher-forced conditional log-likelihood of the full candidate "
                "sequence, summed, with a fixed answer prefix. The same convention "
                "on every context. The attack loss, when an attack is authorized, "
                "uses this classification decision."
            ),
            "adopted_as_checkpoint_1_6_official": False,
            "reason_not_adopted_here": (
                "The row-color table was scored under the strict parser. "
                "The class-loss argmin is higher on some contexts. "
                "That gap is not a reason to change the official score."
            ),
        },
        "h_star": {
            "name": "beta_a",
            "definition": (
                "Linear probability coefficient on sign alignment, among scenes "
                "that are clean-correct under both primary contexts, outcome "
                "strict error of an attacked image, clusters are counterfactual groups."
            ),
            "equals_any_kickoff_metric": False,
            "fitted_in_this_package": False,
        },
        "positive_control": {
            "context": "answer_revealing",
            "role": "The text names the answer. A high score is the control working.",
            "counts_as_visual_robustness": False,
        },
    }


def partition_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Count the strict partition and the class-loss argmin on the same rows."""

    by_context: dict[str, dict[str, int]] = {}
    for row in rows:
        cell = by_context.setdefault(
            row["context"],
            {"n": 0, "correct": 0, "valid_wrong": 0, "invalid": 0, "class_argmin_matches_label": 0},
        )
        cell["n"] += 1
        cell["correct"] += int(bool(row["correct"]))
        cell["valid_wrong"] += int(bool(row["valid_wrong"]))
        cell["invalid"] += int(bool(row["invalid"]))
        cell["class_argmin_matches_label"] += int(row.get("class_argmin") == row.get("answer"))
    return {
        "by_context": by_context,
        "strict_and_class_argmin_are_different_columns": True,
    }
