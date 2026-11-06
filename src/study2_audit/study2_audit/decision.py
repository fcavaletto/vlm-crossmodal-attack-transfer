"""Go or no-go for a 96-scene development sample as support for H*.

Support would be a confirmatory cluster interval on real attack rows. This
function does not fit that interval. A scene count can still be a development
sample. The decision below is about the claim, not about whether the scenes
may be built.
"""

from __future__ import annotations

from typing import Any, Mapping

PROPOSED_SCENES = 96
SCENES_PER_GROUP = 2
PROPOSED_GROUPS = PROPOSED_SCENES // SCENES_PER_GROUP
PUBLISHED_COVERAGE_GROUPS = 24


def decide_scene_count(
    eligibility: Mapping[str, Any],
    precision: Mapping[str, Any],
    *,
    n_scenes: int = PROPOSED_SCENES,
) -> dict[str, Any]:
    """Decide whether ``n_scenes`` could support H*.

    The answer stays no-go while attacks are absent, gradients are missing,
    or the precision description still refuses to treat 24 groups as enough.
    A flattering exclusion rate in the fabricated grid does not flip the
    decision, because that grid is not a development packet.
    """

    if precision.get("h_star_fitted_on_real_data") is True or eligibility.get("h_star_fitted") is True:
        raise RuntimeError("H* was marked fitted. This checkpoint does not fit it.")
    if precision.get("data_origin") != "fabricated_dgp":
        raise RuntimeError("the precision argument is not the fabricated design")
    if n_scenes != PROPOSED_SCENES:
        raise ValueError("this decision is about the proposed 96-scene development sample")
    clean_groups = int(eligibility["n_groups_clean_both_primary"])
    development_groups = int(eligibility["n_groups"])
    eligible = eligibility["n_groups_eligible"]
    attacks_observed = False
    product = PROPOSED_GROUPS * clean_groups / development_groups
    row_9 = _size(precision, clean_groups)
    row_24 = _size(precision, PUBLISHED_COVERAGE_GROUPS)
    row_48 = _size(precision, PROPOSED_GROUPS)
    reasons = [
        (
            "H* needs strict error on an attacked image and a sign alignment. "
            "No attack was run, and alignments were not measured on the clean-both scenes. "
            "Ninety-six clean scenes do not contain that regression."
        ),
        (
            f"The development table has {clean_groups} groups that are strict-correct under both "
            f"primary contexts, out of {development_groups}. The eligible-group count is "
            f"{eligible}. Gradients exist for one scene, and that scene is outside the clean-both set."
        ),
        (
            f"{clean_groups} groups sit above the algebraic floors and below the published "
            f"coverage description of {PUBLISHED_COVERAGE_GROUPS}. "
            "An interval that can be returned is not an interval that lies above zero. "
            "Nine groups do not establish H*."
        ),
        (
            f"{PROPOSED_SCENES} scenes are {PROPOSED_GROUPS} counterfactual groups when each group "
            f"has {SCENES_PER_GROUP} arms. The arithmetic product of {PROPOSED_GROUPS} and the "
            f"development clean-both share {clean_groups}/{development_groups} is {product:g}. "
            "That product is not a forecast, and it is not an eligible-group count."
        ),
        (
            "Twenty-four groups is the published coverage description on a different fabricated "
            "design. It does not promise an interval excluding zero. This plan does not treat "
            "24 groups as automatically enough."
        ),
        _precision_sentence(row_9, row_24, row_48),
        (
            "Multiplicity, exclusions, missing gradients, invalid generations, and ceilings "
            "remain open. A later fit that picks a metric after seeing those tables would "
            "not be the pre-specified coefficient."
        ),
    ]
    return {
        "decision": "no-go",
        "question": "Could 96 development scenes support H*?",
        "supports_h_star": False,
        "blocks_h_star_claim": True,
        "blocks_scene_construction": False,
        "n_scenes": PROPOSED_SCENES,
        "n_groups_if_every_scene_is_kept": PROPOSED_GROUPS,
        "scenes_per_group": SCENES_PER_GROUP,
        "attacks_observed": attacks_observed,
        "n_groups_eligible": eligible,
        "hypothetical_clean_both_product": product,
        "hypothetical_product_is_a_forecast": False,
        "twenty_four_groups_automatically_enough": False,
        "nine_groups_establish_h_star": False,
        "h_star_fitted": False,
        "model_inference": False,
        "precision_at_9": _compact(row_9),
        "precision_at_24": _compact(row_24),
        "precision_at_48": _compact(row_48),
        "reasons": reasons,
    }


def _size(precision: Mapping[str, Any], n_groups: int) -> Mapping[str, Any] | None:
    for row in precision.get("by_n_groups", []):
        if int(row["n_groups"]) == n_groups:
            return row
    return None


def _compact(row: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "n_groups": row["n_groups"],
        "n_scenes": row["n_scenes"],
        "fraction_interval_returned": row["fraction_interval_returned"],
        "fraction_covered_among_returned": row["fraction_covered_among_returned"],
        "fraction_excludes_zero_among_returned": row["fraction_excludes_zero_among_returned"],
        "median_width": row["median_width"],
        "median_width_row": row["median_width_row"],
    }


def _precision_sentence(
    row_9: Mapping[str, Any] | None,
    row_24: Mapping[str, Any] | None,
    row_48: Mapping[str, Any] | None,
) -> str:
    def phrase(row: Mapping[str, Any] | None, label: str) -> str:
        if row is None:
            return f"{label} was not in this precision grid"
        share = row["fraction_excludes_zero_among_returned"]
        share_text = "undefined" if share is None else f"{share:.3f}"
        width = row["median_width"]
        width_text = "undefined" if width is None else f"{width:.3f}"
        return (
            f"at {row['n_groups']} groups the share of returned intervals with lower "
            f"endpoint above zero is {share_text} and the median width is {width_text}"
        )

    return (
        "On the fabricated one-family design with known coefficient 0.25, "
        + "; ".join((phrase(row_9, "9 groups"), phrase(row_24, "24 groups"), phrase(row_48, "48 groups")))
        + ". The row-level interval, which pretends each row is its own cluster, is a different width. "
        "Neither width is a measurement of SmolVLM2 or Qwen."
    )
