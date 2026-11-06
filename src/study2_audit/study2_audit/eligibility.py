"""Clean-both groups and eligible groups are different counts.

An eligible group has at least one scene that is strict-correct under both
primary contexts and whose primary-context image gradients are finite and
nonzero, with the leakage and ambiguity gates clear. Checkpoint 1.6 measured
the clean-both half on 24 row-color scenes. It measured gradients on one
scene. The eligible-group count stays unset.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

PRIMARY = ("neutral", "helpful_nonrevealing")
ONE_FAMILY_PARAMETERS = 2
THREE_FAMILY_PARAMETERS = 4
COVERAGE_DESCRIPTION_GROUPS = 24
GRADIENT_SCENE_ID = "dev3-row-00-a"


def minimum_groups(n_parameters: int) -> int:
    """Smallest group count that can return a k-parameter cluster interval."""

    if isinstance(n_parameters, bool) or not isinstance(n_parameters, int) or n_parameters < 1:
        raise ValueError("n_parameters must be an integer >= 1")
    return n_parameters + 1


def explain_nine(
    predictions: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    """Say what the nine groups are, and why they do not establish H*."""

    clean_ids, membership = _clean_both(predictions)
    clean_groups = sorted(membership)
    sizes = {group_id: len(scene_ids) for group_id, scene_ids in membership.items()}
    gradient_scene_clean = GRADIENT_SCENE_ID in clean_ids
    full_gradient_pass = False
    n_gradient_scenes = 1 if gate.get("scene_id") == GRADIENT_SCENE_ID and gate.get("passed") is True else 0
    algebraic_one_family = minimum_groups(ONE_FAMILY_PARAMETERS)
    algebraic_three_family = minimum_groups(THREE_FAMILY_PARAMETERS)
    return {
        "sampling_unit": "counterfactual_group",
        "n_scenes": len({row["scene_id"] for row in predictions}),
        "n_groups": len({row["counterfactual_group_id"] for row in predictions}),
        "n_scenes_clean_both_primary": len(clean_ids),
        "n_groups_clean_both_primary": len(clean_groups),
        "group_ids_clean_both_primary": clean_groups,
        "scene_ids_clean_both_primary": clean_ids,
        "scenes_per_clean_group": sizes,
        "n_clean_groups_with_both_scenes": sum(size == 2 for size in sizes.values()),
        "n_clean_groups_with_one_scene": sum(size == 1 for size in sizes.values()),
        "full_gradient_pass": full_gradient_pass,
        "n_scenes_with_gradients": n_gradient_scenes,
        "gradient_scene_id": gate.get("scene_id"),
        "gradient_scene_is_clean_both_primary": gradient_scene_clean,
        "n_groups_eligible": None,
        "eligible_count_is_the_clean_both_count": False,
        "algebraic_minimum_groups_one_family": algebraic_one_family,
        "algebraic_minimum_groups_three_family": algebraic_three_family,
        "coverage_description_groups": COVERAGE_DESCRIPTION_GROUPS,
        "nine_is_above_both_algebraic_floors": len(clean_groups) >= algebraic_three_family,
        "nine_meets_coverage_description": len(clean_groups) >= COVERAGE_DESCRIPTION_GROUPS,
        "twenty_four_groups_automatically_enough": False,
        "establishes_h_star": False,
        "h_star_fitted": False,
        "reasons": _reasons(
            n_groups=len(clean_groups),
            n_scenes=len(clean_ids),
            sizes=sizes,
            gradient_scene_clean=gradient_scene_clean,
            algebraic_one_family=algebraic_one_family,
            algebraic_three_family=algebraic_three_family,
        ),
    }


def _clean_both(
    predictions: Sequence[Mapping[str, Any]],
) -> tuple[list[str], dict[str, list[str]]]:
    scenes: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in predictions:
        if row["context"] in PRIMARY:
            scenes.setdefault(row["scene_id"], {})[row["context"]] = row
    membership: dict[str, list[str]] = defaultdict(list)
    clean_ids = []
    for scene_id in sorted(scenes):
        pair = scenes[scene_id]
        if set(pair) != set(PRIMARY):
            continue
        if all(bool(pair[context]["correct"]) for context in PRIMARY):
            clean_ids.append(scene_id)
            membership[pair["neutral"]["counterfactual_group_id"]].append(scene_id)
    return clean_ids, dict(membership)


def _reasons(
    *,
    n_groups: int,
    n_scenes: int,
    sizes: Mapping[str, int],
    gradient_scene_clean: bool,
    algebraic_one_family: int,
    algebraic_three_family: int,
) -> list[str]:
    both = sum(size == 2 for size in sizes.values())
    one = sum(size == 1 for size in sizes.values())
    return [
        (
            f"{n_scenes} scenes sit in {n_groups} counterfactual groups "
            f"({both} groups contribute two scenes and {one} groups contribute one). "
            "The sampling unit is the group. The scene count is not an independent sample size."
        ),
        (
            "These groups are strict-correct under both neutral and helpful_nonrevealing. "
            "The eligible-group count also requires finite nonzero image gradients on those contexts. "
            "Gradients were measured on dev3-row-00-a only. "
            f"That scene is clean under both primary contexts: {gradient_scene_clean}. "
            "The eligible-group count is unset."
        ),
        (
            "H* regresses strict error of an attacked image on sign alignment. "
            "No attack was run, so the outcome is unobserved. "
            "Alignment is unobserved on the clean-both scenes because their gradients were not measured."
        ),
        (
            f"A one-family design has {algebraic_one_family} as the smallest group count that can return "
            f"a cluster interval. A three-family design has {algebraic_three_family}. "
            f"{n_groups} is above both floors, so an interval could be returned if the other gates passed. "
            "Returning an interval is not the same event as the interval lying above zero."
        ),
        (
            f"{n_groups} is below the fabricated three-family coverage description of "
            f"{COVERAGE_DESCRIPTION_GROUPS} groups. That description does not promise an interval "
            "excluding zero, and 24 groups is not treated here as automatically enough."
        ),
    ]
