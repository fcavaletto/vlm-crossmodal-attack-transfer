"""Development-pilot tables. No model, no attack, and no confirmatory fit.

Every block carries the label DEVELOPMENT ONLY. Invalid output and a
valid-but-wrong answer stay in separate rates. Both are task errors, and
neither rate is an attack success.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from .metrics import proportion
from .pilot_schema import PRIMARY_CONTEXTS
from .schemas import CONTEXTS, SchemaError

WEAK_ACCURACY_BELOW = 0.5
HIGH_INVALID_AT_OR_ABOVE = 0.25
UNSTABLE_REPEAT_MAX_ABS = 1e-4
LIBRARY_NLL_TOLERANCE = 1e-3
REVEALING_GAP_AT_OR_ABOVE = 0.25

DEFINITIONS = {
    "loss": "L = -log P(canonical answer | image, prompt), the sum of token negative log-likelihoods.",
    "mean_nll": "mean token NLL is sum_nll / token_count. It is stored beside the sum and is not the optimized loss.",
    "correct": "The parse is one vocabulary item and it matches the canonical answer or a documented alias.",
    "valid_wrong": "The parse is one recognized color or count and it does not match the answer or an alias.",
    "invalid": "The generation is not one task-vocabulary item. It is a task error and not an attack success.",
    "alignment": "A(s, t) = dot(g_t, sign(g_s)) / ||g_t||_1. A zero target L1 norm is null, not zero.",
    "flat_source": "A zero source L1 norm can make the algebra return 0. That row is excluded from the proposed H* population.",
    "counterfactual": "An answer change after a visual edit is not proof of mechanistic visual grounding.",
    "helpful_text": "The current helpful text is task guidance, not additional factual scene evidence.",
    "h_star": "H* was not fit. The cluster-robust coefficient is still required before any confirmatory evaluation.",
}


def build_pilot_summary(
    predictions: Sequence[Mapping[str, Any]],
    gradients: Sequence[Mapping[str, Any]],
    alignments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate validated rows. The caller has already checked the schema."""
    if not predictions:
        raise SchemaError("no pilot predictions")
    pairs = _pairs(predictions)
    summary = {
        "label": "DEVELOPMENT ONLY",
        "checkpoint": "checkpoint-1-development-pilot",
        "attacks_run": False,
        "h_star_fitted": False,
        "confirmatory_estimator": "not_implemented",
        "confirmatory_note": (
            "Agent A's regression estimand, with counterfactual-group clustering, "
            "is still required before any confirmatory evaluation."
        ),
        "definitions": DEFINITIONS,
        "n_scenes": len({row["scene_id"] for row in predictions}),
        "n_groups": len(pairs),
        "n_predictions": len(predictions),
        "clean_by_context": _clean_table(predictions, "context", CONTEXTS),
        "clean_by_task_family": _clean_table(
            predictions, "task_family", ("attribute", "counting", "spatial")
        ),
        "clean_by_family_and_context": _family_context_table(predictions),
        "counterfactual": _counterfactual(pairs),
        "gradients": _gradient_summary(gradients, alignments),
        "proposed_h_star_population": _population(predictions, gradients, alignments),
        "quality_flags": _flags(predictions, gradients, alignments, pairs),
    }
    return summary


def _pairs(predictions: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, Mapping[str, Any]]]]:
    grouped: dict[str, dict[str, dict[str, Mapping[str, Any]]]] = defaultdict(dict)
    for row in predictions:
        grouped[row["counterfactual_group_id"]].setdefault(row["scene_id"], {})[row["context"]] = row
    pairs = {}
    for group_id in sorted(grouped):
        scenes = grouped[group_id]
        if len(scenes) != 2:
            raise SchemaError(f"{group_id}: expected 2 scenes, found {len(scenes)}")
        left_id, right_id = sorted(scenes)
        for scene_id, contexts in scenes.items():
            if set(contexts) != set(CONTEXTS):
                raise SchemaError(f"{scene_id}: missing a context")
        pairs[group_id] = {left_id: scenes[left_id], right_id: scenes[right_id]}
    return pairs


def _clean_table(
    rows: Sequence[Mapping[str, Any]], key: str, order: Sequence[str]
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return {name: _clean_block(grouped.get(name, [])) for name in order}


def _family_context_table(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["task_family"], row["context"])].append(row)
    table = {}
    for family in ("attribute", "counting", "spatial"):
        table[family] = {
            context: _clean_block(grouped.get((family, context), [])) for context in CONTEXTS
        }
    return table


def _clean_block(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = [(row["scene_id"], row["context"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise SchemaError("duplicate scene and context in one clean cell")
    n_correct = sum(int(row["correct"]) for row in rows)
    n_valid_wrong = sum(int(row["valid_wrong"]) for row in rows)
    n_invalid = sum(int(row["invalid"]) for row in rows)
    n_rows = len(rows)
    n_unique_scenes = len({row["scene_id"] for row in rows})
    if n_correct + n_valid_wrong + n_invalid != n_rows:
        raise SchemaError("clean partition does not add up")
    return {
        "label": "DEVELOPMENT ONLY",
        "n_rows": n_rows,
        "n_unique_scenes": n_unique_scenes,
        "n_scenes": n_unique_scenes,
        "n_correct": n_correct,
        "n_valid_wrong": n_valid_wrong,
        "n_invalid": n_invalid,
        "accuracy": proportion(n_correct, n_rows, empty_reason="no scenes"),
        "valid_wrong_rate": proportion(n_valid_wrong, n_rows, empty_reason="no scenes"),
        "invalid_output_rate": proportion(n_invalid, n_rows, empty_reason="no scenes"),
        "denominator": "scene-context rows",
    }


def _counterfactual(pairs: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]]) -> dict[str, Any]:
    by_context = {}
    for context in CONTEXTS:
        both_parsed = 0
        changed = 0
        unchanged = 0
        both_correct = 0
        changed_to_truth = 0
        truths_differ = 0
        texts_identical = 0
        for group_id, scenes in pairs.items():
            left, right = _arms(scenes, context)
            identical = left["context_text"] == right["context_text"]
            texts_identical += int(identical)
            truths_differ += int(left["answer"] != right["answer"])
            if left["correct"] and right["correct"]:
                both_correct += 1
                if left["answer"] != right["answer"] and left["parsed_answer"] != right["parsed_answer"]:
                    changed_to_truth += 1
            if not left["invalid"] and not right["invalid"]:
                both_parsed += 1
                if left["parsed_answer"] == right["parsed_answer"]:
                    unchanged += 1
                else:
                    changed += 1
        n_pairs = len(pairs)
        n_scenes = n_pairs * 2
        relevant = n_pairs > 0 and texts_identical == n_pairs
        by_context[context] = {
            "label": "DEVELOPMENT ONLY",
            "n_pairs": n_pairs,
            "n_scenes": n_scenes,
            "pair_denominator": n_pairs,
            "scene_denominator": n_scenes,
            "texts_identical_pairs": texts_identical,
            "relevant_to_visual_edit": relevant,
            "true_answers_differ": proportion(
                truths_differ, n_pairs, empty_reason="no pairs"
            ),
            "both_parsed": proportion(both_parsed, n_pairs, empty_reason="no pairs"),
            "parsed_answer_changed": proportion(
                changed, both_parsed, empty_reason="no pair had two valid parses"
            ),
            "unchanged_despite_edit": proportion(
                unchanged, both_parsed, empty_reason="no pair had two valid parses"
            ),
            "correct_on_both_arms": proportion(both_correct, n_pairs, empty_reason="no pairs"),
            "changed_to_correct_counterfactual": proportion(
                changed_to_truth, n_pairs, empty_reason="no pairs"
            ),
            "scenes_in_pairs_whose_parsed_answers_changed": proportion(
                changed * 2, n_scenes, empty_reason="no scenes"
            ),
            "group_ids_unchanged": _unchanged_groups(pairs, context) if relevant else [],
        }
    return {
        "label": "DEVELOPMENT ONLY",
        "definition": DEFINITIONS["counterfactual"],
        "by_context": by_context,
    }


def _gradient_summary(
    gradients: Sequence[Mapping[str, Any]], alignments: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    by_context = {}
    for context in PRIMARY_CONTEXTS:
        rows = [row for row in gradients if row["context"] == context]
        l1 = [float(row["l1"]) for row in rows if row["l1"] is not None]
        nonzero = [float(row["nonzero_fraction"]) for row in rows]
        seconds = [float(row["seconds"]) for row in rows]
        repeats = [float(row["repeat_max_abs_diff"]) for row in rows if row["repeat_max_abs_diff"] is not None]
        by_context[context] = {
            "label": "DEVELOPMENT ONLY",
            "n": len(rows),
            "n_finite": sum(int(row["finite"]) for row in rows),
            "n_zero_l1": sum(int(float(row["l1"]) == 0.0) for row in rows),
            "l1_min": min(l1) if l1 else None,
            "l1_median": _median(l1) if l1 else None,
            "l1_max": max(l1) if l1 else None,
            "nonzero_fraction_min": min(nonzero) if nonzero else None,
            "seconds_sum": sum(seconds) if seconds else None,
            "seconds_max": max(seconds) if seconds else None,
            "n_repeated": len(repeats),
            "max_repeat_max_abs_diff": max(repeats) if repeats else None,
        }
    directions = {}
    for source, target, name in (
        ("neutral", "helpful_nonrevealing", "neutral_to_helpful"),
        ("helpful_nonrevealing", "neutral", "helpful_to_neutral"),
    ):
        rows = [
            row
            for row in alignments
            if row["source_context"] == source and row["target_context"] == target
        ]
        defined = [float(row["alignment_a"]) for row in rows if row["alignment_a"] is not None]
        directions[name] = {
            "label": "DEVELOPMENT ONLY",
            "source_context": source,
            "target_context": target,
            "n": len(rows),
            "n_defined": len(defined),
            "n_undefined": sum(int(row["alignment_a"] is None) for row in rows),
            "n_source_excluded": sum(int(row["source_gradient_excludes_h_star"]) for row in rows),
            "mean": (sum(defined) / len(defined)) if defined else None,
            "min": min(defined) if defined else None,
            "max": max(defined) if defined else None,
        }
    by_scene = []
    scenes = sorted({row["scene_id"] for row in alignments})
    for scene_id in scenes:
        scene_rows = [row for row in alignments if row["scene_id"] == scene_id]
        item = {"scene_id": scene_id, "counterfactual_group_id": scene_rows[0]["counterfactual_group_id"]}
        for row in scene_rows:
            key = f"{row['source_context']}__to__{row['target_context']}"
            item[key] = row["alignment_a"]
            item[f"{key}_excluded"] = row["source_gradient_excludes_h_star"]
        by_scene.append(item)
    return {
        "label": "DEVELOPMENT ONLY",
        "definition": DEFINITIONS["alignment"],
        "by_context": by_context,
        "alignment": directions,
        "by_scene": by_scene,
    }


def _population(
    predictions: Sequence[Mapping[str, Any]],
    gradients: Sequence[Mapping[str, Any]],
    alignments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """How many scenes meet the proposed inclusion rule. This is not a fit of H*."""
    correct = defaultdict(dict)
    for row in predictions:
        if row["context"] in PRIMARY_CONTEXTS:
            correct[row["scene_id"]][row["context"]] = row["correct"]
    usable = {}
    for row in gradients:
        usable.setdefault(row["scene_id"], {})[row["context"]] = bool(
            row["finite"]
            and row["l1"] is not None
            and float(row["l1"]) > 0.0
            and float(row["nonzero_fraction"]) > 0.0
        )
    excluded = defaultdict(bool)
    for row in alignments:
        if row["source_gradient_excludes_h_star"] or row["alignment_a"] is None:
            excluded[row["scene_id"]] = True
    included = []
    for scene_id in sorted(correct):
        flags = correct[scene_id]
        gradients_ok = usable.get(scene_id, {})
        if (
            flags.get("neutral") is True
            and flags.get("helpful_nonrevealing") is True
            and gradients_ok.get("neutral") is True
            and gradients_ok.get("helpful_nonrevealing") is True
            and not excluded[scene_id]
        ):
            included.append(scene_id)
    groups = {
        row["counterfactual_group_id"]
        for row in predictions
        if row["scene_id"] in set(included)
    }
    return {
        "label": "DEVELOPMENT ONLY",
        "coefficient_fit": False,
        "definition": DEFINITIONS["h_star"],
        "n_scenes_included": len(included),
        "n_groups_included": len(groups),
        "scene_ids": included,
    }


def _flags(
    predictions: Sequence[Mapping[str, Any]],
    gradients: Sequence[Mapping[str, Any]],
    alignments: Sequence[Mapping[str, Any]],
    pairs: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    flags.extend(_accuracy_flags(predictions))
    flags.extend(_gradient_flags(gradients, alignments))
    flags.extend(_pair_flags(pairs))
    flags.extend(_shortcut_flags(predictions))
    for row in predictions:
        if row["ambiguity_flags"]:
            flags.append(
                _flag(
                    "nonempty_ambiguity_flags",
                    f"{row['scene_id']} {row['context']}",
                    {"ambiguity_flags": list(row["ambiguity_flags"])},
                )
            )
        if row["scoring_generation_disagreement"]:
            flags.append(
                _flag(
                    "scoring_generation_disagreement",
                    f"{row['scene_id']} {row['context']}",
                    {
                        "parsed_answer": row["parsed_answer"],
                        "scored_argmin": row["scored_argmin"],
                    },
                )
            )
        if float(row["library_mean_abs_diff"]) > LIBRARY_NLL_TOLERANCE:
            flags.append(
                _flag(
                    "library_nll_mismatch",
                    f"{row['scene_id']} {row['context']}",
                    {"library_mean_abs_diff": row["library_mean_abs_diff"]},
                )
            )
    return sorted(flags, key=lambda item: (item["flag_id"], item["where"]))


def _accuracy_flags(predictions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flags = []
    tables = {
        "context": _clean_table(predictions, "context", CONTEXTS),
        "family": _clean_table(predictions, "task_family", ("attribute", "counting", "spatial")),
    }
    nested = _family_context_table(predictions)
    cells = [(f"context:{name}", block) for name, block in tables["context"].items()]
    cells.extend((f"family:{name}", block) for name, block in tables["family"].items())
    for family, contexts in nested.items():
        cells.extend((f"family:{family} context:{context}", block) for context, block in contexts.items())
    for where, block in cells:
        accuracy = block["accuracy"]["value"]
        invalid = block["invalid_output_rate"]["value"]
        if accuracy is not None and accuracy < WEAK_ACCURACY_BELOW:
            flags.append(
                _flag(
                    "weak_clean_accuracy",
                    where,
                    {"accuracy": accuracy, "n_scenes": block["n_scenes"], "threshold": WEAK_ACCURACY_BELOW},
                )
            )
        if invalid is not None and invalid >= HIGH_INVALID_AT_OR_ABOVE:
            flags.append(
                _flag(
                    "high_invalid_rate",
                    where,
                    {"invalid_output_rate": invalid, "n_scenes": block["n_scenes"], "threshold": HIGH_INVALID_AT_OR_ABOVE},
                )
            )
    return flags


def _gradient_flags(
    gradients: Sequence[Mapping[str, Any]], alignments: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    flags = []
    for row in gradients:
        where = f"{row['scene_id']} {row['context']}"
        if not row["finite"]:
            flags.append(_flag("nonfinite_gradient", where, {}))
        if row["finite"] and row["l1"] is not None and (
            float(row["l1"]) == 0.0 or float(row["nonzero_fraction"]) == 0.0
        ):
            flags.append(_flag("zero_gradient", where, {"l1": row["l1"]}))
        repeat = row["repeat_max_abs_diff"]
        if repeat is not None and float(repeat) > UNSTABLE_REPEAT_MAX_ABS:
            flags.append(_flag("unstable_gradient", where, {"repeat_max_abs_diff": repeat}))
    for row in alignments:
        where = f"{row['scene_id']} {row['source_context']}->{row['target_context']}"
        if row["alignment_a"] is None:
            flags.append(_flag("undefined_alignment", where, {"undefined_reason": row["undefined_reason"]}))
        elif row["source_l1_zero"]:
            flags.append(_flag("flat_source_gradient", where, {"alignment_a": row["alignment_a"]}))
    return flags


def _pair_flags(pairs: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]]) -> list[dict[str, Any]]:
    flags = []
    for group_id, scenes in pairs.items():
        for context in CONTEXTS:
            left, right = _arms(scenes, context)
            identical = left["context_text"] == right["context_text"]
            expect_identical = context != "answer_revealing"
            if identical != expect_identical:
                flags.append(
                    _flag(
                        "counterfactual_text_mismatch",
                        f"{group_id} {context}",
                        {"texts_identical": identical, "expected_identical": expect_identical},
                    )
                )
            if left["answer"] == right["answer"]:
                flags.append(_flag("counterfactual_same_answer", f"{group_id} {context}", {}))
            if (
                expect_identical
                and identical
                and not left["invalid"]
                and not right["invalid"]
                and left["parsed_answer"] == right["parsed_answer"]
            ):
                flags.append(
                    _flag(
                        "counterfactual_unchanged_answer",
                        f"{group_id} {context}",
                        {"parsed_answer": left["parsed_answer"]},
                    )
                )
    return flags


def _shortcut_flags(predictions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flags = []
    counts: dict[str, set[int]] = defaultdict(set)
    seen = set()
    for row in predictions:
        if row["scene_id"] in seen:
            continue
        seen.add(row["scene_id"])
        counts[row["task_family"]].add(int(row["n_objects"]))
    if counts and all(len(values) == 1 for values in counts.values()):
        flags.append(
            _flag(
                "generator_constant_object_count",
                "task_family",
                {family: sorted(values)[0] for family, values in sorted(counts.items())},
            )
        )
    for family in ("attribute", "counting", "spatial"):
        rows = [row for row in predictions if row["task_family"] == family]
        neutral = _clean_block([row for row in rows if row["context"] == "neutral"])
        revealing = _clean_block([row for row in rows if row["context"] == "answer_revealing"])
        if neutral["accuracy"]["value"] is None or revealing["accuracy"]["value"] is None:
            continue
        gap = revealing["accuracy"]["value"] - neutral["accuracy"]["value"]
        if gap >= REVEALING_GAP_AT_OR_ABOVE:
            flags.append(
                _flag(
                    "possible_revealing_shortcut",
                    f"family:{family}",
                    {"accuracy_gap": gap, "threshold": REVEALING_GAP_AT_OR_ABOVE},
                )
            )
    return flags


def _arms(scenes: Mapping[str, Mapping[str, Mapping[str, Any]]], context: str):
    left_id, right_id = sorted(scenes)
    return scenes[left_id][context], scenes[right_id][context]


def _unchanged_groups(pairs, context: str) -> list[str]:
    found = []
    for group_id, scenes in pairs.items():
        left, right = _arms(scenes, context)
        if (
            left["context_text"] == right["context_text"]
            and not left["invalid"]
            and not right["invalid"]
            and left["parsed_answer"] == right["parsed_answer"]
        ):
            found.append(group_id)
    return found


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2)


def _flag(flag_id: str, where: str, detail: dict[str, Any]) -> dict[str, Any]:
    return {
        "flag_id": flag_id,
        "where": where,
        "kept": True,
        "detail": detail,
        "label": "DEVELOPMENT ONLY",
    }
