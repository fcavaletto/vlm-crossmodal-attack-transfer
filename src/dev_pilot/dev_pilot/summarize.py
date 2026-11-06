"""Counts fixed by the development protocol. No model and no H* fit."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence

from .lock import (
    CEILING_DENOMINATOR,
    CEILING_NUMERATOR,
    COMPETENCE_MIN_DENOMINATOR,
    COMPETENCE_MIN_NUMERATOR,
    LABEL,
    N_SCENES,
    PRIMARY_CONTEXTS,
)


def at_least_65(correct: int, n: int) -> bool:
    """``correct / n >= 0.65`` with integer arithmetic. 0.65 is 13/20."""
    if correct < 0 or n < 1:
        raise ValueError("counts must be a non-negative correct over a positive n")
    return correct * COMPETENCE_MIN_DENOMINATOR >= n * COMPETENCE_MIN_NUMERATOR


def above_80(correct: int, n: int) -> bool:
    """``correct / n > 0.80``. 0.80 is 4/5, and the inequality is strict."""
    if correct < 0 or n < 1:
        raise ValueError("counts must be a non-negative correct over a positive n")
    return correct * CEILING_DENOMINATOR > n * CEILING_NUMERATOR


def context_counts(rows: Sequence[Mapping[str, Any]], context: str) -> dict[str, Any]:
    subset = [row for row in rows if row["context"] == context]
    if len(subset) != N_SCENES:
        raise ValueError(f"{context} has {len(subset)} rows, expected {N_SCENES}")
    closed = sum(int(bool(row["closed_vocabulary_correct"])) for row in subset)
    correct = valid_but_wrong = invalid = 0
    for row in subset:
        free = row["free_generation"]
        correct += int(bool(free["correct"]))
        valid_but_wrong += int(bool(free["valid_but_wrong"]))
        invalid += int(bool(free["invalid"]))
    if correct + valid_but_wrong + invalid != len(subset):
        raise RuntimeError(f"{context} free-generation flags are not a partition")
    return {
        "label": LABEL,
        "context": context,
        "n": len(subset),
        "closed_vocabulary_correct": closed,
        "closed_vocabulary_accuracy": closed / len(subset),
        "free_generation_correct": correct,
        "free_generation_valid_but_wrong": valid_but_wrong,
        "free_generation_invalid": invalid,
        "free_generation_strict_accuracy": correct / len(subset),
        "free_generation_invalid_rate": invalid / len(subset),
        "metrics_combined": False,
        "pairs": pair_counts(subset),
    }


def pair_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Closed-vocabulary pair partition for one context. Rows are already filtered."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["counterfactual_group_id"]].append(row)
    if any(len(arms) != 2 for arms in grouped.values()):
        raise ValueError("a counterfactual group does not have two arms in this context")
    both = one = neither = differ = same = 0
    strict_both = strict_one = strict_neither = 0
    for arms in grouped.values():
        labels = {arm["answer"] for arm in arms}
        if len(labels) != 2:
            raise ValueError("counterfactual arms do not have different labels")
        closed_flags = [bool(arm["closed_vocabulary_correct"]) for arm in arms]
        closed_hits = sum(closed_flags)
        if closed_hits == 2:
            both += 1
        elif closed_hits == 1:
            one += 1
        else:
            neither += 1
        choices = [arm["chosen"] for arm in arms]
        if choices[0] == choices[1]:
            same += 1
        else:
            differ += 1
        strict_hits = sum(bool(arm["free_generation"]["correct"]) for arm in arms)
        if strict_hits == 2:
            strict_both += 1
        elif strict_hits == 1:
            strict_one += 1
        else:
            strict_neither += 1
    n_groups = len(grouped)
    if both + one + neither != n_groups or differ + same != n_groups:
        raise RuntimeError("pair partition does not cover the groups")
    if strict_both + strict_one + strict_neither != n_groups:
        raise RuntimeError("strict pair partition does not cover the groups")
    return {
        "label": LABEL,
        "n_groups": n_groups,
        "pair_consistency": both,
        "both_correct": both,
        "one_correct": one,
        "neither_correct": neither,
        "decisions_differ": differ,
        "same_decision": same,
        "strict_both_correct": strict_both,
        "strict_one_correct": strict_one,
        "strict_neither_correct": strict_neither,
        "metrics_combined": False,
    }


def clean_both(
    rows: Sequence[Mapping[str, Any]],
    predicate: Callable[[Mapping[str, Any]], bool],
) -> dict[str, Any]:
    """Scenes that pass ``predicate`` on both primary contexts, and their groups."""
    by_scene: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        if row["context"] in PRIMARY_CONTEXTS:
            by_scene.setdefault(row["scene_id"], {})[row["context"]] = row
    scene_ids: list[str] = []
    membership: dict[str, list[str]] = defaultdict(list)
    for scene_id in sorted(by_scene):
        pair = by_scene[scene_id]
        if set(pair) != set(PRIMARY_CONTEXTS):
            continue
        if all(predicate(pair[context]) for context in PRIMARY_CONTEXTS):
            scene_ids.append(scene_id)
            membership[pair["neutral"]["counterfactual_group_id"]].append(scene_id)
    both_scenes = sum(len(scenes) == 2 for scenes in membership.values())
    return {
        "label": LABEL,
        "n_scenes": len(scene_ids),
        "scene_ids": scene_ids,
        "n_groups_at_least_one_scene": len(membership),
        "n_groups_both_scenes": both_scenes,
        "group_ids": sorted(membership),
    }


def competence_gate(contexts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Primary-context closed-vocabulary gate. The eligible count is not an input."""
    rates = {}
    passed = True
    ceiling: list[str] = []
    for context in PRIMARY_CONTEXTS:
        correct = int(contexts[context]["closed_vocabulary_correct"])
        n = int(contexts[context]["n"])
        ok = at_least_65(correct, n)
        high = above_80(correct, n)
        rates[context] = {
            "correct": correct,
            "n": n,
            "at_least_65": ok,
            "above_80": high,
        }
        passed = passed and ok
        if high:
            ceiling.append(context)
    return {
        "label": LABEL,
        "passed": passed,
        "contexts": rates,
        "contexts_above_80": ceiling,
        "eligible_count_is_a_threshold": False,
        "rule_changed_after_results": False,
        "scenes_deleted": False,
    }


def eligible_groups(
    records: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    gradients: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Eligible groups, or unset when the v4 gradient pass was not run."""
    if gradients is None:
        return {
            "label": LABEL,
            "n_groups_eligible": None,
            "eligible_group_ids": None,
            "reason": "development_v4 image gradients were not measured",
            "gradients_measured": False,
            "h_star_fitted": False,
        }
    by_scene = _primary_index(rows)
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_group[record["counterfactual_group_id"]].append(record)
    eligible: list[str] = []
    for group_id, members in sorted(by_group.items()):
        if not _group_invariant(members):
            continue
        if any(_scene_eligible(record, by_scene.get(record["scene_id"], {}), gradients) for record in members):
            eligible.append(group_id)
    return {
        "label": LABEL,
        "n_groups_eligible": len(eligible),
        "eligible_group_ids": eligible,
        "reason": "at least one scene passed the pre-attack gates",
        "gradients_measured": True,
        "h_star_fitted": False,
    }


def summarize_model(
    rows: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    model_key: str,
    gradients: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Full development summary. ``gradients=None`` leaves the eligible count unset."""
    present = sorted({row["context"] for row in rows})
    contexts = {context: context_counts(rows, context) for context in present if _complete(rows, context)}
    for context in PRIMARY_CONTEXTS:
        if context not in contexts:
            raise ValueError(f"missing primary context {context}")
    gate = competence_gate(contexts)
    return {
        "label": LABEL,
        "model_key": model_key,
        "n_rows": len(rows),
        "contexts": contexts,
        "closed_clean_both": clean_both(rows, lambda row: bool(row["closed_vocabulary_correct"])),
        "strict_clean_both": clean_both(rows, lambda row: bool(row["free_generation"]["correct"])),
        "eligible": eligible_groups(records, rows, gradients),
        "competence_gate": gate,
        "metrics_combined": False,
        "attacks_run": False,
        "h_star_fitted": False,
        "scenes_deleted": False,
    }


def _complete(rows: Sequence[Mapping[str, Any]], context: str) -> bool:
    return sum(row["context"] == context for row in rows) == N_SCENES


def _primary_index(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Mapping[str, Any]]]:
    indexed: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        if row["context"] in PRIMARY_CONTEXTS:
            indexed.setdefault(row["scene_id"], {})[row["context"]] = row
    return indexed


def _group_invariant(members: Sequence[Mapping[str, Any]]) -> bool:
    if len(members) != 2:
        return False
    questions = {member["question"] for member in members}
    if len(questions) != 1:
        return False
    for context in PRIMARY_CONTEXTS:
        texts = {member["contexts"][context]["text"] for member in members}
        reveals = {bool(member["contexts"][context]["reveals_answer"]) for member in members}
        if len(texts) != 1 or reveals != {False}:
            return False
    return True


def _scene_eligible(
    record: Mapping[str, Any],
    pair: Mapping[str, Mapping[str, Any]],
    gradients: Mapping[str, Mapping[str, Any]],
) -> bool:
    if set(pair) != set(PRIMARY_CONTEXTS):
        return False
    if not all(bool(pair[context]["free_generation"]["correct"]) for context in PRIMARY_CONTEXTS):
        return False
    if record.get("ambiguity_flags"):
        return False
    answer = str(record["answer"])
    if _contains_word(str(record["question"]), answer):
        return False
    for context in PRIMARY_CONTEXTS:
        text = str(record["contexts"][context]["text"])
        if bool(record["contexts"][context]["reveals_answer"]) or _contains_word(text, answer):
            return False
    scene_gradients = gradients.get(str(record["scene_id"]))
    if not isinstance(scene_gradients, Mapping):
        return False
    alignment = scene_gradients.get("alignment")
    if isinstance(alignment, bool) or not isinstance(alignment, (int, float)):
        return False
    if not (-1 <= float(alignment) <= 1):
        return False
    for context in PRIMARY_CONTEXTS:
        stats = scene_gradients.get(context)
        if not isinstance(stats, Mapping) or stats.get("finite") is not True:
            return False
        l1 = stats.get("l1")
        if isinstance(l1, bool) or not isinstance(l1, (int, float)) or not (float(l1) > 0):
            return False
    return True


def _contains_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word.casefold())}\b", text.casefold()) is not None
