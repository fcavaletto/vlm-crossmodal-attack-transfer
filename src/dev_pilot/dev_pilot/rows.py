"""JSON rows for one scene and context. No tensors and no weights."""

from __future__ import annotations

import hashlib
from typing import Mapping

from choice_eval.decision import choose_color, rank_colors
from choice_eval.diagnostic import diagnose_free_generation
from choice_eval.prompt import build_prompt
from choice_eval.protocol import PROTOCOL_ID

from .lock import ANSWER_PREFIX, EXPECTED_TOKEN_IDS, FORBIDDEN_ROW_KEYS, LABEL


def user_text_sha256(user_text: str) -> str:
    return hashlib.sha256(user_text.encode("utf-8")).hexdigest()


def assemble_row(
    *,
    model_key: str,
    model_id: str,
    revision: str,
    processor_class: str,
    image_processor_class: str,
    dtype: str,
    device: str,
    scene_id: str,
    counterfactual_group_id: str,
    context: str,
    answer: str,
    summed_nll: Mapping[str, float],
    mean_nll: Mapping[str, float],
    token_ids: Mapping[str, tuple[int, ...] | list[int]],
    continuations: Mapping[str, str],
    answer_prefix: str,
    generation_text: str,
    generation_truncated: bool,
    seconds: float,
    user_text: str,
) -> dict[str, object]:
    """One compact prediction. The decision is recomputed from the sums."""
    summed = {color: float(summed_nll[color]) for color in ("red", "blue", "green", "yellow")}
    mean = {color: float(mean_nll[color]) for color in ("red", "blue", "green", "yellow")}
    chosen = choose_color(summed)
    free = diagnose_free_generation(generation_text, truncated=generation_truncated, label=answer)
    ids = {color: [int(token) for token in token_ids[color]] for color in summed}
    row: dict[str, object] = {
        "label": LABEL,
        "protocol_id": PROTOCOL_ID,
        "model_key": model_key,
        "model_id": model_id,
        "revision": revision,
        "processor_class": processor_class,
        "image_processor_class": image_processor_class,
        "dtype": dtype,
        "device": device,
        "scene_id": scene_id,
        "counterfactual_group_id": counterfactual_group_id,
        "context": context,
        "answer": answer,
        "chosen": chosen,
        "closed_vocabulary_correct": chosen == answer,
        "summed_nll": summed,
        "mean_nll": mean,
        "token_ids": ids,
        "continuations": {color: continuations[color] for color in summed},
        "answer_prefix": answer_prefix,
        "rank_by_sum": list(rank_colors(summed)),
        "reductions_agree": choose_color(mean) == chosen,
        "scoring_path": "first_continuation_logit",
        "user_text_sha256": user_text_sha256(user_text),
        "free_generation": {
            "text": free["text"],
            "parsed_answer": free["parsed_answer"],
            "reasons": list(free["reasons"]),
            "correct": free["correct"],
            "valid_but_wrong": free["valid_but_wrong"],
            "invalid": free["invalid"],
            "truncated": generation_truncated,
        },
        "seconds": round(float(seconds), 6),
        "attacks_run": False,
        "h_star_fitted": False,
    }
    _reject_forbidden(row)
    return row


def protocol_errors(row: Mapping[str, object], record: Mapping[str, object]) -> list[str]:
    """Lock checks that do not depend on a model: decision, prompt, ids, label."""
    errors = decision_errors(row)
    context = str(row.get("context"))
    contexts = record.get("contexts")
    if not isinstance(contexts, Mapping) or context not in contexts:
        errors.append("context")
        return errors
    user_text = build_prompt(str(contexts[context]["text"]), str(record["question"])).user_text()
    if row.get("user_text_sha256") != user_text_sha256(user_text):
        errors.append("user_text")
    model_key = str(row.get("model_key"))
    if row.get("answer_prefix") != ANSWER_PREFIX.get(model_key):
        errors.append("answer_prefix")
    expected = EXPECTED_TOKEN_IDS.get(model_key)
    token_ids = row.get("token_ids")
    if expected is None or not isinstance(token_ids, Mapping):
        errors.append("token_ids")
    else:
        for color, ids in expected.items():
            stored = token_ids.get(color)
            if not isinstance(stored, list) or tuple(stored) != tuple(ids):
                errors.append(f"token_ids:{color}")
    if row.get("answer") != record.get("answer"):
        errors.append("answer")
    if row.get("scene_id") != record.get("scene_id"):
        errors.append("scene_id")
    if row.get("counterfactual_group_id") != record.get("counterfactual_group_id"):
        errors.append("group")
    if row.get("revision") != _revision(model_key):
        errors.append("revision")
    return errors


def _revision(model_key: str) -> str:
    from choice_eval.models import require_model

    return require_model(model_key).revision


def decision_errors(row: Mapping[str, object]) -> list[str]:
    """Return reasons a stored row disagrees with the locked decision rule."""
    errors: list[str] = []
    if row.get("label") != LABEL:
        errors.append("label")
    summed = row.get("summed_nll")
    if not isinstance(summed, Mapping):
        return errors + ["summed_nll"]
    try:
        chosen = choose_color(summed)
    except (TypeError, ValueError) as exc:
        return errors + [f"summed_nll:{exc}"]
    if row.get("chosen") != chosen:
        errors.append("chosen")
    if bool(row.get("closed_vocabulary_correct")) != (chosen == row.get("answer")):
        errors.append("closed_vocabulary_correct")
    free = row.get("free_generation")
    if not isinstance(free, Mapping):
        errors.append("free_generation")
    else:
        flags = (bool(free.get("correct")), bool(free.get("valid_but_wrong")), bool(free.get("invalid")))
        if sum(flags) != 1:
            errors.append("free_generation_partition")
    if row.get("attacks_run") is not False or row.get("h_star_fitted") is not False:
        errors.append("attack_or_hstar")
    found = _forbidden_keys(row)
    if found:
        errors.append("forbidden:" + ",".join(sorted(found)))
    return errors


def _reject_forbidden(row: Mapping[str, object]) -> None:
    found = _forbidden_keys(row)
    if found:
        raise RuntimeError(f"prediction row contains {sorted(found)}")


def _forbidden_keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in FORBIDDEN_ROW_KEYS:
                found.add(str(key))
            found.update(_forbidden_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_forbidden_keys(item))
    return found
