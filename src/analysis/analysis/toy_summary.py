"""Build the committed toy summary from the compact fixtures.

The fixtures are hand-written. Rebuilding this summary does not load a model
and does not run an attack. Role labels are proposals, not a locked estimand.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .bootstrap import paired_mean_difference, scene_bootstrap_mean, scene_bootstrap_rate
from .metrics import (
    ANALYSIS_DESCRIPTIVE,
    ANALYSIS_EXPLORATORY,
    ANALYSIS_PRIMARY,
    ANALYSIS_SECONDARY,
    additional_error_by_scene,
    attack_cells,
    attack_success_indicators,
    clean_by_context,
    conditional_transfer,
    mean_alignment,
    mean_loss_by_context,
    pearson_correlation,
    sign_alignment,
)
from .schemas import (
    CONTEXTS,
    SchemaError,
    epsilon_key,
    load_jsonl,
    validate_attacks,
    validate_gradient_summaries,
    validate_gradient_vectors,
    validate_losses,
    validate_predictions,
)

TOY_SEED = 20261106
TOY_N_BOOT = 2000
TOY_ALPHA = 0.05
ALIGNMENT_TOLERANCE = 1e-12

FIXTURE_FILES = (
    "predictions.jsonl",
    "losses.jsonl",
    "gradient_summaries.jsonl",
    "gradient_vectors.jsonl",
    "attacks.jsonl",
)
SUMMARY_FILENAME = "toy_summary.json"

TOY_INTERPRETATION = (
    "Fabricated arithmetic check on hand-written rows. "
    "Not evidence about vision-language models."
)

# Illustration only. Agent A has not locked the primary contrast.
ILLUSTRATION = {
    "method": "pgd",
    "epsilon": 0.01,
    "step_budget": 1,
    "source_context": "neutral",
    "left_target": "contradictory",
    "right_target": "neutral",
}
ASSOCIATION_CELL = {
    "method": "pgd",
    "epsilon": 0.01,
    "step_budget": 1,
    "source_context": "neutral",
    "target_context": "helpful_nonrevealing",
}
TRANSFER_CELL = {
    "method": "pgd",
    "epsilon": 0.01,
    "step_budget": 1,
    "source_context": "helpful_nonrevealing",
    "target_context": "contradictory",
}


def fixture_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "toy"


def build_toy_summary(directory: Path | None = None) -> dict[str, Any]:
    root = Path(directory) if directory is not None else fixture_dir()
    loaded = _load_packet(root)
    _audit_packet(loaded)
    clean = clean_by_context(loaded["predictions"])
    losses = mean_loss_by_context(loaded["losses"])
    cells = attack_cells(loaded["attacks"])
    alignment = mean_alignment(loaded["gradient_summaries"])
    summary = {
        "schema_version": "checkpoint0-toy-1",
        "data_origin": "fabricated_toy_fixtures",
        "not_an_empirical_finding": True,
        "labels_are_proposals": True,
        "primary_estimand_locked": False,
        "verification_command": "python -m analysis.verify_toy",
        "verification_idea": (
            "Rebuild this file from the compact JSONL fixtures. "
            "Study 1 published numbers are not inputs."
        ),
        "scoring": _scoring_notes(),
        "bootstrap_settings": {
            "scope": "toy_verification_only",
            "unit": "scene_id",
            "n_boot": TOY_N_BOOT,
            "seed": TOY_SEED,
            "alpha": TOY_ALPHA,
            "quantile_method": "linear",
            "interval": "equal_tailed_percentile",
            "note": (
                "This seed and replicate count make the toy interval rebuildable. "
                "They are not the confirmatory seed."
            ),
        },
        "grouping": {
            "point_estimate_unit": "scene_id",
            "bootstrap_unit": "scene_id",
            "counterfactual_cluster": "not_applied",
            "note": (
                "Pass cluster_of to the bootstrap to resample counterfactual groups. "
                "These fixtures have no group ids, so the toy intervals do not."
            ),
        },
        "role_catalog": {
            "proposed_primary": (
                "RQ1 paired difference in additional error. Illustration only; not locked."
            ),
            "proposed_secondary": (
                "Clean accuracy, invalid-output rate, attack success, transfer cells, "
                "alignment, noise and FGSM controls."
            ),
            "exploratory": (
                "Conditional transfer given a same-context success. "
                "Representation interventions are exploratory and not implemented."
            ),
            "positive_control": "answer_revealing. Not evidence of visual robustness.",
            "descriptive_not_an_endpoint": "Mean clean loss by context.",
        },
        "fixture_sha256": {
            name: _sha256(root / name) for name in FIXTURE_FILES
        },
        "n_scenes": len({row["scene_id"] for row in loaded["predictions"]}),
        "scene_ids": sorted({row["scene_id"] for row in loaded["predictions"]}),
        "n_prediction_rows": len(loaded["predictions"]),
        "n_loss_rows": len(loaded["losses"]),
        "n_gradient_rows": len(loaded["gradient_summaries"]),
        "n_attack_rows": len(loaded["attacks"]),
        "clean": {
            "analysis_role": ANALYSIS_SECONDARY,
            "by_context": {
                context: _public_clean(clean[context]) for context in CONTEXTS
            },
        },
        "losses": {
            "analysis_role": ANALYSIS_DESCRIPTIVE,
            "by_context": losses,
        },
        "attack_cells": [_public_attack_cell(cell, loaded["attacks"]) for cell in cells],
        "proposed_primary_illustration": _primary_illustration(loaded["attacks"]),
        "alignment": _public_alignment(alignment),
        "conditional_transfer": _public_conditional(loaded["attacks"]),
        "alignment_transfer_association": _association(
            loaded["attacks"], loaded["gradient_summaries"]
        ),
        "precision_simulation": {
            "analysis_role": ANALYSIS_EXPLORATORY,
            "included_in_this_file": False,
            "command": "python -m analysis.simulation",
            "reason": (
                "The simulation draws a fabricated distribution. "
                "It is not rebuilt from the compact fixtures."
            ),
        },
    }
    return _json_safe(summary)


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _load_packet(root: Path) -> dict[str, list[dict[str, Any]]]:
    missing = [name for name in FIXTURE_FILES if not (root / name).is_file()]
    if missing:
        raise SchemaError(f"missing toy fixtures: {missing}")
    return {
        "predictions": validate_predictions(load_jsonl(root / "predictions.jsonl")),
        "losses": validate_losses(load_jsonl(root / "losses.jsonl")),
        "gradient_summaries": validate_gradient_summaries(
            load_jsonl(root / "gradient_summaries.jsonl")
        ),
        "gradient_vectors": validate_gradient_vectors(load_jsonl(root / "gradient_vectors.jsonl")),
        "attacks": validate_attacks(load_jsonl(root / "attacks.jsonl")),
    }


def _audit_packet(loaded: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    problems: list[str] = []
    prediction_keys = {(row["scene_id"], row["context"]) for row in loaded["predictions"]}
    loss_keys = {(row["scene_id"], row["context"]) for row in loaded["losses"]}
    if prediction_keys != loss_keys:
        problems.append("prediction and loss scene-context keys differ")
    scenes = sorted({row["scene_id"] for row in loaded["predictions"]})
    for scene_id in scenes:
        for context in CONTEXTS:
            if (scene_id, context) not in prediction_keys:
                problems.append(f"{scene_id} is missing clean context {context}")
    correct = {(row["scene_id"], row["context"]): row["correct"] for row in loaded["predictions"]}
    for row in loaded["attacks"]:
        key = (row["scene_id"], row["target_context"])
        if key not in correct:
            problems.append(
                f"attack scene {row['scene_id']} has no clean prediction for {row['target_context']}"
            )
        elif row["correct_before"] != correct[key]:
            problems.append(
                "correct_before for "
                f"{row['scene_id']} / {row['target_context']} does not match the clean prediction"
            )
    problems.extend(
        _alignment_mismatches(loaded["gradient_summaries"], loaded["gradient_vectors"])
    )
    if problems:
        raise SchemaError("; ".join(problems))


def _alignment_mismatches(
    summaries: Sequence[Mapping[str, Any]],
    vectors: Sequence[Mapping[str, Any]],
) -> list[str]:
    summary_keys = {
        (row["scene_id"], row["source_context"], row["target_context"]): row for row in summaries
    }
    vector_keys = {
        (row["scene_id"], row["source_context"], row["target_context"]): row for row in vectors
    }
    if set(summary_keys) != set(vector_keys):
        return ["gradient summary keys and gradient vector keys differ"]
    problems = []
    for key, summary in summary_keys.items():
        vector = vector_keys[key]
        recomputed = sign_alignment(vector["g_source"], vector["g_target"])["alignment_a"]
        stored = summary["alignment_a"]
        if recomputed is None or stored is None:
            if recomputed is not None or stored is not None:
                problems.append(f"alignment null mismatch at {key!r}")
            continue
        if abs(float(recomputed) - float(stored)) > ALIGNMENT_TOLERANCE:
            problems.append(
                f"alignment mismatch at {key!r}: stored {stored} recomputed {recomputed}"
            )
    return problems


def _public_clean(block: Mapping[str, Any]) -> dict[str, Any]:
    public = {
        key: block[key]
        for key in (
            "context",
            "analysis_role",
            "positive_control_only",
            "scoring",
            "n_scenes",
            "n_correct",
            "n_invalid",
            "accuracy",
            "invalid_output_rate",
            "clean_error_rate",
            "floor_warning",
        )
    }
    if block["n_scenes"] == 0:
        public["accuracy_bootstrap"] = None
        public["invalid_output_bootstrap"] = None
        return public
    public["accuracy_bootstrap"] = scene_bootstrap_mean(
        block["scene_correct"],
        n_boot=TOY_N_BOOT,
        seed=TOY_SEED,
        alpha=TOY_ALPHA,
    )
    public["invalid_output_bootstrap"] = scene_bootstrap_mean(
        block["scene_invalid"],
        n_boot=TOY_N_BOOT,
        seed=TOY_SEED,
        alpha=TOY_ALPHA,
    )
    return public


def _public_attack_cell(
    cell: Mapping[str, Any], attacks: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    rows = _rows_for_cell(attacks, cell)
    values = additional_error_by_scene(rows)
    indicators = attack_success_indicators(rows)
    public = dict(cell)
    if values:
        public["additional_error_bootstrap"] = scene_bootstrap_mean(
            values, n_boot=TOY_N_BOOT, seed=TOY_SEED, alpha=TOY_ALPHA
        )
    else:
        public["additional_error_bootstrap"] = None
    public["attack_success_bootstrap"] = scene_bootstrap_rate(
        indicators, n_boot=TOY_N_BOOT, seed=TOY_SEED, alpha=TOY_ALPHA
    )
    return public


def _primary_illustration(attacks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    left_rows = _rows_for_spec(attacks, target_context=ILLUSTRATION["left_target"])
    right_rows = _rows_for_spec(attacks, target_context=ILLUSTRATION["right_target"])
    left = additional_error_by_scene(left_rows)
    right = additional_error_by_scene(right_rows)
    interval = paired_mean_difference(
        left, right, n_boot=TOY_N_BOOT, seed=TOY_SEED, alpha=TOY_ALPHA
    )
    shared = sorted(set(left) & set(right))
    return {
        "analysis_role": ANALYSIS_PRIMARY,
        "research_question": "RQ1",
        "locked": False,
        "estimand": (
            "Mean over scenes of additional error under contradictory text "
            "minus additional error under neutral text, for one labeled attack cell."
        ),
        "method": ILLUSTRATION["method"],
        "epsilon": ILLUSTRATION["epsilon"],
        "epsilon_key": epsilon_key(ILLUSTRATION["epsilon"]),
        "step_budget": ILLUSTRATION["step_budget"],
        "source_context": ILLUSTRATION["source_context"],
        "left_target": ILLUSTRATION["left_target"],
        "right_target": ILLUSTRATION["right_target"],
        "positive_control_only": False,
        "scene_differences": [
            {"scene_id": scene_id, "difference": left[scene_id] - right[scene_id]}
            for scene_id in shared
        ],
        "bootstrap": interval,
        "reading": TOY_INTERPRETATION,
    }


def _public_alignment(blocks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    published = []
    for block in blocks:
        defined = block["defined_by_scene"]
        if defined:
            interval = scene_bootstrap_mean(
                defined, n_boot=TOY_N_BOOT, seed=TOY_SEED, alpha=TOY_ALPHA
            )
        else:
            interval = None
        published.append(
            {
                "source_context": block["source_context"],
                "target_context": block["target_context"],
                "positive_control_only": block["positive_control_only"],
                "analysis_role": (
                    "positive_control" if block["positive_control_only"] else ANALYSIS_SECONDARY
                ),
                "research_question": "RQ2",
                "n_scenes": block["n_scenes"],
                "n_defined": block["n_defined"],
                "n_undefined": block["n_undefined"],
                "mean": block["mean"],
                "undefined_reason": block["undefined_reason"],
                "missing_rule": (
                    "Undefined alignments are excluded from the mean and counted. "
                    "They are not imputed as zero."
                ),
                "bootstrap": interval,
            }
        )
    return {"analysis_role": ANALYSIS_SECONDARY, "by_pair": published}


def _public_conditional(attacks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = conditional_transfer(attacks, **TRANSFER_CELL)
    result["reading"] = TOY_INTERPRETATION
    return result


def _association(
    attacks: Sequence[Mapping[str, Any]],
    gradients: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = _rows_for_spec(
        attacks,
        method=ASSOCIATION_CELL["method"],
        epsilon=ASSOCIATION_CELL["epsilon"],
        step_budget=ASSOCIATION_CELL["step_budget"],
        source_context=ASSOCIATION_CELL["source_context"],
        target_context=ASSOCIATION_CELL["target_context"],
    )
    indicators = attack_success_indicators(rows)
    alignment = {
        row["scene_id"]: row["alignment_a"]
        for row in gradients
        if row["source_context"] == ASSOCIATION_CELL["source_context"]
        and row["target_context"] == ASSOCIATION_CELL["target_context"]
        and row["alignment_a"] is not None
    }
    eligible = sorted(
        scene_id
        for scene_id, (denominator, _) in indicators.items()
        if denominator == 1 and scene_id in alignment
    )
    correlation = pearson_correlation(
        [float(alignment[scene_id]) for scene_id in eligible],
        [float(indicators[scene_id][1]) for scene_id in eligible],
    )
    return {
        "analysis_role": ANALYSIS_SECONDARY,
        "research_question": "RQ2",
        "locked": False,
        "reading": TOY_INTERPRETATION,
        "method": ASSOCIATION_CELL["method"],
        "epsilon": ASSOCIATION_CELL["epsilon"],
        "epsilon_key": epsilon_key(ASSOCIATION_CELL["epsilon"]),
        "step_budget": ASSOCIATION_CELL["step_budget"],
        "source_context": ASSOCIATION_CELL["source_context"],
        "target_context": ASSOCIATION_CELL["target_context"],
        "join": (
            "Scenes correct before the attack, with a defined alignment for the "
            "same source and target. Success is 1 only for a valid incorrect output."
        ),
        "n_scenes": len(eligible),
        "pairs": [
            {
                "scene_id": scene_id,
                "alignment_a": float(alignment[scene_id]),
                "transfer_success": int(indicators[scene_id][1]),
            }
            for scene_id in eligible
        ],
        "pearson_r": correlation,
    }


def _rows_for_cell(
    attacks: Sequence[Mapping[str, Any]], cell: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    return _rows_for_spec(
        attacks,
        method=cell["method"],
        epsilon=cell["epsilon"],
        step_budget=cell["step_budget"],
        source_context=cell["source_context"],
        target_context=cell["target_context"],
    )


def _rows_for_spec(
    attacks: Sequence[Mapping[str, Any]],
    *,
    target_context: str,
    method: str = ILLUSTRATION["method"],
    epsilon: float = ILLUSTRATION["epsilon"],
    step_budget: int = ILLUSTRATION["step_budget"],
    source_context: str = ILLUSTRATION["source_context"],
) -> list[Mapping[str, Any]]:
    key = epsilon_key(epsilon)
    return [
        row
        for row in attacks
        if row["method"] == method
        and epsilon_key(row["epsilon"]) == key
        and int(row["step_budget"]) == int(step_budget)
        and row["source_context"] == source_context
        and row["target_context"] == target_context
    ]


def _scoring_notes() -> dict[str, str]:
    return {
        "clean_accuracy": (
            "correct scenes divided by scenes in that context. "
            "Invalid output is not correct. A context with no scenes has a null rate."
        ),
        "invalid_output_rate": "invalid scenes divided by scenes in that context.",
        "attack_success_rate": (
            "Valid incorrect outputs after the attack, divided by scenes correct before it. "
            "Invalid outputs are not successes and stay in the denominator. "
            "A cell with no clean-correct scene has a null rate."
        ),
        "additional_error": (
            "Mean over scenes of (task error after minus task error before). "
            "Task error is 1 when the answer is not correct, so invalid output counts here."
        ),
        "correct_before": (
            "Proposed meaning: clean correctness under the target context, "
            "the context in which the attacked image is scored."
        ),
        "cross_context_attack_transfer": (
            "Attack success and additional error on a cell whose source context "
            "differs from its target context."
        ),
        "gradient_alignment": (
            "A(s,t) = dot(g_t, sign(g_s)) / ||g_t||_1. "
            "Undefined when the L1 norm of g_t is zero."
        ),
        "answer_revealing": (
            "Positive control only. Never evidence of visual robustness by itself."
        ),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        raise TypeError("ndarray leaked into the toy summary")
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value
