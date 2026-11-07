"""Development report. Counts, checks, and the H* statement. No new attack."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .protocol import HSTAR_UNESTIMATED, LABEL, REPRODUCTION_COMMAND


def write_report(
    path: Path,
    analysis: Mapping[str, Any],
    progress: Mapping[str, Any],
    scalars: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(analysis, progress, scalars), encoding="utf-8")


def render_report(
    analysis: Mapping[str, Any],
    progress: Mapping[str, Any],
    scalars: list[dict[str, Any]],
) -> str:
    lines = [
        "# Checkpoint 2.1 development report",
        "",
        f"{LABEL}. Research lead: Federico Cavaletto. Qwen3-VL-2B-Instruct, bfloat16, MPS. "
        "The sample is all 96 development_v4 scenes.",
        "",
        f"Protocol commit: `{analysis.get('protocol_commit') or progress.get('protocol_commit') or ''}`.",
        f"Protocol sha256: `{progress.get('protocol_sha256') or ''}`.",
        f"Attack code commit at the start of the run: `{analysis.get('attack_code_commit') or ''}`.",
        f"Attack status: `{progress.get('status')}`.",
        f"Analysis stop reason: `{analysis.get('stop_reason')}`.",
        "",
        analysis.get("h_star_statement") or HSTAR_UNESTIMATED,
        "",
        "## Sample",
        "",
        "Rule `all-development-v4-scenes`. All 48 counterfactual groups, both arms. "
        "Source context is neutral. `helpful_nonrevealing` is evaluation only. "
        "The attack is PGD at epsilon 1/255 for 3 steps, step length epsilon/4, started from the clean image. "
        "FGSM at 1/255, matched random-sign at 1/255, and step 7 of the same trajectory are the pre-specified comparisons.",
        "",
        f"Scenes finished: {progress.get('n_finished')} of {progress.get('n_scenes')}.",
        "",
        "## Counts",
        "",
        "Source success uses scenes whose clean neutral decision is the labeled color and whose perturbed neutral decision was recorded. "
        "The numerator is a decision other than the labeled color. "
        "Conditional transfer uses source successes whose clean helpful decision is the labeled color and whose perturbed helpful decision was recorded. "
        "The numerator is a helpful decision other than the labeled color. "
        "A source success whose clean helpful decision is already wrong is excluded. "
        "A missing decision stays out of the numerator and the denominator.",
        "",
        "| Configuration | Role | Source success | Source missing | Conditional transfer | Excluded | Transfer missing | Target attack success |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in analysis.get("counts") or []:
        lines.append(
            "| `{configuration}` | {role} | {source} | {source_missing} | {transfer} | {excluded} | {transfer_missing} | {target} |".format(
                configuration=row["configuration"],
                role=row["role"],
                source=_ratio(row["n_source_success"], row["n_source_success_denominator"]),
                source_missing=row["n_source_missing"],
                transfer=_ratio(row["n_conditional_transfer"], row["n_conditional_transfer_denominator"]),
                excluded=row["n_excluded_target_not_clean"],
                transfer_missing=row["n_transfer_missing"],
                target=_ratio(row["n_target_flip"], row["n_target_flip_denominator"]),
            )
        )
    lines.extend(["", "## Group intervals", ""])
    lines.append(
        "Each interval resamples the 48 counterfactual groups with replacement, 4000 draws, seed `20261109`, "
        "and NumPy's linear quantile. A zero denominator leaves the rate unset."
    )
    lines.append("")
    lines.append("| Configuration | Rate | Estimate | 95% interval |")
    lines.append("| --- | --- | ---: | --- |")
    for name, rates in (analysis.get("rate_intervals") or {}).items():
        for kind, stat in rates.items():
            lines.append(
                f"| `{name}` | {kind} | {_estimate(stat)} | {_interval(stat)} |"
            )
    lines.extend(["", "## Effective sample", ""])
    if analysis.get("effective_sample_applied") is not True:
        lines.append("The effective-sample check was not applied. The development run did not finish.")
    else:
        eligible = analysis.get("eligible") or {}
        lines.append(
            f"Eligible scenes: {eligible.get('n_eligible')}. "
            f"Eligible conditional transfers: {eligible.get('n_transfer')}. "
            f"Eligible transfer failures: {eligible.get('n_fail')}. "
            f"Eligible groups: {eligible.get('n_groups')}."
        )
        lines.append("")
        lines.append(
            "The floor is 40 transfer failures and 40 transfers, ten events for each of the four regressors, "
            "and at least 6 counterfactual groups."
        )
        failed = analysis.get("failed_checks") or []
        if failed:
            lines.append("")
            lines.append("Failed checks: " + ", ".join(f"`{name}`" for name in failed) + ".")
        exclusions = analysis.get("predictor_exclusions") or {}
        lines.append("")
        lines.append(
            "Predictor exclusions among the conditional-transfer denominator: "
            f"{exclusions.get('n_scenes', 0)} "
            f"(alignment {exclusions.get('alignment_undefined', 0)}, "
            f"margins {exclusions.get('margin_not_finite', 0)}, "
            f"gradient L1 {exclusions.get('gradient_l1_not_positive', 0)}). "
            "A transfer failure with an undefined predictor is not one of the eligible transfer failures."
        )
    lines.extend(["", "## H*", ""])
    hstar = analysis.get("h_star")
    if analysis.get("h_star_estimated") is True and isinstance(hstar, dict):
        lines.append(
            f"`β_A` = {_number(hstar.get('beta_a'))}. "
            f"95% wild cluster interval [{_number(hstar.get('ci_low'))}, {_number(hstar.get('ci_high'))}]. "
            f"Directional reading holds: {hstar.get('directional_reading_holds')}. "
            "`confirmatory_supported` is false."
        )
        lines.append("")
        lines.append(f"Zero residual fit: {hstar.get('zero_residual_fit')}.")
        logistic = hstar.get("logistic") or {}
        if logistic.get("interval") is None and logistic.get("ci_low") is None:
            lines.append(f"Logistic sensitivity: `{logistic.get('status')}`. The logistic interval is null.")
        else:
            lines.append(
                "Logistic sensitivity on alignment: "
                f"{_number(logistic.get('estimate'))}, "
                f"interval [{_number(logistic.get('ci_low'))}, {_number(logistic.get('ci_high'))}]. "
                "The logistic coefficient is a change in log odds."
            )
    else:
        lines.append(analysis.get("h_star_statement") or HSTAR_UNESTIMATED)
        lines.append("No coefficient was computed.")
    lines.extend(["", "## Verification", ""])
    lines.append(
        f"Verification passed: {analysis.get('verification_passed')}. "
        "Checksums, L-infinity bounds, color decisions, preserved Checkpoint 1.7–2.0 files, "
        "and the memory lines are checked again from the saved run."
    )
    gradient = analysis.get("gradient_l1_comparison") or {}
    lines.append("")
    lines.append(
        "The regression covariate is the stored Checkpoint 1.8 neutral gradient L1. "
        f"Recomputed clean gradients compared: {gradient.get('n_compared')}. "
        f"Missing: {gradient.get('n_missing')}. "
        f"Max absolute difference: {_number(gradient.get('max_abs'))}. "
        f"Median absolute difference: {_number(gradient.get('median_abs'))}."
    )
    margins = analysis.get("margin_comparison") or {}
    for name, block in margins.items():
        if isinstance(block, dict):
            lines.append(
                f"Stored {name} versus the recomputed clean margin: "
                f"max absolute difference {_number(block.get('max_abs'))} "
                f"over {block.get('n_compared')} scenes."
            )
    lines.extend(["", "## Memory", ""])
    lines.append(
        f"Available bytes entering the load decision: {progress.get('available_bytes_entering_decision')}. "
        f"Fresh reading at the decision: {progress.get('available_bytes_at_decision')}. "
        "Bfloat16 requires 11,509,447,200 available bytes. "
        "Float32 requires 20,871,410,752 and was not loaded. "
        "The 6 GiB floor is 6,442,450,944. The guard was not lowered."
    )
    lines.append("")
    loaded = progress.get("loaded_model") or {}
    lines.append(
        f"Attack device: {loaded.get('device')}. Dtype: {loaded.get('dtype')}. "
        f"Scene wall-clock sum: {progress.get('scene_seconds_sum')} seconds. "
        f"Recorded wall clock: {progress.get('wall_seconds')} seconds."
    )
    peak = progress.get("peak_memory") or {}
    if peak:
        lines.append("")
        lines.append(
            f"Peak RSS mebibytes: {peak.get('peak_rss_mb')}. "
            f"Peak MPS current allocation mebibytes: {peak.get('peak_mps_current_allocated_mb')}. "
            f"Peak MPS driver allocation mebibytes: {peak.get('peak_mps_driver_allocated_mb')}. "
            f"Lowest OS-available bytes during the run: {peak.get('min_os_available_bytes')}."
        )
    lines.extend(["", "## Scenes", ""])
    lines.append(
        "Primary decisions are the official 8-bit image after 3 PGD steps. "
        "Status is the conditional-transfer status for that image."
    )
    lines.append("")
    lines.append("| Scene | Group | Label | Clean neutral | Clean helpful | PGD3 neutral | PGD3 helpful | Status |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    scalar_group = {row.get("scene_id"): row for row in scalars}
    for scene_id in progress.get("scene_ids") or []:
        scene = (progress.get("scenes") or {}).get(scene_id) or {}
        if scene.get("finished") is not True:
            continue
        clean = scene.get("clean") or {}
        pgd3 = (((scene.get("epsilons") or {}).get("1/255") or {}).get("pgd") or {}).get("3") or {}
        perturbed = pgd3.get("perturbed") or {}
        lines.append(
            "| `{scene}` | `{group}` | {label} | {clean_n} | {clean_h} | {pgd_n} | {pgd_h} | {status} |".format(
                scene=scene_id,
                group=scene.get("counterfactual_group_id") or (scalar_group.get(scene_id) or {}).get("counterfactual_group_id"),
                label=scene.get("answer"),
                clean_n=_view_decision((clean.get("neutral") or {})),
                clean_h=_view_decision((clean.get("helpful_nonrevealing") or {})),
                pgd_n=_view_decision((perturbed.get("neutral") or {})),
                pgd_h=_view_decision((perturbed.get("helpful_nonrevealing") or {})),
                status=pgd3.get("transfer_status") or "",
            )
        )
    lines.extend(["", "## Files", ""])
    for name, digest in (analysis.get("file_sha256") or {}).items():
        lines.append(f"- `{name}` sha256 `{digest}`")
    lines.extend(["", "## Reproduction", "", f"`{REPRODUCTION_COMMAND}`", ""])
    return "\n".join(lines)


def _ratio(numerator: object, denominator: object) -> str:
    if not isinstance(numerator, int) or isinstance(numerator, bool):
        return "unset"
    if not isinstance(denominator, int) or isinstance(denominator, bool) or denominator == 0:
        return "unset"
    return f"{numerator}/{denominator}"


def _estimate(stat: Mapping[str, Any] | None) -> str:
    if not isinstance(stat, Mapping) or stat.get("estimate") is None:
        return "unset"
    return _number(stat.get("estimate"))


def _interval(stat: Mapping[str, Any] | None) -> str:
    if not isinstance(stat, Mapping) or stat.get("ci_low") is None or stat.get("ci_high") is None:
        reason = "" if not isinstance(stat, Mapping) else stat.get("undefined_reason") or ""
        return "unset" if not reason else f"unset ({reason})"
    return f"[{_number(stat.get('ci_low'))}, {_number(stat.get('ci_high'))}]"


def _number(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "unset"
    return f"{float(value):.6g}"


def _view_decision(view: Mapping[str, Any] | None) -> str:
    if not isinstance(view, Mapping):
        return ""
    decision = view.get("decision")
    return decision if isinstance(decision, str) else ""
