"""Calibration report. Counts only. This module does not load a model and does not fit H*."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .prior import describe_scenes, load_prior
from .protocol import LABEL, REPORT_RELATIVE, SOURCE_CONTEXT, TARGET_CONTEXT


def render_report(progress: dict[str, Any], description: list[dict[str, Any]]) -> str:
    """Markdown for the finished or stopped calibration grid."""
    configurations = progress.get("configurations") or []
    both = [row for row in configurations if row.get("adequate_source_and_transfer_variation") is True]
    lines = [
        "# Checkpoint 2.0 calibration report",
        "",
        f"{LABEL}. Research lead: Federico Cavaletto. Qwen3-VL-2B-Instruct, bfloat16, MPS. "
        "No H* fit. No held-out scenes. The sample is exploratory. It is not a population attack-success rate.",
        "",
        f"Protocol commit: `{progress.get('protocol_commit') or ''}`.",
        f"Protocol sha256: `{progress.get('protocol_sha256') or ''}`.",
        f"Status: `{progress.get('status')}`.",
        f"Stop reason: {progress.get('stopped_reason') or ''}",
        "",
        "## Sample",
        "",
        f"Rule `{progress.get('selection_rule')}`. "
        f"Group count on the development manifest: {progress.get('n_groups')}. "
        "Indexes 0, 3, 6, ..., 45 when that count is 48. Both arms. "
        "The rule reads group id and scene id. It does not read alignment, margin, loss, or a Checkpoint 1.9 outcome.",
        "",
        f"Scenes finished: {progress.get('n_finished')} of {progress.get('n_scenes')}.",
        "",
        "## Counts",
        "",
        "Source context is neutral. The helpful context is evaluation only. "
        "The stored attack is the official 8-bit image. "
        "Source success uses scenes whose clean neutral decision is the labeled color and whose perturbed neutral decision was recorded. "
        "Conditional transfer uses source successes that were also clean-correct under helpful text. "
        "Scenes already wrong on that clean helpful answer are excluded and counted, and they are not transfer failures. "
        "A missing decision stays out of the numerator and the denominator.",
        "",
        "Near the floor means the numerator is 0 or 1. "
        "Near the ceiling means the numerator is the denominator or one less. "
        "A denominator below 2 is undefined. "
        "Adequate source success is an interior or near-ceiling source-success count. "
        "Variation in conditional transfer is an interior conditional-transfer count. "
        "Those labels use the counts only.",
        "",
        "| Configuration | Source success | Source missing | Conditional transfer | Excluded | Transfer missing | Target flips | Position |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in configurations:
        lines.append(
            "| `{configuration}` | {source} | {source_missing} | {transfer} | {excluded} | {transfer_missing} | {target} | {position} |".format(
                configuration=row["configuration"],
                source=_ratio(row["n_source_success"], row["n_source_success_denominator"]),
                source_missing=row["n_source_missing"],
                transfer=_ratio(row["n_conditional_transfer"], row["n_conditional_transfer_denominator"]),
                excluded=row["n_excluded_target_not_clean"],
                transfer_missing=row["n_transfer_missing"],
                target=_ratio(row["n_target_flip"], row["n_target_flip_denominator"]),
                position=_position_text(row),
            )
        )
    lines.extend(["", "## Judgment", "", _judgment(progress, configurations, both), ""])
    if progress.get("status") == "completed":
        lines.extend(
            [
                "Epsilon stayed inside {1/255, 2/255}. The recorded PGD steps stayed 2, 3, 4, 5, and 7. "
                "The objective stayed the summed labeled-color NLL under neutral.",
                "",
            ]
        )
    lines.extend(_description_section(description))
    lines.extend(_memory_section(progress))
    lines.extend(_hash_section(progress.get("file_sha256") or {}))
    lines.append("")
    return "\n".join(lines)


def write_report(root: Path, progress: dict[str, Any], description: list[dict[str, Any]]) -> None:
    path = root / REPORT_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(progress, description), encoding="utf-8")


def checkpoint_1_8_description(root: Path, scene_ids: list[str]) -> list[dict[str, Any]]:
    if not scene_ids:
        return []
    return describe_scenes(load_prior(root), scene_ids)


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _judgment(progress: dict[str, Any], configurations: list[dict[str, Any]], both: list[dict[str, Any]]) -> str:
    if progress.get("status") != "completed":
        return (
            "The recorded grid is not complete, so this is not a judgment of every listed configuration. "
            f"Stop status: `{progress.get('status')}`."
        )
    if not configurations:
        return "No configuration was recorded."
    grouped = _groups(configurations)
    if not both:
        return (
            "No recorded configuration has adequate source success and variation in conditional transfer. "
            "Every recorded configuration is near a floor, near a ceiling, or has an undefined denominator. "
            + grouped
        )
    names = ", ".join(
        "`{name}` ({source} source success, {transfer} conditional transfer)".format(
            name=row["configuration"],
            source=_ratio(row["n_source_success"], row["n_source_success_denominator"]),
            transfer=_ratio(row["n_conditional_transfer"], row["n_conditional_transfer_denominator"]),
        )
        for row in both
    )
    return (
        f"Configurations with adequate source success and variation in conditional transfer: {names}. "
        "The list is not ranked, and alignment was not used to make it. "
        "This report does not pick one of them. "
        + grouped
    )


def _groups(configurations: list[dict[str, Any]]) -> str:
    def names(predicate) -> str:
        chosen = [row["configuration"] for row in configurations if predicate(row)]
        return ", ".join(f"`{name}`" for name in chosen) if chosen else "none"

    return (
        "Adequate source and transfer near the ceiling: "
        f"{names(lambda row: row.get('adequate_source_success') and row.get('transfer_position') == 'near_ceiling')}. "
        "Adequate source and transfer near the floor: "
        f"{names(lambda row: row.get('adequate_source_success') and row.get('transfer_position') == 'near_floor')}. "
        f"Source near the floor: {names(lambda row: row.get('source_position') == 'near_floor')}. "
        "Undefined: "
        f"{names(lambda row: row.get('source_position') == 'undefined' or row.get('transfer_position') == 'undefined')}."
    )


def _position_text(row: dict[str, Any]) -> str:
    if row.get("adequate_source_and_transfer_variation"):
        mark = "adequate source, transfer varies"
    else:
        mark = "no"
    return f"{mark}; source {row.get('source_position')}; transfer {row.get('transfer_position')}"


def _description_section(description: list[dict[str, Any]]) -> list[str]:
    lines = [
        "## Stored Checkpoint 1.8 scalars",
        "",
        "Alignment is A(neutral, helpful). Margins are the stored clean margins. "
        "Gradient L1 is the stored neutral gradient. "
        "The JSON file holds the stored floats. This table rounds them for reading. "
        "The columns were not used to choose the sample or a configuration, and they were not regressed on transfer.",
        "",
        "| Scene | Label | Alignment | Neutral margin | Helpful margin | Neutral gradient L1 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    if not description:
        lines.append("|  |  |  |  |  |  |")
        lines.append("")
        return lines
    for row in description:
        lines.append(
            "| `{scene}` | {answer} | {alignment} | {neutral} | {helpful} | {l1} |".format(
                scene=row.get("scene_id") or "",
                answer=row.get("answer") or "",
                alignment=_num(row.get("alignment_neutral_to_helpful"), 6),
                neutral=_num(row.get("neutral_margin"), 4),
                helpful=_num(row.get("helpful_margin"), 4),
                l1=_num(row.get("neutral_gradient_l1"), 1),
            )
        )
    lines.append("")
    return lines


def _memory_section(progress: dict[str, Any]) -> list[str]:
    loaded = progress.get("loaded_model") or {}
    device = loaded.get("device")
    dtype = loaded.get("dtype")
    lines = [
        "## Memory",
        "",
        f"Available bytes entering the load decision: {progress.get('available_bytes_entering_decision')}. "
        f"Fresh reading at the decision: {progress.get('available_bytes_at_decision')}. "
        "Bfloat16 requires 11,509,447,200 available bytes. "
        "Float32 requires 20,871,410,752 and was not loaded. "
        "The 6 GiB floor is 6,442,450,944. The guard was not lowered.",
        "",
    ]
    if device == "mps" and dtype == "torch.bfloat16":
        lines.append(
            f"Attack device: MPS. Dtype: {dtype}. "
            f"Scene wall-clock sum: {progress.get('scene_seconds_sum')} seconds. "
            f"This process wall clock: {progress.get('wall_seconds')} seconds. "
            "These times are from the MPS run. A CPU timing is not included."
        )
    else:
        lines.append(
            "The attack device is not recorded as MPS bfloat16. "
            f"Loaded device value: {device!r}. Loaded dtype value: {dtype!r}."
        )
    peak = progress.get("peak_memory") or {}
    if peak:
        lines.extend(
            [
                "",
                f"Peak RSS mebibytes: {peak.get('peak_rss_mb')}. "
                f"Peak MPS current allocation mebibytes: {peak.get('peak_mps_current_allocated_mb')}. "
                f"Peak MPS driver allocation mebibytes: {peak.get('peak_mps_driver_allocated_mb')}. "
                f"Lowest OS-available bytes during the run: {peak.get('min_os_available_bytes')}.",
            ]
        )
    lines.extend(
        [
            "",
            f"Source context: `{SOURCE_CONTEXT}`. Evaluation contexts: `{SOURCE_CONTEXT}`, `{TARGET_CONTEXT}`.",
            "",
        ]
    )
    return lines


def _hash_section(hashes: dict[str, Any]) -> list[str]:
    lines = ["## Files", ""]
    if not hashes:
        lines.append("Result hashes are written with the result files.")
        return lines
    for name in sorted(hashes):
        lines.append(f"- `{name}` sha256 `{hashes[name]}`")
    return lines


def _ratio(numerator: object, denominator: object) -> str:
    if isinstance(numerator, bool) or not isinstance(numerator, int):
        return ""
    if isinstance(denominator, bool) or not isinstance(denominator, int):
        return ""
    return f"{numerator}/{denominator}"


def _num(value: object, digits: int) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return ""
    return f"{float(value):.{digits}f}"
