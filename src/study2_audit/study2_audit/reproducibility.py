"""Recompute hashes, parses, labels, and revision pins from committed files.

Nothing here calls a model. Smol generations do not store token ids, so the
tokenization check for that table is the stored four-color NLL surface.
Qwen generations store token ids. Those ids are checked for length, for the
truncated flag, and for a stable map between the stored string and the stored
id list. They are not decoded.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

COLORS = ("blue", "green", "red", "yellow")
CONTEXTS = ("neutral", "helpful_nonrevealing", "contradictory", "answer_revealing")
PRIMARY = ("neutral", "helpful_nonrevealing")
SMOL_REVISION = "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
QWEN_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
PROTOCOL_SHA256 = "fe8a43561f649e996333913f14d2801e923c88061c0318f22fdbafbef16ff284"
MANIFEST_SHA256 = "bf21cdcfeca8f5c9a05e9f53465402c48f4a3175bbd363a38ca36dbc443f2367"
BUDGET_SHA256 = "b7cd9ad2808a5bbfd37c619b3f4d7df1225f95650f60d20b63b8eb49e41981a8"
GIT_HEAD_AT_RUN = "49a3945b42ad132b662178317eea0eb2bac80ddc"
MAX_NEW_TOKENS = 16
DISAGREEMENT_TOLERANCE = 1e-6
PARSE_FAMILY = "attribute"


def check_reproducibility(root: Path) -> dict[str, Any]:
    """Return one report. ``ok`` is true when every check matches the files."""

    checks: list[dict[str, Any]] = []
    checks.append(_hash_file(root, "docs/checkpoint-1.6/EVAL_PROTOCOL.md", PROTOCOL_SHA256, "protocol"))
    checks.append(
        _hash_file(
            root,
            "docs/checkpoint-1.5/run_model_feasibility.py",
            BUDGET_SHA256,
            "checkpoint_1_5_budget_script",
        )
    )
    checks.append(
        _hash_file(
            root,
            "fixtures/development_simple/manifest.jsonl",
            MANIFEST_SHA256,
            "row_color_manifest",
        )
    )
    checks.extend(_image_checksums(root))
    checks.append(_scene_graph_hashes(root))
    checks.append(_pins_module(root))
    predictions = _read_jsonl(root / "docs/checkpoint-1.6/development_simple/predictions.jsonl")
    manifest = _read_jsonl(root / "fixtures/development_simple/manifest.jsonl")
    summary = json.loads((root / "docs/checkpoint-1.6/development_simple/summary.json").read_text(encoding="utf-8"))
    run_meta = json.loads((root / "docs/checkpoint-1.6/development_simple/run_meta.json").read_text(encoding="utf-8"))
    gate = json.loads((root / "docs/checkpoint-1.6/development_simple/gate.json").read_text(encoding="utf-8"))
    qwen = json.loads((root / "docs/checkpoint-1.6/qwen_feasibility.json").read_text(encoding="utf-8"))
    checks.append(_smol_revisions(predictions, run_meta))
    checks.append(_qwen_revision(qwen))
    checks.append(_git_head(root, run_meta))
    parse_generation = _load_parse_generation(root)
    checks.append(_smol_generations(predictions, manifest, parse_generation))
    checks.append(_summary_matches(predictions, summary, parse_generation))
    checks.append(_qwen_labels(root, qwen))
    checks.append(_qwen_generations(qwen, parse_generation))
    checks.append(_qwen_token_map(qwen))
    checks.append(_gradient_gate(gate))
    failed = [item["name"] for item in checks if not item["ok"]]
    return {
        "ok": not failed,
        "n_checks": len(checks),
        "n_failed": len(failed),
        "failed": failed,
        "checks": checks,
        "model_inference": False,
    }


def _hash_file(root: Path, relative: str, expected: str, name: str) -> dict[str, Any]:
    digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    return _result(name, digest == expected, {"path": relative, "sha256": digest, "expected": expected})


def _image_checksums(root: Path) -> list[dict[str, Any]]:
    checksums = json.loads((root / "fixtures/development_simple/checksums.json").read_text(encoding="utf-8"))
    mismatches = []
    for relative, record in checksums["files"].items():
        path = root / "fixtures/development_simple" / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        size = path.stat().st_size
        if digest != record["sha256"] or size != record["bytes"]:
            mismatches.append({"path": relative, "sha256": digest, "bytes": size})
    return [
        _result(
            "row_color_png_checksums",
            not mismatches,
            {"n_files": len(checksums["files"]), "mismatches": mismatches},
        )
    ]


def _scene_graph_hashes(root: Path) -> dict[str, Any]:
    stored = json.loads((root / "fixtures/development_simple/scene_graph_hashes.json").read_text(encoding="utf-8"))
    mismatches = []
    seen = set()
    with (root / "fixtures/development_simple/manifest.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            scene_id = record["scene_id"]
            seen.add(scene_id)
            digest = _scene_graph_hash(record)
            if stored["hashes"].get(scene_id) != digest:
                mismatches.append(scene_id)
    missing = sorted(set(stored["hashes"]) - seen)
    return _result(
        "row_color_scene_graph_hashes",
        not mismatches and not missing and stored["algorithm"] == "sha256",
        {"n_scenes": len(seen), "mismatches": mismatches, "missing_from_manifest": missing},
    )


def _scene_graph_hash(record: Mapping[str, Any]) -> str:
    payload = {
        "question": record["question"],
        "answer": record["answer"],
        "objects": record["objects"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pins_module(root: Path) -> dict[str, Any]:
    pins = _load_module(root / "src/vlm_iface/vlm_iface/pins.py", "checkpoint16_pins")
    ok = pins.PINNED_REVISION == SMOL_REVISION and pins.MODEL_ID == "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    return _result(
        "smol_pin_module",
        ok,
        {"model_id": pins.MODEL_ID, "revision": pins.PINNED_REVISION},
    )


def _smol_revisions(predictions: Sequence[Mapping[str, Any]], run_meta: Mapping[str, Any]) -> dict[str, Any]:
    revisions = sorted({row["model_revision"] for row in predictions})
    ids = sorted({row["model_id"] for row in predictions})
    ok = (
        revisions == [SMOL_REVISION]
        and ids == ["HuggingFaceTB/SmolVLM2-500M-Video-Instruct"]
        and run_meta["model_revision"] == SMOL_REVISION
        and run_meta["protocol_sha256"] == PROTOCOL_SHA256
        and run_meta["manifest_sha256"] == MANIFEST_SHA256
        and run_meta["h_star_fitted"] is False
        and run_meta["qwen_loaded"] is False
        and run_meta["attacks_run"] is False
    )
    return _result(
        "smol_revision_on_every_row",
        ok,
        {"revisions": revisions, "model_ids": ids, "git_head_at_run": run_meta.get("git_head_at_run")},
    )


def _qwen_revision(qwen: Mapping[str, Any]) -> dict[str, Any]:
    revision = qwen["qwen_candidate"]["revision"]
    recorded_budget = qwen["budget_module"]["sha256"]
    ok = revision == QWEN_REVISION and recorded_budget == BUDGET_SHA256 and qwen["h_star_fitted"] is False
    return _result(
        "qwen_revision_and_budget_hash",
        ok,
        {"revision": revision, "budget_module_sha256": recorded_budget},
    )


def _git_head(root: Path, run_meta: Mapping[str, Any]) -> dict[str, Any]:
    head = run_meta["git_head_at_run"]
    ancestor = _is_ancestor(root, head)
    return _result(
        "row_color_git_head_is_ancestor_of_this_checkout",
        head == GIT_HEAD_AT_RUN and ancestor,
        {"git_head_at_run": head, "is_ancestor": ancestor},
    )


def _is_ancestor(root: Path, sha: str) -> bool:
    import subprocess

    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", sha, "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    return completed.returncode == 0


def _smol_generations(
    predictions: Sequence[Mapping[str, Any]],
    manifest: Sequence[Mapping[str, Any]],
    parse_generation,
) -> dict[str, Any]:
    by_scene = {row["scene_id"]: row for row in manifest}
    problems = []
    token_id_rows = 0
    for index, row in enumerate(predictions):
        scene = by_scene.get(row["scene_id"])
        if scene is None:
            problems.append({"index": index, "problem": "scene_missing_from_manifest"})
            continue
        if "token_ids" in row:
            token_id_rows += 1
        parsed = parse_generation(str(row["raw_output"]), truncated=bool(row["truncated"]), task_family=PARSE_FAMILY)
        judgment = _judge(parsed.parsed_answer, parsed.invalid, str(row["answer"]), list(row["answer_aliases"]))
        prompt = row["context_text"] + "\n" + row["question"]
        argmin = _argmin(row["class_nll"])
        disagreement = _disagreement(row["parsed_answer"], bool(row["invalid"]), row["class_nll"])
        partition = int(bool(row["correct"])) + int(bool(row["valid_wrong"])) + int(bool(row["invalid"]))
        manifest_context = scene["contexts"][row["context"]]
        if (
            row["task_family"] != "row_color"
            or row["answer"] != scene["answer"]
            or list(row["answer_aliases"]) != list(scene["answer_aliases"])
            or row["question"] != scene["question"]
            or row["counterfactual_group_id"] != scene["counterfactual_group_id"]
            or row["context_text"] != manifest_context["text"]
            or row["reveals_answer"] is not manifest_context["reveals_answer"]
            or row["prompt"] != prompt
            or "Reply using only the answer" in row["prompt"]
            or parsed.parsed_answer != row["parsed_answer"]
            or parsed.invalid != row["invalid"]
            or judgment["correct"] != row["correct"]
            or judgment["valid_wrong"] != row["valid_wrong"]
            or partition != 1
            or set(row["class_nll"]) != set(COLORS)
            or any(not _finite(value) for value in row["class_nll"].values())
            or argmin != row["class_argmin"]
            or disagreement != row["class_disagreement"]
            or row["label"] != "DEVELOPMENT ONLY"
        ):
            problems.append({"index": index, "scene_id": row["scene_id"], "context": row["context"]})
    groups: dict[str, dict[str, set[str]]] = {}
    for row in predictions:
        if row["context"] in PRIMARY:
            groups.setdefault(row["counterfactual_group_id"], {}).setdefault(row["context"], set()).add(row["context_text"])
    varying = [
        group_id
        for group_id, contexts in groups.items()
        if any(len(texts) != 1 for texts in contexts.values())
    ]
    if varying:
        problems.append({"problem": "primary_text_varies_inside_group", "groups": varying})
    return _result(
        "smol_labels_generations_and_class_nll",
        not problems and len(predictions) == 96,
        {
            "n_rows": len(predictions),
            "token_id_rows": token_id_rows,
            "tokenization": (
                "Smol rows store class NLL for the four color names and do not store token ids. "
                "The leading-space convention is the protocol sentence. It is not re-tokenized here."
            ),
            "n_problems": len(problems),
            "problems": problems[:8],
        },
    )


def _summary_matches(
    predictions: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    parse_generation,
) -> dict[str, Any]:
    del parse_generation
    clean_ids, clean_groups = _clean_both(predictions)
    problems = []
    if summary["n_predictions"] != len(predictions):
        problems.append("n_predictions")
    if summary["n_scenes"] != len({row["scene_id"] for row in predictions}):
        problems.append("n_scenes")
    if summary["n_groups"] != len({row["counterfactual_group_id"] for row in predictions}):
        problems.append("n_groups")
    if summary["n_scenes_clean_both_primary"] != len(clean_ids):
        problems.append("n_scenes_clean_both_primary")
    if summary["n_groups_clean_both_primary"] != len(clean_groups):
        problems.append("n_groups_clean_both_primary")
    if summary["scene_ids_clean_both_primary"] != clean_ids:
        problems.append("scene_ids_clean_both_primary")
    if summary["group_ids_clean_both_primary"] != clean_groups:
        problems.append("group_ids_clean_both_primary")
    if summary["h_star_fitted"] is not False or summary["official_metric"] != "strict_free_generation":
        problems.append("official_metric")
    for context in CONTEXTS:
        rows = [row for row in predictions if row["context"] == context]
        cell = summary["clean_by_context"][context]
        if cell["n_correct"] != sum(bool(row["correct"]) for row in rows):
            problems.append(f"{context}.n_correct")
        if cell["n_invalid"] != sum(bool(row["invalid"]) for row in rows):
            problems.append(f"{context}.n_invalid")
        if cell["n_valid_wrong"] != sum(bool(row["valid_wrong"]) for row in rows):
            problems.append(f"{context}.n_valid_wrong")
        if cell["n_rows"] != len(rows):
            problems.append(f"{context}.n_rows")
    valid = [row for row in predictions if not row["invalid"]]
    matched = sum(row["class_argmin"] == row["parsed_answer"] for row in valid)
    recorded = summary["class_argmin_matches_official_on_valid_rows"]
    if recorded["numerator"] != matched or recorded["denominator"] != len(valid):
        problems.append("class_argmin_matches_official_on_valid_rows")
    neutral = summary["clean_by_context"]["neutral"]["accuracy"]["value"]
    helpful = summary["clean_by_context"]["helpful_nonrevealing"]["accuracy"]["value"]
    unsuitable = neutral < 0.50 or helpful < 0.50
    in_band = 0.60 <= neutral <= 0.80 and 0.60 <= helpful <= 0.80
    if summary["unsuitable"] is not unsuitable or summary["meets_design_band"] is not in_band:
        problems.append("design_band")
    if summary["recommendation"] != ("do_not_adopt" if unsuitable else ("retain_for_later_review" if in_band else "outside_band")):
        problems.append("recommendation")
    return _result("row_color_summary_matches_predictions", not problems, {"problems": problems})


def _qwen_labels(root: Path, qwen: Mapping[str, Any]) -> dict[str, Any]:
    manifest = {row["scene_id"]: row for row in _read_jsonl(root / "fixtures/development_v2/manifest.jsonl")}
    problems = []
    scenes = qwen["subset_definition"]["scenes"]
    if [scene["scene_id"] for scene in scenes] != ["dev2-attribute-00-a", "dev2-counting-00-a", "dev2-spatial-00-a"]:
        problems.append("subset_scene_list")
    for scene in scenes:
        source = manifest[scene["scene_id"]]
        if scene["fixture_answer"] != source["answer"] or scene["question"] != source["question"]:
            problems.append(scene["scene_id"])
        if scene["task_family"] != source["task_family"]:
            problems.append(scene["scene_id"] + ".family")
        for context, text in scene["contexts"].items():
            if text != source["contexts"][context]["text"]:
                problems.append(f"{scene['scene_id']}.{context}")
    return _result(
        "qwen_subset_labels_match_development_v2_manifest",
        not problems,
        {"problems": problems, "note": "fixture_answer is the scene label. It is not a model generation."},
    )


def _qwen_generations(qwen: Mapping[str, Any], parse_generation) -> dict[str, Any]:
    problems = []
    family_for_parser = {
        "attribute": "attribute",
        "spatial": "spatial",
        "counting": "counting",
    }
    for index, row in enumerate(qwen["development_predictions"]):
        family = family_for_parser[row["task_family"]]
        parsed = parse_generation(str(row["raw_output"]), truncated=bool(row["truncated"]), task_family=family)
        reasons = list(parsed.reasons)
        partition = int(bool(row["correct"])) + int(bool(row["valid_wrong"])) + int(bool(row["invalid"]))
        if (
            parsed.invalid is not True
            or parsed.parsed_answer is not None
            or row["invalid"] is not True
            or row["correct"] is not False
            or row["valid_wrong"] is not False
            or row["parsed_answer"] is not None
            or reasons != list(row["parse_reasons"])
            or partition != 1
            or row["label"] != "DEVELOPMENT ONLY"
        ):
            problems.append(
                {
                    "index": index,
                    "scene_id": row["scene_id"],
                    "context": row["context"],
                    "recomputed_reasons": reasons,
                    "stored_reasons": list(row["parse_reasons"]),
                }
            )
    return _result(
        "qwen_generation_parses",
        not problems and len(qwen["development_predictions"]) == 12,
        {"n_rows": len(qwen["development_predictions"]), "problems": problems},
    )


def _qwen_token_map(qwen: Mapping[str, Any]) -> dict[str, Any]:
    """Check the stored id lists against the stored strings. Do not decode them."""

    by_text: dict[str, tuple[int, ...]] = {}
    by_ids: dict[tuple[int, ...], str] = {}
    problems = []
    end_ids = set()
    for index, row in enumerate(qwen["development_predictions"]):
        ids = row["token_ids"]
        if not isinstance(ids, list) or not ids or any(isinstance(item, bool) or not isinstance(item, int) for item in ids):
            problems.append({"index": index, "problem": "token_ids"})
            continue
        if len(ids) > MAX_NEW_TOKENS:
            problems.append({"index": index, "problem": "longer_than_max_new_tokens"})
        truncated = len(ids) == MAX_NEW_TOKENS
        if bool(row["truncated"]) is not truncated:
            problems.append({"index": index, "problem": "truncated_flag", "length": len(ids)})
        if not truncated:
            end_ids.add(ids[-1])
        key = tuple(ids)
        text = row["raw_output"]
        previous = by_text.get(text)
        if previous is not None and previous != key:
            problems.append({"index": index, "problem": "same_text_different_ids"})
        by_text[text] = key
        previous_text = by_ids.get(key)
        if previous_text is not None and previous_text != text:
            problems.append({"index": index, "problem": "same_ids_different_text"})
        by_ids[key] = text
    if len(end_ids) != 1:
        problems.append({"problem": "non_truncated_rows_do_not_share_one_final_id", "end_ids": sorted(end_ids)})
    end_id = next(iter(end_ids)) if len(end_ids) == 1 else None
    if end_id is not None:
        for index, row in enumerate(qwen["development_predictions"]):
            ids = row["token_ids"]
            if row["truncated"] and ids and ids[-1] == end_id:
                problems.append({"index": index, "problem": "truncated_row_ends_with_the_stop_id"})
    return _result(
        "qwen_token_id_map",
        not problems,
        {
            "max_new_tokens": MAX_NEW_TOKENS,
            "n_distinct_strings": len(by_text),
            "observed_final_id_on_non_truncated_rows": end_id,
            "decoded": False,
            "problems": problems,
        },
    )


def _gradient_gate(gate: Mapping[str, Any]) -> dict[str, Any]:
    gradients = gate["gradients"]
    ok = (
        gate["scene_id"] == "dev3-row-00-a"
        and gate["attempted"] is True
        and gate["passed"] is True
        and len(gradients) == 2
        and [item["context"] for item in gradients] == ["neutral", "helpful_nonrevealing"]
        and all(item["finite"] is True and item["nonzero_fraction"] == 1.0 for item in gradients)
        and all(item["height"] == 384 and item["width"] == 512 for item in gradients)
        and all(item["loss"] == "canonical_word" and item["repeat_max_abs_diff"] == 0.0 for item in gradients)
    )
    return _result(
        "row_color_gradient_gate_is_one_scene",
        ok,
        {"scene_id": gate.get("scene_id"), "n_gradients": len(gradients)},
    )


def _clean_both(predictions: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    scenes: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in predictions:
        if row["context"] in PRIMARY:
            scenes.setdefault(row["scene_id"], {})[row["context"]] = row
    clean = []
    for scene_id in sorted(scenes):
        pair = scenes[scene_id]
        if set(pair) == set(PRIMARY) and all(bool(pair[context]["correct"]) for context in PRIMARY):
            clean.append(scene_id)
    groups = sorted({scenes[scene_id]["neutral"]["counterfactual_group_id"] for scene_id in clean})
    return clean, groups


def _judge(parsed_answer: str | None, invalid: bool, answer: str, aliases: list[str]) -> dict[str, bool]:
    if invalid or parsed_answer is None:
        return {"correct": False, "valid_wrong": False, "invalid": True}
    if parsed_answer == answer or parsed_answer in aliases:
        return {"correct": True, "valid_wrong": False, "invalid": False}
    return {"correct": False, "valid_wrong": True, "invalid": False}


def _argmin(scores: Mapping[str, float]) -> str:
    return min(scores, key=lambda name: (float(scores[name]), name))


def _disagreement(parsed_answer: str | None, invalid: bool, scores: Mapping[str, float]) -> bool:
    if invalid or parsed_answer is None:
        return False
    best = min(float(value) for value in scores.values())
    return float(scores[parsed_answer]) > best + DISAGREEMENT_TOLERANCE


def _finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number not in (float("inf"), float("-inf"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_parse_generation(root: Path):
    module = _load_module(root / "src/vlm_iface/vlm_iface/parsing.py", "checkpoint16_parsing")
    return module.parse_generation


def _load_module(path: Path, name: str):
    import sys

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = write_bytecode
    return module


def _result(name: str, ok: bool, detail: Mapping[str, Any]) -> dict[str, Any]:
    return {"name": name, "ok": ok, "detail": dict(detail)}
