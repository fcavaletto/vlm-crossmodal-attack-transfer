"""Checkpoint 4 Stage 1: the unchanged 2.4 comparison on InternVL3.5-1B (single tile).

EXPLORATORY, development data only (the 48 arm-a scenes of development_v4).

    python -m internvl_probe.stage1 run --dtype bf16
    python -m internvl_probe.stage1 run --dtype fp32

Protocol: docs/checkpoint-4-internvl/STAGE1_PROTOCOL.md. The run refuses to start unless the
protocol, this file and the analysis script are committed and unchanged.

Per scene and attack source (pass 1: neutral, all scenes; pass 2: helpful, all scenes):
FGSM at eps, PGD-3 with eps/4 steps (gradients on the float surrogate path, as Qwen's
unclipped differentiable path), top-|clean gradient| and 8 same-count random masks of
FGSM's rounded perturbation. Every perturbed image is rounded to 8 bits and scored in both
contexts on the official processor path and on the float surrogate path.

Memory: the Stage 0 safeguards, unchanged (EMP-1 preload lines, MemoryMonitor with the
10 GiB footprint abort, MPS allocator capped at 10 GiB). Records are appended once per
(scene, source), so an interruption keeps every completed unit. No resume, no retry.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from .common import Stop, development_scenes, git_head, repo_root, scene_image

EPS = 1.0 / 255.0
N_SEEDS = 8
SEED_PREFIX = "checkpoint-4-stage1-mask"
CONTEXTS = ("neutral", "helpful_nonrevealing")
SOURCES = ("neutral", "helpful_nonrevealing")  # pass order: neutral (primary) first, over all scenes
TILING = "single"
CAP_S = 4 * 3600  # per precision; a truncated run keeps what it measured and says so
OUT = Path("docs/checkpoint-4-internvl/stage1")
PROTECTED = ("docs/checkpoint-4-internvl/STAGE1_PROTOCOL.md", "docs/checkpoint-4-internvl/analyze_stage1.py",
             "src/internvl_probe/internvl_probe/stage1.py", "src/internvl_probe/internvl_probe/backend.py",
             "src/internvl_probe/internvl_probe/pixels.py")
GIB = 1024**3


def other(ctx: str) -> str:
    return CONTEXTS[1] if ctx == CONTEXTS[0] else CONTEXTS[0]


def arm_a(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    a = sorted((r for r in rows if r["scene_id"].endswith("-a")), key=lambda r: r["counterfactual_group_id"])
    if len(a) != 48 or len({r["counterfactual_group_id"] for r in a}) != 48:
        raise Stop("arm a does not give one scene in each of 48 groups")
    return a


def protocol_commit(root: Path) -> str:
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *PROTECTED], cwd=root, capture_output=True, text=True).stdout
    if dirty.strip():
        raise Stop("the Stage 1 protocol or code has uncommitted changes:\n" + dirty)
    commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", PROTECTED[0]], cwd=root, text=True).strip()
    if not commit:
        raise Stop("STAGE1_PROTOCOL.md is not committed")
    pushed = subprocess.run(["git", "branch", "-r", "--contains", commit], cwd=root, capture_output=True, text=True).stdout
    if not pushed.strip():
        raise Stop("the protocol commit is not on any remote branch: push it first")
    return commit


def unit_block(m, x: torch.Tensor, prompts: dict, label: str, sid: str, source: str, monitor) -> tuple[list[dict], dict]:
    """One (scene, source) unit: gradients, attacks, masks, scores on both paths in both contexts."""
    from attack_feasibility.steps import pgd_update, project_signed_step, scene_seed
    from path_probe import geometry as geo
    from path_probe import masks as mk
    from qwen_rgb.constraints import round_trip_uint8

    base = {"scene_id": sid, "source": source, "target": other(source)}
    t0 = time.perf_counter()
    _, g0 = m.gradient(x, prompts[source], label, "nll")
    monitor.check()
    fgsm = project_signed_step(x, g0, EPS, "positive") - x
    codes = {"fgsm": geo.encode_delta(fgsm, EPS)}
    grads = [g0]
    current, g_prev = x.detach().clone(), g0
    for k in range(1, 4):
        if k > 1:
            _, g_prev = m.gradient(current, prompts[source], label, "nll")
            grads.append(g_prev)
            monitor.check()
        current = pgd_update(current, x, g_prev, EPS / 4.0, EPS)
        codes[f"pgd{k}"] = geo.encode_delta(current - x, EPS)
    pgd_float = current
    t_grad = time.perf_counter() - t0

    r_f = mk.rounded_delta(x, fgsm)
    r_p = mk.rounded_delta(x, pgd_float - x)
    st = mk.structure(r_f, r_p)
    s_f, s_p = mk.support(r_f), mk.support(r_p)
    outside = s_p & ~s_f
    tie = ((pgd_float - x).abs() - 0.5 * EPS).abs() < 1e-3 * EPS
    st["outside_all_half_level_ties"] = bool((tie | ~outside).all())
    st["outside_where_clean_grad_zero"] = int((outside & (g0 == 0)).sum())
    k = st["n_pgd_support"]
    g1, g2 = grads[1], grads[2]
    align = {"cos_g0_g1": geo.cosine(g0, g1), "sign_g0_g1": geo.sign_agreement(g0, g1),
             "cos_g0_g2": geo.cosine(g0, g2), "sign_g0_g2": geo.sign_agreement(g0, g2),
             "cos_g1_g2": geo.cosine(g1, g2), "sign_g1_g2": geo.sign_agreement(g1, g2)}
    rows: list[dict[str, Any]] = [{"kind": "structure", **base, **st, "alignment": align,
                                   "grad_norms": [geo.norms(g) for g in grads], "t_gradients_s": round(t_grad, 2)}]

    def scored(image, kind, **extra):
        stored = round_trip_uint8(image)
        r = stored - x
        row = {"kind": kind, **base, "k": int(mk.support(r).sum()), "pred_clean_gradient": geo.dot(g0, r),
               "scores": {}, **extra}
        for ctx in CONTEXTS:
            row["scores"][ctx] = {"official": m.score(image, prompts[ctx], label, "official"),
                                  "float": m.score(stored, prompts[ctx], label, "surrogate")}
        monitor.check()
        return row

    rows.append(scored(x + fgsm, "fgsm"))
    rows.append(scored(pgd_float, "pgd3"))
    rows.append(scored(mk.apply(x, r_f, mk.topk_mask(s_f, g0, k)), "topk_grad"))
    for s in range(N_SEEDS):
        seed = scene_seed(f"{SEED_PREFIX}|{source}|seed{s}", sid)
        row = scored(mk.apply(x, r_f, mk.random_mask(s_f, k, seed)), "random", seed_index=s, seed=seed)
        if row["k"] != k:
            raise Stop(f"{sid} {source}: random mask count {row['k']} != PGD-3 count {k}")
        rows.append(row)
    rows[0]["t_unit_s"] = round(time.perf_counter() - t0, 2)
    tensors = {f"{source}__{n}": c for n, c in codes.items()}
    tensors[f"{source}__g0"] = g0
    return rows, tensors


def run(dtype: str) -> None:
    from path_probe.memory import FOOTPRINT_ABORT_BYTES, MemoryMonitor, emp1_preload, os_sample, process_footprint

    from .backend import InternVL
    from .model_stage import user_text

    root = repo_root()
    commit = protocol_commit(root)
    out = root / OUT / dtype
    if (out / "records.jsonl").exists():
        raise Stop(f"{out}/records.jsonl exists. Single shot: no resume, no overwrite.")
    out.mkdir(parents=True, exist_ok=True)
    (out / "tensors").mkdir(exist_ok=True)
    (root / OUT / ".gitignore").write_text("*/tensors/\n*.log\n")
    scenes = arm_a(development_scenes(root))
    progress: dict[str, Any] = {
        "schema": "checkpoint-4.stage1.progress.v1", "label": "EXPLORATORY, DEVELOPMENT ONLY",
        "dtype": dtype, "tiling": TILING, "protocol_commit": commit, "git_head_at_start": git_head(root),
        "epsilon": EPS, "n_seeds": N_SEEDS, "seed_prefix": SEED_PREFIX, "sources_in_pass_order": SOURCES,
        "scene_order": [r["scene_id"] for r in scenes], "cap_seconds": CAP_S, "pid": os.getpid(),
        "confirmation_split_used": False, "units": {}, "memory_by_unit": [], "limitations": []}
    store_records = out / "records.jsonl"

    def save():
        (out / "progress.json").write_text(json.dumps(progress, indent=1, default=str) + "\n")

    pre = emp1_preload(os_sample(), shutil.disk_usage(root).free)
    progress["preload"] = pre
    if not pre["permitted"]:
        progress["status"] = "not started: preload lines " + ", ".join(pre["failed"])
        save(); raise Stop(progress["status"])
    torch.mps.set_per_process_memory_fraction(min(FOOTPRINT_ABORT_BYTES / torch.mps.recommended_max_memory(), 1.0))
    monitor = MemoryMonitor()
    monitor.start()
    t0 = time.perf_counter()
    try:
        m = InternVL(dtype, TILING)
        progress["load"] = m.load()
        progress["footprint_after_load"] = process_footprint()
        monitor.check()
        save()
        clean_first = None
        for source in SOURCES:
            for i, row in enumerate(scenes):
                if time.perf_counter() - t0 > CAP_S:
                    progress["limitations"].append(f"time cap reached before {row['scene_id']} source {source}")
                    raise Stop("time cap")
                sid, label = row["scene_id"], str(row["answer"])
                x = scene_image(root, row)
                prompts = {c: m.prompt(x, user_text(row, c)) for c in CONTEXTS}
                rows: list[dict[str, Any]] = []
                if source == SOURCES[0]:
                    clean = {"kind": "clean", "scene_id": sid, "label": label, "scores": {}}
                    for c in CONTEXTS:
                        clean["scores"][c] = {"official": m.score(x, prompts[c], label, "official"),
                                              "float": m.score(x, prompts[c], label, "surrogate")}
                    rows.append(clean)
                    if clean_first is None:
                        clean_first = (row, clean["scores"]["neutral"]["official"])
                unit, tensors = unit_block(m, x, prompts, label, sid, source, monitor)
                rows.extend(unit)
                with store_records.open("a", encoding="utf-8") as fh:
                    for r in rows:
                        fh.write(json.dumps(r, sort_keys=True) + "\n")
                import numpy as np
                np.savez_compressed(out / "tensors" / f"{sid}__{source}.npz",
                                    **{k: v.detach().cpu().numpy() for k, v in tensors.items()})
                fp = process_footprint() or {}
                snap = monitor.snapshot()
                progress["units"][f"{sid}|{source}"] = unit[0]["t_unit_s"]
                progress["memory_by_unit"].append({"unit": f"{sid}|{source}", "footprint_gib": round(fp.get("current", 0) / GIB, 3),
                                                   "lifetime_peak_gib": round(fp.get("lifetime_max", 0) / GIB, 3),
                                                   "min_available_gib": round((snap.get("min_available") or 0) / GIB, 3),
                                                   "max_swap_growth": snap.get("max_swap_growth"),
                                                   "pressure_levels": snap.get("by_level")})
                if len(progress["units"]) == 1:
                    progress["first_unit_memory"] = progress["memory_by_unit"][-1]
                progress["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
                save()
                print(f"[stage1 {dtype}] {source} {i + 1}/48 {sid} {unit[0]['t_unit_s']:.0f}s "
                      f"peak {progress['memory_by_unit'][-1]['lifetime_peak_gib']} GiB", flush=True)
        r0, first = clean_first
        x = scene_image(root, r0)
        again = m.score(x, m.prompt(x, user_text(r0, "neutral")), str(r0["answer"]), "official")
        progress["drift_check"] = {"scene": r0["scene_id"], "exact": again == first}
        progress["status"] = "completed"
    except Exception as exc:
        progress["status"] = "stopped"
        progress["stopped_reason"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        raise
    finally:
        monitor.stop()
        progress["memory_monitor"] = {k: v for k, v in monitor.snapshot().items() if k != "timeline"}
        fp = process_footprint() or {}
        progress["lifetime_peak_footprint_gib"] = round(fp.get("lifetime_max", 0) / GIB, 3)
        progress["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
        save()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("run",))
    ap.add_argument("--dtype", choices=("bf16", "fp32"), required=True)
    a = ap.parse_args()
    run(a.dtype)


if __name__ == "__main__":
    main()
