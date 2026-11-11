"""Checkpoint 3b partition-and-union pilot runner (docs/checkpoint-3b-partition/PLAN.md).

``python -m path_probe.partition_stage smoke --tensors DIR``  one s16 development scene, with bit-exact
    re-score checks against the 2.3 / 2.3b records, then that scene's planned images, and timing.
``python -m path_probe.partition_stage run --tensors DIR``    the 16 s16 development scenes.

``DIR`` holds the stored Checkpoint 2.3 tensors (``<sid>__B__grads.npz``, ``<sid>__B__neutral__deltas.npz``),
which are git-ignored. Neutral source, neutral scoring context only. No new gradients.
Reads development fixtures and stored 2.3 / 2.3b outputs only; refuses the confirmation split.
Numerics, scorer and memory policy are the frozen ones (bf16 on MPS, EMP-1 / M1).
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from . import partition as pt
from .run import Stop

EPS = 1.0 / 255.0
LEVEL = 1.0 / 255.0
PLAN = Path("docs/checkpoint-3b-partition/PLAN.md")
RUN_OUT = Path("docs/checkpoint-3b-partition/run")
SMOKE_OUT = Path("docs/checkpoint-3b-partition/run_smoke")
SEED = "checkpoint-3b-partition"
SOURCE = pt.CONTEXT  # neutral source
CONTEXT = pt.CONTEXT  # neutral scoring context
CAP_S = 45 * 60
KEEP = ("nll", "margin", "chosen", "summed_nll", "nll32", "margin32")
FORBIDDEN = ("trackB_confirm", "trackB-confirm")
FORWARDS_PER_SCENE = 1 + 1 + 1 + pt.N_PGD_SPLITS * pt.N_QUARTERS + pt.N_RANDOM * (1 + pt.N_QUARTERS)  # 51


def _keep(score: dict[str, Any]) -> dict[str, Any]:
    out = {k: score.get(k) for k in KEEP}
    out["summed_nll"] = {c: float(v) for c, v in score["summed_nll"].items()}
    return out


def _plan_commit(root: Path) -> str:
    dirty = subprocess.run(["git", "status", "--porcelain", "--", str(PLAN)], cwd=root, capture_output=True, text=True).stdout
    if dirty.strip():
        raise Stop("PLAN.md has uncommitted changes")
    commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", str(PLAN)], cwd=root, text=True).strip()
    if not commit:
        raise Stop("PLAN.md is not committed")
    return commit


def _same(mine: dict, theirs: dict) -> bool:
    return all(mine[f] == theirs[f] for f in ("nll", "margin", "chosen"))


def partition_block(backend, x, sid: str, stored: dict, ref: dict, smoke: bool = False) -> list[dict[str, Any]]:
    """All planned images for one scene, scored in the neutral context. Returns rows.

    ``ref`` holds the recorded 2.3 / 2.3b official scores of this scene (``confirm_stage._dev_reference``).
    Raises ``Stop`` on any failed check; the rows produced so far are attached to the exception."""
    import torch
    from attack_feasibility.steps import project_signed_step, scene_seed

    from . import geometry as geo
    from . import masks as mk

    rows: list[dict[str, Any]] = []
    n_forward = 0

    def stop(msg: str):
        exc = Stop(f"{sid}: {msg}")
        exc.rows = rows  # type: ignore[attr-defined]
        raise exc

    def seed(name: str) -> int:
        return scene_seed(f"{SEED}|{name}", sid)

    def score(image) -> dict[str, Any]:
        nonlocal n_forward
        n_forward += 1
        s = _keep(backend.score_official(image, CONTEXT))
        try:
            pt.identity_check(s, backend.label)
        except pt.PartitionError as exc:
            stop(f"identity check: {exc}")
        return s

    grads, codes = stored["grads"], stored["codes"]
    fgsm = project_signed_step(x, grads[f"L_{SOURCE}"], EPS, "positive") - x
    if not torch.equal(geo.encode_delta(fgsm, EPS), codes["fgsm"]):
        stop("FGSM from the stored gradient does not match the stored FGSM iterate")
    r_f = mk.rounded_delta(x, fgsm)
    pgd = mk.replay_pgd(x, codes, 3, EPS)
    r_p = mk.rounded_delta(x, pgd - x)
    try:
        st = pt.pgd_structure(r_f, r_p, LEVEL)
    except pt.PartitionError as exc:
        stop(str(exc))
    P, s_f = st.pop("P"), st.pop("S_F")
    k = st["k_retained"]
    base = {"kind": "image", "scene_id": sid, "label": backend.label}

    clean = score(x)
    rows.append({**base, "family": "clean", "draw": 0, "piece": "clean", "count": 0, "n_pos": 0, "n_neg": 0,
                 "scores": {CONTEXT: clean}})
    checks: dict[str, bool] = {}
    official = score(pgd)
    checks["pgd3_official_vs_2.3b"] = _same(official, ref["pgd3"][SOURCE][CONTEXT])
    rows.append({"kind": "pgd_official", "scene_id": sid, "label": backend.label, "scores": {CONTEXT: official},
                 "bit_exact_vs_record": checks["pgd3_official_vs_2.3b"]})
    if smoke:
        checks["clean_vs_2.3"] = _same(clean, ref["clean"][CONTEXT])
        k_all = st["n_pgd_support"]
        r0 = mk.random_mask(s_f, k_all, scene_seed(f"checkpoint-2.3b-mask|{SOURCE}|seed0", sid))
        checks["random_seed0_vs_2.3b"] = _same(score(mk.apply(x, r_f, r0)), ref["random"][(SOURCE, 0)][CONTEXT])
    rows.append({"kind": "checks", "scene_id": sid, **checks, "all_bit_exact": all(checks.values())})
    if not all(checks.values()):
        stop(f"re-scores are not bit-exact against the 2.3 / 2.3b records: {checks}")

    def masked(family: str, draw: int, piece: str, M) -> None:
        # ``draw``: the random seed index for random masks; the split index for PGD quarters (PGD full is 0).
        y = mk.apply(x, r_f, M)
        try:
            pt.image_change_check(x, y, r_f, M, LEVEL)
        except pt.PartitionError as exc:
            stop(f"{family} {draw} {piece}: {exc}")
        rows.append({**base, "family": family, "draw": draw, "piece": piece, **pt.sign_counts(r_f, M),
                     "scores": {CONTEXT: score(y)}})

    def full_and_quarters(family: str, draw: int, M, split_seeds: list[int], full: bool = True) -> None:
        if int(M.sum()) != k or bool((M & ~s_f).any()):
            stop(f"{family} {draw}: mask count {int(M.sum())} != k {k} or outside S_F")
        if full:
            masked(family, draw, "full", M)
        for a, sd in enumerate(split_seeds):
            pieces = pt.split(M, sd)
            try:
                pt.check_split(M, pieces)
            except pt.PartitionError as exc:
                stop(f"{family} {draw} split {a}: {exc}")
            for j, piece in enumerate(pieces):
                masked(family, draw + a, str(j), piece)

    full_and_quarters("pgd", 0, P, [seed(f"split|pgd|{a}") for a in range(pt.N_PGD_SPLITS)])
    for s in range(pt.N_RANDOM):
        R = mk.random_mask(s_f, k, seed(f"random|{s}"))
        full_and_quarters("random", s, R, [seed(f"split|random|{s}")])
    rows.append({"kind": "structure", "scene_id": sid, "source": SOURCE, "context": CONTEXT, **st,
                 "fgsm_sign_counts": pt.sign_counts(r_f, s_f), "pgd_sign_counts": pt.sign_counts(r_f, P),
                 "quarter_sizes": pt.quarter_sizes(k), "n_forward": n_forward})
    want = FORWARDS_PER_SCENE + (1 if smoke else 0)
    if n_forward != want:
        stop(f"{n_forward} forwards, planned {want}")
    return rows


def _load_npz(path: Path) -> tuple[dict[str, Any], str]:
    import numpy as np
    import torch

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path) as data:
        return {k: torch.from_numpy(data[k].copy()) for k in data.files}, digest


def execute(root: Path, smoke: bool, tensors: Path) -> None:
    import torch
    from attack_calibration.scene import _development_image
    from attack_feasibility.score import prepare_prompt, release_cache
    from choice_eval.prompt import build_prompt
    from dev_pilot.fixtures import load_manifest
    from dev_pilot.rows import user_text_sha256
    from qwen_rgb.measure import prepare_environment
    from qwen_rgb_loss.offline import load_predictions
    from qwen_rgb_loss.protocol import MODEL_KEY
    from vlm_iface.devices import memory_snapshot

    from .backend import QwenBackend
    from .confirm_stage import _dev_reference, _load_model
    from .memory import process_footprint
    from .precision import Fp32Projection
    from .probe import CONTEXTS
    from .run import Store, select_scenes
    from .stages import _check_residents

    out_rel = SMOKE_OUT if smoke else RUN_OUT
    if (root / out_rel / "records.jsonl").exists():
        raise Stop(f"{out_rel}/records.jsonl exists. Single shot: no resume, no overwrite.")
    if any(f in str(tensors) for f in FORBIDDEN):
        raise Stop("the tensor directory names the confirmation split")
    _check_residents(root)
    plan_commit = _plan_commit(root)
    dev = {r["scene_id"]: r for r in load_manifest(root)}
    s16, _ = select_scenes(list(dev.values()))
    sids = s16[:1] if smoke else s16
    for sid in sids:
        r = dev[sid]
        if r.get("split") != "development" or any(f in json.dumps(r) for f in FORBIDDEN):
            raise Stop(f"{sid} is not a development scene")
        for name in (f"{sid}__B__grads.npz", f"{sid}__B__{SOURCE}__deltas.npz"):
            if not (tensors / name).is_file():
                raise Stop(f"missing stored tensor {tensors / name}")
    stored_preds = {(r["scene_id"], r["context"]): r for r in load_predictions(root)}

    store = Store(root, out_rel)
    progress: dict[str, Any] = {
        "schema": "checkpoint-3b.partition.progress.v1",
        "label": ("SMOKE, one development scene" if smoke else
                  "PILOT, 16 development scenes (s16), neutral source and context; EXPLORATORY"),
        "plan_commit": plan_commit,
        "git_head_at_start": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "scene_order": sids, "planned_scenes": s16 if not smoke else sids, "cap_seconds": CAP_S,
        "forwards_planned": len(sids) * FORWARDS_PER_SCENE + (1 if smoke else 0) + 1,
        "tensor_dir": str(tensors), "tensor_sha256": {}, "environment": prepare_environment(), "pid": os.getpid(),
        "memory_policy": "EMP-1 with M1 (unchanged)", "confirmation_split_used": False, "reserved_h_samples_used": False,
        "scenes": {}, "limitations": [], "status": "started",
    }
    (root / out_rel / "resident.json").write_text(json.dumps({"pid": os.getpid(), "model_key": MODEL_KEY}) + "\n")
    store.save_progress(progress)
    model, processor, image_processor, token_ids, monitor = _load_model(progress, store, root, out_rel)
    device = torch.device("mps")
    projection = Fp32Projection(model.lm_head, token_ids)
    t0 = time.perf_counter()

    def backend_for(sid, record, image):
        texts, prompts = {}, {}
        for ctx in CONTEXTS:
            text = build_prompt(record["contexts"][ctx]["text"], record["question"]).user_text()
            if stored_preds[(sid, ctx)].get("user_text_sha256") != user_text_sha256(text):
                raise Stop(f"{sid} {ctx} prompt hash does not match the stored row")
            texts[ctx] = text
            prompts[ctx] = prepare_prompt(processor, image_processor, image, text)
        return QwenBackend(model=model, processor=processor, image_processor=image_processor, prompts=prompts, texts=texts,
                           token_ids=token_ids, label=str(record["answer"]), device=device, projection=projection, guard=monitor.check)

    try:
        durations: list[float] = []
        for i, sid in enumerate(sids):
            mean = sum(durations) / len(durations) if durations else 0.0
            if time.perf_counter() - t0 + mean > CAP_S:
                progress["truncated_by_cap"] = sids[i:]
                progress["limitations"].append(f"time cap: {len(sids) - i} scenes not measured; output INCOMPLETE")
                break
            started = time.perf_counter()
            record = dev[sid]
            x = _development_image(root, record)
            grads, hg = _load_npz(tensors / f"{sid}__B__grads.npz")
            codes, hc = _load_npz(tensors / f"{sid}__B__{SOURCE}__deltas.npz")
            progress["tensor_sha256"][sid] = {"grads": hg, "deltas": hc}
            try:
                rows = partition_block(backend_for(sid, record, x), x, sid, {"grads": grads, "codes": codes},
                                       _dev_reference(root, sid), smoke=smoke)
            except Stop as exc:
                store.append(getattr(exc, "rows", []))
                raise
            store.append(rows)
            release_cache(device)
            monitor.check()
            durations.append(time.perf_counter() - started)
            progress["scenes"][sid] = round(durations[-1], 2)
            progress["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
            progress["memory_monitor"] = monitor.snapshot()
            progress["last_memory"] = {"tag": sid, **memory_snapshot(device), "footprint": process_footprint()}
            store.save_progress(progress)
            print(f"[3b] {i + 1}/{len(sids)} {sid} {durations[-1]:.0f}s elapsed {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        progress["n_measured"] = len(durations)
        progress["mean_scene_seconds"] = round(sum(durations) / max(1, len(durations)), 2)
        if smoke and durations:
            progress["runtime_estimate_16_scenes_min"] = round(16 * durations[0] / 60, 1)
        # Final clean re-score of the first scene (one forward), compared on all four colour scores.
        sid = sids[0]
        x = _development_image(root, dev[sid])
        again = _keep(backend_for(sid, dev[sid], x).score_official(x, CONTEXT))
        first = json.loads(store.records.read_text().splitlines()[0])["scores"][CONTEXT]
        exact = again["summed_nll"] == first["summed_nll"] and again["margin"] == first["margin"] and again["chosen"] == first["chosen"]
        progress["drift_check"] = {"scene": sid, "exact": exact}
        if progress.get("truncated_by_cap"):
            progress["status"] = "incomplete_time_cap"
        elif not exact:
            progress["status"] = "drift_check_failed"
        else:
            progress["status"] = "completed"
    except Stop as exc:
        progress["status"] = "stopped"
        progress["stopped_reason"] = str(exc)
        raise
    except Exception as exc:
        progress["status"] = "error"
        progress["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        progress["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
        monitor.stop()
        progress["memory_monitor"] = monitor.snapshot()
        progress["footprint_end"] = process_footprint()
        projection.remove()
        store.save_progress(progress)
        (root / out_rel / "resident.json").unlink(missing_ok=True)
        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m path_probe.partition_stage")
    ap.add_argument("mode", choices=("smoke", "run"))
    ap.add_argument("--tensors", required=True, type=Path, help="directory with the stored Checkpoint 2.3 tensors")
    args = ap.parse_args(argv)
    from .run import repo_root

    try:
        execute(repo_root(), smoke=args.mode == "smoke", tensors=args.tensors.resolve())
    except Stop as exc:
        raise SystemExit(f"stopped: {exc}") from exc


if __name__ == "__main__":
    main()
