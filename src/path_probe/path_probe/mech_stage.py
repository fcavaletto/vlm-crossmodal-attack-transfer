"""Checkpoint 3 development-only mechanism pilot (docs/checkpoint-3-mechanism/PLAN.md).

``python -m path_probe.mech_stage smoke``  one s16 development scene: bit-exact re-score checks
    against the 2.3/2.3b records, then the full per-scene pilot block, and timing.
``python -m path_probe.mech_stage run``       the 16 s16 development scenes, neutral source.

Reads development fixtures and stored 2.3/2.3b outputs only; refuses the confirmation split.
Numerics, gradient, scorer and memory policy are the frozen ones (bf16 on MPS, EMP-1 / M1).
Prints progress only.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .run import OUT_RELATIVE, Stop, Store, select_scenes

EPS = 1.0 / 255.0
PLAN = Path("docs/checkpoint-3-mechanism/PLAN.md")
PILOT_OUT = Path("docs/checkpoint-3-mechanism/pilot")
SMOKE_OUT = Path("docs/checkpoint-3-mechanism/pilot_smoke")
SEED = "checkpoint-3-pilot"
SOURCE = "neutral"
CAP_S = 120 * 60
N_R, N_Q, N_J = 8, 4, 2
FORBIDDEN = ("trackB_confirm", "trackB-confirm")


def _plan_commit(root: Path) -> str:
    dirty = subprocess.run(["git", "status", "--porcelain", "--", str(PLAN)], cwd=root, capture_output=True, text=True).stdout
    if dirty.strip():
        raise Stop("PLAN.md has uncommitted changes")
    commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", str(PLAN)], cwd=root, text=True).strip()
    if not commit:
        raise Stop("PLAN.md is not committed")
    return commit


def pilot_block(backend, x, sid: str, record: dict, stored: dict, smoke_ref: dict | None = None):
    """All pilot measurements for one scene (neutral source). Returns (rows, tensors)."""
    import torch
    from attack_feasibility.steps import project_signed_step, scene_seed

    from . import consensus as cs
    from . import geometry as geo
    from . import masks as mk
    from .probe import CONTEXTS, _keep

    def seed(name: str) -> int:
        return scene_seed(f"{SEED}|{name}", sid)

    grads0, codes, v32 = stored["grads"], stored["codes"], stored["v32"]
    g0 = grads0[f"L_{SOURCE}"]
    fgsm = project_signed_step(x, g0, EPS, "positive") - x
    if not torch.equal(geo.encode_delta(fgsm, EPS), codes["fgsm"]):
        raise Stop(f"{sid}: FGSM from the stored gradient does not match the stored FGSM iterate")
    r_f = mk.rounded_delta(x, fgsm)
    s_f = mk.support(r_f)
    pgd = mk.replay_pgd(x, codes, 3, EPS)
    s_p = mk.support(mk.rounded_delta(x, pgd - x))
    k = int(s_p.sum())
    P = s_p & s_f
    ref = cs.referent(record)
    tok_P = cs.token_density(P)

    rows: list[dict[str, Any]] = []
    clean = {"kind": "clean", "scene_id": sid, "label": backend.label, "scores": {}}
    for ctx in CONTEXTS:
        clean["scores"][ctx] = _keep(backend.score_official(x, ctx))
    rows.append(clean)

    if smoke_ref is not None:  # bit-exact re-scores against the frozen 2.3 / 2.3b records
        chk = {"clean": all(clean["scores"][c][f] == smoke_ref["clean"][c][f] for c in CONTEXTS for f in ("nll", "margin", "chosen"))}
        img = {"pgd3": pgd, "random0": mk.apply(x, r_f, mk.random_mask(s_f, k, scene_seed(f"checkpoint-2.3b-mask|{SOURCE}|seed0", sid)))}
        theirs = {"pgd3": smoke_ref["pgd3"][SOURCE], "random0": smoke_ref["random"][(SOURCE, 0)]}
        for name, image in img.items():
            mine = {c: _keep(backend.score_official(image, c)) for c in CONTEXTS}
            chk[name] = all(mine[c][f] == theirs[name][c][f] for c in CONTEXTS for f in ("nll", "margin", "chosen"))
        rows.append({"kind": "smoke_check", "scene_id": sid, **chk, "all_bit_exact": all(chk.values())})
        if not all(chk.values()):
            return rows, {}

    # New gradients at attack-independent points.
    pts = {}
    for m in range(N_R):
        pts[f"R{m}"] = cs.point_dense(x, EPS / 4, seed(f"R|{m}"))
    for m in range(N_Q):
        pts[f"Q{m}"] = cs.point_quantized(x, seed(f"Q|{m}"))
    for m in range(N_J):
        pts[f"J{m}"] = cs.point_dense(x, EPS / 16, seed(f"J|{m}"))
    g: dict[str, torch.Tensor] = {}
    for name, p in pts.items():
        t = time.perf_counter()
        g[name] = backend.grad_nll(p, SOURCE)
        rows.append({"kind": "gradient", "scene_id": sid, "point": name, "seconds": round(time.perf_counter() - t, 2),
                     "point_linf": float((p - x).abs().max()), **cs.geometry(g[name], g0)})

    R = [g[f"R{m}"] for m in range(N_R)]
    sign32 = torch.sign(v32["fgsm"].to(torch.float32))  # fp32-vision clean-gradient sign (via its FGSM iterate)
    masks: dict[str, tuple[torch.Tensor, int]] = {}
    for M in (1, 2, 4, 8):
        masks[f"vote_R{M}"] = (cs.vote_mask(r_f, s_f, [g0, *R[:M]], k, seed(f"tie|vote_R{M}")), 1 + M)
    for M in (2, 8):
        masks[f"smooth_R{M}"] = (cs.smooth_mask(r_f, s_f, [g0, *R[:M]], k), 1 + M)
    masks["vote_Q4"] = (cs.vote_mask(r_f, s_f, [g0, *[g[f"Q{m}"] for m in range(N_Q)]], k, seed("tie|vote_Q4")), 1 + N_Q)
    masks["vote_J2"] = (cs.vote_mask(r_f, s_f, [g0, *[g[f"J{m}"] for m in range(N_J)]], k, seed("tie|vote_J2")), 1 + N_J)
    masks["vote_P32"] = (cs.vote_mask(r_f, s_f, [g0, sign32], k, seed("tie|vote_P32")), 2)
    masks["pgd_patch_s0"] = (cs.patch_matched_random(P, s_f, seed("patch|pgd|0")), 3)
    masks["pgd_patch_s1"] = (cs.patch_matched_random(P, s_f, seed("patch|pgd|1")), 3)
    masks["voteR8_patch"] = (cs.patch_matched_random(masks["vote_R8"][0], s_f, seed("patch|voteR8")), 9)

    def describe(A):
        return {**cs.overlap(A, P, s_f), "token_spearman_vs_pgd": cs.spearman(cs.token_density(A), tok_P),
                "referent_enrichment": cs.referent_enrichment(A, ref)}

    def preds(r):
        return {f"M_{c}": geo.dot(grads0[f"M_{c}"], r) for c in CONTEXTS}

    def score(image):
        return {c: _keep(backend.score_official(image, c)) for c in CONTEXTS}

    base = {"scene_id": sid, "source": SOURCE}
    for name, (A, n_grad) in masks.items():
        if int(A.sum()) != k or bool((A & ~s_f).any()):
            raise Stop(f"{sid} {name}: mask count {int(A.sum())} != k {k} or outside S_F")
        r = r_f * A.to(r_f.dtype)
        rows.append({"kind": "mask", **base, "name": name, "k": k, "grad_evals": n_grad, **describe(A),
                     "pred_clean_gradient": preds(mk.rounded_delta(x, r)), "scores": score(x + r)})
    for M in (2, 8):
        mean = torch.stack([g0, *R[:M]]).mean(0)
        delta = project_signed_step(x, mean, EPS, "positive") - x
        rd = mk.rounded_delta(x, delta)
        rows.append({"kind": "attack", **base, "name": f"sfgsm_R{M}", "grad_evals": 1 + M,
                     "k": int(mk.support(rd).sum()),
                     "sign_agree_fgsm": float((torch.sign(rd[s_f]) == torch.sign(r_f[s_f])).float().mean()),
                     "pred_clean_gradient": preds(rd), "scores": score(x + delta)})

    # Reference masks (scored earlier in 2.3 / 2.3b): spatial description only.
    pgd32 = mk.replay_pgd(x, v32, 3, EPS)
    P32 = mk.support(mk.rounded_delta(x, pgd32 - x)) & s_f
    refs = {"pgd3": P, "topk_grad": mk.topk_mask(s_f, g0, k),
            "random0": mk.random_mask(s_f, k, scene_seed(f"checkpoint-2.3b-mask|{SOURCE}|seed0", sid)),
            "pgd3_fp32vision": P32}
    for name, A in refs.items():
        rows.append({"kind": "reference_mask", **base, "name": name, "count": int(A.sum()), **describe(A)})
    rows.append({"kind": "structure", **base, "k": k, "n_fgsm_support": int(s_f.sum()),
                 "fp32_fgsm_sign_agree_on_S_F": float((sign32[s_f] == torch.sign(r_f[s_f])).float().mean()),
                 "referent": ref["object_id"]})
    return rows, {name: t for name, t in g.items()}


def execute_pilot(root: Path, smoke: bool) -> None:
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
    from .stages import _check_residents

    out_rel = SMOKE_OUT if smoke else PILOT_OUT
    if (root / out_rel / "records.jsonl").exists():
        raise Stop(f"{out_rel}/records.jsonl exists. Single shot: no resume, no overwrite.")
    _check_residents(root)
    plan_commit = _plan_commit(root)
    dev = {r["scene_id"]: r for r in load_manifest(root)}
    s16, _ = select_scenes(list(dev.values()))
    sids = s16[:1] if smoke else s16
    for sid in sids:
        r = dev[sid]
        if r.get("split") != "development" or any(f in json.dumps(r) for f in FORBIDDEN):
            raise Stop(f"{sid} is not a development scene")
    src = Store(root, OUT_RELATIVE)
    stored_preds = {(r["scene_id"], r["context"]): r for r in load_predictions(root)}

    store = Store(root, out_rel)
    progress: dict[str, Any] = {
        "schema": "checkpoint-3.pilot.progress.v1",
        "label": "SMOKE, one development scene" if smoke else "PILOT, 16 development scenes (s16), neutral source; EXPLORATORY",
        "plan_commit": plan_commit, "git_head_at_start": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "scene_order": sids, "cap_seconds": CAP_S, "environment": prepare_environment(), "pid": os.getpid(),
        "memory_policy": "EMP-1 with M1 (unchanged)", "confirmation_split_used": False, "reserved_h_samples_used": False,
        "scenes": {}, "limitations": [],
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
                progress["limitations"].append(f"time cap: {len(sids) - i} scenes not measured")
                break
            started = time.perf_counter()
            record = dev[sid]
            x = _development_image(root, record)
            stored = {"grads": src.load_tensors(f"{sid}__B__grads"), "codes": src.load_tensors(f"{sid}__B__{SOURCE}__deltas"),
                      "v32": src.load_tensors(f"{sid}__V32__{SOURCE}__deltas")}
            ref = _dev_reference(root, sid) if smoke else None
            rows, tensors = pilot_block(backend_for(sid, record, x), x, sid, record, stored, ref)
            store.append(rows)
            if tensors:
                store.save_tensors(sid, tensors)
            release_cache(device)
            monitor.check()
            durations.append(time.perf_counter() - started)
            progress["scenes"][sid] = round(durations[-1], 2)
            progress["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
            progress["memory_monitor"] = monitor.snapshot()
            progress["last_memory"] = {"tag": sid, **memory_snapshot(device), "footprint": process_footprint()}
            store.save_progress(progress)
            print(f"[pilot] {i + 1}/{len(sids)} {sid} {durations[-1]:.0f}s elapsed {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
            if smoke:
                chk = next(r for r in rows if r["kind"] == "smoke_check")
                progress["smoke_check"] = chk
                if not chk["all_bit_exact"]:
                    raise Stop("smoke re-scores are not bit-exact against the 2.3/2.3b records")
        progress["n_measured"] = len(durations)
        progress["mean_scene_seconds"] = round(sum(durations) / max(1, len(durations)), 2)
        if smoke:
            progress["runtime_estimate_16_scenes_min"] = round(16 * durations[0] / 60, 1)
        sid = sids[0]
        x = _development_image(root, dev[sid])
        again = backend_for(sid, dev[sid], x).score_official(x, "neutral")
        first = json.loads(store.records.read_text().splitlines()[0])["scores"]["neutral"]
        progress["drift_check"] = {"scene": sid, "exact": again["nll"] == first["nll"] and again["margin"] == first["margin"]}
        progress["status"] = "completed" if not progress.get("truncated_by_cap") else "truncated_by_cap"
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
    """Own entry point, so the files frozen for Checkpoint 2.4 (run.py among them) stay untouched."""
    import sys
    args = sys.argv[1:] if argv is None else argv
    if args not in (["smoke"], ["run"]):
        raise SystemExit("usage: python -m path_probe.mech_stage {smoke|run}")
    from .run import repo_root
    root = repo_root()
    try:
        execute_pilot(root, smoke=args == ["smoke"])
    except Stop as exc:
        raise SystemExit(f"stopped: {exc}") from exc


if __name__ == "__main__":
    main()
