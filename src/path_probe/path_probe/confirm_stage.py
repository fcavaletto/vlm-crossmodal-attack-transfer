"""Checkpoint 2.4: held-out confirmation of the 2.3b mask-control result.

``python -m path_probe confirm-smoke-dev``  one DEVELOPMENT scene through the confirmation
    pipeline, checked bit-exactly against the recorded 2.3 and 2.3b scores. Timing estimate.
    Never reads the confirmation split.
``python -m path_probe confirm``            the frozen confirmation run on
    ``fixtures/trackB_confirm_v1`` (arm a of 48 groups). Refuses unless every frozen file
    matches ``docs/checkpoint-2.4-confirm/FROZEN.json`` and that file is committed.

Numerics are the frozen ones: bf16 weights on MPS, eager attention, the frozen load call,
the frozen gradient (``backward_at_image``), step (``project_signed_step``, ``pgd_update``)
and official scorer (``score_official`` with the 8-bit round trip). Memory policy: EMP-1 with
the M1 thresholds, unchanged. The run prints progress only, never an effect.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .run import OUT_RELATIVE, Stop, Store, select_scenes

EPS = 1.0 / 255.0
N_SEEDS = 8
CONFIRM_DIR = Path("docs/checkpoint-2.4-confirm")
FROZEN = CONFIRM_DIR / "FROZEN.json"
CONFIRM_OUT = CONFIRM_DIR / "run"
SMOKE_OUT = CONFIRM_DIR / "smoke_dev"
FIXTURE = Path("fixtures/trackB_confirm_v1")
CONFIRM_SEED_PREFIX = "checkpoint-2.4-confirm-mask"
DEV_SEED_PREFIX = "checkpoint-2.3b-mask"  # smoke only: reproduces the 2.3b random masks
CAP_S = 180 * 60
N_GROUPS = 48


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_files(root: Path) -> dict[str, str]:
    return json.loads((root / FROZEN).read_text())["files"]


def verify_frozen(root: Path) -> dict[str, Any]:
    """Every listed file hashes to its frozen value, and FROZEN.json is committed and clean."""
    if not (root / FROZEN).is_file():
        raise Stop("FROZEN.json is absent: the protocol is not frozen")
    paths = [str(FROZEN), *frozen_files(root)]
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *paths], cwd=root, capture_output=True, text=True).stdout
    if dirty.strip():
        raise Stop("a frozen file has uncommitted changes:\n" + dirty)
    bad = {p: h for p, h in frozen_files(root).items() if sha256(root / p) != h}
    if bad:
        raise Stop(f"frozen files changed since the freeze: {sorted(bad)}")
    commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", str(FROZEN)], cwd=root, text=True).strip()
    if not commit:
        raise Stop("FROZEN.json is not committed")
    return {"frozen_commit": commit, "n_files": len(frozen_files(root))}


def confirm_records(root: Path) -> list[dict[str, Any]]:
    """Primary arm (a) of each of the 48 groups, in group order."""
    path = root / FIXTURE / "manifest.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    prim = sorted((r for r in rows if r["primary_arm"]), key=lambda r: r["group_index"])
    if len(prim) != N_GROUPS or [r["group_index"] for r in prim] != list(range(N_GROUPS)):
        raise Stop("the confirmation manifest does not give arm a of groups 0..47")
    if any(r["split"] != "trackB-confirm" or not r["scene_id"].endswith("-a") for r in prim):
        raise Stop("a primary row is not a trackB-confirm arm a scene")
    return prim


def scene_block(backend, x, sid: str, seed_prefix: str, n_seeds: int = N_SEEDS):
    """Clean scores and gradients, then per source: FGSM, PGD-3, structure, top-k and random masks.

    Returns (rows, tensors). Every perturbed image is scored with the official scorer in both contexts.
    """
    import torch
    from attack_feasibility.steps import pgd_update, project_signed_step, scene_seed

    from . import geometry as geo
    from . import masks as mk
    from .probe import CONTEXTS, _keep, clean_block, other

    clean, grads = clean_block(backend, x)
    clean.update({"scene_id": sid})
    rows: list[dict[str, Any]] = [clean]
    tensors: dict[str, Any] = {f"grad_{k}": v for k, v in grads.items()}

    def preds(r):
        return {f"{g}_{c}": geo.dot(grads[f"{g}_{c}"], r) for g in ("L", "M") for c in CONTEXTS}

    def scored(image, kind, base, **extra):
        r = image - x
        row = {"kind": kind, **base, "k": int(mk.support(mk.rounded_delta(x, r)).sum()),
               "pred_clean_gradient": preds(mk.rounded_delta(x, r)), "scores": {}, **extra}
        for ctx in CONTEXTS:
            row["scores"][ctx] = _keep(backend.score_official(image, ctx))
        return row

    for source in CONTEXTS:
        base = {"scene_id": sid, "source": source, "target": other(source)}
        g0 = grads[f"L_{source}"]
        fgsm = project_signed_step(x, g0, EPS, "positive") - x
        codes = {"fgsm": geo.encode_delta(fgsm, EPS)}
        current, g_prev = x.detach().clone(), g0
        for k in range(1, 4):
            if k > 1:
                g_prev = backend.grad_nll(current, source)
            current = pgd_update(current, x, g_prev, EPS / 4.0, EPS)
            codes[f"pgd{k}"] = geo.encode_delta(current - x, EPS)
        pgd_float = current
        tensors.update({f"{source}__{n}": c for n, c in codes.items()})

        r_f = mk.rounded_delta(x, fgsm)
        r_p = mk.rounded_delta(x, pgd_float - x)
        st = mk.structure(r_f, r_p)
        s_f, s_p = mk.support(r_f), mk.support(r_p)
        outside = s_p & ~s_f
        tie = ((pgd_float - x).abs() - 0.5 * EPS).abs() < 1e-3 * EPS
        st["outside_all_half_level_ties"] = bool((tie | ~outside).all())
        st["outside_where_clean_grad_zero"] = int((outside & (g0 == 0)).sum())
        k = st["n_pgd_support"]
        rows.append({"kind": "structure", **base, **st})
        rows.append(scored(x + fgsm, "fgsm", base))
        rows.append(scored(pgd_float, "pgd3", base))
        rows.append(scored(mk.apply(x, r_f, mk.topk_mask(s_f, g0, k)), "topk_grad", base))
        for s in range(n_seeds):
            seed = scene_seed(f"{seed_prefix}|{source}|seed{s}", sid)
            row = scored(mk.apply(x, r_f, mk.random_mask(s_f, k, seed)), "random", base, seed_index=s, seed=seed)
            if row["k"] != k:
                raise Stop(f"{sid} {source}: random mask count {row['k']} != PGD-3 count {k}")
            rows.append(row)
        del s_p
    return rows, tensors


def _load_model(progress, store, root, out_rel):
    """The guarded EMP-1 load, as in mask_stage.execute_masks."""
    import torch
    from dev_pilot.scoring import assert_locked_tokenization
    from qwen_rgb.budget import dtype_decisions, load_budget_module
    from qwen_rgb.measure import bf16_probe, load_model_after_budget, load_processor, read_local_header, weight_file
    from qwen_rgb.pins import MODEL_ID, PINNED_REVISION
    from qwen_rgb.rgb_path import assert_pinned_processor
    from qwen_rgb_loss.protocol import MODEL_KEY
    from vlm_iface.devices import available_bytes

    from .memory import MemoryMonitor, emp1_preload, os_sample, process_footprint

    processor, image_processor = load_processor()
    assert_pinned_processor(image_processor)
    identity = assert_locked_tokenization(processor, MODEL_KEY)
    token_ids = {c: tuple(int(t) for t in ids) for c, ids in identity["token_ids"].items()}
    if token_ids != {"red": (1151,), "blue": (12203,), "green": (13250,), "yellow": (27869,)}:
        raise Stop("the color token ids are not the locked ids")
    budget = load_budget_module(root)
    probe = bf16_probe(torch)
    weights = weight_file()
    if not weights.is_file():
        raise Stop("the local safetensors file is absent; this script does not download it")
    header = read_local_header(weights)
    if header["parameter_count"] != budget.QWEN_C0_PARAMETERS or header["dtype_counts"] != {"BF16": header["parameter_count"]}:
        raise Stop("the weight header is not the Checkpoint 1.5 Qwen BF16 header")
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    available = available_bytes()
    decisions = dtype_decisions(budget, parameter_count=header["parameter_count"],
                                float32_weight_bytes=header["float32_weight_bytes"], available_bytes=available,
                                mps_available=bool(torch.backends.mps.is_available()))
    progress["budget"] = {"available_bytes": available, "decisions": decisions, "bf16_probe": probe, "guard_lowered": False}
    emp = emp1_preload(os_sample(), shutil.disk_usage(root).free)
    progress["emp1_preload"] = emp
    if not (emp["permitted"] and probe.get("matmul_ok")):
        progress["status"] = "stopped_budget_emp1"
        store.save_progress(progress)
        (root / out_rel / "resident.json").unlink(missing_ok=True)
        raise Stop("EMP-1 did not permit a load: " + ", ".join(emp["failed"]))
    monitor = MemoryMonitor()
    monitor.start()
    progress["footprint_before_load"] = process_footprint()
    permit = {"permitted": True, "policy": "EMP-1",
              "frozen_bf16_decision_permitted": bool(decisions["bfloat16"].get("permitted")), "emp1_preload": emp}
    progress["load_permit"] = permit
    store.save_progress(progress)
    t_load = time.perf_counter()
    try:
        model = load_model_after_budget(permit, "bfloat16")
        model.to(torch.device("mps"))
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        param = next(model.parameters())
        if param.device.type != "mps" or param.dtype != torch.bfloat16:
            raise Stop("the model is not bf16 on MPS")
        if getattr(model.config, "_commit_hash", None) not in (None, PINNED_REVISION):
            raise Stop("the loaded revision is not the pinned one")
    except BaseException as exc:
        progress["status"] = "stopped_load"
        progress["error"] = f"{type(exc).__name__}: {exc}"
        monitor.stop()
        progress["memory_monitor"] = monitor.snapshot()
        store.save_progress(progress)
        (root / out_rel / "resident.json").unlink(missing_ok=True)
        raise
    progress["loaded_model"] = {"model_id": MODEL_ID, "revision": PINNED_REVISION, "dtype": str(param.dtype),
                                "load_seconds": round(time.perf_counter() - t_load, 3)}
    return model, processor, image_processor, token_ids, monitor


def _dev_reference(root: Path, sid: str) -> dict[str, Any]:
    """Recorded 2.3 (FGSM, PGD-3, clean) and 2.3b (masks) official scores for one dev scene."""
    ref: dict[str, Any] = {"clean": None, "fgsm": {}, "pgd3": {}, "topk_grad": {}, "random": {}}
    for line in (root / OUT_RELATIVE / "records.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("scene_id") != sid or r.get("mode") != "B":
            continue
        if r["kind"] == "clean":
            ref["clean"] = {c: r["scores"][c]["official"] for c in r["scores"]}
        elif r["kind"] == "fgsm":
            ref["fgsm"][r["source"]] = {c: r["scores"][c]["official"] for c in r["scores"]}
        elif r["kind"] == "pgd" and r.get("k") == 3:
            ref["pgd3"][r["source"]] = {c: r["scores"][c]["official"] for c in r["scores"]}
    for line in (root / "docs/checkpoint-2.3-exploratory/masks/records.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r.get("scene_id") != sid or r["kind"] != "mask":
            continue
        if r["mask"] == "topk_grad":
            ref["topk_grad"][r["source"]] = r["scores"]
        elif r["mask"] == "random":
            ref["random"][(r["source"], r["seed_index"])] = r["scores"]
    return ref


def _compare(mine: dict, theirs: dict) -> bool:
    return all(mine[c][f] == theirs[c][f] for c in ("neutral", "helpful_nonrevealing") for f in ("nll", "margin", "chosen"))


def execute_confirm(root: Path, smoke_dev: bool) -> None:
    import torch
    from attack_feasibility.score import prepare_prompt, release_cache
    from choice_eval.prompt import build_prompt
    from dev_pilot.rows import user_text_sha256
    from qwen_rgb.measure import prepare_environment
    from qwen_rgb_loss.protocol import MODEL_KEY
    from vlm_iface.devices import memory_snapshot
    from vlm_iface.images import to_image_01

    from .backend import QwenBackend
    from .memory import process_footprint
    from .precision import Fp32Projection
    from .probe import CONTEXTS

    out_rel = SMOKE_OUT if smoke_dev else CONFIRM_OUT
    if (root / out_rel / "records.jsonl").exists():
        raise Stop(f"{out_rel}/records.jsonl exists. This run is single-shot: no resume, no overwrite.")
    from .stages import _check_residents
    _check_residents(root)
    if (root / CONFIRM_OUT / "resident.json").is_file() or (root / SMOKE_OUT / "resident.json").is_file():
        raise Stop("a confirmation resident.json exists")

    if smoke_dev:
        from attack_calibration.scene import _development_image
        from dev_pilot.fixtures import load_manifest
        from qwen_rgb_loss.offline import load_predictions
        dev = {r["scene_id"]: r for r in load_manifest(root)}
        _, order = select_scenes(list(dev.values()))
        sid0 = order[0]
        scenes = [(sid0, dev[sid0], _development_image(root, dev[sid0]))]
        stored = {(r["scene_id"], r["context"]): r for r in load_predictions(root)}
        seed_prefix, freeze = DEV_SEED_PREFIX, None
    else:
        freeze = verify_frozen(root)
        prim = confirm_records(root)
        scenes = [(r["scene_id"], r, to_image_01(root / FIXTURE / r["image_path"])) for r in prim]
        stored, seed_prefix = None, CONFIRM_SEED_PREFIX

    store = Store(root, out_rel)
    progress: dict[str, Any] = {
        "schema": "checkpoint-2.4.confirm.progress.v1",
        "label": ("SMOKE ON ONE DEVELOPMENT SCENE (pipeline reproduction and timing; not a result)" if smoke_dev
                  else "CONFIRMATION RUN, trackB-confirm-v1, single shot"),
        "freeze": freeze, "git_head_at_start": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "scene_order": [s for s, _, _ in scenes], "seed_prefix": seed_prefix, "n_seeds": N_SEEDS, "epsilon": EPS,
        "cap_seconds": CAP_S, "environment": prepare_environment(), "pid": os.getpid(),
        "memory_policy": "EMP-1 with M1 (unchanged; MEMORY_ASSESSMENT.md section 7)",
        "earlier_outputs_written": False, "reserved_h_samples_used": False, "scenes": {}, "limitations": [],
    }
    (root / out_rel / "resident.json").write_text(json.dumps({"pid": os.getpid(), "model_key": MODEL_KEY}) + "\n")
    store.save_progress(progress)
    model, processor, image_processor, token_ids, monitor = _load_model(progress, store, root, out_rel)
    device = torch.device("mps")
    projection = Fp32Projection(model.lm_head, token_ids)
    t0 = time.perf_counter()

    def elapsed() -> float:
        return time.perf_counter() - t0

    def backend_for(sid, record, image):
        texts, prompts = {}, {}
        for ctx in CONTEXTS:
            text = build_prompt(record["contexts"][ctx]["text"], record["question"]).user_text()
            if stored is not None and stored[(sid, ctx)].get("user_text_sha256") != user_text_sha256(text):
                raise Stop(f"{sid} {ctx} prompt hash does not match the stored row")
            texts[ctx] = text
            prompts[ctx] = prepare_prompt(processor, image_processor, image, text)
        return texts, QwenBackend(model=model, processor=processor, image_processor=image_processor, prompts=prompts,
                                  texts=texts, token_ids=token_ids, label=str(record["answer"]), device=device,
                                  projection=projection, guard=monitor.check)

    try:
        durations: list[float] = []
        for i, (sid, record, x) in enumerate(scenes):
            mean = sum(durations) / len(durations) if durations else 0.0
            if elapsed() + mean > CAP_S:
                progress["truncated_by_cap"] = [s for s, _, _ in scenes[i:]]
                progress["limitations"].append(f"time cap: {len(scenes) - i} groups not measured")
                break
            started = time.perf_counter()
            texts, backend = backend_for(sid, record, x)
            rows, tensors = scene_block(backend, x, sid, seed_prefix)
            rows[0]["user_text_sha256"] = {c: user_text_sha256(t) for c, t in texts.items()}
            rows[0]["label"] = str(record["answer"])
            store.append(rows)
            store.save_tensors(sid, tensors)
            release_cache(device)
            monitor.check()
            durations.append(time.perf_counter() - started)
            progress["scenes"][sid] = round(durations[-1], 2)
            progress["elapsed_seconds"] = round(elapsed(), 1)
            progress["memory_monitor"] = monitor.snapshot()
            progress["last_memory"] = {"tag": sid, **memory_snapshot(device), "footprint": process_footprint()}
            store.save_progress(progress)
            print(f"[confirm] {i + 1}/{len(scenes)} {sid} {durations[-1]:.0f}s elapsed {elapsed() / 60:.1f} min", flush=True)
        progress["n_measured"] = len(durations)
        progress["mean_scene_seconds"] = round(sum(durations) / max(1, len(durations)), 2)

        # Drift check: the first scene's clean official neutral score, again at the end.
        sid, record, x = scenes[0]
        _, backend = backend_for(sid, record, x)
        again = backend.score_official(x, "neutral")
        first = json.loads(store.records.read_text().splitlines()[0])["scores"]["neutral"]["official"]
        progress["drift_check"] = {"scene": sid, "exact": again["nll"] == first["nll"] and again["margin"] == first["margin"]}

        if smoke_dev:
            ref = _dev_reference(root, sid)
            rows = [json.loads(x) for x in store.records.read_text().splitlines()]
            chk = {"clean": _compare({c: rows[0]["scores"][c]["official"] for c in CONTEXTS}, ref["clean"]) if ref["clean"] else None}
            src = Store(root, OUT_RELATIVE)
            dev_deltas = {s: src.load_tensors(f"{sid}__B__{s}__deltas") for s in CONTEXTS}
            mine_t = store.load_tensors(sid)
            chk["codes_equal_2_3"] = {f"{s}__{n}": bool(torch.equal(mine_t[f"{s}__{n}"], dev_deltas[s][n]))
                                      for s in CONTEXTS for n in ("fgsm", "pgd1", "pgd2", "pgd3")}
            dev_grads = src.load_tensors(f"{sid}__B__grads")
            chk["grads_equal_2_3"] = {k: bool(torch.equal(mine_t[f"grad_{k}"], dev_grads[k])) for k in dev_grads}
            per = []
            for r in rows[1:]:
                if r["kind"] in ("fgsm", "pgd3", "topk_grad"):
                    theirs = ref[r["kind"]].get(r["source"])
                elif r["kind"] == "random":
                    theirs = ref["random"].get((r["source"], r["seed_index"]))
                else:
                    continue
                per.append({"kind": r["kind"], "source": r["source"], "seed_index": r.get("seed_index"),
                            "bit_exact": None if theirs is None else _compare(r["scores"], theirs)})
            chk["scores"] = per
            chk["all_bit_exact"] = (chk["clean"] is True and all(chk["codes_equal_2_3"].values())
                                    and all(chk["grads_equal_2_3"].values()) and all(p["bit_exact"] is True for p in per))
            progress["reproduction_check"] = chk
            progress["runtime_estimate_48_scenes_min"] = round(48 * durations[0] / 60, 1)
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
        progress["elapsed_seconds"] = round(elapsed(), 1)
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
