"""Checkpoint 2.3b: random-mask FGSM control (MASK_CONTROL_PLAN.md). EXPLORATORY, DEVELOPMENT ONLY.

Reads Checkpoint 2.3 tensors and records (never writes them). Writes only to
docs/checkpoint-2.3-exploratory/masks[_smoke]/. The guarded load is the same as
stages.execute under EMP-1 (M1 thresholds unchanged). Forward passes only.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .run import GIB, OUT_RELATIVE, Stop, Store, select_scenes

EPS = 1.0 / 255.0
MASK_PLAN = Path("docs/checkpoint-2.3-exploratory/MASK_CONTROL_PLAN.md")
MASK_OUT = Path("docs/checkpoint-2.3-exploratory/masks")
MASK_SMOKE = Path("docs/checkpoint-2.3-exploratory/masks_smoke")
MASK_CAP_S = 60 * 60
SEEDS_A = (0, 1, 2, 3)
SEEDS_B = (4, 5, 6, 7)
SEED_PREFIX = "checkpoint-2.3b-mask"


def _plan_commit(root: Path) -> str:
    dirty = subprocess.run(["git", "status", "--porcelain", "--", str(MASK_PLAN)], cwd=root, capture_output=True, text=True).stdout
    if dirty.strip():
        raise Stop("MASK_CONTROL_PLAN.md has uncommitted changes")
    return subprocess.check_output(["git", "log", "-1", "--format=%H", "--", str(MASK_PLAN)], cwd=root, text=True).strip()


def execute_masks(root: Path, smoke: bool = False) -> None:
    import torch
    from attack_calibration.scene import _development_image
    from attack_feasibility.score import prepare_prompt, release_cache
    from attack_feasibility.steps import scene_seed
    from choice_eval.prompt import build_prompt
    from dev_pilot.fixtures import load_manifest
    from dev_pilot.rows import user_text_sha256
    from dev_pilot.scoring import assert_locked_tokenization
    from qwen_rgb.budget import dtype_decisions, load_budget_module
    from qwen_rgb.measure import bf16_probe, load_model_after_budget, load_processor, prepare_environment, read_local_header, weight_file
    from qwen_rgb.pins import MODEL_ID, PINNED_REVISION
    from qwen_rgb.rgb_path import assert_pinned_processor
    from qwen_rgb_loss.offline import load_predictions, verify_locked_inputs
    from qwen_rgb_loss.protocol import MODEL_KEY
    from vlm_iface.devices import available_bytes, memory_snapshot

    from . import geometry as geo
    from . import masks as mk
    from .backend import QwenBackend
    from .memory import MemoryMonitor, emp1_preload, os_sample, process_footprint
    from .precision import Fp32Projection
    from .probe import CONTEXTS, _keep, other
    from .stages import _check_residents

    commit = _plan_commit(root)
    out_rel = MASK_SMOKE if smoke else MASK_OUT
    store = Store(root, out_rel)
    if store.records.exists() and store.records.stat().st_size > 0:
        raise Stop("masks records.jsonl already has rows. This run does not resume or overwrite.")
    _check_residents(root)
    src = Store(root, OUT_RELATIVE)  # Checkpoint 2.3 outputs, read only
    locked = verify_locked_inputs(root)
    records = {r["scene_id"]: r for r in load_manifest(root)}
    _, order = select_scenes(list(records.values()))
    if smoke:
        order = order[:1]
    stored = {(r["scene_id"], r["context"]): r for r in load_predictions(root)}
    progress: dict[str, Any] = {
        "schema": "checkpoint-2.3b.masks.progress.v1",
        "label": "EXPLORATORY, DEVELOPMENT ONLY" + (" (SMOKE, not a result)" if smoke else ""),
        "plan_commit": commit, "git_head_at_start": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "locked_inputs": locked, "scene_order": order, "epsilon": EPS, "cap_seconds": MASK_CAP_S,
        "seeds_pass_A": SEEDS_A, "seeds_pass_B": SEEDS_B, "environment": prepare_environment(), "pid": os.getpid(),
        "frozen_outputs_written": False, "checkpoint_2_3_outputs_written": False, "sealed_or_new_samples_used": False,
        "memory_policy": "EMP-1 with M1 (unchanged; MEMORY_ASSESSMENT.md section 7)", "units": {}, "limitations": [],
    }
    (root / out_rel / "resident.json").write_text(json.dumps({"pid": os.getpid(), "model_key": MODEL_KEY}) + "\n")
    store.save_progress(progress)

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
    progress["frozen_guard_bf16_permitted"] = bool(decisions["bfloat16"].get("permitted"))
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
        device = torch.device("mps")
        model.to(device)
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
    projection = Fp32Projection(model.lm_head, token_ids)
    t0 = time.perf_counter()

    def elapsed() -> float:
        return time.perf_counter() - t0

    def memory_ok(tag: str) -> None:
        monitor.check()
        progress["memory_monitor"] = monitor.snapshot()
        progress["last_memory"] = {"tag": tag, **memory_snapshot(device), "footprint": process_footprint()}

    def scene_setup(sid: str):
        record = records[sid]
        if record.get("split") != "development":
            raise Stop(f"{sid} is not development")
        image = _development_image(root, record)
        texts, prompts = {}, {}
        for ctx in CONTEXTS:
            text = build_prompt(record["contexts"][ctx]["text"], record["question"]).user_text()
            if stored[(sid, ctx)].get("user_text_sha256") != user_text_sha256(text):
                raise Stop(f"{sid} {ctx} prompt hash does not match the stored row")
            texts[ctx] = text
            prompts[ctx] = prepare_prompt(processor, image_processor, image, text)
        backend = QwenBackend(model=model, processor=processor, image_processor=image_processor, prompts=prompts,
                              texts=texts, token_ids=token_ids, label=str(record["answer"]), device=device,
                              projection=projection, guard=monitor.check)
        return record, image, backend

    # Recorded official PGD-3 scores (Checkpoint 2.3, read only) for the pipeline check.
    recorded_pgd3 = {}
    for line in src.records.read_text().splitlines():
        r = json.loads(line)
        if r.get("mode") == "B" and r["kind"] == "pgd" and r.get("k") == 3:
            recorded_pgd3[(r["scene_id"], r["source"])] = {c: r["scores"][c]["official"] for c in CONTEXTS}

    def preds(grads, r):
        return {f"{g}_{c}": geo.dot(grads[f"{g}_{c}"], r) for g in ("L", "M") for c in CONTEXTS}

    def score_mask(backend, x, r_f, mask, grads, name, extra, image=None):
        image = mk.apply(x, r_f, mask) if image is None else image
        r = image - x
        row = {"kind": "mask", "mask": name, "k": int(mk.support(r).sum()), "pred_clean_gradient": preds(grads, r), "scores": {}}
        row.update(extra)
        for ctx in CONTEXTS:
            row["scores"][ctx] = _keep(backend.score_official(image, ctx))
        return row

    def unit(sid, x, backend, grads, source, seeds, with_fixed):
        d = src.load_tensors(f"{sid}__B__{source}__deltas")
        r_f = mk.rounded_delta(x, geo.decode_delta(d["fgsm"], EPS))
        pgd_float = mk.replay_pgd(x, d, 3, EPS)  # bit-identical to the 2.3 float iterate (see masks.replay_pgd)
        if not torch.equal(geo.encode_delta(pgd_float - x, EPS), d["pgd3"]):
            raise Stop(f"{sid} {source}: PGD replay does not reproduce the stored iterate")
        r_p = mk.rounded_delta(x, pgd_float - x)
        st = mk.structure(r_f, r_p)
        s_f, s_p = mk.support(r_f), mk.support(r_p)
        k = st["n_pgd_support"]
        base = {"scene_id": sid, "source": source, "target": other(source), "n_fgsm_support": st["n_fgsm_support"]}
        rows = []
        if with_fixed:
            outside = s_p & ~s_f
            tie = ((pgd_float - x).abs() - 0.5 * EPS).abs() < 1e-3 * EPS
            st["outside_all_half_level_ties"] = bool((tie | ~outside).all())
            st["outside_where_clean_grad_zero"] = int((outside & (grads[f"L_{source}"] == 0)).sum())
            rows.append({"kind": "structure", **base, **st})
            # The official PGD-3 image itself (S_P may hold a few saturated-pixel coordinates outside S_F).
            row = score_mask(backend, x, r_f, s_p, grads, "pgd_mask", base, image=pgd_float)
            same_image = bool(torch.equal(mk.rounded_delta(x, pgd_float - x), r_p))
            rec = recorded_pgd3.get((sid, source))
            row["pipeline_check"] = {
                "same_image_as_official_pgd3": same_image,
                "bit_exact_vs_recorded": None if rec is None else all(
                    row["scores"][c]["nll"] == rec[c]["nll"] and row["scores"][c]["margin"] == rec[c]["margin"]
                    and row["scores"][c]["chosen"] == rec[c]["chosen"] for c in CONTEXTS)}
            if row["pipeline_check"]["bit_exact_vs_recorded"] is not True or not same_image:
                store.append(rows + [row])
                raise Stop(f"{sid} {source}: re-scored PGD-3 is not bit-exact against the recorded 2.3 scores")
            rows.append(row)
            rows.append(score_mask(backend, x, r_f, mk.topk_mask(s_f, grads[f"L_{source}"], k), grads, "topk_grad", base))
        for s in seeds:
            seed = scene_seed(f"{SEED_PREFIX}|{source}|seed{s}", sid)
            rows.append(score_mask(backend, x, r_f, mk.random_mask(s_f, k, seed), grads, "random",
                                   {**base, "seed_index": s, "seed": seed}))
        return rows

    def run_pass(name, seeds, with_fixed):
        durations: list[float] = []
        done, skipped = [], []
        for sid in order:
            for source in CONTEXTS:
                mean = sum(durations) / len(durations) if durations else 0.0
                if elapsed() + mean > MASK_CAP_S:
                    skipped.append([sid, source])
                    continue
                started = time.perf_counter()
                _, x, backend = scene_setup(sid)
                grads = src.load_tensors(f"{sid}__B__grads")
                rows = unit(sid, x, backend, grads, source, seeds, with_fixed)
                for r in rows:
                    r["pass"] = name
                store.append(rows)
                release_cache(device)
                memory_ok(f"{name} {sid} {source}")
                durations.append(time.perf_counter() - started)
                done.append([sid, source])
                progress["units"].setdefault(f"{sid}|{source}", {})[name] = round(durations[-1], 2)
                progress["elapsed_seconds"] = round(elapsed(), 1)
                store.save_progress(progress)
            print(f"[{name}] {sid} elapsed {elapsed()/60:.1f} min", flush=True)
        progress.setdefault("passes", {})[name] = {"done": len(done), "skipped_by_cap": skipped,
                                                   "seconds": round(sum(durations), 1)}
        if skipped:
            progress["limitations"].append(f"pass {name}: {len(skipped)} units not measured (60 min cap)")
        store.save_progress(progress)

    try:
        run_pass("A", SEEDS_A, True)
        run_pass("B", SEEDS_B, False)
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
