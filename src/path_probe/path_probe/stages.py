"""Guarded load and stages 1-3 of EXPLORATORY_PLAN.md."""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Any

from .run import (
    FROZEN_SEED_PREFIX,
    GIB,
    HELPFUL_SEED_PREFIX,
    OUT_RELATIVE,
    SMOKE_RELATIVE,
    STAGE12_CAP_S,
    STAGE3_START_CAP_S,
    TOTAL_CAP_S,
    Stop,
    Store,
    plan_commit,
    select_scenes,
)

EPS = 1.0 / 255.0
LEGACY_RESIDENTS = (
    "docs/checkpoint-1.7/development_v4/resident.json",
    "docs/checkpoint-1.8/qwen_rgb_loss/resident.json",
    "docs/checkpoint-1.9/attack_feasibility/resident.json",
    "docs/checkpoint-2.0/calibration/resident.json",
    "docs/checkpoint-2.1/development/resident.json",
)


def _git_head(root: Path) -> str:
    import subprocess
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _check_residents(root: Path) -> None:
    from development_attack.run import _other_model_processes, _resident_alive

    others = _other_model_processes()
    if others:
        raise Stop("another model process is running: " + others[0])
    for rel in LEGACY_RESIDENTS:
        p = root / rel
        if p.is_file() and _resident_alive(p):
            raise Stop(f"a model is resident according to {p}")
    mine = root / OUT_RELATIVE / "resident.json"
    if mine.is_file() and _resident_alive(mine):
        raise Stop(f"a model is resident according to {mine}")


def execute(root: Path, smoke: bool = False, policy: str = "frozen") -> None:
    import json

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
    from .backend import QwenBackend
    from .precision import Fp32Projection, VisionPrecision
    from .memory import EMP1_V32_AVAILABLE_BYTES, MemoryMonitor, emp1_preload, os_sample, process_footprint
    from .probe import CONTEXTS, clean_block, cross_evaluate, direction_block

    if policy not in ("frozen", "emp1"):
        raise Stop(f"unknown memory policy {policy!r}")

    commit = plan_commit(root)
    out_rel = SMOKE_RELATIVE if smoke else OUT_RELATIVE
    store = Store(root, out_rel)
    if store.records.exists() and store.records.stat().st_size > 0:
        raise Stop("records.jsonl already has rows. This exploratory run does not resume or overwrite.")
    _check_residents(root)
    locked = verify_locked_inputs(root)
    records = {r["scene_id"]: r for r in load_manifest(root)}
    s16, order = select_scenes(list(records.values()))
    if smoke:
        # One scene, 3 steps, every code path including fp32 vision. Not part of the results.
        s16, order = [s16[0]], [s16[0]]
    stored = {(r["scene_id"], r["context"]): r for r in load_predictions(root)}
    progress: dict[str, Any] = {
        "schema": "checkpoint-2.3.paths.progress.v1",
        "label": "EXPLORATORY, DEVELOPMENT ONLY" + (" (SMOKE, not a result)" if smoke else ""),
        "plan_commit": commit,
        "git_head_at_start": _git_head(root),
        "locked_inputs": locked,
        "scene_order": order,
        "subset_s16": s16,
        "epsilon": EPS,
        "caps_seconds": {"stage12": STAGE12_CAP_S, "stage3_start": STAGE3_START_CAP_S, "total": TOTAL_CAP_S},
        "environment": prepare_environment(),
        "pid": os.getpid(),
        "frozen_outputs_written": False,
        "sealed_or_new_samples_used": False,
        "stages": {},
        "scenes": {},
        "limitations": [],
        "memory_policy": "frozen Checkpoint 1.5 guard" if policy == "frozen" else "EMP-1 (exploratory; MEMORY_ASSESSMENT.md)",
        "numerical_configuration": "identical to Checkpoint 2.1: bf16 weights on MPS, eager attention, frozen load call, frozen gradient/step/score functions",
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
    progress["limitations"].append(
        "Full fp32 was not attempted: the frozen guard requires "
        f"{decisions['float32'].get('required_available_bytes')} available bytes, and {available} were available "
        f"(float32 permitted: {decisions['float32'].get('permitted')}). The guard was not lowered.")
    monitor = None
    progress["frozen_guard_bf16_permitted"] = bool(decisions["bfloat16"].get("permitted"))
    if policy == "frozen":
        if not (decisions["bfloat16"].get("permitted") and probe.get("matmul_ok")):
            progress["status"] = "stopped_budget"
            store.save_progress(progress)
            raise Stop("the bf16 budget did not permit a load")
    else:
        import shutil

        emp = emp1_preload(os_sample(), shutil.disk_usage(root).free)
        progress["emp1_preload"] = emp
        if not (emp["permitted"] and probe.get("matmul_ok")):
            progress["status"] = "stopped_budget_emp1"
            store.save_progress(progress)
            raise Stop("EMP-1 did not permit a load: " + ", ".join(emp["failed"]))
        if not decisions["bfloat16"].get("permitted"):
            progress["limitations"].append(
                "The frozen Checkpoint 1.5 guard refused this bf16 load; it ran under EMP-1 with pressure monitoring. "
                "Numerics are unchanged; only the memory headroom rule differs.")
        monitor = MemoryMonitor()
        monitor.start()
        progress["footprint_before_load"] = process_footprint()
        store.save_progress(progress)

    # The frozen load function takes the permit as its only authority. Under the frozen policy that
    # is the frozen bf16 decision. Under EMP-1 it is EMP-1's own decision, recorded in full.
    if policy == "frozen":
        permit = decisions["bfloat16"]
    else:
        permit = {"permitted": bool(progress["emp1_preload"]["permitted"]), "policy": "EMP-1",
                  "frozen_bf16_decision_permitted": bool(decisions["bfloat16"].get("permitted")),
                  "emp1_preload": progress["emp1_preload"]}
    progress["load_permit"] = permit
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
        if monitor is not None:
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

    def guard() -> None:
        if monitor is not None:
            monitor.check()

    def memory_ok(tag: str) -> None:
        if monitor is not None:
            monitor.check()
            progress["memory_monitor"] = monitor.snapshot()
            progress["last_memory"] = {"tag": tag, **memory_snapshot(device), "footprint": process_footprint()}
            return
        avail = available_bytes()
        progress.setdefault("min_available_bytes", avail)
        progress["min_available_bytes"] = min(progress["min_available_bytes"], avail)
        progress["last_memory"] = {"tag": tag, **memory_snapshot(device), "available_bytes": avail}
        if avail < 6 * GIB:
            raise Stop(f"available memory {avail} fell below the 6 GiB OS floor at {tag}")

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
                              projection=projection, guard=guard)
        return record, image, backend

    def tag(rows, sid, mode):
        for r in rows:
            r["scene_id"] = sid
            r["mode"] = mode
        return rows

    try:
        # Stages 1 and 2, mode B.
        durations: list[float] = []
        unmeasured: list[str] = []
        for i, sid in enumerate(order, 1):
            mean = sum(durations) / len(durations) if durations else 0.0
            if elapsed() + mean > STAGE12_CAP_S:
                unmeasured = order[i - 1:]
                break
            started = time.perf_counter()
            print(f"[B] {sid} ({i}/{len(order)}) elapsed {elapsed()/60:.1f} min", flush=True)
            record, x, backend = scene_setup(sid)
            clean, grads = clean_block(backend, x)
            store.save_tensors(f"{sid}__B__grads", grads)
            rows = [clean]
            steps = 3 if smoke else (7 if sid in s16 else 3)
            official = tuple(k for k in (3, 7) if k <= steps)
            for source, prefix in (("neutral", FROZEN_SEED_PREFIX), ("helpful_nonrevealing", HELPFUL_SEED_PREFIX)):
                d_rows, saved = direction_block(
                    backend, x, source, grads, eps=EPS, steps=steps, official_steps=official,
                    random_seed=scene_seed(prefix, sid),
                    clean_source_nll=clean["scores"][source]["float"]["nll"],
                    deadline=lambda: elapsed() > TOTAL_CAP_S)
                rows += d_rows
                store.save_tensors(f"{sid}__B__{source}__deltas", saved)
            store.append(tag(rows, sid, "B"))
            del grads
            release_cache(device)
            memory_ok(sid)
            dur = time.perf_counter() - started
            durations.append(dur)
            progress["scenes"][sid] = {"mode_B": True, "seconds": round(dur, 2), "steps": steps}
            progress["elapsed_seconds"] = round(elapsed(), 1)
            store.save_progress(progress)
        progress["stages"]["B"] = {"measured": len(durations), "unmeasured_by_cap": unmeasured,
                                   "seconds": round(sum(durations), 1)}
        store.save_progress(progress)

        # Stage 3, mode V32, on S16 measured in B.
        targets = [s for s in s16 if progress["scenes"].get(s, {}).get("mode_B")]
        # transformers 5.x: the vision tower is model.model.visual (get_image_features casts to its dtype).
        visual = getattr(model, "visual", None) or model.model.visual
        vp = VisionPrecision(visual)
        need = (6 * GIB + 2 * vp.fp32_bytes() + GIB) if monitor is None else EMP1_V32_AVAILABLE_BYTES
        avail = available_bytes()
        pressure_now = os_sample()["pressure"]
        if monitor is not None and pressure_now != 1:
            avail = -1  # EMP-1: fp32 vision only at normal pressure
        stage3: dict[str, Any] = {"available_bytes": avail, "required_bytes": need, "vision_fp32_bytes": vp.fp32_bytes(),
                                  "pressure_level_at_check": pressure_now}
        progress["stages"]["V32"] = stage3
        if elapsed() >= STAGE3_START_CAP_S:
            stage3["skipped"] = "the stage 3 start cap was reached"
            progress["limitations"].append("Stage 3 (fp32 vision) was not run: time cap.")
        elif avail < need:
            stage3["skipped"] = "the memory check before conversion failed"
            progress["limitations"].append(f"Stage 3 (fp32 vision) was not run: {avail} bytes available, {need} required.")
        else:
            reference_sid = targets[0]
            _, x_ref, b_ref = scene_setup(reference_sid)
            before = b_ref.score_official(x_ref, "neutral")
            stage3["converted_elements"] = vp.to_fp32()
            stage3["visual_dtype_after"] = str(next(visual.parameters()).dtype)
            v_durations: list[float] = []
            done: list[str] = []
            try:
                for sid in targets:
                    mean = sum(v_durations) / len(v_durations) if v_durations else 0.0
                    if elapsed() + mean > TOTAL_CAP_S:
                        stage3.setdefault("skips", []).append({"scene": sid, "reason": "total time cap"})
                        break
                    if monitor is not None:
                        now_s = os_sample()
                        if now_s["pressure"] != 1 or now_s["available"] < EMP1_V32_AVAILABLE_BYTES:
                            stage3.setdefault("skips", []).append(
                                {"scene": sid, "reason": "fp32-vision headroom insufficient before scene",
                                 "pressure": now_s["pressure"], "available_bytes": now_s["available"],
                                 "required_bytes": EMP1_V32_AVAILABLE_BYTES})
                            progress["limitations"].append(f"fp32 vision stopped before {sid}: insufficient headroom.")
                            break
                    started = time.perf_counter()
                    print(f"[V32] {sid} elapsed {elapsed()/60:.1f} min", flush=True)
                    record, x, backend = scene_setup(sid)
                    clean, grads = clean_block(backend, x, ("L_neutral", "M_neutral", "L_helpful_nonrevealing"))
                    b_grads = store.load_tensors(f"{sid}__B__grads")
                    clean["agreement_with_B"] = {
                        k: {"cos": geo.cosine(grads[k], b_grads[k]), "sign_agree": geo.sign_agreement(grads[k], b_grads[k]),
                            "sign_agree_weighted": geo.sign_agreement(grads[k], b_grads[k], b_grads[k]),
                            "l1_ratio": geo.norms(grads[k])["l1"] / geo.norms(b_grads[k])["l1"]}
                        for k in grads}
                    rows = [clean]
                    d_rows, saved = direction_block(
                        backend, x, "neutral", grads, eps=EPS, steps=3, official_steps=(3,),
                        random_seed=scene_seed(FROZEN_SEED_PREFIX, sid),
                        clean_source_nll=clean["scores"]["neutral"]["float"]["nll"],
                        deadline=lambda: elapsed() > TOTAL_CAP_S)
                    rows += d_rows
                    store.save_tensors(f"{sid}__V32__neutral__deltas", saved)
                    b_deltas = store.load_tensors(f"{sid}__B__neutral__deltas")
                    cross = {k: geo.decode_delta(b_deltas[k], EPS) for k in ("fgsm", "pgd3", "pgd7") if k in b_deltas}
                    rows += cross_evaluate(backend, x, cross, b_grads, EPS, crafted_in="B")
                    store.append(tag(rows, sid, "V32"))
                    release_cache(device)
                    memory_ok(f"V32 {sid}")
                    dur = time.perf_counter() - started
                    v_durations.append(dur)
                    done.append(sid)
                    progress["scenes"][sid]["mode_V32"] = True
                    progress["scenes"][sid]["seconds_V32"] = round(dur, 2)
                    progress["elapsed_seconds"] = round(elapsed(), 1)
                    store.save_progress(progress)
            finally:
                stage3["restored_elements"] = vp.restore()
                stage3["visual_dtype_restored"] = str(next(visual.parameters()).dtype)
            after = b_ref.score_official(x_ref, "neutral")
            stage3["restore_check"] = {"scene": reference_sid, "nll_before": before["nll"], "nll_after": after["nll"],
                                       "margin_before": before["margin"], "margin_after": after["margin"],
                                       "exact": before["nll"] == after["nll"] and before["margin"] == after["margin"]}
            stage3["measured"] = done
            stage3["unmeasured_by_cap"] = [s for s in targets if s not in done]
            # Cross-evaluate the V32 perturbations under B (after restore).
            for sid in done:
                if elapsed() > TOTAL_CAP_S:
                    stage3.setdefault("cross_B_unmeasured", []).append(sid)
                    continue
                record, x, backend = scene_setup(sid)
                b_grads = store.load_tensors(f"{sid}__B__grads")
                v_deltas = store.load_tensors(f"{sid}__V32__neutral__deltas")
                cross = {k: geo.decode_delta(v_deltas[k], EPS) for k in ("fgsm", "pgd3") if k in v_deltas}
                store.append(tag(cross_evaluate(backend, x, cross, b_grads, EPS, crafted_in="V32"), sid, "B_cross"))
                release_cache(device)
            stage3["seconds"] = round(sum(v_durations), 1)
        progress["status"] = "completed"
        progress["reproduction_check"] = reproduction_check(root, store)
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
        if monitor is not None:
            monitor.stop()
            progress["memory_monitor"] = monitor.snapshot()
            progress["footprint_end"] = process_footprint()
        projection.remove()
        store.save_progress(progress)
        try:
            (root / out_rel / "resident.json").unlink()
        except FileNotFoundError:
            pass
        del model
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


FROZEN_KEYS = {("fgsm", None): ("fgsm", 1), ("random", None): ("random", 1), ("pgd", 3): ("pgd", 3), ("pgd", 7): ("pgd", 7)}


def reproduction_check(root: Path, store: Store) -> dict[str, Any]:
    """Neutral-source official scores against the frozen Checkpoint 2.1 records (read only)."""
    import json

    frozen = {}
    for line in (root / "docs/checkpoint-2.1/development/records.jsonl").read_text().splitlines():
        r = json.loads(line)
        frozen[(r["scene_id"], r["method"], r["steps"])] = r
    rows = [json.loads(x) for x in store.records.read_text().splitlines()] if store.records.exists() else []
    out = {"compared": 0, "exact_nll": 0, "exact_margin": 0, "same_decision": 0, "max_abs_nll_diff": 0.0, "mismatches": []}
    for r in rows:
        if r.get("mode") != "B" or r.get("source") != "neutral":
            continue
        key = FROZEN_KEYS.get((r["kind"], r.get("k") if r["kind"] == "pgd" else None))
        if key is None or "official" not in r["scores"]["neutral"]:
            continue
        f = frozen.get((r["scene_id"], *key))
        if f is None:
            continue
        for ctx in ("neutral", "helpful_nonrevealing"):
            mine = r["scores"][ctx]["official"]
            theirs = f["perturbed"][ctx]
            out["compared"] += 1
            d = abs(mine["nll"] - theirs["nll"])
            out["max_abs_nll_diff"] = max(out["max_abs_nll_diff"], d)
            out["exact_nll"] += int(d == 0)
            out["exact_margin"] += int(mine["margin"] == theirs["margin"])
            out["same_decision"] += int(mine["chosen"] == theirs["decision"])
            if d != 0 or mine["chosen"] != theirs["decision"]:
                out["mismatches"].append({"scene_id": r["scene_id"], "method": key, "context": ctx,
                                          "nll_diff": d, "decision": [mine["chosen"], theirs["decision"]]})
    return out
