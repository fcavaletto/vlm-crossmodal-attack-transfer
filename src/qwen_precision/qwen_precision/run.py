"""Checkpoint 5 runner: Qwen3-VL-2B, bf16 vs full fp32, the unchanged 2.4 scene block.

EXPLORATORY, development data only (the 48 arm-a scenes of development_v4). The confirmation
split is never read.

    python -m qwen_precision.run repro-mps                       # local, free: bit-exact vs 2.3/2.3b
    python -m qwen_precision.run inputs                          # model-free: prompt/pixel hashes
    python -m qwen_precision.run audit --dtype fp32 --device cuda
    python -m qwen_precision.run clean --device cuda             # G5/G6, both precisions
    python -m qwen_precision.run fd --dtype fp32 --device cuda   # G7
    python -m qwen_precision.run main --device cuda --hourly-usd 0.59 --pod-start-epoch ... --deadline-epoch ...

The scene computation is path_probe.confirm_stage.scene_block, imported and called unchanged,
through path_probe.backend.QwenBackend (observed by RecordingBackend, not modified).
Protocol: docs/checkpoint-5-qwen-fp32/PLAN.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .numerics import DTYPES, DtypeAudit, assert_no_autocast, configure, fp32_matmul_probe, non_matching, settings

EPS = 1.0 / 255.0
CONTEXTS = ("neutral", "helpful_nonrevealing")
MAIN_SEED_PREFIX = "checkpoint-5-qwen-precision-mask"
REPRO_SEED_PREFIX = "checkpoint-2.3b-mask"  # reproduces the 2.3b random masks exactly
OUT = Path("docs/checkpoint-5-qwen-fp32")
PROTECTED = ("docs/checkpoint-5-qwen-fp32/PLAN.md", "docs/checkpoint-5-qwen-fp32/analyze_qwen_precision.py",
             "src/qwen_precision/qwen_precision/run.py", "src/qwen_precision/qwen_precision/backend.py",
             "src/qwen_precision/qwen_precision/numerics.py", "src/qwen_precision/qwen_precision/load.py",
             "src/qwen_precision/qwen_precision/guard.py")
GIB = 1024**3
FD_STEPS_LEVELS = (1 / 1024, 1 / 256, 1 / 64, 1 / 16)
FD_DIRECTIONS = 4
FD_SCENES = 6


class Stop(RuntimeError):
    pass


# ------------------------------------------------------------------ setup
def repo_root() -> Path:
    here = Path(__file__).resolve()
    for c in here.parents:
        if (c / "fixtures" / "development_v4").is_dir() and (c / "src").is_dir():
            return c
    raise Stop("repository root not found")


def git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True).stdout.strip()


def protocol_commit(root: Path) -> str:
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *PROTECTED], cwd=root, capture_output=True, text=True).stdout
    if dirty.strip():
        raise Stop("the Checkpoint 5 protocol or code has uncommitted changes:\n" + dirty)
    return git(root, "log", "-1", "--format=%H", "--", PROTECTED[0])


def scenes(root: Path) -> list[dict[str, Any]]:
    from dev_pilot.fixtures import load_manifest
    from path_probe.run import select_scenes

    rows = {r["scene_id"]: r for r in load_manifest(root)}
    if any(r.get("split") != "development" for r in rows.values()):
        raise Stop("a non-development row was loaded")
    _, order = select_scenes(list(rows.values()))  # s16 first, then the other 32 arm-a scenes
    return [rows[s] for s in order]


def image(root: Path, row):
    from attack_calibration.scene import _development_image

    return _development_image(root, row)


def texts_for(row) -> dict[str, str]:
    from choice_eval.prompt import build_prompt

    return {c: build_prompt(row["contexts"][c]["text"], row["question"]).user_text() for c in CONTEXTS}


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")


class Session:
    """Processor, models (one per dtype), projections and backends on one device."""

    def __init__(self, device: str, dtypes: tuple[str, ...]):
        from path_probe.precision import Fp32Projection

        from .load import load_model, load_processor_and_tokens

        self.device = torch.device(device)
        self.numerics = configure(self.device)
        assert_no_autocast(self.device)
        self.processor, self.image_processor, self.token_ids = load_processor_and_tokens()
        self.models, self.projections, self.load_seconds = {}, {}, {}
        for d in dtypes:
            t = time.perf_counter()
            self.models[d] = load_model(d, self.device)
            self.projections[d] = Fp32Projection(self.models[d].lm_head, self.token_ids)
            self.load_seconds[d] = round(time.perf_counter() - t, 1)

    def backend(self, dtype: str, row, x, guard, *, float_check: bool = True):
        from attack_feasibility.score import prepare_prompt

        from .backend import RecordingBackend

        texts = texts_for(row)
        prompts = {c: prepare_prompt(self.processor, self.image_processor, x, t) for c, t in texts.items()}
        assert_no_autocast(self.device)
        return RecordingBackend(model=self.models[dtype], processor=self.processor, image_processor=self.image_processor,
                                prompts=prompts, texts=texts, token_ids=self.token_ids, label=str(row["answer"]),
                                device=self.device, projection=self.projections[dtype], guard=guard,
                                float_check=float_check)


def make_guard(device: torch.device, deadline: float | None):
    if device.type == "cuda":
        from .guard import CudaGuard

        g = CudaGuard(device, deadline_epoch=deadline)
        return g, g.arm(), g.check
    # Local MPS: the existing safeguards, unchanged (EMP-1 preload lines and the 10 GiB monitor).
    import shutil

    from path_probe.memory import MemoryMonitor, emp1_preload, os_sample

    pre = emp1_preload(os_sample(), shutil.disk_usage(repo_root()).free)
    if not pre["permitted"]:
        raise Stop("EMP-1 preload lines not met: " + ", ".join(pre["failed"]))
    mon = MemoryMonitor()
    mon.start()
    return mon, {"policy": "EMP-1 + MemoryMonitor (unchanged)", "preload": pre}, mon.check


# ------------------------------------------------------------------ per-scene unit
def unit(session: Session, dtype: str, row, x, guard_check, seed_prefix: str):
    """Rows and tensors for one scene in one precision: the frozen scene_block plus diagnostics."""
    from path_probe import geometry as geo
    from path_probe.confirm_stage import scene_block
    from qwen_rgb.measure import encode_chat
    from qwen_rgb.rgb_path import differentiable_model_pixels
    from vlm_iface.images import image_01_to_pil

    from .backend import CALLS_PER_SOURCE, N_CLEAN_CALLS

    sid = row["scene_id"]
    backend = session.backend(dtype, row, x, guard_check)
    t0 = time.perf_counter()
    rows, tensors = scene_block(backend, x, sid, seed_prefix)
    seconds = time.perf_counter() - t0
    if len(backend.official_ctx) != N_CLEAN_CALLS + 2 * CALLS_PER_SOURCE:
        raise Stop(f"{sid}: unexpected number of official calls {len(backend.official_ctx)}")
    if [c for c, _ in backend.nll_grads] != ["neutral", "helpful_nonrevealing", "neutral", "neutral",
                                             "helpful_nonrevealing", "helpful_nonrevealing"]:
        raise Stop(f"{sid}: NLL gradient call order is not the expected one")
    g = [t for _, t in backend.nll_grads]
    align = {}
    for k, (src, g0, g1, g2) in enumerate((("neutral", g[0], g[2], g[3]), ("helpful_nonrevealing", g[1], g[4], g[5]))):
        align[src] = {"cos_g0_g1": geo.cosine(g0, g1), "sign_g0_g1": geo.sign_agreement(g0, g1),
                      "cos_g0_g2": geo.cosine(g0, g2), "sign_g0_g2": geo.sign_agreement(g0, g2),
                      "cos_g1_g2": geo.cosine(g1, g2), "sign_g1_g2": geo.sign_agreement(g1, g2)}
        tensors[f"{src}__g1"], tensors[f"{src}__g2"] = g1, g2
    official_px = encode_chat(session.processor, image_01_to_pil(x), texts_for(row)["neutral"])["pixel_values"].float()
    diff_px, _ = differentiable_model_pixels(x, session.image_processor)
    rows[0]["label"] = str(row["answer"])
    rows.append({"kind": "diagnostics", "scene_id": sid, "dtype": dtype, "seconds": round(seconds, 2),
                 "alignment": align, "float_path": backend.float_scores,
                 "clean_pixels_official_vs_differentiable_max_abs": float((official_px - diff_px).abs().max())})
    for r in rows:
        r["dtype"] = dtype
    return rows, {k: v.detach().cpu() for k, v in tensors.items()}


def append(path: Path, rows: list[dict[str, Any]]) -> None:
    from path_probe.run import _json_ready

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(_json_ready(r), sort_keys=True, allow_nan=False) + "\n")


# ------------------------------------------------------------------ commands
def cmd_inputs(args) -> None:
    """Model-free: hashes of every prompt tensor and pixel tensor (gate G3 compares the pod's to these)."""
    from attack_feasibility.score import prepare_prompt
    from qwen_rgb.measure import encode_chat
    from qwen_rgb.rgb_path import differentiable_model_pixels
    from vlm_iface.images import image_01_to_pil

    from .load import load_processor_and_tokens

    root = repo_root()
    processor, image_processor, _ = load_processor_and_tokens()
    out = {}
    for row in scenes(root):
        x = image(root, row)
        diff_px, grid = differentiable_model_pixels(x, image_processor)
        for c, t in texts_for(row).items():
            p = prepare_prompt(processor, image_processor, x, t)
            off = encode_chat(processor, image_01_to_pil(x), t)["pixel_values"]
            h = hashlib.sha256()
            for k in sorted(p):
                if torch.is_tensor(p[k]):
                    h.update(k.encode()); h.update(p[k].contiguous().numpy().tobytes())
            out[f"{row['scene_id']}|{c}"] = {
                "prompt_tensors_sha256": h.hexdigest(),
                "official_pixels_sha256": hashlib.sha256(off.contiguous().numpy().tobytes()).hexdigest(),
                "differentiable_pixels_sha256": hashlib.sha256(diff_px.contiguous().numpy().tobytes()).hexdigest()}
    dest = Path(args.out) if args.out else root / OUT / "reference" / "input_hashes.json"
    write_json(dest, out)
    if args.compare:
        ref = json.loads(Path(args.compare).read_text())
        bad = sorted(k for k in ref if ref[k] != out.get(k))
        write_json(dest.with_name("input_hashes_compare.json"), {"n": len(ref), "mismatched": bad, "pass": not bad and set(ref) == set(out)})
        print("G3 inputs:", "pass" if not bad else f"FAIL {len(bad)} mismatched")
    print(dest)


def cmd_audit(args) -> None:
    """G1 environment and TF32, G2 dtype audit, weights facts. One clean forward and one gradient."""
    from .load import weights_facts

    root = repo_root()
    device = torch.device(args.device)
    guard, _, check = make_guard(device, None)  # preload lines before any load
    session = Session(args.device, (args.dtype,))
    row = scenes(root)[0]
    x = image(root, row)
    audit = DtypeAudit(session.models[args.dtype])
    try:
        b = session.backend(args.dtype, row, x, check, float_check=False)
        s1 = b.score_official(x, "neutral")
        g1 = b.grad_nll(x, "neutral")
    finally:
        seen = audit.remove()
    try:  # G8: a repeated score and gradient, without hooks, must be bit-identical
        b = session.backend(args.dtype, row, x, check, float_check=False)
        s2, s3 = b.score_official(x, "neutral"), b.score_official(x, "neutral")
        g2, g3 = b.grad_nll(x, "neutral"), b.grad_nll(x, "neutral")
        determinism = {"score_repeat_identical": s2["nll"] == s3["nll"] and s2["margin"] == s3["margin"],
                       "gradient_repeat_bit_identical": bool(torch.equal(g2, g3)),
                       "gradient_repeat_max_abs": float((g2 - g3).abs().max()),
                       "hooked_vs_unhooked_score_identical": s1["nll"] == s2["nll"],
                       "hooked_vs_unhooked_gradient_identical": bool(torch.equal(g1, g2))}
    finally:
        if hasattr(guard, "stop"):
            guard.stop()
    out = {"dtype": args.dtype, "numerics": settings(device), "fp32_matmul_probe": fp32_matmul_probe(device),
           "weights": weights_facts(), "dtype_audit": seen, "modules_not_in_dtype": non_matching(seen, args.dtype),
           "determinism": determinism,
           "git_head": git(root, "rev-parse", "HEAD")}
    write_json(Path(args.out) if args.out else root / OUT / "audit" / f"audit_{args.dtype}_{device.type}.json", out)
    print(json.dumps({"tf32_probe": out["fp32_matmul_probe"], "not_in_dtype": list(out["modules_not_in_dtype"])}, indent=1))


def cmd_clean(args) -> None:
    """Official clean scores for all 96 prompts in every requested precision (G5, G6)."""
    root = repo_root()
    device = torch.device(args.device)
    dtypes = tuple(args.dtypes.split(","))
    guard, _, check = make_guard(device, None)  # preload lines before any load
    session = Session(args.device, dtypes)
    out: dict[str, Any] = {"numerics": settings(device), "scores": {}}
    try:
        for row in scenes(root):
            x = image(root, row)
            for d in dtypes:
                b = session.backend(d, row, x, check, float_check=False)
                for c in CONTEXTS:
                    s = b.score_official(x, c)
                    out["scores"][f"{d}|{row['scene_id']}|{c}"] = {"margin": s["margin"], "nll": s["nll"], "chosen": s["chosen"],
                                                                    "label": str(row["answer"])}
    finally:
        if hasattr(guard, "stop"):
            guard.stop()
    write_json(Path(args.out) if args.out else root / OUT / "audit" / f"clean_{device.type}.json", out)
    print("clean scores written:", len(out["scores"]))


def cmd_fd(args) -> None:
    """G7: central finite differences of the frozen NLL on the frozen differentiable path."""
    root = repo_root()
    device = torch.device(args.device)
    guard, _, check = make_guard(device, None)  # preload lines before any load
    session = Session(args.device, (args.dtype,))
    gen = torch.Generator().manual_seed(20261112)
    res = {}
    try:
        for row in scenes(root)[:FD_SCENES]:
            x = image(root, row)
            b = session.backend(args.dtype, row, x, check, float_check=False)
            g = b.grad_nll(x, "neutral")
            f0 = b.score_float(x, "neutral")["nll"]
            dirs = []
            for _ in range(FD_DIRECTIONS):
                u = torch.randint(0, 2, x.shape, generator=gen).float().mul(2).sub(1)
                an = float((g.double() * u.double()).sum())
                steps = []
                for hl in FD_STEPS_LEVELS:
                    h = hl / 255
                    fp = b.score_float(x + h * u, "neutral")["nll"]
                    fm = b.score_float(x - h * u, "neutral")["nll"]
                    fd = (fp - fm) / (2 * h)
                    steps.append({"h_levels": hl, "fd": fd, "rel_err": abs(fd - an) / max(abs(fd) + abs(an), 1e-12)})
                dirs.append({"analytic": an, "steps": steps, "best_rel_err": min(s["rel_err"] for s in steps)})
            res[row["scene_id"]] = {"f0": f0, "directions": dirs}
    finally:
        if hasattr(guard, "stop"):
            guard.stop()
    by_h = {hl: float(np.median([d["steps"][i]["rel_err"] for r in res.values() for d in r["directions"]]))
            for i, hl in enumerate(FD_STEPS_LEVELS)}
    out = {"dtype": args.dtype, "numerics": settings(device), "scenes": res, "median_rel_err_by_step": by_h,
           "G7_pass": min(by_h.values()) <= 0.05}
    write_json(Path(args.out) if args.out else root / OUT / "audit" / f"fd_{args.dtype}_{device.type}.json", out)
    print(json.dumps(by_h), "G7", out["G7_pass"])


def cmd_repro_mps(args) -> None:
    """Free local check: the new runner on MPS bf16 reproduces 2.3/2.3b bit-for-bit on one scene."""
    from path_probe.confirm_stage import _compare, _dev_reference
    from path_probe.run import OUT_RELATIVE, Store

    root = repo_root()
    row = scenes(root)[0]
    sid = row["scene_id"]
    device = torch.device("mps")
    guard, gconf, check = make_guard(device, None)
    try:
        session = Session("mps", ("bf16",))
        x = image(root, row)
        rows, tensors = unit(session, "bf16", row, x, check, REPRO_SEED_PREFIX)
    finally:
        guard.stop()
    ref = _dev_reference(root, sid)
    chk: dict[str, Any] = {"scene": sid, "clean": _compare({c: rows[0]["scores"][c]["official"] for c in CONTEXTS}, ref["clean"])}
    src = Store(root, OUT_RELATIVE)
    deltas = {s: src.load_tensors(f"{sid}__B__{s}__deltas") for s in CONTEXTS}
    chk["codes_equal_2_3"] = {f"{s}__{n}": bool(torch.equal(tensors[f"{s}__{n}"], deltas[s][n]))
                              for s in CONTEXTS for n in ("fgsm", "pgd1", "pgd2", "pgd3")}
    grads = src.load_tensors(f"{sid}__B__grads")
    chk["grads_equal_2_3"] = {k: bool(torch.equal(tensors[f"grad_{k}"], grads[k])) for k in grads}
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
    chk["diagnostics"] = rows[-1]
    chk["guard"] = gconf
    chk["memory_monitor"] = {k: v for k, v in guard.snapshot().items() if k != "timeline"}
    chk["numerics"] = session.numerics
    chk["git_head"] = git(root, "rev-parse", "HEAD")
    write_json(root / OUT / "repro_mps" / "repro_mps_check.json", chk)
    print("all_bit_exact:", chk["all_bit_exact"], "| scores compared:", len(per),
          "| peak GiB:", round((chk["memory_monitor"].get("process_peak_footprint") or 0) / GIB, 2))


def cmd_main(args) -> None:
    """Main run: per scene, bf16 then fp32, both models resident. Records appended per (scene, dtype)."""
    root = repo_root()
    commit = protocol_commit(root)
    device = torch.device(args.device)
    out = root / OUT / "run"
    for d in ("bf16", "fp32"):
        if (out / d / "records.jsonl").exists():
            raise Stop(f"{out}/{d}/records.jsonl exists. Single shot: no resume, no overwrite.")
    guard, gconf, check = make_guard(device, args.deadline_epoch)
    session = Session(args.device, ("bf16", "fp32"))
    order = scenes(root)
    progress: dict[str, Any] = {"schema": "checkpoint-5.progress.v1", "label": "EXPLORATORY, DEVELOPMENT ONLY",
                                "protocol_commit": commit, "git_head": git(root, "rev-parse", "HEAD"),
                                "numerics": session.numerics, "guard": gconf, "load_seconds": session.load_seconds,
                                "seed_prefix": MAIN_SEED_PREFIX, "scene_order": [r["scene_id"] for r in order],
                                "hourly_usd": args.hourly_usd, "pod_start_epoch": args.pod_start_epoch,
                                "budget_gate_usd": args.budget_gate_usd, "units": [], "status": "running"}
    t_start = time.time()
    try:
        for i, row in enumerate(order):
            x = image(root, row)
            for d in ("bf16", "fp32"):
                check()
                rows, tensors = unit(session, d, row, x, check, MAIN_SEED_PREFIX)
                append(out / d / "records.jsonl", rows)
                (out / d / "tensors").mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out / d / "tensors" / f"{row['scene_id']}.npz", **{k: v.numpy() for k, v in tensors.items()})
                mem = guard.record(f"{row['scene_id']}|{d}") if hasattr(guard, "record") else {}
                progress["units"].append({"scene": row["scene_id"], "dtype": d, "seconds": rows[-1]["seconds"], "memory": mem})
                write_json(out / "progress.json", progress)
                print(f"[main] {i + 1}/48 {row['scene_id']} {d} {rows[-1]['seconds']:.0f}s", flush=True)
            if i == 0 and args.hourly_usd:
                per_scene = time.time() - t_start
                since_pod = time.time() - (args.pod_start_epoch or t_start)
                projected = args.hourly_usd * (since_pod + per_scene * (len(order) - 1) + 1800) / 3600  # +0.5 h teardown
                progress["projection_after_first_scene"] = {"seconds_per_scene_both_dtypes": round(per_scene, 1),
                                                            "projected_total_usd": round(projected, 2)}
                write_json(out / "progress.json", progress)
                # throughput gate (amendment 2): the remaining scenes must finish before the workload deadline
                finish = time.time() + per_scene * (len(order) - 1)
                progress["projection_after_first_scene"]["projected_finish_epoch"] = round(finish, 1)
                progress["projection_after_first_scene"]["workload_deadline_epoch"] = args.deadline_epoch
                write_json(out / "progress.json", progress)
                if projected > args.budget_gate_usd:
                    raise Stop(f"G9 budget gate: projected ${projected:.2f} > ${args.budget_gate_usd:.2f}")
                if finish > args.deadline_epoch:
                    raise Stop(f"throughput gate: projected finish {finish:.0f} is after the workload deadline {args.deadline_epoch:.0f}")
        # end-of-run drift check on the first scene's clean score, both precisions
        drift = {}
        row = order[0]
        x = image(root, row)
        for d in ("bf16", "fp32"):
            first = json.loads((out / d / "records.jsonl").read_text().splitlines()[0])["scores"]["neutral"]["official"]
            again = session.backend(d, row, x, check, float_check=False).score_official(x, "neutral")
            drift[d] = again["nll"] == first["nll"] and again["margin"] == first["margin"]
        progress["drift_check_exact"] = drift
        progress["status"] = "completed"
    except Exception as exc:
        progress["status"] = "stopped"
        progress["stopped_reason"] = f"{type(exc).__name__}: {str(exc)[:500]}"
        raise
    finally:
        progress["elapsed_seconds"] = round(time.time() - t_start, 1)
        if hasattr(guard, "stop"):
            guard.stop()
        write_json(out / "progress.json", progress)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("inputs"); s.add_argument("--out"); s.add_argument("--compare")
    s = sub.add_parser("audit"); s.add_argument("--dtype", choices=DTYPES, required=True); s.add_argument("--device", default="cuda"); s.add_argument("--out")
    s = sub.add_parser("clean"); s.add_argument("--dtypes", default="bf16,fp32"); s.add_argument("--device", default="cuda"); s.add_argument("--out")
    s = sub.add_parser("fd"); s.add_argument("--dtype", choices=DTYPES, default="fp32"); s.add_argument("--device", default="cuda"); s.add_argument("--out")
    sub.add_parser("repro-mps")
    s = sub.add_parser("main"); s.add_argument("--device", default="cuda"); s.add_argument("--hourly-usd", type=float, required=True)
    s.add_argument("--pod-start-epoch", type=float, required=True); s.add_argument("--deadline-epoch", type=float, required=True)
    s.add_argument("--budget-gate-usd", type=float, default=12.0)
    a = ap.parse_args()
    {"inputs": cmd_inputs, "audit": cmd_audit, "clean": cmd_clean, "fd": cmd_fd, "repro-mps": cmd_repro_mps, "main": cmd_main}[a.cmd](a)


if __name__ == "__main__":
    main()
