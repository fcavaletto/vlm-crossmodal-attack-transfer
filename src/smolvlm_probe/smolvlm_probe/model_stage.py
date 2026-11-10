"""Stage 0 model gates for SmolVLM2-2.2B (Checkpoint 6). EXPLORATORY, development data only. No attacks.

    python -m smolvlm_probe.model_stage verify
    python -m smolvlm_probe.model_stage smoke --dtype bf16 --tiling split
    python -m smolvlm_probe.model_stage clean --dtype bf16 --tiling single
    python -m smolvlm_probe.model_stage grad  --dtype fp32 --tiling single

Memory safeguards are the existing ones, unchanged: the EMP-1 preload lines and the
path_probe MemoryMonitor (10 GiB footprint abort, pressure and swap rules). In addition the
MPS allocator is capped at the same 10 GiB, so an oversized pass raises an out-of-memory
error instead of swapping. Nothing here lowers or raises a line.
"""

from __future__ import annotations

import argparse
import shutil
import time
import traceback
from typing import Any

import numpy as np
import torch

from .common import Stop, development_scenes, git_head, repo_root, s16, scene_image, write_json
from .pins import MODEL_ID, PINNED_REVISION

GIB = 1024**3
CONTEXTS = ("neutral", "helpful_nonrevealing")


def user_text(row: dict[str, Any], ctx: str) -> str:
    from choice_eval.prompt import build_prompt

    return build_prompt(row["contexts"][ctx]["text"], row["question"]).user_text()


def guarded(fn):
    """Preload lines, monitor, MPS cap; records the outcome whatever happens."""
    def run(args) -> None:
        from path_probe.memory import FOOTPRINT_ABORT_BYTES, MemoryMonitor, emp1_preload, os_sample, process_footprint

        root = repo_root()
        name = f"{args.part}_{args.dtype}_{args.tiling}{'_smallsteps' if getattr(args, 'small_steps', False) else ''}.json"
        rec: dict[str, Any] = {"small_steps": bool(getattr(args, "small_steps", False)), "schema": "checkpoint-6.stage0.v1", "label": "EXPLORATORY, DEVELOPMENT ONLY, NO ATTACKS",
                               "part": args.part, "dtype": args.dtype, "tiling": args.tiling, "model": MODEL_ID,
                               "revision": PINNED_REVISION, "git_head": git_head(root), "torch": torch.__version__,
                               "device": "mps", "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        pre = emp1_preload(os_sample(), shutil.disk_usage(root).free)
        rec["preload"] = pre
        if not pre["permitted"]:
            rec["outcome"] = "not started: preload lines " + ", ".join(pre["failed"])
            write_json(root, name, rec); print(rec["outcome"]); return
        cap = FOOTPRINT_ABORT_BYTES / torch.mps.recommended_max_memory()
        torch.mps.set_per_process_memory_fraction(min(cap, 1.0))
        rec["mps_cap_bytes"] = int(min(cap, 1.0) * torch.mps.recommended_max_memory())
        monitor = MemoryMonitor()
        monitor.start()
        t0 = time.time()
        try:
            rec.update(fn(args, root, monitor))
            rec["outcome"] = "completed"
        except Exception as exc:  # recorded, never hidden
            text = str(exc).lower()
            memory = "out of memory" in text or "memory abort" in text or ("mps" in text and "alloc" in text)
            rec["outcome"] = ("failed: memory" if memory else "failed: error") + f": {type(exc).__name__}: {str(exc)[:400]}"
            rec["traceback"] = traceback.format_exc()[-2000:]
        finally:
            monitor.stop()
            rec["seconds"] = round(time.time() - t0, 1)
            rec["memory"] = {k: v for k, v in monitor.snapshot().items() if k != "timeline"}
            fp = process_footprint() or {}
            rec["memory"]["lifetime_peak_footprint_gib"] = round(fp.get("lifetime_max", 0) / GIB, 3)
            rec["memory"]["abort_line_gib"] = FOOTPRINT_ABORT_BYTES / GIB
            write_json(root, name, rec)
            print(name, rec["outcome"], rec["seconds"], "s, peak", rec["memory"]["lifetime_peak_footprint_gib"], "GiB")
    return run


def _load(args, monitor):
    from path_probe.memory import process_footprint

    from .backend import SmolVLM

    m = SmolVLM(args.dtype, args.tiling)
    t = time.time()
    info = m.load()
    monitor.check()
    info["load_seconds"] = round(time.time() - t, 1)
    info["footprint_after_load_gib"] = round((process_footprint() or {}).get("current", 0) / GIB, 3)
    return m, info


def verify(args) -> None:
    """sha256 of the pinned weight shards against the repository's LFS ids (no model load)."""
    from .common import verify_weights
    root = repo_root()
    t = time.time()
    got = verify_weights()
    write_json(root, "weights_sha256.json", {"schema": "checkpoint-6.stage0.weights.v1", "model": MODEL_ID,
                                             "revision": PINNED_REVISION, "sha256": got, "matches_pinned": True,
                                             "seconds": round(time.time() - t, 1)})
    print("weights verified", got)


@guarded
def smoke(args, root, monitor) -> dict[str, Any]:
    from path_probe.memory import process_footprint

    rows = {r["scene_id"]: r for r in development_scenes(root)}
    sid = s16(list(rows.values()))[0]
    row = rows[sid]
    m, info = _load(args, monitor)
    img = scene_image(root, row)
    p = m.prompt(img, user_text(row, "neutral"))
    out: dict[str, Any] = {"load": info, "scene": sid, "n_input_tokens": int(p["input_ids"].shape[-1])}
    t = time.time(); out["official"] = m.score(img, p, row["answer"], "official"); out["t_forward_s"] = round(time.time() - t, 2)
    monitor.check()
    out["surrogate"] = m.score(img, p, row["answer"], "surrogate")
    out["footprint_peak_after_forwards_gib"] = round((process_footprint() or {}).get("lifetime_max", 0) / GIB, 3)
    t = time.time(); v, g = m.gradient(img, p, row["answer"]); out["t_backward_s"] = round(time.time() - t, 2)
    monitor.check()
    v2, g2 = m.gradient(img, p, row["answer"])
    out["gradient"] = {"value": v, "norm": float(g.norm()), "nonzero_frac": float((g != 0).float().mean()),
                       "repeat_bit_identical": bool(torch.equal(g, g2)), "repeat_max_abs_diff": float((g - g2).abs().max()),
                       "repeat_value_identical": v == v2}
    out["official_repeat_identical"] = m.score(img, p, row["answer"], "official") == out["official"]
    return out


@guarded
def clean(args, root, monitor) -> dict[str, Any]:
    """Clean accuracy on all 96 development scenes x 2 contexts, image dependence, scorer checks."""
    rows = development_scenes(root)
    m, info = _load(args, monitor)
    gray = torch.full((3, 384, 512), 128 / 255)
    by_group: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_group.setdefault(r["counterfactual_group_id"], []).append(r)
    items = []
    t = time.time()
    for row in rows:
        img = scene_image(root, row)
        for ctx in CONTEXTS:
            p = m.prompt(img, user_text(row, ctx))
            off = m.score(img, p, row["answer"], "official")
            sur = m.score(img, p, row["answer"], "surrogate")
            pg = m.prompt(gray, user_text(row, ctx))
            gr = m.score(gray, pg, row["answer"], "official")
            # controlled image swap: the counterfactual partner's image with this row's own text
            partner = next(r for r in by_group[row["counterfactual_group_id"]] if r["scene_id"] != row["scene_id"])
            pimg = scene_image(root, partner)
            if user_text(partner, ctx) != user_text(row, ctx):
                raise Stop("counterfactual partners do not share the prompt text")
            sw = m.score(pimg, m.prompt(pimg, user_text(row, ctx)), partner["answer"], "official")
            items.append({"scene_id": row["scene_id"], "group": row["counterfactual_group_id"], "arm": row["scene_id"][-1],
                          "context": ctx, "label": row["answer"], "official": off, "surrogate": sur, "gray": gr,
                          "swap": {"partner": partner["scene_id"], "partner_label": partner["answer"], **sw}})
            monitor.check()
    secs = time.time() - t
    # scorer repeat on the first four items, bit-exact
    rep = []
    for it in items[:4]:
        row = next(r for r in rows if r["scene_id"] == it["scene_id"])
        img = scene_image(root, row)
        rep.append(m.score(img, m.prompt(img, user_text(row, it["context"])), it["label"], "official") == it["official"])
    return {"load": info, "items": items, "seconds_per_item": round(secs / len(items), 2),
            "official_repeat_bit_identical": rep, "summary": summarize_clean(items)}


def summarize_clean(items: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ctx in CONTEXTS:
        its = [i for i in items if i["context"] == ctx]
        correct = [i["official"]["chosen"] == i["label"] for i in its]
        margins = [i["official"]["margin"] for i in its if i["official"]["chosen"] == i["label"]]
        groups: dict[str, list[bool]] = {}
        for i, c in zip(its, correct):
            groups.setdefault(i["group"], []).append(c)
        gray_choice = [i["gray"]["chosen"] for i in its]
        out[ctx] = {
            "n": len(its), "accuracy": float(np.mean(correct)),
            "accuracy_arm_a": float(np.mean([c for i, c in zip(its, correct) if i["arm"] == "a"])),
            "groups_both_arms_correct": int(sum(all(v) for v in groups.values())), "n_groups": len(groups),
            "median_margin_correct": float(np.median(margins)) if margins else None,
            "gray_accuracy": float(np.mean([i["gray"]["chosen"] == i["label"] for i in its])),
            "gray_choice_counts": {c: gray_choice.count(c) for c in sorted(set(gray_choice))},
            "swap_follows_image": float(np.mean([i["swap"]["chosen"] == i["swap"]["partner_label"] for i in its])),
            "swap_keeps_original_label": float(np.mean([i["swap"]["chosen"] == i["label"] for i in its])),
            "swap_follows_image_given_original_correct": float(np.mean([i["swap"]["chosen"] == i["swap"]["partner_label"]
                                                                       for i, c in zip(its, correct) if c])) if any(correct) else None,
            "surrogate_vs_official_chosen_agree": float(np.mean([i["surrogate"]["chosen"] == i["official"]["chosen"] for i in its])),
            "surrogate_minus_official_margin_abs": {
                "median": float(np.median([abs(i["surrogate"]["margin"] - i["official"]["margin"]) for i in its])),
                "max": float(np.max([abs(i["surrogate"]["margin"] - i["official"]["margin"]) for i in its]))},
        }
    return out


@guarded
def grad(args, root, monitor) -> dict[str, Any]:
    """Gradient validation on 4 s16 scenes, neutral context. Diagnostics, not verdicts."""
    from .gradcheck import SMALL_STEPS_LEVELS, STEPS_LEVELS, validate

    rows = {r["scene_id"]: r for r in development_scenes(root)}
    sids = s16(list(rows.values()))[: args.n_scenes]
    m, info = _load(args, monitor)
    out = {"load": info, "scenes": {}}
    tensors = {}
    for sid in sids:
        row = rows[sid]
        img = scene_image(root, row)
        p = m.prompt(img, user_text(row, "neutral"))
        res, g = validate(m, img, p, row["answer"], monitor, SMALL_STEPS_LEVELS if args.small_steps else STEPS_LEVELS)
        out["scenes"][sid] = res
        tensors[sid] = g.numpy()
    tdir = root / "docs/checkpoint-6-smolvlm2/stage0/tensors"
    tdir.mkdir(parents=True, exist_ok=True)
    if not args.small_steps:
        np.savez_compressed(tdir / f"clean_grad_{args.dtype}_{args.tiling}.npz", **tensors)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("verify", "smoke", "clean", "grad"))
    ap.add_argument("--dtype", choices=("bf16", "fp32"))
    ap.add_argument("--tiling", choices=("split", "single"))
    ap.add_argument("--n-scenes", type=int, default=4)
    ap.add_argument("--small-steps", action="store_true", help="grad: finite differences at 1/1024 to 1/16 level")
    args = ap.parse_args()
    if args.part == "verify":
        verify(args); return
    if not (args.dtype and args.tiling):
        ap.error("--dtype and --tiling are required")
    {"smoke": smoke, "clean": clean, "grad": grad}[args.part](args)


if __name__ == "__main__":
    main()
