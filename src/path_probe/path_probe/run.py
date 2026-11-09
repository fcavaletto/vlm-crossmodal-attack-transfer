"""Checkpoint 2.3 exploratory run. DEVELOPMENT ONLY.

``python -m path_probe q0``   model-free bf16 input survival (no weights).
``python -m path_probe run``  the guarded bf16 load, stages 1-3 (EXPLORATORY_PLAN.md).

Frozen outputs are read, never written. Outputs go to
``docs/checkpoint-2.3-exploratory/paths/``. The Checkpoint 1.5 memory budget and
the 6 GiB OS floor are reused as they are.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PLAN_RELATIVE = Path("docs/checkpoint-2.3-exploratory/EXPLORATORY_PLAN.md")
OUT_RELATIVE = Path("docs/checkpoint-2.3-exploratory/paths")
SMOKE_RELATIVE = Path("docs/checkpoint-2.3-exploratory/paths_smoke")
SUBSET_PREFIX = "checkpoint-2.3-exploratory-subset|"
FROZEN_SEED_PREFIX = "checkpoint-2.0-calibration"
HELPFUL_SEED_PREFIX = "checkpoint-2.3-exploratory|helpful"
Q0_SEED = 20261109
STAGE12_CAP_S = 3.9 * 3600
STAGE3_START_CAP_S = 4.0 * 3600
TOTAL_CAP_S = 4.6 * 3600
GIB = 1024**3


class Stop(RuntimeError):
    pass


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for c in (here, *here.parents):
        if (c / PLAN_RELATIVE).is_file():
            return c
    raise RuntimeError("the exploratory plan is not above this file")


def plan_commit(root: Path) -> str:
    """The plan must be committed and unchanged since its commit."""
    status = subprocess.run(["git", "status", "--porcelain", "--", str(PLAN_RELATIVE)], cwd=root,
                            capture_output=True, text=True, check=True).stdout.strip()
    if status:
        raise Stop("EXPLORATORY_PLAN.md has uncommitted changes. Refusing to run.")
    return subprocess.check_output(["git", "log", "-1", "--format=%H", "--", str(PLAN_RELATIVE)], cwd=root, text=True).strip()


def select_scenes(records: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    arm_a = sorted((r for r in records if str(r["scene_id"]).endswith("-a")), key=lambda r: r["counterfactual_group_id"])
    if len(arm_a) != 48 or len({r["counterfactual_group_id"] for r in arm_a}) != 48:
        raise Stop("arm a does not give one scene in each of 48 groups")
    for r in arm_a:
        if r.get("split") != "development":
            raise Stop(f"{r['scene_id']} is not development")
    subset = sorted(arm_a, key=lambda r: hashlib.sha256((SUBSET_PREFIX + r["counterfactual_group_id"]).encode()).hexdigest())[:16]
    s16 = sorted(r["scene_id"] for r in subset)
    rest = [r["scene_id"] for r in arm_a if r["scene_id"] not in s16]
    return s16, s16 + rest


def _json_ready(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _json_ready(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_ready(x) for x in v]
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


class Store:
    def __init__(self, root: Path, relative: Path = OUT_RELATIVE):
        self.dir = root / relative
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "tensors").mkdir(exist_ok=True)
        gi = self.dir / ".gitignore"
        if not gi.is_file():
            gi.write_text("tensors/\nresident.json\nrun.log\n")
        self.records = self.dir / "records.jsonl"
        self.progress_path = self.dir / "progress.json"

    def append(self, rows: list[dict[str, Any]]) -> None:
        with self.records.open("a", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(_json_ready(r), sort_keys=True, allow_nan=False) + "\n")

    def save_progress(self, progress: dict[str, Any]) -> None:
        self.progress_path.write_text(json.dumps(_json_ready(progress), indent=2, allow_nan=False) + "\n")

    def save_tensors(self, name: str, tensors: dict[str, Any]) -> None:
        import numpy as np
        arrays = {k: v.detach().cpu().numpy() for k, v in tensors.items()}
        np.savez_compressed(self.dir / "tensors" / f"{name}.npz", **arrays)

    def load_tensors(self, name: str) -> dict[str, Any]:
        import numpy as np
        import torch
        with np.load(self.dir / "tensors" / f"{name}.npz") as data:
            return {k: torch.from_numpy(data[k].copy()) for k in data.files}


def q0(root: Path) -> dict[str, Any]:
    """Model-free: bf16 survival of RGB perturbations in the normalized pixel tensor."""
    from attack_calibration.scene import _development_image
    from dev_pilot.fixtures import load_manifest
    from qwen_rgb.measure import load_processor
    from qwen_rgb.rgb_path import assert_pinned_processor, differentiable_model_pixels

    from .precision import bf16_input_survival

    plan_commit(root)
    records = {r["scene_id"]: r for r in load_manifest(root)}
    _, order = select_scenes(list(records.values()))
    _processor, image_processor = load_processor()
    assert_pinned_processor(image_processor)
    fractions = (0.125, 0.25, 0.5, 0.75, 1.0)
    rows = []
    for sid in order:
        image = _development_image(root, records[sid])
        res = bf16_input_survival(lambda im: differentiable_model_pixels(im, image_processor)[0], image, 1 / 255,
                                  fractions, Q0_SEED)
        rows.append({"scene_id": sid, "rows": res})
    p0 = differentiable_model_pixels(_development_image(root, records[order[0]]), image_processor)[0]
    out = {"schema": "checkpoint-2.3.q0.v1", "status": "exploratory, model-free", "seed": Q0_SEED,
           "image_mean": list(image_processor.image_mean), "image_std": list(image_processor.image_std),
           "pixel_values_abs_median": float(p0.abs().median()), "pixel_values_abs_max": float(p0.abs().max()),
           "scenes": rows}
    store = Store(root)
    (store.dir / "q0.json").write_text(json.dumps(_json_ready(out), indent=1) + "\n")
    return out


def main(argv: list[str]) -> None:
    root = repo_root()
    if argv[:1] == ["q0"]:
        out = q0(root)
        t1 = [r["rows"] for r in out["scenes"]]
        for i, t in enumerate((0.125, 0.25, 0.5, 0.75, 1.0)):
            vals = sorted(rr[i]["frac_changed_bf16_among_moved"] for rr in t1)
            print(f"t={t}: median frac of moved coords whose bf16 value changes = {vals[len(vals)//2]:.4f}")
        return
    if argv[:1] in (["masks"], ["masks-smoke"]):
        from .mask_stage import execute_masks
        execute_masks(root, smoke=argv[:1] == ["masks-smoke"])
        return
    if argv[:1] in (["confirm"], ["confirm-smoke-dev"]):
        from .confirm_stage import execute_confirm
        execute_confirm(root, smoke_dev=argv[:1] == ["confirm-smoke-dev"])
        return
    if argv[:1] not in (["run"], ["smoke"]):
        raise SystemExit("usage: python -m path_probe {q0|smoke|run|masks|masks-smoke|confirm-smoke-dev|confirm}")
    from .stages import execute
    policy = "frozen"
    if "--memory-policy" in argv:
        policy = argv[argv.index("--memory-policy") + 1]
    execute(root, smoke=argv[:1] == ["smoke"], policy=policy)


if __name__ == "__main__":
    main(sys.argv[1:])
