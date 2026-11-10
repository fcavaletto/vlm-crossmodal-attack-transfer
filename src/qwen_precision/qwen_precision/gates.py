"""Pilot gates G1-G3 and G5-G8 (PLAN.md section 6.1), evaluated from the audit outputs.

    python -m qwen_precision.gates <audit_dir> [--expected-weights-sha256 ...]

Writes <audit_dir>/gates.json and exits 0 only if every gate passes. Thresholds are the
pre-registered ones and are not parameters.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

EXPECTED_WEIGHTS_SHA256 = "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0"
MPS_BF16_FP32_ISLANDS = {"Qwen3VLVisionRotaryEmbedding": ["float32"]}  # reference/audit_bf16_mps.json
G6_MIN_ACCURACY = 0.75
G7_MAX_MEDIAN_REL_ERR = 0.05


def _load(p: Path) -> Any:
    return json.loads(p.read_text()) if p.is_file() else None


def evaluate(audit_dir: Path, root: Path) -> dict[str, Any]:
    a32 = _load(audit_dir / "audit_fp32_cuda.json")
    a16 = _load(audit_dir / "audit_bf16_cuda.json")
    inputs = _load(audit_dir / "input_hashes_compare.json")
    clean = _load(audit_dir / "clean_cuda.json")
    fd = _load(audit_dir / "fd_fp32_cuda.json")
    g: dict[str, Any] = {}
    if a32:
        n = a32["numerics"]
        g["G1_environment"] = {"pass": bool(a32["fp32_matmul_probe"]["passes_1e-5"] and n.get("cuda_matmul_allow_tf32") is False
                                            and n.get("cudnn_allow_tf32") is False and n.get("autocast_enabled") is False
                                            and n.get("deterministic_algorithms") is True
                                            and n.get("bf16_reduced_precision_reduction") is False),
                               "probe": a32["fp32_matmul_probe"], "gpu": n.get("gpu"), "cuda_runtime": n.get("cuda_runtime")}
    if a32 and a16:
        g["G2_dtype_audit"] = {"pass": a32["modules_not_in_dtype"] == {} and a16["modules_not_in_dtype"] == MPS_BF16_FP32_ISLANDS,
                               "fp32_not_fp32": a32["modules_not_in_dtype"], "bf16_islands": a16["modules_not_in_dtype"],
                               "bf16_islands_expected": MPS_BF16_FP32_ISLANDS}
        g["G8_determinism"] = {"pass": all(a["determinism"]["score_repeat_identical"] and a["determinism"]["gradient_repeat_bit_identical"]
                                           for a in (a16, a32)), "bf16": a16["determinism"], "fp32": a32["determinism"]}
    if inputs is not None and a32:
        w = a32["weights"]
        g["G3_inputs_and_weights"] = {"pass": bool(inputs["pass"] and w["sha256"] == EXPECTED_WEIGHTS_SHA256
                                                   and w["stored_parameter_dtypes"] == {"BF16": 2127532032}),
                                      "inputs_mismatched": inputs["mismatched"], "weights": w}
    if clean is not None:
        spec = importlib.util.spec_from_file_location("aq", root / "docs/checkpoint-5-qwen-fp32/analyze_qwen_precision.py")
        aq = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(aq)
        q, _ = aq.mps_reference()
        g5 = aq.gate_g5(clean, q)
        g["G5_cross_device_clean"] = {"pass": bool(g5 and g5["pass"]), **(g5 or {})}
        acc = {}
        for c in ("neutral", "helpful_nonrevealing"):
            items = [v for k, v in clean["scores"].items() if k.startswith("fp32|") and k.endswith("|" + c)]
            acc[c] = sum(v["chosen"] == v["label"] for v in items) / max(len(items), 1)
        g["G6_fp32_sanity"] = {"pass": all(v >= G6_MIN_ACCURACY for v in acc.values()) and len(clean["scores"]) == 192,
                               "fp32_accuracy": acc}
    if fd is not None:
        best = min(fd["median_rel_err_by_step"].values())
        g["G7_fp32_gradients"] = {"pass": best <= G7_MAX_MEDIAN_REL_ERR, "median_rel_err_by_step": fd["median_rel_err_by_step"]}
    want = ("G1_environment", "G2_dtype_audit", "G3_inputs_and_weights", "G5_cross_device_clean", "G6_fp32_sanity",
            "G7_fp32_gradients", "G8_determinism")
    missing = [k for k in want if k not in g]
    return {"gates": g, "missing": missing, "all_pass": not missing and all(v["pass"] for v in g.values())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audit_dir")
    a = ap.parse_args()
    from .run import repo_root

    d = Path(a.audit_dir)
    out = evaluate(d, repo_root())
    (d / "gates.json").write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(json.dumps({k: v["pass"] for k, v in out["gates"].items()} | {"missing": out["missing"], "ALL_PASS": out["all_pass"]}))
    sys.exit(0 if out["all_pass"] else 2)


if __name__ == "__main__":
    main()
