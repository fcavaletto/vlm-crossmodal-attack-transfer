"""Gate evaluator and gradient-similarity script on synthetic inputs."""

import json
from pathlib import Path

import numpy as np

from qwen_precision import compare_grads, gates

ROOT = Path(__file__).resolve().parents[2]


def audit(dtype, islands):
    return {"numerics": {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False, "autocast_enabled": False,
                         "deterministic_algorithms": True, "bf16_reduced_precision_reduction": False, "gpu": "A40"},
            "fp32_matmul_probe": {"max_rel_err_vs_fp64": 1e-7, "passes_1e-5": True},
            "weights": {"sha256": gates.EXPECTED_WEIGHTS_SHA256, "stored_parameter_dtypes": {"BF16": 2127532032}},
            "modules_not_in_dtype": islands,
            "determinism": {"score_repeat_identical": True, "gradient_repeat_bit_identical": True}}


def test_gates_pass_and_fail(tmp_path):
    (tmp_path / "audit_fp32_cuda.json").write_text(json.dumps(audit("fp32", {})))
    (tmp_path / "audit_bf16_cuda.json").write_text(json.dumps(audit("bf16", gates.MPS_BF16_FP32_ISLANDS)))
    (tmp_path / "input_hashes_compare.json").write_text(json.dumps({"pass": True, "mismatched": []}))
    (tmp_path / "fd_fp32_cuda.json").write_text(json.dumps({"median_rel_err_by_step": {"0.0156": 0.013}}))
    out = gates.evaluate(tmp_path, ROOT)
    assert out["missing"] == ["G5_cross_device_clean", "G6_fp32_sanity"] and not out["all_pass"]
    assert all(v["pass"] for v in out["gates"].values())
    bad = audit("fp32", {"Linear": ["bfloat16"]})
    bad["numerics"]["cuda_matmul_allow_tf32"] = True
    (tmp_path / "audit_fp32_cuda.json").write_text(json.dumps(bad))
    out = gates.evaluate(tmp_path, ROOT)
    assert not out["gates"]["G1_environment"]["pass"] and not out["gates"]["G2_dtype_audit"]["pass"]


def test_compare_grads_identical_and_flipped(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    for d in ("bf16", "fp32"):
        (tmp_path / d / "tensors").mkdir(parents=True)
    g = {f"grad_{k}": rng.standard_normal((3, 8, 8)).astype(np.float32) for k in compare_grads.KEYS}
    np.savez(tmp_path / "bf16" / "tensors" / "s.npz", **g)
    np.savez(tmp_path / "fp32" / "tensors" / "s.npz", **{k: -v for k, v in g.items()})
    monkeypatch.setattr("sys.argv", ["x", str(tmp_path)])
    compare_grads.main()
    r = json.loads((tmp_path / "gradient_similarity.json").read_text())["bf16_vs_fp32_cuda"]["scenes"]["s"]["L_neutral"]
    assert abs(r["cosine"] + 1) < 1e-9 and r["sign_agreement"] == 0.0 and abs(r["top30_overlap"] - 1) < 1e-9
