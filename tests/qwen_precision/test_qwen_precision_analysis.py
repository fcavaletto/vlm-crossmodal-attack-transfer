"""The Checkpoint 5 analysis on synthetic records with known effects."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("aq", ROOT / "docs/checkpoint-5-qwen-fp32/analyze_qwen_precision.py")
aq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aq)
CTX = aq.CTX


def write(d: Path, drops: dict, n=48):
    rows = []
    for i in range(n):
        sid = f"dev4-ref-{i:02d}-a"
        j = (i % 4) * 0.1
        rows.append({"kind": "clean", "scene_id": sid, "label": "red",
                     "scores": {c: {"official": {"margin": 10.0, "chosen": "red"}, "float": {"margin": 10.0}} for c in CTX}})
        for src in CTX:
            rows.append({"kind": "structure", "scene_id": sid, "source": src, "sign_reversals": 0, "pgd_levels_max": 1,
                         "pgd_outside_fgsm": 0, "outside_where_clean_grad_zero": 0})
            for k, key in (("fgsm", "fgsm"), ("pgd3", "pgd"), ("topk_grad", "topk")):
                rows.append({"kind": k, "scene_id": sid, "source": src,
                             "scores": {c: {"margin": 10 - drops[key] - j, "chosen": "red"} for c in CTX}})
            for s in range(8):
                rows.append({"kind": "random", "scene_id": sid, "source": src, "seed_index": s,
                             "scores": {c: {"margin": 10 - drops["rand"] - j - 0.01 * s, "chosen": "red"} for c in CTX}})
        al = {"cos_g0_g1": 0.0, "sign_g0_g1": 0.53, "cos_g0_g2": 0.0, "cos_g1_g2": -0.03}
        rows.append({"kind": "diagnostics", "scene_id": sid, "alignment": {c: al for c in CTX},
                     "float_path": [{"kind": "fgsm", "official": {"margin": 1.0}, "float": {"margin": 1.0}}],
                     "clean_pixels_official_vs_differentiable_max_abs": 1e-6})
    d.mkdir(parents=True)
    (d / "records.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


QWEN_LIKE = {"fgsm": 3.4, "pgd": 9.8, "topk": 2.4, "rand": 4.0}


def test_outcome_A_and_equivalence(tmp_path):
    write(tmp_path / "bf16", QWEN_LIKE)
    write(tmp_path / "fp32", {**QWEN_LIKE, "pgd": 9.9})
    out = aq.analyze(tmp_path)
    r = out["readings"]
    assert r["reproduction_band_met"] is True
    assert r["A_both_persist"] is True and not r["B_selection_persists_inversion_disappears"]
    # a 0.1-logit shift with no noise: non-zero, but well inside the +/-1.0 margin
    assert out["paired_fp32_minus_bf16"]["neutral"]["selection"]["verdict"] == "non-zero but within the equivalence margin"
    assert out["paired_fp32_minus_bf16"]["neutral"]["topk_minus_rand"]["verdict"] == "equivalent within margin"


def test_outcome_B(tmp_path):
    write(tmp_path / "bf16", QWEN_LIKE)
    write(tmp_path / "fp32", {**QWEN_LIKE, "topk": 4.1})
    r = aq.analyze(tmp_path)["readings"]
    assert r["B_selection_persists_inversion_disappears"] is True and r["A_both_persist"] is False


def test_outcome_C(tmp_path):
    write(tmp_path / "bf16", QWEN_LIKE)
    write(tmp_path / "fp32", {"fgsm": 3.4, "pgd": 6.0, "topk": 3.6, "rand": 4.0})
    r = aq.analyze(tmp_path)["readings"]
    assert r["C_both_substantially_weaker"] is True and not r["A_both_persist"]


def test_outcome_D_when_band_missed(tmp_path):
    write(tmp_path / "bf16", {**QWEN_LIKE, "pgd": 6.0})
    write(tmp_path / "fp32", {**QWEN_LIKE, "pgd": 6.0})
    r = aq.analyze(tmp_path)["readings"]
    assert r["reproduction_band_met"] is False and r["D_uncontrolled_or_not_reproduced"] is True
    assert not r["A_both_persist"]


def test_verdicts():
    b = lambda e, c95, c90: {"estimate": e, "ci95": c95, "ci90": c90}  # noqa: E731
    assert aq.verdict(b(0.1, [-0.4, 0.6], [-0.3, 0.5]), 1.0) == "equivalent within margin"
    assert aq.verdict(b(0.1, [-1.4, 1.6], [-1.2, 1.4]), 1.0) == "inconclusive"
    assert aq.verdict(b(2.0, [1.4, 2.6], [1.5, 2.5]), 1.0) == "different, not within the equivalence margin"
