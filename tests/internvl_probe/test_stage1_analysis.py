"""The Stage 1 analysis on synthetic records with known effects."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("analyze_stage1", ROOT / "docs/checkpoint-4-internvl/analyze_stage1.py")
an = importlib.util.module_from_spec(spec)
spec.loader.exec_module(an)
CTX = an.CTX


def score(margin):
    return {"official": {"margin": margin, "chosen": "red" if margin > 0 else "blue"},
            "float": {"margin": margin, "chosen": "red" if margin > 0 else "blue"}}


def write(dirpath: Path, drops: dict, float_drops: dict | None = None, n=48):
    float_drops = float_drops or drops
    rows = []
    for i in range(n):
        sid = f"s{i:02d}"
        rows.append({"kind": "clean", "scene_id": sid, "label": "red", "scores": {c: score(10.0) for c in CTX}})
        for src in CTX:
            rows.append({"kind": "structure", "scene_id": sid, "source": src, "sign_reversals": 0, "pgd_levels_max": 1,
                         "pgd_outside_fgsm": 0, "outside_all_half_level_ties": True, "density_in_fgsm_support": 0.3,
                         "alignment": {"cos_g0_g1": 0.1, "sign_g0_g1": 0.55, "cos_g0_g2": 0.05, "cos_g1_g2": 0.0}})
            jitter = (i % 3) * 0.1
            for kind, key in (("fgsm", "fgsm"), ("pgd3", "pgd"), ("topk_grad", "topk")):
                sc = {c: {"official": {"margin": 10 - drops[key] - jitter, "chosen": "red"},
                          "float": {"margin": 10 - float_drops[key] - jitter, "chosen": "red"}} for c in CTX}
                rows.append({"kind": kind, "scene_id": sid, "source": src, "scores": sc})
            for s in range(8):
                sc = {c: {"official": {"margin": 10 - drops["rand"] - jitter, "chosen": "red"},
                          "float": {"margin": 10 - float_drops["rand"] - jitter, "chosen": "red"}} for c in CTX}
                rows.append({"kind": "random", "scene_id": sid, "source": src, "seed_index": s, "scores": sc})
    dirpath.mkdir(parents=True)
    (dirpath / "records.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_known_effects_and_readings(tmp_path):
    write(tmp_path / "bf16", {"fgsm": 3, "pgd": 9, "topk": 2, "rand": 3.5})
    write(tmp_path / "fp32", {"fgsm": 3, "pgd": 7, "topk": 2, "rand": 3.5}, {"fgsm": 3, "pgd": 3.5, "topk": 2, "rand": 3.5})
    out = an.analyze(tmp_path)
    b = out["margin_drop"]["bf16"]["official"]["source"]["neutral"]
    assert b["selection_pgd_minus_rand"]["estimate"] == pytest.approx(5.5)
    assert b["magnitude_topk_minus_rand"]["estimate"] == pytest.approx(-1.5)
    assert b["share_of_gap_from_sparsity"]["estimate"] == pytest.approx(0.5 / 6, abs=1e-5)
    d = out["paired"]["fp32_minus_bf16"]["official"]["neutral"]["selection"]
    assert d["estimate"] == pytest.approx(-2.0)
    r = out["readings"]
    assert r["R1_persists_in_both_precisions"] is True
    assert r["R2_much_stronger_in_bf16"] is False  # 3.5 is not below half of 5.5
    assert r["R3_only_after_official_preprocessing"] == {"bf16": False, "fp32": True}
    assert r["where_selection_works"] == {"bf16": "both paths", "fp32": "official path only"}
    assert r["R4_does_not_replicate"] is False


def test_no_replication(tmp_path):
    flat = {"fgsm": 3, "pgd": 3.5, "topk": 3.5, "rand": 3.5}
    write(tmp_path / "bf16", flat)
    write(tmp_path / "fp32", flat)
    assert an.analyze(tmp_path)["readings"]["R4_does_not_replicate"] is True


def test_qwen_reference_reproduces_committed_dev_analysis():
    q = an.load_qwen_dev()
    qs = [s for s in sorted(q["clean"]) if all(an.qwen_values(q, s, so, "source") for so in CTX)]
    v = an.arrays(lambda s, so: an.qwen_values(q, s, so, "source"), qs, "neutral")
    committed = json.loads((ROOT / "docs/checkpoint-2.3-exploratory/analysis/masks_analysis.json").read_text())
    ref = committed["margin_drop_source"]["neutral"]
    assert len(qs) == 48
    assert round(float((v["pgd"] - v["rand"]).mean()), 5) == ref["selection_pgd_minus_rand"]["mean"]["estimate"]
    assert round(float((v["topk"] - v["rand"]).mean()), 5) == ref["magnitude_topk_minus_rand"]["mean"]["estimate"]


def test_much_stronger_in_bf16(tmp_path):
    write(tmp_path / "bf16", {"fgsm": 3, "pgd": 9, "topk": 2, "rand": 3.5})
    write(tmp_path / "fp32", {"fgsm": 3, "pgd": 5, "topk": 2, "rand": 3.5})
    assert an.analyze(tmp_path)["readings"]["R2_much_stronger_in_bf16"] is True
