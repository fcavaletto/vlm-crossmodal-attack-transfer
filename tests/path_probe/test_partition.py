"""Checkpoint 3b: splits, four-score decomposition and analysis, on synthetic records. No model."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from choice_eval.decision import choose_color
from path_probe import partition as pt
from qwen_rgb_loss.records import clean_margin

LABEL = "red"
CLEAN = {"red": 0.1, "blue": 3.0, "green": 4.0, "yellow": 5.0}  # clean competitor: blue; margin 2.9


def score(s: dict[str, float], label: str = LABEL) -> dict:
    s = {c: float(v) for c, v in s.items()}
    return {"summed_nll": s, "margin": clean_margin(s, label), "chosen": choose_color(s), "nll": s[label]}


def damaged(d: float, base: dict = CLEAN) -> dict[str, float]:
    """Fixed-contrast damage d (blue minus red falls by d), no change of strongest rival."""
    return {**base, "red": base["red"] + d / 2, "blue": base["blue"] - d / 2}


def scene_rows(sid: str, k: int, damage, clean: dict = CLEAN, override=None) -> list[dict]:
    """``damage(family, draw, piece)`` gives the fixed-contrast damage of each image; ``override`` may
    replace an image's four scores."""
    sizes = pt.quarter_sizes(k)
    rows = [{"kind": "image", "scene_id": sid, "label": LABEL, "family": "clean", "draw": 0, "piece": "clean",
             "count": 0, "n_pos": 0, "n_neg": 0, "scores": {"neutral": score(clean)}}]
    keys = [("pgd", 0, "full")] + [("pgd", a, str(j)) for a in range(2) for j in range(4)]
    keys += [("random", s, p) for s in range(8) for p in ["full", "0", "1", "2", "3"]]
    for fam, d, p in keys:
        n = k if p == "full" else sizes[int(p)]
        s = damaged(damage(fam, d, p), clean)
        if override and (fam, d, p) in override:
            s = override[(fam, d, p)]
        rows.append({"kind": "image", "scene_id": sid, "label": LABEL, "family": fam, "draw": d, "piece": p,
                     "count": n, "n_pos": n - n // 3, "n_neg": n // 3, "scores": {"neutral": score(s)}})
    return rows


PIECE = [1.0, 0.5, 2.0, 1.5]
RPIECE = [0.4, 0.3, 0.5, 0.2]


def additive(fam, d, p):
    w = PIECE if fam == "pgd" else RPIECE
    return sum(w) if p == "full" else w[int(p)]


# ---------------------------------------------------------------- splits

@pytest.mark.parametrize("k,sizes", [(8, [2, 2, 2, 2]), (10, [3, 3, 2, 2]), (7, [2, 2, 2, 1]), (13, [4, 3, 3, 3])])
def test_quarter_sizes_when_k_is_not_divisible_by_four(k, sizes):
    assert pt.quarter_sizes(k) == sizes and sum(sizes) == k


def test_quarter_sizes_refuses_tiny_masks():
    with pytest.raises(pt.PartitionError):
        pt.quarter_sizes(3)


def test_split_is_disjoint_complete_seeded_and_sized():
    g = torch.Generator().manual_seed(0)
    mask = torch.rand((3, 7, 5), generator=g) < 0.3
    k = int(mask.sum())
    pieces = pt.split(mask, 123)
    pt.check_split(mask, pieces)
    assert [int(p.sum()) for p in pieces] == pt.quarter_sizes(k)
    again = pt.split(mask, 123)
    other = pt.split(mask, 124)
    assert all(torch.equal(a, b) for a, b in zip(pieces, again))
    assert not all(torch.equal(a, b) for a, b in zip(pieces, other))


def test_split_with_k_ten_gives_three_three_two_two():
    mask = torch.zeros(40, dtype=torch.bool)
    mask[torch.tensor([1, 3, 4, 9, 11, 20, 21, 30, 33, 39])] = True
    pieces = pt.split(mask, 7)
    assert [int(p.sum()) for p in pieces] == [3, 3, 2, 2]
    pt.check_split(mask, pieces)


def test_check_split_catches_overlap_and_leaks():
    mask = torch.zeros(12, dtype=torch.bool)
    mask[:8] = True
    pieces = pt.split(mask, 1)
    bad = [pieces[0] | pieces[1], pieces[1], pieces[2], pieces[3]]
    with pytest.raises(pt.PartitionError):
        pt.check_split(mask, bad)
    leak = [p.clone() for p in pieces]
    leak[0][10] = True
    with pytest.raises(pt.PartitionError):
        pt.check_split(mask, leak)


def test_sign_counts_and_pgd_structure():
    L = 1 / 255
    r_f = torch.tensor([L, -L, L, -L, 0.0, L])
    r_p = torch.tensor([L, -L, 0.0, 0.0, L, 0.0])  # index 4: a tie coordinate outside FGSM's support
    st = pt.pgd_structure(r_f, r_p, L)
    assert st["k_retained"] == 2 and st["n_excluded_outside_fgsm"] == 1 and st["sign_disagreements_on_overlap"] == 0
    assert pt.sign_counts(r_f, st["P"]) == {"count": 2, "n_pos": 1, "n_neg": 1}
    with pytest.raises(pt.PartitionError, match="disagree in sign"):
        pt.pgd_structure(r_f, torch.tensor([-L, -L, 0, 0, 0, 0]), L)
    with pytest.raises(pt.PartitionError, match="levels"):
        pt.pgd_structure(r_f, torch.tensor([2 * L, 0, 0, 0, 0, 0]), L)


def test_image_change_check():
    from path_probe import masks as mk
    L = 1 / 255
    x = torch.tensor([0.0, 10 * L, 254 * L, 100 * L, 3 * L, 7 * L]).view(3, 1, 2)
    r_f = torch.tensor([L, -L, L, L, -L, L]).view(3, 1, 2)
    m = torch.tensor([True, False, True, True, False, True]).view(3, 1, 2)
    pt.image_change_check(x, mk.apply(x, r_f, m), r_f, m, L)
    with pytest.raises(pt.PartitionError):
        pt.image_change_check(x, mk.apply(x, r_f, m), r_f, torch.ones_like(m), L)


# ---------------------------------------------------------------- scores, ties, switching

def test_competitor_tie_breaks_by_colour_name():
    assert pt.competitor({"red": 0.1, "blue": 3.0, "green": 3.0, "yellow": 5.0}, "red") == "blue"
    assert pt.competitor({"red": 0.1, "blue": 4.0, "green": 3.0, "yellow": 3.0}, "red") == "green"
    assert pt.competitor({"red": 3.0, "blue": 0.1, "green": 3.0, "yellow": 3.0}, "blue") == "green"


def test_identity_check_catches_tampered_margin_and_decision():
    s = score(damaged(1.0))
    pt.identity_check(s, LABEL)
    with pytest.raises(pt.PartitionError, match="margin"):
        pt.identity_check({**s, "margin": s["margin"] + 1e-12}, LABEL)
    with pytest.raises(pt.PartitionError, match="decision"):
        pt.identity_check({**s, "chosen": "yellow"}, LABEL)


def test_exact_tie_label_with_rival():
    tied = {"red": 2.0, "blue": 2.0, "green": 4.0, "yellow": 5.0}
    s = score(tied)
    assert s["margin"] == 0.0 and s["chosen"] == "blue"  # equal scores: blue precedes red by name
    out = pt.decompose(score(CLEAN), s, LABEL)
    assert out["label_ties_rival"] and out["switching"] == 0.0
    assert out["fixed"] == pytest.approx(2.9) and out["margin"] == pytest.approx(2.9)


def test_rival_switch_decomposition_is_exact():
    # Green overtakes blue as the strongest rival: margin damage = fixed damage + switching term.
    img = {"red": 1.0, "blue": 2.5, "green": 0.5, "yellow": 5.0}
    out = pt.decompose(score(CLEAN), score(img), LABEL)
    assert out["rival_switch"]
    assert out["fixed"] == pytest.approx(2.9 - 1.5)
    assert out["switching"] == pytest.approx(2.0)
    assert out["margin"] == pytest.approx(out["fixed"] + out["switching"], abs=1e-12)


def test_clean_rival_tie_is_reported_and_c0_follows_name_rule():
    clean = {"red": 0.1, "blue": 3.0, "green": 3.0, "yellow": 5.0}
    q = pt.scene_quantities(scene_rows("s", 10, additive, clean=clean))
    assert q["clean_rival_tie"] and q["clean_competitor"] == "blue"


# ---------------------------------------------------------------- per scene

def test_additive_example_gives_zero_delta():
    q = pt.scene_quantities(scene_rows("s", 10, additive))
    f = q["fixed"]
    assert f["pgd_full_minus_sum"] == pytest.approx(0, abs=1e-12)
    assert f["random_full_minus_sum"] == pytest.approx(0, abs=1e-12)
    assert f["A"] == pytest.approx(sum(PIECE) - sum(RPIECE))
    assert f["Q"] == pytest.approx(f["A"]) and f["Delta"] == pytest.approx(0, abs=1e-12)
    assert q["switching"]["Delta"] == 0 and q["quarter_sizes"] == [3, 3, 2, 2]


def test_positive_delta_from_pgd_additive_and_random_subadditive():
    # PGD's set is exactly additive; random sets are sub-additive (full = half the sum of their quarters).
    # Delta is positive although nothing about PGD is super-additive.
    def dmg(fam, d, p):
        if fam == "random" and p == "full":
            return 0.5 * sum(RPIECE)
        return additive(fam, d, p)
    q = pt.scene_quantities(scene_rows("s", 10, dmg))["fixed"]
    assert q["pgd_full_minus_sum"] == pytest.approx(0, abs=1e-12)
    assert q["random_full_minus_sum"] == pytest.approx(-0.5 * sum(RPIECE))
    assert q["Delta"] > 0 and q["Delta"] == pytest.approx(0.5 * sum(RPIECE))
    assert q["Delta"] == pytest.approx(q["pgd_full_minus_sum"] - q["random_full_minus_sum"])


def test_two_pgd_splits_and_eight_seeds_are_averaged():
    def dmg(fam, d, p):
        if fam == "pgd" and p != "full":
            return [1.0, 1.0, 1.0, 1.0][int(p)] * (1 + d)  # split 1 sums to twice split 0
        if fam == "random" and p != "full":
            return 0.1 * (d + 1)
        return 4.0 if fam == "pgd" else 0.0
    q = pt.scene_quantities(scene_rows("s", 12, dmg))["fixed"]
    assert q["pgd_sum_quarters"] == pytest.approx((4 + 8) / 2)
    assert q["random_sum_quarters"] == pytest.approx(sum(0.4 * (s + 1) for s in range(8)) / 8)


def test_switching_identity_for_A_Q_and_Delta_with_a_rival_switch():
    switch = {("pgd", 0, "full"): {"red": 1.0, "blue": 2.5, "green": 0.5, "yellow": 5.0},
              ("random", 3, "2"): {"red": 0.6, "blue": 2.9, "green": 1.0, "yellow": 5.0}}
    q = pt.scene_quantities(scene_rows("s", 10, additive, override=switch))
    assert q["rival_switches"] == {"pgd": 1, "random": 1}
    assert q["switching"]["A"] > 0 and q["switching"]["Q"] < 0
    for name in ("A", "Q", "Delta", "pgd_full_minus_sum", "random_full_minus_sum"):
        assert q["margin"][name] == pytest.approx(q["fixed"][name] + q["switching"][name], abs=1e-12)


def test_scene_refuses_count_mismatch_and_missing_images():
    rows = scene_rows("s", 10, additive)
    bad = [dict(r) for r in rows]
    next(r for r in bad if r["family"] == "random" and r["piece"] == "3")["count"] = 3
    with pytest.raises(pt.PartitionError, match="count"):
        pt.scene_quantities(bad)
    with pytest.raises(pt.PartitionError, match="incomplete scene"):
        pt.scene_quantities(rows[:-1])


# ---------------------------------------------------------------- panel

def _panel_scenes(n=16, shift=0.0):
    out = []
    for i in range(n):
        def dmg(fam, d, p, i=i):
            if fam == "pgd" and p == "full":
                return sum(PIECE) + shift + 0.1 * (i % 3)
            return additive(fam, d, p)
        out.append(pt.scene_quantities(scene_rows(f"s{i:02d}", 10 + i, dmg)))
    return out


def test_panel_paired_bootstrap_directions_and_rho():
    scenes = _panel_scenes(shift=1.0)
    res = pt.panel(scenes, [s["scene_id"] for s in scenes], draws=999)
    d = res["fixed"]["Delta"]
    assert d["direction"] == "positive" and d["ci95"][0] <= d["mean"] <= d["ci95"][1]
    assert res["fixed"]["random_full_minus_sum"]["direction"] == "unresolved"  # exactly 0 everywhere
    assert res["rho"]["defined"] and res["rho"]["estimate"] == pytest.approx(
        sum(s["fixed"]["Q"] for s in scenes) / sum(s["fixed"]["A"] for s in scenes))
    again = pt.panel(scenes, [s["scene_id"] for s in scenes], draws=999)
    assert again == res  # fixed seed


def test_rho_undefined_when_denominator_reaches_zero():
    scenes = []
    for i in range(16):
        def dmg(fam, d, p, i=i):
            if fam == "pgd":
                return additive("random", d, p) + (0.5 if i % 2 else -0.5) * (p == "full")
            return additive(fam, d, p)
        scenes.append(pt.scene_quantities(scene_rows(f"s{i:02d}", 12, dmg)))
    res = pt.panel(scenes, [s["scene_id"] for s in scenes], draws=999)
    assert res["rho"]["defined"] is False


def test_panel_refuses_incomplete_panel():
    scenes = _panel_scenes(n=3)
    with pytest.raises(pt.PartitionError, match="incomplete"):
        pt.panel(scenes[:2], [s["scene_id"] for s in scenes])


# ---------------------------------------------------------------- the analysis script

def _script():
    path = Path(__file__).resolve().parents[2] / "docs/checkpoint-3b-partition/analyze_partition.py"
    spec = importlib.util.spec_from_file_location("analyze_partition", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_run(tmp_path, status, planned=None, record_scenes=None):
    """A synthetic run (records are made up; no real smoke or pilot output is read)."""
    planned = list(_script().FROZEN_S16) if planned is None else planned
    rows = []
    for sid in (planned if record_scenes is None else record_scenes):
        rows += scene_rows(sid, 10, additive)
        rows.append({"kind": "structure", "scene_id": sid, "k_retained": 10, "n_pgd_support": 11, "n_excluded_outside_fgsm": 1,
                     "sign_disagreements_on_overlap": 0, "pgd_sign_counts": {}, "quarter_sizes": [3, 3, 2, 2]})
        rows.append({"kind": "pgd_official", "scene_id": sid, "bit_exact_vs_record": True})
    (tmp_path / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "progress.json").write_text(json.dumps({"status": status, "planned_scenes": planned, "scene_order": planned,
                                                        "plan_commit": "x", "drift_check": {"exact": True}}))


def test_canonical_s16_matches_the_frozen_list():
    mod = _script()
    assert mod.canonical_s16() == list(mod.FROZEN_S16) and len(set(mod.FROZEN_S16)) == 16


def test_analysis_script_runs_on_a_complete_synthetic_run(tmp_path):
    _write_run(tmp_path, "completed")
    res = _script().analyze(tmp_path)
    assert res["panel"]["n_scenes"] == 16 and res["structure"]["dev4-ref-00-a"]["n_excluded_outside_fgsm"] == 1
    assert res["panel"]["fixed"]["Delta"]["mean"] == pytest.approx(0, abs=1e-12)


def test_a_completed_one_scene_smoke_is_rejected(tmp_path):
    # A successful smoke is marked "completed" and its progress file plans exactly its one scene.
    _write_run(tmp_path, "completed", planned=["dev4-ref-00-a"])
    with pytest.raises(SystemExit, match="frozen s16"):
        _script().analyze(tmp_path)
    out = Path(__file__).resolve().parents[2] / "docs/checkpoint-3b-partition/analysis/partition_analysis.json"
    assert not out.exists()


def test_records_covering_other_scenes_are_rejected(tmp_path):
    mod = _script()
    s16 = list(mod.FROZEN_S16)
    _write_run(tmp_path, "completed", record_scenes=s16[:15])  # one planned scene has no records
    with pytest.raises(SystemExit, match="cover 15 scenes"):
        mod.analyze(tmp_path)
    _write_run(tmp_path, "completed", record_scenes=s16 + ["dev4-ref-01-a"])  # an extra scene
    with pytest.raises(SystemExit, match="cover 17 scenes"):
        mod.analyze(tmp_path)
    _write_run(tmp_path, "completed", planned=s16[::-1])  # right scenes, wrong order
    with pytest.raises(SystemExit, match="frozen s16"):
        mod.analyze(tmp_path)


@pytest.mark.parametrize("status", ["incomplete_time_cap", "stopped", "drift_check_failed"])
def test_analysis_script_refuses_incomplete_runs(tmp_path, status):
    _write_run(tmp_path, status)
    with pytest.raises(SystemExit, match="INCOMPLETE"):
        _script().analyze(tmp_path)
