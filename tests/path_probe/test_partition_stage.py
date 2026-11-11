"""Checkpoint 3b runner block on a synthetic image, stored iterates and a fake scorer. No model."""

from __future__ import annotations

import pytest
import torch
from attack_feasibility.steps import pgd_update, project_signed_step
from choice_eval.decision import choose_color
from path_probe import geometry as geo
from path_probe import masks as mk
from path_probe import partition as pt
from path_probe import partition_stage as ps
from path_probe.run import Stop
from qwen_rgb_loss.records import clean_margin

EPS = 1 / 255


def _scene(seed=0, shape=(3, 12, 16)):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 256, shape, generator=g).float() / 255
    g0 = torch.randn(shape, generator=g)
    g0[0, 0, :3] = 0.0  # zero-gradient coordinates: candidate rounding ties outside FGSM's support
    codes = {"fgsm": geo.encode_delta(project_signed_step(x, g0, EPS, "positive") - x, EPS)}
    cur, grad = x.clone(), g0
    for step in range(1, 4):
        cur = pgd_update(cur, x, grad, EPS / 4, EPS)
        codes[f"pgd{step}"] = geo.encode_delta(cur - x, EPS)
        grad = torch.randn(shape, generator=g)
    return x, {"grads": {"L_neutral": g0}, "codes": codes}


class FakeBackend:
    """Deterministic four colour scores of the 8-bit image; a nonlinear (saturating) response."""
    label = "green"

    def __init__(self, x):
        self.x = x
        g = torch.Generator().manual_seed(99)
        self.w = {c: torch.randn(x.shape, generator=g) for c in pt.COLORS}
        self.calls = 0

    def score_official(self, image, ctx):
        from qwen_rgb.constraints import round_trip_uint8
        assert ctx == "neutral"
        self.calls += 1
        d = (round_trip_uint8(image) - self.x) * 255
        s = {c: float(1.0 + 3.0 * (c != self.label) + torch.tanh((self.w[c] * d).sum() / 20)) for c in pt.COLORS}
        return {"summed_nll": s, "margin": clean_margin(s, self.label), "chosen": choose_color(s), "nll": s[self.label],
                "nll32": s[self.label], "margin32": clean_margin(s, self.label)}


def _keepref(score):
    return {"neutral": {k: score[k] for k in ("nll", "margin", "chosen")}}


def _ref(backend, x, stored, k_all=None):
    from attack_feasibility.steps import scene_seed
    r_f = mk.rounded_delta(x, project_signed_step(x, stored["grads"]["L_neutral"], EPS, "positive") - x)
    pgd = mk.replay_pgd(x, stored["codes"], 3, EPS)
    s_p = mk.support(mk.rounded_delta(x, pgd - x))
    r0 = mk.random_mask(mk.support(r_f), int(s_p.sum()), scene_seed("checkpoint-2.3b-mask|neutral|seed0", "dev4-test-a"))
    ref = {"clean": _keepref(backend.score_official(x, "neutral")),
           "pgd3": {"neutral": _keepref(backend.score_official(pgd, "neutral"))},
           "random": {("neutral", 0): _keepref(backend.score_official(mk.apply(x, r_f, r0), "neutral"))}}
    backend.calls = 0
    return ref


@pytest.mark.parametrize("smoke", [False, True])
def test_block_produces_the_planned_images_and_forward_count(smoke):
    x, stored = _scene()
    b = FakeBackend(x)
    rows = ps.partition_block(b, x, "dev4-test-a", stored, _ref(b, x, stored), smoke=smoke)
    assert b.calls == ps.FORWARDS_PER_SCENE + (1 if smoke else 0) == 51 + (1 if smoke else 0)
    images = [r for r in rows if r["kind"] == "image"]
    assert len(images) == 1 + 1 + 8 + 8 * 5
    st = next(r for r in rows if r["kind"] == "structure")
    assert st["sign_disagreements_on_overlap"] == 0 and st["n_forward"] == b.calls
    assert st["n_pgd_support"] == st["k_retained"] + st["n_excluded_outside_fgsm"]
    k = st["k_retained"]
    for r in images[1:]:
        assert r["n_pos"] + r["n_neg"] == r["count"]
        assert r["count"] == (k if r["piece"] == "full" else pt.quarter_sizes(k)[int(r["piece"])])
    chk = next(r for r in rows if r["kind"] == "checks")
    assert chk["all_bit_exact"] and (("clean_vs_2.3" in chk) == smoke)
    q = pt.scene_quantities(images)
    for name in ("A", "Q", "Delta"):
        assert q["margin"][name] == pytest.approx(q["fixed"][name] + q["switching"][name], abs=1e-9)


def test_block_stops_when_a_recorded_score_does_not_match():
    x, stored = _scene(1)
    b = FakeBackend(x)
    ref = _ref(b, x, stored)
    ref["pgd3"]["neutral"]["neutral"]["margin"] += 0.125
    with pytest.raises(Stop, match="bit-exact") as info:
        ps.partition_block(b, x, "dev4-test-a", stored, ref)
    assert any(r["kind"] == "checks" and not r["all_bit_exact"] for r in info.value.rows)


def test_block_stops_when_stored_fgsm_does_not_match_the_gradient():
    x, stored = _scene(2)
    stored["codes"]["fgsm"] = -stored["codes"]["fgsm"]
    b = FakeBackend(x)
    with pytest.raises(Stop, match="FGSM"):
        ps.partition_block(b, x, "dev4-test-a", stored, {})


def test_forward_budget_constants():
    assert ps.FORWARDS_PER_SCENE == 51
    run_total = 16 * ps.FORWARDS_PER_SCENE + 1  # plus the final clean re-score
    smoke_total = ps.FORWARDS_PER_SCENE + 1 + 1  # plus the 2.3b random seed 0 check and the final clean re-score
    assert (run_total, smoke_total) == (817, 53)
