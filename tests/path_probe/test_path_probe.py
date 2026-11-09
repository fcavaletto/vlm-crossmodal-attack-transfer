"""Toy-model tests for the Checkpoint 2.3 probe. No weights are loaded."""

from __future__ import annotations

import math

import pytest
import torch
from path_probe import geometry as geo
from path_probe.precision import Fp32Projection, VisionPrecision, bf16_input_survival
from path_probe.probe import CONTEXTS, clean_block, cross_evaluate, direction_block
from path_probe.run import select_scenes

EPS = 1 / 255
COLORS = ("red", "blue", "green", "yellow")


class Toy:
    """Logits z_c(x) = w_c . x + 0.5 * a_c * (w_c . x)^2 per context. Label red."""

    label = "red"

    def __init__(self, shape=(3, 6, 6), seed=0, curvature=0.0):
        g = torch.Generator().manual_seed(seed)
        self.w = {ctx: torch.randn((4, *shape), generator=g, dtype=torch.float64) for ctx in CONTEXTS}
        self.bias = torch.tensor([3.0, 1.0, 0.5, 0.0], dtype=torch.float64)
        self.curv = curvature
        self.calls = {"grad_nll": 0, "grad_margin": 0}

    def _z(self, x, ctx):
        lin = (self.w[ctx] * x.double().unsqueeze(0)).flatten(1).sum(1)
        return self.bias + lin + 0.5 * self.curv * lin**2

    def _score(self, x, ctx):
        z = self._z(x, ctx)
        nll = {c: float(torch.logsumexp(z, 0) - z[i]) for i, c in enumerate(COLORS)}
        margin = min(nll[c] for c in COLORS if c != "red") - nll["red"]
        chosen = min(nll, key=nll.get)
        return {"nll": nll["red"], "margin": margin, "chosen": chosen, "summed_nll": nll,
                "nll32": nll["red"], "margin32": margin}

    score_float = _score

    def score_official(self, x, ctx):
        return self._score((x.clamp(0, 1) * 255).round() / 255, ctx)

    def grad_nll(self, x, ctx):
        self.calls["grad_nll"] += 1
        leaf = x.detach().double().requires_grad_(True)
        z = self._z(leaf, ctx)
        (torch.logsumexp(z, 0) - z[0]).backward()
        return leaf.grad.float()

    def grad_margin(self, x, ctx, comp):
        self.calls["grad_margin"] += 1
        leaf = x.detach().double().requires_grad_(True)
        z = self._z(leaf, ctx)
        (z[0] - z[COLORS.index(comp)]).backward()
        return leaf.grad.float()


def _image(shape=(3, 6, 6)):
    return (torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape) % 200 + 30) / 255


def test_geometry_basics():
    a = torch.tensor([1.0, -2.0, 0.0, 3.0])
    b = torch.tensor([1.0, 2.0, 5.0, 3.0])
    assert geo.dot(a, b) == pytest.approx(1 - 4 + 9)
    assert geo.sign_agreement(a, b) == pytest.approx(2 / 3)
    assert geo.sign_agreement(a, b, weight=a) == pytest.approx(4 / 6)
    assert geo.cosine(a, a) == pytest.approx(1.0)
    assert geo.cosine(a, torch.zeros(4)) is None
    # A(source, target): target . sign(source) / ||target||_1
    assert geo.alignment(b, a) == pytest.approx((1 - 2 + 0 + 3) / 11)


def test_delta_encoding_roundtrip_and_geometry():
    d = torch.tensor([EPS, -EPS / 4, 0.0, EPS * 0.75])
    code = geo.encode_delta(d, EPS)
    assert code.tolist() == [4, -1, 0, 3]
    assert torch.allclose(geo.decode_delta(code, EPS), d)
    with pytest.raises(ValueError):
        geo.encode_delta(torch.tensor([EPS / 3]), EPS)
    g = geo.delta_geometry(d, EPS, reference=torch.tensor([EPS, EPS, EPS, -EPS]))
    assert g["frac_at_eps"] == pytest.approx(0.25)
    assert g["frac_zero"] == pytest.approx(0.25)
    assert g["sign_disagreement_ref"] == pytest.approx(2 / 3)


def test_interpolate():
    assert geo.interpolate_at([0, 1, 3], [0, 10, 30], 2) == pytest.approx(20)
    assert geo.interpolate_at([0, 1], [0, 10], 5) is None


def test_clean_block_and_pairwise():
    toy = Toy()
    row, grads = clean_block(toy, _image())
    assert set(grads) == {"L_neutral", "M_neutral", "L_helpful_nonrevealing", "M_helpful_nonrevealing"}
    assert row["repeat_float"]["neutral"]["nll_abs_diff"] == 0
    assert -1 <= row["pairwise"]["A_nh"] <= 1
    # In the toy, the NLL gradient is minus a positive mix of logit gradients: cos(g_L, g_M) < 0.
    assert row["pairwise"]["cos_Ln_Mn"] < 0
    reduced, g2 = clean_block(toy, _image(), ("L_neutral", "M_neutral"))
    assert set(g2) == {"L_neutral", "M_neutral"} and "A_nh" not in reduced["pairwise"]


def test_direction_block_linear_model_is_first_order_exact_in_margin():
    toy = Toy(curvature=0.0)
    x = _image()
    clean, grads = clean_block(toy, x)
    rows, saved = direction_block(toy, x, "neutral", grads, eps=EPS, steps=3, official_steps=(3,),
                                  random_seed=7, clean_source_nll=clean["scores"]["neutral"]["float"]["nll"])
    kinds = [r["kind"] for r in rows]
    assert kinds.count("pgd") == 3 and kinds.count("pgd_gradient") == 2 and kinds.count("line") == 4
    assert set(saved) == {"fgsm", "pgd1", "pgd2", "pgd3"}
    m0 = clean["scores"]["neutral"]["float"]["margin"]
    for r in rows:
        if r["kind"] in ("fgsm", "line", "random", "pgd"):
            # Linear logits: the margin is linear in x, so the clean margin gradient predicts it exactly
            # whenever the competitor does not change.
            realized = r["scores"]["neutral"]["float"]["margin"] - m0
            pred = r["pred_clean_gradient"]["neutral"]["margin"]
            if r["scores"]["neutral"]["float"]["chosen"] == "red":
                assert realized <= pred + 1e-4 * (1 + abs(pred))  # margin is a min over competitors: concave
    pgd3 = [r for r in rows if r["kind"] == "pgd" and r["k"] == 3][0]
    assert "official" in pgd3["scores"]["neutral"] and "geometry_rounded" in pgd3
    assert pgd3["geometry"]["linf"] <= 0.75 * EPS + 1e-6
    # Linear model: the gradient direction never changes along the path.
    for r in rows:
        if r["kind"] == "pgd_gradient":
            assert r["sign_agree_g0"] > 0.999
    # FGSM maximizes the clean first-order NLL prediction over the ball.
    fgsm = [r for r in rows if r["kind"] == "fgsm"][0]
    assert fgsm["pred_clean_gradient"]["neutral"]["nll"] >= pgd3["pred_clean_gradient"]["neutral"]["nll"] - 1e-9
    assert all(r["source"] == "neutral" for r in rows)


def test_direction_block_curvature_rotates_gradient_and_deadline():
    toy = Toy(curvature=2000.0, seed=3)
    x = _image()
    clean, grads = clean_block(toy, x)
    rows, _ = direction_block(toy, x, "helpful_nonrevealing", grads, eps=EPS, steps=7, official_steps=(3, 7),
                              random_seed=1, clean_source_nll=clean["scores"]["helpful_nonrevealing"]["float"]["nll"])
    assert [r["k"] for r in rows if r["kind"] == "pgd"] == list(range(1, 8))
    steps_sum = sum(r["realized_nll_step"] for r in rows if r["kind"] == "pgd")
    last = [r for r in rows if r["kind"] == "pgd"][-1]["scores"]["helpful_nonrevealing"]["float"]["nll"]
    assert steps_sum == pytest.approx(last - clean["scores"]["helpful_nonrevealing"]["float"]["nll"], abs=1e-9)
    calls = {"n": 0}

    def deadline():
        calls["n"] += 1
        return calls["n"] >= 2

    rows2, _ = direction_block(toy, x, "neutral", grads, eps=EPS, steps=7, official_steps=(3,), random_seed=1,
                               clean_source_nll=0.0, deadline=deadline, line=False)
    assert any(r["kind"] == "pgd_truncated" for r in rows2)


def test_cross_evaluate():
    toy = Toy()
    x = _image()
    _, grads = clean_block(toy, x)
    _, saved = direction_block(toy, x, "neutral", grads, eps=EPS, steps=3, official_steps=(3,), random_seed=1,
                               clean_source_nll=0.0, line=False)
    deltas = {k: geo.decode_delta(v, EPS) for k, v in saved.items() if k in ("fgsm", "pgd3")}
    rows = cross_evaluate(toy, x, deltas, grads, EPS, crafted_in="B")
    assert [r["perturbation"] for r in rows] == ["fgsm", "pgd3"]
    assert all(r["crafted_in"] == "B" and "official" in r["scores"]["neutral"] for r in rows)


def test_fp32_projection_matches_float64_reference():
    torch.manual_seed(0)
    head = torch.nn.Linear(16, 40000, bias=False).to(torch.bfloat16)
    ids = {"red": (5,), "blue": (17,), "green": (30000,), "yellow": (39999,)}
    proj = Fp32Projection(head, ids)
    h = torch.randn(1, 3, 16).to(torch.bfloat16)
    head(h)
    out = proj.score("red")
    ref = head.weight.double() @ h[0, -1].double()
    lz = torch.logsumexp(ref, 0)
    assert out["nll32"] == pytest.approx(float(lz - ref[5]), abs=1e-3)
    assert out["margin32"] == pytest.approx(float(ref[5] - max(ref[17], ref[30000], ref[39999])), abs=1e-3)
    with pytest.raises(RuntimeError):
        proj.score("red")
    proj.remove()


def test_vision_precision_restore_is_exact():
    m = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.LayerNorm(8)).to(torch.bfloat16)
    m.register_buffer("inv", torch.arange(4, dtype=torch.float32))
    before = {k: v.clone() for k, v in m.state_dict().items()}
    vp = VisionPrecision(m)
    assert vp.fp32_bytes() == sum(p.numel() for p in m.parameters()) * 4
    vp.to_fp32()
    assert next(m.parameters()).dtype == torch.float32 and m.inv.dtype == torch.float32
    vp.restore()
    after = m.state_dict()
    for k in before:
        assert after[k].dtype == before[k].dtype and torch.equal(after[k], before[k])


def test_bf16_input_survival_identity_path():
    x = torch.full((3, 4, 4), 0.5)
    rows = bf16_input_survival(lambda im: (im - 0.48) / 0.27, x, EPS, (0.125, 1.0), seed=1)
    small, full = rows
    assert small["n_moved_fp32"] == full["n_moved_fp32"] == 48
    assert small["frac_changed_bf16_among_moved"] <= full["frac_changed_bf16_among_moved"]


def test_select_scenes_rule():
    recs = [{"scene_id": f"dev4-ref-{i:02d}-{arm}", "counterfactual_group_id": f"cf4-ref-{i:02d}", "split": "development"}
            for i in range(48) for arm in "ab"]
    s16, order = select_scenes(recs)
    assert len(s16) == 16 and len(order) == 48 and order[:16] == s16
    assert all(s.endswith("-a") for s in order)
    assert "dev4-ref-00-a" in s16 and "dev4-ref-47-a" in s16
