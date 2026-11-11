"""Tests for the Checkpoint 2.3b masked-FGSM control. No weights are loaded."""

from __future__ import annotations

import torch
from attack_feasibility.steps import pgd_update, project_signed_step
from path_probe import masks as mk

EPS = 1 / 255


def _image(seed=0, shape=(3, 16, 16)):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, shape, generator=g).float() / 255  # includes 0 and 255 pixels


def test_three_random_sign_steps_round_to_a_sign_preserving_subset_of_fgsm():
    # The structural property holds for ANY step-2/3 signs, here pure noise.
    for seed in range(20):
        x = _image(seed)
        g = torch.Generator().manual_seed(1000 + seed)
        g0 = torch.randn(x.shape, generator=g)
        fgsm = project_signed_step(x, g0, EPS, "positive") - x
        cur = pgd_update(x, x, g0, EPS / 4, EPS)
        for _ in range(2):
            cur = pgd_update(cur, x, torch.randn(x.shape, generator=g), EPS / 4, EPS)
        r_f, r_p = mk.rounded_delta(x, fgsm), mk.rounded_delta(x, cur - x)
        st = mk.structure(r_f, r_p)
        assert st["sign_reversals"] == 0 and st["pgd_levels_max"] <= 1, st
        assert 0.15 < st["density_in_fgsm_support"] < 0.35  # about 1/4 for independent signs
        # The only exceptions: coordinates FGSM left unchanged (clipped at 0/1, or zero gradient)
        # where steps 2-3 moved the pixel by exactly eps/2, a half-level tie that half-to-even rounding can move.
        outside = mk.support(r_p) & ~mk.support(r_f)
        assert bool((((cur - x).abs()[outside] - EPS / 2).abs() < 1e-3 * EPS).all())


def test_replay_reproduces_the_accumulated_iterate_bit_exactly():
    from path_probe import geometry as geo
    for seed in range(10):
        x = _image(seed)
        g = torch.Generator().manual_seed(2000 + seed)
        grads = [torch.randn(x.shape, generator=g) for _ in range(3)]
        grads[0][0, 0, :4] = 0.0  # zero-gradient coordinates, as in the real data
        cur, codes = x.clone(), {}
        for k, gk in enumerate(grads, 1):
            cur = pgd_update(cur, x, gk, EPS / 4, EPS)
            codes[f"pgd{k}"] = geo.encode_delta(cur - x, EPS)
        assert torch.equal(mk.replay_pgd(x, codes, 3, EPS), cur)


def test_random_mask_is_exact_count_subset_and_deterministic():
    x = _image(3)
    r_f = mk.rounded_delta(x, project_signed_step(x, torch.randn(x.shape), EPS, "positive") - x)
    s_f = mk.support(r_f)
    a = mk.random_mask(s_f, 100, seed=7)
    assert int(a.sum()) == 100 and not bool((a & ~s_f).any())
    assert torch.equal(a, mk.random_mask(s_f, 100, seed=7))
    assert not torch.equal(a, mk.random_mask(s_f, 100, seed=8))
    img = mk.apply(x, r_f, a)
    assert torch.equal(mk.rounded_delta(x, img - x), img - x)  # already on the 8-bit grid
    assert int(mk.support(img - x).sum()) == 100


def test_topk_mask_picks_largest_gradient_inside_support():
    s_f = torch.tensor([True, True, False, True, True])
    g = torch.tensor([0.1, -5.0, 100.0, 2.0, 2.0])
    m = mk.topk_mask(s_f, g, 2)
    assert m.tolist() == [False, True, False, True, False]  # stable tie-break by index
