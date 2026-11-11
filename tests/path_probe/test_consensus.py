"""Toy tests for the Checkpoint 3 pilot (no weights)."""

from __future__ import annotations

import numpy as np
import torch
from path_probe import consensus as cs
from path_probe import masks as mk
from path_probe.mech_stage import pilot_block
from path_probe.probe import CONTEXTS, clean_block, direction_block
from test_path_probe import EPS, Toy, _image

RECORD = {"scene_id": "toy", "question": "What color is the small circle in the top row?", "answer": "red",
          "objects": [{"object_id": "o0", "category": "circle", "attributes": {"color": "red", "size": "small"},
                       "position": {"row": 0, "col": 1}},
                      {"object_id": "o1", "category": "circle", "attributes": {"color": "red", "size": "large"},
                       "position": {"row": 0, "col": 2}}]}


def test_points():
    x = _image((3, 8, 8))
    d = cs.point_dense(x, EPS / 4, 1) - x
    inner = (x > EPS) & (x < 1 - EPS)
    assert torch.allclose(d[inner].abs(), torch.full_like(d[inner], EPS / 4))
    q = cs.point_quantized(x, 2, density=0.5) - x
    assert set(torch.round(q[inner] / EPS).unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert torch.equal(cs.point_dense(x, EPS / 4, 1), cs.point_dense(x, EPS / 4, 1))


def test_vote_and_smooth_masks():
    torch.manual_seed(0)
    r_f = torch.sign(torch.randn(3, 8, 8)) * EPS
    s_f = torch.ones_like(r_f, dtype=torch.bool)
    s_f[0, 0, :4] = False
    g = [torch.sign(r_f) * torch.rand(3, 8, 8) for _ in range(3)]
    g[1][1] = -g[1][1]  # channel 1 disagrees once
    m = cs.vote_mask(r_f, s_f, g, 124, seed=3)  # 60 + 64 coordinates have full agreement
    assert int(m.sum()) == 124 and not bool((m & ~s_f).any())
    assert bool(m[0][s_f[0]].all()) and bool(m[2].all()) and not bool(m[1].any())
    sm = cs.smooth_mask(r_f, s_f, g, 10)
    assert int(sm.sum()) == 10 and not bool((sm & ~s_f).any())


def test_patch_matched_random_keeps_counts_per_patch():
    torch.manual_seed(1)
    s_f = torch.rand(3, 32, 48) > 0.1
    mask = (torch.rand(3, 32, 48) > 0.7) & s_f
    out = cs.patch_matched_random(mask, s_f, seed=4, side=16)
    pid = cs.patch_ids((3, 32, 48), 16)
    assert np.array_equal(np.bincount(pid[mask.flatten().numpy()], minlength=6),
                          np.bincount(pid[out.flatten().numpy()], minlength=6))
    assert not bool((out & ~s_f).any()) and not torch.equal(out, mask)
    outside = mask.clone()
    outside[0, 0, 0], s_f[0, 0, 0] = True, False  # one mask coordinate outside S_F
    padded = cs.patch_matched_random(outside, s_f, seed=4, side=16, k=int(outside.sum()))
    assert int(padded.sum()) == int(outside.sum()) and not bool((padded & ~s_f).any())


def test_spearman_and_referent():
    assert abs(cs.spearman(np.array([1, 2, 3, 3]), np.array([2, 4, 6, 6])) - 1) < 1e-12
    assert cs.referent(RECORD)["object_id"] == "o0"


def test_pilot_block_on_toy():
    shape = (3, 64, 128)
    x = _image(shape)
    _, grads = clean_block(Toy(shape, seed=3, curvature=40.0), x)
    _, codes = direction_block(Toy(shape, seed=3, curvature=40.0), x, "neutral", grads, eps=EPS, steps=3,
                               official_steps=(3,), random_seed=1, clean_source_nll=0.0, line=False)
    stored = {"grads": grads, "codes": codes, "v32": codes}
    rows, tensors = pilot_block(Toy(shape, seed=3, curvature=40.0), x, "toy", RECORD, stored)
    st = next(r for r in rows if r["kind"] == "structure")
    masks = [r for r in rows if r["kind"] == "mask"]
    assert len(masks) == 12 and all(r["k"] == st["k"] for r in masks)
    assert len([r for r in rows if r["kind"] == "attack"]) == 2
    assert all(set(r["scores"]) == set(CONTEXTS) for r in masks)
    assert len([r for r in rows if r["kind"] == "gradient"]) == 14 and len(tensors) == 14
    pgd_ref = next(r for r in rows if r["kind"] == "reference_mask" and r["name"] == "pgd3")
    assert pgd_ref["overlap"] == 1.0
    # replayed PGD equals the stored iterate's support
    s_p = mk.support(mk.rounded_delta(x, mk.replay_pgd(x, codes, 3, EPS) - x))
    assert int(s_p.sum()) == st["k"]
