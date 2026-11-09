"""Toy-model tests for the Checkpoint 2.4 confirmation unit (no weights)."""

from __future__ import annotations

import torch
from path_probe import geometry as geo
from path_probe.confirm_stage import scene_block
from path_probe.probe import CONTEXTS, clean_block, direction_block
from test_path_probe import EPS, Toy, _image


def test_scene_block_matches_direction_block_and_counts():
    shape = (3, 12, 12)
    x = _image(shape)
    toy = Toy(shape, seed=3, curvature=40.0)
    rows, tensors = scene_block(toy, x, "toy-scene", "prefix")
    kinds = [r["kind"] for r in rows]
    assert kinds.count("clean") == 1
    for k in ("structure", "fgsm", "pgd3", "topk_grad"):
        assert kinds.count(k) == 2
    assert kinds.count("random") == 16

    _, grads = clean_block(Toy(shape, seed=3, curvature=40.0), x)
    for source in CONTEXTS:
        _, saved = direction_block(Toy(shape, seed=3, curvature=40.0), x, source, grads, eps=EPS, steps=3,
                                   official_steps=(3,), random_seed=1, clean_source_nll=0.0, line=False)
        for n in ("fgsm", "pgd1", "pgd2", "pgd3"):
            assert torch.equal(tensors[f"{source}__{n}"], saved[n]), (source, n)
        st = next(r for r in rows if r["kind"] == "structure" and r["source"] == source)
        pgd = next(r for r in rows if r["kind"] == "pgd3" and r["source"] == source)
        assert st["sign_reversals"] == 0 and st["pgd_levels_max"] <= 1
        assert pgd["k"] == st["n_pgd_support"]
        for r in rows:
            if r.get("source") == source and r["kind"] in ("random", "topk_grad"):
                assert r["k"] == st["n_pgd_support"]
                assert set(r["scores"]) == set(CONTEXTS)


def test_random_masks_depend_on_seed_prefix_only_through_the_seed():
    shape = (3, 10, 10)
    x = _image(shape)
    a, _ = scene_block(Toy(shape, seed=5, curvature=40.0), x, "s", "p1")
    b, _ = scene_block(Toy(shape, seed=5, curvature=40.0), x, "s", "p1")
    c, _ = scene_block(Toy(shape, seed=5, curvature=40.0), x, "s", "p2")
    seeds = lambda rows: [r["seed"] for r in rows if r["kind"] == "random"]  # noqa: E731
    assert seeds(a) == seeds(b) and seeds(a) != seeds(c)
    assert geo is not None
