"""The Stage 1 unit block with a fake model: shapes, counts, subset identity, both paths."""

import torch

from internvl_probe.stage1 import N_SEEDS, unit_block


class Fake:
    def __init__(self):
        self.gen = torch.Generator().manual_seed(0)
        self.calls = {"gradient": 0, "official": 0, "surrogate": 0}

    def gradient(self, x, prompt, label, kind):
        self.calls["gradient"] += 1
        return 0.0, torch.randn(x.shape, generator=self.gen)

    def score(self, image, prompt, label, path):
        self.calls[path] += 1
        return {"margin": float(image.sum()), "chosen": "red", "nll": {}, "label_nll": 0.0, "top_token": 0}


class Monitor:
    def check(self):
        pass


def test_unit_block_counts_and_identity():
    x = torch.rand(3, 24, 32).mul(253).round().add(1).div(255)  # 8-bit, away from 0 and 1
    m = Fake()
    rows, tensors = unit_block(m, x, {"neutral": {}, "helpful_nonrevealing": {}}, "red", "s", "neutral", Monitor())
    kinds = [r["kind"] for r in rows]
    assert kinds == ["structure", "fgsm", "pgd3", "topk_grad"] + ["random"] * N_SEEDS
    assert m.calls["gradient"] == 3
    assert m.calls["official"] == m.calls["surrogate"] == 11 * 2
    st = rows[0]
    assert st["sign_reversals"] == 0 and st["pgd_levels_max"] <= 1 and st["pgd_outside_fgsm"] == 0
    k = rows[2]["k"]
    assert k == st["n_pgd_support"] and all(r["k"] == k for r in rows[3:])
    assert set(rows[1]["scores"]["neutral"]) == {"official", "float"}
    assert set(st["alignment"]) == {"cos_g0_g1", "sign_g0_g1", "cos_g0_g2", "sign_g0_g2", "cos_g1_g2", "sign_g1_g2"}
    assert set(tensors) == {"neutral__fgsm", "neutral__pgd1", "neutral__pgd2", "neutral__pgd3", "neutral__g0"}
