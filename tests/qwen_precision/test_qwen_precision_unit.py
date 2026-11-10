"""Model-free tests: the frozen 2.4 scene block driven through RecordingBackend with stubbed model calls."""

import torch

import path_probe.backend as pb
from path_probe.confirm_stage import scene_block
from qwen_precision.backend import CALLS_PER_SOURCE, N_CLEAN_CALLS, RecordingBackend
from qwen_precision.numerics import DtypeAudit, configure, fp32_matmul_probe, non_matching


def _score(image, ctx):
    m = float(image.sum()) % 7 - 3
    return {"nll": 1.0 + 0.01 * m, "margin": m, "chosen": "red" if m > 0 else "blue", "nll32": 1.0, "margin32": m,
            "summed_nll": {"red": 1.0, "blue": 1.5 - 0.01 * m, "green": 3.0, "yellow": 4.0}, "summed_nll32": {}}


def test_scene_block_call_order_matches_recording_assumptions(monkeypatch):
    gen = torch.Generator().manual_seed(0)
    monkeypatch.setattr(pb.QwenBackend, "score_float", lambda self, image, ctx: _score(image, ctx))
    monkeypatch.setattr(pb.QwenBackend, "score_official", lambda self, image, ctx: _score(image, ctx))
    monkeypatch.setattr(pb.QwenBackend, "grad_nll", lambda self, image, ctx: torch.randn(image.shape, generator=gen))
    monkeypatch.setattr(pb.QwenBackend, "grad_margin", lambda self, image, ctx, comp: torch.randn(image.shape, generator=gen))
    b = RecordingBackend.__new__(RecordingBackend)
    b.label, b.float_check, b.nll_grads, b.official_ctx, b.float_scores = "red", True, [], [], []
    x = torch.rand(3, 16, 16).mul(250).round().add(2).div(255)
    rows, tensors = scene_block(b, x, "s", "test-prefix")
    assert len(b.official_ctx) == N_CLEAN_CALLS + 2 * CALLS_PER_SOURCE
    assert [c for c, _ in b.nll_grads] == ["neutral", "helpful_nonrevealing", "neutral", "neutral",
                                           "helpful_nonrevealing", "helpful_nonrevealing"]
    assert [(e["source_block"], e["kind"], e["context"]) for e in b.float_scores] == [
        (s, k, c) for s in (0, 1) for k in ("fgsm", "pgd3") for c in ("neutral", "helpful_nonrevealing")]
    kinds = [r["kind"] for r in rows]
    assert kinds.count("random") == 16 and kinds.count("fgsm") == 2 and kinds.count("pgd3") == 2


def test_numerics_cpu():
    s = configure(torch.device("cpu"))
    assert s["autocast_enabled"] is False
    assert fp32_matmul_probe(torch.device("cpu"), n=128)["passes_1e-5"]


def test_dtype_audit_finds_downcast():
    class Down(torch.nn.Module):
        def forward(self, x):
            return x.to(torch.bfloat16)

    m = torch.nn.Sequential(torch.nn.Linear(4, 4), Down())
    a = DtypeAudit(m)
    m(torch.randn(2, 4))
    seen = a.remove()
    assert seen["Linear"] == ["float32"] and seen["Down"] == ["bfloat16"]
    assert non_matching(seen, "fp32") == {"Down": ["bfloat16"]}
