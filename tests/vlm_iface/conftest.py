"""Shared fixtures. The processor download is cached; the 500M weights are not loaded."""

from __future__ import annotations

import pytest
import torch

from vlm_iface.pins import MODEL_ID, PINNED_REVISION
from vlm_iface.session import VLMSession
from vlm_iface.tokens import answer_token_ids


class PixelEnergyProbe(torch.nn.Module):
    """Tiny stand-in whose logits depend on the energy of the processed pixels."""

    def __init__(self, vocab_size: int, favored_id: int) -> None:
        super().__init__()
        direction = torch.zeros(vocab_size)
        direction[favored_id] = 1.0
        self.register_buffer("direction", direction)
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        pixel_attention_mask=None,
        use_cache=None,
        **kwargs,
    ):
        del attention_mask, pixel_attention_mask, use_cache, kwargs
        signal = pixel_values.float().square().mean()
        batch, steps = input_ids.shape
        logits = signal * self.direction.view(1, 1, -1)
        logits = logits.expand(batch, steps, -1) + self.anchor
        return _Logits(logits)


class _Logits:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


@pytest.fixture(scope="session")
def processor():
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(MODEL_ID, revision=PINNED_REVISION)


@pytest.fixture
def probe_session(processor):
    favored = answer_token_ids(processor.tokenizer, "red", leading_space=True)
    model = PixelEnergyProbe(len(processor.tokenizer), int(favored[0]))
    return VLMSession(model=model, processor=processor, device="cpu", max_new_tokens=8)
