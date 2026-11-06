"""Numerical checks for the sum log-probability and its image gradient."""

from __future__ import annotations

import pytest
import torch

from vlm_iface import answer_logprob, generate, image_loss_gradient
from vlm_iface.devices import resolve_device
from vlm_iface.load import load_pinned_model
from vlm_iface.logprob import mean_token_nll, sequence_logprob
from vlm_iface.pins import QWEN_CANDIDATE_ID
from vlm_iface.session import VLMSession
from vlm_iface.tokens import answer_token_ids

PROMPT = "What color is the central square?"


def test_sequence_logprob_is_the_sum_not_the_mean():
    logits = torch.zeros(1, 3, 5)
    logits[0, 0, 1] = 4.0
    logits[0, 1, 2] = 1.0
    input_ids = torch.tensor([[0, 1, 2]])
    total = sequence_logprob(logits, input_ids, prompt_length=1)
    pieces = torch.stack([
        torch.log_softmax(logits[0, 0], dim=-1)[1],
        torch.log_softmax(logits[0, 1], dim=-1)[2],
    ])
    assert torch.allclose(total, pieces.sum())
    assert not torch.allclose(total, pieces.mean())
    assert torch.allclose(mean_token_nll(total, 2), -pieces.mean())


def test_loss_gradient_matches_a_central_difference():
    favored = 1

    def loss_at(values: torch.Tensor) -> torch.Tensor:
        # One prompt token and one answer token. Logit of the answer class is the sum of values.
        score = values.sum()
        logits = torch.zeros(1, 2, 4)
        logits[0, :, favored] = score
        input_ids = torch.tensor([[0, favored]])
        return -sequence_logprob(logits, input_ids, prompt_length=1)

    leaf = torch.tensor([0.2, -0.4, 0.7], requires_grad=True)
    loss_at(leaf).backward()
    epsilon = 1e-4
    for index in range(3):
        step = torch.zeros(3)
        step[index] = epsilon
        numerical = (loss_at(leaf.detach() + step) - loss_at(leaf.detach() - step)) / (2 * epsilon)
        assert abs(float(numerical) - float(leaf.grad[index])) < 1e-3


def test_probe_gradient_is_finite_nonzero_and_image_dependent(probe_session):
    image = torch.rand(3, 36, 36) * 0.4 + 0.3
    other = torch.rand(3, 36, 36) * 0.4 + 0.3
    gradient = image_loss_gradient(image, PROMPT, "red", session=probe_session)
    other_gradient = image_loss_gradient(other, PROMPT, "red", session=probe_session)
    assert gradient.shape == image.shape
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0
    assert not torch.allclose(gradient, other_gradient)
    red = answer_logprob(image, PROMPT, "red", session=probe_session)
    blue = answer_logprob(image, PROMPT, "blue", session=probe_session)
    assert red > blue


def test_moving_along_the_gradient_decreases_the_answer_probability(probe_session):
    image = torch.rand(3, 36, 36) * 0.4 + 0.3
    gradient = image_loss_gradient(image, PROMPT, "red", session=probe_session)
    direction = gradient / gradient.norm()
    step = 0.02
    higher_loss = (image + step * direction).clamp(0, 1)
    lower_loss = (image - step * direction).clamp(0, 1)
    logprob_higher_loss = answer_logprob(higher_loss, PROMPT, "red", session=probe_session)
    logprob_lower_loss = answer_logprob(lower_loss, PROMPT, "red", session=probe_session)
    assert logprob_lower_loss > logprob_higher_loss


def test_multi_token_answer_is_scored(probe_session, processor):
    ids = answer_token_ids(processor.tokenizer, "red square", leading_space=True)
    assert ids.numel() >= 2
    value = answer_logprob(torch.rand(3, 24, 24), PROMPT, "red square", session=probe_session)
    assert value < 0


def test_prompt_text_reaches_the_token_ids(probe_session):
    captured: dict[str, torch.Tensor] = {}
    original = probe_session.model.forward

    def wrapped(**kwargs):
        captured["ids"] = kwargs["input_ids"].detach().cpu()
        return original(**kwargs)

    probe_session.model.forward = wrapped
    answer_logprob(torch.rand(3, 24, 24), "unique-prompt-marker", "red", session=probe_session)
    decoded = probe_session.processor.tokenizer.decode(captured["ids"][0])
    assert "unique-prompt-marker" in decoded


def test_generate_contract_marks_truncated_output_invalid(processor):
    eos = int(processor.tokenizer.eos_token_id)
    word = answer_token_ids(processor.tokenizer, "red", leading_space=True).tolist()

    class Scripted(torch.nn.Module):
        def __init__(self, continuation: list[int]) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.continuation = continuation

        def forward(self, **kwargs):
            raise AssertionError(kwargs)

        def generate(self, input_ids, do_sample, max_new_tokens, **kwargs):
            assert do_sample is False
            assert max_new_tokens == 8
            extra = torch.tensor([self.continuation], dtype=input_ids.dtype)
            return torch.cat([input_ids, extra], dim=-1)

    finished = VLMSession(
        model=Scripted(word + [eos]), processor=processor, device="cpu", max_new_tokens=8
    )
    result = generate(torch.full((3, 16, 16), 0.2), PROMPT, task_family="attribute", session=finished)
    assert set(result) == {"text", "parsed_answer", "invalid"}
    assert result["invalid"] is False
    assert result["parsed_answer"] == "red"

    truncated = VLMSession(model=Scripted(word), processor=processor, device="cpu", max_new_tokens=8)
    rejected = generate(torch.full((3, 16, 16), 0.2), PROMPT, task_family="attribute", session=truncated)
    assert rejected["invalid"] is True
    assert rejected["parsed_answer"] is None


def test_loader_refuses_the_qwen_candidate_without_downloading_it():
    with pytest.raises(RuntimeError, match="not loaded"):
        load_pinned_model("cpu", model_id=QWEN_CANDIDATE_ID, revision="main")


def test_device_resolution_does_not_invent_mps_or_cuda():
    assert resolve_device("cpu").type == "cpu"
    if torch.cuda.is_available():
        assert resolve_device("cuda").type == "cuda"
    else:
        with pytest.raises(RuntimeError, match="CUDA"):
            resolve_device("cuda")
    if torch.backends.mps.is_available():
        assert resolve_device("mps").type == "mps"
        if not torch.cuda.is_available():
            assert resolve_device("auto").type == "mps"
    else:
        with pytest.raises(RuntimeError, match="MPS"):
            resolve_device("mps")
