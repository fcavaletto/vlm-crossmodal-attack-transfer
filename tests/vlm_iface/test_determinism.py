"""Same inputs, same tokens, same gradient. CPU probe only; not an MPS claim."""

from __future__ import annotations

import torch

from vlm_iface import answer_logprob, image_loss_gradient

PROMPT = "What color is the central square?"


def test_logprob_and_gradient_repeat_exactly(probe_session):
    image = torch.rand(3, 40, 48)
    first_logprob = answer_logprob(image, PROMPT, "red", session=probe_session)
    second_logprob = answer_logprob(image, PROMPT, "red", session=probe_session)
    first_gradient = image_loss_gradient(image, PROMPT, "red", session=probe_session)
    second_gradient = image_loss_gradient(image, PROMPT, "red", session=probe_session)
    assert first_logprob == second_logprob
    assert torch.equal(first_gradient, second_gradient)


def test_tokenization_repeat_matches_the_session_tokens(processor):
    from vlm_iface.tokens import answer_token_ids

    assert answer_token_ids(processor.tokenizer, "left of").tolist() == answer_token_ids(
        processor.tokenizer, "left of"
    ).tolist()
