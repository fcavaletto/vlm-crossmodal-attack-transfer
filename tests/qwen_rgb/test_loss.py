"""Chosen-answer summed NLL and the text-mask extension. No model."""

from __future__ import annotations

import torch

from qwen_rgb.loss import chosen_answer_nll, extend_with_answer


def test_summed_nll_is_the_joint_not_the_mean():
    logits = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 3.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        ]
    )
    input_ids = torch.tensor([[7, 1, 3, 0]])
    loss = chosen_answer_nll(logits, input_ids, prompt_length=2)
    predicted = torch.log_softmax(logits[0, 1:3].float(), dim=-1)
    expected = -(predicted[0, 3] + predicted[1, 0])
    assert torch.allclose(loss, expected)
    mean = expected / 2
    assert not torch.allclose(loss, mean)


def test_extend_with_answer_lengthens_text_and_leaves_vision():
    inputs = {
        "input_ids": torch.tensor([[4, 5, 6]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "mm_token_type_ids": torch.tensor([[1, 1, 0]]),
        "pixel_values": torch.zeros(4, 8),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }
    extended, prompt_length = extend_with_answer(inputs, torch.tensor([9, 8]))
    assert prompt_length == 3
    assert extended["input_ids"].tolist() == [[4, 5, 6, 9, 8]]
    assert extended["attention_mask"].tolist() == [[1, 1, 1, 1, 1]]
    assert extended["mm_token_type_ids"].tolist() == [[1, 1, 0, 0, 0]]
    assert torch.equal(extended["pixel_values"], inputs["pixel_values"])
    assert torch.equal(extended["image_grid_thw"], inputs["image_grid_thw"])
    assert torch.equal(inputs["input_ids"], torch.tensor([[4, 5, 6]]))
