"""Tokenization of short answers, without loading model weights."""

from __future__ import annotations

import pytest

from vlm_iface.tokens import answer_token_ids


def test_leading_space_changes_the_ids(processor):
    bare = answer_token_ids(processor.tokenizer, "red", leading_space=False)
    spaced = answer_token_ids(processor.tokenizer, "red", leading_space=True)
    assert bare.tolist() != spaced.tolist()
    pieces = processor.tokenizer.convert_ids_to_tokens(spaced.tolist())
    assert pieces[0].startswith("Ġ") or pieces[0].startswith("▁")


def test_multi_token_answer_is_not_collapsed(processor):
    ids = answer_token_ids(processor.tokenizer, "red square", leading_space=True)
    assert ids.ndim == 1
    assert ids.numel() >= 2


@pytest.mark.parametrize("word", ["red", "blue", "three", "left of"])
def test_tokenization_is_deterministic(processor, word):
    first = answer_token_ids(processor.tokenizer, word)
    second = answer_token_ids(processor.tokenizer, word)
    assert torch_equal(first, second)


def test_empty_answer_is_rejected(processor):
    with pytest.raises(ValueError):
        answer_token_ids(processor.tokenizer, "   ")


def test_special_tokens_are_not_added(processor):
    ids = answer_token_ids(processor.tokenizer, "red", leading_space=True)
    eos = processor.tokenizer.eos_token_id
    assert eos not in ids.tolist()


def torch_equal(left, right) -> bool:
    return left.tolist() == right.tolist()
