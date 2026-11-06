"""Closed-vocabulary rules. These tests do not load weights."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from class_score.decision import argmin_class, class_decision, is_disagreement
from class_score.generation import adopt_embedded_reading, score_free_generation
from class_score.loss import (
    class_nll_from_surfaces,
    finite_difference_positions,
    single_position_class_nll,
    word_only_nll,
)
from class_score.protocol import (
    CLASS_ARGMIN_IS_PRIMARY,
    DISAGREEMENT_TOLERANCE,
    EMBEDDED_READING_ADOPTED,
    FREE_GENERATION_ROLE,
    OFFICIAL_METRIC,
)
from class_score.sensitivity import losses_without_kinds
from class_score.surfaces import DIGIT_KINDS, build_surfaces
from class_score.vocabulary import COUNT_DIGIT_ALIASES, alias_to_canonical, task_vocabulary
from vlm_iface.parsing import COUNT_DIGIT_ALIASES as PARSER_DIGITS
from vlm_iface.parsing import task_vocabulary as parser_vocabulary

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "class_score"


class TableTokenizer:
    def __init__(self) -> None:
        self.table: dict[str, list[int]] = {}
        self.decode_map: dict[int, str] = {}
        self.next_id = 1

    def single(self, text: str, decoded: str, token_id: int | None = None) -> int:
        token = self.next_id if token_id is None else token_id
        self.next_id = max(self.next_id, token + 1)
        self.table[text] = [token]
        self.decode_map[token] = decoded
        return token

    def multi(self, text: str) -> None:
        self.table[text] = [self.next_id, self.next_id + 1]
        self.next_id += 2

    def __call__(self, text: str, add_special_tokens: bool = False, return_tensors: str = "pt"):
        del add_special_tokens, return_tensors
        return {"input_ids": torch.tensor([self.table[text]], dtype=torch.long)}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(self.decode_map[int(token)] for token in ids)


def _color_tokenizer() -> TableTokenizer:
    tokenizer = TableTokenizer()
    for word in ("red", "blue", "green", "yellow"):
        capitalized = word.capitalize()
        tokenizer.single(f" {word}", word)
        tokenizer.single(word, word)
        tokenizer.single(f" {capitalized}", capitalized)
        tokenizer.single(capitalized, capitalized)
        tokenizer.multi(f" {word.upper()}")
        tokenizer.single(word.upper(), word.upper())
    return tokenizer


def _counting_tokenizer() -> TableTokenizer:
    tokenizer = TableTokenizer()
    for word in ("one", "two", "three", "four"):
        tokenizer.single(f" {word}", word)
        tokenizer.single(word, word)
        tokenizer.single(f" {word.capitalize()}", word.capitalize())
        tokenizer.single(word.capitalize(), word.capitalize())
        tokenizer.multi(f" {word.upper()}")
        tokenizer.multi(word.upper())
    tokenizer.decode_map[216] = " "
    for digit, token in (("1", 33), ("2", 34), ("3", 35), ("4", 36)):
        tokenizer.table[digit] = [token]
        tokenizer.decode_map[token] = digit
        tokenizer.table[f" {digit}"] = [216, token]
    return tokenizer


def test_vocabulary_is_the_checkpoint_1_parser_vocabulary():
    for family in ("attribute", "counting", "spatial"):
        assert task_vocabulary(family) == parser_vocabulary(family)
    assert alias_to_canonical("attribute") == {color: color for color in task_vocabulary("attribute")}
    assert "1" not in alias_to_canonical("spatial")
    assert alias_to_canonical("counting")["4"] == "four"
    assert COUNT_DIGIT_ALIASES == dict(PARSER_DIGITS)
    assert task_vocabulary("attribute") == ("red", "blue", "green", "yellow")
    assert task_vocabulary("counting") == ("one", "two", "three", "four")


def test_split_uppercase_spellings_are_skipped_and_the_digit_is_kept():
    colors = build_surfaces("attribute", _color_tokenizer(), leading_space=True)
    assert any(item["reason"] == "multi_token" and item["source_text"] == "BLUE" for item in colors.skipped)
    assert all(surface.kind != "digit_alias" for group in colors.by_class.values() for surface in group)
    assert colors.canonical_token_id("red") == colors.by_class["red"][0].token_id
    counting = build_surfaces("counting", _counting_tokenizer(), leading_space=True)
    two = counting.by_class["two"]
    assert any(surface.kind == "digit_alias" and surface.token_ids == (216, 34) for surface in two)
    assert any(surface.kind == "digit_alias_bare" and surface.token_ids == (34,) for surface in two)
    assert any(item["reason"] == "multi_token" and item["source_text"] == "TWO" for item in counting.skipped)
    assert any(item["reason"] == "multi_token" and item["source_text"] == "FOUR" for item in counting.skipped)


def test_duplicate_bare_token_is_not_counted_twice():
    tokenizer = _color_tokenizer()
    tokenizer.table["red"] = list(tokenizer.table[" red"])
    surfaces = build_surfaces("attribute", tokenizer, leading_space=True)
    red_ids = [surface.token_ids for surface in surfaces.by_class["red"]]
    assert len(red_ids) == len(set(red_ids))
    assert any(item["reason"] == "duplicate_token" for item in surfaces.skipped)


def test_required_canonical_token_must_be_one_token():
    tokenizer = TableTokenizer()
    tokenizer.multi(" red")
    with pytest.raises(RuntimeError, match="Required surface"):
        build_surfaces("attribute", tokenizer, leading_space=True)


def test_duplicate_token_ids_do_not_double_count_probability():
    logits = torch.tensor([0.0, 2.0, -1.0], dtype=torch.float64)
    once = single_position_class_nll(logits, [1])
    twice = single_position_class_nll(logits, [1, 1])
    assert torch.allclose(once, twice)


def test_class_loss_is_the_negative_log_sum_of_surface_probabilities():
    logits = torch.tensor([0.0, 0.2, 1.5, 0.0, -1.0], dtype=torch.float64)
    log_probs = torch.log_softmax(logits, dim=-1)
    expected = -torch.log(torch.exp(log_probs[1]) + torch.exp(log_probs[2]))
    assert torch.allclose(single_position_class_nll(logits, [1, 2]), expected)
    assert float(single_position_class_nll(logits, [1, 2])) < float(word_only_nll(logits, 1))


def test_finite_difference_matches_autograd_on_constructed_cpu_logits():
    logits = torch.tensor([0.2, -0.4, 1.1, 0.3, -0.7, 0.5], dtype=torch.float64, requires_grad=True)
    loss = single_position_class_nll(logits, [1, 4])
    loss.backward()
    numerical = finite_difference_positions(
        lambda values: single_position_class_nll(values, [1, 4]),
        (logits.detach(),),
        epsilon=1e-5,
    )[0]
    assert numerical.device.type == "cpu"
    assert torch.allclose(numerical, logits.grad.detach(), atol=1e-6, rtol=1e-5)
    assert float(numerical[1]) < 0
    assert float(numerical[4]) < 0
    assert float(numerical[2]) > 0


def test_two_token_digit_gradient_is_a_logit_finite_difference():
    inventory = build_surfaces("counting", _counting_tokenizer(), leading_space=True)
    first = torch.full((220,), -4.0, dtype=torch.float64)
    second = torch.full((220,), -4.0, dtype=torch.float64)
    four_id = inventory.canonical_token_id("four")
    two_id = inventory.canonical_token_id("two")
    first[four_id] = 1.0
    first[two_id] = 0.0
    first[216] = 6.0
    second[34] = 5.0
    scores = {
        name: float(class_nll_from_surfaces((first, second), surfaces))
        for name, surfaces in inventory.by_class.items()
    }
    words = {name: float(word_only_nll(first, inventory.canonical_token_id(name))) for name in inventory.by_class}
    assert argmin_class(scores) == "two"
    assert argmin_class(words) == "four"
    without_digits = losses_without_kinds((first, second), inventory, DIGIT_KINDS)
    assert argmin_class(without_digits) == "four"
    assert without_digits["two"] > scores["two"]

    leaf_first = first.clone().requires_grad_(True)
    leaf_second = second.clone().requires_grad_(True)
    two_surfaces = inventory.by_class["two"]
    class_nll_from_surfaces((leaf_first, leaf_second), two_surfaces).backward()
    numerical = finite_difference_positions(
        lambda left, right: class_nll_from_surfaces((left, right), two_surfaces),
        (leaf_first.detach(), leaf_second.detach()),
        epsilon=1e-5,
    )
    assert torch.allclose(numerical[0], leaf_first.grad.detach(), atol=1e-6, rtol=1e-5)
    assert torch.allclose(numerical[1], leaf_second.grad.detach(), atol=1e-6, rtol=1e-5)
    assert float(numerical[0][216]) < 0
    assert float(numerical[1][34]) < 0
    assert float(numerical[0][four_id]) > 0


def test_raising_an_alias_logit_increases_the_word_loss_and_decreases_the_class_loss():
    logits = torch.zeros(5, dtype=torch.float64, requires_grad=True)
    word_only_nll(logits, 1).backward()
    assert float(logits.grad[1]) < 0
    assert float(logits.grad[2]) > 0
    logits.grad = None
    single_position_class_nll(logits, [1, 2]).backward()
    assert float(logits.grad[1]) < 0
    assert float(logits.grad[2]) < 0


def test_non_cpu_logits_are_refused():
    logits = torch.zeros(4, device="meta")
    with pytest.raises(RuntimeError, match="CPU"):
        single_position_class_nll(logits, [0, 1])


def test_decision_tie_break_and_disagreement_tolerance():
    assert argmin_class({"yellow": 1.0, "blue": 1.0, "red": 2.0, "green": 3.0}) == "blue"
    scores = {"red": 0.2, "blue": 1.0, "green": 2.0, "yellow": 3.0}
    assert is_disagreement(None, True, scores) is False
    assert is_disagreement("red", False, scores) is False
    assert is_disagreement("blue", False, scores) is True
    near = {"red": 1.0, "blue": 1.0 + DISAGREEMENT_TOLERANCE, "green": 3.0, "yellow": 4.0}
    assert is_disagreement("blue", False, near) is False
    over = {"red": 1.0, "blue": 1.0 + DISAGREEMENT_TOLERANCE + 1e-8, "green": 3.0, "yellow": 4.0}
    assert is_disagreement("blue", False, over) is True
    decision = class_decision(scores, "red")
    assert decision["chosen"] == "red"
    assert decision["correct"] is True
    assert decision["is_primary_outcome"] is False


def test_free_generation_is_the_official_metric_and_embedded_reading_is_refused():
    assert OFFICIAL_METRIC == "strict_free_generation"
    assert FREE_GENERATION_ROLE == "official"
    assert CLASS_ARGMIN_IS_PRIMARY is False
    assert EMBEDDED_READING_ADOPTED is False
    color = score_free_generation("Blue.", truncated=False, task_family="attribute", answer="blue")
    assert color["role"] == "official"
    assert color["is_secondary"] is False
    assert color["correct"] is True
    assert color["parsed_answer"] == "blue"
    sentence = score_free_generation(
        "The large circle is red.",
        truncated=False,
        task_family="attribute",
        answer="red",
    )
    assert sentence["invalid"] is True
    assert sentence["correct"] is False
    assert sentence["embedded_label"] == "red"
    assert sentence["embedded_adopted"] is False
    assert is_disagreement(sentence["parsed_answer"], True, {"red": 0.1, "blue": 1.0, "green": 2.0, "yellow": 3.0}) is False
    digit = score_free_generation("2", truncated=False, task_family="counting", answer="two")
    assert digit["correct"] is True
    assert digit["parsed_answer"] == "two"
    wrong_family = score_free_generation("2", truncated=False, task_family="attribute", answer="red")
    assert wrong_family["invalid"] is True
    with pytest.raises(RuntimeError, match="not an outcome"):
        adopt_embedded_reading("The large circle is red.")


def test_package_source_does_not_call_an_image_gradient_or_load_weights():
    text = "\n".join(path.read_text(encoding="utf-8") for path in PACKAGE.rglob("*.py"))
    for banned in ("image_loss_gradient", "from_pretrained", "FGSM", "PGD"):
        assert banned not in text
