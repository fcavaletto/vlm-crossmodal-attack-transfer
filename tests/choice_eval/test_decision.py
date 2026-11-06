"""Ranking, ties, and the sum-versus-mean decision."""

from __future__ import annotations

import torch

from choice_eval.colors import COLORS
from choice_eval.decision import choose_color, rank_colors
from choice_eval.score import score_example


def test_rank_follows_summed_nll():
    scores = {"red": 0.4, "blue": 0.1, "green": 0.2, "yellow": 0.3}
    assert rank_colors(scores) == ("blue", "green", "yellow", "red")
    assert choose_color(scores) == "blue"
    reverse = {color: scores[color] for color in reversed(COLORS)}
    assert rank_colors(reverse) == rank_colors(scores)


def test_ties_break_by_color_name_not_by_insertion_order():
    tied = 1.25
    blue_red = {"red": tied, "blue": tied, "green": tied + 1, "yellow": tied + 2}
    assert choose_color(blue_red) == "blue"
    red_yellow = {"yellow": tied, "red": tied, "green": 5.0, "blue": 5.0}
    assert choose_color(red_yellow) == "red"
    assert choose_color({color: 3.0 for color in reversed(COLORS)}) == "blue"
    almost = {"blue": 1.0 + 1e-9, "red": 1.0, "green": 2.0, "yellow": 3.0}
    assert choose_color(almost) == "red"
    other = {"blue": 1.0, "red": 1.0 + 1e-9, "green": 2.0, "yellow": 3.0}
    assert choose_color(other) == "blue"


def test_sum_and_mean_can_choose_different_colors(peaked):
    logits = {color: peaked(1, 0.0) for color in COLORS}
    token_ids = {color: [1] for color in COLORS}
    logits["red"] = peaked(1, 1.5)
    logits["yellow"] = peaked(4, 2.25)
    token_ids["yellow"] = [1, 1, 1, 1]
    row = score_example(logits, token_ids, label="yellow")
    assert row["chosen"] == "red"
    assert row["rank_by_mean"][0] == "yellow"
    assert row["reductions_agree"] is False
    assert row["attack_loss"] is row["target_loss"]
    assert row["attack_loss"] is row["summed_nll"]["yellow"]
    assert float(row["attack_loss"]) > float(row["mean_nll"]["yellow"])
    assert float(row["summed_nll"]["red"]) < float(row["summed_nll"]["yellow"])
    assert float(row["mean_nll"]["yellow"]) < float(row["mean_nll"]["red"])
    assert row["primary_reduction"] == "sum"
    assert row["closed_vocabulary_correct"] is False


def test_equal_lengths_share_the_argmin_and_differ_in_scale(peaked):
    logits = {color: peaked(2, margin) for color, margin in zip(COLORS, (0.5, 2.0, 1.0, 0.0), strict=True)}
    token_ids = {color: [1, 1] for color in COLORS}
    row = score_example(logits, token_ids, label="blue")
    assert row["reductions_agree"] is True
    assert row["chosen"] == "blue"
    for color in COLORS:
        assert torch.allclose(row["summed_nll"][color], row["mean_nll"][color] * 2)


def test_labeled_loss_is_not_replaced_by_the_winner(peaked):
    logits = {color: peaked(1, 0.0) for color in COLORS}
    logits["red"] = peaked(1, 0.0).requires_grad_(True)
    logits["blue"] = peaked(1, 3.0)
    token_ids = {color: [1] for color in COLORS}
    row = score_example(logits, token_ids, label="red")
    assert row["chosen"] == "blue"
    assert row["attack_loss"] is row["summed_nll"]["red"]
    assert float(row["attack_loss"].detach()) > float(row["summed_nll"]["blue"].detach())
    row["attack_loss"].backward()
    assert logits["red"].grad is not None
    assert float(logits["red"].grad[0, 1]) < 0
    assert logits["blue"].grad is None
