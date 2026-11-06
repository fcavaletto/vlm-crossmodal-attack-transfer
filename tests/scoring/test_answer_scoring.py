"""Checkpoint 1.5 scoring rules. These tests do not load SmolVLM2 weights."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from answer_scoring.audit import audit_predictions, load_predictions
from answer_scoring.class_loss import argmin_name, class_nll, mixture_nll, two_token_logprob, word_only_nll
from answer_scoring.compare import attach_first_token, score_logits
from answer_scoring.instruction import FORMAT_INSTRUCTION, format_user_prompt
from answer_scoring.reading import side_by_side
from answer_scoring.run_development import assert_writable_scoring_path, main, scoring_records_dir
from answer_scoring.surfaces import Surface, SurfaceSet, build_surfaces
from pilot.manifest import load_committed_manifest
from pilot.prompt import user_prompt
from scene_gen.normalize import leakage_aliases

ROOT = Path(__file__).resolve().parents[2]


class TableTokenizer:
    def __init__(self) -> None:
        self.table: dict[str, list[int]] = {}
        self.decode_map: dict[int, str] = {}
        self.next_id = 1

    def single(self, text: str, decoded: str, token_id: int | None = None) -> int:
        token = self.next_id if token_id is None else token_id
        if token_id is None:
            self.next_id += 1
        else:
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


def _counting_surfaces() -> SurfaceSet:
    def canonical(word: str, token_id: int) -> Surface:
        return Surface(word, word, (token_id,), "canonical_word", True, word)

    return SurfaceSet(
        "counting",
        {
            "one": (canonical("one", 1),),
            "two": (
                canonical("two", 2),
                Surface("two", "2", (5,), "digit_alias", True, "2"),
            ),
            "three": (canonical("three", 3),),
            "four": (canonical("four", 4),),
        },
        (),
    )


def test_digit_mass_increases_word_loss_and_decreases_class_loss():
    logits = torch.zeros(8, requires_grad=True)
    with torch.no_grad():
        logits[4] = 2.0
        logits[2] = 0.5
        logits[5] = 3.0
    word = word_only_nll(logits, 2)
    pooled = class_nll(logits, [2, 5])
    assert float(pooled.detach()) < float(word.detach())
    word.backward(retain_graph=True)
    assert float(logits.grad[5]) > 0
    logits.grad = None
    class_nll(logits, [2, 5]).backward()
    assert float(logits.grad[5]) < 0
    assert float(logits.grad[2]) < 0
    assert argmin_name({"four": 1.0, "two": 1.0}) == "four"


def test_two_token_digit_logprob_enters_the_class_mixture():
    word_lp = torch.tensor(-2.0, requires_grad=True)
    digit_lp = torch.tensor(-0.1, requires_grad=True)
    loss = mixture_nll(torch.stack([word_lp, digit_lp]))
    loss.backward()
    assert float(word_lp.grad) < 0
    assert float(digit_lp.grad) < 0
    first = torch.tensor([0.0, 2.0, -1.0])
    nxt = torch.tensor([-2.0, 0.5, 3.0])
    manual = torch.log_softmax(first, dim=-1)[1] + torch.log_softmax(nxt, dim=-1)[2]
    assert torch.allclose(two_token_logprob(first, nxt, 1, 2), manual)
    surfaces = SurfaceSet(
        "counting",
        {
            "one": (Surface("one", "one", (1,), "canonical_word", True, "one"),),
            "two": (
                Surface("two", "two", (2,), "canonical_word", True, "two"),
                Surface("two", "2", (9, 5), "digit_alias", True, " 2"),
            ),
            "three": (Surface("three", "three", (3,), "canonical_word", True, "three"),),
            "four": (Surface("four", "four", (4,), "canonical_word", True, "four"),),
        },
        (),
    )
    logits = torch.full((10,), -4.0)
    logits[4] = 3.0
    logits[2] = 0.0
    row = score_logits(
        logits,
        surfaces,
        raw_output="2",
        truncated=False,
        answer="two",
        aliases=["two", "2"],
        extra_logprobs={(9, 5): -0.05},
    )
    assert row["word_only_argmin"] == "four"
    assert row["class_argmin"] == "two"
    assert row["word_only_disagreement"] is True
    assert row["class_disagreement"] is False


def test_class_nll_is_the_log_sum_of_surface_probabilities():
    logits = torch.tensor([0.0, 0.2, 1.5, 0.0, -1.0])
    log_probs = torch.log_softmax(logits, dim=-1)
    expected = -torch.log(torch.exp(log_probs[1]) + torch.exp(log_probs[2]))
    assert torch.allclose(class_nll(logits, [1, 2]), expected)


def test_color_surfaces_keep_case_variants_and_skip_multi_token_spellings():
    surfaces = build_surfaces("attribute", _color_tokenizer(), leading_space=True)
    red = surfaces.by_class["red"]
    kinds = {surface.kind for surface in red}
    assert "canonical_word" in kinds
    assert "case_variant" in kinds
    assert "digit_alias" not in kinds
    assert any(item["reason"] == "multi_token" for item in surfaces.skipped)
    assert surfaces.canonical_token_id("red") == red[0].token_id
    sequences = [surface.token_ids for surface in red]
    assert len(sequences) == len(set(sequences))


def test_duplicate_bare_token_is_not_counted_twice():
    tokenizer = _color_tokenizer()
    tokenizer.table["red"] = list(tokenizer.table[" red"])
    surfaces = build_surfaces("attribute", tokenizer, leading_space=True)
    red_ids = [surface.token_ids for surface in surfaces.by_class["red"]]
    assert len(red_ids) == len(set(red_ids))
    assert any(item["reason"] == "duplicate_token" for item in surfaces.skipped)


def test_leading_space_digit_may_be_two_tokens():
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
    surfaces = build_surfaces("counting", tokenizer, leading_space=True)
    two = surfaces.by_class["two"]
    assert any(surface.kind == "digit_alias" and surface.token_ids == (216, 34) for surface in two)
    assert any(surface.kind == "digit_alias_bare" and surface.token_ids == (34,) for surface in two)
    assert any(item["reason"] == "multi_token" and item["source_text"] == "TWO" for item in surfaces.skipped)


def test_required_canonical_token_must_be_one_token():
    tokenizer = TableTokenizer()
    tokenizer.multi(" red")
    with pytest.raises(RuntimeError, match="Required surface"):
        build_surfaces("attribute", tokenizer, leading_space=True)


def test_digit_generation_matches_class_loss_and_not_word_loss():
    logits = torch.full((6,), -4.0)
    logits[4] = 3.0
    logits[2] = 0.0
    logits[5] = 4.5
    row = score_logits(
        logits,
        _counting_surfaces(),
        raw_output="2",
        truncated=False,
        answer="two",
        aliases=["two", "2"],
    )
    assert row["strict_correct"] is True
    assert row["strict_invalid"] is False
    assert row["word_only_argmin"] == "four"
    assert row["class_argmin"] == "two"
    assert row["word_only_disagreement"] is True
    assert row["class_disagreement"] is False
    assert row["true_class_nll"] < row["true_word_nll"]
    assert row["alternative_selected"] is False
    attached = attach_first_token(row, [5])
    assert attached["first_token_class"] == "two"
    outside = attach_first_token(row, [9])
    assert outside["first_token_in_surface_set"] is False


def test_embedded_label_stays_beside_the_strict_invalid_parse():
    row = score_logits(
        torch.zeros(6),
        _counting_surfaces(),
        raw_output="The count is 2.",
        truncated=False,
        answer="two",
        aliases=["two", "2"],
    )
    assert row["strict_invalid"] is True
    assert row["strict_correct"] is False
    assert row["strict_reasons"] == ["not_exact"]
    assert row["alternative_parsed_answer"] == "two"
    assert row["alternative_matches_answer"] is True
    assert row["alternative_selected"] is False
    assert row["word_only_disagreement"] is False
    assert row["class_disagreement"] is False
    reading = side_by_side("2", truncated=False, task_family="counting")
    assert reading["strict_parsed_answer"] == "two"
    assert reading["alternative_applicable"] is False


def test_format_line_is_the_same_on_every_context_and_does_not_leak():
    _manifest, records = load_committed_manifest(ROOT)
    assert len(records) == 24
    assert "{" not in FORMAT_INSTRUCTION
    for record in records:
        for context, payload in record["contexts"].items():
            base = user_prompt(payload["text"], record["question"])
            prompted = format_user_prompt(payload["text"], record["question"])
            assert prompted == base + "\n" + FORMAT_INSTRUCTION
            assert prompted.splitlines()[-1] == FORMAT_INSTRUCTION
            if context != "answer_revealing":
                assert leakage_aliases(prompted, list(record["answer_aliases"])) == []


def test_stored_predictions_keep_the_strict_parser_and_show_the_alias_gap():
    report = audit_predictions(load_predictions(ROOT))
    assert report["n_rows"] == 96
    assert report["n_reparse_mismatches"] == 0
    assert report["official_counts"] == {
        "correct": 31,
        "valid_wrong": 45,
        "invalid": 20,
        "disagreement": 15,
    }
    assert report["invalid_reason_counts"] == {"not_exact": 20}
    assert report["invalid_embedded_equals_answer"] == 10
    assert report["invalid_embedded_differs_from_answer"] == 10
    assert report["invalid_not_embedded"] == 0
    assert report["invalid_by_family"]["attribute"]["n"] == 14
    assert report["invalid_by_family"]["attribute"]["embedded_equals_answer"] == 9
    assert report["invalid_by_family"]["spatial"]["embedded_equals_answer"] == 1
    assert report["invalid_by_family"]["counting"]["n"] == 0
    assert report["disagreement_digit_generations"] == 13
    assert report["disagreement_case_color_generations"] == 2
    assert report["alternative_selected"] is False
    assert sum(report["official_counts"][key] for key in ("correct", "valid_wrong", "invalid")) == 96


def test_runner_refuses_cpu_and_checkpoint1_paths(tmp_path: Path):
    with pytest.raises(RuntimeError, match="checkpoint-1"):
        assert_writable_scoring_path(tmp_path / "docs" / "checkpoint-1" / "predictions.jsonl", tmp_path)
    records = scoring_records_dir(ROOT)
    assert records == (ROOT / "src" / "scoring" / "records").resolve()
    assert "checkpoint-1" not in records.parts
    with pytest.raises(SystemExit):
        main(["--device", "cpu"])
