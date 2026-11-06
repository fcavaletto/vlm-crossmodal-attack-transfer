"""Tokenization checks. These tests do not download a checkpoint."""

from __future__ import annotations

from pathlib import Path

import pytest
from vlm_iface.tokens import answer_token_ids

from choice_eval.tokenize import TokenizationError, encode_text, verify_model_tokenization
from choice_eval.verify_live import (
    ALLOW_PATTERNS,
    IGNORE_PATTERNS,
    assert_no_weight_files,
    prefer_generation_suffix,
)

REPO = Path(__file__).resolve().parents[2]


def test_leading_space_changes_ids_and_selects_from_the_prefix(scripted):
    tokenizer = scripted
    smol = verify_model_tokenization(tokenizer, model_key="smolvlm2", answer_prefix="Assistant:")
    assert smol.prefix_ends_with_whitespace is False
    assert smol.leading_space_changes_every_color
    assert [item.text for item in smol.selected] == [" red", " blue", " green", " yellow"]
    assert smol.selected[0].token_ids == (11,)
    assert smol.selected[0].pieces == ("Ġred",)
    assert "  red" not in tokenizer.calls
    qwen = verify_model_tokenization(tokenizer, model_key="qwen3-vl-2b", answer_prefix="assistant\n")
    assert qwen.prefix_ends_with_whitespace is True
    assert [item.text for item in qwen.selected] == ["red", "blue", "green", "yellow"]
    assert qwen.selected[0].token_ids == (10,)
    assert qwen.joint_matches_every_color
    assert smol.joint_matches_every_color


def test_candidate_encoding_matches_the_vlm_iface_rule(scripted):
    spaced = encode_text(scripted, " red")
    assert spaced == tuple(answer_token_ids(scripted, " red", leading_space=False).tolist())
    assert spaced == tuple(answer_token_ids(scripted, "red", leading_space=True).tolist())


def test_multi_token_candidates_are_kept(multi_yellow):
    report = verify_model_tokenization(multi_yellow, model_key="smolvlm2", answer_prefix="Assistant:")
    yellow = report.selected[3]
    assert yellow.color == "yellow"
    assert yellow.multi_token
    assert yellow.token_ids == (42, 43)
    assert len(report.selected) == 4
    assert report.any_selected_multi_token


def test_duplicate_sequences_are_rejected(scripted):
    tokenizer = scripted
    tokenizer.table[" blue"] = list(tokenizer.table[" red"])
    with pytest.raises(TokenizationError, match="not unique"):
        verify_model_tokenization(tokenizer, model_key="smolvlm2", answer_prefix="Assistant:")


def test_byte_level_space_marker_still_means_the_color(scripted):
    scripted.decode_map[11] = "Ġred"
    report = verify_model_tokenization(scripted, model_key="smolvlm2", answer_prefix="Assistant:")
    assert report.selected[0].decoded == "Ġred"
    assert report.selected[0].token_ids == (11,)


def test_decode_must_be_the_color(scripted):
    tokenizer = scripted
    tokenizer.decode_map[11] = "blue"
    with pytest.raises(TokenizationError, match="red"):
        verify_model_tokenization(tokenizer, model_key="smolvlm2", answer_prefix="Assistant:")


def test_special_tokens_in_a_candidate_are_rejected(scripted):
    tokenizer = scripted
    tokenizer.eos_token_id = 11
    with pytest.raises(TokenizationError, match="eos_token_id"):
        verify_model_tokenization(tokenizer, model_key="smolvlm2", answer_prefix="Assistant:")


def test_nondeterministic_encoding_is_rejected(scripted):
    tokenizer = scripted
    seen = {"red": 0}

    def rewrite(text, ids):
        if text == "red":
            seen["red"] += 1
            if seen["red"] == 2:
                return ids.clone().fill_(99)
        return ids

    tokenizer.rewrite = rewrite
    tokenizer.decode_map[99] = "red"
    with pytest.raises(TokenizationError, match="not deterministic"):
        verify_model_tokenization(tokenizer, model_key="qwen3-vl-2b", answer_prefix="assistant\n")


def test_joint_boundary_mismatch_is_reported(scripted):
    tokenizer = scripted
    tokenizer.table["Assistant: red"] = [1, 9, 11]
    report = verify_model_tokenization(tokenizer, model_key="smolvlm2", answer_prefix="Assistant:")
    assert report.joint_matches_appended[0] is False
    assert report.joint_matches_appended[1:] == (True, True, True)
    assert report.selected[0].token_ids == (11,)


def test_weight_files_are_refused(tmp_path):
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert_no_weight_files(tmp_path, max_file_bytes=100)
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"weights")
    with pytest.raises(RuntimeError, match="weight file"):
        assert_no_weight_files(tmp_path, max_file_bytes=100)
    weight.unlink()
    huge = tmp_path / "tokenizer.json"
    huge.write_text("x" * 20, encoding="utf-8")
    with pytest.raises(RuntimeError, match="cap"):
        assert_no_weight_files(tmp_path, max_file_bytes=10)


def test_image_chat_shape_wins_when_suffixes_agree():
    found = [
        {"attempt": "string", "generation_suffix": "Assistant:"},
        {"attempt": "image-marker", "generation_suffix": "<end_of_utterance>\nAssistant:"},
        {"attempt": "text-part", "generation_suffix": "<end_of_utterance>\nAssistant:"},
    ]
    chosen = prefer_generation_suffix(found)
    assert chosen["attempt"] == "image-marker"
    assert chosen["suffixes_agree"] is False
    agreed = [
        {"attempt": "text-part", "generation_suffix": "same"},
        {"attempt": "pil-image", "generation_suffix": "same"},
    ]
    assert prefer_generation_suffix(agreed)["attempt"] == "pil-image"
    assert prefer_generation_suffix(agreed)["suffixes_agree"] is True


def test_live_downloader_does_not_allow_weight_patterns():
    allowed = " ".join(ALLOW_PATTERNS).lower()
    ignored = " ".join(IGNORE_PATTERNS).lower()
    assert "safetensors" not in allowed
    assert ".bin" not in allowed
    assert "safetensors" in ignored
    source = (REPO / "src" / "choice_eval" / "choice_eval" / "verify_live.py").read_text(encoding="utf-8")
    assert "AutoModel" not in source
    assert "from_pretrained" in source
