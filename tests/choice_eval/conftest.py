"""Stub tokenizers. They do not load a checkpoint."""

from __future__ import annotations

import pytest
import torch


class ScriptedTokenizer:
    def __init__(
        self,
        table: dict[str, list[int]],
        decode_map: dict[int, str],
        pieces: dict[int, str] | None = None,
    ) -> None:
        self.table = table
        self.decode_map = decode_map
        self.pieces = decode_map if pieces is None else pieces
        self.calls: list[str] = []
        self.eos_token_id: int | None = None
        self.bos_token_id: int | None = None
        self.pad_token_id: int | None = None
        self.unk_token_id: int | None = None
        self.rewrite = None

    def __call__(self, text: str, add_special_tokens: bool = False, return_tensors: str = "pt"):
        if add_special_tokens:
            raise AssertionError("special tokens must not be added")
        if return_tensors != "pt":
            raise AssertionError(return_tensors)
        self.calls.append(text)
        ids = torch.tensor([self.table[text]], dtype=torch.long)
        if self.rewrite is not None:
            ids = self.rewrite(text, ids)
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(self.decode_map[int(token)] for token in ids)

    def convert_ids_to_tokens(self, ids) -> list[str]:
        return [self.pieces[int(token)] for token in ids]


def color_tokenizer(*, multi_yellow: bool = False) -> ScriptedTokenizer:
    """One id per color, and a different id for the leading-space form.

    ``Assistant:`` does not end in whitespace. ``assistant\\n`` does. Joint
    encodings of those prefixes plus the selected continuation match the
    appended ids, except yellow when ``multi_yellow`` is set.
    """
    table: dict[str, list[int]] = {"Assistant:": [1, 2], "assistant\n": [7, 8]}
    decode = {1: "Assistant", 2: ":", 7: "assistant", 8: "\n"}
    pieces = dict(decode)
    bare_ids = {"red": 10, "blue": 20, "green": 30, "yellow": 40}
    spaced_ids = {"red": 11, "blue": 21, "green": 31, "yellow": 41}
    for color, bare in bare_ids.items():
        spaced = spaced_ids[color]
        if multi_yellow and color == "yellow":
            table[color] = [40, 46]
            table[" " + color] = [42, 43]
            decode[40] = "yel"
            decode[46] = "low"
            decode[42] = " yel"
            decode[43] = "low"
            pieces[40] = "yel"
            pieces[46] = "low"
            pieces[42] = "Ġyel"
            pieces[43] = "low"
            bare_tokens = [40, 46]
            spaced_tokens = [42, 43]
        else:
            table[color] = [bare]
            table[" " + color] = [spaced]
            decode[bare] = color
            decode[spaced] = " " + color
            pieces[bare] = color
            pieces[spaced] = "Ġ" + color
            bare_tokens = [bare]
            spaced_tokens = [spaced]
        table["Assistant: " + color] = [1, 2, *spaced_tokens]
        table["assistant\n" + color] = [7, 8, *bare_tokens]
    return ScriptedTokenizer(table, decode, pieces)


def _peaked_rows(n_tokens: int, margin: float, *, vocab: int = 4, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """``n_tokens`` rows whose token 1 beats token 0 by ``margin``."""
    row = [0.0, margin] + [0.0] * (vocab - 2)
    return torch.tensor([row for _ in range(n_tokens)], dtype=dtype)


@pytest.fixture
def peaked():
    return _peaked_rows


@pytest.fixture
def scripted() -> ScriptedTokenizer:
    return color_tokenizer()


@pytest.fixture
def multi_yellow() -> ScriptedTokenizer:
    return color_tokenizer(multi_yellow=True)
