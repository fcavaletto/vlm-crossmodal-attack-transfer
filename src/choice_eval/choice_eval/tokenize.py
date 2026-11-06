"""Per-model tokenization of the four color continuations.

Candidates are not dropped when they are more than one token. Leading-space
and bare encodings are both recorded. The selected encoding is the
continuation required by the answer prefix. Two colors may not share a
selected token sequence. This module does not load a checkpoint.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from vlm_iface.parsing import parse_generation

from .colors import COLORS
from .models import require_model
from .prompt import continuation_for, prefix_ends_with_whitespace
from .protocol import COLOR_PARSER_TASK_FAMILY


class TokenizationError(ValueError):
    """The candidate strings do not have a usable tokenization."""


@dataclass(frozen=True)
class EncodedColor:
    color: str
    text: str
    token_ids: tuple[int, ...]
    pieces: tuple[str, ...] | None
    decoded: str
    multi_token: bool


@dataclass(frozen=True)
class TokenizationReport:
    model_key: str
    model_id: str
    revision: str
    answer_prefix: str
    prefix_ends_with_whitespace: bool
    selected: tuple[EncodedColor, ...]
    bare: tuple[EncodedColor, ...]
    leading_space: tuple[EncodedColor, ...]
    leading_space_changes_ids: tuple[bool, ...]
    any_selected_multi_token: bool
    joint_matches_appended: tuple[bool, ...]

    @property
    def leading_space_changes_every_color(self) -> bool:
        return all(self.leading_space_changes_ids)

    @property
    def joint_matches_every_color(self) -> bool:
        return all(self.joint_matches_appended)


def encode_text(tokenizer, text: str) -> tuple[int, ...]:
    """Encode ``text`` with no added specials. The same call must repeat."""
    if not isinstance(text, str) or text == "":
        raise TokenizationError("text to encode must be a non-empty string")
    first = _encode_once(tokenizer, text)
    second = _encode_once(tokenizer, text)
    if first != second:
        raise TokenizationError(f"tokenization of {text!r} is not deterministic")
    if not first:
        raise TokenizationError(f"{text!r} produced no tokens")
    return first


def verify_model_tokenization(tokenizer, *, model_key: str, answer_prefix: str) -> TokenizationReport:
    """Verify bare, leading-space, and selected encodings for one pinned model."""
    model = require_model(model_key)
    ends_with_space = prefix_ends_with_whitespace(answer_prefix)
    bare_ids = {color: encode_text(tokenizer, color) for color in COLORS}
    spaced_ids = {color: encode_text(tokenizer, " " + color) for color in COLORS}
    changes = tuple(bare_ids[color] != spaced_ids[color] for color in COLORS)
    selected_text = {color: continuation_for(color, answer_prefix=answer_prefix) for color in COLORS}
    selected_ids = {
        color: bare_ids[color] if selected_text[color] == color else spaced_ids[color] for color in COLORS
    }
    _reject_collisions(selected_ids)
    bare = tuple(_finish_color(tokenizer, color, color, bare_ids[color]) for color in COLORS)
    spaced = tuple(_finish_color(tokenizer, color, " " + color, spaced_ids[color]) for color in COLORS)
    selected = tuple(
        _finish_color(tokenizer, color, selected_text[color], selected_ids[color]) for color in COLORS
    )
    joint = tuple(
        _joint_matches(tokenizer, answer_prefix, item.text, item.token_ids) for item in selected
    )
    return TokenizationReport(
        model_key=model.key,
        model_id=model.model_id,
        revision=model.revision,
        answer_prefix=answer_prefix,
        prefix_ends_with_whitespace=ends_with_space,
        selected=selected,
        bare=bare,
        leading_space=spaced,
        leading_space_changes_ids=changes,
        any_selected_multi_token=any(item.multi_token for item in selected),
        joint_matches_appended=joint,
    )


def _finish_color(tokenizer, color: str, text: str, token_ids: tuple[int, ...]) -> EncodedColor:
    _reject_specials(tokenizer, token_ids, text)
    decoded = _decode(tokenizer, token_ids)
    if not _means_color(decoded, color):
        raise TokenizationError(f"tokens for {text!r} decode to {decoded!r}, not {color}")
    return EncodedColor(
        color=color,
        text=text,
        token_ids=token_ids,
        pieces=_pieces(tokenizer, token_ids),
        decoded=decoded,
        multi_token=len(token_ids) > 1,
    )


def _joint_matches(
    tokenizer, answer_prefix: str, continuation: str, continuation_ids: tuple[int, ...]
) -> bool:
    prefix_ids = encode_text(tokenizer, answer_prefix)
    joint_ids = encode_text(tokenizer, answer_prefix + continuation)
    return prefix_ids + continuation_ids == joint_ids


def _reject_collisions(selected_ids: Mapping[str, tuple[int, ...]]) -> None:
    owner: dict[tuple[int, ...], str] = {}
    for color, token_ids in selected_ids.items():
        previous = owner.get(token_ids)
        if previous is not None:
            raise TokenizationError(
                f"token sequence {token_ids} is not unique: {previous} and {color}"
            )
        owner[token_ids] = color


def _encode_once(tokenizer, text: str) -> tuple[int, ...]:
    try:
        encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    except TypeError as exc:
        raise TokenizationError(f"tokenizer rejected {text!r}: {exc}") from exc
    ids = encoded["input_ids"]
    if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1:
        raise TokenizationError("tokenizer must return input_ids with shape [1, time]")
    return tuple(int(token) for token in ids[0].tolist())


def _decode(tokenizer, token_ids: tuple[int, ...]) -> str:
    decode = getattr(tokenizer, "decode", None)
    if decode is None:
        raise TokenizationError("tokenizer.decode is required")
    decoded = decode(list(token_ids), skip_special_tokens=True)
    if not isinstance(decoded, str):
        raise TokenizationError("tokenizer.decode did not return a string")
    return decoded


def _pieces(tokenizer, token_ids: tuple[int, ...]) -> tuple[str, ...] | None:
    convert = getattr(tokenizer, "convert_ids_to_tokens", None)
    if convert is None:
        return None
    raw = convert(list(token_ids))
    if isinstance(raw, str):
        return (raw,)
    return tuple(str(piece) for piece in raw)


def _means_color(decoded: str, color: str) -> bool:
    # GPT-2 and SentencePiece display a leading space as Ġ or ▁. decode()
    # sometimes returns that marker instead of a space character.
    displayed = decoded.replace("Ġ", " ").replace("▁", " ")
    folded = " ".join(displayed.split())
    if folded == "":
        return False
    parsed = parse_generation(folded, truncated=False, task_family=COLOR_PARSER_TASK_FAMILY)
    return (not parsed.invalid) and parsed.parsed_answer == color


def _reject_specials(tokenizer, token_ids: tuple[int, ...], text: str) -> None:
    for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        value = getattr(tokenizer, name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            continue
        if value in token_ids:
            raise TokenizationError(f"{text!r} contains {name} {value}")
