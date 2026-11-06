"""Practical surface set S(y) for the class loss.

A spelling is kept when its tokens decode to a string the strict parser accepts
as class y. The candidates are fixed in advance: the canonical word, its
capitalized form, and its uppercase form, each with and without a leading
space, plus the documented counting digits with and without a leading space.

A leading-space digit may be two tokens and is kept. Any other multi-token
spelling is skipped and listed. On the SmolVLM tokenizer recorded in
Checkpoint 1.5, the skipped uppercase spellings are BLUE, GREEN, YELLOW, TWO,
THREE, and FOUR. This module does not load that tokenizer. It applies the
same rule to whatever tokenizer the caller passes. The leading-space canonical
word is required to be one token.
"""

from __future__ import annotations

from dataclasses import dataclass

from vlm_iface.parsing import parse_generation
from vlm_iface.tokens import answer_token_ids

from .vocabulary import COUNT_DIGIT_ALIASES, spelling_forms, task_vocabulary

DIGIT_KINDS = frozenset({"digit_alias", "digit_alias_bare"})


@dataclass(frozen=True)
class Surface:
    canonical: str
    source_text: str
    token_ids: tuple[int, ...]
    kind: str
    leading_space: bool
    decoded: str

    @property
    def token_id(self) -> int:
        if len(self.token_ids) != 1:
            raise RuntimeError(f"{self.source_text!r} is not a single token")
        return self.token_ids[0]


@dataclass(frozen=True)
class SurfaceInventory:
    task_family: str
    by_class: dict[str, tuple[Surface, ...]]
    skipped: tuple[dict[str, str | bool], ...]

    def canonical_token_id(self, canonical: str) -> int:
        matches = [
            surface.token_id
            for surface in self.by_class[canonical]
            if surface.kind == "canonical_word" and surface.leading_space and len(surface.token_ids) == 1
        ]
        if len(matches) != 1:
            raise RuntimeError(f"{canonical} does not have one leading-space canonical token")
        return matches[0]

    def without_kinds(self, kinds: frozenset[str] | set[str]) -> SurfaceInventory:
        """Drop surfaces of the named kinds. Every class must keep one surface."""
        dropped = frozenset(kinds)
        kept: dict[str, tuple[Surface, ...]] = {}
        for name, group in self.by_class.items():
            remaining = tuple(surface for surface in group if surface.kind not in dropped)
            if not remaining:
                raise ValueError(f"{name} has no surfaces after dropping {sorted(dropped)}")
            kept[name] = remaining
        return SurfaceInventory(self.task_family, kept, self.skipped)


def build_surfaces(task_family: str, tokenizer, *, leading_space: bool = True) -> SurfaceInventory:
    """Build S(y) for every class in ``task_family``."""
    if leading_space is not True:
        raise ValueError("Answers are scored with a leading space.")
    vocabulary = task_vocabulary(task_family)
    by_class: dict[str, list[Surface]] = {word: [] for word in vocabulary}
    skipped: list[dict[str, str | bool]] = []
    owner: dict[tuple[int, ...], str] = {}
    for canonical, source_text, kind, required, use_leading in _candidates(task_family):
        surface, reason = _one_surface(
            tokenizer,
            task_family=task_family,
            canonical=canonical,
            source_text=source_text,
            kind=kind,
            leading_space=use_leading,
        )
        if surface is None:
            record = {
                "canonical": canonical,
                "source_text": source_text,
                "kind": kind,
                "leading_space": use_leading,
                "reason": reason or "skipped",
            }
            if required:
                raise RuntimeError(
                    f"Required surface {source_text!r} for {canonical} is unusable: {record['reason']}"
                )
            skipped.append(record)
            continue
        if kind == "canonical_word" and use_leading and len(surface.token_ids) != 1:
            raise RuntimeError(f"Canonical word {source_text!r} is not one token.")
        previous = owner.get(surface.token_ids)
        if previous is not None and previous != canonical:
            raise RuntimeError(
                f"Token sequence {surface.token_ids} maps to both {previous!r} and {canonical!r}."
            )
        if any(existing.token_ids == surface.token_ids for existing in by_class[canonical]):
            skipped.append(
                {
                    "canonical": canonical,
                    "source_text": source_text,
                    "kind": kind,
                    "leading_space": use_leading,
                    "reason": "duplicate_token",
                }
            )
            continue
        owner[surface.token_ids] = canonical
        by_class[canonical].append(surface)
    frozen = {word: tuple(by_class[word]) for word in vocabulary}
    for word, surfaces in frozen.items():
        if not surfaces:
            raise RuntimeError(f"{word} has no surfaces")
    return SurfaceInventory(task_family, frozen, tuple(skipped))


def _candidates(task_family: str) -> list[tuple[str, str, str, bool, bool]]:
    rows: list[tuple[str, str, str, bool, bool]] = []
    for word in task_vocabulary(task_family):
        for index, form in enumerate(spelling_forms(word)):
            kind = "canonical_word" if index == 0 else "case_variant"
            rows.append((word, form, kind, index == 0, True))
            rows.append((word, form, f"{kind}_bare", False, False))
    if task_family == "counting":
        for digit, canonical in COUNT_DIGIT_ALIASES.items():
            rows.append((canonical, digit, "digit_alias", True, True))
            rows.append((canonical, digit, "digit_alias_bare", False, False))
    return rows


def _one_surface(
    tokenizer,
    *,
    task_family: str,
    canonical: str,
    source_text: str,
    kind: str,
    leading_space: bool,
) -> tuple[Surface | None, str | None]:
    try:
        ids = answer_token_ids(tokenizer, source_text, leading_space=leading_space)
    except (TypeError, ValueError) as exc:
        return None, str(exc)
    token_ids = tuple(int(token) for token in ids.tolist())
    if len(token_ids) != 1 and kind not in DIGIT_KINDS:
        return None, "multi_token"
    if len(token_ids) > 2:
        return None, "longer_than_two_tokens"
    decoded = tokenizer.decode(list(token_ids), skip_special_tokens=True)
    if not isinstance(decoded, str):
        return None, "decode_not_string"
    parsed = parse_generation(decoded, truncated=False, task_family=task_family)
    if parsed.invalid or parsed.parsed_answer != canonical:
        return None, "does_not_parse_to_class"
    return (
        Surface(
            canonical=canonical,
            source_text=source_text,
            token_ids=token_ids,
            kind=kind,
            leading_space=leading_space,
            decoded=decoded,
        ),
        None,
    )
