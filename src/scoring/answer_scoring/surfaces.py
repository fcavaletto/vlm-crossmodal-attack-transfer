"""Parser-accepted spellings for one canonical answer.

The included spellings are the canonical word, its capitalized and uppercase
forms, and, for counting, the documented digit. Each spelling is tokenized with
and without the leading-space convention. A spelling is kept when decoding its
token ids parses back to the same canonical word.

Single-token spellings enter the first-position mixture. The documented digit
is kept even when the leading-space form is two tokens, because that is the
alias D4 already accepts. Multi-token case variants are skipped and recorded.
The leading-space canonical word must be one token. This set is fixed by the
parser and the tokenizer. It is not chosen by counting correct scenes.
"""

from __future__ import annotations

from dataclasses import dataclass

from vlm_iface.parsing import COUNT_DIGIT_ALIASES, parse_generation, task_vocabulary
from vlm_iface.tokens import answer_token_ids

_DIGIT_KINDS = frozenset({"digit_alias", "digit_alias_bare"})


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

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "canonical": self.canonical,
            "source_text": self.source_text,
            "token_ids": list(self.token_ids),
            "kind": self.kind,
            "leading_space": self.leading_space,
            "decoded": self.decoded,
        }
        if len(self.token_ids) == 1:
            payload["token_id"] = self.token_ids[0]
        return payload


@dataclass(frozen=True)
class SurfaceSet:
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


def build_surfaces(task_family: str, tokenizer, *, leading_space: bool = True) -> SurfaceSet:
    """Build the practical alias set for ``task_family``.

    ``leading_space`` is accepted so callers pass the session setting explicitly.
    The required canonical word uses that setting. Bare tokens are still inspected.
    """
    if leading_space is not True:
        raise ValueError("Checkpoint 1 scores answers with a leading space.")
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
    return SurfaceSet(task_family, frozen, tuple(skipped))


def _candidates(task_family: str) -> list[tuple[str, str, str, bool, bool]]:
    rows: list[tuple[str, str, str, bool, bool]] = []
    for word in task_vocabulary(task_family):
        forms = [word]
        capitalized = word[:1].upper() + word[1:]
        upper = word.upper()
        if capitalized != word:
            forms.append(capitalized)
        if upper not in forms:
            forms.append(upper)
        for index, form in enumerate(forms):
            kind = "canonical_word" if index == 0 else "case_variant"
            required = index == 0
            rows.append((word, form, kind, required, True))
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
    if len(token_ids) != 1 and kind not in _DIGIT_KINDS:
        return None, "multi_token"
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
