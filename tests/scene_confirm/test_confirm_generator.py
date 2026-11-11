"""Tests for the Track B confirmation generator (no model)."""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in ("scene_v4", "scene_confirm"):
    sys.path.insert(0, str(ROOT / "src" / p))

from scene_confirm.draw import (  # noqa: E402
    CONFIRM_SEED_STRING, Rejected, _arms, attempt_seed, check_group, draw_group, draw_split, mirrors, propose,
)
from scene_confirm.generate import build  # noqa: E402
from scene_v4.checks import ambiguity_flags  # noqa: E402
from scene_v4.design import _signature, build_scenes  # noqa: E402


def test_reserved_seed_strings_refused():
    for s in ("study2-checkpoint-2.2-dev-v5", "study2-checkpoint-2.2-sealed-v1", "anything-else"):
        with pytest.raises(PermissionError):
            draw_split(s, 1, set())


def test_group_reproducible_and_independent_of_other_groups():
    a = draw_group(CONFIRM_SEED_STRING, 7, set())
    b = draw_group(CONFIRM_SEED_STRING, 7, set())
    assert a == b
    split = draw_split(CONFIRM_SEED_STRING, 10, set())
    assert split[7].arm_a == a.arm_a  # group 7 does not depend on groups 0..6 (no collision here)
    assert attempt_seed(CONFIRM_SEED_STRING, 7, 0) != attempt_seed(CONFIRM_SEED_STRING, 8, 0)


def test_proposal_marginals_match_the_stated_distribution():
    n = 20000
    sizes, shapes, rows, cols, pairs = Counter(), Counter(), Counter(), Counter(), Counter()
    for i in range(n):
        rng = np.random.Generator(np.random.PCG64(10_000 + i))
        q, a, b, foil, objs = propose(rng)
        w = q.split()
        sizes[w[4]] += 1
        shapes[w[5]] += 1
        rows[w[8]] += 1
        cols[objs[0].col] += 1
        pairs[(a, b)] += 1
    def close(counter, k):
        expected = n / k
        sd = (n * (1 / k) * (1 - 1 / k)) ** 0.5
        assert len(counter) == k and all(abs(v - expected) < 4.5 * sd for v in counter.values()), counter
    close(sizes, 2)
    close(shapes, 3)
    close(rows, 2)
    close(cols, 4)
    close(pairs, 12)


def test_accepted_groups_pass_every_v4_check_and_rejection_works():
    recs, stats = build(ROOT)
    assert len(recs) == 96 and stats["max_attempt_index"] < 1000
    for r in recs:
        assert ambiguity_flags(r) == []
    # A broken group (the partner keeps the referent's shape and size) is rejected.
    g = draw_group(CONFIRM_SEED_STRING, 0, set())
    o0 = g.arm_a.objects[0]
    bad = tuple(replace(o, shape=o0.shape, size=o0.size) if o.object_id == "o4" else o for o in g.arm_a.objects)
    left, right = _arms(0, g.arm_a.question, g.arm_a.intended_answer, g.arm_b.intended_answer, g.arm_a.foil, bad)
    with pytest.raises(Rejected):
        check_group(left, right)


def test_novelty_rejects_planted_development_layout_and_its_mirror():
    g = draw_group(CONFIRM_SEED_STRING, 3, set())
    sig = _signature(g.arm_a.objects)
    for planted in mirrors(sig):
        again = draw_group(CONFIRM_SEED_STRING, 3, mirrors(planted))
        assert again.attempt > g.attempt and again.rejections.get("novelty", 0) >= 1


def test_no_layout_shared_with_development_v4_or_within_split():
    dev = set()
    for s in build_scenes():
        dev |= mirrors(_signature(s.objects))
    split = draw_split(CONFIRM_SEED_STRING, 48, {_signature(s.objects) for s in build_scenes()})
    seen = set()
    for g in split:
        sig = _signature(g.arm_a.objects)
        assert sig not in dev and sig not in seen
        seen |= mirrors(sig)
