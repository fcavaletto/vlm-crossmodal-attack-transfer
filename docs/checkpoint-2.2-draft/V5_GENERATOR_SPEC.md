# `scene_v5` generator specification (DRAFT, not built)

Status: specification only. No v5 code exists, and no v5 scene has been generated or rendered. `development_v4` (48 groups, 96 scenes) stays development data and is never pooled with v5.

## 1. Why a new generator

`scene_v4` (`src/scene_v4/scene_v4/design.py`) is a closed form. Index `i` in `0..47` builds one counterfactual pair, and the seed chooses nothing. The 96 scenes are the whole design. Its groups were also built to balance one another (every descriptor in every column, each ordered answer pair exactly 4 times), so they are not independent draws. v5 keeps every v4 constraint and replaces each index rule with a seeded random draw. Groups then become independent, identically distributed draws from one stated distribution.

## 2. What stays exactly as in v4

These are imported from `scene_v4`, not copied:
- canvas 512 × 384, a 4 × 2 grid of 128 × 192 cells, radii 60 and 28, `MIN_BOX_GAP` 8, `MIN_LARGE_COL_GAP` 2;
- palette, background, and outline;
- `SHAPES`, `COLORS`, `SIZES`, and the row names;
- the four context strings, built by `build_contexts`;
- the question template `What color is the {size} {shape} in the {row} row?`;
- the oracle (`answer_for`) and the renderer (no jitter, no speckle);
- the per-group checks in `_validate` and `_require_decoys`:
  - the two arms share the layout, the question, and the foil;
  - only `o0` and `o4` change color, and the arm swaps them;
  - the color histogram is [1, 2, 2, 3], and the answer is a double on both arms;
  - neither the modal nor the singleton color is ever an answer;
  - the foil is the singleton cross-row copy;
  - the size decoy and the row decoy force the size and the shape word;
  - 8 objects, 4 per row, exactly 2 large shapes per row at least 2 columns apart;
  - the oracle agrees with the design.

What v4 checked across groups (exact balance of answers, questions, shapes, sizes, rows, foils, and arm orders) is **not** enforced in v5. Enforcing balance would make groups depend on one another. Balance is reported, not imposed.

## 3. The sampling distribution, one group

Each group `k` of split `S` gets its own RNG, `numpy.random.Generator(PCG64(seed))`. The seed is the first 8 bytes of `SHA-256("{S.seed_string}|group|{k}|attempt|{a}")`, big-endian, starting at attempt `a = 0`. A group never shares random state with another group, so group `k` is reproducible on its own and does not depend on how many groups come before it.

Draws, in this order:

| # | Quantity | Distribution |
| --- | --- | --- |
| 1 | target size | uniform over {large, small} |
| 2 | target shape | uniform over {circle, square, triangle} |
| 3 | target row | uniform over {top, bottom} |
| 4 | target column | uniform over {0, 1, 2, 3} |
| 5 | ordered answer pair (arm A, arm B) | uniform over the 12 ordered pairs of distinct colors |
| 6 | modal and singleton colors | the remaining two colors, assigned by a fair coin |
| 7 | foil column (other row, same size and shape) | uniform over the 3 columns other than the target column |
| 8 | large-column set, target row | uniform over the sets in {(0,2), (0,3), (1,3)} consistent with the target's size and column |
| 9 | large-column set, other row | uniform over the sets consistent with the foil's size and column |
| 10 | size-decoy column (target row, target shape, other size) | uniform over the valid columns |
| 11 | row-decoy column and shape (target row, target size, other shape) | uniform over the valid columns, then uniform over the 2 other shapes |
| 12 | filler shapes | each remaining cell, in row-major order, uniform over the shapes `allowed` there |
| 13 | swap partner `o4` | uniform over the remaining cells whose shape is not the target shape |
| 14 | colors of the 3 other remaining cells | uniform random permutation of (answer A, answer B, modal) |

Arm B is arm A with the colors of `o0` and `o4` swapped, exactly as in v4.

**Rejection.** If any v4 check fails, or a step has no valid option, attempt `a` is discarded and the group is redrawn with `a + 1`. The target distribution is therefore the proposal above, conditioned on passing every check. This is exact and reproducible. It is not uniform over all valid layouts, and the protocol says so. Attempts per group are capped at 1,000, and hitting the cap is a generator error, not a silent skip. The attempt count is recorded per group, and the total rejection rate is reported.

**Novelty rule.** A group is also rejected and redrawn (counted separately) if its layout signature, or any of its three mirror images, equals:
- any `development_v4` group;
- any dev-v5 group (for the sealed split);
- an earlier group in the same split.

This is sampling without replacement from a very large finite set. Rough count: about 10⁶ or more valid layouts before colors. Duplicates should be rare, and the count is reported.

## 4. Splits and seeds

| Split | Seed string | Groups | Use |
| --- | --- | --- | --- |
| dev-v5 | `study2-checkpoint-2.2-dev-v5` | 32 | Generator tests. Clean scoring plus clean gradients only, no attack (gate G1). Predictors only, never an outcome. |
| sealed-v5 | `study2-checkpoint-2.2-sealed-v1` | `G_final` (§5) | The H-ΔL test. Rendered only after the protocol and analysis-script hashes are committed. The manifest hash is committed before any model run. |

Scene ids are `v5-{split}-{k:03d}-{a|b}`. Group ids are `cf5-{split}-{k:03d}`. Groups `0..G_final−1` are generated in order. If `G_final` is set from the range in §5, the first `G_final` groups are the sample. A group can never be added after outcomes exist, and a group can never be chosen by its outcome.

## 5. Sample size, fixed before the sealed render

`G_final` is fixed by a rule committed now, which uses **predictors only**:
1. Run G1 on dev-v5: clean scoring and both clean gradients for 64 scenes (about 50 minutes). It yields `A`, both norms, and both margins. No attack is run, and no `ΔL` is seen.
2. Rerun `sample_size.py`, with the predictor pool replaced by the dev-v5 groups and the error model kept from development (§13 of the protocol). Take the smallest `G` on the grid {128, 192, 256, 320, 384} with WCR power ≥ 0.80 for M1 at the smallest effect of interest (`r = 1`, full transfer).
3. `G_final = min(that G, G_max)`. `G_max` is decision D3b, set by the time available. If the cap binds, the report says that the study is under-powered for `r = 1`, and what power it has, before any outcome exists.

Using predictors to set the sample size does not bias the test, because no outcome enters the rule. Using dev-v5 instead of v4 matters: whether v5 spreads alignment more or less than v4 is the largest unknown in the power calculation.

## 6. Gates the generator must pass (C3, before any model run)

- Every scene passes the v4 per-scene checks (re-run on the written records, not just in memory).
- Contexts are constant within a group. The answer word is not a substring of any non-revealing context. `reveals_answer` is false for neutral and helpful.
- The novelty rule holds against `development_v4`, and dev-v5 against sealed.
- The same seed string gives byte-identical records and images on a second run.
- Realized balance is reported as counts per answer, ordered pair, descriptor, column, and foil. There is no pass/fail threshold, because imbalance under independent sampling is expected.

## 7. What the v5 population is, and what it is not

The H-ΔL estimand is defined over the v5 distribution in §3. It is the same family as v4, with the same constraints, but with different mixing proportions: v4 was exactly balanced, and v5 is random. Results generalize to that distribution, not to natural images, not to other prompts, and not to other models.

## 8. Tests to write with the generator

- **Reproducibility:** a fixed `(seed, k)` gives a fixed group, and changing `k` does not change other groups.
- **Distribution:** in 20,000 proposal draws, each step's marginal matches its stated distribution within Monte Carlo error. This is checked before rejection, since rejection changes it.
- **Rejection:** the rejection rate is finite and stable, and no accepted group fails any v4 check.
- **Novelty:** a planted v4 layout is rejected, and so is its mirror.
- **Sealed refusal:** the CLI refuses to render `sealed-v5` unless `PROTOCOL.md` and the analysis script hashes are given and match the committed values. The same guard style as `refuse_non_development_split`.
