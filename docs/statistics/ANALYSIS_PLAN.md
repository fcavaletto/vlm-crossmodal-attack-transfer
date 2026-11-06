# Analysis plan

Status: proposal for Checkpoint 0. This document defines tables, metrics, and uncertainty for a later confirmatory study. It is not a result about a vision-language model. The primary estimand is not locked. Agent A recommends that estimand. The research lead decides it. Labels below are proposals.

The numbers in the worked example are arithmetic on hand-written rows in `src/analysis/analysis/fixtures/toy/`. They were typed so the formulas can be checked by hand. They are fabricated. They are not measurements.

## What Checkpoint 0 scores, and what it leaves open

The study questions, copied from the shared checkpoint spec, are questions rather than conclusions:

- RQ1. Does textual context change the adversarial vulnerability of a VLM when the text does not directly reveal the correct answer?
- RQ2. Can directional agreement between input-image gradients under different textual contexts predict cross-context adversarial attack transfer?
- RQ3. Can context-specific or multi-context attacks exploit vulnerabilities that neutral-context attacks miss?

Representation interventions are exploratory only. They are named here and not implemented.

This package does not load a model, does not differentiate through one, and does not run an attack. It scores tables that some later run may write. A loss value in a table is an input. A gradient vector in the toy audit file is an input. Clean `correct` and `invalid` flags are inputs. This package does not re-parse answer text and does not recompute a likelihood.

## Layout

The installable project is `src/analysis`. The Python import package inside it is `analysis`. Dependencies are pinned in `src/analysis/pyproject.toml` (`numpy==1.26.4` for the algebra that was tested; `pytest==8.3.5` for the tests). PyTorch is not a dependency. Image gradients, if a later caller has them as tensors, can be passed in as numeric arrays. Agent C owns the model interface.

## Units and grouping

A scene is the unit. One scene has one image seed, one question, and four textual contexts: `neutral`, `helpful_nonrevealing`, `contradictory`, and `answer_revealing`. Contexts that share a scene are paired measurements. A mean is a mean over scenes. A bootstrap draw resamples scenes. It does not draw a context row from one scene and a context row from another and then pretend those two rows are a pair.

Counterfactual scenes may also be dependent. The analysis tables in this checkpoint do not carry `counterfactual_group_id`. The bootstrap accepts an optional `cluster_of` map from scene id to cluster id. The toy summary does not pass one. Passing a counterfactual group id later resamples whole groups. A drawn group contributes each of its scenes, so a larger group weighs more than a smaller one. That sensitivity is recommended before any confirmatory interval. It is not applied to the toy file.

Scenes present on only one side of a paired contrast are left out of the mean and counted. The toy illustration is a complete grid, so that count is zero there.

## Data dictionary

JSONL, one object per line. Unknown fields are rejected at this checkpoint. A later relaxation is a lead decision.

Contexts, exact keys:

| Key | Role in the proposal |
| --- | --- |
| `neutral` | Evaluation context and a source context for neutral-context attacks |
| `helpful_nonrevealing` | Helpful text that must not contain the answer. Leakage checks belong to the scene generator |
| `contradictory` | Text that conflicts with the answer |
| `answer_revealing` | Positive control only. A high score here is the text telling the model the answer. It is never evidence of visual robustness by itself |

### Predictions

Fields: `scene_id`, `context`, `parsed_answer`, `invalid`, `correct`.

One row per scene and context. `correct` and `invalid` are booleans. An invalid row cannot be marked correct. `parsed_answer` is stored and not re-scored. On the toy rows, a correct parse is the string `alpha`, a wrong parse is `beta`, and an invalid parse is an empty string. Those strings are labels for the exercise.

### Losses

Fields: `scene_id`, `context`, `loss`.

`loss` is a finite number. The candidate loss elsewhere in the project is

```
L_c(x) = -log P(y | x, c)
```

`x` is the image, `c` is the textual context, `y` is the ground-truth answer, and `P` is the model's probability of that answer given the image and the context. This package does not estimate `P`. Mean loss by context is descriptive. It is not an endpoint.

The loss table has no attack field. These rows are clean-image losses under each context.

### Gradient summaries

Fields: `scene_id`, `source_context`, `target_context`, `alignment_a`.

`alignment_a` is the scalar `A(s, t)` defined below, or null when that scalar is undefined. Raw image gradients are not part of this schema. They can be large. The toy packet adds an audit file, `gradient_vectors.jsonl`, with fields `scene_id`, `source_context`, `target_context`, `g_source`, and `g_target`. Verification recomputes `A(s, t)` from those vectors and checks the stored scalar. The vectors in that file have length 2 to 4. They are not images.

### Attacks

Fields: `scene_id`, `source_context`, `target_context`, `method`, `epsilon`, `step_budget`, `correct_before`, `correct_after`, `invalid_after`.

Proposed meaning of the flags:

- `correct_before` is clean correctness under `target_context`, the context in which the attacked image is scored.
- `correct_after` is correctness of the attacked image under that same target context.
- `invalid_after` marks an unusable attacked generation under the target context.

The toy builder rejects a packet where `correct_before` disagrees with the clean prediction for that scene and target context. Epsilon and step budget on these rows are cell labels. They are not chosen attack budgets. Two epsilons that match at 8 decimal places share a cell key of the form `0.01000000`. That key is bookkeeping.

Method labels:

| Rows | How the label is used |
| --- | --- |
| `method=pgd`, `source_context=neutral` | Neutral-context PGD |
| `method=pgd`, `source_context` a single context | Context-specific PGD |
| `method=pgd_multi`, `source_context=multi` | Multi-context PGD. `multi` is a proposed extra source label so one scalar field can name it |
| `method=fgsm` | FGSM control |
| `method=noise` | Noise control |

`pgd_multi` and `multi` have to appear together. `target_context` is always one of the four contexts, because a row is scored under one text condition.

There is no `perturbation_id`. Two rows with the same source label are not checked to be the same pixels. Cross-context transfer in this package means "rows that carry this source label and this target label." Adding a shared perturbation id before any real attack log is a recommendation, not a field in this checkpoint.

## Metrics

### Clean accuracy

For one context, count scenes whose `correct` flag is true and divide by the number of scenes in that context. An invalid output is not correct, so it lowers accuracy. The invalid-output rate, below, keeps that failure visible as its own count.

If a context has no scenes, the rate is null. The reason string says there were no scenes. The rate is not stored as zero. Zero would look like a measured total failure.

On the fabricated clean rows the counts are 5/6 for `neutral`, 4/6 for `helpful_nonrevealing`, 2/6 for `contradictory` (1 of those 6 is invalid), and 5/6 for `answer_revealing` (1 of those 6 is invalid). The revealing row is a positive control.

### Invalid-output rate

For one context, count scenes with `invalid` true and divide by the number of scenes. The same empty-context rule applies: null, with a reason, when the denominator is zero.

Read this beside accuracy and beside attack success. An attack that turns answers into unparseable text raises the invalid-output rate. It does not, under the rule below, raise the attack success rate.

### Attack success rate

Start from scenes that were correct before the attack. A success is a later answer that is valid and wrong. The rate is

```
successes / scenes that were correct before the attack
```

An invalid later answer stays in the denominator and does not enter the success count. If the denominator is zero, the rate is null. The reason says that no scene was correct before the attack. A cell of all failures-before is not an attack success rate of zero.

The toy same-context cell `pgd`, epsilon label `0.01`, step label `1`, source `neutral`, target `neutral`, has 3 successes out of 5 clean-correct scenes, so the rate is 3/5. Scene `toy-05` was correct before, then invalid after. It is one of the 5 and it is not one of the 3. The scene bootstrap interval for that rate, stored in the summary, has estimate 0.600000 and endpoints 0.166667 and 1.000000. Six typed-in scenes leave a wide interval. That width is a property of those rows and of the bootstrap. It is not a property of a model.

The noise cell contains only `toy-04`, which is already wrong on clean neutral. Its attack success rate is null. The cell exists so the null is visible in the committed file.

### Additional error

For one scene, task error `e` is 0 when the answer is correct and 1 when it is not. Invalid output is not correct, so it has task error 1.

```
Delta = e_after - e_before
```

`Delta` is in {-1, 0, +1}. The cell summary is the mean of `Delta` over scenes in the cell. A scene that was right and stays right contributes 0. A scene that was right and becomes wrong or invalid contributes +1. A scene that was wrong and becomes right contributes -1. A scene that was already wrong cannot contribute +1. When clean error is at least half the scenes in the cell, the summary carries a floor warning: a small additional error can be the ceiling imposed by that clean error.

On the toy neutral PGD cell, the six contributions are +1, +1, 0, 0, +1, +1. The mean is 4/6. Scene `toy-05` contributes +1 here and contributes nothing to the success count above. The two metrics answer different questions. Additional error asks whether the task error went up. Attack success asks whether a clean-correct scene became a valid wrong answer.

### Cross-context attack transfer

Each attack row names the context used to build the perturbation (`source_context`) and the context used to score it (`target_context`). When those differ, the cell's attack success rate and additional error are the transfer summaries for that ordered pair. When they match, the cell is the same-context reference.

`cross_context_attack_transfer` returns the differing pairs and leaves out positive-control pairs. Positive-control rows remain in the full cell list with `positive_control_only` true.

An exploratory rate, `conditional_transfer`, asks a narrower question. Among same-context successes that were also correct on the target before the attack, what fraction are target successes? Scenes that were already incorrect on the target are excluded and counted, not treated as transfer failures. Invalid output is still not a success. If the eligible count is zero, the rate is null.

On the fabricated rows with source `helpful_nonrevealing` and target `contradictory`, there are 4 same-context successes, 2 of those are excluded because the target was already incorrect, and 1 of the 2 eligible scenes is a target success. The rate is 1/2. That fraction is the arithmetic of those rows.

### Gradient alignment

Let `g_s` be the image gradient of `L_s` and `g_t` the image gradient of `L_t`. The candidate definition of each gradient is the derivative of `L_c` with respect to the image `x`. The proposed measurement point is the clean image, before any attack step. The function in this package only sees the arrays it is given. It cannot tell which image they came from.

`sign(g_s)` replaces each component by 1 when it is positive, by -1 when it is negative, and by 0 when it is zero, including negative zero. `||g_t||_1` is the sum of the absolute values of `g_t`. Then

```
A(s, t) = dot(g_t, sign(g_s)) / ||g_t||_1
```

`dot` is the sum of component-wise products. `s` is the source context. `t` is the evaluation context.

`A(s, t)` equals 1 when every nonzero component of `g_t` points the same way as `g_s` and `g_s` is not flat on that component. It equals -1 when every nonzero component of `g_t` points the opposite way. A component of `g_t` where `g_s` is zero adds to the denominator and nothing to the numerator, which pulls `A(s, t)` toward 0. When the denominator is positive, `A(s, t)` lies in [-1, 1]. Components of `g_t` with larger absolute value count more.

If `||g_t||_1` is 0, `A(s, t)` is null. A flat target gradient has no direction to compare. Those scenes are dropped from the mean alignment and counted. They are not filled in as 0. Filling them in as 0 would treat "no target direction" as "the directions disagree."

If `g_s` is all zeros and `g_t` is not, every sign is 0, the dot product is 0, and `A(s, t)` is 0. That is the formula. It says the source gradient contributed no sign. It does not say the two contexts were studied and found to be unrelated.

Worked vectors for scene `toy-01`, source `neutral`, target `helpful_nonrevealing`:

```
g_s = (1, -2, 0, 0.5)
g_t = (3, -1, 4, -2)
sign(g_s) = (1, -1, 0, 1)
dot = 3 + 1 + 0 - 2 = 2
||g_t||_1 = 3 + 1 + 4 + 2 = 10
A = 2/10 = 0.2
```

Scene `toy-04` on that same pair has a zero target vector, so `A` is null. The five defined values on that pair are 0.2, 1, -1, 0, and 0. Their ordinary average is 0.04. The summary stores the float64 mean of those stored numbers. Verification rebuilds that float. It is not a second estimate.

A first-order sign alignment is not proof that a PGD attack will transfer. The association between `A(s, t)` and a transfer indicator is a proposed secondary description for RQ2. On the toy join the Pearson correlation is -0.485071. The sign is an artifact of the hand-written pairs. It is not evidence about vision-language models.

### Paired comparison

The proposed primary illustration, unlocked, is the mean over scenes of

```
Delta(contradictory) - Delta(neutral)
```

for the toy PGD cell whose source label is `neutral`, epsilon label is `0.01`, and step label is `1`. The six scene differences are 0, -1, 0, -1, -1, -1. The mean is -4/6. The scene bootstrap interval in the summary has estimate -0.666667 and endpoints -1.000000 and -0.333333 (seed 20261106, 2000 replicates, NumPy linear quantile). The interval lies below zero because four of the six typed-in differences are -1 and none are positive. Replace the rows and the interval moves. This is a check of the paired mean. It is not evidence that contradictory text reduces vulnerability.

`answer_revealing` is not an arm of this illustration.

This contrast has the algebra of a difference in differences: each arm is already an attacked error minus a clean error, and the reported number is the difference of those arms. The arms are an open choice. Study 1 used a paired additional-error contrast on its own context names. Those names and that contrast are not adopted here. This study's contexts are the four keys above, and the revealing context is a positive control.

### Zero denominators

Every rate in this package is a numerator, a denominator, a value, and a reason. The value is null when the denominator is zero. Callers can still see the numerator. Means over an empty scene list are null as well. Alignment with a zero target L1 norm is null. A bootstrap interval whose point estimate is undefined is left undefined. Ratio replicates that draw a zero denominator are omitted from the percentile interval and counted in `n_undefined_replicates`.

## Uncertainty

The interval is an equal-tailed percentile interval. Draw scenes with replacement, recompute the estimate, repeat, and take the 2.5% and 97.5% points with NumPy's linear quantile. The toy file uses 2000 draws and seed 20261106 so a fresh clone rebuilds the same endpoints on NumPy 1.26.4. That seed is for the toy file. It is not the confirmatory seed. The confirmatory seed, the replicate count, and the single primary contrast should be registered before a sealed split is scored.

For a mean, each scene contributes one number. The draw averages those numbers.

For attack success, each scene contributes a denominator bit and a numerator bit. The draw recomputes the ratio. Scenes that were already wrong stay in the draw. They add neither a success nor a denominator, and they still occupy a slot, so they change how often the other scenes are drawn. A draw whose denominator is zero is omitted from the percentile interval and counted.

There is no p-value and no normal approximation. There is no multiplicity adjustment. A single pre-specified contrast does not need one. A family of contrasts would.

A hypothetical with two scenes shows the grouping. On each scene the two contexts have the same additional error, so every paired difference is 0. The scene bootstrap interval for the mean difference is exactly [0, 0]. The scenes still differ from each other. An analysis that broke the pair could move. The interval here does not move, because the resample never separates the two contexts of one scene.

## Proposed roles

These labels are proposals. `labels_are_proposals` is true in the toy summary. `primary_estimand_locked` is false.

| Analysis | Role | Question |
| --- | --- | --- |
| Paired difference in additional error for one pre-specified attack cell, with arms that are not `answer_revealing` | Proposed primary | RQ1 |
| Clean accuracy and the invalid-output rate | Proposed secondary | Floor, always shown beside a vulnerability contrast |
| Attack success rate | Proposed secondary | RQ1 and RQ3 |
| Additional error by cell | Proposed secondary | RQ1 and RQ3 |
| Cross-context cell summaries | Proposed secondary | RQ2 and RQ3 |
| Same-context cells, FGSM, and noise | Proposed secondary | Reference and controls |
| Mean `A(s, t)` | Proposed secondary | RQ2 |
| Association of `A(s, t)` with a transfer indicator | Proposed secondary, unlocked | RQ2 |
| Multi-context cells (`pgd_multi`) | Proposed secondary | RQ3 |
| Conditional transfer given a same-context success | Exploratory | RQ2 |
| Precision simulation below | Exploratory planning tool | Design |
| Internal-representation interventions | Exploratory | Outside the primary claim. Not coded |
| Anything with `answer_revealing` | Positive control | Not a vulnerability claim |
| Mean loss by context | Descriptive, not an endpoint | None |

## Sample-size simulation

`python -m analysis.simulation` draws a fabricated distribution. Each scene contributes an independent difference `D` in {+1, -1, 0} with `P(D = +1) = p_plus` and `P(D = -1) = p_minus`. The default is `p_plus = 0.25` and `p_minus = 0.05`, so the true mean is 0.20. The command then runs the same scene percentile bootstrap used for paired means, on a grid of scene counts, and records the median interval width and the fraction of intervals that cover that known mean.

The default grid is 12, 24, and 48. Those counts bracket the provisional Checkpoint 0 development sample of 24–48 scenes. They are not a power calculation for a sealed split of 256, and they are not a recommendation that any particular `n` is enough. Real scenes will be dependent across counterfactuals. Real outcomes will not follow this three-point distribution. The primary contrast is not locked. The printed JSON sets `not_an_empirical_finding` to true. Widths from that printout are properties of the draw. They are not sample-size results for the study.

The simulation does not read the compact fixtures. The verification command does not run it. Tests use a small grid and check two properties that follow from the procedure: the same seed repeats, and under a spread-out distribution the median width at 48 scenes is smaller than the median width at 12.

## What is reused from Study 1

Study 1 is read-only. Its published numbers are not inputs.

The verification idea is reused. Study 1 keeps a compact table and a published aggregate, and a command rebuilds the aggregate without a GPU and without gitignored dumps. That split still applies: scoring tables is cheap, and a fresh clone should be able to check the arithmetic. The command here is `python -m analysis.verify_toy`. The compact inputs are the toy JSONL files. The aggregate is `toy_summary.json`. Exit status 0 means they match.

The scene as the resampling unit is reused in spirit. Study 1 resampled images because contexts on one image are paired. This study's contexts are paired on a scene for the same reason. The interval is a percentile bootstrap of scene contributions. The arms are not Study 1's arms.

Strict clean accuracy is reused in spirit. Both studies score a parser against a known answer, and an unusable string is not a correct answer. Invalid-output rate is reported beside accuracy so the two failures stay distinct.

Additional error is reused in spirit. A low clean accuracy leaves little room for a new error, which is the clean-floor threat named in the checkpoint spec. The floor warning is there so a small `Delta` is not told as "the attack did nothing."

Attack success is defined differently. The checkpoint spec says invalid generation is not an attack success. A rule that treated every non-correct output as a success would fold invalid strings into the success count. This plan keeps them in the denominator and out of the numerator.

## Verification

From the repository root, after the install in the next section:

```bash
python -m analysis.verify_toy
```

The command loads the five JSONL fixtures, checks the schema, checks that every attack row's `correct_before` matches the clean target prediction, recomputes `A(s, t)` from the audit vectors, rebuilds the summary, and compares it to the committed JSON. Floats may differ by at most `1e-9`. Non-finite tokens are rejected. The command does not rewrite the file.

`--write` is a maintainer switch that regenerates `toy_summary.json`. Reviewers run the command without it.

The summary records SHA-256 hashes of the five JSONL files. Editing a fixture without rebuilding the summary fails the comparison.

## How to run

From a fresh clone, at the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e 'src/analysis[dev]'
python -m pytest -q tests/analysis
python -m analysis.verify_toy
```

Optional planning sketch, fabricated distribution only:

```bash
python -m analysis.simulation --n 12 24 48 --replications 100 --boot 400 --seed 20261106
```

Tested interpreter for the checkpoint report: the run recorded there. Requires Python 3.11 or newer. The pin that the tests imported is `numpy==1.26.4`.

## Open choices

The report lists these for the research lead. Short form:

1. Primary summary for RQ1: additional-error contrast, attack success, or another contrast Agent A names. Recommendation: additional error as the primary contrast, with attack success beside it. Not locked.
2. Invalid attacked outputs stay in the attack-success denominator. Alternative: drop them and condition on a parseable attacked answer.
3. `correct_before` means the clean target context. Alternative: the clean source context.
4. `A(s, t)` is proposed at the clean image. The function itself is only algebra.
5. Bootstrap unit is the scene now. Add a counterfactual-group sensitivity before confirmatory intervals.
6. Attack-success draws resample every scene in the cell. Alternative: resample only clean-correct scenes.
7. Multi-context rows use `method=pgd_multi` and `source_context=multi`.
8. Exploratory conditional transfer excludes scenes already wrong on the target and counts those exclusions.
9. Undefined alignments are dropped from the mean and counted, not imputed as 0.
10. Register the confirmatory seed, replicate count, and one primary contrast before any sealed split. Do not reuse seed 20261106 for that split. No multiplicity adjustment is implemented.
11. Add a `perturbation_id` before real attack logs exist. It is not in the shared field list, so this checkpoint does not add it.
12. Intervals are percentile bootstrap intervals. Wald and BCa intervals are not implemented. No p-values are computed.
