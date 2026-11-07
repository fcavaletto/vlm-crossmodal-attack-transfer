# Checkpoint 2.1 analysis protocol

Status: fixed before any new attack and before any fit of H*. Research lead: Federico Cavaletto. Label of every number a later run produces under this protocol: DEVELOPMENT ONLY.

This file is the authority for the outcome, the predictors, the clustering, the rare-event rule, the separation rule, the missing-outcome rule, the insufficient-variation rule, and the effective-sample check. It does not replace `docs/checkpoint-2.0/CALIBRATION_PROTOCOL.md`, `docs/checkpoint-2.0/CALIBRATION_REPORT.md`, or `docs/checkpoint-2.0/REVIEW_PACKET.md`. It does not retune the attack those documents froze.

Started from `checkpoint-2.0/review-packet` at `ccc5895b489ef3354c382b52e56f4fdb5c22b57e`. No H* fit. No held-out scene. No merge to main. Pull requests 20, 21, and 22 stay open.

## Frozen configuration

The attack is the primary configuration already chosen in the Checkpoint 2.0 review packet. This protocol does not change it after the calibration counts, and a later fit does not change it either.

| Piece | Locked value |
| --- | --- |
| Model | `Qwen/Qwen3-VL-2B-Instruct`, revision `89644892e4d85e24eaac8bacfd4f463576704203` |
| Weights | bfloat16, eager attention |
| Processor and answer prefix | Checkpoint 1.7 lock. Prefix `<|im_end|>\n<|im_start|>assistant\n`. Bare color continuation. |
| Color token ids | red 1151, blue 12203, green 13250, yellow 27869. Token 2518 is not the continuation. |
| Loss | Summed negative log-likelihood of the labeled color |
| Decision | Lowest summed NLL among red, blue, green, yellow |
| Attack | PGD, epsilon `1/255`, 3 steps, step length epsilon/4 |
| Initialization | The clean image. One deterministic start. No restarts. |
| Source context | `neutral` |
| Official image | Half-to-even 8-bit round trip. Clip to `[0, 1]` and project the L-infinity ball. |
| Memory lines | bfloat16 11,509,447,200 available bytes; float32 20,871,410,752; 6 GiB floor 6,442,450,944 |

`helpful_nonrevealing` is an evaluation context. It is not a second source. Epsilon stays `1/255`. The primary step count stays 3. The memory lines stay the Checkpoint 1.5 lines. The guard is not lowered.

## Sample

All 96 `development_v4` scenes, 48 counterfactual groups, both arms. The manifest is the one named in `docs/checkpoint-1.7/DEVELOPMENT_PROTOCOL.md`: `fixtures/development_v4/manifest.jsonl`, sha256 `0d9d592b6700acf610453e4f4adf639fc778b4efb24530f60deb01045440d597`. Every row has `split` `development` and `task_family` `referent_color`.

The Checkpoint 2.0 stride of 16 groups is prior information. It is not the analysis sample. No held-out scene is opened. A manifest hash mismatch stops the run before a perturbation.

The independent sampling unit is the counterfactual group. Sibling arms share a group. A scene that enters the regression contributes one row, for the ordered pair neutral to `helpful_nonrevealing`. The two arms are not averaged into one row, and they are not treated as two independent groups.

## Primary outcome

The primary outcome is conditional transfer from `neutral` to `helpful_nonrevealing` at the frozen configuration.

- A scene is source-clean-correct when the official clean neutral decision equals the labeled color.
- Source success is a source-clean-correct scene whose official perturbed neutral decision was recorded and is not the labeled color.
- The conditional-transfer denominator is the set of source successes whose official clean helpful decision equals the labeled color and whose official perturbed helpful decision was recorded.
- Conditional transfer is a scene in that denominator whose perturbed helpful decision is not the labeled color.
- A transfer failure is a scene in that denominator whose perturbed helpful decision equals the labeled color.
- A source success whose clean helpful decision is not the labeled color is excluded and counted. The exclusion is a separate count. It is not a transfer failure.
- A missing helpful decision on a source success stays missing and is counted apart from the denominator.

The regression codes the outcome as `T = 1` for conditional transfer and `T = 0` for a transfer failure. H* is the coefficient on alignment in the regression below. The primary rate and H* answer different questions. The rate is a count. H* is that coefficient, and only after the effective-sample check passes.

The endpoint is the four-color decision. A recorded decision is one of red, blue, green, or yellow. The proposal's split of strict error into invalid text and valid-wrong text is a generation endpoint. It is not an endpoint of this protocol.

## Secondary outcomes

These comparisons are reported for the same 96 scenes. They are not a search for a better attack. They are not a second regression. A larger count on one of them does not replace the frozen configuration, and H* is not moved onto a comparison because the primary check failed.

| Outcome | What is counted |
| --- | --- |
| Source success | Source-clean-correct scenes with a recorded perturbed neutral decision. The numerator is a decision other than the labeled color. Missing perturbed decisions are counted apart from the denominator. |
| Target attack success | Scenes whose clean helpful decision equals the labeled color and whose perturbed helpful decision was recorded. The numerator is a helpful decision other than the labeled color. Source success is not required. This is the calibration target-flip count. |
| FGSM at `1/255` | The positive sign step of length epsilon, from the clean image, using the clean neutral gradient. Source success and conditional transfer use the definitions above. |
| Matched random-sign at `1/255` | The Checkpoint 2.0 matched random-sign control at this epsilon, including the seed string `checkpoint-2.0-calibration|{scene_id}` in `docs/checkpoint-2.0/CALIBRATION_PROTOCOL.md`. Source success and conditional transfer use the definitions above. |
| PGD at step 7 | The official image after step 7 of the same trajectory: epsilon `1/255`, step length epsilon/4. Source success and conditional transfer use the definitions above. |

FGSM is the linear comparison. Matched random-sign is the sign comparison. Step 7 is the saturation comparison. Each comparison keeps the epsilon of the frozen attack. A new epsilon, a new step count, a new seed, or a helpful-source attack is outside this protocol.

Each secondary rate is reported as numerator, denominator, exclusions, and missing decisions. The denominator is the one in the table. A zero denominator leaves the rate unset.

## Pre-specified regression

On the eligible scenes defined in the next section, and only when the effective-sample check and the insufficient-variation checks pass:

```
T = β0 + β_A A + β_nm M_neutral + β_hm M_helpful + β_L1 ||g_neutral||_1 + error
```

| Column | Role | Definition |
| --- | --- | --- |
| `A` | Primary predictor | `A(neutral, helpful_nonrevealing)`, the stored Checkpoint 1.8 scalar `alignment_neutral_to_helpful` |
| `M_neutral` | Covariate | Clean neutral margin: best other summed NLL minus the labeled summed NLL |
| `M_helpful` | Covariate | Clean helpful margin, same definition under `helpful_nonrevealing` |
| `||g_neutral||_1` | Covariate | L1 norm of the clean neutral image gradient, the stored Checkpoint 1.8 scalar |

The parameter count is `k = 5`: intercept, alignment, and three covariates. There are four regressors besides the intercept. `referent_color` is the only task family in the manifest, so a task indicator is not a column. One budget is frozen, so a budget indicator is not a column. `A(helpful_nonrevealing, neutral)` is not a column. Group indicators are not fixed effects.

`β_A` is a change in the conditional-transfer probability per unit of alignment, with the three covariates held fixed. Alignment is not standardized. An alignment of 0 is the reference point in this coding. Moving from −1 to +1 changes the modeled probability by `2 β_A`. The margins stay in summed-NLL units. The gradient L1 stays in L1 units.

The covariates are the stored Checkpoint 1.8 clean-image scalars. A recomputed clean gradient can be compared with the stored Checkpoint 1.8 record and the mismatch reported. The stored scalar remains the covariate.

The columns are fixed here. There is no hypothesis-driven variable selection on the observed outcomes. A column is not added, dropped, transformed, or replaced after the outcomes are seen. Interactions, splines, and label indicators are not added.

This population is the conditional-transfer denominator. Source success defines that population. It is not a regressor. The fit, when it is allowed, is the source-success population in one direction. Helpful-to-neutral is not estimated.

The primary fit is ordinary least squares of this linear probability model. A logistic regression with the same columns and the same clusters is a sensitivity. It is attempted only after the pre-fit checks pass. The logistic coefficient is a change in log odds. It does not replace `β_A`.

## Eligible scenes

A scene enters the regression only when every line below holds. The effective-sample counts are counts of this set.

1. The scene is in the conditional-transfer denominator: source success, clean helpful decision equal to the label, perturbed helpful decision recorded.
2. `A` is finite and inside [−1, 1]. An undefined alignment is an exclusion. It is not coded as 0.
3. Both clean margins are finite.
4. The neutral gradient L1 is finite and strictly positive. A zero L1 is an exclusion. It is not coded as 0.

A scene that fails one of these lines is counted under that reason. A transfer failure with an undefined predictor is a predictor exclusion. It is not one of the transfer failures in the effective-sample count. The check is not inflated by rows the regression cannot use.

Scenes outside the conditional-transfer denominator stay outside. An excluded clean-helpful error stays an exclusion. A missing decision stays missing.

## Effective-sample check

This check is applied to the eligible set of the frozen configuration before any coefficient, any bootstrap, and any logistic fit. It reads the eligible-scene count, the transfer-failure count, and the group ids. It does not estimate `β_A`.

Let `n_fail` be the number of eligible scenes with `T = 0`. Let `G` be the number of distinct counterfactual groups among eligible scenes.

The effective-sample check passes only when both of these hold. A pass is required before any fit. It is not sufficient on its own: the insufficient-variation checks are required as well.

- `n_fail` is at least 40, which is ten transfer failures for each of the four regressors.
- `G` is at least 6, which is `k + 1` for `k = 5`. The cluster-score matrix has at most `G` rows, so fewer groups cannot support a five-coefficient cluster interval. The algebraic floor is the one already used by the H* instrument.

If either line fails, the result is that H* cannot be meaningfully estimated. The analysis stops. No point estimate is stored. No diagnostic coefficient is stored. No interval is stored. No logistic coefficient is stored. The floor is not lowered because the count is close, and a covariate is not removed to shrink the floor.

The same stop covers the rare-event rule: fewer than 40 transfer failures means the failure class is too small for this pre-specified model. The response is to stop. Firth penalization, exact logistic regression, a pseudo-count, a changed link, and a model with fewer covariates are not substitutes that are then called H*.

## Insufficient variation

These checks are applied to the same eligible set, still before any fit of `T`. If any one of them fails, H* cannot be meaningfully estimated, and the analysis stops under the same prohibition on a forced coefficient.

| Status | When the analysis stops |
| --- | --- |
| `flat_outcome` | `n_fail` is 0 or `n_transfer` is 0. `T` does not vary. |
| `insufficient_variation` | `n_transfer` is below 40. The transfer class is under the same ten-events-per-regressor floor. A scarce transfer class is not fit. |
| `near_flat_outcome` | The mean of `T` is below 0.05 or above 0.95. Equality with 0.05 or 0.95 stays inside the band. This is the H* instrument's near-flat band, applied here as a stop before the fit. |
| `flat_predictor` | Alignment is constant, a covariate is constant, or alignment has no residual variation after the intercept and the three covariates. The residual check uses the instrument's RSS ratio `1e-10`. |
| `singular_design` | The five-column design is not full rank, or the condition number of `X'X` exceeds `1e12`. |
| `too_few_groups` | `G` is below 6. This is the group half of the effective-sample check. |

On a sample of at most 96 scenes, `n_fail ≥ 40` and `n_transfer ≥ 40` already place the mean of `T` inside (0.05, 0.95). The near-flat line remains in force.

A failed design check does not drop the collinear column and refit. The Checkpoint 2.0 calibration already reports that the two clean margins rank together and that the neutral gradient L1 ranks opposite those margins. That description is why collinearity is anticipated. It is not a reason to remove a covariate, and it is not a fit of `T`.

## Missing outcomes

A missing decision, margin, alignment, or gradient L1 stays missing.

- A missing perturbed neutral decision keeps the scene out of the source-success numerator and out of that denominator. It is counted as missing.
- A missing perturbed helpful decision on a source success keeps the scene out of the conditional-transfer denominator. It is counted as missing. It is not filled in as a transfer, and it is not filled in as a transfer failure.
- A missing clean decision keeps the scene out of every rate that would have required that decision.
- A missing predictor keeps the scene out of the eligible set under the predictor rule above.

Empty cells are reported as unset rates with their missing counts. They are not filled with zero.

## Separation

Separation is a property of the logistic sensitivity. That sensitivity is reached only after every pre-fit check has passed.

The logistic fit uses the separation rule already stated for the H* instrument. A fit that separates or does not converge returns a null logistic interval. A pairs cluster bootstrap draw that separates or does not converge is dropped. If fewer than 95 percent of the replicates converge, the logistic interval is null. The numerical checks are the instrument's: linear predictor absolute value above 30, a working weight below `1e-12`, or a coefficient absolute value above 15.

A null logistic interval does not replace the linear interval, and it does not authorize another model. Complete separation and quasi-complete separation are not repaired by dropping a covariate, merging groups, changing epsilon, or changing the step count.

The linear probability model is the primary estimand. A zero residual vector on that fit, if a fit was allowed, is recorded as `zero_residual_fit`. A zero-width interval is the arithmetic of those residuals. It is not a picture of sampling uncertainty, and it is not support for H*.

## Group-clustered uncertainty

Uncertainty for H* is a group-clustered interval, computed only after the pre-fit checks pass.

The interval is an unrestricted wild cluster bootstrap of the linear probability model. Clusters are `counterfactual_group_id`. Fit ordinary least squares. Multiply each cluster's residual vector by one Webb six-point weight. The weights are −√1.5, −1, −√0.5, √0.5, 1, √1.5. Their mean is 0 and their mean square is 1. Refit, repeat, and take the equal-tailed percentile interval with NumPy's linear quantile. There is no normal approximation and no p-value.

Clusters are weighted in sorted order. The seed draws the weights in that order. The development seed is `20261109`. The draw count is 999. The interval is 95 percent. This seed is the development seed for this protocol. It is not a confirmatory registration seed. The Checkpoint 1.5 fabricated seed `20261107` and the Checkpoint 1.7 descriptive seed `20261108` are not reused for this fit.

The reported interval is the interval for `β_A`. Covariate intervals may be stored beside it. They are not used to change the columns.

The logistic sensitivity, when the linear fit was allowed, uses a pairs cluster bootstrap at the same seed and the same draw count. Clusters are drawn with replacement and the logistic model is refit.

Secondary rates use a separate group bootstrap. Resample the 48 counterfactual groups with replacement, keep every eligible scene that belongs to a drawn group, and recompute the rate. A drawn group contributes each of its scenes. The draw count is 4000, the quantile is NumPy's linear quantile, and the seed is `20261109`. A zero denominator leaves the rate unset and returns no interval. These intervals describe the rates. They are not `β_A`. They are computed whether or not the H* check passes. A scene-iid interval is not the uncertainty for these rates.

The directional reading of H* is `β_A > 0`. The interval is two-sided. The reading holds only when the lower endpoint is strictly greater than 0. An endpoint of 0 does not qualify. `confirmatory_supported` stays false. This sample is the development sample. An interval that lies above 0 is a development association. It is not confirmatory support, and it is not by itself a causal effect of alignment. Alignment and transfer are both consequences of the same model.

If the pre-fit checks stop the analysis, the secondary rate intervals may still be reported. They do not fill in a coefficient.

## Checkpoint 2.0 calibration as information already in hand

The calibration is prior information. It is not refit. It is not the 96-scene result. The counts below are the ones already printed in `docs/checkpoint-2.0/REVIEW_PACKET.md` for the frozen configuration and its named comparisons. The sample there is the stride of 32 scenes.

| Configuration | Source success | Conditional transfer | Transfer failures |
| --- | ---: | ---: | ---: |
| PGD, `1/255`, 3 steps | 16/32 | 13/16 | 3 |
| FGSM, `1/255` | 7/32 | 2/7 | 5 |
| Matched random-sign, `1/255` | 0/32 | 0/0 | undefined |
| PGD, `1/255`, step 7 | 28/32 | 27/28 | 1 |

Target attack success at the frozen configuration is 15/32. All 32 calibration scenes were clean-correct under both contexts, with no exclusions and no missing decisions. The review packet states that three non-transfers, in one direction, are not a sample on which to estimate `β_A`. H* was not fit on those scenes. This protocol does not fit it on them either.

Three transfer failures are below the floor of 40. That is why the floor is written before the full run. The full-run check uses the eligible set of all 96 scenes. It does not multiply the calibration count by 96/32 and call the product the result.

On the same 32 scenes the review packet reports Spearman rank correlations among the stored clean scalars: the two margins at 0.883, neutral gradient L1 with the neutral margin at −0.769 and with the helpful margin at −0.653, and alignment only weakly ranked with either. Those ranks are associations among predictors. They are not a model of transfer. They do not drop a covariate, and they do not predict attack success.

The calibration rule that called 13/16 interior, and that called a numerator within one of its denominator a ceiling, was the rule that froze this attack. It remains that rule. It is not the effective-sample floor. An interior conditional-transfer rate can still have too few transfer failures to estimate H*.

## Order of operations

1. Lock this protocol in the commit that adds it.
2. Run the frozen attack on the 96 development scenes only after that commit is on the remote.
3. Build the eligible set from recorded decisions and stored Checkpoint 1.8 scalars.
4. Apply the effective-sample check and the insufficient-variation checks.
5. If either check fails, report that H* cannot be meaningfully estimated, report the counts, the exclusions, and the missing decisions, and stop without a coefficient.
6. If every check passes, fit the linear model, the wild cluster interval, and the logistic sensitivity. Read `β_A` by the endpoint rule above. Leave `confirmatory_supported` false.

The secondary rates are reported at step 3's counts. They are not a route around step 4.

## What this protocol leaves in place

The attack stays PGD at epsilon `1/255` for 3 steps, step length epsilon/4, source `neutral`. The secondary comparisons stay FGSM at `1/255`, matched random-sign at `1/255`, and the same trajectory at step 7. Held-out scenes stay unopened. H* stays unfitted in this commit. The branch does not merge to main.
