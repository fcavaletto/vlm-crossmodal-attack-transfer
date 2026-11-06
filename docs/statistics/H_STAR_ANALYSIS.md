# H* estimator

Status: Checkpoint 1.5 instrument. This document defines the confirmatory estimator for H*. It is not a result about a vision-language model. Every fitted number below is arithmetic on fabricated rows, or a coverage count on a fabricated design. The 24 development scenes are not fit. No attack is run.

The hypothesis and the symbols are the ones in [`../research/RESEARCH_PROPOSAL.md`](../research/RESEARCH_PROPOSAL.md). The Checkpoint 0 tables in [`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md) still score clean accuracy, attack success, and the scalar `A(s, t)`. This document adds the regression those tables did not fit.

## What H* says

Among scenes the model already answers correctly under both `neutral` and `helpful_nonrevealing`, a higher sign alignment `A(s, t)` goes with a higher probability that an image crafted under one of those contexts is strictly wrong under the other.

The coefficient is called β_A. The hypothesis is directional: β_A > 0. The reported interval is two-sided. A positive coefficient would be an association. Alignment and the transfer indicator are both consequences of the same model, so the coefficient would not by itself be a causal effect of alignment.

This package does not compute `A(s, t)` from an image. The caller supplies the scalar. The Checkpoint 0 function `sign_alignment` remains the definition of that scalar: the share of the target gradient's L1 norm that lies along the sign pattern of the source gradient. It is undefined when the target L1 norm is 0. Those rows are excluded. They are not coded as 0.

## The row

One row is one ordered pair on one scene. The ordered pairs are (`neutral`, `helpful_nonrevealing`) and (`helpful_nonrevealing`, `neutral`). Both enter the regression. They are not averaged. They share a `counterfactual_group_id` with each other and with sibling scenes, so they are not independent observations.

The outcome fields are booleans the caller has already decided:

| Field | Role |
| --- | --- |
| `strict_error` | Primary outcome. 1 when the attacked image is strictly wrong under the target context. |
| `invalid_target` | The target output is unparseable. |
| `valid_wrong_target` | The target output parses and is not the ground truth. |
| `source_strict_error` | Strict error under the source context. Used only to split the sample. It is not a regressor. |

A strict error is exactly one of invalid or valid-wrong. A row that is not a strict error is neither. The schema rejects any other combination.

`method` is `pgd`, `fgsm`, or `noise`. Those are labels on rows the caller fabricated. This package does not build a perturbation. `epsilon` and `step_budget` are cell labels. The primary fit allows one pair of them. Two pairs are a `multiple_budgets` refusal, not a second set of indicators.

`clean_confidence`, when supplied, is a finite probability in (0, 1]. The intended later meaning is the clean-image probability of the true answer under the target context. This package does not compute it. `gradient_target_l1` is the magnitude used in the comparison below.

`data_origin` must be the string `fabricated`. Any other origin is refused. Scene ids that start with `dev-`, the 24 committed development scene ids, and the 12 committed development group ids are refused. Refusing them is a guard. It is not a fit, and it does not recompute the pilot's inclusion count.

## Who enters the regression

A row is included only if every gate below passes. The gates do not read `strict_error`, `invalid_target`, `valid_wrong_target`, or `source_strict_error`. Changing those fields does not change the included set.

1. The two contexts are `neutral` and `helpful_nonrevealing`, in either order.
2. The scene is clean-correct under both.
3. Neither context is marked as revealing the answer, and neither string is marked as containing it.
4. The caller marks the primary strings as invariant inside the counterfactual group.
5. The scene is not flagged ambiguous.
6. Both gradient L1 norms are finite and strictly positive.
7. `alignment_a` is a finite number in [-1, 1]. Null alignment is an exclusion, not a zero.

The scene generator owns the real invariance and leakage checks. This module only reads the flags.

## The primary fit

On the included `pgd` rows,

```
T = β0 + β_A A + β_counting I(counting) + β_spatial I(spatial) + error
```

`T` is `strict_error`. The reference task family is `attribute`, because that is the first name in the family order already used by the scene generator. It is not chosen from a development accuracy. Swapping the reference family changes the task coefficients and does not change β_A when the design is otherwise the same. The worked packet checks that. If a family is absent, its indicator is absent, and the parameter count drops. β_A is a change in probability per unit of alignment. Alignment is not standardized. An alignment of 0 is the reference point in this coding. Moving from -1 to +1 changes the modeled probability by 2 β_A.

The error is clustered on `counterfactual_group_id`. Sibling scenes and both ordered directions share that id.

The interval is an unrestricted wild cluster bootstrap. Fit ordinary least squares. Multiply each cluster's residual vector by one Webb six-point weight. The weights are -√1.5, -1, -√0.5, √0.5, 1, √1.5. Their mean is 0 and their mean square is 1. Refit, repeat, and take the equal-tailed percentile interval with NumPy's linear quantile. There is no normal approximation and no p-value. Clusters are weighted in sorted order, and the seed draws the weights in that order, so a repeated seed rebuilds the endpoints on NumPy 1.26.4.

The bootstrap is undefined as an interval, and none is returned, in these cases:

| Status | When |
| --- | --- |
| `no_included_rows` | Nothing passed the gates. |
| `multiple_budgets` | The included rows use more than one epsilon and step pair. |
| `flat_outcome` | `T` does not vary. |
| `flat_predictor` | Alignment does not vary, or it has no residual variation after the intercept and the task indicators. |
| `singular_design` | The design matrix is not full rank for another reason. |
| `too_few_groups` | The number of groups `G` is below `k + 1`, where `k` is the number of coefficients. |

`G >= k + 1` is the algebraic rule. The cluster-score matrix has at most `G` rows, so fewer groups cannot support a `k`-parameter cluster interval. For the three-family design, `k = 4` (intercept, alignment, two task indicators) and the minimum is 5 groups. A point estimate may still be computed and stored as `diagnostic_point_estimate`. It is labeled `evidence: false`. It is not an interval and it is not support for H*.

A mean of `T` below 0.05 or above 0.95 sets `near_flat_outcome`. The linear interval can still be returned, and `identifiable` is false. The mechanical reading is `non_identifiable`. This band is an estimator diagnostic. It is not the development budget window of 0.20 to 0.80 in the proposal, which is a rule for choosing a budget before any confirmatory fit.

If every residual is numerically zero, every bootstrap replicate equals the point estimate and the interval has width zero. `zero_residual_fit` records that. The worked packet is built that way so the arithmetic can be checked by hand. A zero-width interval is not a picture of sampling uncertainty.

## Logistic sensitivity

The same columns and the same clusters are fit with a logit link. The coefficient is a change in log odds, not a change in probability. The interval is a pairs cluster bootstrap: clusters are drawn with replacement and the logistic model is refit. A replicate that separates or does not converge is dropped. If fewer than 95 percent of the replicates converge, the logistic interval is null.

Separation, a flat outcome, or a failure to converge returns a null logistic interval. That null does not replace the linear interval. The proposal already says the logistic coefficient is unstable when `T` barely varies. The worked packet separates on purpose, and its logistic status is `separation`.

## Invalid outputs and valid-wrong outputs

The primary model is refit twice, with `invalid_target` and with `valid_wrong_target` as the outcome. Same alignment column, same task indicators, same clusters.

`invalid_only_pattern` is true when the invalid-outcome interval lies entirely above 0 and the valid-wrong interval does not. A missing valid-wrong interval counts as "does not." The label means a positive association in that packet is carried by unparseable outputs. It is a property of the rows. It is not a measurement of a model.

## Alignment, clean confidence, and gradient magnitude

Three linear models are fit separately. Each uses the task indicators and the cluster bootstrap. The focal column is one of:

- `alignment_a`
- `clean_confidence`
- `gradient_magnitude`, which is `gradient_target_l1`

The coefficients are in different units. A larger number is not a better predictor. None of the alternatives is added to the primary model. The proposal keeps magnitude out of the primary regression. This comparison is pre-specified so that question has a place to be answered later without changing the primary columns after a fit.

Pearson correlations of alignment with each alternative are reported on the included rows. A correlation at or above 0.95 in absolute value is listed under `high_correlation`. A constant alternative has a null correlation and a `flat_predictor` status. A joint model with all three predictors is stored as `joint_diagnostic` and marked `not_a_primary_estimand`. If confidence is an exact shift and scale of alignment, that joint design is singular. The worked comparison test builds that case.

## Source slices and the two controls

The included `pgd` rows are split by `source_strict_error`. Each slice is fit on its own when it has rows. The signs of β_A are compared only when both slices returned an interval. Opposite nonzero signs set `sign_disagreement`. The pooled coefficient is then not one mechanism. A slice that was refused does not contribute a sign.

`fgsm` and `noise` rows, when they pass the same inclusion gates, are fit with the same linear cluster interval. The mechanical reading uses those intervals. It does not declare H* supported. On fabricated rows `confirmatory_supported` is always false, with reason `fabricated_data_cannot_support_h_star`.

The reading labels, in order, are:

| Condition | Label |
| --- | --- |
| The primary fit is not identifiable, including too few groups, a flat column, a near-flat outcome, or more than one budget | `non_identifiable` |
| FGSM or noise did not return an identifiable interval | `controls_not_supplied` |
| The noise interval lies entirely above 0 | `alignment_confounded_with_difficulty` |
| The two source slices disagree in sign | `pooled_coefficient_is_not_one_mechanism` |
| The invalid-only pattern is set | `association_in_invalid_outputs` |
| The FGSM interval lies entirely above 0 and the PGD interval covers 0 | `fgsm_above_pgd_covers_zero` |
| The PGD interval lies entirely above 0 | `interval_above_zero` |
| The PGD interval covers 0 | `interval_covers_zero` |
| The PGD interval lies entirely below 0 | `interval_below_zero` |

"Entirely above 0" means the lower endpoint is strictly greater than 0. An endpoint of 0 does not qualify. These labels describe intervals. A later confirmatory packet would still have to be a real one. This module will not set the supported flag on fabricated rows.

## Worked packet

`src/analysis/analysis/fixtures/hstar/packet.jsonl` has 46 fabricated rows. Thirty-six are the included block: 6 groups, two of each task family, one scene per group, both ordered directions, and the three method labels. Ten further rows fail one inclusion gate each. The zero-gradient row also has null alignment, so it carries two exclusion reasons.

On the included `pgd` rows the outcome is the exact line

```
strict_error = 1/2 + (1/2) A
```

with `A` equal to -1 or +1. The task coefficients are 0. Residuals are 0, so the wild-bootstrap interval is the point `[1/2, 1/2]`. The noise rows flip the outcome, and their alignment coefficient is -1/2. The logistic fit separates. The mechanical reading is `interval_above_zero`. `confirmatory_supported` is false. `python -m analysis.verify_hstar` rebuilds `hstar_summary.json` from the packet and compares it. Floats may differ by at most `1e-9`.

The epsilon label on these rows is 0.01 and the step label is 2. They keep the schema able to refuse a mixed budget. They are not a chosen attack budget.

## Precision under group dependence

`python -m analysis.hstar_simulation` draws a different fabricated design. It does not read the worked packet and it does not read development scenes.

Each counterfactual group gets one `Uniform(0, 1)` draw. Each row has a probability

```
0.50 + 0.25 A + task effect
```

with `A` uniform on [-0.5, 0.5], task effects 0, -0.04, and +0.04 for attribute, counting, and spatial, and two scenes per group. Every such probability lies in [0.335, 0.665]. The row is an error when the group's uniform draw falls below that probability. Sibling scenes and both ordered directions share the uniform draw, so their outcomes move together. The known alignment coefficient is 0.25. The cluster interval and a row-level wild bootstrap, which pretends each row is its own cluster, are both applied to the same draw.

The committed run is `src/analysis/analysis/fixtures/hstar/precision_simulation.json`. Seed 20261107, 200 replications, 399 bootstrap draws, grid 6, 12, 24, 48, 96 groups. Coverage counts for the known coefficient 0.25:

| Groups | Cluster interval | Row-level interval | Cluster median width | Row median width |
| --- | ---: | ---: | ---: | ---: |
| 6 | 143/197 | 161/197 | 0.888 | 1.026 |
| 12 | 168/200 | 174/200 | 0.793 | 0.848 |
| 24 | 183/200 | 178/200 | 0.668 | 0.640 |
| 48 | 190/200 | 184/200 | 0.500 | 0.461 |
| 96 | 186/200 | 183/200 | 0.363 | 0.335 |

Widths are rounded to three decimals from that file. At 6 groups the cluster fit also recorded 2 near-flat outcomes and 1 flat outcome, which is why the denominator is 197 rather than 200.

At 6 and 12 groups the cluster interval's estimated coverage is below the row-level coverage. Clustering is not automatically conservative when the number of groups is small. From 24 groups upward on this grid, the cluster median width is larger than the row median width, and the cluster coverage count is higher. Those orderings are properties of this run. They are not a theorem about every dependence structure.

With 200 replications, a coverage estimate of 183/200 = 0.915 has a binomial standard error of about 0.020. The step from 168/200 at 12 groups to 183/200 at 24 groups is one grid point.

## How many groups, without a promise

Two different numbers are easy to mix up.

The code refuses an interval when `G < k + 1`. For the three-family primary design that floor is 5. A cited sentence in `docs/checkpoint-1/PILOT_REPORT.md` says three counterfactual groups were clean-correct under both primary contexts, and that no coefficient was fit. This module does not recompute that count. Three is below 5. The estimator would not return a three-family interval for a packet of that size.

The planning description is separate, and it does not promise significance. The pre-specified bar asks for a returned cluster interval in every replication and coverage of the known coefficient of at least 90 percent, on this fabricated grid only. The smallest such grid point in the committed run is 24 groups. The generator's sentence is:

> Under this fabricated design, the smallest grid point at which every replication returned a cluster interval and at least 90% of those intervals covered the known coefficient is 24 eligible groups. That count does not promise that an interval will exclude zero. The algebraic minimum for returning any interval on the three-family design is 5 groups. A cited Checkpoint 1 inclusion count of 3 groups is below that minimum. This is not a measurement of a vision-language model.

Meeting 24 under a different dependence structure is not guaranteed by this file. Failing to meet it does not say the coefficient is zero. The number is not a power calculation for SmolVLM2, and it is not a sample size at which the interval is expected to exclude zero.

Seed 20261107 rebuilds this file. It is not the confirmatory seed. The Checkpoint 0 toy seed 20261106 is not reused for that purpose either. A confirmatory seed belongs to a later registration, before any sealed split is scored.

## What this checkpoint does not estimate

- No coefficient from the 24 development scenes, their alignments, their losses, or their generations.
- No held-out or sealed scene.
- No attack success rate, because no attack was run.
- No statement that H* is supported. The supported flag in the fit object is false.
- No p-value.

## How to run

From a fresh clone, at the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e 'src/analysis[dev]'
python -m pytest -q tests/analysis
python -m analysis.verify_hstar
python -m analysis.verify_toy
```

`verify_hstar` exits 0 when the worked packet rebuilds `hstar_summary.json`. It does not need a GPU, a model, or a download. `--write` regenerates that summary and is not part of the check.

The precision file is rebuilt by the same function the tests call:

```bash
python -m analysis.hstar_simulation --groups 6 12 24 48 96 --replications 200 --boot 399 --seed 20261107
```

That command prints JSON. It does not rewrite the committed file. The printed document sets `not_an_empirical_finding` to true.

Tested interpreter for this checkpoint: Python 3.12.3. The pin the tests import is `numpy==1.26.4`. A different quantile implementation can change bootstrap endpoints. The worked packet's endpoints do not use residual noise, so they are the OLS point on any build that solves the same linear system.
