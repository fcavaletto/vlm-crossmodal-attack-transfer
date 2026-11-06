# Statistical plan

Status: Checkpoint 1.7 proposal. This document is prospective. It leaves the Checkpoint 1.6 official scores where they are. It does not fit H*. It does not run a model. The numbers below are either recomputed from the committed development table or drawn from a fabricated design. Research lead: Federico Cavaletto.

The sampling unit is the counterfactual group. A scene shares its group with the other arm, and a scene that enters H* contributes two ordered directions. Those rows are one cluster.

## Metrics

Checkpoint 0 named six summaries. They answer different questions, and this plan keeps the names apart.

| Metric | Question | Denominator | Attack required |
| --- | --- | --- | --- |
| Clean accuracy | Share of scenes that are strict-correct on the clean image | Scenes in that context | No |
| Invalid-output rate | Share of generations the strict parser rejects | Scenes in that context | No |
| Attack success rate | Share of previously strict-correct scenes that become valid and wrong | Scenes that were strict-correct before the attack | Yes |
| Additional error | Mean rise in task error from the clean image to the attacked image | Scenes in the cell | Yes |
| Cross-context attack transfer | Attack success and additional error when the source context and the target context differ | The denominator of the rate being summarized | Yes |
| Gradient alignment `A(s, t)` | Share of the target gradient's L1 norm that lies along `sign` of the source gradient | L1 norm of the target gradient | No |

An invalid generation is a task error. It stays in the attack-success denominator and stays out of the success count. A zero denominator leaves the rate unset. Filling it with zero would turn an empty cell into a measured failure.

Correct, valid-but-wrong, and invalid are a partition of one generation. Closed-vocabulary accuracy is a different column. On the row-color table the closed-vocabulary column is the class-loss argmin. Checkpoint 1.6 kept the strict parse as the official score. This plan does the same. The class-loss argmin matches the label more often on some contexts, including 22/24 under `answer_revealing` where the strict score is 0/24. That gap is a record of the parser. It is the reason the columns stay separate.

The prospective classification score, for a protocol written before the result it will judge, is the teacher-forced conditional log-likelihood of the full candidate sequence. The reduction is the sum. The answer prefix stays fixed, and the same convention is used in every context. The four candidates are red, blue, green, and yellow. When an attack is later authorized, the attack loss uses that same classification decision. This audit does not install that score as the Checkpoint 1.6 official metric, and it does not install it because a table would look higher.

`answer_revealing` is a positive control. A high score there means the text named the answer.

H* is a coefficient, β_A. Among scenes that are clean-correct under both `neutral` and `helpful_nonrevealing`, strict error of an attacked image is regressed on sign alignment. The interval is a two-sided wild cluster bootstrap. The directional reading β_A > 0 uses that interval only when the lower endpoint is strictly greater than 0. An endpoint of 0 does not qualify. β_A is none of the six kickoff metrics.

## Why nine groups cannot establish H*

The row-color summary reports 14 scenes in 9 groups that are strict-correct under both primary contexts. Recomputing the predictions gives the same 14 scenes and the same 9 groups: `cf3-row-00`, `cf3-row-01`, `cf3-row-02`, `cf3-row-05`, `cf3-row-06`, `cf3-row-07`, `cf3-row-08`, `cf3-row-10`, and `cf3-row-11`. Five of those groups contribute both scenes. Four contribute one. Fourteen scenes are nine clusters.

Those nine groups are the clean-both count. An eligible group also needs finite, strictly positive image gradients on both primary contexts, empty ambiguity flags, non-revealing primary text, and primary strings that match inside the group. The gradient file names one scene, `dev3-row-00-a`. That scene is outside the clean-both set. The eligible-group count is unset. It is not 9, and it is not 0.

H* also needs an attack outcome and a defined alignment. No attack was run. Alignments on the nine groups were not measured. The regression has no response and no focal column on those groups.

Algebra still matters, because a later packet could be misread as soon as it clears the smallest group count. A cluster interval for `k` coefficients needs at least `k + 1` groups. The row-color task is one family, so `k = 2` (intercept and alignment) and the floor is 3 groups. The three-family design has `k = 4` and the floor is 5. Nine is above both floors, so the estimator could return an interval if the other gates passed and the outcome varied. Returning an interval is a different event from the lower endpoint lying above zero.

The published three-family precision file, which this package reads and does not refit, names 24 eligible groups as the smallest grid point on that fabricated design where every replication returned an interval and at least 90 percent of those intervals covered a known coefficient of 0.25. The same sentence says that count does not promise an interval excluding zero. Nine is below 24. Twenty-four itself is a coverage description of one design. This plan does not treat 24 groups as automatically enough.

## Clustered intervals for the development table

The intervals below redraw the 12 committed groups. The point estimate is the mean over scenes, or, where the estimand is already one number per group, the mean over groups. A drawn group contributes each of its scenes. The quantile is NumPy's linear quantile, seed `20261108`, 4000 draws. The seed is a descriptive seed for this table. It is not a confirmatory registration. H* is not fit.

Strict accuracy, cluster interval, then the scene-iid interval beside it:

| Context | Estimate | Cluster interval | Scene-iid interval |
| --- | ---: | --- | --- |
| `neutral` | 14/24 | 0.333 to 0.792 | 0.375 to 0.792 |
| `helpful_nonrevealing` | 17/24 | 0.458 to 0.917 | 0.542 to 0.875 |
| `contradictory` | 2/24 | 0.000 to 0.208 | 0.000 to 0.208 |
| `answer_revealing` | 0/24 | 0.000 to 0.000 | 0.000 to 0.000 |

Both arms strict-correct, one number per group: `neutral` 5/12, interval 0.167 to 0.667; `helpful_nonrevealing` 8/12, interval 0.417 to 0.917; `contradictory` and `answer_revealing` are 0/12, and the interval has width zero because every group in the table is zero. A zero width here is the table. It is not a precise population zero.

The group indicator "at least one scene is clean-correct under both primary contexts" is 9/12, interval 0.500 to 1.000. The scene indicator for the same event is 14/24, interval 0.333 to 0.792.

The neutral cluster interval runs from about one third to about four fifths. It covers the design-band edge 0.60 and the unsuitable line 0.50. Twelve groups do not pin the neutral rate inside the band that was written before the run. The helpful cluster interval is wider than the scene-iid interval. The neutral pair is close. Clustering is the sampling unit either way.

## Precision under group dependence

`study2_audit.precision` draws a fabricated one-family design. It does not read the development predictions. Each group gets one `Uniform(0, 1)` number. Each row has probability `0.50 + 0.25 A`, with `A` uniform on [-0.5, 0.5], two scenes per group, and both ordered directions. Sibling rows share the uniform draw, so they move together. The known alignment coefficient is 0.25. The interval is the unrestricted wild cluster bootstrap with Webb's six-point weights, the same weight set as the Checkpoint 1.5 instrument. A second interval treats each row as its own cluster. Development scene ids are refused before any coefficient is computed.

Seed `20261108`, 200 replications, 399 bootstrap draws. Every replication returned an interval. Counts are out of 200.

| Groups | Scenes | Covered 0.25 | Lower endpoint above 0 | Median cluster width | Median row width |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 9 | 18 | 175 | 42 | 1.020 | 1.043 |
| 12 | 24 | 180 | 34 | 0.947 | 0.907 |
| 24 | 48 | 179 | 64 | 0.686 | 0.646 |
| 48 | 96 | 190 | 90 | 0.512 | 0.467 |

Widths are the medians, rounded to three decimals. The command in the reproduction section reprints them.

Covering 0.25 happens on most replications. The lower endpoint lies above zero on 42, 34, 64, and 90 of the 200 replications. With 200 replications a share near 0.21 has a binomial standard error of about 0.03, and a share near 0.45 has a standard error of about 0.035. The step from 42/200 at 9 groups to 34/200 at 12 groups is inside that noise. The grid does not say that 12 groups are worse than 9. It does say that, across the whole grid, exclusion of zero stays below one half while coverage of the known coefficient stays near 0.90.

At 9 groups the row-level median width is larger than the cluster width. From 12 groups upward on this grid the cluster width is larger. Clustering is not automatically the more conservative interval at a small group count. That ordering is a property of this run.

A median width of 1.020 at 9 groups is about four times the coefficient 0.25. A median width of 0.512 at 48 groups is about twice the coefficient. An interval of width 0.51 centered on 0.25 has a lower endpoint near zero, which this plan does not read as support. Forty-eight groups are what 96 scenes become when every scene is eligible and each group has two arms. On this design, that complete sample still puts the lower endpoint above zero on 90 of 200 replications.

The coefficient 0.25 is the planning value already used in Checkpoint 1.5. It is not an estimate from these models. A smaller coefficient would clear zero less often. The published three-family file remains the source for the 24-group coverage sentence. This one-family run does not replace that sentence, and it does not promote 48 groups to a new recommended minimum.

## Multiplicity, exclusions, missing gradients, invalids, and ceilings

One confirmatory estimand is available for a later registration, and this checkpoint does not register the fit. It is β_A in

```
T = β0 + β_A A + task indicators + error
```

`T` is strict error. The reference coding and the control readings stay the ones in `docs/statistics/H_STAR_ANALYSIS.md`: noise, FGSM, the invalid outcome, the valid-wrong outcome, and the two source slices. The kickoff metrics, the four contexts, the strict-versus-class comparison, and the summed-NLL score are separate summaries. They do not become primary because a development table moved.

A row reaches that regression only through the gates that ignore the attack outcome: both primary contexts, clean-correct under both, no leakage, group-invariant primary strings, no ambiguity flag, both gradient L1 norms finite and strictly positive, and alignment finite in [-1, 1]. A missing gradient is an exclusion. It is not coded as zero. An incomplete gradient pass leaves the eligible-group count unset. The clean-both count is still reported, under its own name.

Strict error is exactly one of invalid or valid-wrong. The primary model is the strict-error model. The two pieces are refit separately. A positive association that shows up only for invalid outputs keeps the reading `association_in_invalid_outputs`. Invalid output is not attack success.

Additional error on one scene lies in {-1, 0, +1}. A scene that is already wrong cannot contribute +1. On the row-color table the neutral clean-error rate is 10/24, which is under one half, so the Checkpoint 0 floor warning does not fire. The helpful clean-error rate is 7/24. The attack rows that would carry additional error do not exist. A mean of `T` below 0.05 or above 0.95 sets `near_flat_outcome` on the H* instrument and turns the mechanical reading to non-identifiable. The design band 0.60 through 0.80 is a rule for adopting the row-color task on strict accuracy. It is a different rule from the H* reading.

The revealing context is a parser ceiling on this table: 0/24 strict-correct, 24/24 invalid, class-loss argmin matching the label on 22/24. The strict zero is the parser rejecting sentences. The class-loss column is higher. The official score stays the strict parse.

## Decision for 96 scenes

**No-go.** Ninety-six development scenes cannot support H*.

Ninety-six scenes are 48 counterfactual groups when each group has two arms. Support would require eligible groups, measured alignments, attack outcomes, one pre-specified coefficient, and a cluster interval whose lower endpoint is strictly above zero, with the control readings clear. The scene count supplies none of that.

The development clean-both share is 9/12 groups. The product of 48 and 9/12 is 36. That product is arithmetic. It is not a forecast of how many groups will be eligible, because gradients removed nothing yet and can remove more. The one measured gradient scene is outside the clean-both set.

On the fabricated design, 48 fully eligible groups still exclude zero on 90 of 200 replications at a known coefficient of 0.25. Nine groups exclude zero on 42 of 200, with a median width near 1.02. Twenty-four groups exclude zero on 64 of 200. None of those shares is a warrant for the claim.

The scenes can still be built. This decision blocks an H* claim from a 96-scene development sample. It does not block construction of the sample, a clean-accuracy table, or a count of eligible groups once gradients exist.

## What a later registration would still write down

Before any sealed split is scored, a registration would name the confirmatory seed, the replication count, the eligible-group rule, the single budget, and the primary columns. It would say what happens when the eligible-group count is below 3 for one family or below 5 for three families, and it would keep the coverage description of 24 separate from the question of excluding zero. This file does not choose that seed.

## Reproduction

From the repository root. The tests do not load a model. The module prints the audit JSON and does not write `docs/checkpoint-1.6/qwen_feasibility.json`.

```bash
python3 -m pytest -q tests/study2_audit
PYTHONPATH=src/study2_audit python3 -m study2_audit
```

Exit 0 from the module means the pinned Qwen hashes match, the reproducibility checks match, H* was not fit, and the decision is no-go. The precision block in that JSON is the fabricated grid above.

## Git

- Branch: `checkpoint-1.7/study-audit`
- Parent commit: `004783b5604c34aa066befbcd5356bfd7bd6e7b2`
- Implementation commit: `336fe962f0bbb213703aa527a5c9eaf27e6327ad`
- Pull request: https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/pull/13
- Head SHA: the commit that records this pull request URL. `git rev-parse HEAD` on the branch prints it.
