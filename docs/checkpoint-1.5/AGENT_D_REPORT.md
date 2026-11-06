# Agent and responsibility

Agent D, confirmatory statistics, Checkpoint 1.5 of `vlm-crossmodal-attack-transfer`. Research lead: Federico Cavaletto.

This workstream owns the H* estimator, its diagnostics, its fabricated-data simulations, `docs/statistics/H_STAR_ANALYSIS.md`, and this report. It does not fit the 24 development scenes, does not run a model or an attack, and does not edit `docs/checkpoint-1/**`, `PROTOCOL_CHANGELOG.md`, or `REVIEW_PACKET.md`.

# Executive summary

The H* estimator is implemented for fabricated rows. It fits a linear probability of strict-error transfer on sign alignment, with task-family indicators, and it keeps both ordered directions between `neutral` and `helpful_nonrevealing`. The interval is a wild cluster bootstrap on `counterfactual_group_id`. A logistic fit, an invalid-versus-valid-wrong split, and separate fits for clean confidence and target-gradient magnitude sit beside that primary model. The estimator returns no interval for too few groups, a flat alignment column, or a constant outcome, and it marks a near-constant outcome non-identifiable.

Nothing here is a coefficient for SmolVLM2. Development scene ids are refused. The worked packet is the exact line strict error = 1/2 + (1/2) alignment, the interval collapses to that point, and the supported flag stays false. A separate simulation shares one uniform draw inside each counterfactual group. On that run the three-family design needs 5 groups before an interval is returned, and 24 was the smallest grid point whose estimated coverage of the known coefficient met a pre-specified 90 percent description. That count does not promise an interval that excludes zero. The pilot report's count of 3 eligible groups is cited and was not refit. It sits below the algebraic minimum.

# Work completed

- Added `analysis.hstar`, the inclusion gates, the linear cluster interval, the logistic sensitivity, the outcome split, the predictor comparison, the source-slice sign check, and the mechanical reading.
- Added a worked packet of 46 fabricated rows and `python -m analysis.verify_hstar`, which rebuilds `hstar_summary.json`.
- Added `analysis.hstar_simulation`, a precision design with one shared uniform draw per counterfactual group, and the committed run in `precision_simulation.json`.
- The estimator refuses `data_origin` other than `fabricated`, scene ids with the `dev-` prefix, the 24 committed development scene ids, and the 12 committed development group ids.
- Tests cover the hand-solved slopes, exclusion that ignores the outcome, too few groups, flat columns, a near-flat outcome, mixed budgets, the invalid-only label, collinear confidence, source-slice signs, and the planning rule that can abstain.
- Checkpoint 0 tests still pass. `verify_toy` still rebuilds the toy summary. Those files were not rewritten.

# Key findings

Findings here are properties of the estimator and of fabricated rows. They are not findings about a vision-language model.

- On the worked `pgd` rows, β_A is 1/2, the intercept is 1/2, and both task coefficients are 0. The wild-bootstrap interval is the point [1/2, 1/2] because the residuals are 0. The noise coefficient is -1/2. The logistic fit separates. The mechanical reading is `interval_above_zero`. `confirmatory_supported` is false.
- The three-family design has 4 coefficients, so the estimator returns an interval only when there are at least 5 groups. A 3-group, 3-family packet is `too_few_groups` and the interval is null.
- In the committed precision run, cluster coverage of the known coefficient 0.25 was 143/197 at 6 groups, 168/200 at 12, 183/200 at 24, 190/200 at 48, and 186/200 at 96. At 6 groups the cluster coverage was lower than the row-level coverage (161/197). At 6 and 12 groups the cluster estimate was below 90 percent. The planning rule names 24 as the smallest grid point that met the bar on this run.
- At 6 groups, 2 replications were near-flat and 1 was a flat outcome, so those three did not return an identifiable interval.
- The pilot report's inclusion count of 3 groups is below both the algebraic minimum of 5 and the planning description of 24. No coefficient was fit to those scenes.

# Evidence

Fabricated inputs and the rebuilt summary:

- `src/analysis/analysis/fixtures/hstar/packet.jsonl` (46 rows)
- `src/analysis/analysis/fixtures/hstar/hstar_summary.json`
- `src/analysis/analysis/fixtures/hstar/precision_simulation.json`

`H_STAR_ANALYSIS.md` records the worked line and the coverage table. The counts in Key findings are the numerators behind that table.

No model weights were loaded. No attack was optimized. No file under `docs/checkpoint-1/` was written. The development manifest was not scored. The run is CPU arithmetic. It is not an MPS measurement.

Commands, from the repository root, in a virtual environment outside the tree (`/tmp/hstar-venv`). Interpreter: Python 3.12.3. Imported pins: `numpy==1.26.4`, `pytest==8.3.5`.

```bash
python3 -m venv /tmp/hstar-venv
source /tmp/hstar-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e 'src/analysis[dev]'
python -m pytest -q tests/analysis
python -m analysis.verify_hstar
python -m analysis.verify_toy
```

`pytest` stdout:

```text
..............................................................           [100%]
62 passed in 2.55s
```

`verify_hstar` stdout, exit status 0:

```json
{
  "ok": true,
  "n_mismatches": 0,
  "mismatches": [],
  "numpy_version": "1.26.4",
  "summary_path": "/agent/repos/vlm-crossmodal-attack-transfer/src/analysis/analysis/fixtures/hstar/hstar_summary.json"
}
```

`verify_toy` stdout, exit status 0:

```json
{
  "ok": true,
  "n_mismatches": 0,
  "mismatches": [],
  "numpy_version": "1.26.4",
  "summary_path": "/agent/repos/vlm-crossmodal-attack-transfer/src/analysis/analysis/fixtures/toy/toy_summary.json"
}
```

# Design decisions and rationale

The long form is `docs/statistics/H_STAR_ANALYSIS.md`. Choices a later confirmatory fit would inherit:

- Primary interval: unrestricted wild cluster bootstrap, Webb six-point weights, equal-tailed NumPy linear quantile, cluster id `counterfactual_group_id`. No Wald interval and no p-value. This matches the Checkpoint 0 refusal of a normal approximation and uses a small-cluster method that does not drop groups.
- An interval is returned only when `G >= k + 1`, the design has full rank, alignment has residual variation, and the outcome is not constant. For three task families that floor is 5 groups.
- A mean outcome outside [0.05, 0.95] is `near_flat_outcome` and the reading is `non_identifiable`. That band is not the proposal's 0.20–0.80 budget window.
- Reference family `attribute`, the first name in the existing family order. β_A on the worked packet is unchanged if `counting` is the reference instead.
- One budget label per primary fit. Mixed epsilon and step pairs are refused rather than given indicators.
- Logistic regression is a sensitivity. Separation yields a null logistic interval and does not overwrite the linear fit.
- Invalid and valid-wrong are separate outcomes. An association that sits only on invalid outputs is labeled and is not read as support.
- Clean confidence and target-gradient L1 are compared in their own models. A joint model is marked not primary.
- Source strict error splits the sample. It is not a covariate. Opposite signs are compared only when both slices returned an interval.
- `confirmatory_supported` is structurally false on fabricated rows. Development scene ids and development group ids cannot be fit.
- The planning bar is a returned interval in every replication and at least 90 percent coverage of the known coefficient, on the committed fabricated grid. The smallest such point in that run is 24 groups. The sentence in the precision file says that count does not promise an interval excluding zero.
- Seeds 20261106 and 20261107 stay with the toy summary and this fabricated check. Neither is proposed as the confirmatory seed.

# Known limitations

- The worked interval has width zero because the line fits exactly. It checks arithmetic. It does not check coverage. Coverage is the precision file.
- At 6 groups in that file, the cluster interval covered the known coefficient less often than the row-level interval (143/197 against 161/197). With very few groups, the wild bootstrap was not the more conservative of the two.
- A coverage estimate of 183/200 has a binomial standard error of about 0.020. Twenty-four is one grid point on this design, not a sharp threshold, and not a sample size for SmolVLM2.
- The shared-uniform dependence is one fabricated mechanism. A different dependence structure would move the coverage counts.
- The linear probability model can produce fitted values outside [0, 1]. The logistic sensitivity is the check against that, and it is null when the outcomes separate.
- Percentile intervals are not BCa intervals. No multiplicity adjustment is applied. The primary coefficient is the only one the reading uses.
- Clean confidence is whatever finite probability the caller stored. This package does not estimate a likelihood.
- The denylist is the committed development ids. A renamed copy of those scenes would not be recognized. The scientific rule is still that those scenes are not fit.
- This workstream did not check answer leakage, scene ambiguity, or gradient correctness. Those belong to the scene generator and the model interface.
- No MPS run was attempted. Nothing here is an Apple-silicon timing or memory figure.

# Blockers

Blocking for a confirmatory fit: the lead has not yet accepted the interval procedure, and the pilot report's eligible-group count of 3 is below the algebraic minimum of 5. This checkpoint does not loosen the inclusion rules to make that count fittable.

Blocking for review of the estimator itself: none. The tests and `verify_hstar` do not need a model.

Non-blocking:

- A confirmatory seed, replicate count, and the eligible-group rule still have to be registered before any sealed split is scored.
- FGSM and noise rows have to be present in a later packet or the reading stays `controls_not_supplied`. This checkpoint does not create those rows from a model.

# Decisions requested from the research lead

1. Primary interval. Recommendation: the unrestricted wild cluster bootstrap with Webb six-point weights and an equal-tailed percentile interval, clustered on `counterfactual_group_id`. Tradeoff: a Wald cluster interval is shorter to compute and is a poor match to very few groups. The wild bootstrap can collapse to a point when residuals are exactly zero, which the worked packet does on purpose.

2. When to return an interval. Recommendation: only when `G >= k + 1`, and treat the committed fabricated run as a planning description in which 24 was the smallest grid point at or above 90 percent estimated coverage. Tradeoff: 5 is the algebraic floor for the three-family design and is not a coverage guarantee. Twenty-four does not promise an interval excluding zero, and it will move if the dependence structure changes. The code enforces 5. It does not enforce 24.

3. Near-flat outcomes. Recommendation: a mean outside [0.05, 0.95] is non-identifiable even if a linear interval exists. Keep this distinct from the 0.20–0.80 budget window. Tradeoff: a mean of 0.94 still has a little variation, and the diagnostic withholds a confirmatory reading anyway.

4. Fabricated and development rows. Recommendation: keep `confirmatory_supported` false on fabricated rows, and keep refusing the committed development ids. Tradeoff: a reviewer cannot ask this command to "just try" the 24 scenes. That refusal is the point of the checkpoint.

5. Source slice. Recommendation: stratify on source strict error, and compare signs only when both slices returned an interval. Tradeoff: strict error counts an invalid source output as a source error. A valid-wrong-only slice would be a different population.

6. Budgets. Recommendation: refuse a primary fit that mixes epsilon and step labels. Do not add budget indicators in this estimator. Tradeoff: a packet that intentionally pools two frozen budgets cannot be fit until it is split.

7. Confirmatory seed. Recommendation: register a new seed and replicate count before any sealed split. Do not reuse 20261106 or 20261107.

8. Eligible groups before a later confirmatory fit. Recommendation: do not fit H* on a three-family packet with fewer than 5 groups, and do not describe a fit with fewer than 24 groups as having met this run's coverage description. Tradeoff: 24 is a property of the shared-uniform design. It is not a power calculation. Using it as a hard inclusion rule would change the population if the real dependence is milder or stronger. The lead can accept it as a description without promoting it to an inclusion gate.

# Reproduction instructions

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

`verify_hstar` exits 0 when the fixtures rebuild `src/analysis/analysis/fixtures/hstar/hstar_summary.json`. `--write` regenerates that summary and is not part of the check.

Optional rebuild of the printed precision document. It does not rewrite the committed file:

```bash
python -m analysis.hstar_simulation --groups 6 12 24 48 96 --replications 200 --boot 399 --seed 20261107
```

The Evidence section records the virtualenv path used so that `.venv` was not committed.

# Git information

- Branch: `checkpoint-1.5/h-star-estimator`
- Base of this branch: `90c6ffe83d61514da217a20f8dfc2f202afbbe69` (`checkpoint-1/dev-pilot`)
- The pull request targets `main`. The diff against `main` contains Checkpoint 1 because this branch starts at that commit. Checkpoint 1 files are not edited here.
- Estimator, fixtures, and the first version of this report: `9b8fbbbff69c61396daedc5026a4a9f39d63e8fd`
- Pull request: https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/pull/9
- Not merged.
- The commit after `9b8fbbb` records this pull-request URL.
