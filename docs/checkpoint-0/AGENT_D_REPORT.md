# Agent and responsibility

Agent D, statistics and reproducibility, Checkpoint 0 of `vlm-crossmodal-attack-transfer`. Research lead: Federico Cavaletto.

This workstream owns `src/analysis/**`, `tests/analysis/**`, `docs/statistics/ANALYSIS_PLAN.md`, and this report. It defines schemas, metrics, scene-level intervals, a fabricated sample-size sketch, and a command that rebuilds a toy summary from committed fixtures. It does not run models or attacks, does not read a held-out split, and does not start Checkpoint 1.

# Executive summary

Agent D owns Checkpoint 0 statistics and reproducibility for this repository. The package defines the JSONL tables for predictions, losses, gradient summaries, and attack rows. It computes clean accuracy, the invalid-output rate, attack success rate, additional error, cross-context attack transfer, and gradient alignment A(s, t). An invalid generation stays out of the success count. A rate with a zero denominator is null. Paired comparisons use a scene-level percentile bootstrap, so contexts that share a scene stay together. A separate routine records how interval width changes for fabricated scene differences. No model was loaded and no attack was run. The committed summary is arithmetic on hand-written toy rows. Role labels are proposals. The primary estimand is not locked. `python -m analysis.verify_toy` rebuilds that summary from the compact fixtures and checks it.

# Work completed

- Added an installable project at `src/analysis` (import package `analysis`) with pinned `numpy==1.26.4` and dev pin `pytest==8.3.5`. PyTorch is not a dependency.
- Defined the JSONL contracts for predictions, losses, gradient summaries, and attacks, plus a toy-only gradient-vector audit file.
- Implemented clean accuracy, the invalid-output rate, attack success rate, additional error, cross-context transfer, conditional transfer, and `A(s, t) = dot(g_t, sign(g_s)) / ||g_t||_1`.
- Implemented scene-level percentile bootstrap, paired differences, and an optional cluster resample for a future counterfactual group.
- Implemented a precision simulation on an i.i.d. three-point distribution. The simulation does not read fixtures and is not a sample-size result.
- Committed hand-written toy fixtures and `toy_summary.json`. `python -m analysis.verify_toy` rebuilds the summary and compares it.
- Wrote `docs/statistics/ANALYSIS_PLAN.md` with beginner-level definitions, the toy arithmetic, proposed roles, and the open choices.
- Tests cover schema rejection, zero denominators, the invalid-output rule, alignment against an independent oracle, scene grouping, the simulation's ordering property, and the verification command.

# Key findings

Findings here are properties of the definitions and of the hand-written fixtures. They are not findings about a vision-language model.

- On the committed toy rows, strict clean accuracy is 5/6, 4/6, 2/6, and 5/6 for neutral, helpful non-revealing, contradictory, and answer-revealing. The contradictory block has an invalid output and a clean-error floor warning. The revealing block is tagged as a positive control.
- On the toy neutral-source PGD cell scored on neutral text, attack success is 3/5 and additional error is 4/6. Scene `toy-05` is correct before, invalid after: additional error +1, and not an attack success. The noise cell has a null attack success rate because its only scene was already wrong.
- The unlocked illustration contrast, contradictory additional error minus neutral additional error on that PGD label, equals -4/6 on these rows. The scene bootstrap interval in the summary is the interval produced by seed 20261106 and 2000 replicates. It describes those six numbers.
- The run in Evidence passed 41 tests. `verify_toy` reported `"ok": true` and zero mismatches.
- Design labels (proposed primary, proposed secondary, exploratory, positive control) are proposals. The primary estimand is not locked.

# Evidence

Toy fixtures, all scene ids `toy-01` through `toy-06`:

- `src/analysis/analysis/fixtures/toy/predictions.jsonl` (24 rows)
- `src/analysis/analysis/fixtures/toy/losses.jsonl` (24 rows)
- `src/analysis/analysis/fixtures/toy/gradient_summaries.jsonl` (12 rows)
- `src/analysis/analysis/fixtures/toy/gradient_vectors.jsonl` (12 rows)
- `src/analysis/analysis/fixtures/toy/attacks.jsonl` (49 rows)
- `src/analysis/analysis/fixtures/toy/toy_summary.json`

Worked fractions checked by `tests/analysis/test_metrics.py` against those files: clean counts above; attack success 3/5; additional error 4/6; paired differences 0, -1, 0, -1, -1, -1; conditional transfer 1/2 with 4 source successes and 2 exclusions; `A` for `toy-01` equal to 0.2; `toy-04` alignment null.

No model weights were loaded. No attack was optimized. No held-out manifest was opened. The run is CPU arithmetic in the Checkpoint 0 cloud environment. It is not an MPS measurement and it is not a timing benchmark.

This VM did not have the `venv` standard-library module until `python3.12-venv` was installed. After that, the commands below were run from the repository root. The virtual environment was `/tmp/c0-analysis-venv`, outside the tree, so it was not committed. Interpreter: Python 3.12.3. Resolver pins imported by the tests: `numpy==1.26.4`, `pytest==8.3.5`, plus transitive `iniconfig==2.3.1`, `packaging==26.3`, and `pluggy==1.7.0`.

```bash
python3 -m venv /tmp/c0-analysis-venv
source /tmp/c0-analysis-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e 'src/analysis[dev]'
python -m pytest -q tests/analysis
python -m analysis.verify_toy
python -m analysis.simulation --n 12 24 --replications 10 --boot 50 --seed 20261106
```

`pytest` stdout:

```text
.........................................                                [100%]
41 passed in 0.34s
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

The simulation command exited 0 and wrote a JSON document with `data_origin` `fabricated_dgp` and `not_an_empirical_finding` true. Widths from that printout are not study results and are not copied here.

# Design decisions and rationale

The analysis plan is the long form. The choices that change what a later study can claim are listed in the next decision section. Implemented behavior, still reversible by the lead:

- Strict clean accuracy: invalid output is not correct. The invalid-output rate is a separate ratio with the same denominator.
- Attack success counts only a valid wrong answer. Invalid attacked output stays in the denominator. Additional error still treats invalid output as task error, so the two numbers can diverge. Scene `toy-05` is the toy example of that split.
- `correct_before` is checked against the clean prediction under the target context.
- `A(s, t)` uses an explicit sign that maps zeros, including negative zero, to 0. A zero target L1 norm returns null. The mean skips nulls and counts them.
- Bootstrap unit is the scene. Paired contrasts subtract inside the scene before the resample. Positive-control rows are tagged and are omitted from the transfer helper. They remain in the full cell list.
- Multi-context rows use `pgd_multi` with source `multi`. That pair is a small extension of the scalar source field so RQ3 has a place to log rows. It is marked as a proposal.
- The toy verification seed is 20261106 with 2000 replicates and NumPy's linear quantile. It is recorded so the file rebuilds. It is not proposed as the confirmatory seed.
- Study 1 contributed the verification pattern (compact rows, published aggregate, rebuild command) and the reasons for scene resampling, strict accuracy, and additional error. It did not contribute its estimates, its context names, its seed, or its attack budgets. Attack success follows the checkpoint rule that invalid generation is not a success.

# Known limitations

- There are no model predictions, gradients, or attacks in this branch. Every signed number in the summary was typed or computed from typed rows.
- The primary contrast in the summary is an illustration. Quoting its interval as a context effect would misread the file. The file says `locked: false` and carries an explicit fabricated-data reading.
- The analysis schema has no counterfactual group id, no perturbation id, and no device field. Scene-level intervals can be too narrow if sibling counterfactuals are dependent. Transfer cells are labels, not checked pixel hashes.
- Percentile intervals for a ratio omit zero-denominator replicates. With a rare clean-correct scene, many replicates can be undefined. The count is reported. BCa and Wald intervals are not implemented. No multiplicity adjustment and no p-value is computed.
- The precision simulation is i.i.d. and three-point. It does not represent counterfactual dependence, parser failures, or a VLM. It does not select the study sample size.
- Alignment in the toy audit uses vectors of length 2 to 4. The function accepts larger arrays, and this checkpoint does not exercise image-shaped gradients.
- Toy intervals were produced with NumPy 1.26.4. A different quantile implementation can change endpoints. The pin is the one the tests import.
- This workstream does not check answer leakage, scene ambiguity, or gradient correctness. Those belong to the scene generator and the model interface.
- No MPS run was attempted. Nothing here should be read as an Apple-silicon timing or memory figure.

# Blockers

Blocking: none. Checkpoint 0 statistics can be reviewed without a model run.

Non-blocking:

- The primary estimand is intentionally unlocked pending Agent A and the research lead. The code labels the illustration as a proposal and continues.
- Confirmatory tables need scene manifests and model logs from the other workstreams. This branch does not read or edit those paths.
- A counterfactual-group sensitivity needs group ids joined from the scene manifest. The bootstrap argument exists. The toy summary does not use it.
- Real attack logs should gain a perturbation id before they are written. That field is outside the shared list, so it is not added here.

# Decisions requested from the research lead

1. Primary summary for RQ1. Recommendation: the scene-level paired difference in additional error, with attack success reported beside it. Tradeoff: additional error is defined on every scene and subtracts the clean error, and it counts an invalid output as a new error. Attack success does not count invalid output as success, and it is null when nobody was correct before. Not locked. Agent A's estimand overrides this recommendation.

2. Invalid attacked outputs in the attack-success denominator. Recommendation, already coded: keep them in the denominator as non-successes. Alternative: drop them and estimate the rate only among parseable attacked outputs. Tradeoff: dropping them can make the rate high when the attack mostly produces garbage.

3. Meaning of `correct_before`. Recommendation, checked in the toy packet: clean correctness under the target context. Alternative: clean correctness under the source context. Tradeoff: the source-context rule mixes the target's clean floor into a rate that is supposed to describe the target.

4. Where `A(s, t)` is evaluated. The function is only algebra. Recommendation: the clean image, before attack steps. Gradients at intermediate attack iterates are a different estimand.

5. Bootstrap unit. Recommendation: scene is the default, and a counterfactual-group resample is a pre-specified sensitivity before any confirmatory interval. Tradeoff: scene-level intervals can be too narrow when sibling scenes are dependent. The toy file has no group ids.

6. Attack-success resample. Recommendation, already coded: resample every scene in the cell, recompute the ratio, and omit draws with a zero denominator. Alternative: resample only the clean-correct scenes, which conditions on the observed baseline.

7. Multi-context encoding. Recommendation: `method=pgd_multi` and `source_context=multi`. Alternative: a list-valued source field, which breaks the scalar contract. Single-context rows still use the four context keys.

8. Exploratory conditional transfer. Recommendation, already coded: among same-context successes that were clean-correct on the target, the fraction of target successes. Scenes already wrong on the target are excluded and counted. Alternative: count those scenes as transfer failures. The exclusion count has to travel with the rate.

9. Undefined alignment. Recommendation, already coded: drop scenes with a zero target L1 norm from the mean and count them. Do not impute 0. Imputing 0 would treat a flat target gradient as disagreement.

10. Confirmatory seed, replicate count, and contrast family. Recommendation: register one primary contrast, the seed, and `B` before any sealed split is scored. Do not reuse the toy seed 20261106. No multiplicity adjustment is implemented. If several contrasts are co-primary, name the family and the adjustment first.

11. `perturbation_id`. Recommendation: add it to the shared attack schema before real attack logs are written, so transfer rows can be tied to one perturbation. It is not added in this checkpoint because it is outside the shared field list. Until then, transfer means rows that share source and target labels.

12. Interval estimator. Recommendation: keep the equal-tailed scene percentile bootstrap (NumPy linear quantile) and do not add p-values at this checkpoint. Revisit BCa only if a fabricated-DGP check shows a coverage problem the lead cares about. Do not switch methods after seeing a sealed split.

# Reproduction instructions

From a fresh clone, at the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e 'src/analysis[dev]'
python -m pytest -q tests/analysis
python -m analysis.verify_toy
```

`verify_toy` exits 0 when the fixtures rebuild `src/analysis/analysis/fixtures/toy/toy_summary.json`. It does not need a GPU, a model, or a download. `--write` regenerates the summary and is not part of the check.

Optional fabricated planning sketch:

```bash
python -m analysis.simulation --n 12 24 48 --replications 100 --boot 400 --seed 20261106
```

The Evidence section records the commands actually executed for this report, including the virtualenv path used so that `.venv` was not committed.

# Git information

- Branch: `checkpoint-0/analysis-plan`
- Base: `main` (`65a8d8ffb2f22f3d284b64e2d481151474761c5e`, initial commit)
- Commit: filled in after the commit that contains this test log
- Pull request: opened to `main`, not merged. URL filled in after the PR exists
- Test status: see the command output in Evidence
