# Checkpoint 2.2 protocol: alignment and the continuous target-loss change (DRAFT)

Status: **draft for review, not frozen.** Prepared 2026-11-08 for research lead Federico Cavaletto. This protocol becomes binding only after the decisions in `REVIEW_PACKET.md` are approved and its sha256 is recorded. Nothing in it has run:
- No model was loaded, no attack was run, and no pod was created.
- No held-out scene was generated or opened.
- No frozen protocol was edited.

Started from `checkpoint-2.1/independent-audit` (PR 26). Planning numbers come from `power_continuous.py` (results in `power_continuous.json`).

## 1. What this hypothesis is, and what it is not

H\* is the proposal's confirmatory hypothesis (`docs/research/RESEARCH_PROPOSAL.md` §4, §6), operationalized for development in `docs/checkpoint-2.1/ANALYSIS_PROTOCOL.md`. The Checkpoint 2.1 development run stopped it as written: there were 4 transfer failures against a floor of 40 (`effective_sample`). This protocol does not test H\*, refit it, or relax its gate.

The new hypothesis is named **H-ΔL** throughout, so no result can be quoted as H\*.

| | H\* (proposal, 2.1) | H-ΔL (this draft) |
| --- | --- | --- |
| Outcome | Binary `T`: perturbed helpful decision is not the label | Continuous `ΔL_h`: change in the labeled-color summed NLL under `helpful_nonrevealing` |
| Population | Source successes only (the attack already flipped `neutral`) | All scenes that are clean-correct under both contexts. Source success is not required. |
| Why the population differs | Conditioning on source success left 55 of 96 scenes, and only 4 failures | No conditioning on a post-attack event. About 94% of scenes are eligible. |
| Model | Linear probability, `A` plus three covariates | Two pre-specified linear models for `ΔL_h` (§8), co-primary under a Bonferroni split |
| Covariates | `M_neutral`, `M_helpful`, `‖g_neutral‖₁` (raw) | `M_neutral`, `M_helpful`, `log ‖g_neutral‖₁`, `log ‖g_helpful‖₁` (model a), or the first-order scale (model b) |
| Data | Development only (96 scenes) | A new sealed sample of 128 groups from a sampled generator (§4) |
| Status in the proposal | Confirmatory | §6 listed `ΔL_t` as a "loss diagnostic, not a second primary." It is promoted here after the binary design failed on development. |

Disclosure: the decision to promote `ΔL_t` was made after seeing the 2.1 development counts. Only three properties of the 2.1 development `ΔL_h` were examined:
- its standard deviation (2.96 nats);
- its skew (0.66);
- the correlation between sibling arms (0.48).

They set the planning error model. Nobody computed its association with `A` or any covariate on development, and this protocol forbids computing it before the sealed analysis. H-ΔL is prospective with respect to the sealed sample only.

A positive H-ΔL result would say that, at this budget, scenes with more aligned clean gradients lose more labeled-color likelihood under helpful text after a neutral-text attack. It is an association between two quantities computed from the same model. It is not a causal effect of alignment. It does not establish that alignment predicts whether the decision flips.

## 2. Hypotheses

Both are directional. Each is tested one-sided at 0.0125, so the family-wise one-sided level is 0.025.

- **H-ΔL(a), additive association.** In model (a), `β_A > 0`. Higher `A(neutral, helpful)` goes with a larger `ΔL_h`, holding both clean margins and both log gradient scales fixed.
- **H-ΔL(b), first-order scaling.** In model (b), `γ_A > 0`. The target-loss change grows with `A · S_h`, the local linear prediction, beyond what the gradient scale `S_h` alone predicts.

Why two: the power analysis (§13) shows that each model is powered in one world and nearly blind in the other.
- If the effect is additive in `A`, model (a) detects it and model (b) does not.
- If the effect follows the first-order formula `ΔL ≈ A · 0.75 ε ‖g_h‖₁`, model (b) detects it, and model (a) has power below its own false-positive rate. The heavy-tailed gradient norm absorbs the signal.

Choosing one model after the data would be a forking path, so both are fixed now and α is split. Decision D1 in the review packet offers the alternative of (a) primary and (b) secondary.

## 3. Frozen pieces carried over unchanged

| Piece | Value |
| --- | --- |
| Model | `Qwen/Qwen3-VL-2B-Instruct`, revision `89644892e4d85e24eaac8bacfd4f463576704203`, bfloat16, eager attention, MPS on the M4 |
| Processor, prefix, color token ids | Checkpoint 1.7 lock. red 1151, blue 12203, green 13250, yellow 27869. |
| Loss and score | Summed NLL of the labeled color from the first-answer logit row. The frozen scorer upcasts the bf16 logit row to float32 before `log_softmax`. |
| Attack | PGD, ε = 1/255, 3 steps, step ε/4, from the clean image, no restarts. Source `neutral` for the primary. |
| Official image | Half-to-even 8-bit round trip, clip to [0, 1], project the L∞ ball |
| Memory lines | Checkpoint 1.5 lines. The guard is not lowered. |
| Contexts | `neutral` and `helpful_nonrevealing` strings exactly as in `development_v4` |

Unchanged on purpose: ε, the step count, the source context, the model, and the scorer.

## 4. Sample

**The current generator cannot supply a sealed sample.** `scene_v4` is a closed-form design (`src/scene_v4/scene_v4/design.py`). Index `i` in 0..47 deterministically builds one counterfactual pair, and the seed chooses nothing. The 96 development scenes are the whole design, so there is no further sample to seal.

**Proposed: a sampled generator, `scene_v5`** (decision D2). It keeps every v4 constraint:
- the referent-color question form;
- the same four context strings;
- the color vocabulary and color-count structure (one color three times, two colors twice, one color once; the answers are the twice-colors);
- the arm-B referent swap;
- the geometry and spacing rules;
- the oracle and the ambiguity checks.

It draws, per group and from a seeded RNG, the things v4 fixed by index: descriptor, column placement, color cycle, arm order, and distractor arrangement. Scenes whose scene-graph hash appears in `development_v4` are rejected and redrawn. The rejection count is recorded.

- **Master seed and rule.** The seed string `study2-checkpoint-2.2-sealed-v1` and the rule "groups `0..127` in order, rejection sampling as above" are recorded in this protocol before any sealed image is rendered.
- **Development split of v5.** 32 groups from the separate seed string `study2-checkpoint-2.2-dev-v5`. It is used for exactly two things: generator tests, and a clean-accuracy check (clean scoring only, no attack). If the clean accuracy under both contexts on dev-v5 is below 0.80, the run stops and the generator is revised before anything is sealed.
- **Sealed split.** 128 groups, 256 scenes (decision D3). Rendering happens only after the protocol hash is recorded. The manifest sha256 is committed before the model sees any sealed image.
- **Independent sampling unit.** The counterfactual group. Groups are independent draws given the master seed. In v4, groups shared descriptors by construction, so v4 groups were not independent draws. v5 removes that.
- **Generalization.** The sealed population is the v5 generator's distribution, which is not identical to `development_v4`. Development results are not pooled with sealed results.

## 5. Units, rows, clusters

- Sampling unit: the counterfactual group.
- Analysis row: one scene and one direction. The primary uses only `neutral → helpful_nonrevealing`, one row per eligible scene. The helpful → neutral secondary uses a separate set of rows.
- Cluster: `counterfactual_group_id`, in every interval and every bootstrap.
- The two arms of a group are not averaged and are not treated as independent.

## 6. Eligibility, fixed before any outcome

A scene enters the primary analysis only when all of the following hold:

1. The clean `neutral` margin is > 0 and the clean `helpful_nonrevealing` margin is > 0 in the frozen scorer. An exact tie (margin 0) is ineligible and counted. This replaces the alphabetical tie rule for eligibility.
2. `A(neutral, helpful)` is finite and in [−1, 1]. Undefined means excluded, not coded as 0.
3. `‖g_neutral‖₁` and `‖g_helpful‖₁` are finite and > 0.
4. Both clean margins are finite.
5. Both perturbed helpful NLLs (PGD and the random-sign control) are recorded.

Each exclusion reason is counted. A missing outcome stays missing: it is not imputed and not replaced by a rerun. If more than 5% of eligible scenes are missing an outcome, the report says so in the first paragraph.

## 7. Outcome and predictors

- **Primary outcome.** `ΔL_h = L_h(x̃_n) − L_h(x)`, in nats. `L_h` is the summed NLL of the labeled color under `helpful_nonrevealing` in the frozen scorer. `x` is the official clean image. `x̃_n` is the official 8-bit image after step 3 of the neutral-source PGD. No trimming, winsorizing, or transformation.
- **Predictors, all on the clean official image, bf16 gradients of the summed NLL with respect to RGB, method of Checkpoint 1.8:**
  - `A = A(neutral, helpful) = ⟨g_h, sign(g_n)⟩ / ‖g_h‖₁`;
  - `M_neutral`, `M_helpful`: best-other summed NLL minus labeled summed NLL;
  - `log ‖g_neutral‖₁`, `log ‖g_helpful‖₁`;
  - `S_h = 0.75 · ε · ‖g_helpful‖₁`: the largest first-order increase available from three sign steps of ε/4.
- Why logs, and why the helpful norm: on development, `‖g_helpful‖₁` has median 297, mean 1935, and SD 6062. A raw norm would let a few scenes set the fit. The target gradient, not the source gradient, sets the target's first-order scale. Both choices are fixed here, before any sealed outcome.

## 8. Models

Ordinary least squares, one row per eligible scene.

**(a)** `ΔL_h = β0 + β_A·A + β1·M_neutral + β2·M_helpful + β3·log‖g_n‖₁ + β4·log‖g_h‖₁ + e`

**(b)** `ΔL_h = γ0 + γ_A·(A·S_h) + γ_S·S_h + γ1·M_neutral + γ2·M_helpful + e`

**Estimands.** `β_A` and `γ_A` are linear-projection coefficients in the eligible population of the v5 sealed generator, at the frozen attack.
- `β_A` is nats of target-loss change per unit of alignment, with the other columns fixed. Results are also reported per SD of `A`.
- `γ_A` is nats of target-loss change per nat of first-order prediction. The local linear picture predicts `γ_A = 1`. On development, the mean FGSM `ΔL_h` was about 0.08 of its first-order prediction, so a value well below 1 is expected and is not itself a failure.

No column is added, dropped, transformed, or interacted after outcomes are seen. There are no group fixed effects.

## 9. Inference

- **Decision test.** Wild cluster restricted bootstrap (WCR): Webb six-point weights, clustered by group, B = 9,999, seed `checkpoint-2.2|wcr`. One-sided p-value for `β_A > 0` and for `γ_A > 0`, each judged at 0.0125.
- **Intervals.** WCU percentile-t intervals at 95% and 97.5% (97.5% matches the split level).
- **Cross-check.** CR2 (Bell–McCaffrey) with Satterthwaite degrees of freedom. It does not decide.
- The analysis script is committed and its sha256 recorded before the sealed run. It runs once, on the complete sealed table.

## 10. Pre-fit checks (a failure means "not identifiable", reported as such)

- At least 100 of 128 groups contribute at least one eligible scene, and at least 180 eligible scenes in total.
- No predictor is constant. The design matrix condition number is < 1e12.
- SD of `A` is at least 0.05. On development it was 0.13.

If a check fails, the report gives the counts and stops. No column is dropped to rescue the fit, and the sample is not topped up after outcomes are seen.

## 11. Controls

| Control | Specification | Role |
| --- | --- | --- |
| C1 matched random-sign | The 2.0 matched random-sign construction at ε = 1/255, seed string `checkpoint-2.2-random|{scene_id}`. Same models, outcome `ΔL_h` under the random-sign image. | **Gate.** If the control's coefficient has one-sided p < 0.0125 in a model, that model's H-ΔL result is "not supported": `A` would be standing in for general fragility. This is the proposal's §7 rule. |
| C2 FGSM at 1/255 | One positive sign step of length ε from the clean neutral gradient. Same models. | Linear-regime comparison. Interpretive, not a gate. |
| C3 leakage and text-only | Contexts constant within the group, the answer is not a substring, `reveals_answer: false`. Checked by the generator before rendering. | Generation gate. |
| C4 float32 readout | `ΔL_h` recomputed with the final projection in float32 (hidden state upcast, vocabulary in chunks, `log_softmax` in float32), from the same forward pass. | Sensitivity. A sign change versus the frozen scorer is reported in the first paragraph. |
| C5 inference sensitivity | Scene-level clustering; rank outcome (`rank(ΔL_h)/n` in model a). | Sensitivity only. |

## 12. Secondary analyses (reported with 95% intervals; no gate, no promotion)

- **S1, helpful → neutral.** PGD-3 crafted on the `helpful_nonrevealing` loss, same ε and steps, evaluated under `neutral`. Outcome `ΔL_n = L_n(x̃_h) − L_n(x)`. Models (a) and (b) with the roles swapped: `A(helpful, neutral)`, `S_n`. Eligibility as in §6.
- **S2, directional asymmetry.** `D = mean[ΔL_h(x̃_n)] − mean[ΔL_n(x̃_h)]` over scenes eligible in both directions. 95% group-bootstrap percentile interval, 9,999 draws. Planning minimum detectable `D` at 128 groups is 0.41–0.65 nats (§13).
- **S3, binary transfer, descriptive.** The 2.1 counts (source success, conditional transfer, target success) in both directions. Ties are reported under two rules: "a tie is not the label" and the alphabetical rule. No fit.
- **S4, conditioning on source damage.** Model (a) plus `ΔL_n(x̃_n)` as a column. Source damage is a post-attack quantity, so this is a different estimand. It is labeled secondary and never replaces (a).

## 13. Power (planning; `power_continuous.py`)

Predictors are resampled as whole development groups. Errors come from the development `ΔL_h` shape (SD 2.96 nats, skew 0.66), with group correlation 0.48 from the development siblings. About 94% of scenes are eligible. Effect sizes are assumptions.

Model (a) in an additive world, WCR one-sided 0.025 (CR1 in parentheses). Rows are partial R² of `A`:

| Partial R² (nats per SD of A) | 64 groups | 128 groups |
| --- | --- | --- |
| 0 (test size) | 0.03 (0.05) | 0.04 (0.04) |
| 0.01 (0.30) | 0.16 (0.23) | 0.33 (0.39) |
| 0.02 (0.42) | 0.28 (0.37) | 0.50 (0.57) |
| 0.05 (0.68) | 0.58 (0.68) | 0.89 (0.93) |
| 0.10 (0.99) | 0.82 (0.93) | 0.99 (1.00) |

At the split level of 0.0125 and 128 groups (CR1), model (a) has power 0.50, 0.86, and 0.99 at R² 0.02, 0.05, and 0.10.

Model (b) in a first-order world at 128 groups and 0.0125 (CR1) has power 0.50 when the effect is 0.62 × the first-order prediction, and 0.83 at 1.0 ×.

Cross-world checks at 128 groups and 0.025 (CR1):

| Model | Additive world, R² 0.05 | First-order world, 1.0 × |
| --- | --- | --- |
| (a) | 0.93 | 0.01 |
| (b) | 0.20 | 0.90 |
| Rank version of (a) | 0.89 | 0.13 |

Reading:
- 128 groups is enough for a moderate effect (R² ≥ 0.05, about 0.7 nats per SD of `A`) in either world, if the right model is in the family. It is not enough for R² ≈ 0.02.
- Small or null results must be reported as "not detected at this size," not "no relation."
- Development FGSM moved `ΔL_h` by about 0.08 of its first-order prediction, which suggests a first-order effect would be weak at this budget.
- 192 groups would lift model (a) at R² 0.02 from 0.50 to about 0.76 (at 0.025).

## 14. Order of operations (each arrow is an approval gate)

1. Approve the decisions → freeze this file, record its sha256. Freeze the analysis script and record its sha256.
2. Build `scene_v5` with tests. Generate dev-v5 (32 groups). Run the C3 checks. → Run the clean-accuracy check on dev-v5 (MPS, loads weights, no attack).
3. Development pilot of the new pipeline on 8 `development_v4` groups (16 scenes). It only checks code: the helpful-source attack, the C4 readout, record schema, and timing. Its numbers are labeled DEVELOPMENT and are not used to choose anything. → 
4. Record the master seed rule (already in §4). Render the sealed split. Commit the manifest sha256 before any model run. →
5. Sealed run on MPS: 256 scenes, one pass, resumable from progress records. A crash resumes where it stopped and does not restart scenes already recorded. →
6. Run the frozen analysis once. Report everything in §§11–12, whatever the result.

## 15. Forbidden after the first sealed outcome is recorded

Changing:
- ε, the steps, or the source context;
- the scorer or the eligibility rules;
- the columns, the models, or the α split;
- the cluster unit or the sample size.

Also forbidden: rerunning sealed scenes for any reason other than a recorded crash, and presenting any changed fit as H-ΔL. Deviations go into a dated deviations section with the reason, before the analysis runs.
