# Checkpoint 2.2 protocol: H-ΔL, the continuous target-loss change (DRAFT v2)

Status: **draft for review, not frozen.** Revised 2026-11-08 for research lead Federico Cavaletto. It replaces draft v1 (commit `2127206`), which stays in git history. It becomes binding only after the decisions in `REVIEW_PACKET.md` are approved and its sha256 is recorded.

Update 2026-11-09: `RESEARCH_STRATEGY.md` (decision S2) proposes that this protocol be frozen with M1s and deferred, and that its primary deliverable be an interval estimate of r at 256 groups. Pending approval, the text below is unchanged.

Nothing in it has run:
- no model was loaded, and no attack was run;
- no v5 scene was generated or rendered;
- no pod was created;
- no frozen protocol was edited.

> ### H\* and H-ΔL are different hypotheses
>
> - **H\*** is the proposal's original confirmatory hypothesis: binary cross-context error transfer, among source successes, predicted by sign alignment `A` (`docs/research/RESEARCH_PROPOSAL.md` §4, §6; `docs/checkpoint-2.1/ANALYSIS_PROTOCOL.md`). On development it stopped at Checkpoint 2.1 with `effective_sample`: 4 transfer failures against a floor of 40. That result stands as written. This protocol does not test, refit, rename, or relax H\*, and it does not edit any H\* document.
> - **H-ΔL** is a **new hypothesis, formulated on 2026-11-08, after the H\* development counts had been seen.** It was **not** preregistered before the earlier experiments, and it must never be described as if it had been. It was developed on `development_v4`. That includes the development Q analysis (`mechanistic_predictor.py`), which ran after the model choice was committed (`MODEL_CHOICE_PRECOMMIT.md`, commit `d2321f7`). H-ΔL is prospective **only** with respect to the sealed v5 sample, which does not exist yet.
> - A positive H-ΔL result is never "support for H\*". Any write-up names H-ΔL, its date, and this disclosure in the same paragraph as the result.

## 1. Comparison

| | H\* (proposal, 2.1) | H-ΔL (this draft) |
| --- | --- | --- |
| When fixed | Before any attack (proposal, 2.1 protocol) | 2026-11-08, after the 2.1 binary counts and on development data |
| Outcome | Binary `T`: the perturbed helpful decision is not the label | Continuous `ΔL_h`: change in the labeled-color summed NLL under `helpful_nonrevealing` |
| Population | Source successes only | All scenes that are clean-correct under both contexts. Source success is not required. |
| Predictor of interest | `A(neutral, helpful)` | `f(P_h)`, where `P_h = 0.75 ε Q` and `Q = g_hᵀ sign(g_n) = A · ‖g_h‖₁` (§7) |
| Model | Linear probability, `A` plus three covariates | M1, a saturating first-order model (§8). One primary model. |
| Data | `development_v4` (96 scenes) | A new sealed v5 sample of independently drawn groups (§4) |
| Status | Stopped, `effective_sample` | Proposed |

## 2. Hypothesis

**H-ΔL.** In the primary model (M1s, §8; plain M1 if decision D1 keeps it), `β_Q > 0`: scenes whose clean helpful gradient agrees more with the sign of the clean neutral gradient lose more labeled-color likelihood under helpful text after the frozen neutral attack. "Agrees more" means agreement weighted by the helpful gradient's magnitude, holding the magnitudes and both clean margins fixed. The test is one-sided at α = 0.025 with the WCR bootstrap. There is one primary test and no split.

What a positive result would mean:
- at this budget, the mechanistically predicted direction of cross-context damage carries information beyond gradient size and confidence;
- it is an association between quantities computed from one model, not a causal effect of alignment;
- it says nothing about whether decisions flip.

## 3. Frozen pieces carried over unchanged

| Piece | Value |
| --- | --- |
| Model | `Qwen/Qwen3-VL-2B-Instruct`, revision `89644892e4d85e24eaac8bacfd4f463576704203`, bf16, eager attention, MPS on the M4 |
| Processor, prefix, color token ids | Checkpoint 1.7 lock (red 1151, blue 12203, green 13250, yellow 27869) |
| Loss and score | Summed NLL of the labeled color from the first-answer logit row. The bf16 logit row is upcast to float32 before `log_softmax`. |
| Attack | PGD, ε = 1/255, 3 steps, step ε/4, from the clean image, no restarts, source `neutral` |
| Official image | Half-to-even 8-bit round trip, clip to [0, 1], project the L∞ ball |
| Gradients | bf16 gradient of the summed NLL with respect to RGB on the clean official image, the Checkpoint 1.8 method |
| Contexts | `neutral` and `helpful_nonrevealing`, exactly as in `development_v4` |
| Memory lines | Checkpoint 1.5 |

Note on the first-order scale. `0.75 ε` is the pre-round-trip size of three agreeing steps of ε/4. After the 8-bit round trip, a coordinate at ±0.75/255 rounds to ±1/255, and one at ±0.25/255 rounds to 0, so the official perturbation's first-order size is closer to `ε · Q`. A constant factor only rescales `c`: `c · asinh(0.75 P / c) = 0.75 · c′ · asinh(P / c′)` with `c′ = c / 0.75`. So it does not change any test. The factor stays 0.75, as committed in `d2321f7`. The realized `g_hᵀ δ` (§7) measures the true value.

## 4. Sample

- **Generator.** `scene_v5`, specified in `V5_GENERATOR_SPEC.md`. It keeps every v4 constraint. Every choice that v4 fixed by index is replaced with a seeded draw from a stated distribution, and invalid draws are rejected and redrawn. Each group has its own RNG keyed by `(split seed, group index, attempt)`, so groups are independent and identically distributed draws.
- **Dev-v5.** 32 groups, seed `study2-checkpoint-2.2-dev-v5`. Used for generator tests and gate G1 only: clean scoring plus both clean gradients, with no attack and no `ΔL`.
- **Sealed-v5.** Seed `study2-checkpoint-2.2-sealed-v1`, groups `0..G_final−1`. Rendered only after this file and the analysis script are hashed. The manifest hash is committed before the model sees any sealed image.
- **`G_final`.** Fixed before the sealed render by the predictors-only rule in `V5_GENERATOR_SPEC.md` §5. Rerun `sample_size.py` with the dev-v5 predictor pool, and take the smallest G on {128, 192, 256, 320, 384} with WCR power ≥ 0.80 for M1 at `r = 1`, capped at `G_max` (decision D3).
- **Development data.** The 48 `development_v4` groups (96 scenes) remain development data. They are never pooled with sealed results, never used as a test, and never reweighted into the estimand.

## 5. Units, rows, clusters

- Independent sampling unit: the counterfactual group.
- Analysis row: one eligible scene. The primary uses the `neutral → helpful` direction only.
- Cluster: the group, in every interval, bootstrap, and cross-check.
- The two arms are neither averaged nor treated as independent.

## 6. Eligibility, fixed before any outcome

A scene is eligible when all of the following hold. Each exclusion reason is counted.
1. Both clean margins are strictly > 0 (an exact tie is ineligible).
2. `A(neutral, helpful)` is finite and in [−1, 1].
3. Both gradient L1 norms are finite and > 0.
4. The PGD-3 and random-sign perturbed helpful NLLs are recorded.

A missing outcome is not imputed and not rerun. If more than 5% of eligible scenes are missing an outcome, the first paragraph of the report says so.

## 7. Outcome, predictors, stored diagnostics

- **Outcome.** `ΔL_h = L_h(x̃_n) − L_h(x)`, in nats, with no trimming or transformation. `x̃_n` is the official image after PGD step 3.
- **Primary predictor.** `f(P_h)`, with:
  - `P_h = 0.75 · ε · Q`;
  - `Q = g_hᵀ sign(g_n) = A(n,h) · ‖g_h‖₁`, exact;
  - `f(P) = c · asinh(P / c)`.
- **Link scale `c`.** `c = 0.25` nats, frozen. It was chosen on development by the precommitted source-side rule: `ΔL_n(PGD-3)` regressed on `f(0.75 ε ‖g_n‖₁)` over the grid {0.25, …, 32}, keeping the smallest RSS. The rule picked the grid end, 0.25 (R² 0.364, against 0.181 for a linear link). The disclosure: the source-side response is nearly logarithmic, and pushing `c` lower barely changes the fit (R² 0.364 at 0.25 versus 0.363 at 0.5). The sealed data never re-chooses `c`, except in sensitivity Σ2.
- **Covariates.** `M_n` and `M_h`, `log‖g_n‖₁` and `log‖g_h‖₁`, and `f(S_h)` with `S_h = 0.75 ε ‖g_h‖₁`.
- **Stored per scene** (new and cheap, all computed from tensors the pipeline already holds):
  - `‖g‖₂` for both gradients;
  - `cos(g_n, g_h)`;
  - `A(h,n)`;
  - the realized `δ = x̃_n − x` sha256 and `‖δ‖₁`;
  - `g_hᵀδ` and `g_nᵀδ`, the exact first-order predictions for the realized perturbation;
  - the same quantities for FGSM and random sign.
- **Gradient tensors.** Saved locally as float32 (about 2.4 MB each, about 2.4 GB for 512 scenes). Their sha256 is committed. The tensors stay out of git.

## 8. Models: one primary, one secondary, baselines, sensitivity

OLS, one row per eligible scene.

**M1s, primary (saturating first-order with flexible magnitude) — recommended, pending decision D1.**
`ΔL_h = β0 + β_Q·f(P_h) + γ1·f(S_h) + γ2·M_n + γ3·M_h + γ4·log‖g_n‖₁ + γ5·log‖g_h‖₁ + RCS₄(log‖g_h‖₁) + RCS₄(log‖g_n‖₁) + e`

`RCS₄` is a restricted cubic spline with 4 knots at the 5th, 35th, 65th, and 95th sealed percentiles. These are predictor quantiles, fixed before any outcome is read. Each spline adds 2 nonlinear columns.

The splines are a departure from the precommitted M1 (`d2321f7`, the same model without the splines), and the departure is disclosed:
- `spline_check.py` showed that M1's WCR test rejects 10.6–11.8% at nominal 2.5% when the truth has a quadratic magnitude effect and no alignment effect;
- M1s holds 2.3–2.7%;
- M1s keeps M1's power (0.709 against 0.710 at 256 groups, r = 1).

The change rests on simulation alone, with no outcome data. If D1 keeps M1, the splines become sensitivity row Σ3.

Why this is the theoretically justified choice:
- For a sign step along `sign(g_n)`, the first-order change in the helpful loss is `ε·Q`. Q is the mechanistic quantity.
- `Q = A · ‖g_h‖₁` mixes direction and magnitude. Magnitude predicts damage for reasons unrelated to cross-context alignment, so M1 includes the magnitude in the same nonlinear form, `f(S_h)`, plus the log norms. Then `β_Q` is identified only by direction.
- Realized changes are far from linear in the prediction:
  - FGSM realizes about 7% of its source-side first-order prediction;
  - PGD-3's realized/predicted ratio ranges 0.57–2.37 across prediction terciles;
  - log–log slopes are 0.56–0.88.

  The saturating link keeps the few largest-gradient scenes from dominating the fit. Its scale was set on the source side, not on the hypothesis.

What M1 can detect: an alignment effect that works through the first-order channel with saturation. With `c = 0.25`, `f(P_h) − f(S_h) ≈ c · log A` for most scenes, so M1 asks whether lower alignment goes with less helpful damage, on a log-alignment scale.

What M1 cannot detect:
- an effect of alignment measured by cosine rather than sign agreement (see B3);
- an effect that is linear in raw `P_h`;
- an effect that appears only through decision flips;
- any effect smaller than its power (§14).

It can also be fooled. A magnitude effect that is nonlinear in a way `f(S_h)` and the log norms miss could load on `f(P_h)`. §14 measures the size of that risk.

**M2, secondary (additive sign alignment).**
`ΔL_h = β0 + β_A·A + γ2·M_n + γ3·M_h + γ4·log‖g_n‖₁ + γ5·log‖g_h‖₁ + e`

It is agnostic about mechanism. It detects a linear-in-`A` shift that does not scale with the gradient magnitude. It weights the alignment effect evenly across scenes, while M1 weights it by log alignment. In practice M1 and M2 rest on the same identifying variation in `A`. On development, five low-alignment scenes (A = 0.20–0.60) carry 62–72% of it. Reported with a 95% WCU interval and a WCR p-value, but it is not a test of H-ΔL.

**Baselines, nested, all reported.** Each is reported with in-sample R², leave-one-group-out R², and group-bootstrap intervals:

| Name | Columns |
| --- | --- |
| B0 | `M_n`, `M_h` |
| B1 | B0 + `log‖g_n‖₁`, `log‖g_h‖₁`, `log‖g_n‖₂`, `log‖g_h‖₂` |
| B2 | B1 + `f(S_h)` |
| B3 | B2 + `cos(g_n, g_h)` |

Increments reported, descriptive only, no gate:
- `f(P_h)` added to B2;
- `A` added to B2;
- `cos` added to B2;
- `f(P_h)` added to B3 ("Q beyond cosine");
- raw `Q` with no magnitude control. On development this one looks significant (p = 0.03), purely because Q contains `‖g_h‖₁`. It is shown as the cautionary contrast.

**Sensitivity analyses (Σ), reported in one table, never replacing M1.**

| # | Change |
| --- | --- |
| Σ1 | `c` ∈ {0.5, 1, 2, 4}, and the linear link (the draft v1 model (b)) |
| Σ2 | `c` re-chosen by the same source-side rule on the sealed neutral side |
| Σ3 | Magnitude adjusted by a restricted cubic spline (4 knots) in `log‖g_h‖₁` and in `log‖g_n‖₁` |
| Σ4 | Add both L2 norms (B1 covariates) |
| Σ5 | Group fixed effects (within-pair identification only) |
| Σ6 | Rank outcome `rank(ΔL_h)/n` |
| Σ7 | Huber M-estimation (k = 1.345), cluster bootstrap |
| Σ8 | Drop the 5% of scenes with the largest leverage on `β_Q` |
| Σ9 | Float32 readout outcome (control C4) |
| Σ10 | Answer-color and descriptor indicators added |
| Σ11 | CR2 with Imbens–Kolesár degrees of freedom instead of WCR |

The report states how many Σ rows agree in sign with M1 and whether any reverses it. No Σ row can change the H-ΔL decision.

## 9. Inference

- **Decision.** Wild cluster restricted bootstrap: Webb six-point weights, clustered by group, null imposed, B = 9,999, seed `checkpoint-2.2|wcr`. One-sided p for `β_Q > 0`, judged at 0.025.
- **Intervals.** WCU percentile-t, 95%.
- **Cross-check, not deciding.** CR2 (Bell–McCaffrey) with Imbens–Kolesár degrees of freedom. CR1 with t(G−1) is reported only for comparison: it over-rejects here (§14).
- **Leverage report.** The IK degrees of freedom, and the share of the tested column's residualized variance carried by the top 5 and top 10 groups. On development, IK dof was 3.4 at 46 groups.
- The analysis script (`hdl_stats.py` plus a sealed driver) is committed and hashed before the sealed run, and it runs once on the complete table.

## 10. Pre-fit checks (a failure reports "not identifiable" and stops)

- At least 85% of groups contribute an eligible scene.
- The residualized SD of `f(P_h)` is ≥ 0.02. On development it was 0.040.
- The design matrix condition number is < 1e12.

No column is dropped to rescue the fit, and the sample is not topped up after outcomes.

## 11. Controls

| # | Control | Role |
| --- | --- | --- |
| C1 | Matched random-sign at ε = 1/255, seed `checkpoint-2.2-random\|{scene_id}`. the primary model with outcome `ΔL_h(random)`, `f(P_h)` unchanged. | **Gate.** If C1's `β_Q` has one-sided WCR p < 0.025, H-ΔL is "not supported": `f(P_h)` would be tracking fragility that is not directional. On development, C1 gave p = 0.34. |
| C2 | FGSM at ε, the one step along exactly `sign(g_n)`. the primary model with outcome `ΔL_h(FGSM)`. | Interpretive. This is where first-order should be most accurate. On development, p = 0.54. |
| C3 | Generator leakage and text checks (`V5_GENERATOR_SPEC.md` §6) | Generation gate |
| C4 | Float32 final-projection readout of `ΔL_h` from the same forward pass | Sensitivity Σ9. A sign change is reported in the first paragraph. |

## 12. Secondary analyses (95% intervals, no gate, no promotion)

- **S1, helpful → neutral.** PGD-3 crafted on the helpful loss, evaluated under neutral. M1 with the roles swapped. This is the helpful-to-neutral comparison approved on 2026-11-08 (decision D5 sets whether it is run).
- **S2, asymmetry.** `mean ΔL_h(x̃_n) − mean ΔL_n(x̃_h)` over scenes eligible in both directions, with a group-bootstrap interval.
- **S3, first-order accuracy, descriptive.** Realized `ΔL` against `ε·Q` (prediction from the clean gradient) and against `g_hᵀδ` (prediction from the realized perturbation), for FGSM and PGD-3, on both sides. This separates two failures: the direction drifting over steps, and the loss being nonlinear.
- **S4, binary transfer, descriptive.** The 2.1-style counts, ties under both rules, no fit.
- **S5, source damage.** M1 plus `ΔL_n(x̃_n)`. This is a different, post-attack estimand.

## 13. What the development data say (exploratory; `mechanistic_predictor.json`)

These results come from the 91 eligible development scenes (46 groups). They did not change the model choice, which was committed in `d2321f7`.

- **Q is exact on development.** Checkpoint 1.8 defined `A(n,h) = g_hᵀ sign(g_n) / ‖g_h‖₁`, so `Q = A·‖g_h‖₁`. Cosine was never stored for this pair, so the cosine comparisons are sealed-only.
- **Gradients are far from Gaussian.** `‖g‖₁ / (√d‖g‖₂)` has median 0.25–0.26 (p10–p90 0.16–0.35), against 0.80 for Gaussian coordinates. Sign alignment and cosine can therefore differ, and the sealed data will measure by how much.
- **First-order accuracy:**

  | Attack, side | Realized / predicted, median | Spearman | Log–log slope |
  | --- | --- | --- | --- |
  | FGSM, source | 0.07 | 0.67 | 0.88 |
  | FGSM, transfer | 0.08 | 0.58 | 0.80 |
  | PGD-3, source | 1.14 (terciles 2.05, 2.37, 0.57) | 0.62 | 0.56 |
  | PGD-3, transfer | 1.33 (terciles 1.71, 2.48, 0.73) | 0.60 | 0.70 |
  | PGD-7, source | 4.0 | 0.50 | 0.17 |

  The clean-gradient prediction ranks the damage moderately well. It does not get the scale right: FGSM undershoots by more than 10×, and PGD overshoots for small predictions and undershoots for large ones. First-order is not a quantitative model of multi-step PGD here. That is why M1 uses a saturating link, and why S3 is planned.
- **Incremental information, PGD-3 `ΔL_h`.** None detected:

  | Test | t | one-sided WCR p | 95% WCU | nats per residual SD |
  | --- | --- | --- | --- | --- |
  | M1 `f(P_h)` | 0.69 | 0.24 | [−3.0, 8.0] | 0.10 |
  | M2 `A` | 0.57 | 0.29 | [−1.6, 3.0] | 0.09 |

  - Leave-one-group-out R²: B2 0.397, M1 0.391.
  - Raw Q without magnitude control: p = 0.033, entirely a magnitude effect.
  - Random-sign control: nothing. PGD-7: negative point estimates.
- **Why power is hard.** After the magnitude controls, `f(P_h)` keeps a residualized SD of only 0.040 (raw SD 0.335). Its correlation with `f(S_h)` is 0.992. Alignment varies little compared with magnitude, and the few low-alignment scenes carry most of the variation.
- **Error structure (null model).** Residual SD 2.09 nats, sibling correlation 0.53, skew 0.64. `|e|` rises 0.23 nats per unit of `log‖g_h‖₁`. The sibling correlation of `f(P_h)` is 0.02, and of `A` it is −0.21. Pairs therefore cost almost no efficiency for the coefficient (design effect ≈ 1 + ρ_x ρ_e ≈ 1).

## 14. Inference validation and sample size (`sample_size.py`, `sample_size.json`)

Setup:
- **Predictors:** whole development groups resampled.
- **Errors:** `σ(x)(√ρ u_group + √(1−ρ) e)`, with:
  - the residual SD (2.09), skew, and `|e|`-on-`log‖g_h‖₁` slope of the development null model;
  - ρ ∈ {0.3, 0.53, 0.75}.
- **Effect anchor:** `β_Q = r · b`, with `b = 5.70` the source-side response. `r = 1` is "full transfer": helpful damage tracks its own first-order prediction as tightly as neutral damage tracks its own.
- **Repetitions:** 2,000 for size and 1,000 for power, with B = 399.

**Size**, nominal 0.025:

| Condition | Rejection rate |
| --- | --- |
| WCR, M1/M2, clean worlds | 0.018–0.035 |
| WCR, 256 groups | 0.026–0.028 |
| CR2-IK | 0.019–0.031 |
| CR1 | 0.03–0.05 (over-rejects; reported only) |
| WCU 95% coverage | 0.913 at 128 groups, 0.934 at 256 |
| Nonlinear magnitude null, M1 | 0.07–0.12 |
| Same null, M1s | 0.023–0.027 |

**Power**, WCR, M1, r = 1:

| G | 128 | 192 | 256 | 320 | 384 | 512 |
| --- | --- | --- | --- | --- | --- | --- |
| Power | 0.42 | 0.56 | 0.72 | 0.82 | 0.86 | 0.95 |

Other effect sizes:
- r = 0.5: 0.44 at 512 groups.
- r = 1.5: 0.94 at 256.
- r = 2: 0.88 at 128.
- Additive world, 0.25 nats per SD of `A`: M1 0.69 and M2 0.76 at 256.

Robustness at 256 / 384 groups:

| Variation | Power |
| --- | --- |
| ρ = 0.3 | 0.68 / 0.83 |
| ρ = 0.75 | 0.73 / 0.90 |
| True link scale c = 2 | 0.40 / 0.52 |

**Sample size.** 80% WCR power at r = 1 needs **about 305 groups**. Other effect sizes need about 168 groups (r = 1.5), 128 or fewer (r = 2), and more than 512 (r = 0.5).

**Assurance** over `r ~ LogNormal(0, 0.5)`, ρ, and residual-SD uncertainty is 0.72 at 256 groups, 0.78 at 320, and 0.81 at 384. These are CR1-based upper bounds.

**Implied partial R²** at r = 1 is 0.013.

**Reading.**
- The study can detect alignment effects at or above full first-order transfer.
- It cannot detect weaker transfer.
- A null result must be worded with that power stated.

`G_final` is fixed by §4 after G1.

## 15. Order of operations (each arrow needs a separate approval)

1. Approve the decisions. Freeze this file and the analysis script, and record both sha256s.
2. Build `scene_v5` with tests. Generate dev-v5 and run C3. → **G1**: clean scoring plus both clean gradients on dev-v5 (MPS, about 50 min, no attack). Then:
   - clean accuracy must be ≥ 0.80 under both contexts, or the generator is revised;
   - rerun `sample_size.py` on the dev-v5 predictors, and fix `G_final` by the §4 rule. Commit it.
3. → **G2**: a pipeline pilot on 8 `development_v4` groups. It checks only code, schema, timing, and the new stored diagnostics. Its numbers are labeled DEVELOPMENT and choose nothing.
4. → **G3**: render sealed-v5 (groups `0..G_final−1`) and commit the manifest sha256.
5. → **G4**: the sealed MPS run, one pass, resumable. A crash resumes and never restarts a recorded scene.
6. Run the frozen analysis once. Report §§8–12 whatever the result.

## 15a. Runtime (MacBook Air M4; measured bases: 1.8 clean gradients 43 s per scene, 2.1 PGD 7 steps 161 s per scene)

| Bundle per scene | Seconds |
| --- | --- |
| Clean scoring + both clean gradients | about 43 |
| PGD-3 neutral + evaluations | about 60 |
| FGSM + random sign (gradient reused) | about 6 |
| PGD-3 helpful (S1) | about 60 |
| Float32 readout + stored diagnostics | about 8 |
| **Total with S1 / without S1** | **about 177 / about 117** |

| G | Scenes | Without S1 | With S1 |
| --- | --- | --- | --- |
| 256 | 512 | about 17 h | about 25 h |
| 320 | 640 | about 21 h | about 31 h |
| 384 | 768 | about 25 h | about 38 h |

Add about 50 min for G1 and about 1 h for G2. Sealed runs are resumable across nights.

## 16. Forbidden after the first sealed outcome is recorded

Changing any of:
- ε, the steps, or the source;
- the scorer, the eligibility rules, or `c`;
- the columns, M1, the α level, or the cluster unit;
- `G_final`.

Also forbidden: rerunning sealed scenes except after a recorded crash; selecting any Σ row as the result; and describing H-ΔL as preregistered before Checkpoint 2.1. Deviations go into a dated deviations section, with the reason, before the analysis runs.

## 17. How results will be worded

| Outcome | Wording |
| --- | --- |
| WCR p < 0.025 and C1 not significant | "H-ΔL supported on sealed v5" with `β_Q`, its interval, and nats per residual SD. Always next to: the post-hoc origin of H-ΔL, the C1 and C2 results, the Σ table, and the statement that H\* remains stopped. |
| WCR p < 0.025 and C1 significant | "Not supported: the predictor also tracks non-directional fragility." |
| WCR p ≥ 0.025 | "Not detected at G = … (power … at r = 1)". Never "no relation". The interval is reported. |
| Pre-fit check fails | "Not identifiable" with the counts |
