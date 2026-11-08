# Review packet v2: Checkpoint 2.2 (H-ΔL) and CUDA readiness (2026-11-08)

For research lead Federico Cavaletto. It replaces packet v1 (commit `2127206`).

Nothing was run on the model:
- no weights were loaded, and no attack was run;
- no v5 scene was generated or rendered;
- no pod was created, and no RunPod account call was made;
- nothing was merged, and no frozen protocol was edited.

Every number below comes from committed development artifacts or from simulation.

## 1. Summary

1. **Q is already in hand on development, and it adds nothing detectable there.**
   - Checkpoint 1.8's `A(n,h)` is defined as `g_hᵀ sign(g_n) / ‖g_h‖₁`, so `Q = A · ‖g_h‖₁` is exact from stored scalars. The previous draft's model (b) was already the Q model.
   - Once magnitude is controlled, neither Q (t = 0.69, p = 0.24) nor `A` (t = 0.57, p = 0.29) predicts the PGD-3 `ΔL_h`.
   - Raw Q *without* magnitude control looks significant (p = 0.03), and that comes entirely from `‖g_h‖₁`.
   - Cosine was never stored for this pair, so the cosine comparison can only happen on sealed data.
2. **First-order is not a quantitative model of PGD here.**
   - It ranks damage moderately well (Spearman 0.58–0.67).
   - FGSM realizes about 7–8% of its first-order prediction.
   - PGD-3 overshoots for small predictions and undershoots for large ones.
   - Gradients are far from Gaussian: `‖g‖₁ / (√d‖g‖₂)` is about 0.25, against 0.80 for Gaussian coordinates.
3. **One primary model, M1s.** A saturating first-order model in Q, with the magnitude adjusted in the same functional form plus restricted cubic splines.
   - The splines were added after a simulation showed that the precommitted M1 rejects 11–12% under a nonlinear magnitude effect with no alignment effect at all. M1s holds 2.3–2.7% with no loss of power.
   - Secondary: M2 (additive `A`). Baselines: B0–B3. Eleven sensitivity rows.
4. **Alignment barely varies, so the study must be larger than planned.**
   - After the magnitude controls, `f(P_h)` keeps a residualized SD of 0.04, and five low-alignment development scenes carry 62–72% of its identifying variation.
   - For 80% power at "full transfer", where the helpful loss responds to its first-order prediction as strongly as the neutral loss responds to its own, about **310 groups** are needed. At 128 groups (draft v1) power is 0.42.
   - Half of full transfer is undetectable at any feasible size (power 0.44 at 512 groups).
5. **The cluster bootstrap is valid, CR1 is not.**
   - Under realistic dependence (sibling correlation 0.3–0.75, heteroskedastic, skewed errors), WCR holds 0.018–0.035 at nominal 0.025.
   - CR1 rejects 0.04–0.05, and the IK degrees of freedom are only about 10 at 128 groups.
   - WCU 95% intervals cover 0.91 at 128 groups and 0.93 at 256.
6. **CUDA: one more defect found and fixed** (PR 27, `691ee06`). Self-terminating right after the result could have erased the only copy of the log.
   - The job now stops itself (which costs $0 for this spec), waits 120 s for a log capture, and the operator saves the log before terminating.
   - Spending plan: $3 hard limit through the prepaid balance, about $0.30–0.60 planned. Not launched.

## 2. Mechanistic predictor analysis (`mechanistic_predictor.py`, `.json`)

**Order of work.** The model choice and the rule for the link scale were committed and pushed **before** this script existed (`MODEL_CHOICE_PRECOMMIT.md`, commit `d2321f7`). The analysis is exploratory, on development data.

**Incremental association with PGD-3 `ΔL_h`** (91 eligible scenes, 46 groups; WCR B = 9,999):

| Predictor (added to) | Coefficient | one-sided p | 95% WCU | nats per residual SD | LOGO R² |
| --- | --- | --- | --- | --- | --- |
| B0: margins only | — | — | — | — | 0.396 |
| B2: + log L1/L2 + f(S_h) | — | — | — | — | 0.397 |
| M1: `f(P_h)` given magnitudes and margins | 2.50 | 0.24 | [−3.0, 8.0] | 0.10 | 0.391 |
| `f(P_h)` added to B2 | 2.49 | 0.23 | [−2.7, 7.6] | 0.10 | 0.392 |
| M2: `A` given log L1 and margins | 0.74 | 0.29 | [−1.6, 3.0] | 0.09 | 0.377 |
| `A` added to B2 | 0.55 | 0.34 | [−1.8, 2.8] | 0.07 | 0.389 |
| Raw `P_h` given `S_h` and margins (v1 model b) | −0.18 | 0.69 | [−0.9, 0.5] | −0.08 | 0.406 |
| Raw Q given margins only (**no magnitude control**) | — | 0.033 | — | 0.40 | 0.410 |
| `cos(g_n, g_h)` | not stored on development | | | | |

**Comparison outcomes, M1:**

| Outcome | one-sided p |
| --- | --- |
| FGSM | 0.54 |
| Random sign (control) | 0.34 |
| PGD-7 | 0.73 (negative estimate) |

**First-order accuracy** (realized / predicted change):

| Attack | Source side | Transfer side |
| --- | --- | --- |
| FGSM | median 0.07 | median 0.08 |
| PGD-3 | median 1.14, terciles 2.05 / 2.37 / 0.57 | median 1.33 |
| PGD-7 | median 4.0, log–log slope 0.17 | — |

**Reading:**
- Q's extra information beyond the baselines is, by construction, the direction `A`.
- On development that information is undetectable, and its residualized spread is small.
- Q's apparent predictive power without magnitude control is a magnitude artifact, which is why every model conditions on magnitude.
- The first-order approximation ranks damage but does not get its scale right. A saturating link is the defensible way to use it, and S3 in the protocol separates "direction drift over PGD steps" from "nonlinear loss" using the realized `g_hᵀδ`.

## 3. Revised statistical protocol (`PROTOCOL_DRAFT.md` v2)

**Primary model, M1s (decision D1):**

`ΔL_h ~ f(P_h) + f(S_h) + M_n + M_h + log‖g_n‖₁ + log‖g_h‖₁ + RCS₄(log‖g_h‖₁) + RCS₄(log‖g_n‖₁)`

- `f(P) = 0.25 · asinh(P / 0.25)`, `P_h = 0.75 ε Q`, `S_h = 0.75 ε ‖g_h‖₁`.
- `H-ΔL: β_Q > 0`, one-sided WCR at 0.025.

| Model | Can detect | Cannot detect |
| --- | --- | --- |
| **M1s (primary)** | Alignment acting through the first-order channel with saturation. Also has good power for a purely additive `A` effect (0.69 at 256 groups, at 0.25 nats per SD of `A`). | Cosine-type alignment, effects linear in raw `P_h`, decision-only effects, effects below r ≈ 1 |
| **M2 (secondary)** | A linear shift in `A` that does not scale with magnitude | Mechanistic weighting. More exposed to leverage from the low-`A` tail. |
| **B0–B3 (baselines)** | How much margins, norms, the perfect-alignment prediction `f(S_h)`, and cosine explain | They are the comparison, not tests |
| **Raw Q, no magnitude control** | Nothing about alignment. Shown only as the confound. | — |

Draft v1's two-model split is withdrawn. Its "each blind in the other's world" finding came from an unsaturated linear first-order world, which development contradicts (FGSM realizes about 7–8%). With the saturating link, M1 and M2 each keep power in the other's world (M2 0.62 in the M1 world at 256 groups; M1 0.69 in the M2 world), so a single primary loses little.

**Sensitivity rows Σ1–Σ11.** Link scale, `c` re-estimated on the sealed source side, splines, L2 norms, group fixed effects, rank outcome, Huber, leverage trimming, float32 readout, color and descriptor indicators, CR2-IK. They are reported in one table and cannot change the decision.

**Controls:**
- C1 random-sign is a **gate**: if C1 is significant, H-ΔL is "not supported".
- C2 FGSM is interpretive.
- C3 is the generator gate.
- C4 float32 readout.

## 4. Sample-size simulation (`sample_size.py`, `spline_check.py`, `.json`)

**How the simulation works:**
- **Predictors:** whole development groups resampled.
- **Errors:** `σ(x) · (√ρ u_group + √(1−ρ) e)`, using the null-model residual SD of 2.09 nats, skew 0.64, and the observed heteroskedasticity. ρ ∈ {0.3, 0.53 (observed), 0.75}.
- **Effect anchor:** `β_Q = r × b`, where `b = 5.70` is the source-side response of `ΔL_n` to its own `f(P_n)`. `r = 1` is "full transfer". The anchor never uses a development estimate of `β_Q`.

**Size at nominal 0.025:**

| World | G | WCR | CR2-IK | CR1 | IK dof |
| --- | --- | --- | --- | --- | --- |
| ρ 0.3 / 0.53 / 0.75 | 128 | 0.034 / 0.035 / 0.024 | 0.031 / 0.026 / 0.020 | 0.053 / 0.048 / 0.041 | 9.6 |
| ρ 0.3 / 0.53 / 0.75 | 256 | 0.027 / 0.028 / 0.027 | 0.024 / 0.028 / 0.024 | 0.033 / 0.042 / 0.037 | 19.4 |
| Nonlinear magnitude, no alignment, M1 | 256 / 384 | **0.106 / 0.118** | — | — | — |
| Same world, M1s (splines) | 256 / 384 | **0.027 / 0.023** | — | — | — |
| tanh magnitude, M1 / M1s | 256 | 0.006 / 0.023 | — | — | — |

A "resampled-residual" world is also in the JSON. It is flawed by construction (the residuals are orthogonal to the tested columns), so it is not used. WCU 95% coverage under the null: 0.913 at 128 groups and 0.934 at 256. Intervals are slightly narrow at 128 groups.

**Power, WCR one-sided 0.025, observed ρ:**

| Effect | 128 | 192 | 256 | 320 | 384 | 512 |
| --- | --- | --- | --- | --- | --- | --- |
| M1, r = 0.5 | 0.15 | 0.23 | 0.26 | 0.30 | 0.39 | 0.44 |
| **M1, r = 1** | **0.42** | **0.56** | **0.72** | **0.82** | **0.86** | **0.95** |
| M1, r = 1.5 | 0.71 | 0.85 | 0.94 | 0.98 | 1.00 | 1.00 |
| M1, r = 2 | 0.88 | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 |
| M1s, r = 1 | — | — | 0.71 | — | 0.87 | — |
| M2 in the M1 world, r = 1 | 0.35 | 0.46 | 0.62 | 0.75 | 0.80 | 0.91 |
| M2, additive world, 0.25 nats per SD of `A` | 0.47 | 0.63 | 0.76 | 0.88 | 0.93 | 0.98 |
| M1 in that additive world | 0.43 | 0.54 | 0.69 | 0.83 | 0.87 | 0.94 |

**Robustness** (M1, r = 1):

| Variation | 256 groups | 384 groups |
| --- | --- | --- |
| ρ = 0.3 | 0.68 | 0.83 |
| ρ = 0.75 | 0.73 | 0.90 |
| True link scale c = 2, frozen 0.25 | 0.40 | 0.52 |

**Assurance.** Power averaged over the uncertainty in:
- the effect: `r ~ LogNormal(0, 0.5)`, median 1, 95% between 0.38 and 2.7;
- the sibling correlation: `ρ ~ N(0.53, 0.12)`;
- the residual SD: ×LogNormal(0, 0.15).

| | 128 | 192 | 256 | 320 | 384 | 512 |
| --- | --- | --- | --- | --- | --- | --- |
| M1 assurance | 0.54 | 0.66 | 0.72 | 0.78 | 0.81 | 0.86 |

These used CR1 for speed, and CR1 over-rejects here by 0.01–0.03 in size. Read them as upper bounds about 0.03–0.06 above WCR. The assurance never reaches 0.90 at a feasible size, because a third of the prior mass sits below r = 0.7, where nothing feasible has power.

**Implied partial R² of the alignment term:**

| Effect | Partial R² |
| --- | --- |
| r = 0.5 | 0.003 |
| r = 1 | 0.013 |
| r = 1.5 | 0.028 |
| r = 2 | 0.049 |

Even full mechanistic transfer explains about 1% of residual variance, because alignment varies so little across scenes.

**Sample size needed for 80% WCR power:**

| Effect | Groups |
| --- | --- |
| r = 1 | about 310 |
| r = 1.5 | about 170 |
| r = 2 | 128 or fewer (power 0.88 at 128) |
| r = 0.5 | more than 512 |

**What counts as scientifically meaningful.** `r = 1` means the helpful context inherits the attack's first-order structure as fully as the attacked context does. That is the natural "the mechanism works" benchmark, and smaller effects would not support a mechanistic reading at this budget. The power curve is honest about the cost: a study sized for r = 1 cannot see r = 0.5, and the report must say so.

**Nuisance uncertainty is resolved prospectively, from predictors only.** The biggest unknown is how much v5 spreads alignment. G1 measures it on dev-v5 (clean gradients only, no outcomes), and `G_final` is then fixed by the committed rule before any sealed render.

## 5. v5 dataset specification (`V5_GENERATOR_SPEC.md`)

- Every v4 per-scene constraint is kept, by importing the v4 checks.
- The 14 index rules are replaced with stated uniform draws, followed by rejection on any failed check. The target distribution is therefore "proposal conditioned on validity", which is exact and reproducible.
- Each group has its own PCG64 RNG, seeded by `SHA-256(seed | group k | attempt a)`, so groups are i.i.d. and reproducible one at a time.
- A novelty rule rejects any layout, or mirror image, that matches a development_v4 group, a dev-v5 group, or an earlier group in the same split.
- Cross-group balance is reported, not enforced, because enforcing it would make groups dependent.
- Splits:
  - dev-v5: 32 groups, seed `study2-checkpoint-2.2-dev-v5`;
  - sealed: seed `study2-checkpoint-2.2-sealed-v1`, groups `0..G_final−1`.
- The 48 development_v4 groups stay development data.

## 6. Estimated runtime (MacBook Air M4)

Per scene: about 117 s without the helpful → neutral secondary (S1), and about 177 s with it.

| G | Scenes | Without S1 | With S1 |
| --- | --- | --- | --- |
| 256 | 512 | about 17 h | about 25 h |
| 320 | 640 | about 21 h | about 31 h |
| 384 | 768 | about 25 h | about 38 h |

Plus:
- the generator build, about half a day of CPU work;
- G1 (dev-v5 clean gradients), about 50 min;
- G2 (pipeline pilot on 16 v4 scenes), about 1 h.

**Feasibility.** With approval on 9 November and the generator ready on 10 November, a 320-group run without S1 finishes around 11–12 November. With S1 it finishes around 12 November. Either leaves time to write before 16 November, as long as nothing slips. A 384-group run with S1 ends around 12–13 November, which is tight.

## 7. CUDA readiness (`CUDA_READINESS.md`)

**Documented:**
- an in-pod stop (`runpodctl pod stop $RUNPOD_POD_ID`);
- the pod-scoped key;
- a stopped pod's container disk is not billed;
- a $0 balance stops and terminates pods without a volume.

**Not documented:**
- whether the pod-scoped key may call the REST action route;
- whether pod logs survive termination.

**Fixed (PR 27, `691ee06`, 51 tests pass):**
- stop before terminate;
- a 120 s result grace;
- `capture_logs.sh` on the operator side.

**Safeguards, outermost first:**
1. balance ≤ $5 with auto-pay off;
2. a 60-minute operator terminate guard;
3. a 50-minute in-pod watchdog;
4. per-step 45-minute timeouts;
5. `JOB_DONE` printed only after full success.

**Plan:**

| | |
| --- | --- |
| Hard limit | **$3** |
| Planned spend | about $0.30–0.60 |
| Abort | `STARTING` for more than 15 min, or no `JOB_START` within 20 min |
| Pods | One pod per approval |

## 8. Decisions requiring your approval

Protocol (before freezing):

| # | Decision | Recommendation | Alternative |
| --- | --- | --- | --- |
| D1 | Primary model | **M1s**: M1 plus restricted cubic splines in both log norms. It departs from the precommitted M1, and that departure is justified by simulation alone (size 0.11 → 0.025, no power loss). | Precommitted M1 as primary, with splines as a sensitivity row. Not recommended: that test is invalid under plausible nonlinear magnitude effects. |
| D2 | Secondary model | M2 (additive `A`) | The cosine model (needs sealed cosine, not validated on development) |
| D3 | Sample size | Rule: `G_final` = smallest G with WCR power ≥ 0.80 at r = 1 on dev-v5 predictors, with **G_max = 320** | G_max 384 (+4 to 7 h). Or a fixed 256 (power 0.72 at r = 1, stated as under-powered). |
| D4 | Smallest effect of interest | r = 1 (full first-order transfer) | r = 1.5 (about 170 groups), which detects only stronger transfer |
| D5 | Helpful → neutral secondary (S1) | **Run it only if G_final ≤ 256.** At G_final = 320 it adds about 10 h. | Always run it (about 31 h at 320 groups), or drop it |
| D6 | Link scale | `c = 0.25`, frozen by the precommitted rule, which chose the grid end (disclosed) | Extend the grid below 0.25. Not recommended: the fit is flat there (R² 0.364 against 0.363). |
| D7 | Random-sign gate | Keep C1 as a gate | Report only |
| D8 | Stored diagnostics | Add cosine, L2 norms, `g_hᵀδ`, `g_nᵀδ`, and gradient tensors kept locally (about 2.4 GB) | Scalars only |
| D9 | Eligibility, ties | An exact tie is ineligible | Alphabetical rule |
| D10 | H\* / H-ΔL wording | As in the protocol banner and §17: H-ΔL is post hoc relative to H\*, prospective only on sealed v5, never called preregistered | — |

Execution gates (each a separate go, each loads weights on the Mac):

| # | Gate |
| --- | --- |
| G1 | Dev-v5 clean scoring plus both clean gradients (about 50 min). Sets `G_final`. |
| G2 | Pipeline pilot, 16 v4 scenes (about 1 h) |
| G3 | Render sealed-v5, commit the manifest hash |
| G4 | Sealed MPS run (about 17–31 h, depending on D3 and D5) |

CUDA (not needed for 2.2):

| # | Decision |
| --- | --- |
| C0 | Auto-pay off, balance ≤ $5 (your console action) |
| C1 | Local MPS one-scene sidecar (loads weights) |
| C2 | Allow the RTX A6000 as an alternative GPU |
| C3 | Build and publish the public GHCR image |
| C4 | Create one pod under the §7 plan, $3 hard limit |

## 9. Corrections to earlier work in this pass

| Item | Correction |
| --- | --- |
| Draft v1 model (b) | Its `A · S_h` is exactly the requested Q prediction `0.75 ε Q`. It was not a separate idea. |
| Draft v1 "each model blind in the other world" | An artifact of an unsaturated first-order world that development contradicts. Withdrawn. |
| Draft v1 power, 128 groups | Not comparable. The v2 effect anchor is mechanistic (r), and v2 uses heteroskedastic, skewed, realistic dependence. At 128 groups, power at r = 1 is 0.42. |
| Draft v1 runtime, 190–250 s per scene | Measured bases give about 117 s without S1 and about 177 s with it |
| The CUDA second pass | Self-terminated before the log could be saved (§7) |
| My own simulation | Its first power pass used the flawed resampled-residual world. It was stopped and rerun in the synthetic world. Only the rerun is reported. |
| First-order scale `0.75 ε` | The 8-bit round trip makes it about ε. This is a constant, so it only rescales `c`, and no test changes (protocol §3). |

## 10. Files

| File | What |
| --- | --- |
| `PROTOCOL_DRAFT.md` | Protocol v2 |
| `MODEL_CHOICE_PRECOMMIT.md` | Model choice, committed before the development Q analysis (`d2321f7`) |
| `mechanistic_predictor.py`, `.json` | Development Q analysis |
| `hdl_stats.py` | Shared OLS, CR1, CR2-IK, WCR, WCU, LOGO-CV code |
| `sample_size.py`, `.json` | Size, coverage, power, assurance, required G |
| `spline_check.py`, `.json` | M1 against M1s under magnitude misspecification |
| `V5_GENERATOR_SPEC.md` | v5 generator specification |
| `CUDA_READINESS.md` | Self-termination review and the spending plan |
| `power_continuous.py`, `.json` | Draft v1 simulation, kept for the record and superseded |
| PR 27 branch: `docker/cuda/run_job.sh`, `self_stop.py`, `docker/runpod/capture_logs.sh`, tests | CUDA fixes |
