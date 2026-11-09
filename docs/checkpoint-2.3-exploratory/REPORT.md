# Checkpoint 2.3: Why single-step attacks fail on a bf16 VLM, and what precision does and does not change

**Status: exploratory, development data only, descriptive.** Nothing here confirms a hypothesis. The frozen Checkpoint 2.1 results and the preregistered H\* are unchanged. H-ΔL is post hoc and is not tested here. No sealed or new samples were used.

Model: Qwen3-VL-2B-Instruct, bf16 on Apple MPS, eager attention. Attack: the frozen Checkpoint 2.0 attack: L∞ ε = 1/255, sign steps of ε/4, on the labeled-color summed NLL. Official scores use an 8-bit round trip. Scenes: arm a of all 48 development groups (S48). A fixed 16-scene subset (S16) also gets PGD-7 and the fp32 vision tower. Plan: [`EXPLORATORY_PLAN.md`](EXPLORATORY_PLAN.md), commit 0c6f176, written before any full-run data existed.

**Revised 2026-11-09** after the pre-registered random-mask control (Checkpoint 2.3b, [`MASK_CONTROL_PLAN.md`](MASK_CONTROL_PLAN.md); results in §2b). The control replaced the §2 interpretation of *why* PGD's sparse perturbation wins. Earlier text that the control superseded is marked.

Intervals are 95% percentile bootstraps over scenes: 9,999 draws, seed 20261109, one scene per group. "Paired" means the difference is computed within each scene before bootstrapping. Margin is z_label − z_best-other in bf16 logits, which are quantized to steps of 0.125. Three scenes (dev4-ref-20, 26, 31) are not clean-correct in both contexts; they stay in continuous measures and drop out only of ratios that need a positive clean margin.

---

> **Terminology note (2026-11-09, Checkpoint 2.4).** Earlier drafts said "pixels" where they meant **RGB channel coordinates**: single entries of the 3 × 384 × 512 image tensor, one color channel of one pixel. Counts, masks and supports in this report are over coordinates. The wording was corrected; no number changed. The pre-registered plans keep their original wording. The held-out confirmation is in [`../checkpoint-2.4-confirm/`](../checkpoint-2.4-confirm/).

## 0. Summary

| Question | Answer, on these 48 scenes | Strength |
|---|---|---|
| Why does PGD-3 beat FGSM on the same ε-ball? | FGSM spends most of its budget past the point of useful damage. One ε/4 step in FGSM's direction does as much damage as the full ε step (ratio 1.00, CI 0.89–1.37). The gradient becomes **uncorrelated with the clean gradient after a single ε/4 step** (cosine 0.002, CI −0.007 to 0.008). After the 8-bit round trip, **the official PGD-3 perturbation is FGSM restricted to about 30% of its RGB channel coordinates**, with no sign reversals. This follows from the attack's arithmetic and is not evidence of anything by itself (§2.5). That sparser perturbation does **2.55× the damage** (CI 2.30–3.28). | Robust observation |
| Is it sparsity or PGD's choice of coordinates? (2.3b) | **Choice of coordinates.** A random mask of exactly the same number of FGSM coordinates recovers only **8% of PGD's advantage** (CI 1–14%). PGD's mask beats **all 8 random masks in 95 of 96 scene–source units**. It flips **60–69%** of clean-correct answers; random masks flip 28%. Selecting the coordinates with the largest clean-gradient magnitude does **worse than random** (−1.6 margin units), although the linear model predicts 2.8× more damage for it. PGD's coordinates look slightly *less* important than random to the clean linear model (0.92×). | Robust observation, pre-registered. Why those coordinates matter is open. |
| Does the clean linear model rank attacks correctly? | No. It predicts that FGSM beats PGD-3 by 3.6×, and it ranks them the wrong way round in **98% of neutral scenes and 100% of helpful scenes**. FGSM achieves 2.3% of its first-order predicted margin drop; PGD-3 achieves 27%. | Robust |
| How much does bf16 matter? | The output projection does not matter: fp32 projection moves margins by 0.07 (median), and every first-order ratio is unchanged. The vision-tower precision **changes the gradient direction a lot**: the clean-gradient cosine between bf16 and fp32 vision is **0.29–0.36**. That is less than the 0.87 cosine between the two text contexts. **It does not change the attack outcome**: PGD-3 damage differs by −0.31 (CI −0.88 to 0.56), and attacks cross-evaluate between precisions with no loss. The FGSM/PGD gap and the one-step decorrelation both persist with an fp32 vision tower. | Robust for the vision tower. The bf16 language model is untested. |
| Is neutral→helpful transfer asymmetric? | Slightly, and only at higher damage. Both sources are equally strong (paired difference 0.5, CI −0.75 to 1.0). Helpful-crafted attacks hurt the neutral context **1.1 margin units more** than the reverse (CI 0.5–2.0). At matched source damage the gap grows with strength: 0.22 at a drop of 2, 0.67 at 4, and **1.35 at 6 (CI 0.61–1.88)**. Binary flip-based "conditional transfer" points the **opposite** way (100% vs 85%). | Modest. The continuous measure and the binary measure disagree. |

The strongest claim the evidence supports, and the next experiment that would most raise its credibility, are in §9.

The main safety lesson is that the cheap, gradient-local tools are misleading on this model: single-step attacks, first-order damage predictions, and gradient-direction diagnostics. The expensive, outcome-level measure is stable: iterative attacks scored after quantization.

---

## 1. Run integrity

| Check | Result |
|---|---|
| Status | Completed. All 48 S48 scenes and all 16 V32 scenes were measured, and none was dropped by a time cap. |
| Runtime | 4.23 h after model load, within the 4.6 h limit. Stage B took 3.56 h and V32 took 0.63 h. |
| Reproduction against the frozen 2.1 records | **320/320** official NLLs, margins and decisions bit-exact; max difference 0.0. A second independent check in [`check_smoke.py`](check_smoke.py) agrees on 384/384, including clean scores. |
| Determinism | A repeated clean pass differs by exactly 0. |
| V32 conversion | 406,957,056 vision parameters converted to fp32 and restored. The bf16 score after restore equals the score before, exactly. |
| Memory (M1) | Peak process footprint **9.45 GiB**, below the 10 GiB abort. Minimum available memory 3.93 GiB. Swap grew 0. Pressure was "warn" for a single 0.5 s sample out of 29,939 and "normal" otherwise. No abort. |
| Preload | 11.71 GiB available, above the 9.91 GiB line. On this occasion the frozen 1.5 guard would also have permitted bf16. The guard was not edited. |
| Not attempted | Full fp32: the frozen guard needs 20.87 GB, and the guard was not lowered. |

Deviations from the plan:
- The log-log slopes in §6.1 were not computed. Medians, CIs and IQRs are reported instead.
- The B+P comparison covers margins and first-order ratios, but not separate B+P line scans. The fp32 projection is a readout hook, so these would equal the line-scan readouts below.
- Nothing else was changed after data were seen.

---

## 2. FGSM vs PGD: what explains the difference

### Observations

**2.1 Damage on the same ball.** These are official margin drops (positive means toward an error):

| | Neutral source | Helpful source |
|---|---|---|
| FGSM | 3.13 (2.25–4.50) | 3.63 (2.44–4.50) |
| PGD-3 | 10.25 (8.50–11.25) | 10.50 (9.00–11.25) |
| PGD-7 (S16) | 17.75 (14.75–18.75) | 16.13 (15.00–18.25) |
| PGD-3 − FGSM, paired | **6.25 (5.00–7.13)** | **6.25 (5.50–6.75)** |
| PGD-3 / FGSM, paired | **2.55 (2.30–3.28)** | **2.67 (2.31–3.32)** |
| Random sign, \|change\| | 0.5 (0.25–0.75) | — |

FGSM is about 6× the random-sign noise floor, so its gradient carries signal. The ratio matches the post-hoc Track B estimate from the frozen records (2.75×).

**2.2 The first-order model fails, and it inverts the ranking.** Realized margin drop divided by the drop that the clean gradient predicts (median, CI):

| | n→n | h→h | V32 n→n |
|---|---|---|---|
| FGSM (ε) | 0.023 (0.015–0.029) | 0.023 (0.019–0.029) | 0.018 (0.015–0.032) |
| PGD-1 (ε/4) | 0.098 (0.073–0.111) | 0.092 (0.076–0.110) | 0.096 (0.056–0.121) |
| PGD-3 | 0.27 (0.22–0.29) | 0.21 (0.19–0.29) | 0.24 (0.19–0.28) |
| PGD-7 | 0.49 (0.30–0.86) | 0.47 (0.29–0.71) | — |

On the rounded perturbations, the linear model predicts that PGD-3 achieves 0.28× FGSM's drop (CI 0.27–0.29). In reality it achieves 2.5×. **The linear model ranks FGSM above PGD-3 in 47/48 neutral scenes and 48/48 helpful scenes.** The fp32-projection version of the ratio is the same to two digits in every cell. The NLL version is worse behaved: PGD-3 reaches 3.3× its prediction, because softmax saturation makes the NLL convex near a confident answer. For this reason the margin is the primary scale.

**2.3 Where the FGSM budget goes (line scan along the FGSM direction, float scores).** The table gives damage at t·ε divided by damage at ε:

| t | 1/8 | 1/4 | 1/2 | 3/4 | 1 |
|---|---|---|---|---|---|
| B, neutral | 0.42 (0.30–0.50) | **1.00 (0.89–1.37)** | **1.21 (1.11–1.42)** | 1.10 (1.00–1.27) | 1 |
| B, helpful | 0.36 | 1.00 | 1.13 | 1.00 | 1 |
| V32, neutral | 0.74 (0.64–1.06) | 0.97 | 1.28 (1.13–1.61) | 1.10 | 1 |

Damage along the FGSM direction is **non-monotone**. It reaches its full value by ε/4 and peaks near ε/2. Beyond ε/2, extra movement along the clean gradient sign *undoes* damage. The PGD-1 first-order ratio and the t = 1/4 line point coincide (0.0977 in both), which is a built-in consistency check: they are the same image.

**2.4 Gradient rotation along the PGD path (neutral, bf16).**

| Step k | cos(g_k, g_0) | Sign agreement with g_0 | ‖g_k‖₁ / ‖g_0‖₁ | Local realized / predicted NLL |
|---|---|---|---|---|
| 1 | **0.002 (−0.007 to 0.008)** | 0.535 | 5.6 | 0.27 |
| 2 | −0.000 | 0.537 | 15.8 | 0.16 |
| 3 (S16) | 0.004 | 0.524 | 48 | 0.10 |
| 6 (S16) | 0.004 | 0.516 | 63 | 0.04 |

After one sign step of L∞ = 1/1020, **the gradient is essentially orthogonal to the clean gradient**, with 53% sign agreement against 50% for chance. The same holds with an fp32 vision tower (cos −0.010, CI −0.030 to 0.006) and in the helpful context. The step-to-step cosine becomes slightly *negative* from step 3 onward (about −0.03 to −0.06; CIs exclude 0 at steps 3, 4 and 6, not at 5), which is a mild sign of oscillation. The growth in gradient norm is expected for an NLL leaving saturation; the direction change is not.

**2.5 The official PGD-3 perturbation is a masked FGSM, by construction.** Three sign steps of ε/4 sum to ±ε/4 or ±3ε/4 on each coordinate. The 8-bit round trip maps these to 0 or ±1/255. The full argument is in [`MASK_CONTROL_PLAN.md`](MASK_CONTROL_PLAN.md) §2 and §9. In short, a coordinate survives rounding exactly when all three step signs agree, and its sign is then the first step's sign, which is FGSM's. **This holds for any second and third steps, including random ones.** So "subset of FGSM with no sign reversal" is a structural property of 3 × ε/4 steps with ε = one 8-bit level. It is not evidence that PGD selects coordinates. With independent random steps 2–3, the subset would be a uniform random ~25% of FGSM's support.

One exact exception was found when the replay was verified, and it is described in plan §9. At most 7 coordinates per unit, where the clean gradient is exactly 0 (FGSM leaves them alone), move by exactly ε/2. Half-to-even rounding turns that into one level at odd 8-bit values. This affects under 0.005% of PGD's support, and there are no sign reversals anywhere.

After rounding:

- 70.1% of coordinates are 0 and 29.9% are at ±ε;
- **sign disagreement with FGSM is 0.000 in every scene**, because the first step *is* the FGSM sign;
- L2 is 1.65, against FGSM's 3.01.

So the official PGD-3 image is, apart from those ≤ 7 tie coordinates, the FGSM image with 70% of its changed coordinates reverted to clean. The kept 30% are *not* the coordinates where the clean gradient is largest: the linear prediction for the kept set is 0.28 of FGSM's, slightly *below* the 0.30 that a random 30% would give. The kept set is about what three independent signs would agree on (25%), plus a small excess (consistent with the 53% step-to-step sign agreement).

### Interpretation (hypotheses, not established)

The observations fit one picture. The margin, as a function of the image, behaves like a **weak smooth trend under a large, rapidly varying component**:

1. The clean gradient's *sign* finds the smooth trend well enough to beat random by about 6× in damage. In about 590k input dimensions, a gradient whose trend component is small relative to the rough component still gives a sign vector that is well correlated with the trend.
2. The rough component changes completely within 1/1020 in L∞ (§2.4). The first-order prediction counts the rough component as damage that FGSM never realizes, which explains why FGSM achieves only 2% of its prediction.
3. Along any fixed sign direction, damage saturates and then reverses within about ε/2 (§2.3). PGD gains by **re-measuring the gradient and moving in fresh directions**, not by refining one direction.
4. *(Superseded by §2b.)* This paragraph originally offered two candidate explanations for the 2.5×: sparsity, or the agreement-based coordinate selection acting as an average. The random-mask control now shows the gain is almost entirely selection: sparsity alone gives 8% of it. The "weak trend under a rough component" picture is *compatible* with that result, but the control does not test it. The gradient decorrelation is not evidence for it either.

The source of the rough component is also open. It could be bf16 rounding inside the language model, the bilinear resize and patching in preprocessing, or genuine high-frequency structure (activation kinks, attention). §3 rules out the vision tower and the output projection as its *main* source.

---

## 2b. Random-mask control: sparsity or adaptive selection? (Checkpoint 2.3b)

Pre-registered in [`MASK_CONTROL_PLAN.md`](MASK_CONTROL_PLAN.md) (commit 107f087, amended before any result in f6f8b36 and c7e5f10). The setup:

- **Units:** 48 scenes × 2 source contexts = 96 units; the scene is the statistical unit.
- **Masks:** for each unit, the FGSM signs restricted to exactly k = |PGD-3 support| coordinates, chosen in three ways:
  - **PGD's own mask** (the official PGD-3 image);
  - **8 uniformly random masks** from FGSM's support, with pre-determined seeds;
  - **top-k by |clean NLL gradient|**.
- **Scoring:** the frozen official 8-bit path, in both contexts.
- **Baseline:** dense FGSM is read from the 2.3 records.

**Integrity.**
- The re-scored PGD-3 images are **bit-exact against the recorded 2.3 scores in 96/96 units**.
- The first smoke attempt failed this check. It rebuilt PGD-3 as x + int8 perturbation, which flips half-level ties. The cause was found and fixed by replaying PGD with the frozen update, and the failed attempt is kept in `masks_smoke_attempt1/`.
- Runtime: 28 min. Peak footprint: 5.2 GiB. Pressure: normal in all 3,326 samples. Swap growth: 0. No unit was skipped.

**Margin drop on the source context.** Each cell is the mean over 48 scenes with its CI. "Rand" is the mean of the 8 seeds within each unit.

| | Neutral source | Helpful source | Pooled |
|---|---|---|---|
| Dense FGSM | 3.36 (2.72–3.99) | 3.67 (3.09–4.29) | 3.52 (2.94–4.10) |
| Random mask, same k | 4.01 (3.38–4.65) | 3.99 (3.46–4.55) | 4.00 (3.46–4.56) |
| Top-k \|g\| mask | 2.36 (1.84–2.87) | 2.45 (1.93–3.01) | 2.41 (1.91–2.91) |
| **PGD-3 mask** | **9.81 (8.75–10.82)** | **9.91 (8.97–10.87)** | **9.86 (8.98–10.74)** |
| Sparsity: rand − FGSM | 0.65 (0.19–1.11) | 0.31 (−0.11 to 0.73) | 0.48 (0.07–0.89) |
| **Selection: PGD − rand** | **5.80 (5.04–6.56)** | **5.93 (5.19–6.62)** | **5.87 (5.24–6.50)** |
| Magnitude: top-k − rand | −1.65 (−2.17 to −1.15) | −1.54 (−1.94 to −1.14) | −1.59 (−2.03 to −1.18) |
| **Share of PGD−FGSM gap from sparsity** | 0.10 (0.03–0.17) | 0.05 (−0.02 to 0.11) | **0.08 (0.01–0.14)** |

**Per unit.** PGD's mask beats all 8 random masks in 48/48 neutral units and 47/48 helpful units. Its median rank among the seeds is 1.0. The seed-to-seed SD of a random mask's damage is about 0.3 within a unit, against a PGD − random gap of about 5.9.

**Attack success.** Flip rate on the 45 clean-correct scenes, source context:

| | FGSM | Random mask | Top-k \|g\| | PGD-3 mask | PGD − rand, paired |
|---|---|---|---|---|---|
| Neutral | 0.16 | 0.28 | 0.13 | **0.60** | +0.32 (0.19–0.46) |
| Helpful | 0.24 | 0.28 | 0.11 | **0.69** | +0.41 (0.26–0.57) |

**Cross-context target damage.** The same pattern holds. Pooled selection effect: 5.32 (4.68–5.97). Sparsity share: 0.05 (−0.02 to 0.12). Top-k − rand: −1.42. PGD − rand target flips: +0.41 to +0.44. The coordinates PGD selects under one prompt are also the damaging ones under the other.

**What the clean linear model expects.** The predicted margin drop is −g_M · r with the clean margin gradient. Relative to a random mask of the same size:
- PGD's mask is predicted to do **0.92×** (CI 0.90–0.94 neutral, 0.90–1.01 helpful);
- the top-k mask is predicted to do **2.79×**.

In reality, PGD's mask does 2.5× random, and top-k does 0.6×. The ordering is inverted again: by the clean model's standard, the coordinates PGD keeps are slightly *less* important than average.

### Reading against the pre-registered rule

Pooled sparsity share = 0.08, below the 0.25 threshold, so this is **"selection primarily"**. The criterion for "PGD selects better than chance" is also met:
- the pooled selection CI excludes 0;
- the effect is positive in both sources.

The criterion for "magnitude selection beats chance" is met in the *negative* direction: the clean-gradient magnitude is **anti-informative**.

### What this does and does not establish

- **Established, on these 48 scenes:** the advantage of official PGD-3 over FGSM on this model comes from **which** FGSM coordinates are kept, not from keeping fewer of them. A random subset of the same size is about as damaging as dense FGSM, with a small benefit at most. The agreement-based subset that PGD's later steps produce is about 2.5× as damaging. The subset that the clean gradient's magnitude ranks highest is *less* damaging than random.
- **Not established:**
  - *why* the agreement subset is damaging;
  - whether the individual coordinates are special, or only the set acting together;
  - whether the information comes from g₁ and g₂ as directions or from something incidental to the path.

  The one-step gradient decorrelation is consistent with "the later gradients carry information the clean gradient lacks", but it does not prove a mechanism, and none is claimed.
- **Scope.** This is not a statement about PGD in general. The subset structure depends on 3 steps of ε/4 with ε equal to one 8-bit level. Other step counts or sizes need not produce a subset, and were not tested here.

---

## 3. How much bf16 precision matters

### Observations

**3.1 Output projection (B vs B+P).** Over 2,816 scored evaluations, the bf16 margin and the fp32-projected margin differ by 0.072 at the median (CI 0.068–0.076; maximum 0.243), which is below one quantum of 0.125. The sign of the margin differs in 0.9% of evaluations (CI 0.6–1.2%), all of them near zero margin. Every first-order ratio in §2.2 is the same under the fp32 projection. **Output precision explains none of the FGSM/PGD effects.**

**3.2 Input survival (Q0, from the plan).** Fraction of coordinates that a perturbation of t·ε actually changes after the bf16 cast: 0.15 at t = 1/8, 0.55 at t = 1/4, 1.00 from t = 1/2. This explains the one place where the vision tower's precision changes the line scan: at t = 1/8, V32 (which keeps fp32 pixel values) gets 0.74 of FGSM's damage, while bf16 gets 0.42. At the attack's actual step sizes, ε/4 and ε, the inputs survive the cast.

**3.3 Gradient direction (S16, the same image, bf16 vs fp32 vision tower).**

| Gradient | cos(B, V32) | Sign agreement | ‖·‖₁ ratio |
|---|---|---|---|
| NLL, neutral | **0.29 (0.15–0.40)** | 0.75 | 1.04 |
| NLL, helpful | 0.32 (0.18–0.56) | 0.75 | 1.04 |
| Margin, neutral | 0.36 (0.20–0.56) | 0.77 | 1.10 |

For comparison, the *same-precision* gradients for the two text contexts have cosine **0.87** (0.81–0.92). Changing the vision tower from bf16 to fp32 rotates the image gradient **more than changing the prompt does**.

**3.4 Attack outcome is precision-insensitive (S16, paired).**

| | V32 − B, official margin drop |
|---|---|
| FGSM n→n | −0.75 (−1.25 to −0.06) |
| PGD-3 n→n | −0.31 (−0.88 to 0.56) |
| PGD-3 n→h | −0.06 (−0.75 to 0.75) |
| B-crafted PGD-3, scored under V32 minus scored under B | 0.06 (−0.25 to 0.19) |
| V32-crafted PGD-3, scored under B minus scored under V32 | 0.19 (−0.38 to 0.50) |

The random-sign noise floor is 0.5 in B and 0.25 in V32. FGSM crafted with the fp32 vision gradient is slightly *weaker*. PGD-3 is unchanged, and attacks move between precisions without any measurable loss. In V32, the first-order ratios (FGSM 0.018, PGD-3 0.24), the one-step gradient decorrelation, and the line-scan shape above t = 1/8 all reproduce the bf16 pattern.

### Interpretation

Vision-tower bf16 rounding adds a large, input-specific component to the gradient. That component does not affect the margin on the scale of an ε-ball. The attack-relevant part is shared across precisions, and a gradient cosine of about 0.3 is enough to find it. **bf16 in the vision tower is therefore not the explanation for the FGSM/PGD gap.** The bf16 language model is untested: full fp32 would need 20.9 GB, and the frozen guard was not lowered. The one-step decorrelation could still come from the language model's rounding.

The practical consequence is a warning about **gradient-level diagnostics**: saliency maps, gradient similarity between prompts, and gradient-based attributions. On this model they depend on numerical precision more than on the prompt, while outcome-level measures do not.

---

## 4. Is transfer asymmetric between neutral and helpful contexts?

### Observations (S48, PGD-3)

| | n→h (neutral-crafted, scored on helpful) | h→n |
|---|---|---|
| Source margin drop | 10.25 (8.50–11.25) | 10.50 (9.00–11.25) |
| Source flip rate | 0.56 (0.42–0.71) | 0.69 (0.56–0.81) |
| Conditional transfer (target flips / source flips) | **27/27 = 1.00** | **28/33 = 0.85** (0.73–0.97) |
| Target margin drop | 8.63 (7.75–9.63) | 10.13 (8.25–11.00) |
| Transfer efficiency (target / source drop) | 0.83 (0.79–0.94) | 0.97 (0.88–1.07) |
| First-order predicted efficiency (clean gradients) | 0.85 (0.77–0.92) | 0.90 (0.81–0.98) |

Paired, n-source minus h-source:
- source drop: 0.5 (−0.75 to 1.0), so the two sources are equally strong;
- **target drop: −1.13 (−2.0 to −0.5)**;
- efficiency: −0.16 (−0.28 to 0.04);
- first-order predicted efficiency: −0.07 (−0.23 to 0.17).

At **matched source drop**, interpolated along each float PGD path (paired n−h target drop):

| Source drop | n_pairs | n-source target drop | h-source target drop | Paired n − h |
|---|---|---|---|---|
| 2 | 47 | 1.76 | 2.00 | −0.22 (−0.51 to 0.15) |
| 4 | 45 | 3.50 | 4.00 | −0.67 (−0.99 to 0.04) |
| 6 | 41 | 4.92 | 6.23 | **−1.35 (−1.88 to −0.61)** |

Clean margins are similar: neutral 7.75 and helpful 6.75 at the median, with a paired difference of −0.125 and helpful smaller in exactly half the scenes.

### Interpretation

- **The continuous measures show a modest asymmetry.** Attacks crafted under the helpful context carry over to the neutral context almost completely (about 97%). Attacks crafted under the neutral context lose about 15% on the helpful context. The gap is absent for small perturbations, where the clean gradients predict similar transfer (cosine 0.87 between context gradients, and no difference in first-order predicted efficiency). It appears as the path moves away from the clean point, so it is a **path-dependent rather than first-order effect**.
- **The binary measure points the other way.** Every neutral flip also flipped the helpful answer, but 5 of 33 helpful flips did not flip the neutral answer. This mostly reflects the helpful context being easier to flip overall (69% vs 56% source flip rate on the same-strength attack). Binary transfer rates mix "how much damage transferred" with "how close to the boundary the target started". This is why the plan made continuous damage primary.
- The asymmetry is **not large**. Its CIs come from 48 scenes, and many contrasts were examined without multiplicity correction. The 1.35-unit gap at matched drop 6 is the strongest single number, and it should be checked on a fresh split before it is relied on.
- A **safety-relevant reading** that does *not* depend on the asymmetry: adding a helpful, non-revealing instruction does not protect the model. Attacks crafted under it, or against it, transfer at 83–97% efficiency, and the helpful context flips at least as often as the neutral one.

---

## 5. What is robust, what is surprising, what matters for safety

**Robust.** These hold in both contexts and in both precision modes, with CIs far from the null and paired within scenes:
1. PGD-3 does about 2.5× FGSM's damage on the same ε-ball, and the clean linear model ranks them the wrong way round in at least 98% of scenes.
2. FGSM achieves about 2% of its first-order predicted margin drop, and PGD-3 about 25%.
3. One ε/4 step along FGSM's direction already achieves all of FGSM's damage.
4. The gradient is uncorrelated with the clean gradient after one ε/4 step.
5. The official PGD-3 perturbation is a 30%-sparse subset of FGSM, with identical signs. This is structural (§2.5).
8. **Selection, not sparsity (2.3b, pre-registered):** a random subset of FGSM of exactly PGD's size recovers 8% of PGD's advantage, and PGD's subset beats all 8 random subsets in 95/96 units. This holds for source damage, cross-context damage and flips.
9. **The clean-gradient magnitude is anti-informative:** top-|g| subsets do worse than random, although the linear model predicts 2.8× more.
6. Output-projection precision is irrelevant to all of the above.
7. The vision-tower precision rotates gradients (cosine about 0.3) but does not change attack damage or transfer.

**Surprising.**
- The decorrelation scale: the gradient "forgets" its direction within 1/1020 in L∞, in about 590k dimensions, yet sign steps still make steady progress.
- Keeping the *right* 30% of FGSM's coordinates makes it 2.5× more damaging, while keeping a random 30% changes almost nothing. The "right" coordinates are the ones where two further gradient steps happen to agree. To the clean gradient they look slightly less important than average, and the coordinates it ranks highest are worse than random.
- Precision changes the gradient more than the prompt does.

**Relevant to safety.**
- **Evaluation.** Robustness numbers from single-step attacks, or from gradient-norm and linearization arguments, understate this VLM's vulnerability by a large and *scene-inconsistent* factor. They also mis-order perturbations, so they can't be used even for ranking. Multi-step attacks scored after realistic quantization should be the minimum.
- **Defense claims.** Precision differences cause gradient mismatch without any robustness gain. A defense that changes numerical precision, or that makes gradients look different, should be expected to show apparent robustness against single-step or transfer-from-gradient tests that disappears under PGD. This is the familiar gradient-masking failure, here produced by bf16 rather than by design.
- **Saliency-guided defenses and evaluations.** Methods that rank coordinates by clean-gradient magnitude would protect, monitor or attack the wrong coordinates here: top-|g| is worse than random. The damaging coordinate set becomes visible only after taking further gradient steps.
- **Interpretability.** Gradient-based attributions on bf16 VLMs are dominated by a component that depends on precision and on the exact input. Conclusions drawn from them need a precision-robustness check.
- **Prompt-side mitigations.** A helpful, non-revealing system instruction does not reduce, and does not isolate, image-borne attacks. Transfer between the two contexts is high in both directions.

---

## 6. Limitations

- **One model, one size, one hardware stack.** Qwen3-VL-2B on Apple MPS in bf16. MPS kernels differ from CUDA, and the rough gradient component may be specific to this stack.
- **Synthetic color-naming scenes**, 48 development groups and one arm each. The two contexts are neutral and helpful-non-revealing. These are not open-ended or harmful-content tasks.
- **The language model stays bf16 in all modes.** "Precision doesn't matter" is established only for the vision tower and the output projection.
- **Controls.** The random-mask control (§2b) separates sparsity from selection, but only at PGD-3's own sparsity. There was no smaller-step gradient-decorrelation curve, so the decorrelation length scale is only bounded above by ε/4. No experiment here identifies *why* PGD's subset is damaging.
- **A reconstruction pitfall, caught.** Rebuilding iterates from int8 storage is not bit-exact at half-level ties. The pre-registered bit-exact check caught it, and the analysis uses replayed iterates.
- **Small subsets.** PGD-7 and V32 use 16 scenes, and their CIs are wide.
- **Multiplicity and post-hoc framing.** Many contrasts were examined without multiplicity correction. The interpretive paragraphs were written after seeing the data. Every claim here is hypothesis-generating. Confirmation needs the reserved split `study2-trackB-confirm-v1`, which has not been drawn.
- **Quantized margins.** bf16 logits move in steps of 0.125, so small drops (FGSM, the random controls) are coarse.
- **Memory history.** The original EMP-1 smoke criterion (peak ≤ 7.5 GiB) **failed** at 8.41 GiB and stays recorded as failed in [`MEMORY_ASSESSMENT.md`](MEMORY_ASSESSMENT.md) §6. This run used the later M1 calibration (§7 there), which the research lead approved before the run. M1 is not a retroactive pass of the original criterion. Monitoring could not rule out a transient OOM; none happened.

---

## 7. Next experiments (not run, not approved)

The random-mask control is done (§2b). The rest are small, on the same machine.

1. **Confirmation on fresh scenes (most valuable for credibility; see §9).**
2. **Decorrelation curve.** cos(g(x + t·sign g₀), g₀) for t from ε/64 to ε/4. This finds the length scale of the rough component and checks it against the bf16 ulp scale.
3. **Language-model precision.** Run one or two scenes with the language model in fp32 on CPU (slow but feasible). This tests whether the language model's rounding is the rough component.
4. **Gradient smoothing.** Average the gradient over a few nearby points before taking the sign. If the "weak trend + rough component" picture holds, a single smoothed step should close much of the FGSM/PGD gap.
5. **Which step carries the selection?** Use the stored iterates, plus masks where step 2 or step 3 is replaced by a random sign, and score the resulting subsets. This is mostly forward passes and takes under an hour. It tests whether the damaging subset comes from g₁, from g₂, or only from their agreement.
6. **A second model.** Repeat the minimal design on another local VLM: dense FGSM, PGD-3, and the matched random-mask control. This tests whether the selection effect is a property of Qwen3-VL on MPS or of sign-step attacks under 8-bit storage.

---

## 8. Reproducing

```bash
cd repo
export PYTHONPATH=$(ls -d src/*/ | tr '\n' ':')
PY=/path/to/venv/bin/python   # torch 2.15.0, transformers 5.22.1

# Full run (~4.3 h on a 24 GB M-series Mac; needs ≥ 9.91 GiB available and normal memory pressure at start)
caffeinate -i $PY -m path_probe run --memory-policy emp1

# Random-mask control, 2.3b (~30 min; reads paths/tensors, writes masks/)
caffeinate -i $PY -m path_probe masks

# Integrity checks and analysis (read-only, seconds)
$PY docs/checkpoint-2.3-exploratory/check_smoke.py paths
$PY docs/checkpoint-2.3-exploratory/analyze_paths.py paths
$PY docs/checkpoint-2.3-exploratory/analyze_masks.py masks
$PY -m pytest tests/path_probe -q
```

Outputs:
- [`paths/records.jsonl`](paths/records.jsonl): 1,632 rows.
- [`paths/progress.json`](paths/progress.json): run state, memory monitor timeline, and limitations.
- [`paths/run_check.json`](paths/run_check.json).
- [`analysis/analysis.json`](analysis/analysis.json): every 2.3 number in this report.
- [`masks/records.jsonl`](masks/records.jsonl), [`masks/progress.json`](masks/progress.json), [`analysis/masks_analysis.json`](analysis/masks_analysis.json): the 2.3b control.

Perturbation tensors (313 MB, int8 multiples of ε/4) are kept locally and are not committed. The analysis does not need them. Code is in `src/path_probe/path_probe/`, and the frozen scoring and attack functions are imported unchanged.

---

## 9. The strongest claim the evidence supports

> **On Qwen3-VL-2B, a 3-step sign-gradient attack is 2.5× more damaging than FGSM because of *which* RGB channel coordinates it perturbs, not how many. Those coordinates are invisible to, and in fact ranked below average by, the model's clean gradient.**
>
> Under an 8-bit-stored L∞ attack with ε = 1/255 and steps of ε/4, PGD-3's perturbation is, by arithmetic, a ~30% subset of FGSM's signs.
> - A random subset of exactly the same size recovers only 8% (CI 1–14%) of PGD's advantage.
> - PGD's subset beats all 8 random subsets in 95 of 96 scene–context units.
> - It raises the flip rate from 28% to 60–69%, and the effect carries across prompts.
> - The subset that the clean gradient's magnitude ranks highest does worse than random.
> - Single-step attacks and first-order, saliency-style reasoning therefore misjudge both which attacks and which coordinates matter.

**Qualifiers that must travel with it:**
- exploratory;
- 48 development scenes of one synthetic color task;
- one 2B model in bf16 on Apple MPS;
- one ε and step grid.

The structural part (subset, no sign reversal) is arithmetic and holds anywhere. The empirical part (selection ≫ sparsity, magnitude anti-informative) is shown only here. Nothing here establishes *why* the subset is damaging. The gradient decorrelation is a consistent observation, not an explanation.

**Why it is interesting for safety.** It is a concrete, quantified case where the cheapest evaluation and attribution tools give the wrong answer, and specifically the *wrong ranking*, not just a noisy one. A pipeline that ranks coordinates by clean-gradient saliency picks coordinates that are worse than random. A robustness evaluation based on FGSM or linearization would rank perturbations backwards. The finding is cheap to reproduce: 48 scenes, one laptop, a bit-exact pipeline check against frozen records.

**The next experiment that would most improve credibility before 16 November: a pre-registered confirmation on scenes never used before.**

| | |
|---|---|
| Fixed in advance | The 2.3b claim, as directional predictions: sparsity share ≤ 0.25, PGD − random > 0 in both sources, top-k − random < 0. |
| Run | Dense FGSM, PGD-3, and the matched mask control. |
| Compute | About 1–1.5 h locally. |
| Scenes | Not available today. The only fresh scenes are the reserved split `study2-trackB-confirm-v1` (not yet drawn), or a newly generated development batch. Either needs the research lead's approval, because it uses samples not touched before. |
| What it changes | Every number here comes from the scenes on which the question was formed. A successful directional confirmation turns "exploratory" into "confirmed on held-out data" for the headline claim, which is the change a reviewer will weigh most. |

A second-model replication (§7, item 6) would do more for *generality*. It needs porting the scoring pipeline, and it would not finish with the same rigor in the time available.

> **Update, 2026-11-09 (Checkpoint 2.4).** This confirmation has now been run, pre-registered on a newly generated split (`study2-trackB-confirm-v1`, 48 groups) with the protocol hash-frozen before inference. All three directional predictions were supported. Selection (PGD − random) was +6.50 logits [6.08, 6.92]. The sparsity share was 0.03 [−0.02, 0.09]. Top-|∇| minus random was −1.38 [−1.71, −1.06]. PGD's subset beat all 8 random seeds in 96/96 units. The headline claim above is therefore confirmed on held-out scenes, with the same qualifiers on model, task family, ε and precision. The text above is left as written. See [`../checkpoint-2.4-confirm/REPORT.md`](../checkpoint-2.4-confirm/REPORT.md).
