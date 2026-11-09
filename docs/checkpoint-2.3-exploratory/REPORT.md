# Checkpoint 2.3: Why single-step attacks fail on a bf16 VLM, and what precision does and does not change

**Status: exploratory, development data only, descriptive.** Nothing here confirms a hypothesis. The frozen Checkpoint 2.1 results and the preregistered H\* are unchanged. H-ΔL is post hoc and is not tested here. No sealed or new samples were used.

Model: Qwen3-VL-2B-Instruct, bf16 on Apple MPS, eager attention. Attack: the frozen Checkpoint 2.0 attack: L∞ ε = 1/255, sign steps of ε/4, on the labeled-color summed NLL. Official scores use an 8-bit round trip. Scenes: arm a of all 48 development groups (S48). A fixed 16-scene subset (S16) also gets PGD-7 and the fp32 vision tower. Plan: [`EXPLORATORY_PLAN.md`](EXPLORATORY_PLAN.md), commit 0c6f176, written before any full-run data existed.

Intervals are 95% percentile bootstraps over scenes: 9,999 draws, seed 20261109, one scene per group. "Paired" means the difference is computed within each scene before bootstrapping. Margin is z_label − z_best-other in bf16 logits, which are quantized to steps of 0.125. Three scenes (dev4-ref-20, 26, 31) are not clean-correct in both contexts; they stay in continuous measures and drop out only of ratios that need a positive clean margin.

---

## 0. Summary

| Question | Answer, on these 48 scenes | Strength |
|---|---|---|
| Why does PGD-3 beat FGSM on the same ε-ball? | FGSM spends most of its budget past the point of useful damage. One ε/4 step in FGSM's direction does as much damage as the full ε step (ratio 1.00, CI 0.89–1.37). The gradient becomes **uncorrelated with the clean gradient after a single ε/4 step** (cosine 0.002, CI −0.007 to 0.008). After the 8-bit round trip, **the official PGD-3 perturbation is FGSM with 70% of its pixels reverted** (sign disagreement with FGSM 0.000). That sparser perturbation does **2.55× the damage** (CI 2.30–3.28). | Robust observation. The mechanism is a hypothesis. |
| Does the clean linear model rank attacks correctly? | No. It predicts that FGSM beats PGD-3 by 3.6×, and it ranks them the wrong way round in **98% of neutral scenes and 100% of helpful scenes**. FGSM achieves 2.3% of its first-order predicted margin drop; PGD-3 achieves 27%. | Robust |
| How much does bf16 matter? | The output projection does not matter: fp32 projection moves margins by 0.07 (median), and every first-order ratio is unchanged. The vision-tower precision **changes the gradient direction a lot**: the clean-gradient cosine between bf16 and fp32 vision is **0.29–0.36**. That is less than the 0.87 cosine between the two text contexts. **It does not change the attack outcome**: PGD-3 damage differs by −0.31 (CI −0.88 to 0.56), and attacks cross-evaluate between precisions with no loss. The FGSM/PGD gap and the one-step decorrelation both persist with an fp32 vision tower. | Robust for the vision tower. The bf16 language model is untested. |
| Is neutral→helpful transfer asymmetric? | Slightly, and only at higher damage. Both sources are equally strong (paired difference 0.5, CI −0.75 to 1.0). Helpful-crafted attacks hurt the neutral context **1.1 margin units more** than the reverse (CI 0.5–2.0). At matched source damage the gap grows with strength: 0.22 at a drop of 2, 0.67 at 4, and **1.35 at 6 (CI 0.61–1.88)**. Binary flip-based "conditional transfer" points the **opposite** way (100% vs 85%). | Modest. The continuous measure and the binary measure disagree. |

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

**2.5 The official PGD-3 perturbation is a masked FGSM.** Three sign steps of ε/4 sum to ±ε/4 or ±3ε/4 on each coordinate. The 8-bit round trip maps these to 0 or ±1/255. After rounding:

- 70.1% of coordinates are 0 and 29.9% are at ±ε;
- **sign disagreement with FGSM is 0.000 in every scene**, because the first step *is* the FGSM sign;
- L2 is 1.65, against FGSM's 3.01.

So the official PGD-3 image equals the FGSM image with 70% of its pixels reverted to clean. The kept 30% are *not* the coordinates where the clean gradient is largest: the linear prediction for the kept set is 0.28 of FGSM's, slightly *below* the 0.30 that a random 30% would give. The kept set is about what three independent signs would agree on (25%), plus a small excess (consistent with the 53% step-to-step sign agreement).

### Interpretation (hypotheses, not established)

The observations fit one picture. The margin, as a function of the image, behaves like a **weak smooth trend under a large, rapidly varying component**:

1. The clean gradient's *sign* finds the smooth trend well enough to beat random by about 6× in damage. In about 590k input dimensions, a gradient whose trend component is small relative to the rough component still gives a sign vector that is well correlated with the trend.
2. The rough component changes completely within 1/1020 in L∞ (§2.4). The first-order prediction counts the rough component as damage that FGSM never realizes, which explains why FGSM achieves only 2% of its prediction.
3. Along any fixed sign direction, damage saturates and then reverses within about ε/2 (§2.3). PGD gains by **re-measuring the gradient and moving in fresh directions**, not by refining one direction.
4. The quantization step then keeps only the coordinates where the three sampled directions agree. This acts like a crude average of the trend that discards much of the rough part. That may be why a perturbation that is 70% zeros does 2.5× FGSM's damage.

**What is not shown.** The obvious control was not run, because it is outside the pre-registered plan: an FGSM perturbation with a *random* 30% mask. Some of the gain may come from sparsity or a smaller norm alone. The ε/2 line point (L2 ≈ 1.5) already gives 1.21× FGSM, so a large part of the 2.55× might be "smaller is better along this direction". Only the remainder would be attributable to PGD's coordinate selection. The source of the rough component is also open. It could be bf16 rounding inside the language model, the bilinear resize and patching in preprocessing, or genuine high-frequency structure (activation kinks, attention). §3 rules out the vision tower and the output projection as its *main* source.

---

## 3. How much bf16 precision matters

### Observations

**3.1 Output projection (B vs B+P).** Over 2,816 scored evaluations, the bf16 margin and the fp32-projected margin differ by 0.072 at the median (CI 0.068–0.076; maximum 0.243), which is below one quantum of 0.125. The sign of the margin differs in 0.9% of evaluations (CI 0.6–1.2%), all of them near zero margin. Every first-order ratio in §2.2 is the same under the fp32 projection. **Output precision explains none of the FGSM/PGD effects.**

**3.2 Input survival (Q0, from the plan).** Fraction of coordinates that a perturbation of t·ε actually changes after the bf16 cast: 0.15 at t = 1/8, 0.55 at t = 1/4, 1.00 from t = 1/2. This explains the one place where the vision tower's precision changes the line scan: at t = 1/8, V32 (which keeps fp32 pixels) gets 0.74 of FGSM's damage, while bf16 gets 0.42. At the attack's actual step sizes, ε/4 and ε, the inputs survive the cast.

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
5. The official PGD-3 perturbation is a 30%-sparse subset of FGSM, with identical signs.
6. Output-projection precision is irrelevant to all of the above.
7. The vision-tower precision rotates gradients (cosine about 0.3) but does not change attack damage or transfer.

**Surprising.**
- The decorrelation scale: the gradient "forgets" its direction within 1/1020 in L∞, in about 590k dimensions, yet sign steps still make steady progress.
- A perturbation that is FGSM with 70% of its pixels *reverted* is 2.5× more damaging than FGSM.
- Precision changes the gradient more than the prompt does.

**Relevant to safety.**
- **Evaluation.** Robustness numbers from single-step attacks, or from gradient-norm and linearization arguments, understate this VLM's vulnerability by a large and *scene-inconsistent* factor. They also mis-order perturbations, so they can't be used even for ranking. Multi-step attacks scored after realistic quantization should be the minimum.
- **Defense claims.** Precision differences cause gradient mismatch without any robustness gain. A defense that changes numerical precision, or that makes gradients look different, should be expected to show apparent robustness against single-step or transfer-from-gradient tests that disappears under PGD. This is the familiar gradient-masking failure, here produced by bf16 rather than by design.
- **Interpretability.** Gradient-based attributions on bf16 VLMs are dominated by a component that depends on precision and on the exact input. Conclusions drawn from them need a precision-robustness check.
- **Prompt-side mitigations.** A helpful, non-revealing system instruction does not reduce, and does not isolate, image-borne attacks. Transfer between the two contexts is high in both directions.

---

## 6. Limitations

- **One model, one size, one hardware stack.** Qwen3-VL-2B on Apple MPS in bf16. MPS kernels differ from CUDA, and the rough gradient component may be specific to this stack.
- **Synthetic color-naming scenes**, 48 development groups and one arm each. The two contexts are neutral and helpful-non-revealing. These are not open-ended or harmful-content tasks.
- **The language model stays bf16 in all modes.** "Precision doesn't matter" is established only for the vision tower and the output projection.
- **Missing controls.** There was no random-mask FGSM at matched sparsity, so §2.5's "selection" story is not separated from a pure sparsity or norm effect. There was no smaller-step gradient-decorrelation curve, so the decorrelation length scale is only bounded above by ε/4.
- **Small subsets.** PGD-7 and V32 use 16 scenes, and their CIs are wide.
- **Multiplicity and post-hoc framing.** Many contrasts were examined without multiplicity correction. The interpretive paragraphs were written after seeing the data. Every claim here is hypothesis-generating. Confirmation needs the reserved split `study2-trackB-confirm-v1`, which has not been drawn.
- **Quantized margins.** bf16 logits move in steps of 0.125, so small drops (FGSM, the random controls) are coarse.
- **Memory history.** The original EMP-1 smoke criterion (peak ≤ 7.5 GiB) **failed** at 8.41 GiB and stays recorded as failed in [`MEMORY_ASSESSMENT.md`](MEMORY_ASSESSMENT.md) §6. This run used the later M1 calibration (§7 there), which the research lead approved before the run. M1 is not a retroactive pass of the original criterion. Monitoring could not rule out a transient OOM; none happened.

---

## 7. Next experiments (not run, not approved)

Each is small: one hour or less on the same machine, development scenes only.

1. **Random-mask FGSM control.** FGSM with random masks at 30% density, against PGD-3's mask. This separates sparsity from selection.
2. **Decorrelation curve.** cos(g(x + t·sign g₀), g₀) for t from ε/64 to ε/4. This finds the length scale of the rough component and checks it against the bf16 ulp scale.
3. **Language-model precision.** Run one or two scenes with the language model in fp32 on CPU (slow but feasible). This tests whether the language model's rounding is the rough component.
4. **Gradient smoothing.** Average the gradient over a few nearby points before taking the sign. If the "weak trend + rough component" picture holds, a single smoothed step should close much of the FGSM/PGD gap.

---

## 8. Reproducing

```bash
cd repo
export PYTHONPATH=$(ls -d src/*/ | tr '\n' ':')
PY=/path/to/venv/bin/python   # torch 2.15.0, transformers 5.22.1

# Full run (~4.3 h on a 24 GB M-series Mac; needs ≥ 9.91 GiB available and normal memory pressure at start)
caffeinate -i $PY -m path_probe run --memory-policy emp1

# Integrity checks and analysis (read-only, seconds)
$PY docs/checkpoint-2.3-exploratory/check_smoke.py paths
$PY docs/checkpoint-2.3-exploratory/analyze_paths.py paths
$PY -m pytest tests/path_probe -q
```

Outputs:
- [`paths/records.jsonl`](paths/records.jsonl): 1,632 rows.
- [`paths/progress.json`](paths/progress.json): run state, memory monitor timeline, and limitations.
- [`paths/run_check.json`](paths/run_check.json).
- [`analysis/analysis.json`](analysis/analysis.json): every number in this report.

Perturbation tensors (313 MB, int8 multiples of ε/4) are kept locally and are not committed. The analysis does not need them. Code is in `src/path_probe/path_probe/`, and the frozen scoring and attack functions are imported unchanged.
