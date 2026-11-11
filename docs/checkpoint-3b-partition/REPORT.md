# Checkpoint 3b report: partition-and-union pilot (2026-11-11)

**Status: EXPLORATORY, development data only.**
- The panel is the 16 `s16` scenes of `development_v4`, with neutral source and neutral scoring context.
- No confirmation data were used.
- Every interval below is conditional on this panel and on the specified randomization (8 random masks, 2 PGD splits, seeded quarters). None is a population inference.

**Provenance:**
- Plan: [`PLAN.md`](PLAN.md), committed in `7c1736b` before any inference.
- Analysis-input fix: `2822b31`, before the run. It requires exactly the frozen s16 panel; the estimator is unchanged.
- Run: `run/`, one shot, from `2822b31`.
- Analysis: [`analysis/partition_analysis.json`](analysis/partition_analysis.json), run once on the complete panel.

## 1. Results (fixed clean-competitor contrast, score units = logit differences)

The quantities:
- **A** is the full-mask advantage of PGD over random.
- **Q** is the summed advantage of the four quarters.
- **Δ = A − Q.**
- **I_P** and **I_R** are each family's full mask minus the sum of its quarters, so Δ = I_P − I_R.

Means are over 16 scenes, with paired two-sided 95% percentile scene-bootstrap intervals (9,999 draws, seed 20261111).

| Quantity | Mean [95% CI] | Direction | Scenes > 0 / < 0 |
|---|---|---|---|
| **I_P**, PGD full − Σ quarters | **−6.96** [−10.11, −4.31] | negative | 1 / 15 |
| **I_R**, random full − Σ quarters | **−4.75** [−6.43, −3.23] | negative | 0 / 16 |
| A | +4.75 [+3.59, +5.92] | positive | 16 / 0 |
| Q | +6.96 [+5.05, +8.95] | positive | 16 / 0 |
| **Δ** (primary) | **−2.21** [−4.10, −0.57] | negative | 6 / 10 |
| ρ = mean Q / mean A (secondary; denominator rule met) | 1.47 [1.12, 1.98] | — | — |

**Family means, for scale (descriptive):**

| | Full mask | Sum of its four quarters |
|---|---|---|
| PGD | 8.31 | 15.27 |
| Random | 3.56 | 8.31 |

## 2. Interpretation (exploratory)

**Which family accounts for the difference.**
- Both families are sub-additive at quarter scale: the full mask does less damage than its four quarters summed.
- This holds for every scene in the random family and for 15 of 16 scenes in PGD's.
- PGD's shortfall is larger, by 2.21 on average, and that is all Δ is.
- So the negative Δ is not random sets behaving unusually. **PGD's set is more sub-additive than matched random sets.** Its quarters together over-account for its full-mask advantage. On this panel, ρ puts Q at about 1.5 times A.

**What this does not establish** (PLAN §7).
- The two families sit at different damage levels: PGD's full mask does about 2.3 times the damage of random's.
- If the response to adding coordinates bends differently at different levels, a negative Δ can arise without anything specific to PGD's coordinate choice. This design cannot separate redundancy among PGD's coordinates from such a level-dependent response.
- The result is a net difference at one partition scale. Interactions inside a quarter are not seen, and opposite-signed interaction terms can cancel.
- The earlier Checkpoint 3 hypothesis was that PGD needs a particular joint combination. This result does not support the version in which PGD's advantage needs coordinates from different quarters together. **It does not establish redundancy either.**

## 3. Switching and the official margin

| Quantity | Switching term W | Official margin (= fixed + W) |
|---|---|---|
| A | +1.05 [+0.37, +1.87] | +5.80 [+4.65, +6.87] |
| Q | +0.97 [0.00, +2.32] | +7.93 [+5.98, +9.94] |
| Δ | **+0.08** [−0.81, +0.85] (unresolved) | **−2.13** [−4.44, −0.001] |
| I_P | −0.05 [−0.98, +0.78] | −7.01 [−10.35, −4.02] |
| I_R | −0.13 [−0.63, +0.19] | −4.88 [−6.50, −3.39] |

- Changes of the strongest rival add about 1 to A on the margin. They contribute essentially nothing to Δ.
- The margin's Δ has the same sign as the fixed contrast's, with an upper bound just below 0.
- Rival switches occur in 7 scenes. They are concentrated in `07`, `37` and `38` (PGD/random images switched: 9/26, 9/13 and 9/40); the other 4 scenes have 1–3 each.
- One scene (`38`) has a clean tie between rivals. Its c₀ came from the name rule (blue).
- The switching identity held exactly on every image and aggregate. The largest image-level discrepancy in the smoke was 0.0.

## 4. Scene-level variation (fixed contrast)

| Scene | k | c₀ | Clean margin | A | Q | Δ | I_P | I_R |
|---|---|---|---|---|---|---|---|---|
| 00 | 170,136 | green | 8.75 | 5.84 | 6.59 | −0.75 | −9.50 | −8.75 |
| 03 | 167,900 | yellow | 8.75 | 2.91 | 14.28 | −11.38 | −23.13 | −11.75 |
| 07 | 176,405 | blue | 2.25 | 1.75 | 6.19 | −4.44 | −8.13 | −3.69 |
| 12 | 187,980 | red | 9.75 | 5.38 | 5.97 | −0.59 | −2.50 | −1.91 |
| 16 | 171,200 | red | 5.25 | 8.84 | 8.56 | +0.28 | −3.75 | −4.03 |
| 20 | 177,481 | green | −5.25 | 1.56 | 10.16 | −8.59 | −15.69 | −7.09 |
| 25 | 165,695 | red | 13.25 | 4.92 | 1.91 | +3.02 | +0.75 | −2.27 |
| 27 | 174,631 | blue | 12.25 | 0.95 | 2.00 | −1.05 | −2.88 | −1.83 |
| 29 | 162,718 | red | 2.25 | 6.69 | 9.81 | −3.13 | −11.00 | −7.88 |
| 30 | 174,785 | red | 5.25 | 8.44 | 11.31 | −2.88 | −8.13 | −5.25 |
| 33 | 175,773 | red | 3.25 | 5.50 | 5.00 | +0.50 | −9.63 | −10.13 |
| 37 | 178,818 | red | 11.00 | 2.59 | 2.53 | +0.06 | −3.19 | −3.25 |
| 38 | 168,038 | blue | 11.25 | 3.19 | 2.06 | +1.13 | −0.25 | −1.38 |
| 39 | 166,853 | blue | 7.75 | 7.92 | 12.95 | −5.03 | −6.75 | −1.72 |
| 42 | 176,338 | green | 11.75 | 4.16 | 3.00 | +1.16 | −2.13 | −3.28 |
| 47 | 191,981 | yellow | 10.00 | 5.41 | 9.09 | −3.69 | −5.50 | −1.81 |

Notes on the table:
- Δ ranges from −11.38 to +3.02. Scene `03` alone contributes about a third of the summed Δ.
- Scene `20` is answered incorrectly when clean (its clean margin is negative). The plan includes it.
- 12 of the 16 scenes have k not divisible by 4, so the uneven-quarter path was exercised.

## 5. Integrity, runtime and memory

| Check | Result |
|---|---|
| Official PGD-3 image re-scored against the 2.3b record | bit-exact in **16/16** scenes |
| FGSM from the stored gradient against the stored iterate | equal in 16/16 |
| Sign agreement of rounded PGD-3 and FGSM on the retained overlap | 0 disagreements; max 1 level |
| Excluded tie coordinates (PGD support outside FGSM's) | 0–7 per scene (7, 3, 2, 4, 0, 3, 4, 4, 3, 2, 2, 1, 7, 1, 5, 2) |
| Score identities (stored margin and decision from the four scores) | held on all 800 image rows |
| Mask, split and 8-bit round-trip checks | passed for every masked image (the runner stops otherwise) |
| Final clean re-score of scene 1 (four scores, margin, decision) | identical |
| Forwards | 816 in scenes + 1 final = **817**, as planned; 0 backward |
| Runtime | 721.7 s (12.0 min); 42.7–48.3 s per scene; cap 45 min |
| Memory (EMP-1 / M1) | peak process footprint 5.59 GB; pressure normal in 1,421 of 1,421 samples; swap growth 0; minimum available 9.9 GB |

**Earlier smoke** (`run_smoke/`, from `7c1736b`): all historical replays were bit-exact, 53 forwards, 44.5 s.

## 6. Status

Exploratory pilot only. No confirmation protocol is proposed. Further experiments, cloud spending and merging are separate decisions.
