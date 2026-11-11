# Checkpoint 3b: partition-and-union pilot of PGD-3's selection advantage (development only)

Federico Cavaletto · drafted 2026-11-11 · branch `research/checkpoint-3b-partition`, from `main` at `e2eb037`

**Status: EXPLORATORY. Development data only. Written before any model inference for this checkpoint.**
- No confirmation data are used. The runner refuses any path naming `trackB_confirm`, and the 16 scenes must be `development` rows.
- No `confirm-v2` split is generated.
- No cloud resources and no spending.

**Author's decisions (2026-11-11):**
- candidate A;
- a two-sided primary analysis;
- quarters and full masks only;
- 8 random masks and 2 PGD partitions;
- neutral source and neutral scoring context only;
- all 16 development scenes `s16`.

## 1. Question

At ε = 1/255, PGD-3 keeps about 30% of FGSM's changed RGB channel coordinates. That subset does about +6 logits more source-context margin damage than random subsets of the same size (Checkpoint 2.4 §5; development 2.3b §2b).

This pilot asks whether that full-mask advantage equals the sum of the advantages of four disjoint random quarters of the masks, or differs from it. The quarters are measured against matched random quarters.

The answer is a **net difference at one partition scale** (quarters). It is not a per-coordinate decomposition.

## 2. Scorer facts (verified in code at `e2eb037`)

| Fact | Code |
|---|---|
| Colour score: s_c = −log softmax(z)[t_c]. z is the first answer-position logit row (bf16, cast to fp32, on CPU) over the full vocabulary; t_c is the colour's single locked token. **Units: nats.** Because the normaliser cancels, s_c − s_l = z_l − z_c, a logit difference. Here l is the label. | `attack_feasibility/score.py:94–123`; `dev_pilot/scoring.py:19–36`; `choice_eval/likelihood.py:34–38` |
| Official path: `round_trip_uint8`, then PIL, then processor, then model. | `score.py:108–112` |
| Margin: m = min over c ≠ l of s_c, minus s_l. | `qwen_rgb_loss/records.py:74–79` |
| **Tie rule.** The decision is the lowest s_c; equal scores break by colour name (blue, green, red, yellow). The margin value does not depend on ties. bf16 logits are discrete, so exact ties occur. | `choice_eval/decision.py:35–39` |
| `score_official` returns all four `summed_nll`. Earlier runners kept only `nll`, `margin` and `chosen`. **This runner keeps all four.** | `score.py:48–60`; `path_probe/probe.py:28` |
| The stored 2.3 `competitor` uses the float path and dict-order ties (`probe.py:39–41, 58`). **It is not used here.** | — |

## 3. Images, matching and randomization (per scene)

**FGSM and PGD.**
- x is the clean 8-bit image.
- r_F = round_trip(x + FGSM) − x, and S_F is its support. FGSM comes from the stored clean NLL gradient and must equal the stored FGSM iterate.
- PGD-3 is replayed from the stored int8 iterates (`masks.replay_pgd`, 3 steps).
- r_P is its rounded perturbation.

**The PGD mask.**
- **P = S_F ∩ supp(r_P)**, and k = |P|.
- On every scene the runner verifies, and stops otherwise, that:
  - r_P and r_F agree in sign on every coordinate of P;
  - no coordinate of r_P exceeds one 8-bit level.
- The excluded coordinates supp(r_P) \ S_F are counted and reported per scene. These are the rounding ties at zero clean gradient (2.4 §5.6; at most 7 per unit in 2.3b and 2.4).

**Images.** For any mask M, y(M) = x + r_F ⊙ 1_M.
- The runner checks that the 8-bit image of y(M) differs from x by exactly round(r_F · 255) on M and by 0 elsewhere.

**What is matched.**
- **Counts:** every full mask has k coordinates, and piece j has the same size in every family and split.
- **Absolute magnitudes:** every changed coordinate moves by one 8-bit level.
- **Signs are not matched across families.** Each selected coordinate keeps its own FGSM sign. The runner records the numbers of positive and negative changes for every mask and piece.

**Random masks.**
- R_s, s = 0…7, are uniform draws of k coordinates from S_F, seeded by `scene_seed("checkpoint-3b-partition|random|s", scene_id)`.
- **There are 8 seeds, fixed before the run.**

**Partition.**
- A mask's coordinates are put in a seeded uniform random order and cut into 4 consecutive blocks.
- Sizes are quarter_sizes(k) = ⌊k/4⌋ + [j < k mod 4] for j = 0…3. For example, k = 10 gives 3, 3, 2, 2.
- The runner checks that the pieces are disjoint, that their union is the mask, and that the sizes are as planned.
- **PGD's mask is split twice:** seeds `…|split|pgd|0` and `…|split|pgd|1`.
- **Each random mask is split once,** with its own seed `…|split|random|s`.

**Averaging within the scene.**
- Random quantities are averaged over the 8 seeds; each seed carries its own split.
- PGD quarter sums are averaged over the 2 splits.
- Seeds and splits are not units of analysis.

## 4. Outcome: fixed clean-competitor contrast, and the switching decomposition

All scores are official, from the neutral context.

**The clean competitor.** c₀ is the colour other than l with the lowest clean score, ties broken by name.

**Definitions for any image y:**
- f(y) = s_{c₀}(y) − s_l(y)
- **D(y) = f(x) − f(y)** (primary damage, in nats, a logit difference)
- W(y) = s_{c₀}(y) − min over c ≠ l of s_c(y), which is ≥ 0. W(x) = 0.
- The official margin is m(y) = f(y) − W(y), so the margin damage is exactly

  **D_m(y) = m(x) − m(y) = D(y) + W(y).**

**Checks on every scored image** (any failure stops the run):
1. The stored margin equals `clean_margin` of the stored four scores exactly.
2. The stored `chosen` equals `choose_color` of the stored four scores.
3. W(x) = 0 and W(y) ≥ 0.
4. |D_m − (D + W)| ≤ 1e-9. This tolerance is for floating-point rounding only.

**Also reported:**
- the number of images whose strongest rival is not c₀, per family;
- images where the label ties a rival;
- whether the two lowest clean rivals tie, in which case c₀ comes from the name rule.

## 5. Per-scene quantities

All in score units, computed separately on each channel: fixed (D), switching (W) and margin (D_m).

| Quantity | Definition |
|---|---|
| **A**, full-mask advantage | D(P) − ⅛ Σ_s D(R_s) |
| **Q**, sum of the four quarter-scale advantages | ½ Σ_a Σ_j D(P_j^a) − ⅛ Σ_s Σ_j D(R_{s,j}) |
| **Δ**, their difference | A − Q |
| **I_P**, PGD full minus sum of quarters | D(P) − ½ Σ_a Σ_j D(P_j^a) |
| **I_R**, random full minus sum of quarters | ⅛ Σ_s [D(R_s) − Σ_j D(R_{s,j})] |

These satisfy **Δ = I_P − I_R.** On every channel, margin = fixed + switching for A, Q, Δ, I_P and I_R; the analysis code checks this.

## 6. Primary analysis

- **Primary:** mean Δ over the 16 scenes, on the fixed contrast.
- **Interval:** a paired two-sided 95% percentile bootstrap over scenes.
  - 9,999 draws, seed 20261111.
  - Scenes are resampled with replacement.
  - **The same resampled scenes are used for every quantity.**
- **Independent unit:** the scene.

**Reported together:**
- the mean and 95% CI of Δ, A, Q, I_P and I_R, on each of the three channels;
- per quantity, the number of scenes above and below 0;
- the per-scene table;
- the excluded tie counts, sign counts and quarter sizes;
- the drift checks.

All intervals are **exploratory and conditional on this development panel (`s16`) and the specified randomization** (masks, seeds and splits). They are not population inferences.

**Direction:**
- positive if the CI lower bound is > 0;
- negative if the upper bound is < 0;
- otherwise **unresolved**. A CI that crosses zero leaves the direction unresolved.

**Secondary: ρ = mean(Q) / mean(A),** a ratio of across-scene means, on the fixed contrast. It equals 1 − mean(Δ)/mean(A) and adds no information beyond A and Δ.
- Its percentile CI uses the same draws.
- **ρ is uninformative, and is reported as not defined, if the 95% CI of mean(A) reaches 0 or any bootstrap draw has mean(A) ≤ 0.**

## 7. What the estimates can and cannot establish

The quarters are random, spatially interleaved quarters of each mask. Δ compares the net interaction between quarters of PGD's set with the same quantity for random sets.
- In a four-player game whose players are the quarters, full − Σ singles is the sum of all interaction terms of order 2 to 4 (6 pairs, 4 triples, 1 quadruple).
- **Interactions inside a quarter are part of that quarter's own damage, and Δ cannot see them.**

| Result | Can establish | Cannot establish |
|---|---|---|
| **Δ positive** | On this panel, PGD's full-mask advantage exceeds the summed advantage of its quarters, net of the random reference. Recorded as an **exploratory conjecture** that part of PGD's advantage needs coordinates from different quarters together. | That PGD's set is super-additive. **A positive Δ can arise when PGD is additive (I_P ≈ 0) and random sets are sub-additive (I_R < 0).** I_P and I_R are reported alongside Δ for this reason. It also cannot say which combinations, of which order, or any per-coordinate mechanism. |
| **Δ near zero, CI including 0** | The direction is unresolved. The point estimate and CI bound the net quarter-scale difference on this panel. | Additivity. Opposite-signed interaction terms can cancel, and interactions inside quarters are untested. |
| **Δ negative** | On this panel, PGD's quarters together over-account for its full-mask advantage, net of the random reference. | That PGD's set is redundant or sub-additive: the same pattern arises if PGD is additive and random sets are super-additive (I_R > 0). It also cannot separate redundancy from any other difference in how the two families' responses depend on perturbation size. |

**Response shape.** The response to adding coordinates may be nonlinear in either family, and it may differ between them, because they sit at different damage levels. This pilot does not model the response shape and makes no claim about its curvature or the direction of any resulting bias. I_P and I_R are reported so that a reader can see which family drives Δ.

**Switching.** If Δ on the official margin differs from Δ on the fixed contrast, the switching row shows how much of the margin result comes from changes of the strongest rival rather than from the fixed logit contrast.

**Literature.** I checked these against the arXiv abstract pages and the conference listings, at abstract level only.
- Wang et al., ICLR 2021 (arXiv:2010.04055) relate adversarial transferability to interactions inside perturbations.
- Ren et al., NeurIPS 2021 (arXiv:2103.07364) report that attacks mainly affect high-order "multi-order interactions".

These motivate examining joint effects. Neither supplies this estimator.

## 8. Budget

| | Official forwards |
|---|---|
| Per scene: clean 1, official PGD-3 image 1, PGD full 1, PGD quarters 2 × 4, random 8 × (1 full + 4 quarters) | **51** |
| Run: 16 × 51 + 1 final clean re-score | **817** |
| Smoke: 51 + 1 (2.3b random seed 0 check) + 1 final clean re-score | **53** |
| **Total** | **870** |

- **Backward passes: 0.** All gradients and iterates are the stored Checkpoint 2.3 ones.
- **Runtime (assumption):** about 0.88 s per forward, derived from 2.3b (1,693.8 s for 96 units of 20 forwards, including mask construction). That gives about 13 min of forwards, plus model load and mask construction: **about 15–20 min in total** on local MPS. The smoke measures the actual rate.
- **Cap:** 45 min of wall clock for the run.
- **Memory:** EMP-1 / M1, unchanged (2.3b peak 5.2 GiB).
- **Cost: $0.**

If the cap interrupts the run:
- the output is labelled `incomplete_time_cap`, and completed scenes are kept;
- **the planned 16-scene analysis is reserved for a complete panel**, and `analyze_partition.py` refuses to run.

## 9. Integrity checks and stopping rules

**Smoke** (`dev4-ref-00-a`, separate folder `run_smoke/`). All bit-exact on `nll`, `margin` and `chosen` against the committed 2.3 / 2.3b records:
- the clean score;
- the replayed official PGD-3 image;
- 2.3b random seed 0, rebuilt with its recorded seed and k.

The smoke also runs all structure and identity checks, and that scene's 51 planned images, and measures timing.

**Every scene in the run:**
- the official PGD-3 image is re-scored and must match the 2.3b neutral→neutral record bit-exactly;
- FGSM must match the stored iterate;
- the sign agreement, mask, split, image and score-identity checks of §3–4 apply;
- the scene must have used exactly 51 forwards.

**End of run:** scene 1's clean image is re-scored; all four scores, the margin and the decision must be identical.

**Stopping rules:**
- Any failed check stops the run. The failing scene's rows are kept, with no fix and no retry without a dated note here.
- **Single shot:** no resume and no overwrite.
- The analysis is written and tested on synthetic records before any inference (§11), and runs once on the complete panel.
- An unresolved Δ is reported as such. No scenes are added and no confirmation is proposed. The next step returns to the author.

## 10. Omitted by decision

The following are not part of this pilot:
- pairs and triples of quarters;
- partitioned patch shuffles;
- the helpful context, as source or as scoring context;
- additional splits;
- finer partitions.

## 11. Implementation and tests (no model)

**Code:**
- `src/path_probe/path_probe/partition.py`: splits, checks, the four-score decomposition, per-scene quantities and the panel bootstrap.
- `src/path_probe/path_probe/partition_stage.py`: the runner.
- `docs/checkpoint-3b-partition/analyze_partition.py`: the analysis.

**Tests, all on synthetic records:**
- `tests/path_probe/test_partition.py`;
- `tests/path_probe/test_partition_stage.py` (fake scorer, synthetic stored iterates).

**What the tests cover:**
- an additive example (Δ = 0);
- PGD additive with random sub-additive (Δ > 0 with I_P = 0);
- a rival switch;
- exact ties: between rivals at clean, and label with rival;
- quarter sizes when k is not divisible by 4 (k = 7, 10, 13);
- the switching identity for A, Q, Δ, I_P and I_R;
- refusal of incomplete scenes and panels, and of incomplete runs;
- the per-scene bit-exact re-score guard and the stored-FGSM guard;
- the forward budget.

The end-of-run clean re-score lives in `execute`, which loads the model. It is not unit-tested, and the smoke exercises it.

**Smoke command** (needs a later decision; not run):

```bash
cd wt-3b
PYTHONPATH=$(ls -d src/*/ | tr '\n' ':') caffeinate -i /path/to/venv/bin/python -m path_probe.partition_stage smoke \
  --tensors ../repo/docs/checkpoint-2.3-exploratory/paths/tensors
```

The stored tensors are git-ignored. The runner records the directory and the sha256 of every tensor file it reads.

## 12. Correction recorded for deferred candidate B

Comparing the full PGD-7 image with its FGSM-aligned-only subset changes the count, because it removes the reversed coordinates. That difference mixes removal with sign.

A sign-effect contrast must:
- keep PGD-7's full rounded support fixed;
- change only signs, among feasible quantized alternatives: the opposite one-level value must stay within [0, 255], and infeasible coordinates are reported or excluded.

**B does not depend on A being completed.**
