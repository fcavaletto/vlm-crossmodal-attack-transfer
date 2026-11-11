# Checkpoint 2.3b: random-mask FGSM control (pre-registered)

**EXPLORATORY, DEVELOPMENT ONLY.** Approved by the research lead on 2026-11-09. Budget: about one hour of local computation. This plan was committed before any mask-control output existed. Frozen checkpoints and all Checkpoint 2.3 outputs in `paths/` are read only. No sealed or new samples, no paid infrastructure, no merges.

> **Terminology note added 2026-11-09 (after the run; the plan text below is unchanged).** "Pixels" in §1 and §2 means RGB channel coordinates (one channel of one pixel). §8 and §9 discuss real 8-bit pixel values and use the word correctly.

## 1. Question

In the 2.3 run, the official PGD-3 perturbation does 2.55× FGSM's margin damage. After the 8-bit round trip it is a subset of FGSM's pixels with identical signs. Does PGD's advantage come from:

- **sparsity**: perturbing fewer coordinates, with any choice of coordinates;
- **adaptive selection**: PGD's second and third gradients pick particularly damaging coordinates;
- or **a combination** of the two?

The gradient decorrelation seen in 2.3 is not used as evidence for either answer.

## 2. Why "subset of FGSM" is structural, not evidence of selection

Let x be the clean image, on the 8-bit grid (multiples of 1/255). Let ε = 1/255 = one 8-bit level, and let the step size be ε/4. Then:

1. PGD-3 starts at x. The first step is ε/4 · sign(g₀) = ¼·δ_FGSM. The next two steps are ε/4 · sign(g₁) and ε/4 · sign(g₂).
2. Per coordinate, the sum is s = (ε/4)(σ₀ + σ₁ + σ₂) with σᵢ ∈ {−1, +1}. So s ∈ {±ε/4, ±3ε/4}. Since |s| ≤ 3ε/4 < ε, the ε-ball projection never acts. Clamping to [0, 1] acts only at pixels already at 0 or 255.
3. The 8-bit round trip maps x + s to the nearest level. |s| = ε/4 is a quarter level and rounds back to x. |s| = 3ε/4 is three quarters of a level and rounds to x ± ε. Neither is a tie, so half-to-even rounding plays no role.
4. |s| = 3ε/4 happens exactly when σ₀ = σ₁ = σ₂, and then the sign of the rounded change is σ₀, which is the FGSM sign.

So **the official PGD-3 perturbation is always FGSM restricted to the coordinates where steps 2 and 3 agree with step 1. No coordinate can change sign.** This holds for *any* σ₁ and σ₂, including random signs. It is a property of (3 steps) × (ε/4) × (ε = one 8-bit level) × (rounding). It is not evidence that PGD selects anything.

If σ₁ and σ₂ were independent fair coins, the kept set would be a uniform random subset with expected density 1/4. The observed density was 29.9%. This is why a **uniform random subset of FGSM's support with exactly the same count** is the right null for "adaptive selection".

## 3. Scenes and perturbations

- **Scenes:** the 48 S48 development scenes of Checkpoint 2.3. Sources: neutral and helpful_nonrevealing. That gives 96 (scene, source) units, and the scene is the statistical unit.
- **Inputs:** read from `paths/tensors/`, never written. For each scene and source:
  - the FGSM perturbation δ_F and the PGD-3 iterate δ_P;
  - the clean gradients L (NLL) and M (margin) for both contexts.
- **Rounded perturbations:**
  - r_F = round_trip(x + δ_F) − x and r_P = round_trip(x + δ_P) − x;
  - their supports are S_F and S_P, with k = |S_P|.
- **Structural check** (recorded for every unit, CPU only): S_P ⊆ S_F; sign(r_P) = sign(r_F) on S_P; |r_P| ∈ {0, ε}. If any check fails, the unit is still scored, and the failure is reported.

Masked perturbations: all use r_F restricted to a mask of exactly k coordinates drawn from S_F, so the image is x + r_F ⊙ m.

| Name | Mask |
|---|---|
| `pgd_mask` | S_P itself. This is the official PGD-3 image. Re-scoring it is the pipeline check: it must reproduce the recorded PGD-3 official scores **bit-exactly**. |
| `random_r`, r = 0…7 | k coordinates drawn uniformly from S_F without replacement: `torch.randperm` with a CPU generator seeded by `scene_seed("checkpoint-2.3b-mask|<source>|seed<r>", scene_id)`. |
| `topk_grad` | The k coordinates of S_F with the largest \|L_source\| (the clean NLL gradient that FGSM uses). Ties are broken by a stable sort on the flattened index. |

Dense FGSM and PGD-3 are not re-scored. Their official scores are read from `paths/records.jsonl`, and `pgd_mask` verifies that the pipeline matches them.

## 4. Scoring

- Each perturbed image is scored with the frozen `score_official` in **both** contexts, through the same Qwen backend: bf16 on MPS, eager attention, the fp32-projection readout recorded.
- Recorded per mask: official NLL, margin and decision in each context; k; |S_F|; and the linear predictions g·r for the L and M gradients of both contexts.
- **Order:**
  - **pass A:** scenes in S48 order; for each scene, both sources; for each source, `pgd_mask`, `topk_grad`, then `random_0`…`random_3`;
  - **pass B:** seeds 4…7 for every scene and source.

## 5. Safeguards and caps

- **Memory:** EMP-1 with M1 exactly as in 2.3. Preload requires available ≥ 9.91 GiB, normal pressure, and disk ≥ 10 GiB. The monitor aborts on:
  - footprint above 10 GiB;
  - critical pressure;
  - warn pressure for 120 s;
  - available memory below 0.75 GiB for 30 s;
  - swap growth above 6 GiB;

  and it hard-exits 60 s after an abort if pressure stays critical. Nothing here uses fp32 vision. Thresholds are not changed.
- **Time:** scoring stops before starting a (scene, source) unit if elapsed time plus the mean unit time would exceed **60 min after load**. Pass B starts only for units that fit the same cap.
- **Interruption:** records are flushed per unit. An interrupted run is reported as interrupted and is not restarted.

## 6. Analysis (descriptive; scene bootstrap)

The bootstrap is percentile, 9,999 draws, seed 20261109, over scenes. Random masks are averaged within (scene, source) first. **The primary seed set is the largest set of seeds completed for every unit.** Results are reported per source, and pooled as the within-scene mean of the two sources.

For margin drop D (clean official margin minus perturbed official margin), on the source context (primary) and on the other context (transfer):

1. D_FGSM, D_PGD, D_rand (mean over seeds), D_topk.
2. **Sparsity effect:** D_rand − D_FGSM.
3. **Selection effect:** D_PGD − D_rand.
4. **Magnitude selection:** D_topk − D_rand, and D_PGD − D_topk.
5. **Share of the PGD-over-FGSM gap explained by sparsity:** (mean D_rand − mean D_FGSM) / (mean D_PGD − mean D_FGSM), as a ratio of scene means, bootstrapped.
6. **Per-unit rank:** the fraction of the seeds that PGD's mask beats, and the number of units where PGD beats all seeds. Also the within-unit seed SD of D_rand.
7. **Attack success:** the flip rate (official decision ≠ label) for each mask, on clean-correct scenes, with paired differences.
8. **Linear predictions** g_M · r for each mask, to show what a first-order model expects from the selection.

**Interpretation rule** (fixed now; descriptive, not a test), on the pooled source margin drop:

- **Sparsity primarily:** share ≥ 0.75.
- **Selection primarily:** share ≤ 0.25.
- **Combination:** in between.

"PGD selects better than chance" is stated only if the CI of D_PGD − D_rand excludes 0 in the pooled analysis *and* has the same sign in each source. The same rule holds for "magnitude selection beats chance". Whatever the outcome, no mechanism is claimed from the gradient decorrelation alone.

## 7. Not done

- No other densities, no complement mask, no new PGD runs, no gradients.
- No fp32 vision.
- No sealed data.

## 8. Amendment before any model output (2026-11-09)

The toy test and a CPU-only check of the stored 2.3 tensors found an exception to §2 at **saturated pixels**. The exception arises at a pixel already at 255:

1. FGSM's +ε step is clipped, so FGSM makes no change there.
2. PGD's first step is clipped the same way.
3. If steps 2 and 3 go down, the iterate sits at 255 − ½ level. Half-to-even rounding takes 254.5 to 254, a one-level decrease where FGSM did nothing.

The symmetric case at 0 rounds 0.5 to 0, so it changes nothing. In the real data, every one of the 96 units has such coordinates: median 10 and at most 18 of about 176k PGD-3 coordinates, at most 0.011%. **There are zero sign reversals anywhere, and all changes are a single level.**

The precise statement is therefore: the official PGD-3 perturbation is a sign-preserving subset of FGSM on all non-saturated pixels, plus at most 18 one-level decreases at saturated white pixels per unit.

Changes to the plan:
- `pgd_mask` scores the official PGD-3 image x + r_P itself, not r_F ⊙ (S_P ∩ S_F), so the pipeline check is exact.
- k = |S_P| counts every coordinate PGD-3 changes, including the saturated ones, so the matched count is exact.
- Random and top-k masks are still drawn from S_F.

Nothing else changes.

## 9. Correction to §8, still before any result (2026-11-09)

**§8's diagnosis was wrong.** It came from rebuilding PGD-3 as x + decode(int8 iterate). That is not bit-identical to PGD's accumulated float iterate, and at half-level ties the 8-bit rounding can go the other way.

A one-scene smoke run found the problem. Its re-scored PGD-3 image was not bit-exact against the recorded 2.3 scores, so the run was stopped and nothing was used.

PGD-3 is now **replayed** with the frozen `pgd_update`, using step signs recovered from the stored iterates. The replay reproduces the stored int8 iterate exactly, and it matches the recorded rounded count and L2 in **96/96 units**.

On the replayed images:
- **Sign reversals:** zero anywhere.
- **Changes:** every change is a single level.
- **Coordinates outside FGSM's support:** 86 of 96 units have some, at most 7 per unit. All are coordinates where **the clean gradient is exactly 0**, so FGSM leaves them unchanged. Steps 2 and 3 agreed there, which moves the pixel by exactly ε/2. Half-to-even rounding sends an odd 8-bit value up or down by one level and leaves an even value unchanged. No saturated pixel is involved: the scenes contain no pixels at 255.

Changes to the plan:
- `pgd_mask` scores the replayed float iterate, which is the image 2.3 scored. Its scores must be bit-exact against the recorded PGD-3 scores; the run stops otherwise.
- k = |S_P| as in §8.
- The smoke attempt that exposed the problem is kept in `masks_smoke_attempt1/`.
