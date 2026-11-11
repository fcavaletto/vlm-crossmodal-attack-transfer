# Checkpoint 2.4: held-out confirmation of PGD-3 coordinate selection

**Status: PRE-REGISTERED CONFIRMATION PROTOCOL.** Approved by the research lead on 2026-11-09. This file, the analysis script, the runner, and the confirmation manifest are frozen by hash in `FROZEN.json`. The freeze happens before any model inference on the confirmation scenes. The runner refuses to start unless every frozen hash matches and `FROZEN.json` is committed.

**Origin of the hypotheses.** All three predictions were formed after looking at development data: the 48 `development_v4` arm-a scenes in Checkpoint 2.3b (`../checkpoint-2.3-exploratory/MASK_CONTROL_PLAN.md`, `REPORT.md` §2b). They are confirmatory **only** with respect to the new split defined here, which no model has seen.

**Not touched.** The frozen Checkpoint 2.1 records, the preregistered H\*, the post-hoc H-ΔL, and its drafted `dev-v5` and `sealed-v5` samples. Their seed strings are refused by the generator in code. The Checkpoint 2.2 draft protocol is not edited. All earlier outputs are read only. No paid infrastructure, no PR merges.

## 1. Question

On independently generated scenes, does PGD-3's advantage over FGSM come mainly from **which** RGB channel coordinates it perturbs, rather than from perturbing **fewer** of them?

**Terminology.** The perturbation lives in the 3 × 384 × 512 RGB image tensor, which has 589,824 coordinates. A **coordinate** is one color channel of one pixel. Every count and mask below is over coordinates, not pixels. One pixel can have one, two, or three of its channels perturbed.

## 2. Background: the near-subset property (structural, not a finding)

Setting: ε = 1/255 (one 8-bit level), PGD step size ε/4, three steps, and the official 8-bit round trip (clamp, ×255, round half to even). Under these, the per-coordinate PGD sum is s ∈ {±ε/4, ±3ε/4}. Rounding keeps a change only where |s| = 3ε/4. That happens when all three step signs agree, and the kept sign is then the FGSM sign.

So the official PGD-3 image is FGSM restricted to a subset of coordinates, with the same signs. This holds whatever steps 2 and 3 do. If those steps were random, the kept set would be a uniform random subset of about 25% of FGSM's support.

**Exceptions.** At a coordinate where the clean gradient is exactly 0, FGSM makes no change. If PGD's steps 2 and 3 agree there, the coordinate moves by exactly ε/2. Half-to-even rounding then moves an odd 8-bit value by one level and leaves an even value unchanged. In development this happened at 0–7 of about 176k changed coordinates per unit, in 86 of 96 units. There were never any sign reversals.

The property says nothing about whether PGD selects well. That is why the comparison is against a **random subset of FGSM's support with exactly the same coordinate count**.

## 3. Data: `study2-trackB-confirm-v1`

**Generator.** `src/scene_confirm` (tests in `tests/scene_confirm`).
- It implements the sampling distribution of `../checkpoint-2.2-draft/V5_GENERATOR_SPEC.md` §3, draws 1–14: target size, shape, row, and column; the ordered answer pair; the modal and singleton colors; the foil column; the large-column sets; the decoy columns; the filler shapes; the swap partner; and the filler colors.
- Each draw is conditioned on the v4 per-group checks, reused from `scene_v4`.
- Everything visual is imported from `scene_v4` unchanged: canvas, grid, radii, palette, question template, the neutral and helpful context strings, the oracle, the renderer, and the ambiguity checks.

**Seeding.** Group k uses `PCG64(first 8 bytes BE of SHA-256("study2-trackB-confirm-v1|group|k|attempt|a"))`, with a = 0, 1, … on rejection, up to 1,000 attempts. Groups do not share random state.

**Rejection.** Measured on 20,000 proposals, the constraint rejection rate is 0: the proposal satisfies every v4 check by construction. The realized distribution is therefore the stated proposal distribution, conditioned only on novelty.

**Novelty and separation**, all enforced at generation:
1. The layout signature (shape, size, column, row of all 8 objects) and its three mirror images may not equal any `development_v4` group or any earlier group in this split. Rejections: 0.
2. Scene-graph hashes may not equal any of the 216 hashes in the four development exclusion lists (`development`, `development_v2`, `development_simple`, `development_v4`). Clashes: 0.
3. Decoded-pixel hashes may not equal any of the 216 development images. Clashes: 0.
4. There are no duplicates within the split.

**Reproducibility.** Regenerating gives byte-identical files (checked: identical `checksums.json`).

**Sample.** 48 groups, 96 scenes, in `fixtures/trackB_confirm_v1/`. The manifest sha256 is `fcd8b4ef19b78dfaf6d8a18122b3d47e52ad7deacb88624741a9596aa33b5d55`, also frozen in `FROZEN.json`. Realized balance is reported in `metadata.json`. Balance is not imposed, because groups are independent draws.

**Primary arm.** Arm **a** of every group, fixed before generation. This is the same rule as the development S48 set. One scene per group, so **the group is the statistical unit**. Arm b scenes are rendered and kept, but are not run.

**Size.** 48, fixed before generation, matching development. A planning check resampled the development per-scene selection effects (pooled mean 5.87, SD 2.28) shrunk to a quarter of their mean. In that check the pooled bootstrap lower bound exceeded 0 in 400 of 400 simulated samples. No scene is added after outcomes exist.

**Independence from H-ΔL.** This split uses its own seed string and never draws `study2-checkpoint-2.2-dev-v5` or `study2-checkpoint-2.2-sealed-v1`. `scene_graph_hashes.json` here lists this split's graph and pixel hashes. Any later sealed draw should add it to its exclusion list. That is a recommendation only; the H-ΔL draft is not edited.

## 4. Procedure (identical to Checkpoint 2.3/2.3b, except where noted)

**Model and numerics.** Qwen3-VL-2B-Instruct, pinned revision, bf16 weights on MPS, eager attention, frozen load call. The color token ids are locked.

**Contexts.** `neutral` and `helpful_nonrevealing` (v4 strings), each in turn as the attack source. The other context is the transfer target.

**Per group (arm-a scene x).** Clean official scores in both contexts, plus the four clean gradients (NLL and margin, both contexts), using frozen `clean_block`. Then, for each source context:

- **FGSM:** `project_signed_step(x, g_L_source, ε, "positive")`, with ε = 1/255.
- **PGD-3:** three steps of `pgd_update` with step ε/4 and radius ε, on the frozen labeled-color NLL gradient, starting at x. It is the same loop as `probe.direction_block`.
- **Rounded perturbations:** r_F = round_trip(x + δ_F) − x and r_P = round_trip(x + δ_P) − x, with supports S_F and S_P, and **k = |S_P|**. k counts every coordinate PGD-3 changes, including the zero-gradient exceptions.
- **Structure record:** S_P ⊆ S_F, sign reversals, maximum levels, coordinates outside S_F, whether they are half-level ties, and whether they are zero-gradient.
- **Images scored**, each with the frozen official scorer in **both** contexts:

  | Image | Definition |
  |---|---|
  | `fgsm` | x + δ_F |
  | `pgd3` | the PGD-3 float iterate (the official PGD-3 image) |
  | `topk_grad` | x + r_F ⊙ m, where m holds the k coordinates of S_F with the largest \|g_L_source\|, stable tie-break on the flat index |
  | `random_r`, r = 0…7 | x + r_F ⊙ m, where m holds k coordinates drawn uniformly from S_F without replacement, by `torch.randperm` on a CPU generator seeded with `scene_seed("checkpoint-2.4-confirm-mask|<source>|seed<r>", scene_id)` |

  The runner stops if a random mask's rounded count differs from k.
- **Recorded per image:** official NLL, margin, and decision in each context; k; and the first-order predictions g·r for the four clean gradients.
- **Tensors saved** for later mechanism work, not committed: the gradients and the int8 FGSM and PGD iterates.

**Pipeline validation, before the freeze, on development data only.** `python -m path_probe confirm-smoke-dev` runs the same `scene_block` on development scene `dev4-ref-00-a`, with the 2.3b mask seed prefix. It must reproduce bit-exactly:
- the 2.3 gradients and int8 FGSM/PGD iterates;
- the 2.3 clean, FGSM, and PGD-3 official scores;
- the 2.3b top-k and all eight random-mask official scores.

The result is recorded in `smoke_dev/progress.json`. It gives the timing estimate in §8.

**Order.** Groups 0…47 in index order. Within a group: clean, then neutral as source, then helpful as source.

**Safeguards.** These are unchanged from 2.3b:
- EMP-1 with the M1 thresholds. Before load: available memory ≥ 9.91 GiB, normal pressure, and free disk ≥ 10 GiB.
- Abort on any of: footprint > 10 GiB; critical pressure; warn pressure for 120 s; available memory < 0.75 GiB for 30 s; swap growth > 6 GiB. Hard exit 60 s after an abort if pressure stays critical.
- Thresholds are never changed during the run.

**Cap.** **180 minutes** after load (§8). The runner stops before starting a group if elapsed time plus the mean group time would exceed the cap.

**Single shot.** Records are flushed per group. A run that is stopped, crashes, or is truncated is reported as such. It is not restarted, resumed, or extended, and its completed groups are analyzed as they are, labeled incomplete. The runner refuses to overwrite existing records.

**Drift check.** At the end, the first group's clean neutral official score is recomputed and compared bit-exactly with the first record.

## 5. Outcome definitions

For group g, source context s, and image m:
- **D(m)**, source damage: clean official margin in s minus the official margin of m in s. The margin is z_label − max over other colors of z_c, from the official scorer.
- **T(m)**, transfer damage: the same quantity in the other context.
- **D_rand** is the mean of D over the eight random masks, computed within (g, s) first.
- **Pooled:** the within-group mean of the neutral-source and helpful-source values.
- **Flip:** the official decision ≠ label. It is computed on groups that are clean-correct in both contexts.

**Bootstrap.** Percentile, 9,999 draws, numpy `default_rng(20261116)`, resampling groups.

**Inclusion.** A group enters if all its records exist: clean, and for both sources FGSM, PGD-3, top-k, and eight random masks. No group is excluded for clean correctness, for its margin, or for any outcome.

## 6. Pre-registered predictions and decision rules

| ID | Prediction | Supported if |
|---|---|---|
| **P1 (primary)** | PGD-3's coordinates do more source damage than matched random coordinates. | Mean of D(PGD-3) − D_rand > 0 with neutral as source **and** > 0 with helpful as source, **and** the pooled 95% bootstrap CI lower bound > 0. |
| **S1** | Sparsity explains at most 25% of the PGD-3 vs FGSM gap. | Pooled share = (mean D_rand − mean D_FGSM) / (mean D_PGD3 − mean D_FGSM) has a 95% bootstrap CI upper bound ≤ 0.25. Draws with a non-positive denominator count as +∞. If the full-sample denominator is ≤ 0, S1 is not supported. |
| **S2** | Selecting the top \|clean NLL gradient\| coordinates does less damage than random selection. | Mean of D(top-k) − D_rand < 0 for each source **and** the pooled 95% CI upper bound < 0. |

These are three separate predictions, not one combined test. P1 is the primary claim. There is no multiplicity correction, and S1 and S2 are reported as secondary. The estimates and CIs are reported **whether or not** a rule is met. A prediction that fails is reported as failed. It is not replaced with another endpoint, threshold, subset, or context.

## 7. Descriptive analyses (no decisions)

- D and T for FGSM, PGD-3, top-k, and random, per source and pooled; the sparsity, selection, and magnitude contrasts on both outcomes.
- Per source: PGD-3's rank among the eight random draws, the number of groups where it beats all eight or falls below all eight, and the within-group SD across seeds.
- Flip rates and paired flip differences on clean-correct groups.
- The structure summary: subset, sign reversals, exceptions, and density of S_P in S_F.
- First-order predictions −g_M·r for PGD-3, top-k, and random.
- A side-by-side comparison with the development estimates. This is descriptive only. Development and confirmation are never pooled.

## 8. Runtime estimate

**Smoke.** The development smoke (`smoke_dev/progress.json`, commit before the freeze) took **188 s** for one group. That covers clean scoring, 4 clean gradients, 2 PGD gradients per source, and 44 official scores. The estimate for 48 groups is **150 min** after load.

**Stability.** Per-scene time in 2.3 was very stable: 209–213 s over 32 three-step scenes, which also scored float line scans.

**Memory in the smoke.** Peak footprint was 8.38 GiB (abort line 10 GiB), with 4.34 GiB minimum available. 377 pressure samples were normal and 2 were warn, with no critical samples and no swap growth. The 2.3 full run peaked at 9.45 GiB under the same line without an abort.

**Cap.** **180 min after load**, 1.2× the estimate. An earlier draft of this file set 120 min, before any timing existed. The smoke showed 120 min would truncate the run at about 38 groups, so the cap was set from the measured time. That change happened before the freeze and before any confirmation inference.

## 9. Blindness during the run

- The runner prints only the group index, elapsed time, and memory status.
- `progress.json` holds timings and memory, not scores.
- No one reads `run/records.jsonl` until the run has ended (completed, stopped, or truncated).
- The analysis script then runs once, unchanged. Any later analysis is labeled exploratory.

## 10. Pre-written readings

| Outcome | Reading |
|---|---|
| P1, S1, S2 all supported | On independently drawn scenes, PGD-3's advantage is primarily coordinate selection, and clean-gradient saliency ranks the useful coordinates below random. The development finding replicates, for this model, task family, and ε. |
| P1 supported, S1 not | PGD-3 selects better than chance, but sparsity explains more than a quarter of its advantage on these scenes. The development "selection primarily" reading does not hold as stated. Report as "a combination". |
| P1 not supported | The development selection effect does not replicate. The dev finding is reported as not confirmed, whatever S1 and S2 show. |
| S2 not supported | The "anti-saliency" description is not confirmed. P1 and S1 stand or fall on their own. |
| Truncated or stopped run | Report the completed groups with the label "incomplete confirmation". Do not extend the run. |

Whatever the outcome, no mechanism is claimed. Gradient decorrelation across PGD steps is not evidence for one.

## 11. Deviations

None at freeze. Any deviation is added here, dated, with its reason, before the analysis is run.
