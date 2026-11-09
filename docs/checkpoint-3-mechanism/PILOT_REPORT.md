# Checkpoint 3 pilot: gradient sign consensus does not explain PGD's selection

**Status: EXPLORATORY, development data only (16 scenes).**
- Nothing here is confirmed.
- No confirmation split was used.
- No `confirm-v2` split was generated.

Plan: [`PLAN.md`](PLAN.md), committed in `06f0222` before the full run. Analysis: [`analyze_pilot.py`](analyze_pilot.py), run once, with output in [`analysis/pilot_analysis.json`](analysis/pilot_analysis.json). Records: [`pilot/records.jsonl`](pilot/records.jsonl).

## 1. Summary

**Reading C (consensus fails), by the rule fixed in PLAN §6.**

The question was whether PGD-3 picks good coordinates because their gradient signs are stable around the clean image. Masks built from sign agreement at independently sampled nearby points do almost nothing:
- The best pure consensus mask, `vote_R8` (9 gradient evaluations), recovers **R = 0.05** of PGD's selection advantage [95% CI −0.05, 0.14].
- At PGD's own budget of 3 evaluations, `vote_R2` recovers R = 0.08 [0.01, 0.15].
- Adding gradients does not help. R stays between 0.04 and 0.08 from 1 to 8 extra points.

Three findings are informative beyond the negative:
1. **Averaged gradients are worse than random.** Ranking by smoothed-gradient magnitude (SmoothGrad-style, `smooth_R2` and `smooth_R8`) is worse than random selection, like clean top-|∇|.
2. **About a fifth of PGD's advantage is regional.** Keeping PGD's per-patch coordinate counts but shuffling coordinates inside each 16×16 patch keeps R ≈ 0.22.
3. **PGD's exact coordinate set is not stable.** The PGD mask computed with fp32 vision overlaps the bf16 PGD mask only 0.11 above chance. On this development set, the earlier 2.3 study found that the fp32-vision attack is about as damaging.

Points 2 and 3 suggest that PGD finds *a* damaging set rather than a unique stable one. That is a hypothesis for later work, not a finding.

## 2. Setup (as planned)

| | |
|---|---|
| Scenes | 16 development scenes (s16), neutral context as source, scored in both contexts |
| Model | Qwen3-VL-2B, bf16 on MPS; L∞ ε = 1/255; official 8-bit scorer |
| Gradients per scene | 14 new gradients:<br>• R: 8 dense random points at ±ε/4<br>• Q: 4 points at ±ε on 1/16 of coordinates, on the 8-bit grid<br>• J: 2 jitter points at ±ε/16 |
| Masks | All masks use FGSM's signs on exactly k coordinates of FGSM's support. k is PGD-3's count, median 174,711. |
| Main quantity | R(X) = mean selection gain of X over random ÷ mean selection gain of PGD-3. Bootstrap over scenes, seed 20261110. |
| Integrity | • Smoke reproduced the clean, PGD-3 and random-mask scores bit-for-bit.<br>• The end-of-run drift check was exact.<br>• 80.3 min total, against a 120 min cap.<br>• Memory pressure was normal throughout, with no swap growth. |

## 3. Results

Margin damage D is in logits. "Sel" is D minus the mean of 8 same-count random masks. Source is neutral, target is the other context.

| Condition | Grad evals | D source | Sel source [95% CI] | R source [95% CI] | R target | Flip src / tgt |
|---|---|---|---|---|---|---|
| FGSM (dense) | 1 | 3.41 | −0.53 | −0.09 | −0.06 | 0.25 / 0.19 |
| **PGD-3** | 3 | **9.61** | **+5.67** [4.51, 6.74] | 1 | 1 | 0.44 / 0.63 |
| top-\|∇\| (clean) | 1 | 2.34 | −1.60 | −0.28 [−0.47, −0.15] | −0.29 | 0.13 / 0.13 |
| vote_R1 | 2 | 4.18 | +0.24 | 0.04 [0.00, 0.09] | 0.06 | 0.19 / 0.19 |
| **vote_R2** (equal budget) | 3 | 4.40 | +0.46 | **0.08** [0.01, 0.15] | 0.12 | 0.19 / 0.19 |
| vote_R4 | 5 | 4.23 | +0.29 | 0.05 [−0.02, 0.12] | 0.12 | 0.25 / 0.19 |
| **vote_R8** (primary) | 9 | 4.20 | +0.26 | **0.05** [−0.05, 0.14] | 0.06 | 0.19 / 0.19 |
| smooth_R2 | 3 | 2.60 | −1.34 | −0.24 [−0.43, −0.12] | −0.22 | 0.13 / 0.19 |
| smooth_R8 | 9 | 3.16 | −0.78 | −0.14 [−0.29, −0.02] | −0.16 | 0.19 / 0.13 |
| vote_Q4 (8-bit grid) | 5 | 4.11 | +0.17 | 0.03 [−0.05, 0.11] | 0.04 | 0.19 / 0.19 |
| vote_J2 (±ε/16) | 3 | 4.19 | +0.25 | 0.04 [−0.03, 0.11] | 0.06 | 0.19 / 0.19 |
| vote_P32 (fp32 vote) | 2 | 4.17 | +0.23 | 0.04 [−0.01, 0.10] | 0.08 | 0.19 / 0.19 |
| pgd_patch, seed 0 | 3 | 5.21 | +1.27 | 0.22 [0.18, 0.29] | 0.20 | 0.31 / 0.25 |
| pgd_patch, seed 1 | 3 | 5.16 | +1.21 | 0.21 [0.17, 0.28] | 0.24 | 0.25 / 0.31 |
| voteR8_patch | 9 | 3.71 | −0.23 | −0.04 [−0.13, 0.03] | −0.05 | 0.19 / 0.19 |
| sfgsm_R2 (dense) | 3 | 3.52 | −0.42 | −0.07 | −0.04 | 0.19 / 0.19 |
| sfgsm_R8 (dense) | 9 | 3.48 | −0.46 | −0.08 | −0.10 | 0.19 / 0.06 |

- Flip rates are over all 16 scenes, not only clean-correct ones. One scene changes a rate by 0.06.
- The pgd_patch masks use PGD's mask, so their gradient-evaluation count is nominal.
- Sanity check: PGD-3's selection here (+5.67) is consistent with the earlier development estimate on all 48 groups (+5.87) and with the held-out confirmation (+6.50).

### Plan qualifiers (PLAN §6)

- **Spatial.** pgd_patch keeps R = 0.22 (mean of 2 seeds), below the 0.7 bar for "largely regional". Patch-level placement accounts for about a fifth of PGD's advantage; the rest is within-patch coordinate choice. Consensus density is moderately correlated with PGD's at token scale (Spearman 0.32), but placing vote_R8's coordinates with that density (voteR8_patch) does no better than vote_R8. Whatever regional signal consensus carries does not translate into damage.
- **Quantization and bf16.** vote_Q4 (R 0.03) and vote_R4 (0.05) are within 0.2 of each other, as are vote_J2 (0.04) and vote_R8 (0.05). Grid placement and point scale make no difference, because every variant is near zero.
- **Precision.**
  - A vote with the fp32-vision FGSM sign adds nothing (R 0.04).
  - The fp32 and bf16 FGSM signs agree on only 75% of FGSM's support.
  - The fp32-vision PGD-3 mask overlaps the bf16 PGD-3 mask only 0.11 above chance; the token-density Spearman is 0.48.
- **Budget.** At equal budget R₃ = 0.08; more points give no gain. Consensus is not a cheaper route to PGD's set.
- **Exploit.** Dense steps along the averaged sign (sfgsm) do FGSM-level damage: 3.5 against FGSM's 3.4 and PGD-3's 9.6. This is descriptive only.

### Gradient geometry (development, descriptive)

| Point type | n | cos with clean gradient | Sign agreement with clean gradient |
|---|---|---|---|
| R (±ε/4 dense) | 128 | 0.16 [0.12, 0.20] | 0.71 |
| Q (±ε sparse, on the grid) | 64 | 0.19 [0.13, 0.25] | 0.71 |
| J (±ε/16 dense) | 32 | 0.30 [0.21, 0.40] | 0.75 |

Even a move of ε/16, a sixteenth of one 8-bit level, rotates the gradient to a cosine of 0.30. This matches the 2.3 observation that gradients decorrelate after one ε/4 step, and the "shattered gradients" picture [Balduzzi et al., 2017]. The local sign field is not a stable object that a few samples could estimate.

### Spatial association (association only)

These are correlations of where masks put coordinates. They are **not** evidence about any internal mechanism.

| Mask | Overlap with PGD minus chance | Token-density Spearman with PGD (32 px) | Referent-cell enrichment |
|---|---|---|---|
| PGD-3 | 0.70 (self) | 1 | 1.09 [1.05, 1.14] |
| top-\|∇\| | −0.02 | −0.30 [−0.38, −0.23] | 1.18 [1.10, 1.27] |
| vote_R8 | 0.06 [0.05, 0.07] | 0.32 [0.27, 0.38] | 1.25 [1.14, 1.36] |
| smooth_R8 | 0.00 | −0.23 [−0.30, −0.16] | 1.28 [1.21, 1.37] |
| random | 0.00 | −0.01 | 1.00 |
| PGD-3, fp32 vision | 0.11 | 0.48 | 1.08 |

Clean and smoothed gradient magnitude concentrate on the referent object's cell more than PGD does, in regions anti-correlated with PGD's density. vote_R8 is also referent-enriched (1.25). PGD's set is only mildly referent-enriched. The damaging coordinates are not simply "on the object the question asks about".

## 4. Interpretation

**What this rules out on development data.** A sign-stability account in which PGD's 3-step vote approximates a sign field that is stable across the ε-ball, and which random neighbourhood samples could therefore also estimate. That account predicts R near 1 for vote_R8. The observed R is 0.05.

**What it does not rule out.**
- Functions of the clean gradient other than those tested. The pilot adds three more clean-neighbourhood rules: sign voting, smoothed magnitude and the averaged-sign dense step. All three fail. That still does not show that *every* function of local gradients fails.
- A path-specific account. PGD's second and third gradients are taken at points chosen *by the attack*, and the attack's own moves change the gradient. The pilot was designed to separate exactly this. Attack-independent points carry essentially none of the selection signal, so the signal seems to come from the path.

**A hypothesis for later, not a claim.** Damaging subsets may be many and not unique, so PGD finds one by moving along a direction that the model's response is curved along. Two observations point that way: the fp32-vision PGD set barely overlaps the bf16 one yet does similar damage, and only about 22% of the advantage is regional. Testing this needs path-specific second-order quantities, the planned next candidate (PLAN §8), and a damage comparison of fp32 and bf16 PGD sets on the same scenes.

**Novelty.** Neighbourhood gradient averaging (SmoothGrad, EOT, VMI-FGSM) is established as a way to *improve* attacks or saliency. The narrow point here is diagnostic: at matched count and sign, attack-independent neighbourhood consensus does not reproduce an iterative attack's coordinate choice on this model. This is exploratory and from 16 development scenes.

## 5. Limitations

- 16 development scenes, one model, one ε, one precision, and the neutral context as source only.
- The readings are exploratory rules, not tests.
- Random points were sampled at ±ε/4, ±ε and ±ε/16. Other distributions, for example Gaussian or larger radii, were not tried.
- Overlap and spatial statistics use 16-pixel patches and 32-pixel tokens, which approximate the model's visual tokens.
- The fp32-versus-bf16 PGD damage comparison quoted in §4 comes from Checkpoint 2.3 (development), not from this pilot.

## 6. Decision

Reading C. Per PLAN §8, **no confirmatory protocol is proposed and no `confirm-v2` split is generated.** A negative result on development data does not need a held-out confirmation to be reported as exploratory.

Two candidate next steps, both development-only and not started:
1. Path-specific curvature along the FGSM direction (PLAN §8).
2. Same-scene damage of PGD sets computed under different numerics, to test the "many damaging subsets" hypothesis.

Neither is planned before 16 November.
