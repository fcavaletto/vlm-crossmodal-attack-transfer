# Checkpoint 4, Stage 1: the 2.4 comparison on InternVL3.5-1B, bf16 and fp32 (2026-11-10)

**Status: EXPLORATORY, development data only (48 scenes). Single-tile preprocessing, not InternVL's standard 13-tile configuration.**
- Protocol, runner and analysis: committed and pushed in `c20a87c`, before the first attack.
- Records: [`stage1/`](stage1). Analysis, run once on the complete records: [`analysis/stage1_analysis.json`](analysis/stage1_analysis.json).
- No confirmation split was read or generated. No paid resource was used. No frozen file was changed.

## 1. Summary

**Reading R1 holds: coordinate selection persists in both precisions on this second architecture.** R2, R3 and R4 do not hold.

| Primary cell: neutral source, source context | bf16 | fp32 | fp32 − bf16, paired |
|---|---|---|---|
| PGD-3 selection over matched random (official path) | **+3.39** [2.93, 3.86] | **+3.54** [3.07, 4.02] | +0.15 [−0.08, 0.38] |
| Same, float path | +3.41 [2.95, 3.88] | +3.53 [3.06, 4.01] | +0.12 [−0.12, 0.36] |
| Official − float, selection (paired) | −0.02 [−0.08, 0.03] | +0.01 [−0.03, 0.06] | |
| Scenes where PGD-3 beats all 8 random seeds | 47 / 48 | 47 / 48 | |
| Top-\|∇\| − random | −0.10 [−0.24, 0.03] | +0.05 [−0.10, 0.21] | +0.16 [0.03, 0.29] |
| Sparsity share of the PGD − FGSM gap | −0.14 [−0.22, −0.08] | −0.17 [−0.24, −0.11] | |
| D_FGSM / D_PGD-3 (logits) | 2.76 / 5.73 | 2.94 / 5.96 | D_PGD: +0.23 [0.02, 0.44] |
| PGD-3 / FGSM damage ratio | 2.07 [1.92, 2.25] | 2.03 [1.89, 2.20] | |
| cos(clean gradient, step-2 gradient) | −0.00 [−0.01, 0.01] | −0.01 [−0.02, 0.01] | |
| Sign agreement of the same pair | 0.56 | 0.54 | |

Bootstrap over scenes, 9,999 draws. Paired quantities are formed per scene first.

What this shows, on development data and in this preprocessing configuration:
1. **Selection does not depend on bf16 here.** It is present in both precisions with nearly the same size. The paired difference is small, +0.15 logit, and its interval includes 0. This is read from the paired difference, not from both arms being significant.
2. **Selection does not depend on the second quantization.** Official and float paths give the same selection to within 0.03 logit, in both precisions. Selection works in the underlying differentiable model and in the discretized pipeline alike. Stage 0's main confound is ruled out for this effect.
3. **Selection accounts for more than all of the PGD–FGSM gap.** Same-count random subsets do *less* damage than dense FGSM (negative sparsity share). So PGD-3's advantage is all coordinate choice. In Qwen, by contrast, random subsets did slightly *more* damage than FGSM (share +0.10).
4. **The saliency inversion does not replicate.** On InternVL, the top-\|clean gradient\| mask is indistinguishable from random in both precisions. It is not worse than random, as on Qwen (−1.65 on the same scenes). That is an informative boundary: Qwen's inversion is not a general property of the iterative-versus-single-step gap.
5. **Gradient decorrelation after one ε/4 step persists in full fp32.** The gradient after one ε/4 step is uncorrelated with the clean gradient (cosine ≈ 0), and sign agreement is barely above chance, in both precisions. On InternVL, therefore, this decorrelation is not a bf16 artefact. **This does not establish whether the same holds for Qwen.**

## 2. What was run

| | |
|---|---|
| Model | InternVL3.5-1B (`OpenGVLab/InternVL3_5-1B-HF` @ `9191dbc`), single-tile 448×448, SDPA attention, MPS |
| Precisions | bf16 (weights and input), and full fp32, in separate processes on the same device. Code, scenes, budget, schedule and seeds are identical. |
| Scenes | The 48 arm-a development scenes |
| Attack | Unchanged 2.4: ε = 1/255; FGSM; PGD-3 with ε/4 steps (gradients on the float surrogate path); 8-bit rounding |
| Masks | FGSM's rounded perturbation on k = \|PGD-3 support\| coordinates: top-\|∇\|, and 8 random seeds |
| Attack sources | neutral (primary), then helpful_nonrevealing (secondary, fixed in the protocol). Pass order: all neutral units first. |
| Scoring | Every image in both contexts, on the official processor path and on the float path |
| Runtime | bf16 68.5 min, fp32 92.2 min (35–56 s per unit) |
| Memory | bf16 peak 7.05 GiB. fp32 peak **9.81 GiB**, against the unchanged 10 GiB line. First complete unit: 6.93 GiB (bf16), 9.81 GiB (fp32). Pressure stayed normal in every sample, swap did not grow, and no safeguard fired. |
| Integrity | Exact drift check at the end in both runs. 0 sign reversals. Maximum one level. **0 coordinates outside FGSM's support in all 192 units.** Qwen had up to 7 zero-gradient ties per image. Random mask counts equal k everywhere. |
| Clean, official path | neutral 85.4%, helpful 83.3% in both precisions. Mean margin ≈ 3.1–3.2 logits. |

## 3. Full results

Margin drop D in logits. Source context, official path. Secondary sources are shown for consistency; the readings use neutral.

| | bf16 neutral | bf16 helpful | bf16 pooled | fp32 neutral | fp32 helpful | fp32 pooled |
|---|---|---|---|---|---|---|
| D_FGSM | 2.76 | 3.79 | 3.27 | 2.94 | 3.81 | 3.37 |
| D_PGD-3 | 5.73 | 7.76 | 6.75 | 5.96 | 7.95 | 6.96 |
| D_random (mean of 8) | 2.34 | 3.19 | 2.77 | 2.42 | 3.18 | 2.80 |
| D_top-\|∇\| | 2.24 | 3.10 | 2.67 | 2.47 | 3.11 | 2.79 |
| **Selection** | **+3.39** [2.93, 3.86] | +4.57 [3.94, 5.19] | +3.98 [3.53, 4.42] | **+3.54** [3.07, 4.02] | +4.77 [4.14, 5.37] | +4.16 [3.69, 4.62] |
| Top-\|∇\| − random | −0.10 [−0.24, 0.03] | −0.09 [−0.29, 0.14] | −0.10 [−0.24, 0.07] | +0.05 [−0.10, 0.21] | −0.07 [−0.28, 0.15] | −0.01 [−0.16, 0.15] |
| Sparsity share | −0.14 | −0.15 | −0.15 | −0.17 | −0.15 | −0.16 |
| PGD beats all seeds | 47 / 48 | 46 / 48 | | 47 / 48 | 48 / 48 | |
| Flip rate PGD / random (clean-correct) | 0.72 / 0.44 | 0.85 / 0.49 | | 0.72 / 0.45 | 0.85 / 0.49 | |

**Transfer to the other context** (neutral source scored in helpful): selection +3.36 [2.74, 3.98] in bf16 and +3.42 [2.75, 4.10] in fp32.

**Precision, paired (fp32 − bf16), official path.**

| | Selection | Top-\|∇\| − random | D_PGD-3 | D_FGSM |
|---|---|---|---|---|
| neutral | +0.15 [−0.08, 0.38] | +0.16 [0.03, 0.29] | +0.23 [0.02, 0.44] | +0.17 [−0.02, 0.37] |
| pooled | +0.18 [0.01, 0.34] | +0.09 [−0.04, 0.22] | +0.21 [0.06, 0.37] | +0.10 [−0.05, 0.25] |

**Reading of the precision table.**
- *Selection.* Where the paired interval excludes 0 (pooled), fp32 is slightly *higher*, by about 4% of the effect. Nothing in the data suggests that bf16 inflates selection on this model.
- *Top-\|∇\|.* fp32 moves it from slightly below random to slightly above. The change is small: 0.16 logit, against a cross-model difference of about 1.5.

**Preprocessing, paired (official − float), every method.**
- *Each method's damage.* D changes by at most 0.11 logit for any method in either precision. In bf16 the official path does slightly *less* damage than the float path for every method, by 0.06–0.11. In fp32 the differences are within ±0.07.
- *Selection, official minus float.* bf16 −0.02 [−0.08, 0.03]; fp32 +0.01 [−0.03, 0.06].
- *Conclusion.* The second rounding attenuates damage a little but does not change which coordinates matter.

## 4. Comparison with Qwen3-VL-2B

### What is matched and what is not

| | Matched | Differs |
|---|---|---|
| Scenes | **same 48 development scenes**, same labels, prompts and answer readout | |
| Design | **same attack**: ε, steps, rounding, mask construction, k rule, both sources, both contexts, margin definition | random seeds (different prefix) |
| Analysis | **same definitions**. The analysis reproduces the committed Qwen development numbers exactly. | |
| Architecture | | vision tower (InternViT-300M vs Qwen3-VL ViT), projector, parameter count (1.06 B vs about 2 B) |
| Language model | both Qwen3 family, same tokenizer and colour ids | depth and width |
| Preprocessing | | Qwen: no resize at 512×384 (exact float path). InternVL: 448×448 resize with uint8 rounding, single tile. |
| Precision | bf16 arm in both | InternVL also has full fp32. Qwen has bf16 only (fp32 vision tower in 2.3). |
| Attention, device | MPS in both | Qwen eager, InternVL SDPA |

### Results on the same scenes (neutral source, official path)

Qwen3-VL-2B values are bf16, from the Checkpoint 2.3/2.3b development records. Cross-model differences are paired per scene. They mix architecture, preprocessing and attention implementation, so they are **not** attributable to any one of these.

| | Qwen3-VL-2B bf16 | InternVL bf16 | InternVL fp32 | InternVL bf16 − Qwen |
|---|---|---|---|---|
| D_FGSM | 3.36 | 2.76 | 2.94 | −0.59 [−1.33, 0.15] |
| D_PGD-3 | 9.81 | 5.73 | 5.96 | −4.08 [−5.21, −2.86] |
| PGD / FGSM | 2.92 | 2.07 | 2.03 | |
| Selection | **+5.80** [5.02, 6.58] | **+3.39** | **+3.54** | −2.42 [−3.29, −1.52] |
| Sparsity share | +0.10 [0.03, 0.17] | −0.14 | −0.17 | |
| Top-\|∇\| − random | **−1.65** [−2.17, −1.15] | −0.10 | +0.05 | +1.54 [1.04, 2.06] |
| cos(g0, step-2 gradient) | ≈ 0.00 (2.3 paths) | −0.00 | −0.01 | |
| Held-out confirmation (2.4) | selection +6.50 (pooled), top-\|∇\| −1.38 | — | — | |

**Reading.**
- *What replicates across the two models.* The *existence* of coordinate selection, its dominance of the PGD–FGSM gap, its transfer across contexts, and the near-zero alignment between the clean and next-step gradients. On InternVL all four also hold in full fp32.
- *Size.* InternVL's selection is about 60% of Qwen's on the same scenes, and its PGD/FGSM ratio is lower.
- *What does not replicate.* The finding that magnitude saliency is *worse* than random. On InternVL it is neutral.
- *Why the size and saliency differences exist cannot be said from this comparison.* Architecture, preprocessing and attention all differ.

## 5. Interpretation, by the protocol's readings

| Reading | Result | Meaning |
|---|---|---|
| R1 persists in both precisions | **true** | On this second architecture, the selection effect is not solely caused by bf16 |
| R2 much stronger in bf16 | false | No numerical contribution to selection is visible on InternVL. The paired difference is +0.15 [−0.08, 0.38], or pooled +0.18 [0.01, 0.34] in the *other* direction. |
| R3 only after official preprocessing | false, both precisions | Selection works on both paths. The resize/quantization pipeline is not what produces it. |
| R4 does not replicate | false | |

**What this does not show.**
- **It does not show that Qwen's effect is independent of bf16.** InternVL in fp32 cannot establish the cause of Qwen's behaviour. Qwen in full fp32 has never been run.
- **It does not cover InternVL's standard 13-tile configuration.** A positive result here is a replication with single-tile preprocessing only.
- **It is not a confirmation.** These are development scenes from the distribution that informed the 2.4 hypotheses.
- **It does not explain why** PGD's coordinates are better. It shows that the explanation cannot rest on bf16 rounding of InternVL's inputs or activations, nor on the uint8 resize.

**Post hoc, flagged as such.** On InternVL selection appears without the saliency inversion. So the two 2.4 findings may have different causes. The inversion could be the more model- or numerics-specific of the two, which makes it the part of the Qwen result most worth testing in full fp32.

## 6. Limitations

- Development data only. One task family. Single-tile preprocessing. One ε and schedule.
- InternVL shares its language-model family with Qwen. This replication spans vision towers and preprocessing, not language models.
- 7 of 48 neutral clean images are misclassified (85.4%). Margin damage is the primary quantity because flips near the boundary are fragile.
- The fp32 memory margin was 0.19 GiB. It held throughout, but a larger image or prompt would not fit under the unchanged line.
- The cross-model table compares numbers from separate runs (2.3/2.3b for Qwen). Its random seeds differ by construction.

## 7. Recommended next experiment

**Full-fp32 Qwen3-VL-2B on an on-demand GPU, with a bf16 control on the same GPU.** I agree with the author's expectation. After this run the case is stronger than before, for two reasons.
1. It is the only experiment that tests whether the **already-confirmed** Qwen result depends on numerical precision. Nothing on InternVL can.
2. Stage 1 now makes a specific, falsifiable prediction for it. If the InternVL pattern carries over, Qwen's **selection should persist in fp32**, while the **saliency inversion is the open question**. Both outcomes are informative:
   - the inversion survives fp32: it is a property of Qwen's function;
   - it disappears: it was a bf16 effect on Qwen, and the 2.4 saliency claim needs that qualifier.

**Design.**
- *Data.* The same 48 development scenes and the unchanged 2.4 comparison. This is the same analysis code with a Qwen loader.
- *Arms.* bf16 and fp32 on the **same CUDA device**, so device and precision are not confounded. Also a bit-for-bit reproduction check of a few MPS development scores in bf16, reported as a cross-device difference rather than required to match.
- *Code.* The Qwen pipeline is hash-frozen and macOS-specific: the footprint monitor and the MPS paths. A CUDA run needs a new package, as was done here for InternVL, with no frozen file edited.
- *Memory guard.* A memory guard appropriate to CUDA.

**Cost, needs approval.**
- *Estimate.* 2.4 took 203 s per scene on MPS for both sources in bf16. Two precisions on 48 scenes is about 5.5 h at MPS speed. At an unmeasured 2–4× CUDA speed-up that is 1.5–3 h of compute, plus setup and a timing hour.
- *Price.* At $0.33–0.59/h for a 48 GB card (RTX A6000 or A40; the 2026-11-09 list, nothing created) that is roughly **$2–5**.
- *Card.* 24 GB would probably fit fp32 (projected peak ≈ 16.8 GiB), but leaves little margin.

**Strongest alternative, local and free.** SmolVLM2-2.2B in bf16, the same Stage 1 design. It would test the one generality gap this run leaves, a different language-model family, but it cannot answer the precision question for Qwen. I would run it after the Qwen fp32 experiment.

**Not recommended yet.** A confirmation split for the InternVL result. It would confirm a replication on single-tile preprocessing, a narrower claim than the Qwen fp32 question. `confirm-v2` should be kept for the next important hypothesis.
