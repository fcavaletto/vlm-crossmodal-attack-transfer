# Checkpoint 4, Stage 0: InternVL3.5-1B integration and feasibility gates (2026-11-10)

**Status: EXPLORATORY, development data only. No attack was run.**
- Approved by the author on 2026-11-10: weights download, integration and gates on development scenes.
- No Stage 1 attack, paid resource or new confirmation split was used. `trackB_confirm_v1` was not read.
- Branch `research/internvl-stage0`, kept separate from the publication PR #28.

Code: [`src/internvl_probe`](../../src/internvl_probe). Records: [`stage0/`](stage0). Gradient tensors are git-ignored and local only.

## 1. Summary

| Gate | bf16, single tile | fp32, single tile | bf16, 13 tiles (processor default) | fp32, 13 tiles |
|---|---|---|---|---|
| Integration (geometry, colour tokens, prompt cue) | pass | pass | pass | pass |
| Memory (unchanged 10 GiB line) | **pass**, peak 5.9–7.0 GiB | **pass, narrowly**, peak 9.71–9.80 GiB | **fail**, stopped in the first forward | **fail**, stopped in the first forward |
| Clean accuracy ≥ 75% per context | pass: 88.5% / 84.4% | pass: 88.5% / 82.3% | not run | not run |
| Median clean margin ≥ 2 logits | pass: 3.1 / 3.4 | pass: 2.9 / 3.5 | not run | not run |
| Image dependence | pass: uniform gray image gives 25% (always "blue") | pass: same | not run | not run |
| Scorer determinism | pass: bit-identical repeats | pass | not run | not run |
| Gradient determinism and finiteness | pass | pass | not run | not run |
| Finite differences (diagnostic, 5% line) | **not meaningful in bf16** (§5) | **pass** at small steps after the step-size check (§5) | not run | not run |

Main findings:
1. **The processor's default preprocessing does not fit locally.** InternVLProcessor tiles a 512×384 image into 12 tiles plus a thumbnail, 3,328 image tokens; Qwen3-VL used 192. Both precisions hit the 10 GiB cap during the first forward pass. The line was not changed. This configuration needs a GPU.
2. **Single-tile preprocessing passes every gate in both precisions.** It is the image processor's own default (`crop_to_patches=False`), but not the full processor's default. Stage 1 on this machine would therefore test InternVL3.5-1B *with single-tile preprocessing*, and must say so.
3. **The official input is quantized a second time.** The processor resizes the **uint8** image and rounds after resizing, so the model input is a piecewise-constant function of the RGB image, unlike Qwen3-VL at this image size. Gradients go through a float surrogate of the same geometry, within 1.06 levels of the official tensor (mean 0.02). One-level RGB perturbations survive into the model input only partly (§3).
4. **fp32 gradients are valid. The function is curved far below one 8-bit level.** The finite-difference error is 1.3% (median) at a step of 1/64 level, but 68% at 1/4 level. This is an InternVL observation. **It says nothing about the cause of Qwen's behaviour.**
5. **In bf16 the finite-difference check is not a meaningful comparison.** Logits are quantized in steps of 0.125, and the same forward pass with and without autograd differs by up to 0.12 nats. The bf16 gradient agrees with the fp32 gradient: cosine 0.69, signs 86%, top-30% overlap 0.81.
6. **The unchanged 2.4 comparison is technically valid on this architecture**, with three caveats (§6).

## 2. Setup

| | |
|---|---|
| Model | `OpenGVLab/InternVL3_5-1B-HF` at `9191dbccf312b537016f041b25d61c72e7c5c9f3`, native transformers port (no remote code), Apache-2.0. 1.06 B parameters: InternViT-300M (448 px, patch 14), pixel shuffle to 256 tokens per tile, Qwen3 LM (28 layers, hidden 1024). |
| Runtime | torch 2.15.0, transformers 5.22.1, MPS on MacBook Air M4 24 GB, SDPA attention (the model default; Qwen runs used eager), `logits_to_keep=1` |
| Precisions | bf16: weights and pixel input bf16. fp32: everything fp32. Loaded separately. |
| Data | `development_v4`, 96 scenes in 48 counterfactual groups, hash-checked. Gradient checks use the first 6 of the Checkpoint 2.3/3 subset s16, neutral context. |
| Readout | As in every earlier checkpoint: first answer-token row, full-vocabulary NLL of the four one-token colours. Margin = best other colour's NLL minus the label's. Colour ids 1151, 12203, 13250, 27869: the same as Qwen3-VL, since both use the Qwen tokenizer. Answer cue `<\|im_end\|>\n<\|im_start\|>assistant\n`, checked jointly with each colour. |
| Memory safeguards | Unchanged: EMP-1 preload lines and the `path_probe` MemoryMonitor (10 GiB footprint abort, pressure and swap rules). **Added**: the MPS allocator is capped at the same 10 GiB, so an oversized pass raises out-of-memory instead of swapping. Pressure stayed normal and swap did not grow in every run, including the two failures. |

## 3. Preprocessing (model-free, all 96 images) — [`stage0/pixels.json`](stage0/pixels.json)

| | 13 tiles (default) | single tile |
|---|---|---|
| Tiles / image tokens | 13 / 3,328 | 1 / 256 |
| Surrogate geometry equals processor's | yes | yes |
| Surrogate vs official: mean / max gap | 0.020 / 1.06 levels | 0.021 / 1.05 levels |
| Elements more than 0.5 level apart | 0.6% | 0.6% |
| Dense ±1-level RGB perturbation: model-input elements that change officially | 65% | 63% |
| Same, cosine of official change with float change | 0.94 | 0.93 |
| 30%-sparse ±1-level perturbation: cosine official vs float | 0.89 | 0.88 |
| bf16 cast of the official input keeps every official change | yes | yes |

The gap is the uint8 rounding after the resize, plus small differences between torchvision's integer and float bicubic kernels. Rounding the surrogate gives a maximum gap of exactly 1 level and a mean of 0.006.

## 4. Memory and runtime (measured)

| | bf16 single | fp32 single | bf16 13 tiles | fp32 13 tiles |
|---|---|---|---|---|
| Footprint after load | 3.19 GiB | 5.47 GiB | — | — |
| Lifetime peak | 5.91 GiB (smoke), 6.96 GiB (gradient run) | 9.71 GiB (smoke), **9.80 GiB** (gradient runs) | 10.57 GiB at the cap | 12.06 GiB at the cap |
| Outcome | completed | completed, 0.20 GiB under the line | MPS out of memory, first forward | MPS out of memory, first forward |
| One official forward | ≈ 0.5 s | ≈ 0.7 s | | |
| One RGB gradient | 4.8 s | 4.5 s | | |

The fp32 single-tile margin is thin. The peak should not grow in Stage 1, because every pass has the same 302 tokens. If it does cross the line, the monitor aborts cleanly and records are flushed per scene. The line will not be raised.

## 5. Gradient validity and numerical stability — [`stage0/grad_*.json`](stage0)

Objective: NLL of the label on the surrogate path. Central differences along 4 random ±1 directions per scene, 6 scenes, 24 directions.

| Step (levels) | fp32: median relative error | fp32: within 5% / 10% | bf16: median relative error |
|---|---|---|---|
| 1/1024 | 0.065 | 11 / 16 of 24 | — |
| 1/256 | 0.019 | 19 / 20 | — |
| **1/64** | **0.013** | **18 / 23** | — |
| 1/16 | 0.066 | 10 / 14 | 1.00 |
| 1/4 | 0.68 | 2 / 5 | 0.86 |
| 1 | 1.00 | 0 / 1 | 0.74 |
| 4 | 1.00 | 0 / 0 | 0.82 |

Relative error = \|FD − analytic\| / (\|FD\| + \|analytic\|).

**Investigation, as the author asked, before any verdict.**
- *Step-size dependence.* In fp32 the error falls as the step shrinks, to 1–2% at 1/256–1/64 level, then rises at 1/1024 level where fp32 rounding of the objective dominates. That V-shape is what a correct gradient of a curved function gives. A wrong gradient would not converge.
- *Differentiability.* The surrogate's clamp is active on at most 0.04% of resized values, and no RGB value is at 0 or 1 in these scenes. So kinks are not the source of the error at ≥ 1/4 level. The function is curved at that scale.
- *Preprocessing.* The surrogate is the official geometry without uint8 rounding (§3), so the gradient is of the surrogate. Its scores agree with the official ones: same choice on 100% of items in fp32 and 99% in bf16, with median margin differences of 0.006 (fp32) and 0.12 (bf16) logits.
- *Is the comparison meaningful in bf16?* No.
  - bf16 logits are quantized in steps of 0.125 near their working range, and objective changes at small steps lie below that resolution.
  - The same input evaluated with and without autograd gives NLLs that differ by up to 0.12 nats (fp32: 2×10⁻⁵), since different kernels are selected.
  - So the bf16 finite differences measure arithmetic, not the gradient. The bf16 gradient is instead checked against fp32 on the same scenes: cosine **0.69**, sign agreement **0.86**, top-30% overlap **0.81** (chance 0.30), norm ratio 1.00.

**Verdict.** fp32 gradients are valid. bf16 gradients are deterministic and close to fp32, but their local accuracy cannot be verified by finite differences.

**Non-linearity, descriptive (6 scenes).**
- *Along the FGSM direction*, the actual NLL increase is 0.29× the first-order prediction at ε/4 and 0.10× at ε in fp32; bf16 gives 0.23× and 0.10×.
- *A one-level FGSM step on the official path* lowers the margin by 1.4–4.4 logits in fp32.
- *Comparison with Qwen.* This resembles Qwen's line-scan saturation, but it is a measurement on InternVL in full fp32. As the author noted, it cannot establish the cause of Qwen's behaviour. That needs full-fp32 Qwen (GPU).

## 6. Is the unchanged 2.4 comparison technically valid here?

**Yes, with caveats.**
- The PGD-3 ⊂ FGSM identity is arithmetic in RGB space. Three ε/4 sign steps, projection to ε = 1/255 and 8-bit rounding move only coordinates whose three signs agree, so it holds for any model. Stage 1 must still count violations per scene, as 2.4 did.
- Matched values, signs and count are defined in RGB space, before any model-specific preprocessing. All arms are scored by the same official path.

**Caveats that change interpretation, not validity.**
1. *Second quantization.* The official input rounds after the resize, so a ±1-level RGB change reaches the model only partly (63% of elements, cosine 0.93 for dense). A selection gain could partly mean "coordinates whose change survives the resize rounding". **Control:** score every mask on the surrogate path as well, where there is no second rounding.
2. *Gradient is of the surrogate.* top-\|∇\| and FGSM use the surrogate gradient. This is the standard BPDA-style choice for a non-differentiable input pipeline, and it differs from Qwen, whose differentiable path was exact.
3. *Preprocessing is single-tile*, not the processor's default. Single tile downsamples 512×384 to 448×448, with aspect distortion. The default upsamples 3.5× into 12 tiles. A result does not transfer to the default configuration without a GPU run.

## 7. Recommended Stage 1 design (needs approval; not started)

**Question.** Does the 2.4 coordinate-selection effect appear on a second architecture (Q1), and does it persist in full fp32 on that architecture (Q2)? Q3, whether fp32 removes Qwen's effect, is **not** addressed by this run.

| | |
|---|---|
| Model | InternVL3.5-1B, single-tile preprocessing, bf16 and fp32, same device |
| Scenes | The 48 arm-a development scenes, neutral source, scored in both contexts. s16 reported separately for comparison with the Checkpoint 3 pilot. |
| Attack | Unchanged 2.4: ε = 1/255, PGD-3 with ε/4 steps on the surrogate path, 8-bit rounding |
| Conditions | clean, FGSM, PGD-3, 8 same-count random masks (FGSM's signs on FGSM's support), top-\|∇\| clean surrogate gradient, sparse FGSM |
| Scoring | Official path (primary) **and** surrogate path (control for caveat 1) |
| Integrity | Subset-identity count per scene; clean re-score bit-exact at start and end; plan and analysis script committed before the first attack |
| Diagnostics for later H3 (no extra cost) | sign agreement and cosine between the clean gradient and PGD's step-2 gradient; 5-point line scan along FGSM |
| Measured cost | per scene and precision: 3 gradients + about 57 forwards ≈ 45–60 s. 48 scenes ≈ 40–50 min per precision, **≈ 1.5–2 h for both**, local, no cost. s16 only: ≈ 30 min for both. |
| Memory | bf16 peak ≈ 7 GiB; fp32 peak ≈ 9.8 GiB against the unchanged 10 GiB line |

**Reading, fixed before the run.** Development data, exploratory. The outcome table in the roadmap §9 applies per precision, with the selection gain's bootstrap CI over scenes. A positive result is labelled as a replication on development data with single-tile preprocessing. Confirmation would need a frozen protocol and `confirm-v2`, with approval.

**Strongest alternative explanations to report with any result.**
1. **Resize-rounding survival.** If selection appears only on the official path and not on the surrogate path, it reflects the second quantization, not the model.
2. **Preprocessing-specific.** Single-tile downsampling may favour coordinates differently from the default tiling. A null or a positive says nothing about the default configuration.
3. **bf16 measurement granularity.** Margins move in 0.125-logit steps, and grad and no-grad forwards differ by up to 0.12 nats. Small bf16 effects are unreliable; fp32 is the reference arm.
4. **Low clean margins.** In bf16, 4 of 16 s16 scenes are wrong on clean neutral images, and several margins are under 1.3 logits. Flip rates near the boundary are fragile, so margin damage should be the primary quantity, as in 2.4.
5. **Shared language model.** InternVL3.5-1B and Qwen3-VL-2B both use Qwen3 language models. A replication here generalizes across vision towers and preprocessing, not across language models. SmolVLM2-2.2B is the planned test for that.
6. **Same scene family.** The scenes come from the same synthetic distribution that informed the 2.4 hypotheses.

**Strongest alternative design.** Stage 1 on s16 only, both precisions, about 30 min. It is cheaper and directly comparable with the Checkpoint 3 pilot, but has about a third of the scenes for the CI. I recommend the 48-scene version: the extra hour is small, and a null on 16 scenes would be hard to read.

## 8. What was not done

- No attack, no PGD iterate, no mask scoring.
- No GPU, no paid resource, no new split, no use of `trackB_confirm_v1`.
- The 13-tile configuration was not run beyond the failed memory gate. Chunking the vision tower or gradient checkpointing might fit it locally, but both would change the implementation, so they were not tried without approval.
- No CPU fp64 reference gradient (8.5 GB of weights). fp32 finite differences were used instead.
