# Checkpoint 6, Stage 0: SmolVLM2-2.2B feasibility gates (2026-11-10)

**Status: EXPLORATORY, development data only. No attack was run.**
- Approved by the author on 2026-11-10: the weights download and every Stage 0 gate, run locally.
- No paid resource, no held-out data and no new confirmation split were used. `trackB_confirm_v1` was not read.
- The gate lines, memory safeguards and task are those of the plan ([`STAGE0_PLAN.md`](STAGE0_PLAN.md)) and of InternVL Stage 0. **None was changed after seeing a result.**

Code: [`src/smolvlm_probe`](../../src/smolvlm_probe). Records: [`stage0/`](stage0). Gradient tensors are git-ignored.

## 1. Verdict

**SmolVLM2-2.2B does not qualify for the unchanged 2.4 comparison.** It fails the clean-accuracy gate (≥ 75% per context) and the margin gate (median ≥ 2 logits) in both preprocessing configurations. In the standard configuration it also cannot be differentiated within the local 10 GiB line. Its earlier, smaller sibling (SmolVLM2-500M) also failed competence, in Checkpoint 1.7. This is a negative result about the model on this task, not about the attack question.

| Gate | **13 tiles** (processor default) | Single tile (`do_image_splitting=False`) |
|---|---|---|
| Integration (geometry, colour ids 2382/4461/2654/5724, answer cue) | pass | pass |
| Weights sha256 against the pinned LFS ids | pass | pass |
| Memory, bf16, forward pass | **pass**, peak 6.7 GiB (SDPA) | pass, peak 6.7 GiB |
| Memory, bf16, gradient | **fail**: the 10 GiB MPS cap is hit after the forward pass (SDPA); eager attention fails even the forward | pass, peak 8.6 GiB |
| Memory, fp32 | not attempted (larger than single tile) | **fail at load**: fp32 weights are 9.0 GB |
| **Clean accuracy ≥ 75% per context** | **fail: 65.6% neutral, 68.8% helpful** | **fail: 51.0% neutral, 52.1% helpful** |
| **Median clean margin ≥ 2 logits** | **fail: 1.62 / 1.31** | **fail: 0.75 / 0.50** |
| Image dependence: uniform gray image | pass: 25%, always "blue" | pass: same |
| Image dependence: counterfactual image swap | pass: follows the swapped image in 66–69% of prompts (79–82% of those first answered correctly); keeps the original answer in ≤ 1% | pass: follows in 51–52% (61–64%); keeps the original in 3% |
| Scorer determinism (official path, bit-exact repeats) | pass | pass |
| Gradient finiteness and repeatability | not measurable locally | pass: finite and bit-identical |
| Finite differences (diagnostic, 5% line) | not measurable locally | **not meaningful in bf16**; fp32 check impossible locally (§5) |
| Differentiable vs official preprocessing | pass: max gap 1.23 levels, mean 0.02 | pass: max 1.23, mean 0.03 |

"Prefer the standard configuration" was followed. The 13-tile results are the primary ones. Single-tile results are reported beside them, not instead of them.

## 2. Setup

| | |
|---|---|
| Model | `HuggingFaceTB/SmolVLM2-2.2B-Instruct` at `482adb537c021c86670beed01cd58990d01e72e4`, Apache-2.0, 2.25 B parameters. SigLIP vision (27 layers, 384 px, patch 14), pixel shuffle × 3 (81 tokens per tile), Llama-type LM (24 layers). **Stored in float32**, so bf16 here is a down-cast, the reverse of Qwen. |
| Download | 8.99 GB; both shards' sha256 equal the repository's LFS ids ([`stage0/weights_sha256.json`](stage0/weights_sha256.json)). Free disk before: 27 GiB. |
| Runtime | torch 2.15.0, transformers 5.22.1, torchvision 0.29.2, MPS on MacBook Air M4 24 GB, bf16 weights and input, `logits_to_keep=1` |
| Attention | **SDPA, the model default** (as in InternVL Stage 0) for every gate. The first 13-tile smoke run used eager attention, the Qwen pipeline's setting, and failed in the forward pass. It is kept as [`smoke_bf16_split_eager.json`](stage0/smoke_bf16_split_eager.json). Single-tile results agree between the two (§4). |
| Data and readout | Unchanged: `development_v4`, 96 scenes in 48 counterfactual pairs, two contexts. The readout is the first answer-token row, full-vocabulary NLL of the four one-token colours, and margin = best other colour's NLL minus the label's. |
| Memory safeguards | Unchanged: EMP-1 preload lines, `path_probe` MemoryMonitor (10 GiB footprint abort, pressure and swap rules), MPS allocator capped at 10 GiB. |

## 3. Preprocessing (model-free, all 96 images): [`stage0/pixels.json`](stage0/pixels.json)

The processor (torchvision backend) Lanczos-resizes the **uint8** image to 1536×1152. It then cuts twelve 384-px tiles plus a squashed 384×384 global image, or, for single tile, only the 384×384 resize. Every resize rounds to uint8. The differentiable surrogate follows the same geometry in float32. Like torchvision's uint8 kernel, it resizes horizontally first and saturates to 0–255 before the vertical pass, which cut the largest gap from 4.3 to 1.1 levels on the upscale.

| | 13 tiles | single tile |
|---|---|---|
| Images / image tokens | 13 / 1,053 | 1 / 81 |
| Surrogate geometry equal to the processor's | yes | yes |
| Surrogate vs official: mean / max gap | 0.021 / 1.23 levels | 0.034 / 1.23 levels |
| Elements more than 0.5 level apart | 0.6% | 1.3% |
| Dense ±1-level perturbation: official input elements that change | 66% | 56% |
| Same, cosine of official change with float change | 0.92 | 0.91 |
| 30%-sparse ±1-level: cosine official vs float | 0.87 | 0.84 |
| bf16 cast keeps every official change | yes | yes |
| Resized values saturated by the surrogate (kinks) | 1.4% | 11% |

As with InternVL, the official input is quantized a second time after resizing. Small RGB perturbations reach the model only partly.

## 4. Memory and runtime (measured)

| Run | Outcome | Peak footprint | Time |
|---|---|---|---|
| bf16, 13 tiles, SDPA, smoke | forward 5.2 s, then the cap was hit in the gradient step | 11.5 GiB (process abort line exceeded; MPS out of memory) | – |
| bf16, 13 tiles, eager, smoke | cap hit in the first forward pass (vision attention, 13 × 16 × 729 × 729) | 11.5 GiB | – |
| bf16, single, SDPA, smoke | completed: forward 0.95 s, backward 4.6 s | 8.6 GiB | 17 s |
| bf16, single, eager, smoke | completed: forward 1.0 s, backward 4.1 s; same margin as SDPA | 8.6 GiB | 16 s |
| fp32, single, SDPA, smoke | cap hit while loading; swap grew 0.81 GB, pressure briefly at level 2 | 10.5 GiB | 10 s |
| bf16, single, clean (192 prompts × 4 scorings) | completed | 6.7 GiB, no swap growth | 471 s, 2.4 s per prompt |
| bf16, single, gradients (4 scenes) | completed | 8.6 GiB | 136 s |
| bf16, 13 tiles, clean (192 × 4) | completed | 6.7 GiB, no swap growth | 5,504 s, 28.5 s per prompt |

The 13-tile clean run took longer per forward pass than its smoke test, at 100% GPU utilization. That is consistent with sustained-load throttling on a fanless machine.

## 5. Gradients (bf16, single tile, 4 scenes of s16, neutral): [`stage0/grad_bf16_single_sdpa.json`](stage0/grad_bf16_single_sdpa.json)

- **Finite and repeatable.** All gradients are finite, non-zero everywhere and bit-identical on repeat.
- **Same forward with and without autograd:** the objective differs by 0.0001–0.07 nats.
- **Finite differences, central, 4 random sign directions:**
  - the relative error is ≈ 1.0 at steps of 1/16 and 1/4 level;
  - it falls to 0.05–0.74 only at 4 levels;
  - the best per direction ranges from 0.05 to 0.97.
  - As with InternVL in bf16, this is **not a meaningful check**: bf16 logits are coarse, and 11% of resized values sit on saturation kinks. The InternVL fix, an fp32 reference, is not possible here because fp32 does not fit.
- **One-level FGSM step on the official 8-bit path:** the label's NLL rises by 0.81–1.42 on all four scenes, 13–36% of the first-order prediction. So the gradient direction does reach the model through the official path.

## 6. Implications

1. **The model is not competent enough on this task for the 2.4 comparison.** With 66–69% accuracy and a median margin below 2 logits, a third of clean prompts are already wrong, and the correct ones sit near the decision boundary. Selection effects measured on such a model would mix attack strength with weak baseline competence. The unchanged rule excludes it, and the rule stays.
2. **Preprocessing matters for competence.** Thirteen tiles raise accuracy by about 15 points over a single tile (51→66%, 52→69%). The study's earlier SmolVLM2-500M runs used single tile, which may explain part of their low scores. It does not rescue the 2.2B model.
3. **Even a competent 13-tile model would need a GPU** for gradients and for fp32: bf16 gradients exceed the local line, and the fp32 weights alone are 9 GB.
4. **Nothing here bears on Qwen's PGD-3 > FGSM selection effect** or on InternVL's results. It is a failed qualification, not a replication failure.

## 7. Recommendation

- **Do not run Stage 1 on SmolVLM2-2.2B.**
- The Qwen full-fp32 CUDA run (Checkpoint 5) remains the most informative next experiment. It is blocked only by GPU capacity: on 2026-11-10, A40, A6000 and L40S were all refused or out of stock. The RTX PRO 6000 (96 GB, $2.49/h, LOW) is available but not authorized; see the Checkpoint 5 deployment status.
- If a third architecture is still wanted, choose it by clean competence on `development_v4` first. A model below the 75% line should not be carried into attack measurement.
