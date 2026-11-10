# Checkpoint 6, Stage 0 plan: SmolVLM2-2.2B local feasibility (2026-11-10)

**Status: PLAN ONLY.** No weights are downloaded and nothing is run until the author approves. Development data only. No attack, no paid resource, no held-out data, no use of `trackB_confirm_v1`, no confirm-v2.

The Qwen full-fp32 CUDA run (Checkpoint 5) is still the next priority once a suitable GPU is available. This plan runs locally in the meantime.

## 1. Question

Is SmolVLM2-2.2B a valid third architecture for the unchanged 2.4 comparison (PGD-3 against matched-sparsity random masks at ε = 1/255)? Stage 0 answers only that. It uses the same gates as InternVL Stage 0 ([`../checkpoint-4-internvl/STAGE0_REPORT.md`](../checkpoint-4-internvl/STAGE0_REPORT.md)). Stage 1 attacks need separate approval.

## 2. Model facts (read today from the public repository; no weights fetched)

| | |
|---|---|
| Checkpoint | `HuggingFaceTB/SmolVLM2-2.2B-Instruct`, revision `482adb537c021c86670beed01cd58990d01e72e4`, Apache-2.0 |
| Architecture | `SmolVLMForConditionalGeneration`. SigLIP vision encoder (27 layers, 384 px, patch 14), pixel shuffle × 3, so **81 image tokens per 384 px tile**. Llama-type LM: 24 layers, hidden 2048, vocabulary 49,280. |
| **Stored dtype** | **float32**, 8.99 GB (about 2.25 B parameters). Qwen3-VL-2B is stored in bf16. So for SmolVLM2 **fp32 is the native precision and bf16 is a lossy down-cast**, the reverse of Qwen, where fp32 is an exact up-cast. Any precision comparison must state this. |
| Colour tokens | `red` 2382, `blue` 4461, `green` 2654, `yellow` 5724 (the space-prefixed forms). Same ids as SmolVLM2-500M in `dev_pilot.lock`. Checked in `tokenizer.json`. |
| Answer cue | `<end_of_utterance>\nAssistant:`, from the chat template. Same as `dev_pilot.lock`. Checked jointly with each colour in Stage 0, as for the other models. |
| Default preprocessing | `do_image_splitting=True`, longest edge 1536. A 512×384 image is upscaled to 1536×1152 and split into 4×3 = 12 tiles of 384 px plus a global 384 px image: **13 tiles, 1,053 image tokens**. |
| Single-tile option | `do_image_splitting=False`: one 384 px image, **81 image tokens**. This is the setting the study used for SmolVLM2-500M in Checkpoints 1.6–1.7. |
| Resampling | `resample=1` (PIL Lanczos) on uint8 images, so the official input is quantized again after resizing, as with InternVL. Gradients need a float surrogate of the same geometry, and its gap to the official tensor must be measured. |

**Prior result to keep in mind.** SmolVLM2-**500M** failed the competence gate on these scenes in Checkpoint 1.7 (32/96 neutral, 27/96 helpful, below 65%). The 2.2B model is a different, larger checkpoint, but it may also fail. If it does, Stage 0 stops there and the model does not qualify.

## 3. Feasibility on this machine (MacBook Air M4, 24 GB, MPS)

The memory safeguards are unchanged: the EMP-1 preload lines, the `path_probe` MemoryMonitor (10 GiB footprint abort, pressure and swap rules) and the MPS allocator capped at 10 GiB.

| Configuration | Weights | Expectation | Decided by |
|---|---|---|---|
| bf16, single tile | ≈ 4.5 GB | likely fits | measurement |
| bf16, 13 tiles | ≈ 4.5 GB | uncertain: 1,053 image tokens, against 3,328 for InternVL (failed) and 192 for Qwen | measurement |
| fp32, single tile | ≈ 9.0 GB | **likely over the 10 GiB cap** once activations and gradients are added | preload check; if it fails, fp32 is not loaded |
| fp32, 13 tiles | ≈ 9.0 GB | over the cap | not attempted unless fp32 single tile passes with room |

If fp32 does not fit, the fp32 arm becomes a GPU job. It could share the Checkpoint 5 pod later, with separate approval. The memory line is not raised.

## 4. Steps and gates (the InternVL Stage 0 set, unchanged)

0. *(free, no weights)* **Model-free preprocessing analysis** on all 96 `development_v4` images, hash-checked, for both tilings:
   - tile count and token count;
   - surrogate geometry equal to the processor's;
   - surrogate against official input (mean and maximum gap in levels, share of elements more than 0.5 level apart);
   - survival of dense and 30%-sparse ±1-level RGB perturbations into the official input, including after the bf16 cast.

   This needs only the processor and tokenizer files (a few MB).
1. *(needs the 8.99 GB weights download)* **Integration:** geometry, colour tokens, answer cue.
2. **Memory** per configuration, against the unchanged lines.
3. **Clean accuracy ≥ 75%** in each context (neutral, helpful_nonrevealing).
4. **Median clean margin ≥ 2 logits.**
5. **Image dependence:** a uniform gray image must not give the trained answer distribution.
6. **Scorer determinism:** bit-identical repeats.
7. **Gradient determinism and finiteness.**
8. **Finite differences**, as a diagnostic with the 5% line. The bf16 caveat from InternVL applies. Use the first 6 scenes of subset s16, neutral context, with the same step ladder.

Readout and data are unchanged from every earlier checkpoint: first answer-token row, full-vocabulary NLL of the four colours, margin = best other colour's NLL minus the label's; `development_v4`, 96 scenes in 48 counterfactual groups.

## 5. Code and records

- New package `src/smolvlm_probe`, built like `src/internvl_probe` (pins, pixels, common, backend, pixel_stage, model_stage, gradcheck). Frozen packages are reused, not edited.
- Records go in `docs/checkpoint-6-smolvlm2/stage0/`. Gradient tensors are git-ignored and kept locally.
- Model-free tests go in `tests/smolvlm_probe`.
- Branch `research/smolvlm2-stage0`, separate from PR #28 and from the Checkpoint 5 branch. No merge or tag.

## 6. Cost and time

- No paid resources.
- Disk: about 9 GB for the weights, in the study cache `/Users/fedecava/Development/models/huggingface`.
- Expected runtime: under 2 hours of local compute for steps 0–8 in bf16, based on InternVL Stage 0 and the 2.4 runtimes.

## 7. Decisions for the author

1. Approve step 0 (model-free; downloads only the processor and tokenizer files).
2. Approve the 8.99 GB weights download and steps 1–8, locally, under the unchanged memory lines.
3. Confirm that both tilings are measured and reported side by side, as for InternVL, with neither replacing the other.
