# Checkpoint 5 report: Qwen3-VL-2B, bf16 vs full fp32 on one CUDA GPU (2026-11-10)

**Status: EXPLORATORY, development data only** (the 48 arm-a scenes of `development_v4`). No held-out data. The protocol ([`PLAN.md`](PLAN.md)) and the analysis script were frozen before any GPU run. The infrastructure amendments ([1](AMENDMENT_1_HARDWARE.md)–[4](AMENDMENT_4_L40S_STARTUP.md)) changed hardware and operations only. The analysis was run once, on the committed records (`7ed0992`): [`analysis/qwen_precision_analysis.json`](analysis/qwen_precision_analysis.json).

## 1. Answer

**Pre-registered reading A: both effects persist in full fp32.**
- *Selection advantage:* PGD-3 beats matched-sparsity random masks.
- *Clean-gradient saliency inversion:* top-|∇| coordinates do worse than random ones.

Both are essentially unchanged when the whole model runs in fp32. Every paired fp32 − bf16 difference is **equivalent within the pre-registered margins** (90% CI inside ±1.0 for selection, ±0.5 for top-k − random). The confirmed 2.4 result is not an artifact of bf16 evaluation.

## 2. Setup

| | |
|---|---|
| Hardware | NVIDIA A40 48 GB (Ampere, sm_86), RunPod Secure Cloud, CA-MTL-1, $0.59/h; pod `s3uabwvn1a10q9` |
| Software | torch 2.15.0+cu132 (CUDA 13.2), transformers 5.22.1; commit `a940c46` |
| Model | Qwen3-VL-2B-Instruct, weights sha256 `7de1838c…` (stored in bf16, so fp32 is an exact up-cast) |
| Precisions | bf16 (the 2.4 setting) and full fp32, both loaded on the same GPU, per scene bf16 then fp32 |
| Numerical controls | TF32 off (fp64 probe max rel. err 5.3e-7), no autocast, deterministic algorithms, eager attention |
| Protocol | unchanged 2.4 `scene_block`: ε = 1/255, FGSM, PGD-3, top-k and 8 matched random masks; two contexts; bootstrap 9,999 draws over scenes, seed 20261112 |

## 3. Gates (all pass)

| Gate | Result |
|---|---|
| G1 environment, TF32 off | pass |
| G2 dtype audit | pass: bf16 shows only the known fp32 rotary-embedding island; fp32 shows no other dtype |
| G3 inputs and weights | pass: all 96 prompt/pixel hashes equal the local reference |
| G4 memory (≤ 36 GiB reserved) | pass: peak 15.5 GiB in the paired unit |
| G5 CUDA bf16 vs MPS bf16 clean margins | **pass, at the limits**: median abs 0.2500 (limit 0.25), p95 0.7500 (limit 0.75), max 1.00 (limit 1.5), chosen colour equal on 96/96. These sit exactly on bf16's 0.125-logit grid. The pass is by the pre-registered rule, with no margin to spare. |
| G6 fp32 clean accuracy ≥ 75% | pass: 93.8% neutral, 97.9% helpful |
| G7 fp32 finite differences ≤ 5% | pass: median relative error 0.4% at a step of 1/256 level |
| G8 determinism | pass |
| G9 projected spend ≤ $12 | pass: $0.87 projected after scene 1, $0.92 at resume |
| Throughput | pass: 63.9 s per paired scene; finished at 21:11Z against a 00:08Z workload deadline |
| Drift (scene 1, preflight process vs resumed process) | exact in both precisions |
| Reproduction band (CUDA bf16 neutral selection in [4.3, 7.3], CI > 0, top-k − rand upper < 0) | met |
| Attack structure | 96 units per precision, no sign reversals, perturbations within one 8-bit level |

## 4. Results (source context, 48 scenes; estimate [95% CI])

| Quantity | bf16 | fp32 | **fp32 − bf16 (paired)** | Verdict |
|---|---|---|---|---|
| Selection, neutral | 5.86 [5.04, 6.68] | 5.89 [5.16, 6.63] | **+0.03 [−0.52, 0.56]**, 90% [−0.42, 0.47] | equivalent within ±1.0 |
| Selection, helpful | 6.33 [5.66, 6.99] | 5.92 [5.38, 6.50] | **−0.41 [−0.95, 0.11]**, 90% [−0.86, 0.03] | equivalent within ±1.0 |
| Selection, pooled | 6.10 [5.43, 6.76] | 5.91 [5.35, 6.49] | **−0.19 [−0.61, 0.22]**, 90% [−0.54, 0.16] | equivalent within ±1.0 |
| Top-k − random, neutral | −1.60 [−2.18, −1.07] | −1.70 [−2.20, −1.23] | **−0.11 [−0.49, 0.27]**, 90% [−0.42, 0.21] | equivalent within ±0.5 |
| Top-k − random, helpful | −1.54 [−1.96, −1.17] | −1.75 [−2.18, −1.33] | **−0.20 [−0.47, 0.05]**, 90% [−0.42, 0.01] | equivalent within ±0.5 |
| Top-k − random, pooled | −1.57 [−2.04, −1.15] | −1.73 [−2.16, −1.31] | **−0.15 [−0.41, 0.09]**, 90% [−0.37, 0.05] | equivalent within ±0.5 |
| PGD-3 − FGSM total, pooled | 6.58 [5.97, 7.25] | 6.49 [5.91, 7.13] | −0.09 [−0.48, 0.32] | – |
| Share of the gap from sparsity, pooled | 0.07 [0.01, 0.14] | 0.09 [0.02, 0.15] | – | – |

Selection, the advantage of PGD-3 over random masks of the same sparsity, is about 6 logits in both precisions, and its CI excludes 0 in each. Saliency inversion, top-|∇| worse than random, is about −1.6 to −1.7 in both, CI below 0 in each. Sparsity alone explains under 10% of the PGD-3 − FGSM gap in both precisions.

**Other measurements:**
- *Clean behaviour:* clean accuracy is identical across precisions (93.8% / 97.9%), and mean clean margins differ by under 0.06 logits.
- *Official vs float scoring paths:* identical for every FGSM and PGD-3 scoring in both precisions (max abs 0).
- *CUDA bf16 against the committed MPS bf16 run:* selection +0.06 [−0.51, 0.62] (neutral) and +0.23 [−0.10, 0.58] (pooled); top-k − random +0.02 [−0.15, 0.19] (pooled).

## 5. A nuance: per-scene gradients are precision-sensitive, the effects are not

For the same scene and context, the bf16 and fp32 clean gradients agree only loosely:
- mean cosine 0.26–0.34 (range −0.54 to 0.94);
- sign agreement about 73%;
- top-30% overlap 0.68–0.69 ([`run/gradient_similarity.json`](run/gradient_similarity.json)).

The paired attack-level effects are nonetheless equivalent. So the effects do not depend on reproducing a particular per-scene gradient direction. This run does not measure why. It only shows that the aggregate comparison is stable across precisions.

## 6. What this does and does not show

- **Shows:** on Qwen3-VL-2B, with these scenes, evaluating the model in full fp32 instead of bf16 changes neither the selection advantage nor the saliency inversion beyond the pre-registered margins.
- **Does not show:** behaviour of other models, of other ε, or of held-out scenes in fp32. This is development data only. The held-out 2.4 confirmation remains the study's evidence; this run tests one alternative explanation of it.
- **Limitations:**
  - G5 passed exactly at its limits.
  - The hardware is an A40, not the MPS device of the 2.4 study. The CUDA bf16 run reproduced the MPS bf16 effects within the CIs above.
  - One run on one card. No cross-card replication was attempted.

## 7. Cost and operations

- **This run:** 72 min of A40 time, ≈ $0.71 expected. RunPod had posted $0.15 at 21:14Z, with posting still catching up.
- **All of Checkpoint 5's compute today:** $0.55 posted so far (two RTX PRO 6000 pods $0.21 and $0.18, CPU tests $0.01, this pod partially). That is well under the $20 ceiling.
- **Network volume `0uhv9a501u`:** unchanged, about $0.02 so far today.
- **Outputs:**
  - exported while the pod was running;
  - verified against the pod's SHA256SUMS (115/116; `pipeline.log` grows after the manifest is written, and matches up to that point);
  - checked against the hashes printed in the pod's log stream.

  The pod was then stopped and terminated (404, `list-pods` empty).
- **Tensors:** 1.3 GB, kept locally under `c5-export/` and git-ignored.
- **Attempt history:** [`DEPLOYMENT_STATUS.md`](DEPLOYMENT_STATUS.md).

## 8. Recommendation

The precision question for Qwen is answered. No further fp32 work on this model is needed for the publication's claim. If more compute is spent, the higher-value next step is a held-out replication on a second competent architecture. The candidates screened on 2026-11-10 either qualified only with caveats (InternVL3.5-1B, single tile) or failed competence (SmolVLM2-2.2B).
