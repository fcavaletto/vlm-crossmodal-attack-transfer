# Checkpoint 4, Stage 1 protocol: the 2.4 comparison on InternVL3.5-1B (2026-11-10)

**Status: EXPLORATORY, development data only.** Approved by the author on 2026-11-10. This protocol, the runner [`stage1.py`](../../src/internvl_probe/internvl_probe/stage1.py) and the analysis [`analyze_stage1.py`](analyze_stage1.py) are committed and pushed before the first attack. The runner refuses to start otherwise.

Nothing here is a confirmation. `trackB_confirm_v1` is not read, no new split is generated, no paid resource is used, and no frozen file or record is changed.

## 1. Questions

- **Q1.** Does the 2.4 coordinate-selection effect (PGD-3's coordinates beat same-count random coordinates of FGSM's perturbation) appear on a second architecture?
- **Q2.** Does it persist when that architecture runs in full fp32?
- **Out of scope.** Whether full fp32 removes Qwen3-VL-2B's own effect. An InternVL bf16-versus-fp32 result cannot establish the cause of Qwen's behaviour.

## 2. Model and preprocessing

| | |
|---|---|
| Model | `OpenGVLab/InternVL3_5-1B-HF` at `9191dbccf312b537016f041b25d61c72e7c5c9f3` |
| Preprocessing | **Single tile (448×448), not the processor's default 13-tile configuration**, which fails the 10 GiB memory line (Stage 0). No result is to be described as a replication under the standard 13-tile configuration. |
| Precisions | bf16 (weights and input), and full fp32. Separate processes, same device (MPS), same code, same scenes, budget, schedule and seeds. |
| Gradient path | Float surrogate of the official geometry without the uint8 rounding after the resize (Stage 0 §3) |

## 3. Data

The 48 arm-a scenes of `development_v4`, one per counterfactual group, in group order.

## 4. Attack and conditions (unchanged from 2.4)

- **Budget and gradient.** ε = 1/255 in RGB [0, 1]. Gradient of the label NLL.
- **FGSM.** `project_signed_step(x, g0, ε, "positive")`.
- **PGD-3.** Three steps of ε/4 with `pgd_update`, each step's gradient taken at the current float iterate.
- **Rounding.** Every perturbed image is rounded to 8 bits, as the official scorer stores it.
- **Masks.** These use FGSM's rounded perturbation r_F on exactly k coordinates of its support, where k is PGD-3's rounded support size:
  - **top-\|∇\|:** the k largest \|g0\|;
  - **random:** 8 random masks, seed = `scene_seed("checkpoint-4-stage1-mask|<source>|seed<s>", scene)`.
- **Attack sources.**
  - **Primary:** neutral.
  - **Secondary:** helpful_nonrevealing, fixed here before any attack. The measured Stage 0 cost is 37–44 s per (scene, source) unit, about 1 h per precision for both sources.
  - **Order:** all 48 neutral units run first, then all 48 helpful units. An interruption therefore costs secondary units first.
- **Scoring.** Every perturbed image (FGSM, PGD-3, top-\|∇\|, 8 random) is scored in **both contexts** on **both paths**:
  - the official processor path (8-bit image, uint8 resize);
  - the float path (the same 8-bit image through the surrogate, with no second rounding).
  - Clean images are scored the same way.
- **Diagnostics, no extra cost.** Cosine and sign agreement between the clean gradient g0 and PGD's step-2 and step-3 gradients g1 and g2, and between g1 and g2.

## 5. Integrity and memory

- **Integrity.**
  - Subset structure recorded per unit (support sizes, sign reversals, levels, coordinates outside FGSM's support).
  - Random mask counts must equal k.
  - The first scene's clean official score is re-computed at the end and must be bit-identical.
- **Memory safeguards: Stage 0's, unchanged.**
  - EMP-1 preload lines.
  - MemoryMonitor: 10 GiB footprint abort, pressure, available-memory and swap rules.
  - MPS allocator capped at 10 GiB.
- **Memory recording.** Footprint is recorded after every unit. The first complete unit's memory is stored separately for each precision.
- **On any safeguard or error, the run stops and is reported.** No threshold is raised. Precision and numerical settings are not changed. Nothing is retried automatically. Records are appended after every unit, so completed observations are kept. There is no resume and no overwrite.
- **Time cap:** 4 h per precision. A truncated run says so.

## 6. Quantities (analysis script, committed now)

D = clean margin − perturbed margin, on the same path and in the scored context. The primary quantities use the source context and the official path, with neutral as source; helpful and the pooled mean of both sources are secondary. All quantities are computed for each precision and each path.

| Quantity | Definition |
|---|---|
| Selection advantage | D_PGD3 − mean over 8 seeds of D_rand |
| Sparsity | D_rand − D_FGSM |
| Sparsity share of the PGD–FGSM gap | (mean D_rand − mean D_FGSM) / (mean D_PGD3 − mean D_FGSM) |
| Top-\|∇\| vs random | D_topk − D_rand |
| FGSM vs PGD | D_PGD3 − D_FGSM, and mean D_PGD3 / mean D_FGSM |
| Gradient alignment | cos(g0, g1), sign agreement(g0, g1), cos(g0, g2), cos(g1, g2) |
| **Precision difference** | per scene: selection(fp32) − selection(bf16), and the same for the other quantities |
| **Path difference** | per scene and precision: official − float, for every method's D and for selection |
| Cross-model, matched scenes | per scene: InternVL (official) − Qwen3-VL-2B bf16 (Checkpoint 2.3/2.3b records on the same 48 scenes, reproduced exactly by this script) |

**Uncertainty.** Percentile bootstrap over scenes, 9,999 draws, seed 20261111.
- Every paired quantity is formed per scene before resampling. So the dependence between precisions, paths, sources and contexts is kept.
- **A precision effect is read only from the paired difference and its interval**, never from one arm being significant and the other not.

Flip rates on clean-correct scenes are descriptive.

## 7. Readings, fixed now

Selection is "present" in a cell when its 95% CI lower bound is above 0. Primary cell: neutral source, source context. The readings are not exclusive, and each is reported as true or false.

| Reading | Rule | Interpretation allowed |
|---|---|---|
| R1 persists in both precisions | present in bf16 and in fp32, official path | Evidence that on this second architecture the effect is not solely caused by bf16 |
| R2 much stronger in bf16 | paired (fp32 − bf16) selection CI entirely below 0, and the fp32 estimate below half the bf16 estimate | Evidence of a numerical contribution on InternVL. **Not** an explanation of Qwen's result. |
| R3 only after official preprocessing | per precision: present on the official path, not present on the float path, and paired (official − float) selection CI above 0 | The resize and quantization pipeline matters |
| R4 does not replicate | not present in any precision on any path | A potential architectural or preprocessing boundary. Reported as such. |
| Where selection works | per precision: both paths, official only, float only, or neither | Descriptive |

- **Mixed outcomes.** A pattern that matches none of these, or contradicts across sources, is reported as mixed or ambiguous, as it is.
- **Thresholds.** The 2.4 thresholds (share ≤ 0.25, top-\|∇\| below random) are reported descriptively only. They are not decision rules here.
- **Hypotheses are not changed after the data are seen.** Any new idea from this run is labelled post hoc.

## 8. Outputs

- `stage1/{bf16,fp32}/records.jsonl` and `progress.json`, committed. Tensors are git-ignored.
- `analysis/stage1_analysis.json`, from one run of `analyze_stage1.py` on the complete records.
- A comparative report with Qwen3-VL-2B. It keeps matched measurements (same scenes, design and scorer logic) separate from differences in architecture, language model family, preprocessing (Qwen has no resize; InternVL is single-tile with a second rounding), attention implementation (Qwen eager, InternVL SDPA) and precision.
