# Checkpoint 5 plan: does Qwen3-VL-2B's confirmed effect survive full fp32? (2026-11-10)

**Status: PLAN FOR REVIEW, infrastructure revised 2026-11-10 (network storage, termination). Nothing has been launched, created or bought.**
- Prices are live reads made on 2026-11-10 (§7).
- This plan asks for approval of a GPU pilot with a hard ceiling of $20 (§8).
- Everything in §5–§6 is fixed before any inference, and is frozen with the runner and analysis code before the first paid hour (§9, step 0).
- Exploratory, development data only. No confirmation split is read or generated.

## 1. Question

Do Qwen3-VL-2B's confirmed **coordinate-selection advantage** and **clean-gradient saliency inversion** survive full-fp32 computation?

The comparison is bf16 against fp32 on **the same CUDA GPU**. Scenes, prompts, attack parameters, preprocessing, scoring rules and analysis definitions are identical in both. The only change is the dtype of weights and activations.

Why it matters now:
- The InternVL3.5-1B run (Checkpoint 4) found selection in both precisions, but not the saliency inversion.
- So the inversion is the part of the Qwen result most likely to be numerics- or model-specific.
- Qwen in full fp32 has never been run: the 2.3 "V32" arm changed only the vision tower.

## 2. What fp32 can and cannot change (numerical controls)

**The checkpoint is stored in bf16.** I read the `model.safetensors` header at the pinned revision `89644892`: 625 tensors and 2,127,532,032 parameters, **all BF16**. So an "fp32 model" here is the *same* bf16 weights upcast exactly to fp32.
- It does **not** recover fp32 weight information. None exists.
- What it removes is reduced precision in **computation**: the bf16 input cast, bf16 activations, and bf16 rounding of intermediate sums and residual streams.
- That makes the comparison cleaner: weights are bit-identical in value, so any difference comes from arithmetic. It also means the experiment cannot say what a natively fp32-trained model would do.

| Control | How | Check (gate in §6) |
|---|---|---|
| No TF32 | `torch.backends.cuda.matmul.allow_tf32 = False`, `torch.backends.cudnn.allow_tf32 = False`, `torch.set_float32_matmul_precision("highest")` | fp32 matmul against an fp64 reference: max relative error < 1e-5. TF32 gives about 1e-3. |
| No autocast | No `autocast` context anywhere; assert `torch.is_autocast_enabled()` is false before every pass | asserted in code |
| No silent down-casts | Forward hooks on every module during the smoke run record the dtype of each floating-point output | fp32 model: every floating output fp32. Exceptions are listed, not assumed. bf16 model: the fp32 islands (for example eager attention's fp32 softmax) are listed and compared with the MPS run. |
| Same graph and attention | `attn_implementation="eager"` in both precisions, as in every Qwen run since 1.8. The same frozen `from_pretrained` arguments with only `dtype` changed. The same frozen scoring and gradient functions (`attack_feasibility.score.*`, `qwen_rgb.rgb_path.differentiable_model_pixels`), called unchanged. | config recorded |
| Determinism | `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, fixed seeds | repeat gradient and score bit-identical in each precision |
| Same inputs | Preprocessing runs on CPU in both modes: the pinned processor, with no resize at 512×384 | `pixel_values` and `input_ids` hashes on the pod equal the local ones, for all 96 prompts |
| Same weights | sha256 of `model.safetensors` on the pod equals the local file | gate |

**Cross-device reproduction.** MPS and CUDA are not expected to agree bit-for-bit. Tolerances are fixed in §6, gate G5.

## 3. Engineering: a new CUDA runner, no frozen file edited

New package `src/qwen_precision/`. It reuses the frozen, device-agnostic functions by import only, without changing them, and adds:
- **Loader.** Same pinned id and revision, eager attention, `dtype ∈ {bf16, fp32}`, `device ∈ {mps, cuda}`. The frozen loader is bf16-on-MPS-only by design.
- **CUDA memory guard** (§8). The frozen `path_probe.memory` monitor is macOS-only.
- **Runner.** The unchanged 2.4 scene block: per scene and source,
  - clean scores and the clean gradients L (NLL) and M (margin) in both contexts;
  - FGSM, PGD-3 at ε/4, top-\|∇\| and 8 matched-count random masks;
  - every image scored officially in both contexts.
  - New random-seed prefix `checkpoint-5-qwen-precision-mask`.
- **Model-free diagnostics.** The numerical audits of §2, and finite differences as in Checkpoint 4 Stage 0.

**Local check before any paid hour (free).**
- Run the new runner on MPS in bf16 on one development scene.
- Require **bit-exact** agreement with the frozen 2.4 development smoke and the 2.3/2.3b records for clean, FGSM, PGD-3, top-\|∇\| and random-mask scores: same device, same functions, same seeds (for that check, the 2.3b seed prefix is used).
- This proves the new runner *is* the 2.4 pipeline before it moves to CUDA.

**Order on the GPU: interleaved.** Both models are loaded at once (bf16 ≈ 4 GB, fp32 ≈ 8.5 GB). Each scene runs the frozen `scene_block` (both attack sources) in bf16 and then in fp32 before the next scene. So an interruption leaves complete *pairs*, and slow drift of the machine cannot separate the arms.

*Revision, 2026-11-10, implementation detail only.* The first draft ran the neutral source for all scenes before the helpful one. The frozen `scene_block` computes both sources of a scene together, and calling it unchanged was preferred over splitting it. The scene order is s16 first, then the other 32 arm-a scenes, as in 2.3. No scientific quantity changes.

**Status of the local check (done 2026-11-10):**
- On MPS bf16, scene `dev4-ref-00-a`, the new runner is **bit-exact** with the 2.3/2.3b records: clean scores, all 8 FGSM/PGD codes, all 4 clean gradients and all 22 attack scores.
- Runtime was 203.7 s (2.4: 203 s per scene), with a peak footprint of 8.37 GiB.
- Qwen's official and differentiable pixel tensors are bit-identical (max difference 0.0), and the float-path margins of FGSM and PGD-3 equal the official ones. So for Qwen at this size the "official vs float" comparison is a check, not a separate experiment.
- Records: [`repro_mps/repro_mps_check.json`](repro_mps/repro_mps_check.json). References for G2/G3: [`reference/`](reference).

## 4. Data and design

- **Scenes.** The 48 arm-a development scenes (`development_v4`), the same scenes as Checkpoint 2.3/2.3b and Checkpoint 4.
- **Attack sources.** neutral (primary) and helpful_nonrevealing (secondary), as in 2.4. Both are computed per scene (§3).
- **Budget.** ε = 1/255. FGSM; PGD-3 with ε/4 steps; 8-bit rounding.
- **Official vs differentiable path, "where appropriate".** Qwen's processor does not resize a 512×384 image, so its differentiable path *is* the official input up to float error. The 2.1 check bounded this at 1e-3, unlike InternVL, which has a second rounding. Therefore:
  - clean, FGSM and PGD-3 images are also scored on the float path in both contexts, and the maximum pixel-tensor difference is recorded;
  - masks are scored on the official path only, as in 2.4.
- **Gradient comparisons**, no extra passes:
  - Clean L and M gradients from both precisions are saved. Offline: cosine, sign agreement, norm ratio, top-30% overlap, per scene and context.
  - Each CUDA precision is also compared with the stored MPS bf16 gradients for the same scenes.
  - PGD's step-2 and step-3 gradients give the clean-to-next-step alignment: cos and sign agreement of (g0, g1), (g0, g2), (g1, g2).

## 5. Pre-specified analysis

D = clean official margin − perturbed official margin in the scored context, in logits. Definitions are the 2.4 ones, reused from the Checkpoint 4 analysis code, which reproduces the committed Qwen development numbers exactly.

**Primary comparisons**: neutral source, source context, official path.

| | Quantity | Reported |
|---|---|---|
| **P1** | Δsel = sel(fp32) − sel(bf16), where sel = D_PGD3 − mean of 8 D_rand | estimate, 95% CI, 90% CI |
| **P2** | Δtopk = (D_topk − D_rand)(fp32) − (D_topk − D_rand)(bf16) | estimate, 95% CI, 90% CI |
| levels | sel and topk − rand in each precision | estimate, 95% CI |

**Uncertainty.** Percentile bootstrap over scenes, 9,999 draws, seed 20261112.
- Each scene's paired difference is formed first, so resampling keeps the dependence between precisions, sources and contexts.
- P1 and P2 are reported without multiplicity adjustment, and the report says so.

**Equivalence, fixed now.** Failing to reject a difference is not evidence of equivalence. Equivalence is claimed only if the **90% CI of the paired difference lies inside the margin**: two one-sided tests at α = 0.05.

| | Margin | Rationale |
|---|---|---|
| Δsel | ±1.0 logit | 17% of Qwen's development selection (+5.80); under half the InternVL−Qwen gap (2.4) |
| Δtopk | ±0.5 logit | 30% of Qwen's development inversion (−1.65) |

A difference that is neither clearly non-zero nor within the margin is reported as **inconclusive**.

**Secondary**, descriptive:
- the helpful source and the pooled mean, and the target context;
- D_FGSM and D_PGD3, the PGD/FGSM ratio and the sparsity share;
- flip rates on clean-correct scenes;
- the alignment quantities;
- cross-precision gradient similarity;
- official minus float for clean, FGSM and PGD-3;
- clean accuracy and margins in each precision.

## 6. Decision criteria

### 6.1 Pilot gates (smoke and benchmark, before the main run)

If any gate fails, the run **stops and is reported**. No threshold is changed on the pod.

| Gate | Criterion |
|---|---|
| G1 Environment | TF32 flags off. fp32 matmul vs fp64: max relative error < 1e-5. No autocast. Deterministic algorithms on. GPU name, driver and CUDA version recorded. |
| G2 Dtype audit | fp32 model: every floating module output is fp32. bf16 model: its fp32 islands match the set seen on MPS (recorded locally first). |
| G3 Inputs | `pixel_values` and `input_ids` hashes equal the local ones for all 96 prompts. Weights sha256 equal. |
| G4 Memory | One full scene in each precision, both models resident: peak reserved ≤ 36 GiB on a 48 GB card (§8) |
| G5 Cross-device clean scores | CUDA bf16 vs MPS bf16 official clean margins, 96 prompts: median \|Δ\| ≤ 0.25 logit, 95th percentile ≤ 0.75, max ≤ 1.5. Chosen colour agrees on ≥ 94/96. |
| G6 fp32 sanity | fp32 clean accuracy ≥ 75% in each context. This is a sanity line, not a target: fp32 accuracy itself is a result. |
| G7 fp32 gradients | Finite differences on 6 scenes × 4 random directions, steps 1/1024–1/16 level: median relative error ≤ 5% at the best step, with error falling as the step shrinks. This is a **diagnostic**: on a miss, check step dependence, TF32 and the dtype audit before concluding anything (as in Checkpoint 4). |
| G8 Determinism | A repeated gradient and score are bit-identical in each precision on CUDA. Otherwise the size of the difference is recorded and the run stops for review. |
| G9 Budget | From measured throughput: elapsed spend plus projected remaining ≤ $12. Otherwise stop and ask, with the measured numbers. |

### 6.2 Reproduction band (main run, defines outcome D)

CUDA bf16 counts as reproducing the MPS-confirmed phenomenon if both hold:
- neutral-source selection estimate within **[+4.3, +7.3]** (the MPS development +5.80 ± 1.5), with 95% CI above 0;
- topk − rand 95% CI upper bound below 0.

If it does not, the fp32-versus-bf16 comparison on CUDA is still reported. It is labelled D, because it no longer tests the effect that was confirmed on MPS.

### 6.3 Outcomes, fixed now

"fp32 selection present" means fp32 sel 95% CI above 0.

| | Rule | Establishes | Does not establish | Strongest follow-up |
|---|---|---|---|---|
| **A** Both persist | Reproduction band met. fp32 sel present. fp32 topk − rand 95% CI upper < 0. Not C. | Both confirmed Qwen findings are properties of the model's function, not of bf16 arithmetic. P1/P2 equivalence statements, if they hold, bound any precision contribution. | That the effects are general (InternVL already lacks the inversion), or *why* PGD's coordinates win | SmolVLM2-2.2B (a different language-model family), then a mechanism study: path-specific curvature along PGD's steps, in fp32 |
| **B** Selection persists, inversion disappears | Band met. fp32 sel present. fp32 topk − rand CI includes 0 or lies above it. P2 Δtopk 95% CI lower > 0. | The inversion depends on bf16 on Qwen. The 2.4 S2 claim must be qualified as "under bf16 computation". Selection is a model property. | That bf16 causes saliency inversion in general, or where in the network it arises | A component-precision ablation on the same GPU: fp32 vision + bf16 LM, and bf16 vision + fp32 LM. This localizes the precision effect (about $1–2 more, needs approval). |
| **C** Both substantially weaker | Band met. fp32 estimates of sel *and* of \|topk − rand\| each below 50% of bf16, with P1 and P2 95% CIs excluding 0 | A substantial numerical contribution to the confirmed result. The 2.4 interpretation must be re-framed around evaluation precision. | That the effects vanish in fp32, or that the same holds for other models (InternVL says it does not for selection) | The component-precision ablation, plus an ε sweep (2/255, 4/255) in both precisions, to see whether the effect tracks the bf16 resolution of the input |
| **D** Uncontrolled numerical change | Any of: G1–G3 or G8 fails unresolved; G5 fails; reproduction band missed; fp32 gradients fail G7 after investigation | That the CUDA run is not a clean test of the MPS-confirmed effect. Which component differs (kernels, attention, reductions) is identified as far as the audits allow. | Anything about the precision question | Diagnose on a few scenes: CPU fp32 reference forwards, and kernel-by-kernel comparison of MPS and CUDA bf16 activations |
| Mixed | Anything else, for example selection weaker but inversion intact | Reported as observed. No outcome label is forced onto it. | | Decided after review, labelled post hoc |

Hypotheses are not changed after seeing the data. Anything new is labelled post hoc.

## 7. Provider comparison (2026-11-10)

All RunPod figures come from a live `list-gpu-types` read today, plus the plugin's storage and billing references (per-second billing). Other providers come from web sources found today, not live account reads.

| Provider and card | VRAM | $/h | Availability (read today) | Storage | Notes |
|---|---|---|---|---|---|
| **RunPod Secure, A40** | 48 GB | **0.59** | first read MEDIUM (CA-MTL-1, EU-SE-1); re-read restricted to Sweden: LOW in EU-SE-1, CUDA 13.2 hosts | Network volume ≈ $0.07/GB-month (§8.1); container disk billed while running | **Recommended**, in EU-SE-1 with the network volume. Ampere fp32 without TF32. |
| RunPod Secure, RTX A6000 | 48 GB | 0.59 | LOW: EU-SE-1 | same | Same GA102 architecture; the fallback in EU-SE-1 |
| RunPod Secure, L40S | 48 GB | 1.09 | LOW: EUR-IS-2, US-NC-1 | same | Not in EU-SE-1, so not usable with the planned volume |
| RunPod Secure, RTX 6000 Ada | 48 GB | 0.99 | LOW: US-WA-1 | same | |
| RunPod Community, A40 / A6000 | 48 GB | 0.35 / 0.33 | community hosts | same | About $1.5 cheaper over the run. Less predictable hosts: a mid-run host loss costs more than it saves. |
| Lambda, A6000 | 48 GB | ≈ 0.80–1.09 (web) | not checked | | No live read. Sources disagree. |
| Vast.ai, A40 / A6000 | 48 GB | ≈ 0.28–0.40 (web, marketplace) | not checked | storage rate not found | Marketplace hosts vary widely. Not recommended for a precision study that needs a stable, documented machine. |
| RunPod Secure, RTX A5000 / 4090 | 24 GB | 0.27 / 0.89 | LOW | | fp32 projected peak ≈ 17 GiB for one model. Too little headroom for both models resident and the 36 GiB-equivalent guard. |

Sources for the non-RunPod rows:
- [Lambda pricing](https://www.lambda.ai/pricing), [Lambda cloud](https://lambdalabs.com/cloud), [ComputePrices: Lambda A6000](https://computeprices.com/providers/lambda/gpus/rtxa6000)
- [ComputePrices: Vast A40](https://computeprices.com/providers/vast/gpus/a40), [Beam: RTX A6000 pricing](https://www.beam.cloud/blog/nvidia-rtx-a6000-pricing)

## 8. Budget, safeguards and stop conditions

### Runtime and cost estimate

The MPS baseline is measured: 2.4 took **203 s per scene** in bf16, both sources (8 gradients and about 50 forwards). This plan adds float-path scoring of 3 images per source. The CUDA speed is **not measured**; the benchmark measures it before the main run.

| Phase | Pod hours, low | Pod hours, high |
|---|---|---|
| Setup: image, pinned pip install, 4.3 GB pinned weights, hash checks | 0.3 | 0.6 |
| Smoke and benchmark: G1–G9, 96 clean scorings × 2, FD on 6 scenes, one full scene | 0.4 | 0.8 |
| Main bf16, 48 scenes (assumes 40–100 s per scene) | 0.5 | 1.3 |
| Main fp32, 48 scenes (assumes 80–250 s per scene) | 1.1 | 3.3 |
| Export and teardown | 0.2 | 0.3 |
| **Total** | **2.5 h** | **6.3 h** |

| | A40 Secure at $0.59/h, EU-SE-1 |
|---|---|
| GPU compute, expected range | **$1.5–3.7** |
| Container disk: 30 GB for about 7 h (billed while the pod runs) | < $0.05 |
| With a 50% buffer (≈ 9.5 h) | ≈ $5.6 |
| Worst case: fallback to RTX A6000 at the same price, 9.5 h | ≈ $5.6 |
| Projected-spend gate (G9) | **$12** |
| **Maximum authorized compute** | **$20** |
| Network volume, separate and ongoing | 30 GB × ≈ $0.07/GB-month ≈ **$2.10 per month**, from creation until deletion, whether or not a pod is attached |

L40S is no longer a fallback: it is not in EU-SE-1, and the network volume pins every pod to the volume's data center (§8.1).

### 8.1 Storage: one reusable network volume (revised 2026-11-10, at the author's request)

| | |
|---|---|
| Volume | 30 GB, **STANDARD** tier, new, in **EU-SE-1** |
| Mount | `/workspace` on every pod that uses it |
| Layout | `/workspace/hf-cache` (`HF_HOME`; the pinned Qwen snapshot ≈ 4.3 GB), `/workspace/checkpoint-5/` (records, audits, tensors ≈ 1–2 GB), `/workspace/manifests/` (weight and input hash references) |
| Not on the volume | The software environment. It is installed on the container disk from the pinned requirements and the repository commit at every pod start, so it stays reproducible from code. |
| Replaces | The planned 20 GB pod volume disk. A pod mounts a network volume *or* a volume disk, never both, so nothing is paid twice. |
| Kept after the run | Yes. Terminating the pod detaches the volume without deleting it. It keeps billing until the author deletes it. |

**Why EU-SE-1 (live reads, 2026-11-10).**
- *Network volumes with standard storage:* `list-data-centers` reports 16 data centers.
- *Which of those have a 48 GB card in stock:* only EU-SE-1 (RTX A6000, LOW) and US-NC-1 (L40S, LOW).
- *The specific cards:* `list-gpu-types` restricted to Sweden shows Secure-cloud **A40 LOW** and **RTX A6000 LOW**, both in EU-SE-1, both on CUDA 13.2 hosts, both $0.59/h.
- So EU-SE-1 is the only data center with network storage and both recommended cards in stock today.
- *This account today:* no network volumes and no pods (`list-network-volumes` and `list-pods` both empty).

**Reuse for later experiments, and the limitation.**
- *Fits in EU-SE-1:* SmolVLM2-2.2B, and more Qwen or InternVL arms, all fit a 48 GB Ampere card.
- *Does not fit:* EU-SE-1 showed no 80 GB card (A100, H100, H200) in today's reads. A later experiment on a 7B-class model in fp32 would need another data center.
- *Moving is cheap:* the volume holds about 6 GB of re-creatable data (pinned weights re-download in minutes, and results are also kept locally). So a second volume in another data center can be created then, and this one deleted.
- *Stock is LOW for both cards.* If neither is free at launch, the options are waiting, or the author choosing another data center. I will not move the volume or switch cards on my own.

### 8.2 Automatic termination (revised 2026-11-10)

**Finding: RunPod has no provider-side deadline that terminates a pod.** I checked it before planning around `--terminate-after`, as the author asked.
- *REST API:* REST v2, which the MCP tools and `runpodctl` use, has no stop or terminate timer field ([Manage Pods](https://docs.runpod.io/pods/manage-pods); REST v2 `CreatePodRequest`).
- *The CLI flags:* RunPod's official agent guide states that `runpodctl pod create --stop-after` and `--terminate-after` **were never enforced by the backend**, so a pod kept running and billing past its deadline. runpodctl v2.12.0 removed both flags.
- *Conclusion:* I do not plan on `--terminate-after`.

**What does exist.** Every pod ships with `runpodctl` and a **pod-scoped API key**, preinstalled by RunPod ([runpodctl overview](https://docs.runpod.io/runpodctl/overview)). So a deadline needs no key of ours inside the pod. The docs show this key *stopping* the pod (`runpodctl pod stop $RUNPOD_POD_ID`). I found no statement that it may *terminate* (remove) the pod.

**Design.**

1. **In-pod deadline, started first in the pod's start command, before any setup:**
   `(sleep 28800; runpodctl pod remove "$RUNPOD_POD_ID" || runpodctl pod stop "$RUNPOD_POD_ID") &`
   - This is 8 h from container start, using RunPod's own pod-scoped key.
   - The runner's own deadline (`--deadline-epoch`) is 7.5 h, so records are flushed first.
   - If `remove` is refused, `stop` still ends GPU billing. The network volume keeps billing as planned; a stopped pod with a network volume has no volume disk.
2. **Verify termination before the GPU run, at negligible cost.**
   - Create a CPU-only pod with no volume, whose start command is `sleep 90; runpodctl pod remove "$RUNPOD_POD_ID"`.
   - Read `list-pods` from this session until it disappears, and confirm it in `list-pod-billing`.
   - *If it is removed:* the pod-scoped key can terminate, and the GPU pod's deadline will terminate it.
   - *If it is only stoppable or neither works:* I report it before the GPU pod. The GPU run then relies on the external controls below, with the deadline as a stop.
3. **External verification, independent of the pod.**
   - I terminate the GPU pod myself from this session (`delete-pod` on the id I created) as soon as the work is done.
   - I read back that it is gone (`list-pods`) and its final cost (`list-pod-billing`).
   - I also read the pod's state and cost at every phase boundary (after setup, the benchmark, the main run and export). During the run I stay in this session and check at least every 30 minutes.
4. **Not relied on: the credit balance.** At $0 RunPod stops (does not terminate) pods that have a network volume, and the volume keeps accruing charges. Also, **auto-pay would reload the balance**, so it must be off for the balance to bound anything.

**Provider-side limitations that prevent enforcing the safeguards reliably.**
- No provider-side wall-clock terminate: the only automatic deadline runs inside the pod.
- Whether the pod-scoped key can *remove* (not just stop) its pod is undocumented, so it is tested first (step 2).
- There is no dollar cap per pod or per account. RunPod's only automatic limit is an hourly spend limit, which starts at $80/h, far above this budget. **The $12 gate and the $20 ceiling are enforced by this plan's controls (G9, the deadline, external checks), not by RunPod.**
- A zero balance stops but does not terminate pods with a network volume, and auto-pay defeats it.
- Network-volume storage is billed continuously and outside every GPU deadline. It ends only when the volume is deleted.

### CUDA memory safeguards and stop conditions

- `torch.cuda.set_per_process_memory_fraction(0.75)`: an over-allocation becomes an out-of-memory error at 36 GiB instead of degraded running.
- The **peak reserved** memory is recorded after every unit. If it exceeds 36 GiB the run stops: no retry, no change of precision or settings.
- Host RAM: stop if available memory < 4 GiB. Disk: stop if free < 5 GB.
- **Any other stop condition**: any exception, a non-finite gradient, a failed subset-identity check (coordinates outside FGSM's support beyond zero-gradient ties), or a random-mask count different from k.
- Records are appended once per (scene, source) to the network volume and copied off the pod at each phase boundary. An interruption keeps every completed pair.

### Teardown procedure (revised)

1. On the pod, bundle records, `progress.json`, audits and tensors on the network volume and compute sha256.
2. Copy the bundle to this machine (scp over the pod's SSH, or `runpodctl send`/`receive`) and verify every hash locally. The network volume copy stays as a second copy.
3. **Terminate** the pod from this session (`delete-pod`). Stopping is not enough.
4. Read back:
   - `list-pods`: no pod from this work;
   - `list-network-volumes`: the volume still exists, with its size;
   - `list-pod-billing` for the pod id: the final GPU and disk cost;
   - `list-network-volume-billing`: storage so far.
5. Do **not** delete the network volume. Its ongoing cost (≈ $2.10/month) is reported separately from GPU compute.
6. Commit the results, hashes and billing figures to the repository.

## 9. Execution steps

0. **Locally, free: done 2026-11-10** (§3).
   - `qwen_precision`, the analysis and 8 model-free tests are in; the full suite has 574 passing.
   - The MPS bf16 reproduction check is bit-exact.
   - Reference input hashes and the MPS dtype audit are recorded.
1. *(paid, about $0.01)* Termination test with a CPU pod (§8.2, step 2).
2. *(paid, ongoing)* Create the 30 GB STANDARD network volume in EU-SE-1.
3. *(paid)* Create one A40 Secure pod in EU-SE-1 with the volume at `/workspace`, a 30 GB container disk and CUDA ≥ 12.8. RTX A6000 if no A40 is free. The deadline goes first in the start command. The hourly price is re-read at create time.
4. Setup: clone the repository at the protocol commit, install the pinned environment, download the pinned weights to `/workspace/hf-cache` and verify sha256 `7de1838c…`.
5. Gates G1–G3, G5–G8 (`audit`, `inputs --compare`, `clean`, `fd`). If any fails, stop and report; thresholds are not changed. If CUDA bf16 differs substantially from MPS (G5), stop and investigate.
6. Main run with `--hourly-usd`, `--pod-start-epoch` and `--deadline-epoch`. G4 and G9 are checked after the first scene.
7. Export, terminate, verify (above). Run the committed analysis once and write the report.

**What needs approval.** Steps 1–7 involve paid resources. This plan asks for them together, with the final deployment summary. No confirmation split, no merge and no tag are involved.

## 10. After this experiment

- **SmolVLM2-2.2B** next, as the author prioritized. It tests a genuinely different language-model family, locally in bf16, with fp32 on the same GPU if useful.
- The longer arc stays the roadmap's:
  - generalization boundaries: which models and settings show selection, which show the inversion;
  - mechanism: path-specific curvature, the non-uniqueness of damaging sets;
  - then a predictive diagnostic. Checkpoint 4 already shows that clean-to-next-step gradient alignment is near 0 in models with and without the inversion, so alignment alone cannot be that diagnostic.
