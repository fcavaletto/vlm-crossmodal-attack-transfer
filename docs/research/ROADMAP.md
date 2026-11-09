# Research roadmap after Checkpoint 2.4 (2026-11-09)

**Status: plan only.** Nothing here is a result unless it cites a committed record. No model run, paid resource or confirmation split was started for this document. The numbers in §2 are re-readings of committed **development** records and are exploratory.

16 November is a publication milestone for the confirmed 2.4 study. This roadmap is the work after it.

## 1. Summary and recommendation

- **The proposed question is good but presumes too much.** "When and why do clean-input gradients fail" assumes the failure is a property of vision-language models. The evidence is one model, one precision, one budget. The existing records point at a sharper and more checkable question (§3).
- **Recommended thesis.** At small budgets, white-box evaluation of a VLM run in bf16 operates at the numerical resolution of its own input. The first thing to establish is whether the confirmed selection effect and the saliency inversion belong to the *model* or to the *arithmetic*. Either answer is publishable, and they lead to different papers.
- **Next decisive experiment.** A precision × budget test on a second architecture small enough to run in full fp32 on the laptop, using the same matched comparison as 2.4 (§8, §9). It tests generality and numerical origin in one run, costs nothing, and needs no new confirmation split.
- **What would make this a paper rather than a portfolio piece.** Three architectures, a full-fp32 arm, a budget sweep with a control that works without the 2.4 identity, and attack baselines at matched compute (§6, "essential"). A predictive diagnostic (H3) is the candidate original contribution; it is exploratory and may fail.

## 2. What the committed development records already say

Re-read from `docs/checkpoint-2.3-exploratory/paths/` and `docs/checkpoint-3-mechanism/` (development scenes, Qwen3-VL-2B, bf16 on MPS, ε = 1/255). Exploratory.

| Observation | Value | Why it matters |
|---|---|---|
| Damage along the dense FGSM direction, by fraction t of ε | t = 1/8: 1.5 · 1/4: 3.8 · 1/2: 4.1 · 3/4: 3.7 · 1: ≈ 3.4–3.9 logits | The single-step direction saturates by ε/4. More of it does nothing. |
| PGD damage by step (ε/4 steps) | 3.9, 7.4, 10.0, 11.5, 13.8, 15.1, 16.1 logits at steps 1–7 | Each fresh gradient adds about as much as the first. No saturation by 7 steps. |
| Path gradients against the clean gradient | cosine ≈ 0.00, sign agreement 0.52–0.54 at every step after the first | After one ε/4 step the gradient is nearly independent of the clean one. |
| Gradient at random nearby points against the clean gradient | cosine 0.16 (±ε/4), 0.30 (±ε/16) | Decorrelation is already strong at a sixteenth of one 8-bit level. |
| Fraction of coordinates that a t·ε perturbation changes after the bf16 cast | 0.15 at t = 1/8, 0.55 at t = 1/4, 1.00 from t = 1/2 | **An ε/4 PGD step is below the bf16 resolution of the input for 45% of coordinates.** |
| bf16 gradient against fp32-vision gradient (same clean image) | cosine 0.29, sign agreement 0.75 | Changing the precision of the vision tower alone changes a quarter of the gradient signs. |
| PGD-3 mask, bf16 against fp32 vision | overlap 0.11 above chance; damage similar | Different arithmetic finds a different set that is about as damaging. |
| New model-free check for this roadmap (16–48 scenes, stored perturbations): share of coordinates PGD-k moves one 8-bit level, and share of those moving *against* FGSM's sign | k = 3: 30%, 0% · k = 5: 43%, 15% · k = 7: 50%, 23% | The 2.4 identity (PGD ⊂ FGSM) holds only up to 4 steps of ε/4. Longer or larger attacks need a different control (§7). Even-step ties are ignored in this count. |

Two readings fit these facts, and the records cannot separate them:
1. **Model reading.** The loss surface is highly non-linear at this scale. Single-step and saliency methods fail because the gradient field genuinely rotates.
2. **Numerical reading.** In bf16 the input and activations are quantized near the perturbation size, so the measured gradient is partly rounding noise. Iteration then helps by re-sampling that noise where it counts, and the "rotation" is partly an artefact.

The only precision test so far changed the vision tower to fp32 and kept the language model in bf16. It did not remove the effect, which does not decide between the two readings.

## 3. Challenge to the proposed framing

Proposed: *"When and why do clean-input gradients fail to identify damaging adversarial perturbations in vision-language models, and can the information recovered by iterative optimization be explained or exploited?"*

Three problems:

1. **"Fail … in vision-language models" is not established.** One model at one budget. A reviewer's first question is whether full fp32 or a 4/255 budget makes it disappear. The framing should make that the first hypothesis, not an assumption.
2. **"Iterative beats single-step because of non-linearity" is old** (gradient masking checks, GradAlign, local-linearity work; §5). A paper framed as explaining why PGD beats FGSM will be read as rediscovery. What is not in the literature I could find is narrower: the matched-coordinate control, the saliency-below-random result at matched count, and any study of how half precision affects small-budget white-box evaluation of VLMs.
3. **"Exploit" invites attack tweaks.** A new attack must beat APGD at matched compute to be believed, on a field where AttackBench has shown that most claimed improvements do not survive fair budgets. The diagnostic route (predict when cheap evaluation misleads) has less competition and fits safety evaluation better.

**Proposed replacement question.**
*When does cheap first-order evidence (a single gradient step, or gradient saliency) misjudge a VLM's vulnerability at small budgets, how much of that is the model and how much the numerical precision it is evaluated in, and can a two-gradient measurement predict it in advance?*

This keeps everything already confirmed, and it makes a negative replication informative (see §6, "what a null means").

## 4. Thesis and falsifiable hypotheses

**Thesis.** At budgets near one 8-bit level, the gap between single-step and iterative attacks on VLMs is a gap in *which coordinates move*, and its size is governed by how quickly the gradient sign field decorrelates along the attack path. That rate depends on model and on numerical precision, and it can be measured cheaply.

| | Hypothesis | Prediction | Falsified if |
|---|---|---|---|
| **H1** numerical origin | Part of the selection effect and the saliency inversion in Qwen3-VL-2B comes from bf16 arithmetic. | In full fp32, same scenes and budget: (a) gradient cosine at ±ε/16 rises above 0.6; (b) the PGD-3 selection gain falls by at least half; (c) top-\|∇\| is no longer below random. | Full fp32 gives a selection gain within 25% of bf16 and top-\|∇\| still below random. Then the effect is a property of the model, which is the stronger scientific claim. |
| **H2** generality | The selection effect is not specific to Qwen3-VL-2B. | On at least two further architectures at ε = 1/255, PGD-3: selection gain CI above 0, sparsity share ≤ 25%, top-\|∇\| not better than random. | A model passes the clean-accuracy and gradient gates and shows a selection gain CI that includes 0, or top-\|∇\| clearly better than random. |
| **H3** predictive diagnostic | A sign-stability index from two gradients predicts where single-step evaluation misleads. | Across cells (model × precision × ε × task), the index S = sign agreement between the clean gradient and the gradient after one FGSM step of size ε/4 has Spearman ρ ≤ −0.6 with the PGD-to-FGSM damage ratio, and separates cells with saliency inversion from cells without it. | \|ρ\| < 0.3 with at least 12 cells, or the index does no better than ε alone. |

Notes:
- H1(a)–(c) are graded, so a partial result is reportable. The thresholds are proposals to be frozen before the run, not after.
- H3 has no theory behind the specific index yet. It is the simplest candidate the 2.3 records suggest. The line-scan saturation point (the t at which damage along FGSM's direction stops growing) is the alternative. Choosing between them must happen on development cells only.
- A fourth idea, **non-uniqueness of damaging sets** (different restarts or precisions find near-disjoint sets of equal damage), is kept as Track C exploratory work, not a headline hypothesis. It is interesting, but I do not yet see what it would let anyone predict or do.

## 5. Novelty assessment

Method, as in `LITERATURE_REVIEW.md`: a source counts only if its abstract or text was opened. "Summary" means I read a fetched summary of the paper, not the full text. Every row needs a full-text read before it is cited in a manuscript.

| Closest work | What it established | Overlap with this project | What we could add beyond it |
|---|---|---|---|
| **DMS**, arXiv:2406.07580 (summary) | Integer rounding of adversarial images loses attack strength. DMS-AI rounds each pixel along the loss-gradient sign; DMS-AS picks pixels by Integrated Gradients. CNN classifiers, CIFAR-100 and ImageNet, 10 steps, ε given as 0.3. | **High on the starting observation**: 8-bit rounding changes what an attack is. | DMS treats rounding as a loss to repair. We use it as a control: at ε = 1/255 rounding makes PGD-3 a subset of FGSM, which permits a same-values, same-signs, same-count comparison. DMS has no VLMs, no matched random-coordinate control, no sub-level step analysis, no precision analysis. **DMS-AI must be a baseline** in any budget sweep. |
| **AttackBench**, arXiv:2404.19460 (abstract) | Fair comparison of 100+ gradient attacks under a fixed query budget, with an optimality metric against the best empirical attack. Found implementation flaws in many attacks. | Methodological: matched compute, attacks as measurement instruments. | It covers image classifiers and minimum-norm attacks. It has no per-coordinate decomposition and no VLMs. We should adopt its accounting (forward and backward passes counted), not compete with it. |
| **AutoAttack / APGD**, arXiv:2003.01690 (abstract) | Parameter-free step-size schedule; the standard for reliable evaluation. | APGD is the reference iterative attack. Its step halving adapts to exactly the non-linearity we describe. | Nothing on the method. APGD is the **strongest baseline**: any "selection" claim must hold with APGD's final perturbation, not only PGD-3. |
| **Schlarmann & Hein** (already row in `PRIOR_ART_MATRIX.md`) and later VLM white-box work: arXiv:2606.03713, arXiv:2607.07375, arXiv:2603.16960, VEAttack arXiv:2505.17440 (abstracts) | VLMs break at 1/255–8/255 under APGD/PGD; model families differ in robustness. | **High on the headline** "small perturbations break VLMs". | None of those found reports the precision the attack was run in as a variable, a single-step comparison with a matched control, or a saliency test. That is the gap, and it is an evaluation-method gap, not a new vulnerability. |
| **Gradient masking / obfuscation**: Athalye et al. 2018 checks; Lee & Chandrakasan arXiv:2006.01304 (numerical causes of overestimated robustness); arXiv:1909.12741 (low-bit quantization masks gradients) (abstracts) | Iterative should beat single-step; black-box should not beat white-box; numerical effects can make gradients uninformative and robustness look higher than it is. | **High on H1.** The numerical reading in §2 is a gradient-masking claim. | These concern defended or quantized classifiers. I found no study of bf16 inference as an unintended source of gradient masking in VLM evaluation. The standard check, a gradient-free attack (Square) against the white-box ones, is required in our matrix. |
| **GradAlign**, Andriushchenko & Flammarion 2020; SEAT arXiv:2210.05373; ELLE (ICLR 2024) (abstracts) | Catastrophic overfitting in FGSM training comes with loss of local linearity; gradient alignment inside the ε-ball measures it. | **High on H3.** Our index is a gradient-alignment measure. | They use alignment as a training regularizer or training-time diagnostic. Using a two-gradient index as a *test-time predictor* of single-step unreliability on fixed pretrained VLMs, validated across cells, was not found. The novelty is modest and must be stated as such. |
| **Sign and trajectory studies**: RGD, arXiv:2312.01260; Gradient Relevance Attack (ICCV 2023); arXiv:2608.14594 trajectory-level PGD study (abstracts / summary) | Sign fluctuation across PGD iterations exists and can be exploited; successive gradients decorrelate on small CNNs. | Medium. The "signs change along the path" observation is not new. | None of them ties sign fluctuation to a matched-count coordinate comparison, to 8-bit rounding, or to precision. Cite as prior observation of the phenomenon. |
| **Sparse attacks**: Sparse-RS arXiv:2006.12834; Sparse-PGD arXiv:2405.05075 (abstracts) | Strong L0 attacks; random search is competitive with gradients for coordinate choice. | Medium. Our masks are sparse perturbations with fixed values. | I did not find a report that gradient-magnitude selection is *below* random at matched count. That may be because nobody ran the comparison, so the claim must stay "not found", not "new". Sparse-RS is the right baseline for "best k coordinates". |
| **Gradient smoothing**: SmoothGrad, EOT, VMI-FGSM (known; already cited in the pilot report) | Averaging gradients over a neighbourhood improves saliency maps or transfer. | Direct: Checkpoint 3 tested these and they failed. | The negative is a diagnostic result on one model. It becomes interesting only if H1 explains it (noise that does not average out in bf16) or it replicates. |
| **Quantization and VLM reliability**: Bouguerra et al. ICML 2026; Q-MLLM arXiv:2511.16229 (abstracts) | Quantization changes VLM reliability and can be used for robustness. | Low to medium. | Not about attack evaluation in half precision. |

**Honest reading.** No single element is unprecedented. The defensible contribution is a combination: a matched control that isolates coordinate choice, applied across precision and budget on VLMs, plus a cheap predictor. The risk of being scooped or of having missed a paper is real; a second search before any submission is part of the plan.

## 6. Experiment matrix

Models, chosen for architectural difference, open weights and size:

| Model | Vision path | Language model | Why | Runs where |
|---|---|---|---|---|
| Qwen3-VL-2B (done) | native-resolution ViT, 2×2 merge | Qwen3 | reference | bf16 local; full fp32 needs GPU |
| InternVL3.5-1B | InternViT-300M, 448 px tiles, pixel shuffle | Qwen3 (small) | different vision tower and preprocessing, **same LM family** → isolates the vision side | bf16 and probably fp32 local |
| SmolVLM2-2.2B | SigLIP-SO400M, 384 px, pixel shuffle | SmolLM2-1.7B | different vision tower **and** LM family | bf16 local; fp32 needs GPU |
| SmolVLM2-500M (cached, integrated) | SigLIP-B | SmolLM2-360M | fits fp32 in ~2 GiB; failed clean accuracy on the Checkpoint 1 scenes (10/24) | re-gate on current scenes; use only if it passes |
| Backups: InternVL3-2B, PaliGemma 2 3B-224 (gated), LLaVA-OneVision-0.5B (unverified) | | | if a primary fails a gate | |

Experiments. **E** = essential for credibility. **X** = exploratory, might produce the original contribution.

| # | Track | Experiment | Conditions | Controls | Class | Expected value |
|---|---|---|---|---|---|---|
| 1 | A+B | Second architecture, bf16 and fp32, 2.4 comparison | InternVL3.5-1B; ε = 1/255; PGD-3 at ε/4; dev scenes | same-count random ×8, top-\|∇\|, sparse FGSM; gradient gates | **E** | Highest. Tests H2 and H1 at once, locally, free. |
| 2 | B | Full-fp32 Qwen3-VL-2B | same scenes as 2.3/3 dev sets | bf16 arm re-run on the same device, so device and precision are not confounded | **E** | High. The direct answer to "is 2.4 a bf16 artefact". GPU only. |
| 3 | B | Budget and schedule sweep | ε ∈ {1, 2, 4, 8}/255; steps {3, 10}; step sizes ε/4 and 2.5ε/steps | matched-marginal control (§7); FGSM; line scan | **E** | High. Shows where the effect lives and whether it matters at the budgets others use. |
| 4 | B | Baselines at matched compute | APGD-CE and APGD-margin, MI-FGSM, PGD with random restarts, RGD, DMS-AI rounding, Square (gradient-free) | forward and backward passes counted as in AttackBench | **E** | High. Without it, "PGD's coordinates" may just mean "a weak PGD". Square is the gradient-masking check. |
| 5 | A | Third architecture | SmolVLM2-2.2B, bf16 (fp32 on GPU) | as 1 | **E** | High for H2. |
| 6 | A | Natural-image tasks | a spatial-relation set and a yes/no or multiple-choice VQA subset, chosen and frozen before any attack; candidates VQAv2, GQA, What'sUp | clean-accuracy gate per model and task; margin defined on the answer options | **E** for a conference paper, optional for a workshop paper | Medium. Synthetic scenes are the largest external-validity gap. |
| 7 | C | Diagnostic index (H3) | computed in every cell of 1–6 from two gradients | ε-only and model-only predictors as baselines | **X** | The candidate contribution. Costs almost nothing extra once 1–6 exist. |
| 8 | B | 8-bit weights (bitsandbytes) | Qwen3-VL-2B, CUDA only | bf16 and fp32 on the same device | **X** | Medium. Extends H1 to a precision people deploy. |
| 9 | C | Non-uniqueness of damaging sets | PGD restarts, bf16 vs fp32 masks; union, intersection and swap tests at matched count | random sets of the same size | **X** | Uncertain. Could reframe "selection" as "any of many sets, found only by following the path". |
| 10 | C | Path-conditioned ablations | drop the coordinates chosen at step j; replay steps in another order | matched-count random drops | **X** | Uncertain. Needed only if 9 shows structure. |

**What a null means, by design.**
- If a new model shows no selection gain (experiment 1 or 5), the gates decide what it means. Passing clean accuracy, passing the gradient checks, and showing FGSM ≈ PGD at ε = 1/255 says the model is locally linear at that scale, and H3 predicts a high index there. That is a data point for H3, not a failed replication.
- If full fp32 removes the effect in Qwen (experiment 2), the 2.4 result stands as measured and gains an explanation: it is a property of bf16 evaluation. The paper becomes "half-precision white-box evaluation of VLMs is unreliable at small budgets".
- If it does not, the numerical reading is ruled out for this model and the model reading is strengthened.

**Development and confirmation.**
- Experiments 1–5 and 7–10 run on existing development scenes first.
- Any hypothesis first formed here (H1, H3) is confirmed only on a new split, `confirm-v2`, generated after its protocol is hash-frozen and pushed, and only with approval.
- A decision for the author: `trackB_confirm_v1` could be reused to replicate the *unchanged* 2.4 predictions P1, S1, S2 on new models. Those models have not seen it and the predictions were fixed before it existed. It must not be used for anything whose design was informed by 2.4's results on it. I recommend allowing that single reuse under a frozen replication protocol; the conservative alternative is to use `confirm-v2` for everything.

## 7. A matched control that works without the 2.4 identity

The 2.4 comparison relies on PGD-3 at ε/4 being FGSM on a subset. §2 shows that this breaks from step 5, and it never holds at larger ε, where coordinates move several levels.

**Matched-marginal control.** For any 8-bit perturbation δ from any attack, and FGSM's sign field s:

| | attack's locations | permuted locations |
|---|---|---|
| **attack's signs** | δ itself | the pairs (\|δ\|, sign) moved to random coordinates |
| **FGSM's signs** | \|δ\| at the attack's locations, signed by s | \|δ\| at random coordinates, signed by s |

- Every cell has the same multiset of magnitudes, so the same L∞, L0, L1 and L2 norms.
- Row difference = what the attack's signs add over clean-gradient signs. Column difference = what its locations add over random ones. The interaction is reported, not assumed away.
- Permutations are within colour channel, with a second variant within 16×16 patch to separate regional from fine placement (as `pgd_patch` did in Checkpoint 3).
- At ε = 1/255 with three ε/4 steps the attack's signs equal FGSM's, the two rows coincide, and the control reduces to the 2.4 comparison exactly. This is checkable model-free on the stored 2.4 tensors before any new run.

## 8. Feasibility: MacBook Air M4 24 GB against an on-demand GPU

Measured facts: Qwen3-VL-2B bf16 peaks at 8.4–8.5 GiB (9.4 GiB with fp32 vision). The abort line is 10 GiB. A confirmation group took 203 s (3 gradients, 22 scorings). About 13–14 GiB was reclaimable at the time of the check.

| Work | Laptop | GPU |
|---|---|---|
| Full-fp32 Qwen3-VL-2B | **No.** Projected peak ≈ 16.8 GiB against a 10 GiB abort line and a frozen guard that requires 20.9 GB free. More RAM being free does not change this; running it would mean lowering a safeguard to force the experiment. | Yes. 24 GB fits on paper; 48 GB leaves margin. |
| InternVL3.5-1B, bf16 | Yes (estimated peak 5–6 GiB) | not needed |
| InternVL3.5-1B, fp32 | **Probably.** fp32 weights ≈ 4.1 GiB; estimated peak 9–10 GiB, which is at the abort line. A measured smoke decides. If it exceeds 10 GiB the line is not raised and the arm moves to GPU. | fallback |
| SmolVLM2-2.2B, bf16 | Yes (weights ≈ 4.2 GiB) | not needed |
| SmolVLM2-2.2B, fp32 | No (weights alone ≈ 8.4 GiB) | Yes |
| SmolVLM2-500M, fp32 | Yes (≈ 2 GiB weights), if it passes the accuracy gate | not needed |
| 8-bit weights | No (bitsandbytes is CUDA-only) | Yes |
| Budget sweep and baselines on Qwen bf16 | Yes but slow: APGD-10 on 16 scenes × 4 budgets is roughly 6–8 h. PGD-7 took 492 s per scene. | faster, by an unmeasured factor |
| Gradient-free baseline (Square, thousands of queries) | Impractical beyond a few scenes | Yes |

GPU prices read from the RunPod list on 2026-11-09, **nothing created**: RTX A5000 24 GB $0.16–0.27/h; RTX 3090 $0.22–0.50; RTX 4090 $0.34–0.89; RTX A6000 or A40 48 GB $0.33–0.59; L40S $0.79–1.09.

A rough budget for experiments 2–4 on Qwen3-VL-2B in fp32 and bf16 is 15–30 GPU-hours, so about $5–18 on a 48 GB card, plus storage. This rests on an **unmeasured** assumption that CUDA is 2–4× the laptop's speed; the first paid hour would be a timing run. Two open technical points for a GPU arm: results on CUDA and MPS will not be bit-identical, so the bf16 arm must be re-run on the same device; and the `study_cache` launcher has not been exercised on a second machine.

## 9. Minimum decisive cross-model experiment, and the plan for it

**Minimum decisive experiment.** One new architecture, the unchanged 2.4 comparison, two precisions:

> InternVL3.5-1B at ε = 1/255, PGD-3 with ε/4 steps, in bf16 and in full fp32, on the 16 development scenes used by Checkpoint 3. Conditions: clean, FGSM, PGD-3, 8 same-count random masks, top-\|∇\|, sparse FGSM.

It is the smallest run in which every outcome moves the project: it tests H2 on a different vision tower, tests H1 where full fp32 is locally possible, and needs no new control, since the 2.4 identity is arithmetic and holds for any model.

**Stage 0. Gates, before any attack (lightweight, local, no cost).**
1. *Integration.* Add the model behind the existing `vlm_iface` interface with a pinned revision. Differentiable preprocessing must reproduce the processor's pixel values exactly; test this model-free.
2. *Memory smoke.* One load and one gradient in bf16, then fp32, under the unchanged 10 GiB abort line. If fp32 exceeds it: stop, report, do not raise the line.
3. *Clean-accuracy gate.* All 48 development groups, both contexts, clean images only. Pass at ≥ 75% in each context (Qwen: 45/48). Below that, margins are not meaningful: switch to the next candidate model and report the failure.
4. *Margin gate.* Median clean margin ≥ 2 logits on correct items, so that damage has room to register.
5. *Gradient validation.* In fp32: directional finite differences along 8 random sign directions at three step sizes, relative error ≤ 5% at the best step. In both precisions: repeat-gradient determinism (bit-identical or reported), cosine between bf16 and fp32 gradients recorded, no NaN or zero-gradient images. A model that fails finite differences in fp32 does not have "reliable gradients" and is excluded.
6. *Scorer round-trip.* The 8-bit scorer must reproduce clean scores bit-for-bit on reload, as in earlier checkpoints.

**Stage 1. The run (needs approval; local; estimated 1–2 h per precision, unmeasured).**
- Plan and analysis script committed and pushed before the first attack, as in Checkpoint 3.
- 16 development scenes, neutral context as source, scored in both contexts.
- Per scene and precision: 3 gradients for PGD-3, plus 2 extra gradients for the H3 index and a 5-point line scan.
- Reported: selection gain with bootstrap CI, sparsity share, top-\|∇\| minus random, FGSM/PGD ratio, gradient cosines, subset-identity check (count of PGD coordinates outside FGSM's support).

**Failure and stop criteria, fixed in advance.**
- Any Stage 0 gate fails → no attack run on that model or precision; report.
- Subset identity violated beyond zero-gradient ties → the comparison is invalid for that arm; investigate the rounding path, do not adjust the definition.
- Memory pressure warning, swap growth, or footprint above 10 GiB → abort, as in earlier runs.
- End-of-run drift check not exact → the run is reported as failed.

**How it will be read (exploratory, development data).**

| Outcome | Reading | Next |
|---|---|---|
| Selection gain clearly above 0 in both precisions, top-\|∇\| ≤ random | Effect generalizes and is not bf16-specific on this model | Freeze a replication protocol; GPU fp32 on Qwen becomes confirmatory rather than exploratory |
| Present in bf16, absent or halved in fp32 | Numerical reading supported | Full-fp32 Qwen on GPU is the priority; reframe around precision |
| Absent in both, FGSM ≈ PGD | This model is locally linear at 1/255 | Budget sweep on this model; the cell is an H3 data point |
| Absent in both, PGD ≫ FGSM | Gap is not coordinate selection here | The 2.4 decomposition is model-specific; examine before scaling |

Nothing from this run would be confirmed. Confirmation of H1 or H2 needs a frozen protocol and a held-out split, with approval.

## 10. What this can realistically become

| Level | What it needs | Assessment |
|---|---|---|
| **Portfolio contribution** | The merged, audited 2.4 study and this roadmap | Already there. Its strength is rigour (pre-registration, held-out confirmation, a reported stopped hypothesis, byte-reproducible analysis), not scope. For applications, that is a real and visible signal. |
| **Workshop paper** | Experiments 1–5: three architectures, full fp32 on at least two, budget sweep with the matched-marginal control, APGD and Square baselines. One confirmation split. | Realistic in roughly 6–10 weeks of part-time work and under $50 of GPU. Either H1 outcome gives a clear message. The main risk is that the effect is confined to ε = 1/255 and small models, which would make it a careful note rather than a finding people act on. |
| **Main-conference submission** | All of the above, plus natural-image tasks, at least one widely used model (LLaVA-class, 7B, GPU-only), and H3 working out of sample, or an H1 result general enough to change how VLM robustness is evaluated. | Possible but not likely on current evidence. It depends on H3 or a broad H1, both exploratory. I would not plan around it until experiments 1–3 are in. |

The single most valuable next piece of evidence, for any of the three levels, is whether the effect survives a second architecture and full precision.
