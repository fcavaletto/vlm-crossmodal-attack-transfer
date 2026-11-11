# Which Coordinates, Not How Many: Matched-Count Evidence on the PGD–FGSM Gap in a Vision–Language Model

**Federico Cavaletto**

Review draft, November 2026. Code, data, frozen protocols and records: <https://github.com/fcavaletto/vlm-crossmodal-attack-transfer>

---

## Abstract

We study a synthetic colour-question task. A 2B-parameter vision–language model (Qwen3-VL-2B-Instruct, bf16) sees a rendered scene of eight coloured shapes and answers a question such as "What colour is the large triangle in the bottom row?". We measure an attack by its **answer-margin damage**: how much it lowers the model's score for the correct colour relative to its best-scoring wrong colour, in logits. When the margin falls below zero, a wrong colour outscores the correct one.

We perturb the image within the smallest budget that survives 8-bit image storage: each colour channel of each pixel moves by at most one 8-bit level (L∞ ε = 1/255). At this budget, three-step projected gradient descent (PGD) does about three times the margin damage of the one-step fast gradient sign method (FGSM). We ask whether PGD wins because it perturbs fewer coordinates or because it perturbs particular ones.

With PGD steps of ε/4, 8-bit rounding makes the PGD-3 image equal to FGSM's perturbation restricted to about 30% of its RGB-channel coordinates, up to at most 7 zero-gradient rounding ties per image. We therefore compare subsets matched in count k, in magnitude (one 8-bit level each) and in FGSM's sign at each selected location. Only the choice of coordinates differs, so the totals of positive and negative changes can differ between subsets.

Under a protocol and analysis frozen and published before any model inference on the held-out split, we tested this on 48 new, independent counterfactual scene groups, each attacked from two source contexts (96 group-by-source cases). Intervals bootstrap over the 48 groups.
- **Selection.** PGD's subset caused +6.50 logits more margin damage than random subsets of the same size (95% CI 6.08 to 6.92). In all 96 cases it exceeded all eight random subsets.
- **Sparsity.** Random thinning explained 3% of the PGD–FGSM gap (CI −2% to 9%).
- **Clean-gradient magnitude.** Subsets ranked by clean-gradient magnitude did worse than random (−1.38, CI −1.71 to −1.06), and the clean first-order prediction ranked the subsets in reverse order.
- **Answers.** Among the 45 groups answered correctly when clean, PGD's subset changed the answer in 60% and 78% of groups (by source context), against 18% and 23% for random subsets.

Exploratory checks on development scenes add two qualified statements. With Qwen's bf16 weights upcast exactly to fp32, the paired fp32 − bf16 changes in both contrasts fell within equivalence margins fixed before the run (90% intervals inside ±1.0 and ±0.5 logits). On InternVL3.5-1B (single-tile preprocessing), PGD's subset again beat random subsets (+3.98 [3.53, 4.42], pooled over source contexts), but the magnitude contrast was −0.10 [−0.24, 0.07]: the interval includes zero and no equivalence margin was set, so we state the magnitude-ranking result for Qwen only.

---

![**Figure 1. The subset relationship and the held-out matched contrasts.** **(A)** Schematic with illustrative values for six RGB-channel coordinates (two pixels), shown for the case in which every PGD step has a nonzero gradient and values stay inside [0, 1]. In that case, after rounding, PGD-3 keeps a coordinate only where its three step signs agree, and then with FGSM's sign (shaded). A step with zero gradient can instead leave a half-level rounding tie; held-out images had 0–7 such coordinates each, all with zero clean gradient (§3). **(B)** Held-out results on 48 independent scene groups (Qwen3-VL-2B, bf16). Each dot is one group's contrast in margin drop, averaged over its two source contexts. Bars show the frozen analysis's mean and 95% percentile bootstrap interval over groups. PGD-3's subset minus random: +6.50 [6.08, 6.92]. Top-|∇| subset minus random: −1.38 [−1.71, −1.06]. Source: `docs/checkpoint-2.4-confirm/analysis/` at commit `e2eb037`; drawn by `docs/paper/figures/make_main_figure.py`.](figures/figure1_main.svg)

## 1. Introduction

Cheap first-order tools are widely used to judge adversarial vulnerability:
- one-step attacks such as the fast gradient sign method (FGSM) [Goodfellow et al., 2015];
- gradient saliency, to locate the inputs a model relies on [Simonyan et al., 2014; Smilkov et al., 2017];
- first-order estimates of how much a perturbation changes an output.

All three treat the model as locally linear. Iterative attacks such as projected gradient descent (PGD) [Kurakin et al., 2017; Madry et al., 2018] find much stronger perturbations in the same ball. The gap has been linked to curvature and the loss of local linearity [Qin et al., 2019; Moosavi-Dezfooli et al., 2019; Andriushchenko & Flammarion, 2020].

We ask a narrower question: **when PGD beats FGSM, is it because PGD perturbs fewer coordinates, or because it perturbs particular ones?**

In general the question is confounded, because PGD and FGSM differ in values, signs and support at once. We use a setting where values and signs coincide (§3). At ε equal to one 8-bit level, with PGD steps of ε/4, rounding turns the PGD-3 image into FGSM's perturbation restricted to a subset of coordinates. PGD's subset can then be compared with other subsets of the same perturbation of the same size:
- uniformly random subsets;
- the subset with the largest clean-gradient magnitude.

Each kept coordinate carries exactly the value FGSM gives it: one 8-bit level, with FGSM's sign at that location. A difference in damage is then attributable to *which* coordinates are kept, rather than to how many, how far each moves, or which way each moves. "Which coordinates" is a property of the whole set. It includes the coordinates' spatial arrangement and their interactions; the comparison does not attribute the effect to individual coordinates.

**Contributions.**
1. **A matched-count decomposition** of the iterative-versus-single-step gap. It holds coordinate count, one-level magnitude and per-location FGSM sign fixed, and is available at the 8-bit floor up to a handful of rounding ties.
2. **A pre-registered held-out test** on a vision–language model. On 48 independent scene groups:
   - selection accounts for nearly all of PGD's advantage over FGSM, and sparsity for about 3%;
   - ranking coordinates by clean-gradient magnitude is worse than random;
   - the clean first-order prediction reverses the observed order.
3. **Exploratory robustness and mechanism checks** on development data, reported with their limits (§7–§8).
4. **A reproducible pipeline**, with bit-exact cross-checks, a scene generator with checked separation from development data, and a hash-frozen analysis.

We do **not** claim a new attack, a general property of vision–language models, or a mechanism. Only two clean-gradient rules were tested: magnitude ranking and the first-order prediction.

## 2. Related work

**Single-step and iterative attacks.**
- FGSM linearizes the loss [Goodfellow et al., 2015]. Iterated, projected sign steps (the basic iterative method, BIM, and PGD) give stronger L∞ adversaries [Kurakin et al., 2017; Madry et al., 2018].
- In adversarial training, FGSM can collapse ("catastrophic overfitting"). Regularizing local linearity or gradient alignment closes much of the gap [Wong et al., 2020; Andriushchenko & Flammarion, 2020; Qin et al., 2019; Moosavi-Dezfooli et al., 2019].

These works explain single-step weakness through local geometry. They do not separate the size of the perturbed support from its location.

**Coordinate selection inside single-step perturbations.** Golgooni et al. [2021] (arXiv v1, the version cited here) modify FGSM adversarial training by keeping only some coordinates of the FGSM perturbation.
- ZeroGrad (§3.1, Eq. 5) zeroes coordinates whose absolute input gradient falls below a per-sample lower quantile (q = 0.35 for CIFAR-10). It is a magnitude rule, related to our top-|∇| subset.
- MultiGrad (§3.2, Eq. 6, Algorithm 2) keeps only coordinates where N = 3 random-start FGSM perturbations agree in sign. It is a sign-consensus rule, related to the hypothesis tested in §8.
- Their Appendix C.4 (Table 10) reports how often each method's perturbation signs differ from PGD-10's.

That work selects coordinates to stabilize training. It reports no random subset at matched count, and no comparison of PGD's own support with other supports at fixed values.

**Restricting a fixed dense perturbation to a subset.** Zhao et al. [2022] (arXiv v2) refine the output of a dense attack (BIM, PGD or AutoAttack) as X + Z ⊙ (X^s − X): the source perturbation, unchanged, on a binary support Z (§2.1, Eq. 1). Z is sampled from a vulnerability map produced by a learned network (§2.3–2.4). They report that about 30% of the perturbed pixels often keep the attack effective (β = 0.3). Their baselines are the unrefined attack and dedicated sparse attacks. They report no random or gradient-saliency support at the same β.

**Location controls with fixed values.** The closest location-matched control we found is in Vargas & Su [2020]: §8 and Table 1 of the CEUR-WS version (the same experiment and table appear in §7 of arXiv v1). After a successful one-pixel attack on CIFAR-10, they apply the same pixel value at a random pixel, which drops success from 59% to 4.9% (LeNet) and from 33% to 3.1% (ResNet). At the eight adjacent pixels, success stays at 33.1% and 31.3%. The value is held fixed and only the location changes, as in our comparison, but for a single pixel found by a sparse search rather than a dense iterative attack.

Wang et al. [2021] (ICCV open-access version, §3.2) hold an adversarial perturbation fixed, mask subsets of its pixels, and measure interactions among them. Their interest is in decomposing an attack into interacting components. Random subsets enter their method only in sampling the interaction estimates; we found no random-subset baseline for the attack's own pixels.

**Our specific contribution.** We fix one single-step perturbation (FGSM's, after rounding). At a matched count, we compare the support an iterative attack ends up with against random and clean-gradient-magnitude supports drawn from that same perturbation, in a pre-registered held-out test on a vision–language model. In the work we read, related controls do one of three things:
- move a fixed single-pixel value (Vargas & Su);
- learn a support for a fixed dense perturbation without random or saliency baselines at the same count (Zhao et al.);
- select single-step coordinates by magnitude or sign agreement during training, without a matched random control (Golgooni et al.).

Our literature search was targeted, not systematic, so we make no priority claim.

**Sparse attacks** show that pixel choice matters when the count is constrained [Papernot et al., 2016; Su et al., 2019; Modas et al., 2019; Croce & Hein, 2019; Croce et al., 2022]. They *search* for small supports; we analyze the support a dense attack ends up with after quantization.

**Gradients and saliency.**
- Gradient masking can make single-step evaluations overstate robustness [Tramèr et al., 2018; Athalye et al., 2018a].
- Averaging gradients over noise or transformations recovers more useful directions [Athalye et al., 2018b; Smilkov et al., 2017; Wang & He, 2021], and deep-network gradients can resemble noise [Balduzzi et al., 2017].
- Gradient attributions can be fragile and insensitive to the model [Ghorbani et al., 2019; Adebayo et al., 2018].

**Attacks on vision–language models.** Small L∞ image perturbations can steer and jailbreak multimodal models [Schlarmann & Hein, 2023; Zhao et al., 2023; Carlini et al., 2023; Qi et al., 2024; Bailey et al., 2024; Luo et al., 2024]. We use the smallest budget at which a perturbation survives 8-bit storage.

## 3. Setting and the subset relationship

**Model and task.**
- **Model.** Qwen3-VL-2B-Instruct at a pinned revision, bf16 weights on Apple MPS, eager attention.
- **Questions.** Synthetic referent-colour questions such as "What colour is the large triangle in the bottom row?".
- **Scenes.** Each 512 × 384 scene has 8 shapes on a 4 × 2 grid. Decoys force every word of the question, and a fixed colour histogram prevents frequency from answering it.
- **Counterfactual groups.** Each group has two arms that differ only by swapping the referent's colour with another object's colour.
- **Contexts.** Two non-revealing text contexts: `neutral`, and `helpful`, which tells the model to use all object words together.
- **Readout.** The four single-token colour logits.
- **Outcome.** The **margin** is the label logit minus the largest other colour logit. A frozen scorer computes it after rounding the image to 8 bits (clamp, ×255, round half to even). The margin drop is D = margin(clean) − margin(perturbed), in logits.

**Attacks.**
- An image has 3 × 384 × 512 = 589,824 **RGB-channel coordinates**: one per colour channel of each pixel.
- **FGSM** takes one sign step of size ε = 1/255 on the gradient of the labelled-colour NLL, in the *source* context.
- **PGD-3** takes three sign steps of ε/4, each projected onto the ε-ball.
- Each context serves in turn as source. The image is also scored in the other context, the *target*.

**Proposition (PGD-3 is an FGSM subset after rounding).** Assume the clean image is on the 8-bit grid. Consider a coordinate whose clean value is not at the [0, 1] bounds and whose gradient is non-zero at each of the three steps (the case drawn in Figure 1A).
- Three ε/4 steps never reach the ε-ball boundary, so the coordinate's PGD-3 displacement is ±ε/4 or ±3ε/4.
- Rounding to the 8-bit grid (spacing ε) maps ±ε/4 to 0 and ±3ε/4 to ±ε.
- The coordinate therefore survives exactly when all three step signs agree, and its sign is then the first step's sign, which is FGSM's.

*The rounded PGD-3 image equals the rounded FGSM image restricted to that agreement set, whatever steps 2 and 3 do* (Figure 1A).

If steps 2 and 3 had independent random signs, the agreement set would be a uniform random 25% of FGSM's support. Empirically it is 29.8% of FGSM's support (held-out; mean over the 96 group-by-source cases, 95% CI 29.5–30.1%).

**Exceptions.** A zero gradient at any step falls outside this case: the displacement can then be ±ε/2, a rounding tie. The only exceptions found were coordinates whose clean gradient is exactly zero. FGSM does not move such a coordinate. If steps 2 and 3 agree, PGD moves it by ε/2, a rounding tie, and round-half-to-even can shift an odd 8-bit value by one level.
- This affected 1–7 coordinates in 84 of the 96 held-out cases, out of about 175,000 kept coordinates.
- Every such coordinate had exactly zero gradient.
- No sign was ever reversed.

**Consequence.** That PGD-3 is a sparse, sign-consistent FGSM is arithmetic, not evidence. The informative question is whether PGD's particular coordinates beat other coordinates drawn from the same perturbation at the same count.

## 4. Method: a matched-count decomposition

Notation for each scene and source context:
- r_F is the rounded FGSM perturbation, and S_F its support (median about 589,800 coordinates).
- k is the size of PGD-3's rounded support (about 148,000–196,000).

**What is matched.** Each subset image below is x + r_F ⊙ 1_M for a mask M ⊂ S_F with |M| = k:
- **count:** every subset changes exactly k coordinates;
- **magnitude:** each changed coordinate moves by exactly one 8-bit level;
- **direction:** at each selected location, the change has the sign FGSM assigns there.

Signs are matched location by location, not in aggregate. Two masks of the same size can contain different numbers of positive and negative changes.

Four images are compared:

| Image | Coordinates changed | Values |
|---|---|---|
| FGSM | all of S_F | r_F |
| Random subset (8 seeds) | k coordinates drawn uniformly from S_F | r_F on those coordinates |
| Top-\|∇\| subset | the k coordinates of S_F with the largest \|clean NLL gradient\| | r_F on those coordinates |
| PGD-3 (official image) | k: PGD's agreement set (§3), plus 0–7 zero-gradient tie coordinates outside S_F | r_F on the agreement set |

The decomposition uses three contrasts:
- **selection:** D_PGD − D_rand;
- **sparsity:** D_rand − D_FGSM;
- **saliency selection:** D_topk − D_rand.

The **sparsity share** is (D̄_rand − D̄_FGSM)/(D̄_PGD − D̄_FGSM), a ratio of mean differences. We also compute each subset's clean **first-order prediction**, −∇M·r, where ∇M is the clean margin gradient.

## 5. Samples and pre-registration

**Held-out confirmation (pre-registered).** The split `study2-trackB-confirm-v1` contains 48 newly generated counterfactual groups.
- **Independent unit.** The 48 groups are the independent units. Each group's arm a is attacked from both source contexts, giving 96 group-by-source cases. The cases are not independent: two cases share a scene.
- **Pooling.** Pooled estimates average each group's two source contexts first, then bootstrap over the 48 groups (percentile, 9,999 draws, seed 20261116).
- **Generator.** A new seeded generator draws the groups from a scene distribution specified before this study. Seed strings reserved for other studies are refused in code.
- **Separation.** Each group was checked against all 216 earlier development scenes by layout signature and its mirror images, by scene-graph hash, and by decoded-pixel hash. All three checks found zero overlap.
- **Freeze.** The protocol, analysis code, runner and scene manifest were hash-frozen and pushed to a public repository before any model inference on the held-out split (commit `c35b02c`); earlier inference used development scenes only. The runner refuses to start if any hash differs.
- **Pipeline check.** A smoke test on one development scene first reproduced earlier gradients, attack iterates and 22 official scores bit for bit.
- **Execution.** One run with a time cap and no interim look at outcomes.

**Pre-registered predictions.**
- **P1 (primary).** Mean selection > 0 with each context as source, and the pooled 95% CI lower bound > 0.
- **S1.** The pooled sparsity-share CI upper bound ≤ 0.25.
- **S2.** Mean saliency selection < 0 with each source, and the pooled CI upper bound < 0.

**Development sample (exploratory).** The 48 arm-a scenes of the development design. The question arose while examining these scenes. A random-mask control on them was specified before it ran (Checkpoint 2.3b), but the hypotheses were formed on these data. Only the held-out split is confirmatory.

## 6. Held-out results

**Run integrity.**
- All 48 groups completed (1,200 records) in 162.6 minutes, against a 180-minute cap.
- An end-of-run drift check reproduced the first score bit for bit.
- There were no deviations from the frozen protocol.
- 45 of the 48 groups were answered correctly in both contexts when clean.

**Table 1. Source-context margin drop (logits; mean [95% CI]).** Pooled over the two source contexts per group. Development values are exploratory and shown for comparison.

| Image (k matched) | Development (exploratory) | **Held-out confirmation** |
|---|---|---|
| FGSM | 3.52 [2.93, 4.10] | 3.32 [2.65, 4.03] |
| Random subset (mean of 8 seeds) | 4.00 [3.44, 4.55] | 3.54 [3.07, 4.05] |
| Top-\|∇\| subset | 2.41 [1.92, 2.92] | 2.16 [1.67, 2.69] |
| PGD-3 | 9.86 [8.98, 10.75] | 10.04 [9.35, 10.75] |
| Selection (PGD − random) | 5.87 [5.24, 6.51] | **6.50 [6.08, 6.92]** |
| Sparsity (random − FGSM) | 0.48 [0.06, 0.89] | 0.23 [−0.16, 0.62] |
| Sparsity share of the gap | 0.08 [0.01, 0.14] | **0.03 [−0.02, 0.09]** |
| Saliency (top-\|∇\| − random) | −1.59 [−2.03, −1.17] | **−1.38 [−1.71, −1.06]** |

**Pre-registered decisions.** All three predictions were supported.
- **P1.** Selection was +6.27 [5.71, 6.80] with neutral as source and +6.73 [6.20, 7.27] with helpful as source.
- **S1.** The upper bound, 0.087, is below 0.25.
- **S2.** Saliency selection was −1.33 [−1.70, −0.97] and −1.43 [−1.77, −1.10] per source, with a pooled upper bound of −1.06.

**Consistency across groups (Figure 1B).**
- Within each source context, PGD's subset did more damage than every one of the eight random subsets in 48 of 48 groups, so in all 96 group-by-source cases.
- The median within-group standard deviation across random seeds is about 0.3 logits, against a selection effect of about 6.5.
- Per-group pooled selection ranged from +3.8 to +10.7 logits. Per-group pooled saliency ranged from −4.4 to +0.8.

**Answer changes (descriptive).** The margin is a logit difference. To read it behaviourally, we report how often each image changed the model's answer, among the 45 groups answered correctly in both contexts when clean. The random rate averages the eight seeds.

| Source context | FGSM | Random subset | Top-\|∇\| subset | PGD-3 subset |
|---|---|---|---|---|
| neutral | 5/45 (11%) | 18% | 5/45 (11%) | 27/45 (60%) |
| helpful | 12/45 (27%) | 23% | 6/45 (13%) | 35/45 (78%) |

PGD's subset changed answers 42 and 54 percentage points more often than random subsets: neutral 0.42 [0.29, 0.58], helpful 0.54 [0.40, 0.68], bootstrap over the 45 groups. These rates were reported descriptively and were not pre-registered decision quantities.

**Transfer.** Scored in the context it was not optimized for, PGD's selection advantage stays large: +5.54 [5.03, 6.05] (neutral source) and +6.64 [6.12, 7.20] (helpful source). The sparsity share remains small (0.02). The selected coordinates are not specific to one prompt.

**The first-order ranking is reversed.** The clean first-order prediction rates the subsets as follows:
- PGD's subset at 0.93× (neutral) and 0.92× (helpful) the random subset;
- the top-|∇| subset at 2.81× and 2.84× the random subset.

Observed damage was 2.83× random for PGD and 0.61× random for top-|∇| (pooled means). The linear model also overestimates every subset's damage roughly tenfold, which is expected outside its regime of validity; the ordering error is the informative part.

## 7. Exploratory robustness checks (development data)

These checks use the 48 development scenes, not the held-out split. They were each specified before running, but they are exploratory: hypotheses had been formed on these scenes. Each estimate is labelled with its source commit. All numbers below were read from the committed analysis outputs at those commits.

**Table 2. Robustness checks on the 48 development scenes: pooled estimates.** Every cell is pooled over the two source contexts (each scene's two source-context values are averaged first) and scored in the source context. Logits; mean [95% CI], bootstrap over 48 scenes. The InternVL *primary cell* discussed below (neutral source context only, bf16) is a different, unpooled quantity and is not in this table.

| Check (commit) | Precision | Selection, pooled (PGD − random) | Saliency, pooled (top-\|∇\| − random) |
|---|---|---|---|
| Qwen3-VL-2B, CUDA A40 (`ef2b637`) | bf16 | +6.10 [5.43, 6.76] | −1.57 [−2.04, −1.15] |
| | fp32 (bf16 weights upcast exactly) | +5.91 [5.35, 6.49] | −1.73 [−2.16, −1.31] |
| | paired fp32 − bf16 | −0.19 [−0.61, 0.22]; 90% CI [−0.54, 0.16], inside the prespecified ±1.0 | −0.15 [−0.41, 0.09]; 90% CI [−0.37, 0.05], inside the prespecified ±0.5 |
| InternVL3.5-1B, single tile, MPS (`00e8530`) | bf16 | +3.98 [3.53, 4.42] | −0.10 [−0.24, 0.07] |
| | fp32 | +4.16 [3.69, 4.62] | −0.01 [−0.16, 0.15] |
| | paired fp32 − bf16 | +0.18 [0.01, 0.34]; no equivalence margin specified | +0.09 [−0.04, 0.22]; no equivalence margin specified |

**Qwen in full fp32 (Checkpoint 5, `ef2b637`).**
- **Design.** The same 48 development scenes and the same attack and scoring protocol, run on an NVIDIA A40 rather than the MPS device of the main study. The model is evaluated in bf16 and in full fp32. The checkpoint stores its weights in bf16, so the fp32 model uses the same weights upcast exactly; gradients, attacks and scores are recomputed in fp32.
- **Equivalence rule, fixed in the plan before the run.** Equivalence is claimed only if the 90% interval of the paired fp32 − bf16 difference lies inside a margin of ±1.0 logit for selection or ±0.5 logit for saliency (two one-sided tests at α = 0.05).
- **Result.** Pooled over source contexts, both 90% intervals lie inside their margins: [−0.54, 0.16] for selection and [−0.37, 0.05] for saliency (Table 2).
- **What it supports.** On these scenes, with these exactly upcast weights, evaluating in fp32 instead of bf16 changed neither the selection advantage nor Qwen's saliency result by more than the prespecified margins. It does not test fp32-trained weights, which this checkpoint does not provide.
- **Caveats.** A cross-device clean-margin gate passed exactly at its pre-set limits. The run covers one GPU and development scenes only.

**A second model (Checkpoint 4 Stage 1, `00e8530`).**
- **Design.** InternVL3.5-1B, with the same 48 development scenes, attack, budget and masks. It differs from the Qwen study in several ways at once:
  - architecture and language model;
  - single-tile 448 × 448 preprocessing instead of the model's default 13-tile configuration, which exceeded the memory limit;
  - a resize with a second rounding step inside preprocessing; Qwen's pipeline has no resize;
  - SDPA rather than eager attention.
- **Two kinds of estimate.** The protocol's *primary cell* is neutral as source context, scored in that context on the official preprocessing path. It is computed for each precision; below, "primary cell" means its bf16 value. Table 2 instead pools both source contexts. Both are given where they differ.
- **Selection.** Positive in both precisions and both source contexts. Primary cell: +3.39 [2.93, 3.86], with PGD's subset beating all eight random subsets in 47 of 48 scenes. Pooled, bf16 (Table 2): +3.98 [3.53, 4.42].
- **Sparsity share.** Primary cell: −0.14 [−0.22, −0.08]. Same-count random subsets did less damage than dense FGSM.
- **Saliency.** Primary cell: −0.10 [−0.24, 0.03]. Pooled, bf16 (Table 2): −0.10 [−0.24, 0.07]. Both intervals include zero. The protocol specified no equivalence margin for this contrast, so these intervals do not show that magnitude-ranked subsets match random ones; small negative or positive effects remain compatible with the data.
- **What it supports.** A selection advantage of PGD-3's subset over matched random subsets is not unique to Qwen on these scenes. Qwen's saliency result, magnitude-ranked subsets clearly worse than random, was not reproduced on InternVL in this configuration, so we state it for Qwen only.
- **What it does not support.** The two models differ in several respects at once, so these checks do not identify why the saliency results differ. They also do not show that selection is smaller on InternVL for architectural reasons. A paired same-scene comparison is in the Checkpoint 4 analysis; we do not interpret it causally for the same reason.

## 8. What the selected set is: exploratory notes

These are development-only pilots on a fixed 16-scene subset (s16), with neutral as the source context. None was confirmed on new data.

**Sign consensus at nearby points (Checkpoint 3; report at `e2eb037`).**
- **Hypothesis.** PGD's three-way sign agreement suggests that PGD enriches coordinates whose gradient sign is stable near the clean image. MultiGrad [Golgooni et al., 2021] uses a related agreement rule over random starts.
- **Design.** We built masks of the same count and signs from sign agreement among gradients at attack-independent random nearby points.
- **Result.** The best such mask recovered R = 0.05 [−0.05, 0.14] of PGD's selection gain over random. At PGD's own budget of three gradient evaluations, R = 0.08 [0.01, 0.15]. Ranking by smoothed-gradient magnitude was worse than random, like the clean ranking.
- **Reading.** For the samplers and rules tested, attack-independent neighbourhood gradients carry little of PGD's selection signal. Other estimators were not tested.

**Patch shuffling (same pilot).**
- **Design.** This control keeps PGD's number of selected coordinates in each 16 × 16-pixel patch, and re-draws which coordinates within each patch are selected, each with FGSM's sign.
- **Result.** It retained about a fifth of PGD's selection gain: R = 0.22 [0.18, 0.29] and 0.21 [0.17, 0.28] for two seeds, or +1.27 and +1.21 logits over random, against +5.67 for PGD's own subset.
- **Reading.** Patch-level density carries part of the advantage. Most of it depends on placement within patches. Shuffling also breaks any interaction among nearby coordinates, so this does not separate individual coordinate effects from local arrangement.

**Partition and union (Checkpoint 3b, `6c7ffc4`).**
- **Estimand.** For each scene, the outcome is the change in a **fixed clean-competitor contrast**: the score gap between the label and the colour that was its strongest rival on the clean image. The rival is held fixed, so rival switches do not change the outcome.
- **Design.**
  - PGD's subset was split into four random quarters (two independent splits), and each of eight random subsets of the same size was split the same way.
  - The full-mask advantage is A = PGD full − mean random full.
  - The summed quarter advantage is Q = Σ over quarters of (PGD quarter − matched random quarter).
  - The primary estimate is Δ = A − Q, with a paired two-sided 95% scene bootstrap.
- **Results.**
  - A = +4.75 [3.59, 5.92].
  - An average PGD quarter already beat a matched random quarter, by Q/4 = +1.74 [1.26, 2.24] logits.
  - The summed quarter advantages, Q = +6.96 [5.05, 8.95], exceeded the full-mask advantage: Δ = −2.21 [−4.10, −0.57].
- **Within each family.** Both were sub-additive. The full mask did less damage than its four quarters summed, by 6.96 [4.31, 10.11] for PGD and 4.75 [3.23, 6.43] for random, so Δ reflects PGD's set being more sub-additive.
- **Reading.**
  - PGD's advantage does not require coordinates from different quarters to act together.
  - This does not establish redundancy among PGD's coordinates. PGD's full subset does about 2.3 times the damage of random's, so a response that saturates differently at different damage levels could produce the same pattern.
  - Interactions within a quarter are not observed, and interaction terms of opposite sign can cancel.
  - On the official margin, where rival switches count, the direction was the same (Δ = −2.13 [−4.43, −0.001]).

Taken together, these pilots are consistent with PGD finding *a* damaging coordinate set whose advantage is spread across its parts, but they do not identify a mechanism. How the spatial organization, interactions and saturation contribute remains open.

## 9. Discussion and limitations

**What the held-out result means.** On this model and budget, a perturbation that changes the right 30% of coordinates by one level does about three times the damage of one that changes all of them. Random thinning does almost nothing. Two clean-gradient rules point away from the damaging coordinates: magnitude ranking and the first-order prediction.

For evaluation practice, on this model and budget, two consequences follow:
1. Single-step attack numbers understate vulnerability substantially, even with no gradient masking.
2. Coordinate-level gradient saliency should not be taken to identify where the model is vulnerable without a check against an iterative attack.

The InternVL check shows that the second consequence may not transfer to other models.

**What it does not mean.** The clean gradient may still contain the relevant information in some other function of it. Only two rules were tested. Nor do we know *why* PGD's coordinates are damaging (§8).

**Limitations.**
- **Scope.** The confirmatory evidence covers one model and precision, one budget and step schedule, and one synthetic task family with two non-revealing contexts.
- **Structural condition.** The subset relationship depends on ε being one 8-bit level and on steps of ε/4. It holds up to 0–7 zero-gradient rounding ties per image, which lie outside FGSM's support. Other settings need a different control.
- **Meaning of "selection".** The comparison holds count, one-level magnitude and per-location sign fixed. "Selection" therefore includes spatial arrangement and interactions; it is not a per-coordinate importance claim. The sparsity share is a ratio of mean differences, not an additive decomposition of a nonlinear response.
- **Held-out split.** It shares the development distribution family. It is out-of-sample, not out-of-distribution.
- **Random null.** The confirmatory null is uniform within FGSM's support. Structured nulls (§8) are development-only.
- **Saliency.** Saliency here means raw clean-gradient magnitude. The inversion is shown for Qwen only.
- **Robustness checks.** §7 and §8 are exploratory, on development data, and they change several factors at once.
- **Hypothesis origin.** The hypotheses were formed on development data. Only the held-out split is confirmatory.

## Pre-registration, data and code availability

All locations are in the public repository. Results are pinned to immutable commits.

| Material | Location | Commit |
|---|---|---|
| Held-out protocol, hashes, records, analysis, report | `docs/checkpoint-2.4-confirm/` | frozen at `c35b02c`; results at `e2eb037` |
| Scene generator; runner and attacks | `src/scene_confirm/`; `src/path_probe/` | `e2eb037` |
| Sign-consensus and patch-shuffle pilot | `docs/checkpoint-3-mechanism/` | `e2eb037` |
| Qwen fp32 check | `docs/checkpoint-5-qwen-fp32/` on `research/qwen-fp32-gpu` | `ef2b637` |
| InternVL check | `docs/checkpoint-4-internvl/` on `research/internvl-stage0` | `00e8530` |
| Partition-and-union pilot | `docs/checkpoint-3b-partition/` on `research/checkpoint-3b-partition` | `6c7ffc4` |
| This manuscript, Figure 1 and their sources | `docs/paper/` on `publication/revision-1` | this branch |

On the same hardware and software the pipeline is bit-for-bit deterministic. On other hardware the estimates should reproduce, but the bits may not.

## References

- Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018). Sanity checks for saliency maps. *Advances in Neural Information Processing Systems 31 (NeurIPS 2018)*.
- Andriushchenko, M., & Flammarion, N. (2020). Understanding and improving fast adversarial training. *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*.
- Athalye, A., Carlini, N., & Wagner, D. (2018a). Obfuscated gradients give a false sense of security: circumventing defenses to adversarial examples. *Proceedings of the 35th International Conference on Machine Learning (ICML)*, PMLR 80:274–283.
- Athalye, A., Engstrom, L., Ilyas, A., & Kwok, K. (2018b). Synthesizing robust adversarial examples. *Proceedings of the 35th International Conference on Machine Learning (ICML)*, PMLR 80:284–293.
- Bailey, L., Ong, E., Russell, S., & Emmons, S. (2024). Image hijacks: adversarial images can control generative models at runtime. *Proceedings of the 41st International Conference on Machine Learning (ICML)*, PMLR 235:2443–2455.
- Balduzzi, D., Frean, M., Leary, L., Lewis, J. P., Ma, K. W.-D., & McWilliams, B. (2017). The shattered gradients problem: if resnets are the answer, then what is the question? *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70:342–350.
- Carlini, N., Nasr, M., Choquette-Choo, C. A., Jagielski, M., Gao, I., Koh, P. W., Ippolito, D., Tramèr, F., & Schmidt, L. (2023). Are aligned neural networks adversarially aligned? *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*.
- Croce, F., & Hein, M. (2019). Sparse and imperceivable adversarial attacks. *ICCV 2019*, 4723–4731. doi:10.1109/ICCV.2019.00482.
- Croce, F., Andriushchenko, M., Singh, N. D., Flammarion, N., & Hein, M. (2022). Sparse-RS: a versatile framework for query-efficient sparse black-box adversarial attacks. *Proceedings of the AAAI Conference on Artificial Intelligence* 36(6), 6437–6445. doi:10.1609/aaai.v36i6.20595.
- Ghorbani, A., Abid, A., & Zou, J. (2019). Interpretation of neural networks is fragile. *Proceedings of the AAAI Conference on Artificial Intelligence* 33, 3681–3688. doi:10.1609/aaai.v33i01.33013681.
- Golgooni, Z., Saberi, M., Eskandar, M., & Rohban, M. H. (2021). ZeroGrad: mitigating and explaining catastrophic overfitting in FGSM adversarial training. arXiv:2103.15476v1. Sections cited from v1. A journal article by the same authors with a different title, "ZeroGrad: costless conscious remedies for catastrophic overfitting in the FGSM adversarial training", appeared in *Intelligent Systems with Applications* 19, 200258 (2023), doi:10.1016/j.iswa.2023.200258; we did not read it.
- Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and harnessing adversarial examples. *ICLR 2015*. arXiv:1412.6572.
- Kurakin, A., Goodfellow, I. J., & Bengio, S. (2017). Adversarial examples in the physical world. *ICLR 2017 Workshop Track*. arXiv:1607.02533v4.
- Luo, H., Gu, J., Liu, F., & Torr, P. (2024). An image is worth 1000 lies: transferability of adversarial images across prompts on vision-language models. *ICLR 2024*. (The arXiv version, 2403.09766, has a different subtitle.)
- Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards deep learning models resistant to adversarial attacks. *ICLR 2018*. arXiv:1706.06083.
- Modas, A., Moosavi-Dezfooli, S.-M., & Frossard, P. (2019). SparseFool: a few pixels make a big difference. *CVPR 2019*, 9079–9088. doi:10.1109/CVPR.2019.00930.
- Moosavi-Dezfooli, S.-M., Fawzi, A., Uesato, J., & Frossard, P. (2019). Robustness via curvature regularization, and vice versa. *CVPR 2019*, 9070–9078. doi:10.1109/CVPR.2019.00929.
- Papernot, N., McDaniel, P., Jha, S., Fredrikson, M., Celik, Z. B., & Swami, A. (2016). The limitations of deep learning in adversarial settings. *IEEE European Symposium on Security and Privacy (EuroS&P)*, 372–387. doi:10.1109/EuroSP.2016.36.
- Qi, X., Huang, K., Panda, A., Henderson, P., Wang, M., & Mittal, P. (2024). Visual adversarial examples jailbreak aligned large language models. *Proceedings of the AAAI Conference on Artificial Intelligence* 38(19), 21527–21536. doi:10.1609/aaai.v38i19.30150.
- Qin, C., Martens, J., Gowal, S., Krishnan, D., Dvijotham, K., Fawzi, A., De, S., Stanforth, R., & Kohli, P. (2019). Adversarial robustness through local linearization. *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*.
- Schlarmann, C., & Hein, M. (2023). On the adversarial robustness of multi-modal foundation models. *ICCV Workshops 2023*, 3679–3687. doi:10.1109/ICCVW60793.2023.00395.
- Simonyan, K., Vedaldi, A., & Zisserman, A. (2014). Deep inside convolutional networks: visualising image classification models and saliency maps. *ICLR 2014 Workshop*. arXiv:1312.6034.
- Smilkov, D., Thorat, N., Kim, B., Viégas, F., & Wattenberg, M. (2017). SmoothGrad: removing noise by adding noise. arXiv:1706.03825.
- Su, J., Vargas, D. V., & Sakurai, K. (2019). One pixel attack for fooling deep neural networks. *IEEE Transactions on Evolutionary Computation* 23(5), 828–841. doi:10.1109/TEVC.2019.2890858.
- Tramèr, F., Kurakin, A., Papernot, N., Goodfellow, I., Boneh, D., & McDaniel, P. (2018). Ensemble adversarial training: attacks and defenses. *ICLR 2018*. arXiv:1705.07204.
- Vargas, D. V., & Su, J. (2020). Understanding the one-pixel attack: propagation maps and locality analysis. *Proceedings of the Workshop on Artificial Intelligence Safety 2020 (AISafety), co-located with IJCAI-PRICAI 2020*, CEUR-WS Vol. 2640, paper 4. Section cited from the CEUR-WS version; an earlier version is arXiv:1902.02947v1.
- Wang, X., & He, K. (2021). Enhancing the transferability of adversarial attacks through variance tuning. *CVPR 2021*, 1924–1933. doi:10.1109/CVPR46437.2021.00196.
- Wang, X., Lin, S., Zhang, H., Zhu, Y., & Zhang, Q. (2021). Interpreting attributions and interactions of adversarial attacks. *ICCV 2021*, 1075–1084. doi:10.1109/ICCV48922.2021.00113. Section cited from the CVF open-access version.
- Wong, E., Rice, L., & Kolter, J. Z. (2020). Fast is better than free: revisiting adversarial training. *ICLR 2020*. arXiv:2001.03994.
- Zhao, H., Nguyen, T., Le, T., Montague, P., De Vel, O., Abraham, T., & Phung, D. (2022). Learning to attack with fewer pixels: a probabilistic post-hoc framework for refining arbitrary dense adversarial attacks. arXiv:2010.06131v2 (February 2022). Sections cited from v2.
- Zhao, Y., Pang, T., Du, C., Yang, X., Li, C., Cheung, N.-M., & Lin, M. (2023). On evaluating adversarial robustness of large vision-language models. *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*.
