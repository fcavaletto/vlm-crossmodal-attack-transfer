# Which Coordinates, Not How Many: Pre-registered Evidence that Iterative Attacks Beat Single-Step Attacks on a Vision-Language Model by Coordinate Selection

**Federico Cavaletto**
Preprint draft, November 2026. Code, data, frozen protocols and records: <https://github.com/fcavaletto/vlm-crossmodal-attack-transfer>

---

## Abstract

Single-step attacks and gradient saliency are the cheapest tools for estimating how vulnerable a model is to adversarial inputs and which input features matter. Both assume the model is locally close to linear. We study the gap between FGSM and 3-step PGD on a 2B-parameter vision-language model (Qwen3-VL-2B, bf16) at the smallest 8-bit budget, L∞ ε = 1/255. There, PGD-3 does about three times FGSM's damage to the answer margin.

We first show that, after the 8-bit rounding every valid image undergoes, the PGD-3 image is FGSM's perturbation restricted to about 30% of its RGB channel coordinates, with identical signs, apart from at most 7 zero-gradient rounding ties per image (out of about 175,000 kept coordinates). This near-exact subset relationship allows a matched comparison: subsets of FGSM's own perturbation with the same values, signs and count, differing in which coordinates are kept. The comparison controls magnitude, sign and count; it does not separate a coordinate's individual effect from its spatial arrangement or from interactions among the kept coordinates.

A development study suggested that PGD's advantage is coordinate *selection* rather than sparsity. We then confirmed this on 48 newly generated counterfactual scene groups, under a protocol and analysis that were hash-frozen and published before any inference:
- **Selection.** PGD's subset caused +6.50 logits more margin damage than random subsets of the same size (95% CI 6.08–6.92). It exceeded all eight random subsets in 96 of 96 group-by-source cases.
- **Sparsity.** Sparsity explained 3% of the PGD-versus-FGSM gap (CI −2% to 9%).
- **Saliency.** Subsets chosen by clean-gradient magnitude were worse than random (−1.38, CI −1.71 to −1.06).
- **First-order prediction.** The clean first-order prediction ranked the subsets in the reverse of their true order.

On this model, single-step evaluations and magnitude saliency therefore misjudge both how much and where the model is vulnerable. We describe the limits of the evidence. In a development-only pilot, the sign-consensus masks we tested, built from gradients at attack-independent nearby inputs, did not reproduce PGD's selection.

---

## 1. Introduction

Adversarial evaluation of deployed multimodal models often relies on cheap tools:
- one-step attacks such as FGSM [Goodfellow et al., 2015];
- gradient saliency to locate the inputs a model relies on [Simonyan et al., 2014; Smilkov et al., 2017];
- first-order estimates of how much a perturbation will change an output.

All three treat the model as locally linear around the clean input. When that fails, iterative attacks such as PGD [Kurakin et al., 2017; Madry et al., 2018] find much stronger perturbations in the same ball. The gap between single-step and multi-step attacks has been linked to curvature and the loss of local linearity [Qin et al., 2019; Moosavi-Dezfooli et al., 2019; Andriushchenko & Flammarion, 2020].

We ask a narrower question that, as far as a limited search found, has not been isolated before: **when PGD beats FGSM, is it because PGD perturbs fewer coordinates, or because it perturbs the right ones?**

In general the question is confounded. PGD and FGSM differ in values, signs and support at once. We identify a setting where values and signs coincide, up to a handful of rounding ties. At ε equal to one 8-bit level, with PGD steps of ε/4, the 8-bit rounding that every valid image undergoes turns the PGD-3 image into a sign-preserving subset of FGSM's perturbation, up to at most 7 zero-gradient ties per image (§3). PGD can then be compared with other subsets of FGSM's perturbation that have the same count and signs:
- uniformly random subsets;
- the subset with the largest clean-gradient magnitude.

Any difference in damage is then attributable to *which coordinates* are kept, rather than to perturbation magnitude, sign or count. "Which coordinates" includes where they sit relative to one another; the comparison does not separate individual coordinates from their spatial arrangement or interactions.

**Contributions.**
1. **A matched-count, sign-matched decomposition** of the iterative-versus-single-step gap, available at the 8-bit floor up to a handful of zero-gradient rounding ties.
2. **A pre-registered held-out confirmation** on a VLM. Selection explains nearly all of PGD's advantage, sparsity about 3%, and clean-gradient-magnitude selection is worse than random.
3. **A reversal of the first-order ranking.** The clean linear prediction rates PGD's subset below random and the saliency subset nearly 3× above random. The truth is the reverse.
4. **A reproducible pipeline.** It includes bit-exact cross-checks against earlier records, a scene generator with provable separation from development data, and a hash-frozen analysis.

We do **not** claim a new attack. We do not claim that no function of the clean gradient could identify PGD's coordinates: only magnitude ranking and the first-order prediction were tested. We also do not claim a mechanism.

## 2. Related work

**Single-step and iterative attacks.**
- FGSM linearizes the loss [Goodfellow et al., 2015].
- Iterating sign steps with projection (BIM/PGD) yields stronger L∞ adversaries, and PGD is the standard first-order adversary [Kurakin et al., 2017; Madry et al., 2018].
- FGSM-based adversarial training can collapse ("catastrophic overfitting"). Regularizing gradient alignment or local linearity closes much of the gap [Wong et al., 2020; Andriushchenko & Flammarion, 2020; Qin et al., 2019].
- Curvature regularization has the same aim [Moosavi-Dezfooli et al., 2019].

These works explain *why* single steps underperform in terms of local geometry. They do not separate support size from support location.

**Noisy, masked and smoothed gradients.**
- Gradient masking and obfuscation can make single-step evaluations overstate robustness [Tramèr et al., 2018; Athalye et al., 2018a].
- Averaging gradients over random transformations or noise (EOT [Athalye et al., 2018b], SmoothGrad [Smilkov et al., 2017]) recovers more useful directions.
- Neighborhood gradient averaging improves transfer attacks [Wang & He, 2021].
- Gradients of deep networks can resemble noise [Balduzzi et al., 2017].

Our proposed mechanism study (§8) builds directly on these ideas.

**Saliency reliability.** Gradient attributions can be fragile under imperceptible perturbations, and insensitive to the model [Ghorbani et al., 2019; Adebayo et al., 2018]. We report a specific, attack-relevant failure: a matched-count ranking that is *worse than chance*.

**Sparse attacks.** L0 and sparse attacks show that the choice of pixels matters: JSMA [Papernot et al., 2016], one-pixel attacks [Su et al., 2019], SparseFool [Modas et al., 2019], PGD₀/CornerSearch [Croce & Hein, 2019] and Sparse-RS [Croce et al., 2022]. They typically *search* for small supports. We instead analyze the support an existing dense attack ends up with after quantization, at a matched count.

**Attacks on vision-language models.** Tiny L∞ image perturbations can steer and jailbreak multimodal models, and transfer across prompts in limited ways [Schlarmann & Hein, 2023; Zhao et al., 2023; Carlini et al., 2023; Qi et al., 2024; Bailey et al., 2024; Luo et al., 2024]. We use the smallest budget at which a perturbation survives 8-bit storage.

## 3. Setting and a structural fact

**Model and task.**
- **Model.** Qwen3-VL-2B-Instruct at a pinned revision, bf16 weights on Apple MPS, eager attention.
- **Questions.** Synthetic referent-colour questions, e.g. "What colour is the large triangle in the bottom row?".
- **Scenes.** Each 512 × 384 scene has 8 shapes on a 4 × 2 grid, with decoys that force every word of the question, and a fixed colour histogram so frequency cannot answer the question.
- **Counterfactual groups.** Two arms that differ only by swapping the referent's colour with one other object's colour.
- **Contexts.** Two non-revealing text contexts: `neutral`, and `helpful`, which tells the model to use all object words together.
- **Readout.** The answer is read from the four single-token colour logits.
- **Outcome.** The **margin** is the label logit minus the largest other colour logit. It is computed by a frozen scorer that first rounds the image to 8 bits (clamp, ×255, round half to even). The margin drop is D = margin(clean) − margin(perturbed).

**Attacks.**
- The image tensor has 3 × 384 × 512 = 589,824 **RGB channel coordinates**.
- **FGSM** takes one sign step of size ε = 1/255 on the labelled-colour NLL gradient in the *source* context.
- **PGD-3** takes three sign steps of ε/4, each projected onto the ε-ball.
- Each context serves in turn as source, and the image is also scored in the other (*target*) context.

**Proposition (PGD-3 is an FGSM subset after rounding).** Assume the clean image lies on the 8-bit grid, and consider a coordinate whose clean value is not at the [0, 1] bounds and whose clean gradient is non-zero. Three ε/4 steps never reach the ε-ball boundary, so each such coordinate's PGD-3 displacement is ±ε/4 or ±3ε/4. Rounding to the 8-bit grid (a spacing of ε) maps ±ε/4 to 0 and ±3ε/4 to ±ε. A coordinate therefore survives exactly when all three step signs agree, and its sign is then the first step's sign, which is FGSM's. *The rounded PGD-3 image equals the rounded FGSM image restricted to that agreement set, whatever steps 2 and 3 do.*

If steps 2 and 3 had independent random signs, the agreement set would be a uniform random 25% of FGSM's support. Empirically it is 29.8% (held-out; 95% CI 29.5–30.1%).

Empirically, the only exceptions arose where the clean gradient is exactly zero. There FGSM does not move the coordinate. If steps 2 and 3 agree, PGD moves it by ε/2, a rounding tie, and round-half-to-even can shift an odd 8-bit value by one level. This affected 1–7 coordinates in 84 of 96 held-out units (out of about 175,000 kept). Every such coordinate had exactly zero gradient, and no sign was ever reversed.

**Consequence.** "PGD-3 is a sparse, sign-consistent FGSM" is arithmetic, not evidence. The informative question is whether PGD's particular coordinates beat other coordinates drawn from the same support at the same count.

## 4. Method: a matched-count decomposition

Notation for each scene and source context:
- r_F is the rounded FGSM perturbation and S_F its support (median about 589,800 coordinates).
- k is the size of PGD-3's rounded support (about 148,000–196,000).

Four images are compared:

| Image | Coordinates changed | Values |
|---|---|---|
| FGSM | all of S_F | r_F |
| Random subset (8 seeds) | k coordinates drawn uniformly from S_F | r_F |
| Top-\|∇\| subset | the k coordinates of S_F with the largest \|clean NLL gradient\| | r_F |
| PGD-3 (official) | k | r_F on PGD's agreement set (§3), plus 0–7 zero-gradient tie coordinates outside S_F |

The decomposition uses three contrasts:
- **selection**, D_PGD − D_rand;
- **sparsity**, D_rand − D_FGSM;
- **saliency selection**, D_topk − D_rand.

The **sparsity share** is (D̄_rand − D̄_FGSM)/(D̄_PGD − D̄_FGSM). We also compute each subset's clean **first-order prediction**, −∇M·r, where ∇M is the clean margin gradient.

## 5. Two samples

**Development (exploratory).** The 48 arm-a scenes of the development design. The question arose while examining these scenes. A random-mask control on them was pre-registered before it ran (Checkpoint 2.3b).

**Held-out confirmation (pre-registered).** The confirmation split, `study2-trackB-confirm-v1`, is a set of 48 newly generated counterfactual groups.
- **Generator.** A new seeded generator draws them from a scene distribution specified before this study. Seed strings reserved for other studies are refused in code.
- **Separation from development data.** Each group is checked against all 216 earlier development scenes in three ways: by layout signature and its mirror images, by scene-graph hash, and by decoded-pixel hash. All three found zero overlap.
- **Freeze.** The protocol, the analysis code, the runner and the scene manifest were hash-frozen and pushed to a public repository before any model inference on these scenes (commit `c35b02c`). The runner refuses to start if any hash differs.
- **Pipeline check.** A smoke test on one development scene first reproduced earlier gradients, attack iterates and 22 official scores bit-for-bit.
- **Execution.** The run was a single shot with a time cap and no interim look at outcomes. The primary analysis uses arm a of each group.

**Pre-registered predictions.** All intervals are percentile bootstraps over groups (9,999 draws).
- **P1 (primary).** Mean selection > 0 with each context as source, and the pooled 95% CI lower bound > 0.
- **S1.** The pooled sparsity-share CI upper bound ≤ 0.25.
- **S2.** Mean saliency selection < 0 with each source, and the pooled CI upper bound < 0.

## 6. Results

**Run integrity.**
- All 48 groups completed (1,200 records) in 162.6 minutes against a 180-minute cap.
- An end-of-run drift check reproduced the first score bit-exactly.
- Memory pressure stayed normal throughout.
- There were no deviations from the frozen protocol.
- 45 of 48 groups were answered correctly in both contexts.

**Table 1. Pooled source-context margin drop (logits; mean, 95% CI).**

| Image (k matched) | Development | Held-out confirmation |
|---|---|---|
| FGSM | 3.52 [2.93, 4.10] | 3.32 [2.65, 4.03] |
| Random subset (mean of 8 seeds) | 4.00 [3.44, 4.55] | 3.54 [3.07, 4.05] |
| Top-\|∇\| subset | 2.41 [1.92, 2.92] | 2.16 [1.67, 2.69] |
| **PGD-3** | **9.86 [8.98, 10.75]** | **10.04 [9.35, 10.75]** |
| Selection (PGD − random) | 5.87 [5.24, 6.51] | **6.50 [6.08, 6.92]** |
| Sparsity (random − FGSM) | 0.48 [0.06, 0.89] | 0.23 [−0.16, 0.62] |
| Sparsity share of the gap | 0.08 [0.01, 0.14] | **0.03 [−0.02, 0.09]** |
| Saliency (top-\|∇\| − random) | −1.59 [−2.03, −1.17] | **−1.38 [−1.71, −1.06]** |

**Pre-registered decisions.** All three predictions were supported.
- **P1:** selection was +6.27 [5.71, 6.80] with neutral as source and +6.73 [6.20, 7.27] with helpful as source.
- **S1:** the upper bound of 0.087 is below 0.25.
- **S2:** saliency selection was −1.33 and −1.43 per source, with a pooled upper bound of −1.06.

**Consistency.**
- PGD's subset did more damage than the random mean in 48 of 48 groups for each source.
- It exceeded all eight random seeds in 96 of 96 group-by-source cases. In development the figure was 95 of 96.
- The median within-group standard deviation across random seeds is about 0.3 logits, against a selection effect of about 6.5.
- **Answer flips.** On clean-correct groups, PGD's subset flipped 60% (neutral source) and 78% (helpful source) of answers. Random subsets flipped 18% and 23%, and top-|∇| subsets 11% and 13%.

**Transfer.** Scored in the context it was not optimized for, PGD's selection advantage stays large: +5.54 [5.03, 6.05] and +6.64 [6.12, 7.20] by direction. The sparsity share remains small (0.02). The selected coordinates are not specific to one prompt.

**The first-order ranking is reversed.** The clean first-order prediction rates the subsets as follows:
- PGD's subset at 0.93× (neutral) and 0.92× (helpful) the random subset;
- the top-|∇| subset at 2.81× and 2.84× the random subset.

Observed damage is 2.83× random for PGD and 0.61× random for top-|∇|. Development gave the same ratios. The linear model also overestimates every subset's damage roughly tenfold, which is expected outside its regime of validity. The ordering error is the informative part.

**Figure 1** (repository README) shows Table 1 and the per-group scatter for both samples.

## 7. Discussion and limitations

**What the result means.** On this model and budget, an attacker who perturbs the right 30% of coordinates does about three times the damage of one who perturbs them all. Random thinning does almost nothing. Two common clean-gradient rules point *away* from the right coordinates: magnitude ranking and first-order prediction.

For safety evaluation, two consequences follow:
1. Single-step attack numbers can substantially understate vulnerability even when no gradient masking is intended.
2. Coordinate-level gradient saliency should not be read as identifying where a VLM is vulnerable without checking it against an iterative attack.

**What it does not mean.** The clean gradient may still contain the relevant information in some other function of it. We tested only two rules. Nor do we know *why* PGD's coordinates are damaging. The 3-way sign agreement that defines them suggests that stability of the gradient sign across nearby inputs matters, but that is a hypothesis (§8).

**Limitations.**
- **Scope.** One model at one precision (bf16 on MPS), one budget and step schedule, and one synthetic task family with two non-revealing contexts.
- **Structural property.** The subset property depends on ε being one 8-bit level and on steps of ε/4, and holds up to 0–7 zero-gradient rounding ties per image, which lie outside FGSM's support. Other settings need a different control.
- **What "selection" covers.** The matched comparison holds magnitude, sign and count fixed. "Selection" therefore includes the spatial arrangement of the kept coordinates and their interactions; it is not a per-coordinate importance claim. The sparsity share is a ratio of mean differences, not an additive decomposition of a nonlinear response.
- **Held-out split.** It shares the development distribution family. It is out-of-sample, not out-of-distribution.
- **The random null** is uniform within FGSM's support. Spatially structured nulls were not part of the confirmation.
- **Saliency** means raw clean-gradient magnitude. Smoothed or integrated attributions were not tested.
- **Hypothesis origin.** The hypotheses were formed on development data. Only the held-out split is confirmatory.

## 8. Exploratory follow-up: is PGD's selection gradient-sign consensus?

**Hypothesis.** PGD's selection rule is a three-way vote on gradient signs along its path. If a gradient decomposes into a component that is stable across the ε-ball and a rough component that decorrelates quickly, the vote enriches the stable component.

**Development-only pilot.** The pilot (pre-specified in `docs/checkpoint-3-mechanism/PLAN.md`) builds sign-consensus masks from gradients sampled at *attack-independent* random points. It matches their count and signs to PGD's, and compares them at equal gradient-evaluation budgets. Its controls separate genuine gradient stability from three alternatives:
- bf16 numerics;
- 8-bit grid effects;
- spatial clustering, via patch-matched random masks.

Neighbourhood gradient averaging itself is established [Smilkov et al., 2017; Athalye et al., 2018b; Wang & He, 2021]. The open question is whether it *explains* the coordinate selection of an existing iterative attack at matched count. Any positive pilot result would have needed confirmation on a new, independently seeded split before being claimed.

**Pilot result (exploratory, 16 development scenes; [`PILOT_REPORT.md`](../checkpoint-3-mechanism/PILOT_REPORT.md)).** The pre-specified reading is *consensus fails*.
- The strongest consensus mask (8 random points, 9 gradient evaluations) recovers R = 0.05 [−0.05, 0.14] of PGD-3's selection gain over random.
- At PGD's own budget of 3 evaluations, R = 0.08 [0.01, 0.15]. More points do not help.
- Smoothed-gradient magnitude ranking is worse than random, like clean top-|∇|.
- Shuffling PGD's coordinates within 16×16 patches keeps R ≈ 0.22, so about a fifth of the advantage is regional.
- Even an ε/16 move rotates the gradient to a cosine of 0.30 with the clean one.

For the samplers and consensus rules tested, attack-independent neighbourhood gradients therefore carry little of PGD's selection signal. Other sampling distributions, radii and aggregation rules were not tested, so this does not exclude a stable sign field that a different estimator could recover. A path-dependent account remains a hypothesis. This is consistent with, but does not establish, a hypothesis that many damaging subsets exist and PGD finds one of them. No confirmation was run, because the pilot was negative. The pilot does not show that every function of local gradients fails.

## Pre-registration, data and code availability

- **Frozen confirmation materials:** `docs/checkpoint-2.4-confirm/` holds the protocol, hashes, records, analysis and report.
- **Generator and code:** the scene generator is `src/scene_confirm/`; the runner and attack code are in `src/path_probe/`.
- **Scenes:** all rendered scenes, with manifests and hashes, are in `fixtures/`.
- **Earlier records:** every earlier record is preserved, and changes are annotated with dates.
- **What reproduces where:**
  - On the same hardware and software, the pipeline is bit-for-bit deterministic.
  - On other hardware, the estimates should reproduce but the bits may not.

## References

- Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018). Sanity checks for saliency maps. *NeurIPS*.
- Andriushchenko, M., & Flammarion, N. (2020). Understanding and improving fast adversarial training. *NeurIPS*.
- Athalye, A., Carlini, N., & Wagner, D. (2018a). Obfuscated gradients give a false sense of security. *ICML*.
- Athalye, A., Engstrom, L., Ilyas, A., & Kwok, K. (2018b). Synthesizing robust adversarial examples. *ICML*.
- Bailey, L., Ong, E., Russell, S., & Emmons, S. (2024). Image hijacks: adversarial images can control generative models at runtime. *ICML*.
- Balduzzi, D., Frean, M., Leary, L., Lewis, J. P., Ma, K. W.-D., & McWilliams, B. (2017). The shattered gradients problem. *ICML*.
- Carlini, N., & Wagner, D. (2017). Towards evaluating the robustness of neural networks. *IEEE S&P*.
- Carlini, N., et al. (2023). Are aligned neural networks adversarially aligned? *NeurIPS*.
- Croce, F., & Hein, M. (2019). Sparse and imperceivable adversarial attacks. *ICCV*.
- Croce, F., Andriushchenko, M., Singh, N. D., Flammarion, N., & Hein, M. (2022). Sparse-RS: a versatile framework for query-efficient sparse black-box adversarial attacks. *AAAI*.
- Ghorbani, A., Abid, A., & Zou, J. (2019). Interpretation of neural networks is fragile. *AAAI*.
- Goodfellow, I., Shlens, J., & Szegedy, C. (2015). Explaining and harnessing adversarial examples. *ICLR*.
- Kurakin, A., Goodfellow, I., & Bengio, S. (2017). Adversarial examples in the physical world. *ICLR Workshop*.
- Luo, H., Gu, J., Liu, F., & Torr, P. (2024). An image is worth 1000 lies: transferability of adversarial images across prompts on vision-language models. *ICLR*.
- Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards deep learning models resistant to adversarial attacks. *ICLR*.
- Modas, A., Moosavi-Dezfooli, S.-M., & Frossard, P. (2019). SparseFool: a few pixels make a big difference. *CVPR*.
- Moosavi-Dezfooli, S.-M., Fawzi, A., Uesato, J., & Frossard, P. (2019). Robustness via curvature regularization, and vice versa. *CVPR*.
- Papernot, N., McDaniel, P., Jha, S., Fredrikson, M., Celik, Z. B., & Swami, A. (2016). The limitations of deep learning in adversarial settings. *IEEE EuroS&P*.
- Qi, X., Huang, K., Panda, A., Henderson, P., Wang, M., & Mittal, P. (2024). Visual adversarial examples jailbreak aligned large language models. *AAAI*.
- Qin, C., et al. (2019). Adversarial robustness through local linearization. *NeurIPS*.
- Schlarmann, C., & Hein, M. (2023). On the adversarial robustness of multi-modal foundation models. *ICCV Workshops*.
- Simonyan, K., Vedaldi, A., & Zisserman, A. (2014). Deep inside convolutional networks: visualising image classification models and saliency maps. *ICLR Workshop*.
- Smilkov, D., Thorat, N., Kim, B., Viégas, F., & Wattenberg, M. (2017). SmoothGrad: removing noise by adding noise. *arXiv:1706.03825*.
- Su, J., Vargas, D. V., & Sakurai, K. (2019). One pixel attack for fooling deep neural networks. *IEEE TEVC*.
- Tramèr, F., Kurakin, A., Papernot, N., Goodfellow, I., Boneh, D., & McDaniel, P. (2018). Ensemble adversarial training: attacks and defenses. *ICLR*.
- Wang, X., & He, K. (2021). Enhancing the transferability of adversarial attacks through variance tuning. *CVPR*.
- Wong, E., Rice, L., & Kolter, J. Z. (2020). Fast is better than free: revisiting adversarial training. *ICLR*.
- Zhao, Y., et al. (2023). On evaluating adversarial robustness of large vision-language models. *NeurIPS*.

*Citation details are drawn from the project's literature notes and a limited web search on 2026-11-09. Each entry should be checked against the publisher's record before submission.*
