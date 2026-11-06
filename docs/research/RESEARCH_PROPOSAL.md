# Research proposal: when does one textual context's image attack transfer to another?

**Status.** Checkpoint 0 recommendation. No attack was run for this document, and no number below is a Study 2 measurement. The working title, "Which Modality Wins? Predicting and Exploiting Cross-Modal Attack Transfer in Vision-Language Models," is a label for the program, not a finding.

**Who decides.** The primary hypothesis and estimand below are this workstream's recommendation. They are not frozen until the research lead accepts them. Choices that would change the claim are listed at the end with options and tradeoffs.

**Audience.** A reader who can follow a gradient, and a reader who is new to adversarial machine learning. Symbols are defined where they appear.

## 1. The recommendation in one paragraph

Study 2 should pre-register a single confirmatory claim: on scenes the model already answers correctly, the sign alignment between the image gradients of two non-revealing textual contexts predicts whether a perturbation crafted under one context also causes a strict error under the other. That claim is named H\* below. Two neighbouring questions stay secondary. Whether non-revealing wording changes vulnerability at all is a paired contrast with its own null. Whether optimizing the image against both contexts at once raises cross-context error is a matched-budget comparison with methods that are already published. Causal interventions on internal representations stay exploratory and outside the primary claim.

## 2. Why this hypothesis, given what is already published

Cross-prompt transfer of an adversarial image is an established research topic. Luo, Gu, Liu, and Torr define it, show that an image optimized on one fixed prompt transfers poorly, and show that optimizing the same image against several fixed prompts (their Multi-P baseline) raises targeted success and then saturates. Their CroPA method goes further by perturbing prompt embeddings during the attack. An anonymous under-review manuscript on gradient-regularized cross-prompt attacks describes the same Multi-P construction and reports that joint optimization over many prompts is unstable. Cai and coauthors, at NeurIPS 2025, treat multi-prompt joint optimization as an existing, expensive route to prompt transfer and study a different attacker. Details and limits are in [`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md) and [`PRIOR_ART_MATRIX.md`](PRIOR_ART_MATRIX.md).

The quantity that is not established in the sources opened for this review is narrower. Demontis and coauthors showed that the cosine of two classifiers' input gradients predicts cross-model evasion transfer under a linearized loss. Madry and coauthors showed that multi-step projected gradient descent can leave that linear regime: some adversarial perturbations have a negative inner product with the loss gradient, and the correlation with the gradient direction falls as the perturbation grows. Nobody in the opened set estimates, on one vision-language model, whether the sign alignment of image gradients under two textual contexts predicts scene-level error transfer when neither context contains the answer.

H\* is that missing estimate. It is an application of a known first-order idea to a new axis (context instead of surrogate model), on scenes that block a shortcut Study 1 could not block. It is not a new attack algorithm.

## 3. What Study 1 contributes as a lesson

Study 1 (repository `fcavaletto/crossmodal-vlm-robustness`, commit `e73326b28deae5fe50dacd078eedd0c9c960aec3`) asked whether a misleading caption amplifies L∞ projected gradient descent relative to a supportive caption, on one large colored shape, with SmolVLM2-500M. The attack was crafted only under a neutral prompt. On the sealed 32-image test split the primary color difference-in-differences was negative (held-out PGD at ε = 8/255: −0.500, 95% interval [−0.688, −0.344]). Clean accuracy under the misleading color caption was already a floor (development 6/64 ≈ 9.4%; held-out 4/32 = 12.5%). The supportive caption restated the true label. Held-out color-neutral conditional attack success was 100% at ε = 2/255 and 93.1% at ε = 8/255, so the attack often had nowhere left to go. The technical report lists optimizing under other prompts, and supportive text that cannot be copied as the answer, as untested.

Those observations are lessons for the design. They are not laws about vision-language models in general. Study 2 should not repeat misleading-versus-supportive as the primary contrast. It should not treat a context that contains the answer as evidence of visual robustness. It should not assume a seven-step attack at ε = 2/255 still has variance left to explain.

## 4. Primary hypothesis

Let a scene be one image together with one question, one ground-truth answer string \(y\), and the four context strings named in the shared spec: `neutral`, `helpful_nonrevealing`, `contradictory`, and `answer_revealing`.

**H\* (confirmatory).** Among scenes the model answers correctly under both `neutral` and `helpful_nonrevealing`, higher sign alignment between those two contexts is associated with a higher probability that an L∞ PGD image crafted under one of them is strictly wrong under the other.

The association is the coefficient β_A defined in §6. The hypothesis is directional: β_A > 0. A two-sided interval is still reported. H\* is about association. The alignment and the transfer indicator are both consequences of the same model; a positive coefficient would not by itself be a causal effect of alignment.

### Why the other research questions are not the primary claim

The shared spec lists three research questions. They stay open. Only one is confirmatory.

- Context can change post-attack error even when the image is fixed. Cui and coauthors report that extra textual context mitigates visual attacks, with a much smaller drop on ScienceQA than on the visual encoder they attacked. Study 1's secondary contrast also moved with context, and the supportive side restated the answer. A primary claim that "context matters" would restate those results unless the text is forbidden from containing the answer. That leakage-controlled contrast is secondary (§10.1). It does not require gradient alignment, and it can be null while H\* is true, or the reverse.
- Crafting the perturbation on several prompts at once is Multi-P. Running it on two non-revealing contexts is a useful matched-budget control (§10.2). Calling it a new method would contradict Luo and coauthors.

Causal interventions on hidden states are out of H\* (§11).

## 5. Symbols

| Symbol | Meaning |
| --- | --- |
| \(x\) | Clean image, the tensor the attack differentiates. The implementation records the color space and the preprocessing path. |
| \(c\) | A textual context. The primary contexts are the keys \(s\) (source) and \(t\) (target). |
| \(y\) | Ground-truth answer string for the scene. Machine-checkable. |
| \(P(y \mid x, c)\) | Probability of the full answer token sequence given the image and the context, under teacher forcing. |
| \(L_c(x)\) | \(-\log P(y \mid x, c)\). Negative log probability of the whole answer. This is a sum of per-token negative log probabilities. |
| \(g_c\) | Gradient of \(L_c(x)\) with respect to the image \(x\). Same rank and layout as \(x\). |
| \(\operatorname{sign}(g_s)\) | Elementwise sign of the source gradient. Zero entries stay zero. |
| \(\lVert g_t \rVert_1\) | Sum of absolute values of the entries of \(g_t\). |
| \(A(s, t)\) | \(\langle g_t, \operatorname{sign}(g_s) \rangle / \lVert g_t \rVert_1\). Defined only when \(\lVert g_t \rVert_1 > 0\). |
| \(\varepsilon\) | L∞ radius. Each pixel of the perturbation is at most \(\varepsilon\) in absolute value, and pixels stay in the legal image range. |
| \(\delta_s\) | Perturbation produced by PGD on \(L_s\), with \(\lVert \delta_s \rVert_\infty \le \varepsilon\). |
| \(T(s \to t)\) | 1 if the attacked image \(x + \delta_s\) is strictly wrong under context \(t\), and 0 if it is strictly right. |
| β_A | Coefficient on \(A(s, t)\) in the risk model in §6. |

Strictly wrong means the parsed output is missing, empty, or not the ground-truth answer. Strictly right means the parser accepts the ground-truth answer. Invalid strings are errors, as in Study 1. A required decomposition splits invalid strings from wrong-but-valid answers (§8).

### What \(A(s, t)\) measures

Take a step of L∞ size \(\alpha > 0\) in the signed source direction, \(\alpha \operatorname{sign}(g_s)\). To first order, \(L_t\) changes by \(\alpha \, \langle g_t, \operatorname{sign}(g_s) \rangle\). The largest first-order increase available from any L∞ step of size \(\alpha\) is \(\alpha \lVert g_t \rVert_1\), achieved when the step's signs match \(g_t\). The ratio of those two quantities is \(A(s, t)\).

So \(A(s, t)\) is the share of the best one-step L∞ increase in the target loss that the source sign pattern captures. It lies in \([-1, 1]\) when every source entry is nonzero. The value 1 is perfect sign agreement. The value −1 is perfect disagreement. A zero gradient component in \(g_s\) contributes nothing to the numerator.

\(A(s, t)\) is not the cosine similarity used by Demontis and coauthors. Their \(R\) divides by both Euclidean norms and matches an L2 step. They note that other perturbation norms change the alignment formula. \(A\) is the L∞, sign-step version. Cosine similarity may be stored as a diagnostic. It is not the primary predictor.

\(A(s, t)\) and \(A(t, s)\) are different numbers. Both ordered pairs enter the model. They are not two independent scenes.

A positive constant multiplied into a loss cancels in \(A\), because it scales \(g_t\) in the numerator and the denominator and does not change \(\operatorname{sign}(g_s)\). Study 1 optimized the mean token negative log likelihood. The spec writes the sum, \(-\log P(y \mid x, c)\). Signs used by FGSM and PGD match either way. Reported loss values do not. The recommendation is to implement the sum, and to store the per-token mean as a diagnostic so Study 1 losses are not compared with Study 2 losses as if they were the same quantity.

If \(\lVert g_t \rVert_1 = 0\), \(A(s, t)\) is undefined. The row is excluded. It is not imputed as zero.

## 6. Estimand

The unit inside the regression is an ordered pair \((s, t)\) on one scene, with \(\{s, t\} = \{\texttt{neutral}, \texttt{helpful\_nonrevealing}\}\) and \(s \neq t\).

Restrict to pairs that pass all of the following. The rule is applied before looking at \(T\) or at β_A.

1. The scene is clean-correct under \(s\) and under \(t\).
2. Both contexts are marked `reveals_answer: false`, and the documented normalization finds \(y\) in neither string.
3. Within the scene's counterfactual group, the `neutral` string is identical across siblings and the `helpful_nonrevealing` string is identical across siblings, while \(y\) changes. A text-only reader then cannot score above chance across the group.
4. \(\lVert g_s \rVert_1 > 0\) and \(\lVert g_t \rVert_1 > 0\) in the dtype and tolerance Agent C records. A row that fails the finite, nonzero gradient gate is excluded, not coded as \(A = 0\).
5. The pair uses the frozen attack budget from §9.

On the included rows,

\[
T(s \to t) = \beta_0 + \beta_A A(s, t) + \beta_{\text{task}}^\top u + \beta_\varepsilon^\top v + \text{error}.
\]

\(u\) is a vector of task-family indicators (`attribute`, `counting`, `spatial`). \(v\) is a vector of budget indicators if more than one frozen budget is inside the primary (the recommendation in §9 is one frozen budget). The error is clustered by `counterfactual_group_id`, because sibling scenes share layout and the two directions share a scene. β_A is a change in transfer probability per unit of alignment. An alignment of 0 is the reference in this coding; moving from complete disagreement to complete agreement changes the modeled probability by \(2\beta_A\).

**Recommendation on the link function.** Fit this linear risk model as the primary estimand. Also fit a logistic regression with the same covariates and the same clustering, as a sensitivity analysis. The linear coefficient is readable as a probability difference. It can fall outside \([0, 1]\). The logistic coefficient is a log-odds change and is unstable if \(T\) is nearly constant. Agent D owns the interval estimator. The scientific requirement is that the interval respects the cluster, and that H\* is supported only when the interval for β_A lies entirely above 0.

Source-context success is not a covariate in the primary model. The attack outcome under \(s\) is partly determined by \(g_s\), and \(A\) is also a function of \(g_s\). Conditioning on source success would change the population mid-analysis. Report β_A inside the source-success slice and inside the source-failure slice as a stratification. If those two coefficients disagree in sign, the pooled β_A is not one mechanism, and H\* is not declared supported.

**Loss diagnostic, not a second primary.** Repeat the same linear model with outcome \(\Delta L_t = L_t(x + \delta_s) - L_t(x)\). \(A\) is derived as a predictor of this first-order change, up to the gap between one signed step and full PGD. If β_A is positive for \(\Delta L_t\) and not for \(T\), the attack objective moved and the parsed decision did not. If only \(T\) moves, inspect the invalid rate before treating H\* as supported.

## 7. Confounders

Each row states the threat, the control, and what remains uncontrolled.

**Answer leakage.** Study 1's supportive context restated \(y\). Control: \(y\) is not a substring of either primary context, `reveals_answer` is false, and both strings are constant inside a counterfactual group while \(y\) changes. A semantic paraphrase can still point at the answer without using the string ("the warm one" when the answer is red). The group-invariance control blocks paraphrases that differ across siblings. It does not block a paraphrase that is true for every sibling, because that paraphrase would not identify which sibling is which. Residual risk: the question itself can be easy for reasons that have nothing to do with the context sentence.

**Clean-accuracy floor.** Study 1 could not interpret additional error under a context the model already failed. Control: H\* keeps only scenes that are clean-correct under both primary contexts. `contradictory` is outside H\* because a false caption produced a floor in Study 1. `answer_revealing` is outside H\* because success there can be copying. Tasks whose clean accuracy on development falls below one half, inside a context that is outside H\*, are reported as floors and are not given an attack-amplification story.

**Attack saturation.** Study 1's neutral PGD conditional success was already about 100% at ε = 2/255. If \(T\) is almost always 1, or almost always 0, β_A has nothing to predict. Control: the identifiability window in §9. A saturated run is a non-identifiable result, not a rejection of H\*.

**Invalid outputs versus label swaps.** Study 1 found that many PGD failures were unparseable strings. Control: the primary endpoint counts invalids as errors, and a required table splits invalid, wrong-but-valid, and still-correct. An association that exists only in the invalid column means the attack broke the answer format.

**Scene difficulty masquerading as alignment.** Hard scenes can have messy gradients and high error under any perturbation, including noise. Control: the same regression of strict error on \(A(s, t)\), using an equal-budget uniform L∞ noise image in place of \(\delta_s\). \(A\) is always computed on the clean image. If the noise coefficient is also positive and its interval lies above 0, \(A\) is standing in for difficulty. H\* is then not declared supported, regardless of the PGD coefficient.

**Leaving the linear regime.** \(A\) is a one-step L∞ calculation. PGD takes several projected steps. Madry and coauthors already report adversarial perturbations whose inner product with the gradient is negative. Control: fit the same risk model to FGSM, the single signed step \(\varepsilon \operatorname{sign}(g_s)\), clipped to the image range. FGSM is the setting in which \(A\) is the right first-order summary. If the FGSM coefficient is positive and the PGD coefficient is not, the reading is that multi-step projection left the signed direction. That reading supports the caution in Madry and coauthors. It does not say that context is irrelevant.

**Budget and step count.** More steps or a larger ε can saturate \(T\). Control: one frozen budget for H\*, chosen by the window in §9, not by maximizing β_A.

**Gradient scale and sign noise.** \(A\) ignores the magnitude of \(g_s\) except through its sign. A tiny \(g_s\) can flip signs under dtype noise. Control: the nonzero gate in §6. Magnitudes are stored. They are not extra primary covariates, because adding them after seeing the fit would be a second analysis.

**Task and label imbalance.** Counting may be harder than color, and some answers may be frequent. Control: task-family indicators. Answer-label indicators are a sensitivity analysis if a label dominates a family. They are not required for the primary model unless development shows one answer string covering more than about half of a family. That check is descriptive and happens before the held-out fit.

**Counterfactual dependence.** Sibling images share objects and the same primary strings. Control: cluster on `counterfactual_group_id`. Scene-level intervals, as in Study 1, are a sensitivity analysis. Group clustering is wider when siblings are correlated. Scene clustering is anti-conservative in that case.

**Preprocessing mismatch.** Study 1 accepted a residual gap between the differentiable image path and the generation path. If Agent C's gradient path and `generate` path differ, \(A\) and \(T\) are not derivatives of the same function. The mismatch is recorded. Rows used in H\* must use the path pair Agent C documents. A known mismatch makes a positive β_A harder to interpret, and the report says so rather than adjusting ε to hide it.

**Device.** Study 1's measurements are Apple MPS on a MacBook Air M4 with 24 GB unified memory. A cloud CPU run is not an MPS result. Every later measurement names the device. This proposal contains no device measurement.

**Development contamination.** Choosing the budget, the inclusion floor, or the hypothesis after seeing the held-out coefficient would make the interval exploratory. Control: §9 and §12.

## 8. Endpoint decomposition

For every included pair, store three indicators after the attack, under context \(t\):

- strict error (the primary \(T\));
- invalid output;
- wrong but valid.

The same three numbers are stored under context \(s\), for the stratification in §6. Invalid is a property of the parser. The parser rule is frozen with the hypothesis, before held-out evaluation. Study 1's strict exact match is the starting proposal: after a light strip of whitespace and quotes, the string matches an answer or an alias, or it is invalid.

## 9. Budget rule

Default geometry, carried over from Study 1 so the lesson is comparable: L∞ ball, pixels clipped to the legal range, PGD with 7 steps, step size \(\varepsilon/4\), no random start, ε in the grid \(\{2/255,\ 4/255,\ 8/255\}\). FGSM is one signed step at the same ε. Noise is uniform in \([-\varepsilon, \varepsilon]\) and then clipped, with a recorded seed.

**Identifiability window, applied on development only.** Among rows that already pass the clean-correct and leakage rules, compute the source strict-error rate. A budget is inside the window when that rate is between 0.20 and 0.80 inclusive. These thresholds are a proposed operating rule so that \(T\) has variance. They are not a power calculation.

Order the candidates from milder to stronger. A larger radius is stronger. At a fixed radius, more steps are stronger.

1. ε = 2/255, 1 step
2. ε = 2/255, 3 steps
3. ε = 2/255, 7 steps
4. ε = 4/255, 1 step
5. ε = 4/255, 3 steps
6. ε = 4/255, 7 steps
7. ε = 8/255, 1 step
8. ε = 8/255, 3 steps
9. ε = 8/255, 7 steps

Freeze the first candidate whose rate lies inside the window. That candidate is the only primary budget. One step is FGSM, so if only the one-step candidate qualifies, H\* is a one-step claim and the separate FGSM column duplicates it. Say so.

Stop early if a candidate's rate is already above 0.80. Stronger candidates are not a remedy for an attack that is already too successful, and they are not evaluated for the primary choice. If the list ends with every evaluated rate below 0.20, or the rate jumps from below 0.20 to above 0.80 with no candidate inside the window, H\* is non-identifiable on this grid. Report the rates. Do not add a new ε, a new step count, or a new hypothesis afterward and call it confirmatory.

The walk uses development scenes only. It minimizes nothing about β_A. The mildest budget that still leaves variance is preferred because \(A\) is a local summary. When the frozen budget has more than one step, FGSM at the same ε remains the linear control.

## 10. Secondary analyses

These are specified so they are not promoted later if H\* is null.

### 10.1 Does non-revealing wording change vulnerability?

On scenes that are clean-correct under both `neutral` and `helpful_nonrevealing`, let additional error be the strict-error indicator after the attack minus the clean strict-error indicator. Clean strict error is 0 on this subset, so additional error equals post-attack strict error.

The secondary estimand is the mean paired difference, helpful minus neutral, for an attack crafted under neutral, and the swapped difference for an attack crafted under helpful. Intervals are clustered by counterfactual group. A difference of zero means these two wordings do not change additional error at the frozen budget. A nonzero difference means the wording changed vulnerability without containing \(y\). It does not mean that every context matters, and it does not mean that contradictory text amplifies attacks. `contradictory` and `answer_revealing` are reported as separate slices with their clean accuracy beside them. A low clean accuracy blocks any sentence of the form "the attack did more damage here."

### 10.2 Multi-context PGD is a published baseline

Define three image attacks at the frozen ε, with the same number of image-gradient evaluations:

- neutral-only PGD, then evaluate under `helpful_nonrevealing`;
- helpful-only PGD, then evaluate under `neutral`;
- multi-context PGD: each step averages \(g_{\text{neutral}}\) and \(g_{\text{helpful}}\) and steps on that average.

Averaging two gradients costs two backward passes. The multi-context attack therefore takes half as many outer steps as a single-context attack, or it accumulates the two gradients inside one outer step and counts both. The comparison that matches work is the one with equal backward counts. A comparison at equal outer steps gives multi-context twice the gradient budget and is reported only as a sensitivity.

The secondary estimand is the difference in target strict-error rate between multi-context PGD and the better of the two single-context attacks, on the same included population as H\*. A difference of zero means that, on this leakage-controlled pair, averaging the two losses did not buy cross-context errors beyond the stronger single context. That null does not speak to CroPA, which perturbs prompt embeddings, uses many prompts, optimizes for a chosen target string, and runs for far longer. A positive difference would be Multi-P reproduced on this scene family. It would not be a new optimizer.

`contradictory` may be added to the average only in an exploratory sensitivity. It is not part of the secondary estimand, because a context with a clean floor dominates a summed loss.

### 10.3 Contexts held out of H\*

| Context | Role |
| --- | --- |
| `neutral` | Primary source and target. Constant inside a counterfactual group. |
| `helpful_nonrevealing` | Primary source and target. Helpful relative to a bare question, constant inside the group, and free of \(y\). |
| `contradictory` | Stress slice. Report clean accuracy first. Eligible for an attack contrast only if development clean accuracy in that task is at least one half. |
| `answer_revealing` | Positive control for copying. Expected pattern, from Study 1's supportive condition: high clean accuracy and possibly low additional error. That pattern is leakage, not visual robustness. |

Illustrative strings, not frozen prompts: a helpful sentence can say "Answer from the picture. Several objects are present." A contradictory sentence can assert the wrong attribute. An answer-revealing sentence can contain \(y\). Agent B owns the actual strings, subject to the invariance and substring rules.

## 11. Exploratory track: causal interventions

The shared spec keeps causal interventions off the primary claim. The opened literature already contains several different interventions.

- Palit, Pandey, Arora, and Liang corrupt BLIP's image embedding and patch clean hidden states back in, on COCO-QA. Later layers matter for the answers they could score. Their Table 1 lists object-counting accuracy at 3.27%, location at 26.30%, and color at 80.23%. Counting is a poor first target for a tracing study on that setup. They adapt a language-model tracing procedure; the original tracing paper was not opened for this review.
- Li and coauthors estimate natural direct effects of vision, text, and their interaction on hallucinated descriptions, then intervene at test time. That is a hallucination study, not an attack-transfer study.
- Niu and coauthors subtract a question-only causal effect inside VQA models that were trained with an explicit language branch. SmolVLM2 does not have that branch. The subtraction does not transfer as a method.

A later exploratory analysis may ask whether pairs with small or negative \(A(s, t)\) are pairs whose correct answer, under a pre-specified intervention, depends more on text tokens than on image tokens. That analysis needs its own protocol, a model whose interventions Agent C can actually run, and a slice where clean accuracy is high enough to trace. It cannot repair a null H\*. It is not a Checkpoint 0 deliverable.

## 12. Sealed-set procedure

Design only. This procedure was not run.

**Target, later.** 384 scene seeds: 128 development, 256 sealed held-out. Checkpoint 0 may generate a development sample of 24 to 48 scenes. It does not generate the sealed split and does not inspect what that split would contain beyond this paragraph.

**Separation.** Development seeds and sealed seeds are disjoint. The sealed seed list is chosen by a recorded rule from a recorded master seed, written down before any sealed image is rendered. No sealed image, question, or model output is used to change prompts, the parser, the model revision, the budget walk, the inclusion rules, or which coefficient is primary.

**What development may decide.** The budget walk in §9. Descriptive clean-accuracy checks. Whether a task family's contradictory slice is above the one-half floor. Implementation bugs. Development may also be used to debug the estimator on toy or development rows. Any coefficient computed on development is labeled development. The confirmatory interval is the held-out interval only.

**What is forbidden after the first held-out estimate of β_A.** Changing ε, steps, contexts, the parser, the loss reduction (sum versus mean), the cluster unit, or the primary endpoint, and then presenting the new fit as H\*.

**How to read a surprise.** If the held-out interval lies above 0 and the development fit did not, report both and do not replace the held-out interval with the prettier one. If the held-out result is non-identifiable because of a floor that development did not show, report non-identifiable. Do not switch the primary contrast to `contradictory` or `answer_revealing`.

## 13. Data requirements implied by H\*

These are scientific constraints for the scene generator. They are not an implementation.

- Task families: object attribute, object counting, and a simple spatial relation. Several objects and distractors, so the scene is not Study 1's single central shape.
- Exact answers, aliases, and `ambiguity_flags`. A flagged scene is out of H\*.
- Matched counterfactuals that change \(y\) and hold the primary context strings fixed: an attribute swap on the queried object, a controlled count change, or a relation swap.
- The four context keys. Substring test on `helpful_nonrevealing`. Group-invariance test on `neutral` and `helpful_nonrevealing`.
- Label balance reported per family. H\* does not require perfect balance. A family with one answer covering more than half its scenes triggers the sensitivity analysis in §7.
- Development and sealed splits as in §12. No sealed generation at this checkpoint.
- Generalization beyond the generator is untested until an external set is licensed and scored. A synthetic result does not extend to photographs by default.

### CLEVR as a later external set

Johnson and coauthors built CLEVR as a rendered diagnostic for attributes, counts, and relations, with functional programs and an explicit attempt to limit question bias. That makes it a relevant later check on whether a result is an artifact of this repository's generator. It is not a substitute for the four context keys. CLEVR questions are long compositional programs, not `neutral` / `helpful_nonrevealing` / `contradictory` / `answer_revealing` wrappers.

The generator license opened at `facebookresearch/clevr-dataset-gen` (`main`, 6 Nov 2026) is a BSD license for that software, copyright 2017 Facebook. A separate license for a pre-rendered image archive was not opened. Checkpoint 0 does not download or vendor CLEVR. A later checkpoint needs a human-checked license for whatever files would actually be stored, plus a short suitability note that CLEVR questions are not the Study 2 context factor.

## 14. Models

**Primary baseline.** `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`. The model card opened on 6 Nov 2026 describes a 0.5B-parameter Apache-2.0 model for image, multi-image, video, and text, with an example that loads bfloat16 and FlashAttention on CUDA. This workstream did not load the weights and does not pin a revision. Study 1 pinned revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`, float32, eager attention, and MPS. Agent C compares the revision it actually loads with that pin. A mismatch is a protocol deviation, recorded before any confirmatory fit.

**Not adopted.** `Qwen/Qwen3-VL-2B-Instruct`. The card opened the same day describes a 2B-parameter BF16 vision-language model and advertises spatial perception. The card does not report clean accuracy on this scene family, image-gradient checks, MPS memory, or MPS runtime. Those four checks are the adoption criteria in the shared spec. Until Agent C reports them on the target device, Qwen3-VL-2B stays a candidate. The Qwen3 technical report linked from the card was not opened, so its benchmark tables are not evidence here.

No fine-tuning. Weights stay frozen. The adversary edits pixels only.

## 15. Attacks named, not run

| Name | Role in Study 2 | Prior art status in the opened sources |
| --- | --- | --- |
| Neutral-context PGD | Source attack and secondary contrast | Standard PGD (Madry et al.) on \(L_{\text{neutral}}\). Study 1's only crafting context. |
| Context-specific PGD | The other direction of H\* | Same optimizer on \(L_c\). Luo et al. call the one-prompt case Single-P. |
| Multi-context PGD | Secondary matched-budget comparison | Luo et al. Multi-P. Also the baseline in the anonymous gradient-regularization manuscript. |
| FGSM | Linear positive control | Goodfellow, Shlens, and Szegedy. |
| L∞ noise | Difficulty negative control | Equal-budget non-gradient perturbation, as in Study 1. |
| CroPA | Not in the primary protocol | Published attack. Out of scope unless the lead adds it later as a named baseline, with its prompt-embedding update disclosed as theirs. |

PGD here is projected gradient ascent on \(L_c\) because the loss is already the negative log probability of the true answer. Equivalently, it is projected gradient descent on \(\log P(y \mid x, c)\). The constraint set is the L∞ ball around \(x\), intersected with the legal pixel range.

## 16. How to read the confirmatory result

| What happened | Reading |
| --- | --- |
| Interval for β_A lies entirely above 0. The noise coefficient does not. The included development source-error rate was inside the window. Invalid-only and wrong-but-valid decompositions are reported. | H\* is supported on this model, these two contexts, this scene family, and this budget. The result is still an association. It does not establish a defense, a general law, or a new attack. |
| Interval for β_A covers 0, and the design was identifiable (window met, inclusion not tiny). | This sign-alignment summary does not predict strict-error transfer here. Context can still change errors (§10.1). Multi-P can still raise targeted success in the settings Luo et al. studied. Demontis et al.'s cross-model result is untouched. Do not replace H\* with a better-looking secondary contrast. |
| Interval for β_A lies entirely below 0. | The directional prediction fails. Discuss, without promoting any of them to a new primary claim: PGD left the signed direction; invalids dominate \(T\); a shared vulnerability is orthogonal to sign agreement. The FGSM column distinguishes the first of these. |
| FGSM coefficient above 0 and PGD coefficient covering 0. | Consistent with a first-order effect that several projected steps do not preserve. |
| Noise coefficient above 0. | \(A\) is confounded with difficulty. Do not read the PGD coefficient as directional transfer. |
| No mild-to-strong candidate has a source strict-error rate inside [0.20, 0.80], or the nonzero-gradient or clean-correct rules leave too few groups to cluster. | Non-identifiable. Report the rates and the exclusion counts. β_A is not evidence for or against H\*. |
| Held-out sign disagrees with a development fit. | Report both. The confirmatory number is the held-out interval. |

"Too few groups" is deliberately not given a fake sample-size calculation. Agent D should say, before held-out evaluation, how many groups the cluster estimator needs in order to return an interval. If development already falls short of that number, the plan stops at non-identifiable rather than loosening the inclusion rules.

## 17. What a later positive paper could honestly claim

If H\* is supported under the rules above, the claim is: for this frozen SmolVLM2 revision, on this synthetic family, with text that does not contain the answer, scene-level sign alignment of the answer-loss image gradients ranks cross-context PGD errors. The claim can sit next to three established facts without rewriting them. Prompts already change attack success. Multi-prompt PGD already exists. Cross-model gradient cosine already predicts transfer for the classifiers Demontis and coauthors studied.

## 18. Decisions requested

Numbered in `docs/checkpoint-0/AGENT_A_REPORT.md`. Work on the literature and on the written estimand can continue without those decisions. A confirmatory experiment should not start until decision 1 is accepted or explicitly revised.

## 19. Sources

Every source named above was opened. The bibliography and the limits of the search are in [`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md).
