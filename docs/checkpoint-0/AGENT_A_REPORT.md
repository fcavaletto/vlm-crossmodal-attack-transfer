# Agent and responsibility

Agent A, literature and research design, Checkpoint 0 of Study 2 (`vlm-crossmodal-attack-transfer`).

This workstream writes the proposal, the literature review, and the prior-art matrix. It does not implement the study, generate scenes, load model weights, run attacks, or start Checkpoint 1. It does not edit `README.md` or any other workstream's paths.

Owned paths:

- `docs/research/RESEARCH_PROPOSAL.md`
- `docs/research/LITERATURE_REVIEW.md`
- `docs/research/PRIOR_ART_MATRIX.md`
- `docs/checkpoint-0/AGENT_A_REPORT.md`

Branch: `checkpoint-0/research-design`.

# Executive summary

Study 2 should pre-register one confirmatory hypothesis, H\*. On scenes a frozen vision-language model already answers correctly, the sign alignment of the image gradients under `neutral` and `helpful_nonrevealing` predicts whether an L∞ PGD image crafted under one context is strictly wrong under the other. The estimand is the cluster-robust coefficient on that alignment. `contradictory` and `answer_revealing` stay out of it: Study 1's false captions floored clean accuracy, and its true captions contained the answer. Cross-prompt transfer and multi-prompt PGD are already published, as CroPA and its Multi-P baseline, so multi-context PGD is a matched-budget control rather than a method. Demontis and coauthors already showed that gradient cosine predicts transfer across classifiers; the open test is this sign-alignment functional, across two contexts of one model, with the answer string absent. A noise control catches difficulty confounding. An FGSM control matches the first-order story. If every budget saturates, the result is non-identifiable and is not read as evidence against H\*. Causal interventions stay exploratory. No experiment was run. The documents are the deliverable.

# Work completed

- Read the Checkpoint 0 shared spec and followed its file ownership, context keys, loss notation, and report sections.
- Read Study 1 at commit `e73326b28deae5fe50dacd078eedd0c9c960aec3`: the README, the frozen protocol, the freeze pins, the consolidation note, the pre-test addendum, and the technical report's abstract, related-work table, and held-out result paragraphs. Study 1 was not modified.
- Opened the papers listed in `docs/research/LITERATURE_REVIEW.md` §9. Citations in the three research documents are restricted to those pages.
- Wrote the proposal, including H\*, the estimand, confounders, controls, the budget window, the sealed-set procedure, the CLEVR license note, and the model recommendation.
- Wrote the literature review and the prior-art matrix, with Multi-P identified as the published form of multi-context PGD.
- Did not generate data, load SmolVLM2 or Qwen3-VL, run gradients, or run attacks.

# Key findings

Evidence from sources that were opened:

- Luo, Gu, Liu, and Torr define cross-prompt transfer and show that Single-P transfers poorly. Their Multi-P baseline, PGD against several fixed prompts, improves targeted success and then saturates. CroPA's additional mechanism is a learnable prompt embedding. An anonymous under-review manuscript and a NeurIPS 2025 mutual-information attack treat multi-prompt optimization as existing practice.
- Cui, Aparcedo, Jang, and Lim report that added textual context mitigates visual attacks, and that VQA questions off the attacked content are more resilient. Zhao and coauthors transfer image attacks across models with the question held fixed. Schlarmann and Hein change OpenFlamingo captions at ε = 1/255.
- Demontis and coauthors show that input-gradient cosine predicts cross-model evasion transfer for classifiers, under a linearized L2 argument. Madry and coauthors show that multi-step L∞ PGD can have a negative inner product with the loss gradient.
- Training-time "gradient conflict" and "modality competition" (PCGrad; Huang et al.; Peng et al.; Wei et al.) are about parameter updates while learning. They are not image gradients of a frozen model under two prompts.
- Counterfactual VQA methods split into exact synthetic scenes (CLEVR), subtracting a question-only effect (Niu et al.), and linguistic presuppositions (C-VQA, CFMM). Palit and coauthors' BLIP tracing tool is real; their COCO-QA counting accuracy in that paper is 3.27%.
- Study 1's held-out primary color contrast was negative (−0.500, 95% interval [−0.688, −0.344], n = 32, PGD, ε = 8/255). Misleading-color clean accuracy was 4/32. Neutral conditional PGD success was 100% at ε = 2/255. Supportive text restated the label. Crafting used the neutral prompt only.

Design conclusions, labeled as proposals:

- H\* is the recommended confirmatory hypothesis. The estimand, inclusion rules, and null readings are in the proposal §§4–9 and §16.
- Multi-context PGD is proposed only as a secondary Multi-P comparison at matched gradient-evaluation count.
- `answer_revealing` is a copying control. `contradictory` is a stress slice with a clean-accuracy gate.
- Qwen3-VL-2B-Instruct is not adopted. The model card was opened; clean accuracy, gradients, memory, and runtime were not measured.
- CLEVR is not downloaded. The generator software license opened on GitHub is BSD. A pre-rendered image archive license was not opened.

# Evidence

| Claim | Where it was read |
| --- | --- |
| Multi-P definition, saturation from about ten prompts, CroPA's prompt-ascent / image-descent loop | CroPA arXiv HTML and ICLR 2024 proceedings PDF |
| Non-stationarity of multi-prompt optimization | OpenReview PDF `I05Z6KjQ9K` (anonymous, under review) |
| Cross-model gradient cosine \(R\) | Demontis et al., USENIX Security 2019 PDF, equation for \(R\) |
| PGD leaving the gradient direction | Madry et al., arXiv:1706.06083v4 |
| ScienceQA context summary (8.10% versus 99.73%) | Cui et al., CVPR 2024 PDF |
| Study 1 held-out numbers and opt-at-neutral | Study 1 README, `docs/PROTOCOL.md`, technical report |
| CLEVR generator license text | `raw.githubusercontent.com/facebookresearch/clevr-dataset-gen/main/LICENSE` |
| SmolVLM2 card (0.5B, Apache-2.0) and Qwen3-VL-2B card (2B, BF16) | Hugging Face model cards, 6 Nov 2026 |
| Full source list and what was not opened | `docs/research/LITERATURE_REVIEW.md` §9 and the opening method note |

No Study 2 measurement exists. Nothing in the proposal is an attack success rate, a gradient norm, or a runtime.

# Design decisions and rationale

1. **One primary hypothesis, H\*.** The confirmatory claim is the association between sign alignment \(A(s, t)\) and strict-error transfer on the two non-revealing contexts. Rationale: cross-prompt attack success and multi-prompt PGD are already published; the unpublished estimate inside the opened set is whether this first-order summary ranks transfer once answer text is removed. A single primary coefficient keeps a null result from being replaced by a secondary contrast, which is the failure mode Study 1 deliberately avoided.

2. **Linear risk model, clustered by counterfactual group.** β_A is a change in transfer probability per unit alignment. Logistic regression is the sensitivity analysis. Group clustering matches the dependence created by sibling scenes and by using both directions of one scene.

3. **Source-attack success is a stratification, not a covariate.** It depends on the same source gradient as \(A\). Conditioning on it would change the population after the attack.

4. **Noise and FGSM are part of the reading, not optional plots.** Noise detects difficulty confounding. FGSM is the attack \(A\) actually approximates. Madry et al. give a reason PGD can disagree with that approximation.

5. **Budget ladder on development, frozen before held-out.** Study 1 saturated at seven steps and ε = 2/255. The window (source strict-error rate in [0.20, 0.80]) is an identifiability rule. It is not a search for the largest β_A.

6. **Sum token negative log likelihood, with the mean stored.** The spec writes \(L_c(x) = -\log P(y \mid x, c)\). Study 1 used the mean. Signs match. Numeric losses do not. \(A\) is invariant to a positive constant factor.

7. **CLEVR deferred.** The opened generator license is BSD for the software. The image archive license was not opened, and CLEVR questions are not the four context keys.

8. **SmolVLM2 remains the intended baseline, unpinned by this workstream.** This agent did not load weights. Study 1's revision is the comparison point for Agent C. Qwen3-VL-2B stays a candidate until clean accuracy, gradients, memory, and runtime exist on the target device.

# Known limitations

- The literature base is the pages opened on 6 November 2026, not an exhaustive search. A paper outside that list could already estimate H\*. The proposal claims only that the opened set does not.
- GrCPA is anonymous and under review. It is used as evidence that multi-prompt optimization is already under study, not as a confirmed benchmark number.
- DeBias-Attack was read as a preprint with a placeholder publication identifier.
- The CroPA reproducibility note is a preprint.
- Study 1's technical report was not re-read line by line. Cited numbers come from the README, the protocol, and the report passages that were opened.
- No power calculation was invented. "Too few groups" is left for Agent D to define before held-out evaluation.
- Illustrative context sentences in the proposal are not frozen prompts.
- This workstream did not verify MPS, memory, or the current Hugging Face revision of SmolVLM2. The model card's 1.8 GB video-inference figure is the card's claim.
- Author lists follow the title pages that were opened. Venues are named when the opened PDF or its URL stated them.

# Blockers

**Blocking.** None for this documentation checkpoint. The proposal can be reviewed as written.

**Non-blocking.**

- H\* is a recommendation until the research lead accepts or revises decision 1. Confirmatory experiments should wait. Documentation does not.
- Whether any ε and step count on SmolVLM2 lands inside the identifiability window is unknown, because no attack was run. The proposal treats a miss as a reportable outcome.
- Qwen3-VL suitability is unmeasured.
- The CLEVR image-archive license is unread.
- Agent C has not yet confirmed that image gradients of \(L_c\) are finite and nonzero, or that the gradient path matches generation. H\* is undefined without that gate.
- Device numbers, if any later checkpoint reports them, have to come from MPS on the MacBook Air M4. This cloud VM cannot supply them.

# Decisions requested from the research lead

1. **Accept H\* as the only confirmatory hypothesis.** Recommendation: yes. The estimand is β_A in the proposal §6, population `neutral` × `helpful_nonrevealing`, endpoint strict error. Tradeoff: a positive result is an association on a narrow slice, which is honest and small. Rejecting H\* in favor of "multi-context PGD works better" would put the primary claim on a method Luo et al. already published. Rejecting it in favor of "misleading text amplifies attacks" repeats Study 1's contrast, including the floor and the copying problem.

2. **Keep `contradictory` and `answer_revealing` out of H\*.** Recommendation: yes. Tradeoff: the study will not, as its main result, speak to hostile captions. Those slices remain in the report with clean accuracy shown first. Putting them into H\* reintroduces the Study 1 floor or the copying shortcut.

3. **Loss reduction.** Recommendation: implement \(-\log P(y \mid x, c)\) as the sum of token negative log likelihoods, and store the mean. Tradeoff: numeric losses will not match Study 1's mean. Signs and \(A\) will. Using the mean instead is compatible with H\* if it is declared, because a positive constant cancels in \(A\).

4. **Budget rule.** Recommendation: on development only, walk the proposal §9 list from milder to stronger (ε = 2/255, then 4/255, then 8/255; at each radius, 1 step, then 3, then 7). Freeze the first setting whose source strict-error rate lies in [0.20, 0.80]. If a setting is already above 0.80, stop; a larger attack is not a fix. If the list never enters the window, H\* is non-identifiable. Tradeoff: the thresholds are an operating rule, not a calculated sample size. A stricter window yields non-identifiable more often. A wider window lets a nearly flat \(T\) produce a noisy coefficient. One step is FGSM, so a one-step freeze makes the FGSM column a duplicate and should be labeled as such. Do not choose the budget by maximizing β_A.

5. **Link function and cluster.** Recommendation: linear risk model, clustered by `counterfactual_group_id`; logistic and scene-level clusters as sensitivities. Tradeoff: the linear fit can predict outside [0, 1]. Scene-level intervals match Study 1 and are anti-conservative if siblings are correlated. Agent D owns the estimator; the scientific constraint is the cluster and the directional interval.

6. **Qwen3-VL-2B-Instruct.** Recommendation: do not adopt it in the primary protocol. Revisit only if Agent C reports, on MPS, clean accuracy on the scene family, finite nonzero image gradients, memory, and runtime. Tradeoff: a second model would strengthen a positive H\* and costs budget the spec has not approved. The card's spatial-perception language is not that evidence.

7. **CLEVR.** Recommendation: out of Checkpoint 0 and out of the primary scene set. A later external check requires a human-read license for the files that would be stored. Tradeoff: without an external set, a positive result can be an artifact of the generator. Downloading CLEVR now would violate the checkpoint rule and might violate a license that was not opened.

8. **Causal interventions.** Recommendation: leave them exploratory, off the primary claim, and unstarted at Checkpoint 0. Tradeoff: the working title's "which modality wins" will not be answered by an internal intervention in the confirmatory paper. Palit et al. already show that tracing a task the model cannot do (their counting slice) is uninformative.

9. **Parser.** Recommendation: start from Study 1's strict exact match, with aliases allowed, invalid counted as error, and a required split of invalid versus wrong-but-valid. Tradeoff: a strict parser inflates error with format failures. Dropping invalids from \(T\) would change the endpoint after the fact if it is not pre-specified. The proposal pre-specifies the split so either pattern can be discussed without moving H\*.

# Reproduction instructions

These documents are Markdown. There is no package, no unit test, and no model run in this workstream.

From the repository root, on branch `checkpoint-0/research-design`:

```bash
git diff --check -- docs/research/RESEARCH_PROPOSAL.md docs/research/LITERATURE_REVIEW.md docs/research/PRIOR_ART_MATRIX.md docs/checkpoint-0/AGENT_A_REPORT.md
```

Exit code 0 means those files have no conflict markers and no whitespace errors that `git diff --check` reports. That is the test status: documentation only.

Study 1 was read from a separate clone and was not copied into this repository. Re-reading it requires that repository at commit `e73326b28deae5fe50dacd078eedd0c9c960aec3`. Do not vendor it here.

# Git information

- Branch: `checkpoint-0/research-design`
- Documentation commit: `a9cc0fbaff220f53b9f48016e10faf93b8c86d19`
- Pull request: https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/pull/1 (base `main`, not merged)
- The pull-request head is the commit that adds this Git section. Its parent is the documentation commit above.
- Test status: documentation only. No unit tests ship with this workstream. `git diff --check` on the four owned files passed with exit code 0 before the documentation commit. No attack, gradient, or scene test was run.
