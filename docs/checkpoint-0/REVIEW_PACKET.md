# Checkpoint 0 review packet

Review date: 6 November 2026. Repository: [vlm-crossmodal-attack-transfer](https://github.com/fcavaletto/vlm-crossmodal-attack-transfer). Research lead: Federico Cavaletto.

This packet reviews the four Checkpoint 0 workstreams. It is a review document. It is not a Study 2 measurement, and it does not accept a confirmatory claim. The four workstreams are unmerged. Each remains an open pull request against `main`. This branch does not merge them and does not copy their trees.

`main` at review time is `65a8d8ffb2f22f3d284b64e2d481151474761c5e`, the initial commit, whose tree is `README.md` only. No attack was run for this review. No sealed scene was generated or opened. Nothing was trained. No cloud GPU was used. Study 1 was not re-opened for this packet; where Study 1 numbers appear below, they are Agent A's citations of that repository, not figures re-checked here, and they are not Study 2 results.

## A. Project status and completed work

Checkpoint 0 asked for a research design, a development scene generator, a model interface with a minimal gradient check, and an analysis plan that can rebuild a toy summary. Those four products exist, on four branches, and they are ready for the research lead to read. They are not a finished experiment.

What is done:

- Agent A wrote the proposal, the literature review, and the prior-art matrix, and recommended one confirmatory hypothesis, called H\* there. No scene, weight, gradient, or attack was produced in that workstream.
- Agent B committed a deterministic generator and a 24-scene development sample, with a contact sheet, a manifest, checksums, and tests for leakage, shortcuts, counterfactual edits, ambiguity flags, and a refusal to write any split other than `development`. The sealed-set procedure is written and was not run.
- Agent C committed the scoring interface for `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` at revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`, the revision Agent C reports Study 1 froze. The same branch contains an MPS feasibility record from the author's Mac. This review did not repeat that run.
- Agent D committed table schemas, the rate and alignment formulas, a scene-level bootstrap, hand-written toy rows, and a command that rebuilds the committed toy summary. The primary estimand is explicitly unlocked.

What is not done, on purpose at this checkpoint: development or held-out attacks, a sealed split, training, a second model, and any estimate of H\* on real model output. Clean accuracy on the 24 scenes is unknown. Multi-context projected gradient descent is not a new method in these documents. Agent A's review identifies it with the Multi-P baseline of Luo, Gu, Liu, and Torr. The other three pull requests do not claim it as a contribution. Agent D only reserves a row label for it.

Deliverable paths were listed with `git ls-tree` at the pull-request heads below. All of the following are present: `docs/checkpoint-0/AGENT_A_REPORT.md`, `AGENT_B_REPORT.md`, `AGENT_C_REPORT.md`, and `AGENT_D_REPORT.md` on their own branches; `docs/research/RESEARCH_PROPOSAL.md`, `LITERATURE_REVIEW.md`, and `PRIOR_ART_MATRIX.md`; `docs/data/DATASET_DESIGN.md`; `fixtures/development/manifest.jsonl`, `fixtures/development/contact_sheet.png`, the 24 PNGs, checksums, metadata, and scene-graph hashes; `docs/models/MODEL_COMPATIBILITY.md` together with the feasibility JSON and the measured-environment note; `docs/statistics/ANALYSIS_PLAN.md`; and the packages `src/scene_gen`, `src/vlm_iface`, and `src/analysis` with their tests. There is no shared root `pyproject.toml`.

The only path that appears on more than one branch is `README.md`. Its blob is `245c0cbcdb0f401afa2f69536cec84e5d3042e5d` on `main` and on all four branches.

### Checks run for this packet

Scene generator, on a detached worktree at `d738f5f13868c08824ea94abb8abfe3693d549a2`, Linux, CPython 3.12.3, Pillow 11.3.0, pytest 9.2.0. This is not an MPS run.

```bash
PYTHONPATH=src/scene_gen /tmp/venv-scene/bin/python -m pytest tests/scene_gen -q
```

`44 passed in 1.56s`. Exit code 0. The system Python had no pytest, so the command is the author's command with that virtualenv interpreter.

Analysis, on a detached worktree at `5b522752571d8d9ebcfb86c724cf6023a9997d57`, Linux, CPython 3.12.3, numpy 1.26.4, pytest 8.3.5, after `pip install -e 'src/analysis[dev]'`. CPU arithmetic. Not an MPS run.

```bash
python -m pytest -q tests/analysis
python -m analysis.verify_toy
```

`41 passed in 0.39s`. `verify_toy` exited 0 with `"ok": true` and `"n_mismatches": 0`.

Model unit tests, on a detached worktree at `920fd4a5cce4bd9f84fdf5f804fae63391d538f9`. Linux, CPython 3.12.3, `torch 2.15.0+cpu`, `torchvision 0.29.2+cpu`, transformers 5.22.1, numpy 2.5.4, Pillow 13.0.0, pytest 9.2.0. `torch.backends.mps.is_available()` was false, and CUDA was unavailable. This is a CPU run. It is not the author's MPS measurement.

```bash
HF_HOME=/tmp/hf-c0 python -m pytest tests/vlm_iface -q -ra
```

`37 passed, 1 skipped in 7.06s`. The skip is `test_preprocess_refuses_a_non_cpu_image`: the test body moves a tensor to MPS, and the file skips with "MPS is not available on this interpreter" when MPS is absent. On the author's Mac that test can run, which is consistent with the author's report of 38 passed. The run downloaded the processor and tokenizer for revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`. The largest cached blob was about 3.5 MB. `model.safetensors` was not downloaded.

The feasibility script was not run:

```bash
python -m vlm_iface.feasibility --device mps --output docs/models/feasibility-smolvlm2-mps.json
```

This machine has no MPS device. That command loads the 500M weights. Pointing it at CPU would be a different measurement, so it was not pointed at CPU.

`git diff --check` from `main` to each of the four heads exited 0. Agent A's workstream ships no unit tests. That matches the report.

A separate manifest read of the 24 lines found every contracted field, `split` equal to `development` on every row, eight scenes in each family, 12 `counterfactual_group_id` values, `reveals_answer` true only for `answer_revealing`, empty `ambiguity_flags`, and `image_path` values of the form `images/<scene>.png`, each of which is a file next to the manifest. The dataset note describes the contact sheet as an 800-by-464 review figure. Opened for this review, it shows pair 00 of each family, arm A on the top row and arm B on the bottom, with the attribute recolor, the counting recolor, and the spatial swap that the note describes for those six scenes. It is not a model input.

### Throwaway merge

In a throwaway worktree at `main`, an octopus merge of the four heads was stopped before commit. Git reported "Automatic merge went well." A second check merged the same four heads one at a time with the `ort` strategy, in the order research design, model interface, scene generator, analysis plan. Each merge exited 0. There were no unmerged paths. The merged tree had 106 paths: `README.md` plus the files added by the four branches. That branch was deleted locally and was not pushed. Nothing from either merge is on this branch.

## B. Four-agent status table

| Agent | Result | Tests | PR | Remaining issues |
| --- | --- | --- | --- | --- |
| A. Literature and research design | Proposal, literature review, and prior-art matrix. H\* is a recommendation. No Study 2 measurement. | No unit tests. `git diff --check` passed. | [PR 1](https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/pull/1), head `4bde745901ee0d457d89fd073d34452678c9cb28` | H\* is not accepted. The opened literature is not an exhaustive search. Qwen and CLEVR stay out. |
| B. Scene generator | Deterministic 24-scene development sample, manifest, contact sheet, sealed procedure written and not run. | 44 passed in 1.56s on Linux. Not MPS. | [PR 3](https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/pull/3), head `d738f5f13868c08824ea94abb8abfe3693d549a2` | `image_path` join is an open reading. 128 and 256 do not pair evenly. Clean accuracy is unmeasured. |
| C. Model interface | Sum-log-probability interface, parser, and an author-recorded MPS probe of SmolVLM2. Qwen was not loaded. | CPU re-run: 37 passed, 1 skipped. MPS feasibility was not repeated. | [PR 2](https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/pull/2), head `920fd4a5cce4bd9f84fdf5f804fae63391d538f9` | The probe is 128×128, not the 512×384 scenes. Float32 and uint8 preprocessing differ. The parser does not apply scene aliases. |
| D. Statistics | Schemas, metrics, scene bootstrap, and a toy summary that rebuilds. Primary estimand unlocked. | 41 passed in 0.39s. `verify_toy` matched with 0 mismatches. Not MPS. | [PR 4](https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/pull/4), head `5b522752571d8d9ebcfb86c724cf6023a9997d57` | The coded illustration is not H\*. Group ids are not in the analysis schema. Toy arithmetic is not a finding. |

## C. One-paragraph summary from each agent

**Agent A.** Study 2 should pre-register one confirmatory hypothesis. On scenes a frozen vision-language model already answers correctly, the sign alignment of the image gradients under `neutral` and `helpful_nonrevealing` predicts whether an L∞ PGD image crafted under one of those contexts is strictly wrong under the other. The estimand is the cluster-robust coefficient on that alignment. `contradictory` and `answer_revealing` stay out of it, because Study 1's false captions floored clean accuracy and its true captions contained the answer. Cross-prompt transfer and multi-prompt PGD are already published, including CroPA and its Multi-P baseline, so multi-context PGD is a matched-budget control rather than a method. Demontis and coauthors showed that gradient cosine predicts transfer across classifiers; the open test, inside the sources Agent A opened, is this sign-alignment functional on two contexts of one model with the answer string absent. A noise control is there to catch difficulty confounding, and an FGSM control matches the first-order story. If every budget saturates, the result is non-identifiable and is not read as evidence against the hypothesis. No experiment was run.

**Agent B.** The branch contains a deterministic generator and 24 development scenes on a 4 by 2 grid, with a scene graph, one question, and the four context keys. The families are attribute color, counting, and "immediately left." The 24 scenes form 12 pairs. A pair shares the question and the three non-revealing texts. Attribute and counting pairs change one object's color. Spatial pairs exchange two objects' cells. Under the documented normalization, `helpful_nonrevealing` does not contain the answer or its aliases, and the same absence holds for the neutral sentence, the contradictory sentence, and the question. Only `answer_revealing` is marked as containing the true answer. Forty-four tests passed. No model ran, no attack ran, and no held-out scene was generated. The later totals of 128 and 256 scenes do not divide evenly into three families of pairs. Those counts are an open choice.

**Agent C.** The interface loads SmolVLM2-500M-Video-Instruct at the Study 1 revision. On the author's MacBook Air M4 with 24 GB unified memory, PyTorch 2.15.0 ran the float32 eager model on MPS. For a synthetic red square and the two-token answer "red square," the joint log-probability was −19.2657, and the library mean token loss matched that value divided by 2. Gradients of the sum loss on two 128×128 synthetic images were finite, nonzero in every recorded component, identical on a repeat, and different across the two images. One backward took about 4.4 seconds, and the MPS driver cache reached 4131.7 MB. Those figures are the author's, from `docs/models/feasibility-smolvlm2-mps.json`. This review checked that the report matches that file and did not re-run it. Qwen3-VL-2B-Instruct was not loaded. Its float32 expansion, from a header read, is about 8.51 GB, above a 6 GiB guard. Scoring and generation share a float32 preprocess that does not match the official uint8 resize. One greedy decode returned `Red.`, which the parser accepted as `red`. The committed JSON marks that string as a smoke check, not a clean-accuracy result. No attack was implemented.

**Agent D.** The package defines JSONL tables for predictions, losses, gradient summaries, and attack rows, and it computes clean accuracy, the invalid-output rate, attack success, additional error, cross-context transfer, and the alignment \(A(s, t)\). An invalid generation stays out of the success count. A rate with a zero denominator is null. Paired comparisons use a scene-level percentile bootstrap. A separate routine records how interval width changes for fabricated scene differences, and that routine is marked as not an empirical finding. No model was loaded and no attack was run. The committed summary is arithmetic on hand-written rows. Role labels are proposals. The primary estimand is not locked. `python -m analysis.verify_toy` rebuilds the summary. Agent D's own recommendation for a primary RQ1 contrast is a paired difference in additional error, and the report says Agent A's estimand overrides that recommendation.

## D. Links to all four full reports

Blob links at the reviewed heads:

- Agent A, `4bde745901ee0d457d89fd073d34452678c9cb28`: [AGENT_A_REPORT.md](https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/blob/4bde745901ee0d457d89fd073d34452678c9cb28/docs/checkpoint-0/AGENT_A_REPORT.md)
- Agent B, `d738f5f13868c08824ea94abb8abfe3693d549a2`: [AGENT_B_REPORT.md](https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/blob/d738f5f13868c08824ea94abb8abfe3693d549a2/docs/checkpoint-0/AGENT_B_REPORT.md)
- Agent C, `920fd4a5cce4bd9f84fdf5f804fae63391d538f9`: [AGENT_C_REPORT.md](https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/blob/920fd4a5cce4bd9f84fdf5f804fae63391d538f9/docs/checkpoint-0/AGENT_C_REPORT.md)
- Agent D, `5b522752571d8d9ebcfb86c724cf6023a9997d57`: [AGENT_D_REPORT.md](https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/blob/5b522752571d8d9ebcfb86c724cf6023a9997d57/docs/checkpoint-0/AGENT_D_REPORT.md)

The reports rest on these documents at the same commits: [research proposal](https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/blob/4bde745901ee0d457d89fd073d34452678c9cb28/docs/research/RESEARCH_PROPOSAL.md), [literature review](https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/blob/4bde745901ee0d457d89fd073d34452678c9cb28/docs/research/LITERATURE_REVIEW.md), [prior-art matrix](https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/blob/4bde745901ee0d457d89fd073d34452678c9cb28/docs/research/PRIOR_ART_MATRIX.md), [dataset design](https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/blob/d738f5f13868c08824ea94abb8abfe3693d549a2/docs/data/DATASET_DESIGN.md), [model compatibility](https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/blob/920fd4a5cce4bd9f84fdf5f804fae63391d538f9/docs/models/MODEL_COMPATIBILITY.md), [MPS feasibility JSON](https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/blob/920fd4a5cce4bd9f84fdf5f804fae63391d538f9/docs/models/feasibility-smolvlm2-mps.json), and [analysis plan](https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/blob/5b522752571d8d9ebcfb86c724cf6023a9997d57/docs/statistics/ANALYSIS_PLAN.md).

## E. Proposed primary hypothesis

The following is Agent A's proposal. It is not an accepted claim. The analysis plan still has `primary_estimand_locked` false.

**H\*, as proposed.** Among scenes the model answers correctly under both `neutral` and `helpful_nonrevealing`, higher sign alignment between those two contexts is associated with a higher probability that an L∞ PGD image crafted under one of them is strictly wrong under the other.

The symbols, in the proposal's wording, are these. \(x\) is the clean image. \(c\) is a textual context. \(y\) is the ground-truth answer string. \(P(y \mid x, c)\) is the probability of the full answer token sequence under teacher forcing. The loss is \(L_c(x) = -\log P(y \mid x, c)\), a sum of per-token negative log probabilities. \(g_c\) is the gradient of that loss with respect to the image. For a source context \(s\) and an evaluation context \(t\),

\[
A(s, t) = \langle g_t, \operatorname{sign}(g_s) \rangle / \lVert g_t \rVert_1
\]

when the L1 norm is positive. Zero entries of the source gradient stay zero in the sign. The ratio is the share of the best one-step L∞ increase in the target loss that the source sign pattern captures. It is not the cosine similarity used by Demontis and coauthors, and a first-order alignment is not proof that multi-step PGD will transfer.

The proposed estimand is the coefficient \(\beta_A\) in a linear risk model for the indicator \(T(s \to t)\), which is 1 when the attacked image is strictly wrong under the target context. The model includes task-family indicators. The errors are clustered by `counterfactual_group_id`. The hypothesis is directional, \(\beta_A > 0\), and a two-sided interval is still reported. A positive coefficient would be an association. The alignment and the transfer indicator are both consequences of the same model. Strictly wrong, in the proposal, includes a missing, empty, or non-matching parse, so an invalid string counts as an error, and a separate table is required to split invalid strings from wrong-but-valid answers.

The population is the two ordered pairs of `neutral` and `helpful_nonrevealing` on scenes that are clean-correct under both, that pass the leakage and group-invariance checks, and that have finite nonzero gradients, at one budget frozen on development by an identifiability window rather than by maximizing \(\beta_A\). `contradictory` and `answer_revealing` are outside this population in the proposal. Causal interventions on internal representations are exploratory and unstarted.

Agent A asks that H\* be the only confirmatory hypothesis. Rejecting it in favor of "multi-context PGD works better" would put the primary claim on a method the opened CroPA paper already published. Rejecting it in favor of "misleading text amplifies attacks" would repeat the Study 1 contrast the proposal is trying to leave. This packet does not make either replacement, and it does not treat H\* as accepted.

## F. Assessment of benchmark feasibility

The development instrument is feasible. The generator is deterministic, refuses a sealed split before creating a directory, and the 24-scene sample passes the tests re-run here. Each manifest row has the contracted fields: scene id, seed, `split` of `development`, task family, counterfactual group, question, answer, aliases, the four contexts with text and `reveals_answer`, objects, `image_path`, and ambiguity flags. Answers in this sample are single words. Attribute and spatial answers are colors. Counting answers are `two` and `three`, with digit aliases. Pairs share the question and the three non-revealing strings while the answer changes, which is the dependence structure Agent A asked for. The contact sheet matches that pair-00 description and is labeled as a review figure.

The leakage control that the tests actually enforce is a substring test under `normalize_for_leakage`, including aliases and the digits for count words. That test passed. It does not prove that a paraphrase is free of the answer. `answer_revealing` contains the canonical answer and is the copying control. A high score there would be leakage, not visual robustness.

Several limits are already measured on the sample and are properties of this generator, not of a model. Object counts are 4, 8, and 5 by family, so the picture identifies the family before the question is read. Spatial anchors occupy only columns 2 and 3, and the only relation is image-left. Attribute and spatial questions never target a small square or a small triangle. True counts are only two and three. Counting scenes fill every cell. Pair images differ, but the 24 scenes are 12 dependent pairs. The drawings are flat geometric figures. A later result on them is a result on this generator.

What is not feasible to claim yet is that the benchmark is a task SmolVLM2 can solve, or that it has headroom for an attack. Clean accuracy was not measured. Study 1's misleading-color floor is a lesson about that protocol, not a prediction for these scenes. The contradictory sentence states a wrong answer in a short line, so a model that copies text will be wrong for a textual reason. Crowded eight-object counts may be hard for a visual reason. Neither statement has a measurement behind it.

The sealed procedure is a written sequence: freeze the generator and the templates, publish development graph hashes, draw a master seed from a public beacon, render sealed scenes outside git, and reject collisions with development. It is not implemented. No sealed image exists in the repository. CLEVR was not downloaded. Agent A opened a BSD license for the CLEVR generator software and did not open a pre-rendered image-archive license. Agent B's note reaches the same recommendation. CLEVR questions are not these four context keys.

The headline totals of 128 development scenes and 256 sealed scenes are not divisible by three families and by pairs of two. Agent B recommends, when an expansion is written, 132 development scenes and 252 sealed scenes, and recommends keeping this 24 until that design exists. Generating either total now is outside Checkpoint 0 and outside the next checkpoint recommended below.

## G. Model and hardware feasibility

SmolVLM2-500M-Video-Instruct at revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467` is the primary baseline Agent A named and the only checkpoint Agent C loaded. Agent C's committed record says the Hub revision matches Study 1's pin, the parameter dtype is float32, attention is eager, image splitting is off, and the loaded parameters sit on MPS. The weight header in that record has 507,482,304 float32 parameters and 2,029,929,216 tensor bytes. This review did not reload the weights and did not repeat the timings.

The MPS numbers below are quoted from the author's JSON and the model report. They are one run on an Apple M4 MacBook Air with 25,769,803,776 bytes of unified memory, macOS 26.6, Python 3.12.14, torch 2.15.0, transformers 5.22.1. The requested device, the resolved device, and the parameter device were `mps`. They are not CPU numbers, and the CPU unit tests in this review do not corroborate them.

On the probe prompt "What color is the central square? Reply with one word." and the answer "red square," the joint log-probability was −19.265708923339844 over two tokens. The absolute difference between the library mean negative log likelihood and that sum divided by 2 was 0. Gradients with respect to a 128×128 RGB image had nonzero fraction 1 on a red square and on a blue square. The repeat maximum absolute difference was 0. The red-versus-blue maximum absolute difference was 4.779532432556152. Backward times were 4.594 s, 4.357 s, and 4.388 s. After the gradients, the MPS driver allocation was 4131.7 MB, the current allocation was 1965.6 MB, and the process peak RSS was 602.0 MB. The author says the driver figure is a cache high-water, that this PyTorch build has no separate MPS peak-activation counter, and that 602 MB is not the model footprint. OS available memory before the load was 8,435,777,536 bytes, so the machine was not empty. The finite-difference check of the loss was done on a tiny probe and a closed-form score. It was not repeated on the 500M model. The gradient entries are large enough that the author warns against reading the maximum absolute value, about 4.76 on the blue square, as an attack step size. This packet repeats that warning.

Those timings and memory figures are for 128×128 probes drawn in the script. They are not dataset items. The development canvas is 512 by 384. With splitting off, the processor's square side is 512, so the vision encoder sees a 512×512 tensor either way, but the input-gradient tensor and the resize backward for a scene are larger than the probe. Checkpoint 1 has to measure the scenes. It cannot reuse 4.4 seconds or 4131.7 MB as the scene cost.

The float32 path and the official uint8 path differ. On the author's machine the mean absolute gap in normalized space was about 0.0079 on a 96×128 image and about 0.0089 on a 512×512 image, with a maximum of about 0.79 on the smaller one. Scoring, the gradient, and `generate` share the float32 path, so they see the same pixels as each other. A script that scores with the raw PIL chat template does not. The CPU unit tests exercise that preprocess code path. They do not reproduce the MPS table.

Qwen3-VL-2B-Instruct at revision `89644892e4d85e24eaac8bacfd4f463576704203` was not adopted. The author's header read, 76,240 bytes, counts 2,127,532,032 BF16 parameters and a float32 expansion of 8,510,128,128 bytes. The weight tensor bytes were not downloaded. The 6 GiB guard is a policy for a 24 GB machine, not a measured out-of-memory failure. There is no clean accuracy, gradient, or runtime for Qwen. Agent A and Agent C both recommend leaving it out. The CPU unit test that refuses the Qwen model id passed without downloading it.

The hardware conclusion is narrow. One float32 SmolVLM2 backward fit on the author's laptop for a 128×128 probe, with a driver cache of about 4.1 GB after the run. That is enough to attempt a development-scene pilot on that machine. It is not a budget for a second model, for image splitting, or for an attack sweep. This cloud VM cannot add an MPS data point.

## H. Statistical analysis readiness

The analysis package is ready to check its own definitions. It is not ready to estimate H\*.

What is implemented and tested: clean accuracy, with an invalid output counted as not correct; an invalid-output rate with the same denominator; attack success, where a success is a valid wrong answer and an invalid attacked output stays in the denominator; additional error, where an invalid output counts as task error; cross-context cells; and

\[
A(s, t) = \operatorname{dot}(g_t, \operatorname{sign}(g_s)) / \lVert g_t \rVert_1
\]

with zeros in the source, including negative zero, mapped to sign 0, and a zero target L1 norm returned as null rather than imputed as 0. The scene bootstrap is an equal-tailed percentile interval. Zero-denominator rates are null. The toy verification command rebuilds the published summary from the committed fixtures. The re-run here matched.

What those tests do not say: the toy fractions, including any interval stored in `toy_summary.json`, are arithmetic on six hand-written scenes. The analysis plan says the illustration contrast is unlocked and that quoting its interval as a context effect would misread the file. This packet does not repeat those fractions as results. The precision simulation draws an independent three-point distribution. Its JSON sets `not_an_empirical_finding` true. It does not select a sample size, and it does not answer Agent A's question of how many counterfactual groups the cluster estimator needs.

The scientific gap is the estimand. Agent A's primary quantity is \(\beta_A\) in a linear risk model clustered by `counterfactual_group_id`. Agent D's plan labels a paired difference in additional error as the proposed primary for RQ1, labels the association between \(A(s, t)\) and a transfer indicator as a proposed secondary, and says in the report that Agent A's estimand overrides the RQ1 recommendation. The code does not fit \(\beta_A\). The analysis schemas have no `counterfactual_group_id`, no token count, and no device field. The bootstrap can take a scene-to-cluster map, and the toy summary does not pass one. A flat source gradient yields \(A = 0\) from the function, while Agent A's inclusion rule drops a row whose source L1 norm is 0. Attack success and Agent A's strict-error indicator also disagree when the failure is an invalid string: strict error counts it, attack success does not. On the clean-correct subset Agent A uses, additional error matches post-attack strict error, so the additional-error definition can carry H\* once the population is the H\* population. The contradictory-versus-neutral illustration in the toy file is a different contrast.

Until the lead accepts or revises H\*, and until the estimator and the cluster unit match that choice, a confirmatory interval cannot be computed. Development output from the next checkpoint can be summarized with the existing rates only if every number is labeled development and none of it is used to replace the hypothesis.

## I. Major scientific risks and mitigation options

The mitigations below are the ones the agents already wrote. They are options for the lead, not new claims.

**The primary question is still two different questions.** Agent A wants one coefficient, \(\beta_A\), on a leakage-controlled pair of contexts. Agent D's unlocked illustration is a difference in additional error, and the toy file happens to use `contradictory` against `neutral`. If that illustration is later quoted as the result, the study will have answered a different question from H\*, on a context the proposal held out because of Study 1's floor. Mitigation: the lead accepts H\*, revises it explicitly, or says that no confirmatory contrast exists yet. Do not let a secondary table become the claim after a null.

**Clean accuracy on these scenes is unknown.** Eight-object counts and a contradictory sentence that states a false answer can both put a context on the floor. Additional error is then hard to read, and H\*’s inclusion rule, which keeps only scenes that are clean-correct under both primary contexts, may leave very few of the 12 groups. Mitigation: measure clean accuracy, the invalid rate, and the wrong-but-valid rate before any attack. Agent A’s rule for the contradictory slice is to show clean accuracy first and to allow an attack contrast only if development clean accuracy in that task is at least one half. A low clean accuracy is reported as a floor.

**Saturation can make H\* non-identifiable.** Study 1, as cited by Agent A, had little room left at a small radius under a neutral prompt. That is a lesson, not a law about this generator. If the source strict-error rate is near 0 or near 1, \(\beta_A\) has nothing to predict. Mitigation: Agent A’s development-only walk, from milder to stronger, freezing the first budget whose source strict-error rate lies in [0.20, 0.80], and stopping if a budget is already above 0.80. A miss is reported as non-identifiable. The budget is not chosen by maximizing \(\beta_A\). This walk is not part of the next checkpoint.

**\(A(s, t)\) is a one-step summary.** Multi-step PGD can leave the signed direction. Agent A cites Madry and coauthors for that possibility and pairs PGD with FGSM, which is the attack the alignment actually approximates, and with equal-budget noise, which is there so that a hard scene is not mistaken for alignment. Mitigation: when attacks are eventually run, keep both controls, compute \(A\) on the clean image, and do not read a positive noise coefficient as directional transfer. None of those attacks has been run.

**Twelve groups is a small cluster sample, and the minimum is undefined.** Sibling scenes share layout and text. Treating 24 scenes as 24 independent draws overstates precision. Agent A wants the group as the primary cluster. Agent D’s implemented default is the scene, with the group as a sensitivity, and Agent D did not state how many groups the estimator needs. Mitigation: the lead picks the primary cluster before any interval is treated as confirmatory. The 24-scene pilot cannot confirm H\*. "Too few groups" remains a stop rule that still needs a number before a held-out fit, not before this review.

**The gradient and the parsed decision can come from different image functions.** The float32 path used for the loss and for `generate` agrees with itself and disagrees with uint8 PIL, by the gap Agent C recorded. Mixing a PIL forward with this gradient would make \(A\) and the decision derivatives of different functions. Mitigation: keep scoring, gradients, and generation on the documented float32 path, and report the gap rather than tuning a radius to hide it.

**The prompt that the model will see is not in the manifest.** Agent B tested one join, a context, a newline, and the question, with no extra label words. Agent C’s MPS prompt was a different sentence, on a single central square, and it asked for one word. Using that probe as if it were the study would measure a different task, the one Study 2’s generator was built to leave. Mitigation: freeze the user string before the pilot, using a join the leakage tests cover, and let the interface wrap it in the chat template.

**Answer strings and the parser are not yet one rule.** Agent A’s endpoint counts invalid output as error and also asks for a split between invalid and wrong-but-valid. Agent C’s parser judges format, not the scene label, and a string longer than 4 tokens or 32 characters is invalid. Agent D’s attack success does not count an invalid output as success. The current canonical answers fit the cap. A later spatial phrase might not. Nothing in the packages compares a parse to `answer_aliases`. Mitigation: the lead writes the alias rule before any accuracy is stored. Dropping invalids from the endpoint after seeing the pilot would move H\*.

**The sample can leak the family and can be over-read as natural images.** Constant object counts, a single spatial relation, and a gray geometric canvas are part of the generator. Mitigation: state the population as this generator. Keep CLEVR out until a person has read the license of the files that would actually be stored. Do not describe a synthetic association as a general law, and do not describe Study 1’s held-out contrast as one either.

**Development contamination.** The next checkpoint will be the first time model outputs exist for these scenes. Choosing the parser, the loss reduction, the cluster, or the primary endpoint after seeing those outputs, and then calling the fit confirmatory, would spend the held-out claim early. Mitigation: label every pilot number as development. Do not generate the sealed split. Do not retune H\* to fit the pilot.

**A large probe gradient is not a step size, and a toy interval is not an effect.** The blue-square gradient reached a maximum absolute value of about 4.76 in the author’s record. The toy summary’s intervals describe six typed rows. Mitigation: quote the gradient only as a finiteness check, and quote the toy file only as a rebuild check.

## J. Required decisions before development experiments

Each item states the options, the tradeoff the relevant agent already gave, and that agent's recommendation. Where two agents disagree, both recommendations are stated. This list does not add a new primary claim.

1. **Confirm or revise H\* before any number is treated as the study.** Options: accept Agent A’s H\* as the only confirmatory hypothesis; replace it with "multi-context PGD works better"; or replace it with "misleading text amplifies attacks." Tradeoff, from Agent A: a positive H\* is an association on a narrow slice, which is honest and small. The multi-context replacement would rest on a method Luo and coauthors already published. The misleading-text replacement repeats Study 1’s contrast, including the floor and the copying problem. Recommendation: Agent A recommends accepting H\*. Agent D’s report says that estimand overrides D’s RQ1 recommendation. The analysis code still labels a different contrast as proposed primary and leaves it unlocked. The lead’s choice is what locks it. This packet does not lock it.

2. **Keep `contradictory` and `answer_revealing` out of H\*.** Options: leave them out, or put them into the confirmatory population. Tradeoff, from Agent A: the main result will not speak to hostile captions. Those slices remain in the report with clean accuracy shown first. Putting them into H\* reintroduces the Study 1 floor or the copying shortcut. Recommendation: Agent A recommends leaving them out. They are still scored, as slices, when clean accuracy is measured.

3. **Loss reduction.** Options: implement \(L = -\log P(y \mid x, c)\) as the sum of token negative log likelihoods, or use Study 1’s mean token negative log likelihood. Tradeoff: Agent A notes that a positive scale cancels in \(A(s, t)\) and in the sign pattern, so the mean is compatible with H\* if it is declared, while the stored loss values will not match Study 1. Agent C notes that the mean equals the sum divided by the token count \(T\), and that the two losses share a minimizer across answers only when every answer has the same \(T\). For one fixed answer, the L∞ sign step Agent A specifies is the same under either reduction. Recommendation: both Agent A and Agent C recommend the sum, and both ask to store \(T\) or the per-token mean beside it. Agent D’s loss row is only `scene_id`, `context`, and `loss`. A development log that follows the recommendation has to carry \(T\) as well.

4. **The user string passed to the model.** Options: Agent B’s tested join, one context’s text, a newline, and the question, with no extra label words; or some other assembly. Tradeoff, from Agent B: words such as "Context:" were not part of the leakage-tested string, and the manifest does not store a combined prompt. Agent C wraps whatever user string it is given in the SmolVLM chat template. The feasibility prompt is not a scene prompt. Recommendation: Agent B recommends the newline join. A different string needs the leakage check run again before the pilot.

5. **`image_path`.** Options: resolve paths relative to `fixtures/development/`, the directory that contains the manifest, which is what the committed values do (`images/dev-attribute-00-a.png`); or rewrite them relative to `fixtures/`. The shared spec says "relative to the fixtures root," which can be read either way. Tradeoff, from Agent B: a loader that joins the stored paths onto `fixtures/` will miss the files. The change is small, and it has to happen before a model reads the sample. Recommendation: Agent B recommends the manifest directory, so the folder can move as a unit.

6. **Parser, invalid output, and wrong-but-valid output.** Options: Agent A’s starting rule, a strict match to the answer or an alias after a light strip of whitespace and quotes, with anything else invalid; Agent C’s format parser, which marks truncation, emptiness, line breaks, refusals, and strings over 4 tokens or 32 characters invalid, and does not consult the scene label; and, for attack success only, Agent D’s alternative of dropping invalid attacked outputs from the denominator. Tradeoff: Agent A says a strict parser inflates error with format failures, and dropping invalids from \(T\) after the fact moves the endpoint. Agent C says that, until the answer strings are frozen, a long string should come back invalid rather than be guessed. Agent D says that dropping invalids can make attack success high when the attack mostly produces garbage. The 24 canonical answers are short enough for the current cap. No package yet applies `answer_aliases`. Recommendation, as stated by each: Agent A recommends the strict match, aliases allowed, invalid counted as error, and a required split of invalid versus wrong-but-valid. Agent C recommends keeping the format parser and the provisional cap. Agent D recommends keeping invalids in the attack-success denominator as non-successes. The lead has to say whether "wrong but valid" means "a well-formed string that is not the answer" or "a string inside a closed vocabulary that is not the answer." That choice is a scoring rule. It is not a new hypothesis.

7. **Cluster unit.** Options: cluster on `counterfactual_group_id`, or resample scenes. Tradeoff: Agent A says siblings share layout and the two directions of one scene share a scene, so scene-level intervals are anti-conservative when siblings are correlated, while the linear risk model can predict outside [0, 1] and logistic regression is the sensitivity. Agent D says the scene bootstrap keeps the two contexts of a scene together, and that a group resample is available but unused because the toy rows have no group ids. Recommendation: Agent A recommends the group as the primary cluster and scene-level intervals as a sensitivity. Agent D recommends the scene as the default and the group as a pre-specified sensitivity. They disagree on which one is primary. The lead chooses. This packet does not replace H\* with a scene-level mean difference.

8. **Qwen3-VL-2B-Instruct.** Options: adopt it in the primary protocol, or leave it unadopted. Tradeoff: Agent A says a second model would strengthen a positive H\* and costs budget that has not been approved, and the model card’s spatial-perception language is not evidence of clean accuracy, gradients, memory, or runtime. Agent C adds that the float32 weight expansion is past the 6 GiB guard and that the weights were not loaded. Recommendation: both recommend not adopting it. A later checkpoint could reconsider a lower-precision load only with an explicit memory budget and with the four missing measurements.

9. **Whether to grow the scene sample before the pilot.** Options: keep the committed 24; add a second real layout, up to 48; or generate a 128/256 or 132/252 set now. Tradeoff, from Agent B: 24 is a small picture set, and 48 independent layouts need a new table and a new balance check. A reflected copy would double-count one layout. 128 and 256 cannot be split into three equal pair-friendly family sizes. Recommendation: Agent B recommends keeping 24 until a written expansion, and recommends 132 development and 252 sealed when the totals are revised. Keeping 128 and 256 is acceptable only as an explicit choice of unequal families. Do not generate those totals for the pilot.

10. **SmolVLM2 scoring path.** These are Agent C’s recommendations, and they change the numbers a pilot would store. Leading-space answer tokens: keep them. On the probe they were much more probable than the bare token, which matches a chat template that ends at "Assistant:". The bare tokenization remains available for an audit. Float32 preprocess shared by scoring, gradients, and generation: keep it, and document the uint8 gap. A straight-through uint8 path would match the PIL forward values and would not be the derivative of those values. Image splitting: keep it off. The processor default would tile toward a long edge of 2048, and that setting was not measured. `max_new_tokens`: keep 16, against Study 1’s 12, while answers may be short phrases. The one-token smoke generation stopped on an end token, so the cap did not bind there. The tradeoff of each alternative is a different log-probability, a different gradient, or a different generation, which is why the path should be accepted or revised before the pilot writes a table.

11. **Attack budget, recorded now and not used in the next checkpoint.** Options: Agent A’s mild-to-strong list, or another grid. The list walks \(\varepsilon \in \{2/255, 4/255, 8/255\}\) and, at each radius, 1 step, then 3, then 7. Freeze the first setting whose development source strict-error rate lies in [0.20, 0.80]. If a setting is already above 0.80, stop. If the list never enters the window, H\* is non-identifiable. Tradeoff, from Agent A: the thresholds are an operating rule, not a calculated sample size. A stricter window is non-identifiable more often. A wider window lets a nearly flat transfer indicator produce a noisy coefficient. One step is FGSM, so a one-step freeze makes the FGSM column a duplicate and has to be labeled as such. The budget is not chosen by maximizing \(\beta_A\). Recommendation: Agent A recommends that rule. Checkpoint 1 below does not run it.

Decisions that can wait until after the clean pilot, with the recommendation already on record: CLEVR stays out of the primary set until a human has read the license of the stored files (Agent A and Agent B). Causal interventions stay exploratory and unstarted (Agent A). The contradictory sentence stays the same on both arms of a pair, count answers stay at two and three, the spatial relation stays image-left, and shape stays the category, with material omitted (Agent B). The confirmatory bootstrap seed is not the toy seed 20261106, and no p-value is added at this stage (Agent D). A `perturbation_id` is for real attack logs, which Checkpoint 1 does not write.

## K. Exact recommended next checkpoint

Checkpoint 1 is a development-data pilot on the 24 scenes that are already committed. It is not a confirmatory study.

In scope, after the lead has recorded decisions 3, 4, 5, 6, and 10, or has explicitly kept the agents’ recommendations:

- Use only `fixtures/development/`.
- Load only `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` at revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`, float32, eager attention, image splitting off.
- Run it on the MacBook Air M4 under MPS. Write the device on every timing, memory, loss, and accuracy number. A CPU run is not a substitute.
- For each of the 24 scenes and each of the four contexts, generate a parse, apply the alias rule from decision 6, and store clean correctness, the invalid flag, and the wrong-but-valid flag separately. Store the sum loss and the token count \(T\).
- On the clean 512×384 scene image, compute the image gradient of that loss under `neutral` and under `helpful_nonrevealing`, and store \(A(s, t)\) both ways. Rows with a zero target L1 norm stay undefined. Rows with a zero source L1 norm are reported as excluded from H\*’s population, even if the algebra returns 0.
- Label every rate and every coefficient as development. Do not call the pilot a test of H\*.

Out of scope:

- The sealed held-out set: no generation, no inspection, no evaluation.
- Training or fine-tuning.
- `Qwen/Qwen3-VL-2B-Instruct` or any second model.
- PGD, FGSM, noise, multi-context optimization, or the budget walk in decision 11.
- A confirmatory interval, a p-value, or a revision of H\* fitted to the pilot.
- Cloud GPU spend.

Attacks wait because the inclusion rule and the contradictory gate both depend on clean accuracy, and that accuracy does not exist yet. The budget walk would also spend the identifiability rule on these 12 groups. Agent A asked that a confirmatory experiment wait until H\* is accepted or explicitly revised. The pilot is the measurement that comes first. A later development checkpoint can run the budget walk on development scenes only, still without a sealed split, after this pilot has been read.

## L. A list of all relevant commits and PR URLs, with full SHAs

`main`, the parent of every workstream:

- `65a8d8ffb2f22f3d284b64e2d481151474761c5e` — Initial commit. Tree: `README.md` only.

Pull request 1, open, base `main`, branch `checkpoint-0/research-design`: https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/pull/1

- `a9cc0fbaff220f53b9f48016e10faf93b8c86d19` — Add Checkpoint 0 research design and literature review. Parent: `65a8d8ffb2f22f3d284b64e2d481151474761c5e`.
- `4bde745901ee0d457d89fd073d34452678c9cb28` — Record the Checkpoint 0 pull request in the Agent A report. Head.

Pull request 2, open, base `main`, branch `checkpoint-0/model-gradients`: https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/pull/2

- `5fd7c3cc88d934658ab89aacb81e6350ad0ba5b2` — Add the SmolVLM2 log-probability and image-gradient interface. Parent: `65a8d8ffb2f22f3d284b64e2d481151474761c5e`. Author: fcavaletto.
- `920fd4a5cce4bd9f84fdf5f804fae63391d538f9` — Record the Checkpoint 0 pull request in the model report. Head.

Pull request 3, open, base `main`, branch `checkpoint-0/scene-generator`: https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/pull/3

- `1c04873cec4038137aaf71f462587b75d26e6c62` — Add the Checkpoint 0 development scene generator. Parent: `65a8d8ffb2f22f3d284b64e2d481151474761c5e`.
- `d738f5f13868c08824ea94abb8abfe3693d549a2` — Record the scene-generator pull request URL. Head.

Pull request 4, open, base `main`, branch `checkpoint-0/analysis-plan`: https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/pull/4

- `1058ef2bdf54a3e0b9b1ce53740cff32604cd7e9` — Add Checkpoint 0 toy metrics and a rebuild check. Parent: `65a8d8ffb2f22f3d284b64e2d481151474761c5e`.
- `5b522752571d8d9ebcfb86c724cf6023a9997d57` — Record the Checkpoint 0 pull request URL in the analysis report. Head.

The four pull requests were open at these heads when this packet was written. None is merged.

## M. A brief integration plan

The four branches can be combined. A throwaway merge onto `main` was clean, both as one octopus merge stopped before commit and as four sequential `ort` merges. The only overlapping path is `README.md`, and that file is identical on every branch. The merged tree is the four workstreams side by side. Packages do not import each other. There is no shared root project file. Pillow 11.3.0, which the scene package pins, sits inside the model package’s Pillow range. Numpy 1.26.4, which the analysis package pins, sits inside the model package’s numpy range. Torch is a dependency of the model package only. That is a file-level and dependency-level statement. The merge was not pushed, and this branch does not contain it.

The interfaces that still conflict, or that are not wired, are these.

- **Estimand.** H\* is a cluster-robust coefficient on \(A(s, t)\). The analysis plan’s proposed primary is a paired additional-error contrast, unlocked, and the report defers to Agent A without implementing the regression. Combining the branches does not resolve that.
- **Cluster unit.** Agent A’s primary cluster is `counterfactual_group_id`. Agent D’s default resample is the scene. The group id lives on the scene manifest and is absent from the analysis schemas. A later join can pass it in. The toy summary does not.
- **Loss storage.** Agent A and Agent C agree on the sum, and on storing \(T\) or the mean. For a fixed answer the sign pattern and \(A(s, t)\) agree under the mean as well, because \(T\) scales both contexts of one scene. The analysis loss schema cannot store \(T\). Numeric losses from Study 1’s mean and Study 2’s sum are different quantities.
- **Alignment edge case.** The formula for \(A(s, t)\) matches across the shared spec, the proposal, and the analysis code. A flat source gradient is excluded by Agent A’s inclusion rule and returned as 0 by Agent D’s function when the target norm is positive.
- **Invalid versus wrong.** Agent A’s strict error counts an invalid parse as an error and asks for a separate wrong-but-valid count. Agent C separates format-invalid from factual correctness and does not apply aliases. Agent D’s attack success does not treat invalid output as success; additional error does treat it as task error. The alias comparison is unwritten.
- **`image_path`.** Stored paths are relative to `fixtures/development/`. A loader that uses `fixtures/` as the root will not find the PNGs. Agent B asked for confirmation. The model interface takes an image tensor, not a manifest path, so the join has to be written when the pilot loads scenes.
- **Prompt.** The manifest stores the context and the question separately. The model call takes one user string. Those are not yet the same object.
- **Attacks.** No branch implements PGD, FGSM, or noise. Agent D’s `pgd_multi` / `source_context=multi` pair is a schema label for the published Multi-P control. It is not an optimizer, and it is not a claim that the method is new.

A practical order, after the decisions in section J, is to put the four branches in one development branch without merging to `main`, write the thin loader that joins `image_path`, builds the prompt, calls `vlm_iface`, and emits Agent D’s rows plus \(T\) and the group id, and run the Checkpoint 1 pilot on MPS. That loader is not part of this packet.
