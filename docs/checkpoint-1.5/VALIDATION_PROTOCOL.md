# Checkpoint 1.5 validation protocol

**DEVELOPMENT ONLY.** This file was written before any model call on `fixtures/development_v2`. It locks the candidate setup and the success and failure criteria for that run. It does not authorize an attack, a fit of H\*, a held-out scene, training, or a cloud GPU. Checkpoint 1 files under `docs/checkpoint-1/` are not overwritten.

Research lead: Federico Cavaletto. Common ancestor: `90c6ffe83d61514da217a20f8dfc2f202afbbe69`.

## What was read before this protocol

Agent A, `docs/checkpoint-1.5/AGENT_A_REPORT.md` at `7d6094855ade2ed36eabde6daf46da333603858b`. The strict parser still labels the stored 96 generations as 31 correct, 45 valid-but-wrong, and 20 invalid. The report recommends keeping that parser, including the sentence rule. The embedded single-label reading stays stored and is not selected.

The same report measured one format line, `Reply using only the answer. Do not write a sentence.`, appended to every context. Strict accuracy on the stored 24 scenes moved from 31/96 to 39/96. Agent A says that movement is a description of that sample and is not a reason to adopt the line. Contradictory counting stayed at 0/8 and spatial stayed at or below 3/8. The recommendation, written before any result on the v2 scenes, is to leave the Checkpoint 1 D2 prompt unchanged. This run does not append the line. Adopting it after reading 39/96 would choose a prompt on the old development sample. Adding it would also change the text inside H\*.

The class loss in that report is the negative log-sum-exp over the spellings D4 already accepts: the canonical word and its single-token case variants, plus the documented digit, including the two-token leading-space digit. Agent A recommends differentiating that scalar on a later development gradient and keeping the canonical-word loss beside it. The stated reason is that a gradient of the word alone moves probability off a continuation the official evaluator already counts as that class. The 14-of-15 agreement with the strict parse is a description of the old sample, not the selection criterion. Changing the differentiated loss would change the image gradient inside \(A(s, t)\). The strict-error endpoint would stay the same while D4 remains the label. This run stores the class-loss argmin beside the strict parse. It does not differentiate the class loss. The image-gradient check uses the canonical-word loss, which is the loss the Checkpoint 1 full-resolution gate already differentiates. Switching the scalar is a lead decision and is not taken here.

Agent B, at `28c50d320fb780fa7a7450abd7bdba6d18b443a7`, supplies 72 development scenes in 36 pairs. No scene and no family is dropped before the table exists. Agent C, at `b27cfff03c029e0d0f19052793a09cf645678dd2`, keeps SmolVLM2 and does not load Qwen. Agent D, at `be8be387fe93fd071d31543273baacfa476cb61c`, returns an interval only at 5 or more groups for the three-family design. On the fabricated shared-uniform grid, 24 was the smallest point whose estimated coverage met the pre-specified 90 percent description. That count does not promise an interval that excludes zero. H\* is not fit to these scenes.

## Candidate setup

| Choice | Locked value |
| --- | --- |
| Model | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |
| Device and dtype | MPS, float32, eager attention. A CPU number is not an MPS number |
| Image | `do_image_splitting=False`. Generation and the loss share the float32 preprocess |
| Generation | greedy, `max_new_tokens=16`, leading-space answer tokens |
| Prompt | `context_text + "\n" + question`. No format line. No `Context:` or `Question:` label |
| Helpful text | Task guidance, not additional factual scene evidence. The manifest strings are used as stored |
| Official score | Checkpoint 1 strict parser (D4). Correct, valid-but-wrong, and invalid are a partition |
| Stored beside it | Raw text, parsed label, valid-wrong, invalid, and the class-loss argmin. The canonical-word argmin is stored too and is not official |
| Scenes | Every row of `fixtures/development_v2/manifest.jsonl`, sha256 `b62ab1171a0aec7b2f264615a2961b6d384d00a1a4b17977d619ea27481154e1`, all four contexts. 72 scenes, 36 groups, 24 in each family |
| Qwen | Not loaded |
| Attacks and H\* | Not run and not fit. Alignments of the two clean gradients may be stored as compact scalars. They are not entered in a regression |
| Checkpoint 1 | The 24-scene tables stay the comparison. They are not recomputed and not overwritten |

The manifest sha256 is the committed file. The scenes are not regenerated for this run.

## Full-resolution gate

Before the 288-row table, run the Checkpoint 1 gate on `dev2-attribute-00-a`, the first committed v2 scene. The canvas is 512 by 384. For `neutral` and for `helpful_nonrevealing`, differentiate the canonical-word loss of the true answer with `image_loss_gradient`, and repeat each backward once.

The gate fails if the loaded checkpoint is not float32 on MPS with eager attention, if image splitting is on, if the generation settings differ from the table above, if the float32 and uint8 preprocess shapes differ, or if either gradient is missing, non-finite, or has L1 norm zero. A failed gate stops the run. The 288-row table is not produced. `repeat_max_abs_diff` is recorded and is not itself a failure.

The load starts only when the OS reports at least 6 GiB available, the same guard as Checkpoint 0. A refusal at that guard is not a model measurement.

## Scoring stability and the gradient pass

The scoring run is stable only when all 288 rows exist, every stored class loss and canonical-word loss is finite, the process did not raise, and OS-available memory after scoring is still at least 6 GiB.

If it is stable, compute the canonical-word image gradient on `neutral` and `helpful_nonrevealing` for every v2 scene. A non-finite or zero gradient on a later scene is stored. It does not delete the scene. If the scoring run is not stable, do not start that pass. The gate above is the one-scene gradient check, and the summary records why the full pass was not run.

## What is counted

Clean accuracy, valid-wrong, and invalid are strict-parser counts. They are reported by context and by task family and context. The denominator is every scene in that cell. Nothing is removed to approach 70 percent.

Counterfactual pair correctness, for each context, is the share of the 36 pairs in which both arms are strict-correct. The summary also records how often both arms have a valid parse and whether those parses differ. A pair is the two scenes that share a `counterfactual_group_id`.

Agreement between the official score and the generated answer is whether the stored official fields are the strict parse of `raw_output`. A row that fails that check does not enter the summary. Agreement between the official score and the class-loss argmin is, among valid rows, the share whose class-loss argmin equals the strict label. Invalid rows stay out of that rate, which is the Checkpoint 1 disagreement rule. The disagreement flag uses the same \(10^{-6}\) tolerance and the same name tie break.

A scene is clean-correct on both primary contexts when the strict parse is correct under `neutral` and under `helpful_nonrevealing`.

An eligible group is a `counterfactual_group_id` with at least one scene that meets every Agent D inclusion gate that does not read an attack outcome:

1. Strict-correct under both primary contexts.
2. Canonical-word image gradients on both of those contexts are finite and have L1 norm greater than zero.
3. `ambiguity_flags` is empty.
4. Both primary contexts are marked `reveals_answer` false.
5. The primary context strings are identical across the two scenes in the group.

The eligible-group count is the number of such groups. It is undefined when the full gradient pass was not run. The clean-correct scene count and the clean-correct group count are still reported. Neither number is a coefficient, and neither is produced by fitting H\*.

## Success and failure

These criteria were fixed before the v2 numbers. Missing them is an acceptable result. It is not a reason to delete a row, a scene, or a family.

About 70 percent means the closed band from 0.60 through 0.80 inclusive. A family meets the design target when its strict accuracy is inside that band in both `neutral` and `helpful_nonrevealing`. A family is unsuitable when its strict accuracy is below 0.50 in either of those contexts. The recommendation is then to drop or replace that family. The full table still contains it. A family that is at or above 0.50 but outside the band misses the design target. The recommendation is a lead decision. The scenes are not edited after the number is known.

The instrument is adequate for a later three-family transfer study only when all three families meet the design target and the eligible-group count is at least 5. Five is the algebraic floor at which Agent D returns an interval for the three-family design. Twenty-four is the fabricated coverage description from Agent D. It is reported beside the count. It is not an inclusion rule and it does not promise an interval excluding zero. A count below 5, or below 24, does not delete scenes.

Original Checkpoint 1 development performance, cited and not refit: strict accuracy 10/24 `neutral`, 5/24 `helpful_nonrevealing`, 3/24 `contradictory`, 13/24 `answer_revealing`. By family, those four contexts are attribute 6, 1, 3, 5; counting 2, 3, 0, 8; spatial 2, 1, 0, 0. Three scenes in three groups were clean-correct under both primary contexts. The v2 scenes are a different sample. Scene-by-scene comparison with those 24 is not defined.
