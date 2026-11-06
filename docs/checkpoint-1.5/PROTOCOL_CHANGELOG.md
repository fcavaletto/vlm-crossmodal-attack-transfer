# Protocol changelog

Checkpoint 1.5 integration. Every revision below was specified before any model call on `fixtures/development_v2`. The v2 counts in `REVIEW_PACKET.md` were not used to choose them. No row was deleted. H\* was not fit. `docs/checkpoint-1/` was not edited.

## 1. Development sample

- Development observation: on the Checkpoint 1 graphs, object count identifies the task family, and only 3 groups are clean-correct under both primary contexts.
- Original decision: 24 development scenes, with 4, 8, and 5 objects by family.
- Revised decision: keep those 24 files unchanged. Score all 72 scenes in `fixtures/development_v2` as the development instrument for this checkpoint.
- Scientific justification: a picture that names the task is a bad instrument for a later contrast. Agent B wrote this before any model result on the 72.
- Estimand: the H\* statement is unchanged. The scene distribution a later fit would use is not the Checkpoint 1 distribution.
- Specified before the v2 numbers: yes. `docs/data/DATASET_V2_DESIGN.md` and Agent B's report at `28c50d320fb780fa7a7450abd7bdba6d18b443a7`.

## 2. Spatial question and geometry

- Development observation: Checkpoint 1 spatial accuracy was at or below 2/8 in every context. On those graphs the nearest-left object was the graded answer and a large column-mate was 1.80 times as far.
- Original decision: one relation, "immediately left," with a required column-mate of a different color.
- Revised decision: the question says "in the same row," the neighbor's column contains only the neighbor, and the next left-half object is at least twice as far.
- Scientific justification: the column-mate made two readings of an underspecified question disagree. The item is unambiguous when those readings agree. Geometry does not repair a failure to copy a revealing sentence.
- Estimand: a spatial trial is the same-row neighbor the oracle already defined. The items are new. H\* is unchanged.
- Specified before the v2 numbers: yes, in the same Agent B documents.

## 3. Counting support

- Development observation: contradictory counting on the Checkpoint 1 sample was the foil word on 8/8 scenes, and that sentence said four.
- Original decision: true answers only `two` and `three`, foil always `four`.
- Revised decision: true counts `one` through `four`, four pairs of each adjacent transition, and a foil that is wrong for both arms. The 2↔3 transition stays.
- Scientific justification: one foil word cannot be separated from a model that emits that word. Dropping 2↔3 because the Checkpoint 1 score was low is rejected.
- Estimand: the counting label set is wider. H\* is unchanged.
- Specified before the v2 numbers: yes, in the same Agent B documents.

## 4. Prompt

- Development observation: on the stored 24 scenes, one format line moved strict accuracy from 31/96 to 39/96. Contradictory counting stayed 0/8 and spatial stayed at or below 3/8.
- Original decision: D2, `context_text + "\n" + question`.
- Revised decision: unchanged. The format line is not appended.
- Scientific justification: Agent A says the accuracy movement is not a reason to adopt the line. Adopting it would change the text inside H\*.
- Estimand: unchanged.
- Specified before the v2 numbers: yes. Agent A's report at `7d6094855ade2ed36eabde6daf46da333603858b`, and `VALIDATION_PROTOCOL.md` commit `e8bf93ea1b49e6a2c721431b82c1d132ca9a95bc`.

## 5. Official parser

- Development observation: the strict parser reproduces the stored 96 labels. Twenty invalid strings are sentences. An embedded reading of those sentences is not the official parse.
- Original decision: D4 exact vocabulary, including the sentence rule.
- Revised decision: unchanged. The embedded reading is not selected.
- Scientific justification: Agent A recommends keeping D4. Selecting a parser because accuracy rose is refused.
- Estimand: the strict-error endpoint is unchanged.
- Specified before the v2 numbers: yes, in the same Agent A report and the protocol commit above.

## 6. Class loss and the differentiated gradient

- Development observation: digits and some case variants that D4 accepts are not the canonical-word surface. On the stored prompts the class-loss argmin matched the strict parse on 14 of the 15 word-loss disagreements.
- Original decision: D1 stores the canonical-word loss, and the Checkpoint 1 pilot recommended keeping the gradient on that word.
- Revised decision: store the class-loss argmin beside the strict parse. Do not differentiate the class loss in this run. The image gradient stays the canonical-word loss.
- Scientific justification: Agent A recommends differentiating the class loss later because the official evaluator already accepts those surfaces. That recommendation would change the gradient inside \(A(s, t)\). The validation protocol keeps the Checkpoint 1 gate, which differentiates the canonical word, and treats the 14-of-15 count as a description rather than a selection rule.
- Estimand: unchanged in this run. Adopting the class-loss gradient later would change \(A(s, t)\) and would not change the strict-error endpoint while D4 remains the label.
- Specified before the v2 numbers: yes, in the same Agent A report and the protocol commit above.

## 7. Study model

- Development observation: the Checkpoint 1 MPS pilot is the task evidence for SmolVLM2. Qwen3-VL-2B-Instruct was not loaded. Both float32 and BF16 failed the pre-load memory budget.
- Original decision: D5, SmolVLM2-500M at revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`, MPS, float32, eager, image splitting off.
- Revised decision: unchanged. Qwen stays unloaded.
- Scientific justification: Agent C proposes no protocol revision. The task floors are a scene and scoring question.
- Estimand: unchanged.
- Specified before the v2 numbers: yes. Agent C's report at `b27cfff03c029e0d0f19052793a09cf645678dd2`.

## 8. H\* procedure

- Development observation: the Checkpoint 1 inclusion count is 3 groups, below the algebraic minimum of 5 for a three-family design. No coefficient was fit.
- Original decision: H\* is the provisional primary hypothesis and was not implemented as a cluster estimator in the pilot.
- Revised decision: the estimator in `docs/statistics/H_STAR_ANALYSIS.md` is the proposed primary analysis. This checkpoint does not fit it to real scenes. An interval is returned only at 5 or more groups for the three-family design. Twenty-four is the fabricated coverage description, not an inclusion rule and not a promise that an interval excludes zero.
- Scientific justification: Agent D's rules were fixed on fabricated rows. Using the v2 scenes to choose the interval would spend the procedure on this sample.
- Estimand: the hypothesis statement is unchanged. The interval procedure is now specified. It was not applied to these scenes.
- Specified before the v2 numbers: yes. Agent D's report at `be8be387fe93fd071d31543273baacfa476cb61c`.
