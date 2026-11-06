# Agent and responsibility

Agent C, Checkpoint 1.6, answer scoring. Research lead: Federico Cavaletto.

Branch `checkpoint-1.6/scoring`, started at `b2e7abff72b87b475d0abf8d037e1b4d2f9c0c06`. The external reviewer had already marked the v2 benchmark as failing its clean-competence gate. Attacks were not authorized. This checkpoint is model and task selection only, and the question here is whether closed-vocabulary class scoring should be the primary outcome.

Owned paths: `docs/checkpoint-1.6/AGENT_C_SCORING.md`, `src/class_score/`, and `tests/class_score/`. `src/scoring` is unchanged. No file under `docs/checkpoint-1/`, `docs/checkpoint-1.5/development_v2/`, `fixtures/`, `tests/scene_gen/`, `tests/scene_v2/`, or `tests/analysis/` was edited. No push, pull request, or merge.

No PGD, FGSM, noise, or other attack was run. H\* was not fit. No held-out scene was generated. Nothing was trained. No cloud GPU was used. No VLM weights were loaded. The numerical gradient below is a finite difference of a class loss on constructed CPU logits.

# Executive summary

The clean-competence band was fixed before these rows: strict accuracy from 0.60 to 0.80 in both primary contexts. Closed-vocabulary class scoring should not replace that outcome. On the 288 committed v2 predictions, SmolVLM2 on MPS and labeled DEVELOPMENT ONLY, strict accuracy is 26/72 under neutral and 22/72 under helpful_nonrevealing. The class-loss argmin is correct on 30/72 and 26/72. The canonical-word argmin is correct on 29/72 and 22/72. No family is inside the band on both primary contexts under any of the three readouts, and both primary totals are below one half. Class scoring does not rescue clean accuracy.

The official primary metric is the Checkpoint 1 strict parser of greedy text. The class loss is the negative log of the summed probability of the surfaces that parser already accepts. That scalar is the differentiable class loss. Its argmin is stored beside the parse. Ties break by class name. A valid parse disagrees only when its class is worse by more than 1e-6: 230 of 234 valid rows agree, and the 4 disagreements are spatial. Invalid generations are not disagreements. The embedded-sentence reading is unselected. Using it would move the primary rates to 30/72 and 32/72, still outside the band. Free generation is official. The class argmin is a stored readout.

# Work completed

- Read `docs/methods/ANSWER_SCORING.md`, the Checkpoint 1 parser in `src/vlm_iface/vlm_iface/parsing.py`, and the pre-specified band in `docs/checkpoint-1.5/VALIDATION_PROTOCOL.md`. The band is the closed interval from 0.60 to 0.80 in both `neutral` and `helpful_nonrevealing`.
- Read every row of `docs/checkpoint-1.5/development_v2/predictions.jsonl`. The file has 288 rows. None were deleted. Each row was re-parsed with the Checkpoint 1 strict parser. The stored parse, reasons, and correct / valid-wrong / invalid partition match on 288/288 rows. The stored class argmin and word argmin match the stored losses, including the name tie break. The stored disagreement flags match the 1e-6 rule.
- Implemented that protocol in `src/class_score`. The package builds the practical surface set, evaluates `L(y)` on CPU logits, checks a central difference against autodiff, and audits the committed predictions. It does not import a weight loader and it does not call `image_loss_gradient`.
- Recorded the unselected sensitivities: digit surfaces removed on constructed logits, and the embedded-sentence counts on the v2 file.

# Key findings

Class scoring is not a rescue. The primary-context totals stay below one half: strict 26/72 and 22/72, class argmin 30/72 and 26/72, word argmin 29/72 and 22/72. Across all four contexts the class argmin and the word argmin are each correct on 128/288 rows, against 110/288 strict. The two 128s are different rows. A higher total is a description of this sample. The protocol below was not chosen by comparing these totals.

The gain from strict to class argmin is 18 rows, and one primary cell moves the other way. Spatial under `helpful_nonrevealing` is 12/24 strict and 11/24 class argmin, because `dev2-spatial-06-b` parses as the true color `red` while the class argmin is `yellow` (gap 0.148). Attribute under the helpful sentence moves from 0/24 to 5/24. Those 5 are invalid sentences whose first-position class loss prefers the true color. 5/24 is still below one half. Counting under `contradictory` is 0/24 for all three readouts.

On the 94 counting generations whose text is `1`, `2`, `3`, or `4`, the class argmin equals the parsed class on 94/94. Class-argmin accuracy on those rows is the same number as strict accuracy. The digit surface makes the closed-vocabulary decision agree with the parser. It does not create a new set of correct scenes. The two extra class-correct counting rows under `neutral` are the invalid generations `0` on `dev2-counting-01-a` and `dev2-counting-03-a`.

No simpler attribute task is recommended. Attribute and spatial answers stay the four color names. The helpful-context attribute failure is a sentence under that parser, and a smaller color set would still be a sentence. Counting stays `one` through `four`, with digit aliases `1` through `4`.

# Evidence

Source: `docs/checkpoint-1.5/development_v2/predictions.jsonl`. 288 rows, 72 scenes, 24 in each family, four contexts. Model `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`, device `mps`, `format_line_appended` false. **DEVELOPMENT ONLY.** Denominators are every row in the cell.

Strict partition, correct / valid-wrong / invalid:

| Family | neutral | helpful_nonrevealing | contradictory | answer_revealing |
| --- | --- | --- | --- | --- |
| attribute | 10 / 9 / 5 | 0 / 2 / 22 | 10 / 8 / 6 | 20 / 2 / 2 |
| counting | 10 / 12 / 2 | 10 / 14 / 0 | 0 / 24 / 0 | 23 / 1 / 0 |
| spatial | 6 / 18 / 0 | 12 / 12 / 0 | 7 / 9 / 8 | 2 / 13 / 9 |
| all | 26 / 39 / 7 | 22 / 28 / 22 | 17 / 41 / 14 | 45 / 16 / 11 |

Decision accuracy, strict / class argmin / word argmin. A class or word count is the number of rows whose argmin equals the true answer:

| Family | neutral | helpful_nonrevealing | contradictory | answer_revealing |
| --- | --- | --- | --- | --- |
| attribute | 10 / 11 / 13 | 0 / 5 / 6 | 10 / 11 / 12 | 20 / 22 / 24 |
| counting | 10 / 12 / 8 | 10 / 10 / 4 | 0 / 0 / 0 | 23 / 23 / 24 |
| spatial | 6 / 7 / 8 | 12 / 11 / 12 | 7 / 11 / 12 | 2 / 5 / 5 |
| all | 26 / 30 / 29 | 22 / 26 / 22 | 17 / 22 / 24 | 45 / 50 / 53 |

Primary rates against the band 0.60–0.80. "Outside" means the rate is not in that closed interval. A rate below one half is marked as well. `answer_revealing` is not a primary context.

| Family | neutral strict | neutral class | neutral word | helpful strict | helpful class | helpful word |
| --- | --- | --- | --- | --- | --- | --- |
| attribute | 10/24 outside, below half | 11/24 outside, below half | 13/24 outside | 0/24 outside, below half | 5/24 outside, below half | 6/24 outside, below half |
| counting | 10/24 outside, below half | 12/24 outside | 8/24 outside, below half | 10/24 outside, below half | 10/24 outside, below half | 4/24 outside, below half |
| spatial | 6/24 outside, below half | 7/24 outside, below half | 8/24 outside, below half | 12/24 outside | 11/24 outside, below half | 12/24 outside |

Counting neutral class accuracy is 12/24 = 0.50. Spatial helpful strict and word accuracy are 12/24 = 0.50. Those three cells are outside the band. They are not below one half. Every family is below one half in at least one primary context under each readout. No primary family cell is inside the band.

Agreement of the argmin with the strict parse, valid rows only. Invalid rows are excluded, which is the Checkpoint 1 disagreement rule.

| Readout | Matches the parsed class | Disagreement flags | Valid rows |
| --- | ---: | ---: | ---: |
| class argmin | 230 | 4 | 234 |
| word argmin | 179 | 55 | 234 |

The four class disagreements are all spatial, all valid, and all worse by more than 1e-6:

| Scene | Context | Parsed | Class argmin | Answer | Gap |
| --- | --- | --- | --- | --- | ---: |
| `dev2-spatial-02-a` | neutral | yellow | blue | blue | 0.040187 |
| `dev2-spatial-06-b` | helpful_nonrevealing | red | yellow | red | 0.148155 |
| `dev2-spatial-01-a` | answer_revealing | red | green | yellow | 0.106376 |
| `dev2-spatial-04-b` | answer_revealing | blue | green | red | 0.127045 |

Scenes correct under both primary contexts. This is an inclusion count. It is not a fitted H\*.

| Readout | attribute | counting | spatial | all scenes | all groups |
| --- | ---: | ---: | ---: | ---: | ---: |
| strict | 0 | 8 in 6 groups | 5 in 5 groups | 13 | 11 |
| class argmin | 4 in 4 groups | 10 in 8 groups | 6 in 6 groups | 20 | 18 |
| word argmin | 3 in 3 groups | 3 in 3 groups | 6 in 6 groups | 12 | 12 |

# Design decisions and rationale

One protocol. The official primary outcome is the Checkpoint 1 strict parse of greedy text (`max_new_tokens=16`, `do_sample=False` in the run that wrote the file). Correct, valid-wrong, and invalid are a partition. Invalid is an error for that outcome. Free generation is official. It is not a secondary metric.

The class-loss argmin is stored on the same row and is not the outcome. The canonical-word loss is stored as the one-token special case. The embedded single-label reading is computed in the sensitivity and is unselected. `adopt_embedded_reading` raises.

This stack was fixed from the parser and from the pre-specified outcome. The v2 table is the warning that the class argmin does not enter the band. The same recommendation would stand if the class argmin had been higher. Promoting it after the gate failed would replace the outcome with a different functional of the model: the lowest loss among four classes at the first answer position, including rows whose greedy text is a sentence.

## Candidate answers and aliases

D4, unchanged. Attribute and spatial classes are `red`, `blue`, `green`, `yellow`. Counting classes are `one`, `two`, `three`, `four`. The digits `1`, `2`, `3`, and `4` alias those words and are not extra classes. A recognized color or count that misses the scene label is valid-but-wrong. Sentences, option lists, multiple labels, and refusals are invalid.

No simpler attribute vocabulary is selected. The current attribute question already asks for one of those four colors.

The spellings considered for `S(y)`, before tokenization, are the canonical word, its capitalized form, and its uppercase form, each with and without a leading space. Counting also considers the documented digit with and without a leading space. A spelling is kept only when decoding its tokens parses back to `y`.

## Class loss

For canonical class `y` and practical surface set `S(y)`:

\[
L(y) = -\log \sum_{s \in S(y)} P(s \mid x, c)
\]

`P(s | x, c)` is the joint probability of the token sequence of `s`. A one-token surface is one log-softmax entry at the first answer position. Duplicate sequences are counted once. The canonical-word loss is the special case `S(y) = {leading-space canonical token}`.

The image gradient of `L(y)` was not computed. The unit test differentiates `L` with respect to constructed CPU logits: a central difference with step `1e-5`, in float64, checked against autodiff. Raising a logit inside `S(y)` lowers `L`. Raising a competing token raises the canonical-word loss and, when that token is an accepted alias in the same softmax, lowers the class loss. The test does not call `image_loss_gradient` and does not load a VLM. A non-CPU tensor is refused.

## Multi-token surfaces

The leading-space digit is two tokens on the recorded SmolVLM tokenization, a space piece followed by the digit piece. It is kept. Its joint probability is the first-position probability of the space piece times the next-position probability of the digit. Any other spelling that splits into more than one token is skipped and listed, with reason `multi_token`. Checkpoint 1.5 recorded the skipped uppercase spellings on revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467` as leading and bare `BLUE`, `GREEN`, `YELLOW`, `TWO`, `THREE`, and `FOUR`, and recorded `RED` and `ONE` as one token. This checkpoint did not load the tokenizer, so that list is the earlier record. The code applies the same rule to the tokenizer it is given. The leading-space canonical word is required to be one token. A surface longer than two tokens is rejected.

## Decision

The closed-vocabulary choice is the class with the lowest loss. Equal losses break by class name, so `blue` wins a tie with `yellow`. A scoring/generation disagreement is recorded only when the strict parse is valid and that class's loss exceeds the best loss by more than `1e-6`. An invalid generation is not a disagreement. The four stored flags match this rule. There were no exact ties in the 288 rows.

## Sensitivities, unselected

Digit surfaces. The committed rows store pooled class losses, so the digit piece cannot be subtracted out of those numbers. On constructed logits, dropping `digit_alias` and `digit_alias_bare` raises `L` for the class that owned the digit and can move the argmin to the canonical-word choice. On the file, the canonical-word argmin is the larger change: every non-canonical surface is gone, not only the digit. Class and word argmins differ on 46/96 counting rows and on 21/192 attribute and spatial rows. Counting primary word accuracy is 8/24 and 4/24, against class accuracy 12/24 and 10/24. Word-argmin agreement with the 94 digit parses is 50/94, against 94/94 for the class argmin. Dropping the digit surface would separate the closed-vocabulary decision from the digit strings the official parser already accepts. It would also move counting farther from the band. It is not a second outcome.

Embedded sentences. The reading applies only when the strict reason is exactly `not_exact` and the string contains one vocabulary item. That is 50 rows. The embedded label equals the answer on 26 and a different class on 24. Two truncated attribute sentences (`dev2-attribute-06-a`, `dev2-attribute-06-b`, helpful) and two `0` generations stay outside it. If the reading were scored, accuracy would be:

| Family | neutral | helpful_nonrevealing | contradictory | answer_revealing |
| --- | ---: | ---: | ---: | ---: |
| attribute | 14/24 | 10/24 | 10/24 | 22/24 |
| counting | 10/24 | 10/24 | 0/24 | 23/24 |
| spatial | 6/24 | 12/24 | 13/24 | 6/24 |
| all | 30/72 | 32/72 | 23/72 | 51/72 |

The primary totals would be 30/72 and 32/72. Attribute helpful would be 10/24. No family would enter the band on both primary contexts. The reading stays unselected. Adopting it from this file would spend a parser change on sentences that have already been read.

# Known limitations

The sample is 72 development scenes. Rates with denominator 24 or 72 describe this generator and this checkpoint. They are not a held-out result.

`L(y)` is the probability of the practical surfaces at the answer position. It is not the probability of the greedy sentence. A row can be an invalid sentence and still have its class argmin on the true color. Terminal punctuation is stripped by the parser and is not a surface in `L(y)`. Uppercase spellings that split are omitted.

The stored class loss cannot be split into digit mass and case mass. The word-argmin comparison removes every extra surface.

The finite-difference test is CPU algebra on constructed logits. It is not an MPS measurement and it is not an image gradient.

Scenes correct under both primary contexts are a count. No coefficient was fit. The eligible-group count from Checkpoint 1.5 stays whatever that packet recorded. This file does not recompute gradients.

# Blockers

None for this recommendation. The instrument is outside the pre-specified band under the official strict metric and under the class argmin. That is the result of the gate, not a missing measurement. An image gradient of `L(y)` would not change the 30/72 and 26/72 counts, which are decisions on the stored losses.

Attacks remain unauthorized.

# Decisions requested from the research lead

1. Keep the Checkpoint 1 strict parser of greedy text as the official primary outcome, including the sentence rule. Recommendation: yes. Free generation stays official.
2. Keep the class-loss argmin as a stored readout, and do not make it the primary outcome. Recommendation: yes. The v2 rates 30/72 and 26/72 are outside 0.60–0.80. They are not a reason to change the outcome, and a higher rate would not have been a reason either.
3. Leave the embedded single-label reading unselected. Recommendation: yes. The unselected primary rates would be 30/72 and 32/72.
4. Do not introduce a simpler attribute vocabulary from this file. Recommendation: yes. Attribute and spatial stay on `red`, `blue`, `green`, `yellow`. Counting stays on `one` through `four`, with digits `1` through `4`.
5. When a future run is authorized to differentiate an answer loss, name `L(y)` or the canonical-word loss before that run. This checkpoint does not recompute the v2 image gradients. The 230/234 agreement is a description, not the reason to pick the scalar. Recommendation: treat the definition above as the class scalar, and leave the already stored v2 gradients as canonical-word gradients.
6. Attacks stay unauthorized. Recommendation: yes.

# Reproduction

From the worktree root. The command does not load weights.

```bash
PYTHONPATH=src/class_score:src/scoring:src/vlm_iface:src/pilot:src/analysis \
/Users/fedecava/Development/_c0-work/venv-vlm/bin/python -m pytest tests/class_score -q
```

Result of that command in this checkpoint: `17 passed`.

The audit JSON, including the tables above, is:

```bash
PYTHONPATH=src/class_score:src/scoring:src/vlm_iface:src/pilot:src/analysis \
/Users/fedecava/Development/_c0-work/venv-vlm/bin/python -m class_score
```

Interpreter: `/Users/fedecava/Development/_c0-work/venv-vlm/bin/python`.

# Git

- Branch: `checkpoint-1.6/scoring`
- Parent: `b2e7abff72b87b475d0abf8d037e1b4d2f9c0c06`
- Commit: `COMMIT_SHA_PENDING`
- Not pushed. No pull request.
