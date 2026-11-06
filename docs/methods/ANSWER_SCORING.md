# Answer scoring

Checkpoint 1.5 development proposal. The strict parser from Checkpoint 1 decision D4 remains the official label. The canonical-word loss from decision D1 remains stored beside the class loss defined here. Nothing in this file replaces `docs/checkpoint-1/`. Every model measurement cited from `src/scoring/records/development_mps.json` is **DEVELOPMENT ONLY** and was taken on MPS.

This file does not run an attack and does not fit H\*. A higher accuracy is not a reason to adopt a parser, a prompt, or a loss.

## Official label

D4 is unchanged. Normalization is case, whitespace, surrounding quotes, and terminal punctuation. The normalized string must be exactly one task-vocabulary item or a documented alias. Attribute and spatial answers use `red`, `blue`, `green`, and `yellow`. Counting uses `one` through `four`, and the digits `1` through `4` canonicalize to those words. A recognized color or count that misses the scene label is valid-but-wrong. Sentences, option lists, multiple labels, and refusals are invalid. Invalid output is a task error.

`src/vlm_iface/vlm_iface/parsing.py` is the implementation. This checkpoint does not edit it. Re-parsing the 96 stored `raw_output` strings reproduces the stored parse, the stored reasons, and the stored correct / valid-but-wrong / invalid split.

## Embedded single-label reading

For a string whose strict reason is exactly `not_exact` and which contains one vocabulary item, the scoring record stores that item as `alternative_parsed_answer`. The field `alternative_selected` is false. The official correct / valid-but-wrong / invalid flags stay on the strict parse. Adopting the embedded reading would reassign the 20 invalid rows already stored in Checkpoint 1.

## Uniform format line

Checkpoint 1 decision D2 builds the user message as the context text, a newline, and the question. The format probe appends one fixed line after that message:

```text
Reply using only the answer. Do not write a sentence.
```

The line is the same string for every context, every task family, and every scene. It is not interpolated. It contains none of the color names, count words, or digits under the repository's leakage normalization. The original D2 message is the prefix, so the Checkpoint 1 prompts stay recoverable. The line is a development probe. It is not the stored Checkpoint 1 prompt.

## Closed vocabulary and unconstrained generation

Unconstrained generation is greedy decoding with `max_new_tokens=16`, `do_sample=False`, and the strict parser. Closed-vocabulary scoring ranks the four canonical classes by a loss and does not decode.

Two losses are stored for every row:

- Word loss: the sum negative log-likelihood of the leading-space canonical word. On this tokenizer that word is one token, so the sum equals the token negative log-likelihood. This is the D1 surface.
- Class loss: the negative log-sum-exp of the log-probabilities of the surfaces below.

The closed-vocabulary decision for a loss is the class with the lowest loss. Ties break by the class name, which is the Checkpoint 1 tie break. A scoring/generation disagreement is recorded only when the strict parse is valid and that class's loss exceeds the best loss by more than \(10^{-6}\). An invalid generation is not counted as this disagreement. Both definitions are the Checkpoint 1 rule from `pilot.rows.scoring_disagreement`, applied once to each loss.

## Class loss

For a canonical class \(y\) and a practical surface set \(S(y)\):

\[
L(y) = -\log \sum_{s \in S(y)} P(s \mid x, c)
\]

\(P(s \mid x, c)\) is the joint probability of the token sequence of \(s\). A one-token surface is read from the log-softmax at the first answer position. The leading-space digit is two tokens on this revision, a space piece followed by the digit piece. Its joint probability is the first-position probability of the space piece times the next-position probability of the digit. That sum is the D1 sum of token log-probabilities. On `dev-counting-00-a` under `neutral`, it matches `VLMSession.score_details` for the digit `2` within \(5.42 \times 10^{-5}\).

\(S(y)\) contains a spelling when decoding its tokens parses back to \(y\) under D4. The spellings that are considered are fixed in advance:

- the canonical word, its capitalized form, and its uppercase form
- each of those with and without the leading-space convention
- for counting, the documented digit, with and without the leading-space convention

A multi-token case variant is skipped and listed in `skipped_surfaces`. On revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467` the skipped spellings are leading and bare `BLUE`, `GREEN`, `YELLOW`, `TWO`, `THREE`, and `FOUR`. `RED` and `ONE` are one token and are included. The leading-space canonical word is required to be one token. The leading-space digit is required to parse back to its count word and is kept as two tokens.

The image gradient of \(L(y)\) was not computed. The unit tests differentiate \(L\) with respect to the surface log-probabilities: raising an accepted surface's log-probability lowers \(L\), and, inside one softmax, raising the digit logit raises the canonical-word loss. Those tests are CPU algebra on constructed logits. They are not MPS measurements.

## What stays beside the proposal

| Record | Role |
| --- | --- |
| Strict parse, including invalid versus valid-but-wrong | Official label |
| Embedded single label | Stored, not selected |
| Word loss and its argmin | D1 surface, stored for every row |
| Class loss and its argmin | Proposed differentiated loss |
| Checkpoint 1 `docs/checkpoint-1/` files | Unchanged |

## Development measurements

The MPS run and its tables are in `docs/checkpoint-1.5/AGENT_A_REPORT.md`. Machine-readable rows are `src/scoring/records/development_mps.json` and `src/scoring/records/development_mps.jsonl`. The runner refuses a device other than MPS, refuses to write under `docs/checkpoint-1/`, and stops before loading weights when available memory is below the Checkpoint 0 guard of 6 GiB.
