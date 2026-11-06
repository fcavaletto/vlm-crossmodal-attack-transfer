# Prospective choice scoring

**Status: prospective.** This protocol does not correct Checkpoint 1.6 and does not replace its official scores. Checkpoint 1.6 remains the strict free-generation parser on the rows already stored. Nothing here is a development-set accuracy, an attack, a fit of H\*, or a reason to adopt a rule because a rate moved.

The primary score is the teacher-forced conditional log-likelihood of one full color sequence. The reduction is the sum. The closed-vocabulary decision is the argmin of that sum. The scalar a later image attack differentiates is that same sum for the labeled color.

## Decision

Four classes: `red`, `blue`, `green`, `yellow`. One candidate string per class. No count words. No extra class for a capitalized spelling.

For image \(x\), shared prompt text, and candidate token sequence \(y^{(c)}\):

\[
\log P(y^{(c)} \mid x, \text{prompt}) = \sum_t \log P(y^{(c)}_t \mid y^{(c)}_{<t}, x, \text{prompt})
\]

\[
L(c) = -\log P(y^{(c)} \mid x, \text{prompt})
\]

\(L(c)\) is the sum of the token negative log-likelihoods. The decision is

\[
\hat c = \operatorname{argmin}_c L(c)
\]

Equal sums break by color name, so `blue` precedes `green`, `red`, and `yellow`. The name order is not the order the prompt lists the colors, and it is not dict insertion order. A gap of any finite nonzero size is not a tie.

The attack loss for labeled color \(y\) is \(L(y)\). It is not \(L(\hat c)\), not the mean token NLL, and not a four-way softmax. Minimizing \(L(y)\) differentiates the quantity the argmin compares. A softmax over the four sums would add a gradient that suppresses the other colors. That is a different objective. It is not selected.

The mean token NLL, \(L(c) / T(c)\), is stored and is not the decision. For one fixed candidate, \(T\) does not depend on the image, so the mean and the sum have the same image-gradient sign. Across candidates with different \(T\), the argmin can change. On both pinned tokenizers below, every selected color is one token, so the two reductions currently rank the same. That measurement is not the reason the sum is primary. The sum is primary because it is \(-\log P\) of the whole candidate. The implementation still adds every token, so a later multi-token color is not cut down to its first piece.

## Why this is the image-attack measurement

An image attack changes pixels and leaves the text fixed. Transfer, from one context to another or from SmolVLM2 to Qwen3-VL-2B, is a change in the decision on that image. The decision has to be the argmin of the scalar the attack differentiates. A greedy string passed through the strict parser is a different functional. So is a first-token mixture, a mean taken only when it changes the ranking, and the Checkpoint 1.6 class loss over every spelling the parser accepts. Those can move when \(L(y)\) does not, and the reverse.

\(L(c)\) depends on the image because every answer-token logit is conditioned on the vision tokens. It is defined for both models on every image, including rows whose greedy text is an invalid sentence. The four candidate strings are the same words. Each model uses its own tokenizer, so the token ids are not shared and the numeric values of \(L\) are not a common unit. The comparable outcome is the color. The text, including the context, the question, the instruction, and the answer prefix, is held fixed, so the image is the only channel an image perturbation changes.

Free-generation strict accuracy, the invalid rate, and the split between correct and valid-but-wrong stay beside that decision. They are not added into it. Invalid remains a property of the decoded string. Closed-vocabulary scoring has no invalid bucket: a row that tokenizes has a color.

This checkpoint does not build a perturbation. It fixes the decision a perturbation would be scored against.

## Prompt

Context and question are separate fields. The rendered user text is three lines, in this order, for every context:

1. The context, unchanged, including when it already names a color.
2. The question, unchanged.
3. This fixed instruction: `Reply with one of these colors: red, blue, green, yellow.`

The instruction lists each color once. It is not a format string. It does not take the scene label. It is the same line under `neutral`, `helpful_nonrevealing`, `contradictory`, and `answer_revealing`. The join does not add `Context:` or `Question:` labels. A caller that only has a prejoined string is not the builder; `parse_user_text` accepts only this three-line form.

The instruction contains every answer word, on purpose, identically for every scene. It is not scene leakage. The question and the context are not rewritten to insert it.

## Answer prefix and tokenization

The answer prefix is the chat template's generation cue after the user text. It does not depend on the color or the context. The same rule is used for both models:

- If the cue already ends in whitespace, the continuation is the bare color word.
- Otherwise the continuation is one space plus the color word.

That space is not chosen by which form has higher probability. No probability was measured. The cue is a property of the template.

Both forms are encoded for every color. A selected sequence is kept when it is more than one token. Two colors may not share a selected id sequence. Decoding the ids must be that color under the unchanged strict parser, after a leading `Ġ` or `▁` is read as a space. Special tokens are not added, and a candidate that contains a bos, eos, pad, or unk id is rejected. Encoding must repeat.

The ids that are scored are the continuation's own ids, appended after the template's prompt ids. That is the same concatenation the existing session uses. It is checked against tokenizing the rendered prompt and the continuation as one string. It is not replaced when those two disagree; a disagreement is reported. On the check below, they agreed.

Pins, unchanged from the recorded checkpoints:

| Key | Model | Revision |
| --- | --- | --- |
| `smolvlm2` | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |
| `qwen3-vl-2b` | `Qwen/Qwen3-VL-2B-Instruct` | `89644892e4d85e24eaac8bacfd4f463576704203` |

Tokenizer files only were read. No weight file was downloaded and no forward was run. The render used one fixed neutral sentence and one fixed row-color question, plus the instruction above. An 8×8 placeholder was used only to ask the template for an image turn. It was not scored.

SmolVLM2 generation cue, the same string for the image turn and the text turn:

```text
<end_of_utterance>
Assistant:
```

The cue does not end in whitespace. The selected continuation has a leading space. Each selected color is one token. `Ġred` is id 2382, which is the id already recorded for this revision in `docs/models/MODEL_COMPATIBILITY.md`.

| Color | Selected text | Id | Piece | Bare id |
| --- | --- | ---: | --- | ---: |
| red | ` red` | 2382 | `Ġred` | 455 |
| blue | ` blue` | 4461 | `Ġblue` | 16591 |
| green | ` green` | 2654 | `Ġgreen` | 9298 |
| yellow | ` yellow` | 5724 | `Ġyellow` | 23429 |

Leading-space ids differ from bare ids for all four colors. Appending the selected id to the rendered prompt matches tokenizing the rendered prompt plus the continuation. The template has no assistant-content branch: passing an assistant message does not insert the color, so that path is not the scoring check.

Qwen3-VL-2B generation cue, again the same for the image turn and the text turn:

```text
<|im_end|>
<|im_start|>assistant
```

The cue ends in a newline. The selected continuation is the bare word. Each selected color is one token.

| Color | Selected text | Id | Piece | Leading-space id |
| --- | --- | --- | ---: | ---: |
| red | `red` | 1151 | `red` | 2518 (`Ġred`) |
| blue | `blue` | 12203 | `blue` | 6303 (`Ġblue`) |
| green | `green` | 13250 | `green` | 6176 (`Ġgreen`) |
| yellow | `yellow` | 27869 | `yellow` | 13753 (`Ġyellow`) |

Leading-space ids differ for all four. The rendered-string check matches. An assistant turn tokenizes as the bare color id, then `<|im_end|>` (id 151645) and a newline (id 198). Those two ids are outside the candidate. They are not the color, and their conditional probability depends on the color that preceded them, so folding them into \(L(c)\) would be a different decision.

Logits are aligned with the causal shift. Row `prompt_length - 1` predicts the first answer token. Later answer tokens use the following rows. The final logit, which predicts a token past the candidate, is not in the sum. A multi-token candidate is the sum of those conditional terms, not the first term alone.

Half-precision and bfloat16 logits are promoted inside `log_softmax`. The loss stays on the graph, so a later image tensor can receive the gradient. This package does not build that image tensor.

## Free generation

`vlm_iface.parsing.parse_generation` is not edited. It has no `row_color` family. Attribute and spatial already use these four color names, so a greedy string is parsed with `task_family="attribute"`. That call selects the vocabulary. It does not relabel the scene.

Correct, valid-but-wrong, and invalid are a partition of greedy strings. A sentence that contains one color stays invalid. That embedded reading is not adopted. Strict accuracy is the correct count over every row. The invalid rate uses the same denominator. Valid-but-wrong is the rest of the partition. Closed-vocabulary accuracy is a different count on the same rows. The summary has no blended rate.

## Not selected

- Mean NLL as the decision or the attack loss.
- The Checkpoint 1.6 mixture over case variants and other parser-accepted spellings. That mixture is a different argmin. One lowercase sequence is the string the instruction asks for.
- Dropping a multi-token color. None of these eight forms split. The code does not depend on that.
- Choosing the leading-space form because it scored higher. No score was computed.
- Including the Qwen end-of-turn tokens in \(L(c)\).
- Treating a higher agreement with the parser as a reason to change the loss.

## What this protocol does not authorize

No attack, no H\* fit, no held-out scene, no training, no cloud GPU, and no load of either weight file. Checkpoint 1.6 files are unchanged. A finite gradient of \(L(y)\) with respect to an image is a later measurement. The scalar it should differentiate is the one defined here.
