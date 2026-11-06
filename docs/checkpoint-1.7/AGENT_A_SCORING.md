# Agent and responsibility

Agent A, Checkpoint 1.7, choice scoring. Research lead: Federico Cavaletto.

Branch `checkpoint-1.7/choice-scoring`, started at `004783b5604c34aa066befbcd5356bfd7bd6e7b2` (PR #12, "Name the Checkpoint 1.6 pull request in the review packet"). The branch does not start at `main`. A pull request into `main` therefore also contains Checkpoints 0 through 1.6. Those files are not this change. Checkpoint 1.6 was not edited. Its official scores were not replaced.

Owned paths: `src/choice_eval/**`, `tests/choice_eval/**`, `docs/checkpoint-1.7/AGENT_A_SCORING.md`, `docs/checkpoint-1.7/SCORING_PROPOSAL.md`.

No attack was run. H\* was not fit. No held-out scene was generated. Nothing was trained. No cloud GPU was used. No VLM weights were loaded. The development scenes were not scored. The protocol is prospective.

# Executive summary

The primary score is the summed teacher-forced negative log-likelihood of one full color sequence. The colors are `red`, `blue`, `green`, and `yellow`. The decision is the lowest sum, then the color name. The loss a later image attack would differentiate is that same sum for the labeled color. Free-generation strict accuracy, the invalid rate, and correct versus valid-but-wrong stay a separate diagnostic. The written proposal is `docs/checkpoint-1.7/SCORING_PROPOSAL.md`.

The rule was not chosen because it raises accuracy. No accuracy was computed.

# Work performed

- Read `docs/methods/ANSWER_SCORING.md`, `docs/checkpoint-1.6/EVAL_PROTOCOL.md`, `docs/checkpoint-1.6/AGENT_C_SCORING.md`, `docs/models/MODEL_COMPATIBILITY.md`, and the research proposal's definition of \(L_c(x)\) as \(-\log P(y \mid x, c)\).
- Added `choice_eval`. It builds the shared prompt, selects a continuation from the generation cue, verifies tokenization, sums the aligned token NLL, and keeps the strict parser as a secondary diagnostic. It does not call `from_pretrained` on a model and it does not step an image.
- Checked the pinned SmolVLM2 and Qwen3-VL-2B tokenizers. The download allow-list is tokenizer and config files. A weight suffix, or a file over 40 MB, aborts the snapshot. `from_pretrained` was used only for `AutoTokenizer` on that local snapshot.

# Design decisions

The proposal states the reasons. The short form:

- One sequence per color, not the 1.6 surface mixture. The mixture is a different argmin.
- Sum, not mean. For one candidate the gradient sign matches. Across candidates with different lengths the argmin may not. Both pinned tokenizers give every selected color one token, so the argmins match on this vocabulary. The sum stays primary anyway.
- Leading space only when the generation cue does not already end in whitespace. The same function serves both models. It is not fit to a probability.
- Context and question stay separate fields. The instruction that names the four colors is a third fixed line. It does not change with the context, including when the context already names a color.
- The attack loss is \(L(y)\) for the labeled color. It is not the loss of whichever color won, and it is not a softmax over the four sums.
- Ties break by color name. `blue` beats `red` on an equal sum. A difference of `1e-9` is not a tie.
- The strict parser is called with the attribute vocabulary because that vocabulary is the four colors. `parsing.py` was not edited. Sentences that contain one color stay invalid.

# Tests

From the repository root. This command does not download a checkpoint and does not load weights. The interpreter needs the dependencies `vlm_iface` already imports (`torch`, `numpy`, `Pillow`, `psutil`) and `pytest`.

```bash
PYTHONPATH=src/choice_eval:src/vlm_iface python3 -m pytest tests/choice_eval -q
```

Result in this worktree: `45 passed`.

| Required check | Test |
| --- | --- |
| Logit alignment | `test_causal_logits_read_the_shifted_answer_row`, `test_summed_nll_matches_the_library_shift`, `test_backward_reaches_only_the_aligned_answer_rows` |
| Multi-token likelihood | `test_multi_token_likelihood_sums_the_conditionals`, `test_multi_token_candidates_are_kept` |
| Sum versus mean | `test_sum_and_mean_can_choose_different_colors`, `test_equal_lengths_share_the_argmin_and_differ_in_scale` |
| Ranking | `test_rank_follows_summed_nll` |
| Deterministic ties | `test_ties_break_by_color_name_not_by_insertion_order` |
| Finite differentiable loss | `test_finite_difference_matches_autograd`, `test_bfloat16_loss_is_finite_and_differentiable` |

The alignment tests use the causal shift: answer token \(t\) is read from logit row `prompt_length - 1 + t`. Backward reaches those rows and not the unused final row. The summed NLL matches `vlm_iface.logprob.sequence_logprob` on a seeded float32 sequence. The finite-difference check uses step `1e-5` in float64 and matches autograd. A bfloat16 leaf promotes inside `log_softmax`, returns a finite float32 loss, and receives a finite gradient. Raising the labeled token's logit lowers the loss. The gradient of the labeled loss does not flow into another color's logits.

The sum-versus-mean test uses a one-token candidate and a four-token candidate whose argmins differ. The decision follows the sum. The labeled loss is the four-token sum, not the mean.

Tokenizer unit tests use a stub. They check that a leading space changes the ids, that a two-token yellow is kept, that a shared id sequence is rejected, and that a weight file in a snapshot directory is refused.

# Tokenizer results

Command, after the unit tests. It downloads tokenizer files and renders the chat template. It does not load weights. Transformers 5.22.2 required `jinja2>=3.1.6` to render the template. The unit tests do not render it.

```bash
PYTHONPATH=src/choice_eval:src/vlm_iface python3 -m choice_eval.verify_live
```

Exit code 0. `weights_loaded` is false. No safetensors file was in either snapshot.

The render used the neutral sentence "Several simple objects appear on a plain background.", the question "What color is the circle in the top row?", and the fixed instruction. Image-marker, PIL-placeholder, and text-part suffixes agreed for both models. The image placeholder was an 8×8 black image or a template image marker. It was not a development scene and it was not scored.

SmolVLM2, revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`, vocabulary 49280. Cue: `<end_of_utterance>\nAssistant:`. Selected ids: red 2382, blue 4461, green 2654, yellow 5724, each one `Ġ` piece. Bare ids differ: 455, 16591, 9298, 23429. Appending the selected id matches tokenizing the rendered prompt plus the continuation.

Qwen3-VL-2B, revision `89644892e4d85e24eaac8bacfd4f463576704203`, vocabulary 151669. Cue: `<|im_end|>\n<|im_start|>assistant\n`. Selected ids: red 1151, blue 12203, green 13250, yellow 27869, each the bare piece. Leading-space ids differ: 2518, 6303, 6176, 13753. The same append check matches. The assistant turn continues with `<|im_end|>` (151645) and a newline (198). Those ids are not in \(L(c)\).

None of the eight selected or alternate forms is multi-token. The sum is still defined for more than one token. The stub test scores a two-token yellow that a real color did not produce.

# Limitations

No image was scored. The differentiable loss is with respect to constructed logits, including one bfloat16 leaf. It is not an image gradient and not an MPS measurement.

The token ids come from `AutoTokenizer` and the chat template string. They were not checked by a model forward. A processor that expands image tokens can lengthen the prompt. On SmolVLM2 the image-marker suffix was the same string as the text suffix, and the raw encoding of the rendered string matched the template ids. That is not a forward pass.

Qwen's weight-memory refusal from Checkpoint 1.6 is unchanged. This work does not adopt the model and does not load it.

Closed-vocabulary accuracy and free-generation accuracy are not interchangeable. A row can be closed-vocabulary correct and still be an invalid sentence. The summary keeps both counts.

Numeric \(L\) is not comparable across the two models. Token ids are not comparable. The color decision is the comparable outcome.

# Unexpected findings

SmolVLM's chat template has no assistant-content branch. An assistant message does not insert the color; the tail is a space, `<end_of_utterance>`, and a newline (ids 216, 49279, 198). A user message passed as a plain string is also not rendered as that string, because the template indexes `content[0]` as a typed part. The generation cue `Assistant:` is still well defined, and appending `Ġred` matches the rendered string. Scoring uses that continuation, not the empty assistant-message path.

The string user path on Qwen does render the text, and its suffix matches the image path. The end-of-turn tokens after the color are real template tokens. They were the easy place to lengthen every candidate by the same count and still change the argmin, because their probability depends on the color. They stay outside the candidate.

Every lowercase color, with and without a leading space, is one token on both revisions. The multi-token path is tested and was not observed for these words.

# Unresolved questions

1. Should a later protocol add the end-of-turn token to the sequence? This one does not. Adding it changes \(L(c)\) by a color-dependent term.
2. Should capitalized spellings enter the sum, as in the 1.6 class loss? This one does not. That would be a different decision. Agreement with the parser is not a reason to add them.
3. The image gradient of \(L(y)\) is not measured here. When a forward is authorized, the scalar is this sum, on the continuation verified above.
4. A processor forward could still disagree with `AutoTokenizer` on the prompt ids. That check needs a loaded model and was not run.

# Reproduction

Unit tests, no download:

```bash
PYTHONPATH=src/choice_eval:src/vlm_iface python3 -m pytest tests/choice_eval -q
```

Tokenizer verification, network, no weights:

```bash
PYTHONPATH=src/choice_eval:src/vlm_iface python3 -m choice_eval.verify_live
```

# Git

- Branch: `checkpoint-1.7/choice-scoring`
- Base: `004783b5604c34aa066befbcd5356bfd7bd6e7b2`
- Scoring commit: `3d99e76ada71deba3e1a48d806271ce6c4eff057`
- Pull request: https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/pull/16
- Head: the head of that pull request. This file is on that head.
- The diff against `main` includes earlier checkpoints because this branch does not start at `main`.
