# Agent and responsibility

Agent A, answer scoring, Checkpoint 1.5 of Study 2 (`vlm-crossmodal-attack-transfer`). Research lead: Federico Cavaletto.

This workstream reads the stored Checkpoint 1 generations, keeps the strict parser, and compares closed-vocabulary scoring with unconstrained generation. It adds one format line that is identical across contexts, and a class loss whose surfaces are the spellings that parser already accepts. It does not run PGD, FGSM, noise, or any other attack. It does not fit H\*. It does not edit `docs/checkpoint-1/`, `PROTOCOL_CHANGELOG.md`, or `REVIEW_PACKET.md`.

Owned paths:

- `docs/methods/ANSWER_SCORING.md`
- `docs/checkpoint-1.5/AGENT_A_REPORT.md`
- `src/scoring/`
- `tests/scoring/`
- `VLMSession.prompt_last_logits` in `src/vlm_iface/vlm_iface/session.py`

Branch: `checkpoint-1.5/answer-scoring`, started at `90c6ffe83d61514da217a20f8dfc2f202afbbe69`.

# Executive summary

The strict parser still labels the 96 stored development generations exactly as Checkpoint 1 stored them: 31 correct, 45 valid-but-wrong, and 20 invalid. All 20 invalid strings are one-label sentences. An embedded reading of those sentences matches the answer on 10 rows and a different color on 10. That reading is recorded and is not the official parse. On MPS, SmolVLM2-500M at the pinned revision reproduces the stored canonical-word loss within \(5.44 \times 10^{-5}\). Counting strings that the parser accepts as digits are generated as a space token, a digit token, and an end token. The class loss adds that sequence, the bare digit, and the single-token case variants of the canonical word. On the stored prompts its argmin matches the strict parse on 14 of the 15 rows where the word loss disagrees, and it still disagrees on `Green` versus `blue` for `dev-attribute-00-b` under `contradictory`. The same format line on every context leaves 3 invalid rows, all the string `0.`, and moves strict accuracy from 31/96 to 39/96. That movement is a description of this sample. It is not a reason to adopt the line. No attack was run. H\* was not fit. Every model number below is MPS and DEVELOPMENT ONLY.

# Work completed

- Read `internal/checkpoint-1.5-spec.md` and `docs/checkpoint-1/PILOT_REPORT.md` at `90c6ffe83d61514da217a20f8dfc2f202afbbe69` before editing.
- Re-parsed the committed `docs/checkpoint-1/predictions.jsonl` with the unchanged strict parser and stored an embedded single-label reading beside it.
- Defined the class loss in `src/scoring/answer_scoring/` and the uniform format line. The line does not contain a task-vocabulary alias.
- Added `VLMSession.prompt_last_logits` so one forward scores every single-token surface. A second forward scores the shared space-token prefix of the two-token digits. No image gradient is taken there.
- Ran the 24 committed development scenes on MPS, once under the stored prompts and once under the format line. The runner loads one pinned checkpoint, refuses any device other than MPS, and refuses to write `docs/checkpoint-1/`.
- Did not generate held-out scenes, did not load a second model, and did not fit a coefficient.

# Key findings

Findings about the stored file do not depend on a new model call. Findings about losses and the format line are MPS measurements on the author's Mac, DEVELOPMENT ONLY.

The stored file has 96 rows. Re-parsing `raw_output` with `truncated=false` matches `parsed_answer`, `invalid`, and `parse_reasons` on all 96. The official split remains 31 correct, 45 valid-but-wrong, and 20 invalid. All 20 invalid reasons are `not_exact`. Each of those strings contains one vocabulary item. That item equals the scene answer on 10 rows (9 attribute, 1 spatial) and a different color on 10. Counting contributes none of the 20. The 15 stored `scoring_generation_disagreement` flags are 13 digit strings and 2 case-variant color strings, `Green` and `Blue`.

Fresh greedy generations of `dev-attribute-00-a` and `dev-counting-00-a`, four contexts each, match the stored `raw_output` on 8 of 8 strings. The counting strings are token ids `[216, digit, 49279]`: the space piece, the digit piece, and the end piece. The canonical-word negative log-likelihood, read from the first answer position, matches the stored `sum_nll` on all 96 rows. The maximum absolute gap is \(5.44 \times 10^{-5}\). The word-loss argmin matches `scored_argmin` on all 96 rows, and the word-loss disagreement flag matches the stored flag on all 96. On `dev-counting-00-a` under `neutral`, the two-token log-probability of digit `2` matches `score_details` within \(5.42 \times 10^{-5}\).

Under the stored prompts, the class-loss argmin differs from the word-loss argmin on 20 of 96 rows. On 14 of the 15 word-loss disagreements, the class-loss argmin equals the strict parse. Four of those 14 parses are the true answer and ten are valid-but-wrong. The class loss is agreeing with the emitted alias, including aliases that are the wrong count. One disagreement remains: `dev-attribute-00-b` under `contradictory`, raw string `Green`. The leading-space token ` Green` has log-probability \(-1.404\), which is higher than ` Blue` at \(-1.502\). The blue class still has the lower class loss, 1.133 against 1.187, because ` blue` and the other blue spellings add enough mass to pass ` Green`. No valid row gains a class-loss disagreement that the word loss did not already have. On 6 invalid sentences the class argmin and the word argmin differ. Those rows stay invalid, and the difference is not counted as a scoring/generation disagreement.

The format line was appended to all 96 scene-context prompts. Strict counts under that line, DEVELOPMENT ONLY:

| Context | Correct | Valid-but-wrong | Invalid |
| --- | ---: | ---: | ---: |
| `neutral` | 14 | 7 | 3 |
| `helpful_nonrevealing` | 6 | 18 | 0 |
| `contradictory` | 2 | 22 | 0 |
| `answer_revealing` | 17 | 7 | 0 |

The three invalid strings are `0.` on `dev-counting-01-a`, `dev-counting-02-a`, and `dev-counting-03-a`, all under `neutral`, reason `not_in_vocabulary`. No format-line output is `not_exact`. Eighty of the 96 strings end in a period; D4's existing punctuation rule accepts `Blue.` as `blue`. Attribute and spatial first tokens are single-token color surfaces on all 64 of those rows. All 32 counting first tokens are the space piece `216`, the prefix of the two-token digit. Word-loss disagreements under the format line are 14. Class-loss disagreements are 2: `dev-attribute-03-b` / `helpful_nonrevealing` (`Blue.` parsed as blue, class argmin yellow) and `dev-spatial-02-a` / `answer_revealing` (`Green.` parsed as green, class argmin red).

Family behavior under the format line, correct/invalid out of 8, DEVELOPMENT ONLY: attribute `neutral` 8/0, `helpful_nonrevealing` 3/0, `contradictory` 1/0, `answer_revealing` 8/0. Counting `neutral` 3/3, `helpful_nonrevealing` 2/0, `contradictory` 0/0, `answer_revealing` 8/0. Spatial `neutral` 3/0, `helpful_nonrevealing` 1/0, `contradictory` 1/0, `answer_revealing` 1/0. Contradictory counting remains 0/8 correct. Spatial remains at or below 3/8 in every context. The format line does not make those cells look like solved visual tasks.

# Evidence

| Claim | Where it was read |
| --- | --- |
| Stored labels, 20 invalid `not_exact` rows, 15 disagreements | `docs/checkpoint-1/predictions.jsonl`, re-parsed by `answer_scoring.audit` |
| Word-loss gap \(5.44 \times 10^{-5}\), 8/8 generation matches, class-loss counts, format-line counts | `src/scoring/records/development_mps.json` and `development_mps.jsonl` |
| Digit `2` sequence versus `score_details` | `digit_sequence_checks` in that JSON: token ids `[216, 34]`, absolute gap \(5.42 \times 10^{-5}\) |
| Surface inventory and the six skipped spellings per family | `surfaces` and `skipped_surfaces` in that JSON |
| Memory, device, revision, and runtime | The same JSON, table below |
| Logit derivatives of the class loss | `tests/scoring/test_answer_scoring.py` (constructed tensors, not MPS) |

MPS run, DEVELOPMENT ONLY. One model: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`, float32, eager attention, image splitting off, `max_new_tokens=16`, leading-space answer tokens. Interpreter `/Users/fedecava/Development/_c0-work/venv-vlm/bin/python`, Python 3.12.14, torch 2.15.0, transformers 5.22.1, Pillow 13.0.0. `HF_HOME` was `/Users/fedecava/Development/models/huggingface`. Weight load 3.661 seconds. Scoring 94.81 seconds. The process loaded no second model.

| Snapshot | Current RSS MB | Peak RSS MB | MPS current allocated MB | MPS driver allocated MB | OS available MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before load | 205.7 | 205.7 | 0.0 | 0.5 | 8256.2 |
| After load | 223.1 | 399.8 | 1965.6 | 2856.5 | 4117.3 |
| After the 192 rows | 490.8 | 490.8 | 1965.6 | 2929.8 | 5522.9 |

OS available memory before the load was 8256.2 MB, above the 6 GiB guard in `vlm_iface.pins.MIN_AVAILABLE_BYTES`. `peak_rss_mb` mixes the process with unified memory. The MPS current allocation stayed at 1965.6 MB after the load.

# Design decisions and rationale

1. **Keep D4 as the official label.** The embedded single-label reading is stored on the 20 `not_exact` rows and `alternative_selected` is false. Adopting it would move 10 rows onto the answer and 10 onto another color, using sentences Checkpoint 1 already published. The pilot asked for an explicit revision before any such move. This report does not make that revision.

2. **Class loss as the proposed differentiated scalar.** Development observation: the digit strings the strict parser accepts are the token sequence `(216, digit, end)`, and 13 stored disagreements are those digits. The two color disagreements are `Green` and `Blue`, whose leading-space case tokens are single pieces the same parser accepts after casefolding. Original decision: D1 stores the sum negative log-likelihood of the canonical answer, and the Checkpoint 1 pilot's decision 2 recommended keeping the gradient on that word while leaving digits as a parse rule. Revised decision: when a gradient of a development loss is next computed, differentiate \(L(y)\) from `docs/methods/ANSWER_SCORING.md` and keep the canonical-word loss in the same row. Do not drop the word loss. Scientific justification: a gradient of the word alone moves probability off a continuation the official evaluator already counts as that class. The practical set is those continuations the tokenizer represents as one token, plus the two-token documented digit. Estimand: H\* aligns image gradients of the loss. Changing the loss changes that gradient and therefore \(A(s, t)\). The strict-error endpoint is unchanged while D4 remains the label. No coefficient was fit, and no attack result is being used to judge the proposal. The 14-of-15 agreement count is a description of this development sample. It is not the selection criterion. The criterion is whether the surface is one D4 already accepts.

3. **Format line, measured and not adopted.** The line was fixed before the MPS run, applies to every context, and names no answer. On this sample it replaces the 20 sentences with short strings and leaves contradictory counting at 0/8 and spatial at or below 3/8. Strict accuracy goes from 31/96 to 39/96. Selecting the line because that fraction rose would spend the prompt choice on this development run. The confirmatory prompts stay the Checkpoint 1 D2 messages until the lead freezes a different sentence as a definition, in advance of the result that would be judged on it. Adding the line would change the text inside H\*, so the numerical alignment would be a different random variable even though the hypothesis statement would stay.

4. **Tie break and disagreement threshold.** Both match Checkpoint 1: lowest loss, then the class name, and a disagreement only when the parsed class is worse by more than \(10^{-6}\). Invalid rows stay out of that flag.

5. **No image gradient in this checkpoint.** `prompt_last_logits` runs under `inference_mode`. The scalar \(L(y)\) is what a later backward would differentiate. This run does not claim that the image gradient of \(L(y)\) is finite.

# Known limitations

The sample is the same 24 development scenes in 12 pairs. Rates with denominator 8 or 96 describe this generator and this checkpoint.

The class loss is not the probability of every string D4 would accept. Terminal punctuation is stripped by the parser, and 80 format-line outputs end in a period, but the period token is not a surface in \(L(y)\). Uppercase `BLUE`, `GREEN`, `YELLOW`, `TWO`, `THREE`, and `FOUR` split into more than one token and are omitted. A sum of several surfaces can outrank the single greedy token, which is the remaining `Green` / `blue` row.

The format-line table and the stored-prompt table use the same strict parser. They are not two official scores. The stored-prompt official score is the Checkpoint 1 file.

The image gradient of the class loss was not measured. Available memory after the scoring process was 5522.9 MB. A second load was not started.

No held-out scene was scored. No attack, noise baseline, or H\* coefficient was computed. CPU unit tests are not MPS numbers.

# Blockers

**Blocking.** None for this scoring checkpoint. The lead can accept or reject the two proposals below without another model run.

**Non-blocking.**

- The image gradient of \(L(y)\) is unmeasured. Checkpoint 1 already measured the word-loss image gradient. A later development check can differentiate \(L(y)\) on MPS before any attack is authorized.
- Whether the format line should be frozen is a lead decision. This run does not freeze it.
- H\* still has three development groups that are clean-correct under both primary contexts in the Checkpoint 1 labels. This workstream did not recompute that inclusion set under the format line and did not fit a coefficient.

# Decisions requested from the research lead

1. **Keep D4, including the sentence rule.** Recommendation: yes. Keep the embedded reading in the scoring record and leave `alternative_selected` false. Tradeoff: sentence answers that contain the right color stay invalid. Adopting the embedded reading would change who is correct on 20 rows that have already been seen.

2. **Differentiate the class loss, and keep the word loss beside it.** Recommendation: yes for any later development gradient. The revision is decision 2 in the design section: observation, original D1 surface, revised surface set, justification, and the change to the gradient inside \(A(s, t)\). Tradeoff: \(A(s, t)\) will not match a gradient of the canonical word alone. Keeping only the word loss leaves the digit sequence, which is the continuation this model emits, outside the differentiated scalar.

3. **Leave the D2 prompt unchanged.** Recommendation: yes at this checkpoint. The format line is implemented and its development behavior is tabulated above. Tradeoff: helpful-context attribute answers will keep arriving as sentences under D4. Freezing the format line after reading the 39/96 figure would choose a prompt on this sample. If the lead wants the line, it has to be frozen as a definition before a result that would be judged on it.

4. **Attacks stay unauthorized, and H\* stays unfitted.** Recommendation: yes. Three development groups remain a count from Checkpoint 1, not a fit.

# Reproduction instructions

From the worktree root. Unit tests do not load SmolVLM2 weights. The parsing and pilot tests are included because `session.py` gained `prompt_last_logits`.

```bash
PYTHONPATH=src/scene_gen:src/vlm_iface:src/analysis:src/pilot:src/scoring \
  /Users/fedecava/Development/_c0-work/venv-vlm/bin/python -m pytest \
  tests/scoring tests/vlm_iface tests/pilot -q
```

Status before this report: 60 passed in 4.58s. `tests/scoring` alone was 28 passed. The MPS command writes `src/scoring/records/development_mps.json` and `development_mps.jsonl`. It refuses a non-MPS device. It does not run an attack.

```bash
PYTHONPATH=src/scene_gen:src/vlm_iface:src/analysis:src/pilot:src/scoring \
  HF_HOME=/Users/fedecava/Development/models/huggingface \
  /Users/fedecava/Development/_c0-work/venv-vlm/bin/python -m answer_scoring.run_development \
  --device mps
```

The run recorded here exited 0. `checkpoint1_word_nll_reproduced` is true. `generation_spotcheck_matches` is 8. `h_star_fitted` is false. `attack_run` is false.

# Git information

- Branch: `checkpoint-1.5/answer-scoring`
- Started at: `90c6ffe83d61514da217a20f8dfc2f202afbbe69` (`checkpoint-1/dev-pilot`)
- Scoring commit: `e972cd0eb12cbd252e6daf4882c8d605f9bc7e13`
- Base of the pull request: `main`
- The diff against `main` contains Checkpoint 1 because this branch starts at the Checkpoint 1 pilot commit. Those `docs/checkpoint-1/` files are unchanged by this workstream.
- Pull request: https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/pull/8
- The pull-request head is the commit that adds this URL. Its parent is the scoring commit above.
- Tests: `pytest tests/scoring tests/vlm_iface tests/pilot` — 60 passed. MPS scoring command above — exit 0.
