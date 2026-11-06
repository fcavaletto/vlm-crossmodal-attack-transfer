# Checkpoint 1.7 review packet

Research lead: Federico Cavaletto. Every new count in this file is **DEVELOPMENT ONLY**. The sample is `fixtures/development_v4`. It is not a sealed test. No attack was run. H\* was not fit. No held-out scene was generated. No scene was deleted after the table. The decision rule was not changed after the table.

`docs/checkpoint-1.6/**` is unchanged. This branch starts at `004783b5604c34aa066befbcd5356bfd7bd6e7b2`, not at `main`, so a pull request into `main` also shows Checkpoints 0 through 1.6. Those commits are the base.

## 1. Executive summary

Observation. DEVELOPMENT ONLY. Under the locked closed-vocabulary rule, SmolVLM2 scores 32/96 on `neutral` and 27/96 on `helpful_nonrevealing`. Both are below 0.65. The competence gate fails. Qwen3-VL-2B, loaded as BF16 because that is the dtype the existing budget selected, scores 91/96 on `neutral` and 95/96 on `helpful_nonrevealing`. Both are strictly above 0.80, so the gate passes and the ceiling flag is on. The eligible-group count is unset on both models, because this pilot did not measure development_v4 image gradients. Invalid greedy outputs are 0/96 in every scored context. Strict-correct counts equal the closed-vocabulary counts in every context. The columns stay separate.

Interpretation. SmolVLM2 does not clear the pre-registered competence line on this development sample. Qwen3-VL-2B is near the top of the clean scale, which is the ceiling the protocol said to report. Neither result is an attack success rate. Agent D's no-go for an H\* claim from 96 scenes still stands: there is no attack outcome, no alignment, and the eligible count is unset.

## 2. Commit and protocol provenance

The protocol file is `docs/checkpoint-1.7/DEVELOPMENT_PROTOCOL.md`. Its commit is `fd752d68d8237ff9cca44228867922acd2c99cc1`, at 2026-11-06 20:24:30 +0100. That commit contains the manifest hash, the 0.65 gate, the answer prefixes, and the color ids. It does not contain `development_v4` predictions. `git cat-file` on that commit misses both prediction files. The file sha256 recorded at runtime is `8dde031c41e7e64ea73a0ac173e8f038b44c8f7f7fff4bcd57315eb924f5a45a`.

The scorer commit is `66c2a5a13c1866978fddc33e2ade272eca476be5`, at 2026-11-06 20:37:10 +0100. Both model runs record that commit as `git_head_at_run`. The protocol commit is an ancestor of it. SmolVLM2 wrote `inference_started_unix` 1793993862.579315, which is 2026-11-06 20:37:42 +0100, after the protocol commit and after the scorer commit, and before the first development scene forward. Qwen wrote `inference_started_unix` 1793994216.031633, which is 2026-11-06 20:43:36 +0100. The protocol commit precedes both inferences.

The integration merges, each `--no-ff` and without conflict, are:

| Pull request | Head | Merge on this branch |
| --- | --- | --- |
| 16 choice scoring | `c0edc478d0dc97922b48ba335df4a6c6168d1fad` | `f218b0c` |
| 15 scene v4 | `92d332af876d747c41d4f3fff72f8f1d72442e20` | `0b84dee` |
| 14 Qwen RGB | `a1c06d0c646e1e091c117220b41b4d1b6fa26543` | `9ff372a` |
| 13 audit | `1acb79322503958cb523199eef88457aeb95dbd6` | `d551b5e` |

Manifest sha256 `0d9d592b6700acf610453e4f4adf639fc778b4efb24530f60deb01045440d597`. The scene verifier `tests/scene_v4` passed, 66 tests, before either model load. `tests/choice_eval`, `tests/qwen_rgb`, `tests/study2_audit`, and `tests/dev_pilot` passed together, 97 tests, before either model load. `python -m dev_pilot.verify` passed after both prediction files existed.

Reproduction, from the worktree, one model process at a time:

```bash
PYTHONPATH=src/scene_v4:src/scene_gen \
  /Users/fedecava/Development/_c0-work/venv-scene/bin/python -m pytest tests/scene_v4 -q

HF_HOME=/Users/fedecava/Development/models/huggingface \
PYTHONPATH=src/dev_pilot:src/choice_eval:src/vlm_iface:src/qwen_rgb:src/pilot:src/scene_v4:src/study2_audit \
  /Users/fedecava/Development/_c0-work/venv-vlm/bin/python -m pytest \
  tests/choice_eval tests/qwen_rgb tests/study2_audit tests/dev_pilot -q

HF_HOME=/Users/fedecava/Development/models/huggingface \
PYTHONPATH=src/dev_pilot:src/choice_eval:src/vlm_iface:src/qwen_rgb:src/pilot:src/scene_v4 \
  /Users/fedecava/Development/_c0-work/venv-vlm/bin/python -m dev_pilot --model smolvlm2

HF_HOME=/Users/fedecava/Development/models/huggingface \
PYTHONPATH=src/dev_pilot:src/choice_eval:src/vlm_iface:src/qwen_rgb:src/pilot:src/scene_v4 \
  /Users/fedecava/Development/_c0-work/venv-vlm/bin/python -m dev_pilot --model qwen3-vl-2b

HF_HOME=/Users/fedecava/Development/models/huggingface \
PYTHONPATH=src/dev_pilot:src/choice_eval:src/vlm_iface:src/qwen_rgb:src/pilot:src/scene_v4 \
  /Users/fedecava/Development/_c0-work/venv-vlm/bin/python -m dev_pilot.verify
```

## 3. Agent findings

These are the merged workstreams. This pilot does not replace them.

Agent A, pull request 16. The primary score is the lowest summed teacher-forced NLL of one full color continuation. Ties break by color name. The strict parser stays a secondary diagnostic. Tokenizer-only checks, with no weight load, fixed the cues and the ids used here. Smol ids are 2382, 4461, 2654, and 5724 for red, blue, green, and yellow, each with a leading space. Qwen ids are 1151, 12203, 13250, and 27869, each the bare word, because the cue ends in a newline.

Agent B, pull request 15. The development sample is 96 scenes in 48 counterfactual pairs. The only family is `referent_color`. Both arms share the question and the non-revealing text. The edit swaps two objects and changes the answer while keeping every color count. Neutral and helpful text do not contain the answer. The scenes were fixed before any model score.

Agent C, pull request 14. The differentiable Qwen RGB path matches the official processor on `dev2-attribute-00-a`, where the resize is the identity. The chosen-answer RGB gradient was not computed. The Checkpoint 1.5 budget refused `from_pretrained` at 10914119680 available bytes. The binding line was 11509447200 bytes. The record is `src/qwen_rgb/records/development_rgb_gradient.json`, status `stopped_budget`. The loss token in that note is the leading-space id 2518. This protocol's Qwen continuation is the bare id 1151.

Agent D, pull request 13. The Checkpoint 1.6 Qwen JSON has `from_pretrained_called` true at the top and false on the load gate. The original file was not rewritten. The row-color clean-both count is 14 scenes in 9 groups. The eligible count on that table is unset. The decision for a 96-scene H\* claim is no-go. The fabricated one-family grid, seed `20261108`, 200 replications, puts the lower endpoint above zero on 90 of 200 replications at 48 fully eligible groups.

## 4. SmolVLM2 versus Qwen

Observation. DEVELOPMENT ONLY. Both models were scored on the same 96 scenes, the same four contexts, and the same prompt. They were separate processes. SmolVLM2 is float32, revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`, processor `SmolVLMProcessor`. Qwen3-VL-2B is bfloat16, revision `89644892e4d85e24eaac8bacfd4f463576704203`, processor `Qwen3VLProcessor`, image processor `Qwen2VLImageProcessor`. The pinned preprocessor file names `Qwen2VLImageProcessorFast`. The instantiated class is `Qwen2VLImageProcessor`. The cue and the four ids still matched the protocol, so the run continued. Numeric values of \(L\) are not a common unit. The comparable outcome is the color.

| Model | `neutral` | `helpful_nonrevealing` | Gate | Above 80% | Eligible groups |
| --- | ---: | ---: | --- | --- | --- |
| SmolVLM2 | 32/96 | 27/96 | fail | no | unset |
| Qwen3-VL-2B | 91/96 | 95/96 | pass | both primary contexts | unset |

Interpretation. On this development sample the two models are far apart under one rule. Smol misses the competence line. Qwen sits in the ceiling region the protocol required us to keep. A transfer claim would need both models on a task where each has room to move. This table does not supply that claim.

## 5. Clean accuracy tables

Denominator is every scene in the context. DEVELOPMENT ONLY. Closed-vocabulary accuracy is the argmin of summed NLL. The competence gate uses only the two primary contexts.

| Model | Context | Closed correct | Role in the gate |
| --- | --- | ---: | --- |
| SmolVLM2 | `neutral` | 32/96 | primary, below 0.65 |
| SmolVLM2 | `helpful_nonrevealing` | 27/96 | primary, below 0.65 |
| SmolVLM2 | `contradictory` | 21/96 | scored after a stable primary pass, outside the gate |
| SmolVLM2 | `answer_revealing` | 53/96 | scored after a stable primary pass, outside the gate |
| Qwen3-VL-2B | `neutral` | 91/96 | primary, above 0.80 |
| Qwen3-VL-2B | `helpful_nonrevealing` | 95/96 | primary, above 0.80 |
| Qwen3-VL-2B | `contradictory` | 36/96 | scored after a stable primary pass, outside the gate |
| Qwen3-VL-2B | `answer_revealing` | 96/96 | scored after a stable primary pass, outside the gate |

The extra contexts ran because each model's primary pass wrote all 192 rows and OS-available memory after that pass was still at least the existing 6 GiB floor. They did not enter the gate and they were not used to edit the rule.

Scenes clean on both primary contexts, DEVELOPMENT ONLY:

| Model | Scenes | Groups with at least one such scene | Groups with both scenes |
| --- | ---: | ---: | ---: |
| SmolVLM2 | 21 | 18 | 3 |
| Qwen3-VL-2B | 91 | 48 | 43 |

The same counts are the strict clean-both counts, because the strict-correct flag matches the closed-vocabulary flag on every row of both files. The name of the count is still clean-both, not eligible.

Ceiling, Qwen only. DEVELOPMENT ONLY. 91/96 and 95/96 are strictly above 0.80. 76/96 would not have been. The scenes stay. The rule stays. Interpretation: clean accuracy has little room to rise. A later attack, if one is authorized, would be applied to a decision that is already correct on nearly all of these development scenes. That is a ceiling on the clean task. The clean-error counts are 5/96 and 1/96. Agent D's flat-outcome flag is defined on an attacked error that this pilot did not measure.

Smol's rates are below 0.65 and below 0.80. The failed gate is the result. The scenes stay.

## 6. Strict versus closed-vocabulary diagnostics

Observation. DEVELOPMENT ONLY. Greedy decoding used `max_new_tokens` 16. The parser is the unchanged attribute vocabulary. Correct, valid-but-wrong, and invalid partition each context.

| Model | Context | Strict correct | Valid-but-wrong | Invalid |
| --- | --- | ---: | ---: | ---: |
| SmolVLM2 | `neutral` | 32/96 | 64/96 | 0/96 |
| SmolVLM2 | `helpful_nonrevealing` | 27/96 | 69/96 | 0/96 |
| SmolVLM2 | `contradictory` | 21/96 | 75/96 | 0/96 |
| SmolVLM2 | `answer_revealing` | 53/96 | 43/96 | 0/96 |
| Qwen3-VL-2B | `neutral` | 91/96 | 5/96 | 0/96 |
| Qwen3-VL-2B | `helpful_nonrevealing` | 95/96 | 1/96 | 0/96 |
| Qwen3-VL-2B | `contradictory` | 36/96 | 60/96 | 0/96 |
| Qwen3-VL-2B | `answer_revealing` | 96/96 | 0/96 | 0/96 |

The greedy string was exactly the closed-vocabulary color on all 384 Smol rows and on 381 of 384 Qwen rows. Sum and mean chose the same color on every row, which is what one-token continuations do. The three Qwen exceptions are valid color words, untruncated, and wrong under both columns:

| Scene | Context | Label | Argmin | Greedy text |
| --- | --- | --- | --- | --- |
| `dev4-ref-35-b` | `neutral` | yellow | blue | `red` |
| `dev4-ref-26-a` | `contradictory` | yellow | blue | `red` |
| `dev4-ref-26-b` | `contradictory` | green | blue | `red` |

Interpretation. On this prompt the strict parser and the argmin tell almost the same story, because the model emitted one bare color. That agreement is a property of these strings. It is not a reason to fold the columns together. Checkpoint 1.6 already showed a context where the strict column and a class-loss column diverge. This table does not repeat that divergence.

## 7. Counterfactual groups

Observation. DEVELOPMENT ONLY. Each group has two arms and two different labels. Pair consistency is both arms closed-vocabulary correct. `same_decision` is the count of groups whose two decisions are the same color. No pixel was perturbed. The pair is the vision-dependence check.

| Model | Context | Both correct | One correct | Neither | Decisions differ | Same decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SmolVLM2 | `neutral` | 8/48 | 16/48 | 24/48 | 18/48 | 30/48 |
| SmolVLM2 | `helpful_nonrevealing` | 6/48 | 15/48 | 27/48 | 19/48 | 29/48 |
| SmolVLM2 | `contradictory` | 3/48 | 15/48 | 30/48 | 13/48 | 35/48 |
| SmolVLM2 | `answer_revealing` | 13/48 | 27/48 | 8/48 | 29/48 | 19/48 |
| Qwen3-VL-2B | `neutral` | 43/48 | 5/48 | 0/48 | 48/48 | 0/48 |
| Qwen3-VL-2B | `helpful_nonrevealing` | 47/48 | 1/48 | 0/48 | 48/48 | 0/48 |
| Qwen3-VL-2B | `contradictory` | 14/48 | 8/48 | 26/48 | 23/48 | 25/48 |
| Qwen3-VL-2B | `answer_revealing` | 48/48 | 0/48 | 0/48 | 48/48 | 0/48 |

Strict both-correct equals the closed both-correct column in every row of this table.

Interpretation. On the primary contexts, Qwen's decision changes with the arm on every group. Smol assigns one color to both arms on 30 of 48 neutral groups and 29 of 48 helpful groups. A shared color cannot track an edit that changes the label. That is the vision-dependence check on the clean images. It is not an attack transfer.

The sampling unit for a later interval is the group. The scene rates above are not independent-sample standard errors. This packet does not redraw a cluster interval for the new table.

## 8. Qwen RGB gradient validation

Observation. Agent C's record still says `model_from_pretrained_called` false, `rgb_gradient_finite` null, and `stopped_budget`. The processor comparison on `fixtures/development_v2/images/dev2-attribute-00-a.png` matched the official PIL path with maximum absolute gap 0. A separate sum of `pixel_values` had a finite nonzero RGB gradient. That sum is the preprocessor Jacobian. It is not \(L(y)\).

This pilot's Qwen process did call `from_pretrained`, after the same budget function permitted BF16. The load was for clean scores. No backward was taken. No development_v4 gradient was stored. `gradients_measured_on_development_v4` is false in both run files. The eligible count stays unset. Agent C's scene is not in `fixtures/development_v4`, so that record does not make a v4 group eligible.

The gradient objective named before the run is the summed NLL of the labeled color's protocol continuation, with respect to canonical RGB in \([0, 1]\), when the path exists. For Qwen that continuation is the bare color id, not token 2518.

Interpretation. A clean-score load and a gradient measurement are different operations. The budget allowed the first on this later OS reading. The second was not part of this pilot. The question Agent C left open, whether \(L(y)\) has a finite nonzero RGB gradient, is still open.

## 9. Memory and runtime

Observation. DEVELOPMENT ONLY. Apple M4, 25769803776 bytes of unified memory. One model resident at a time. The 6 GiB OS floor, the 5 GiB spare, and the 2 GiB projection floor were the Checkpoint 1.5 constants. They were not edited. Smol's loader refuses below 6 GiB. Qwen's loader calls `qwen_rgb.budget`. Float32 is selected when its lines pass. BF16 is selected when float32 does not pass, the BF16 lines pass, and the MPS probe has run.

| Model | Dtype | Load seconds | Score seconds | Rows | OS available before load | OS available after load | MPS current after load |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| SmolVLM2 | float32 | 5.409 | 203.343 | 384 | 10164.2 MB | 7399.3 MB | 1965.6 MB |
| Qwen3-VL-2B | bfloat16 | 5.411 | 651.863 | 384 | 11721.1 MB | 6045.4 MB | 4072.9 MB |

The MB figures are the snapshot's `os_available_mb` and `mps_current_allocated_mb`. Qwen's before-load reading is above the BF16 calibrated line of 11509447200 bytes and below the float32 line of 20871410752 bytes. That is the selection the code is written to make. Smol's before-first-scene available bytes were 7759183872, still above 6 GiB, so the extra contexts ran. Qwen's before-first-scene available bytes were 6373998592, also above 6 GiB, so the extra contexts ran. There was no memory refusal and no retry.

Wall clock from process start to exit was about 214 seconds for Smol and about 663 seconds for Qwen. The Qwen process started after the Smol process had exited and the resident file was gone.

## 10. Statistical feasibility from Agent D

Observation. Agent D's decision is no-go. Ninety-six scenes are 48 groups. Support for H\* would require eligible groups, measured alignments, attack outcomes, one pre-specified coefficient, and a cluster interval whose lower endpoint is strictly above zero. This pilot supplies none of the attack side and leaves the eligible count unset.

The clean-both counts are not that count. Smol has 21 scenes in 18 groups that are strict-correct on both primary contexts, 3 of those groups contributing both scenes. Qwen has 91 scenes in 48 groups, 43 of those groups contributing both scenes. DEVELOPMENT ONLY. A missing gradient is an exclusion. It is not coded as zero.

Agent D's fabricated grid, which this pilot did not refit, reports that 48 fully eligible groups put the lower endpoint above zero on 90 of 200 replications at a known coefficient of 0.25. Nine groups do so on 42 of 200. The published three-family coverage description of 24 groups is a different design and was not treated as automatically enough.

Interpretation. A high Qwen clean-both count does not convert the no-go into a go. Eligibility still requires finite, strictly positive gradients on both primary contexts and a finite alignment in \([-1, 1]\), plus the leakage and ambiguity gates. Those gradients were not measured. H\* was not fit. The sampling unit remains the group.

## 11. Threats to validity

The table is a development table. It was not sealed, and model outputs were not used to drop scenes.

There is one family and eight synthetic objects on every scene. A result here does not travel to photographs or to a counting question.

Smol ran in float32 and Qwen in bfloat16. The dtype rule was fixed before the scores. The numeric losses are still not comparable.

The instruction line names all four colors on every scene. Agent B's leakage checks cover the context and the question. They do not say the instruction is free of color words. The instruction is the same line for every scene and every context, which is the protocol.

Qwen's image-processor class name differs from the `Fast` name in the snapshot config. Scoring used the instantiated processor. The cue and the ids matched. A later gradient path that builds `Qwen2VLImageProcessor` by hand has to stay on the same continuation.

Smol's primary pairs often share a decision. An attack on a pair that already ignores the image would be an attack on a vision-independent error. That is a different estimand from transfer off a vision-dependent decision.

Group dependence means the scene fractions are descriptive. This packet does not add a cluster interval, so it does not pretend the 96 scenes are 96 independent draws.

The OS reading moved between Agent C's refusal and this Qwen load. The guard did not move. A later gradient attempt still has to read the OS again.

## 12. Decisions needed before attacks

1. SmolVLM2 failed the 65% line, 32/96 and 27/96, DEVELOPMENT ONLY. An attack on that model would be an attack on a checkpoint that missed the clean bar. The bar stays unless a new protocol replaces it before any new prediction.
2. Qwen3-VL-2B is above 80% on both primary contexts, 91/96 and 95/96, DEVELOPMENT ONLY. The ceiling is part of the record. Whether the task is too easy for a graded attack study is a lead decision. The scenes were not dropped to change the rate.
3. The eligible-group count is unset. A gradient pass, if it is authorized, uses \(L(y)\) for the protocol continuation, one model at a time, the existing budget, and no signed step. Qwen's bare color id is the continuation. Token 2518 is not. This packet does not run that pass.
4. H\* stays unfit. Agent D's no-go is the feasibility statement for a 96-scene claim. Clean-both counts do not replace eligible groups, alignments, or attack rows.
5. `contradictory` and `answer_revealing` are controls. They are outside the competence gate. Qwen's revealing column is 96/96, DEVELOPMENT ONLY. That is the text naming the color. It is not the primary score.
6. This pull request is for review. It is not merged.
