# Checkpoint 1.7 development protocol

Status: prospective. Research lead: Federico Cavaletto. This document is the competence gate and the scoring rule for the clean `fixtures/development_v4` pilot. It is not a correction of Checkpoint 1.6 and it does not replace any Checkpoint 1.6 score. It is committed before any model call that reads `fixtures/development_v4`. No scene is deleted after the table exists. The decision rule in this file is not revised after that table exists. Failure of the competence gate is an acceptable result.

Label of every number this pilot produces: DEVELOPMENT ONLY. The sample is a development sample. It is not a sealed test. Nothing here is an attack, an H\* fit, a held-out scene, a training run, or a cloud-GPU job.

## Integration parent

The worktree branch is `checkpoint-1.7/review-packet`. It starts at `004783b5604c34aa066befbcd5356bfd7bd6e7b2` and merges, with `--no-ff` and without conflict, these heads:

| Pull request | Head | Content |
| --- | --- | --- |
| 16 | `c0edc478d0dc97922b48ba335df4a6c6168d1fad` | choice scoring |
| 15 | `92d332af876d747c41d4f3fff72f8f1d72442e20` | scene v4 |
| 14 | `a1c06d0c646e1e091c117220b41b4d1b6fa26543` | Qwen RGB path |
| 13 | `1acb79322503958cb523199eef88457aeb95dbd6` | audit |

`docs/checkpoint-1.6/**` is not edited. The branch does not start at `main`, so a pull request into `main` also shows earlier checkpoints. Those commits are the base.

## Models and processors

Both the weights and the processor files are the snapshot at the pinned revision. The processor revision is that same commit. Tokenizer-only checks already recorded the generation cues and the color ids. This pilot does not pick a different cue because a probability moved.

| Key | Model id | Model and processor revision | Config processor class | Image processor named in that snapshot |
| --- | --- | --- | --- | --- |
| `smolvlm2` | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` | `SmolVLMProcessor` | `SmolVLMImageProcessor` |
| `qwen3-vl-2b` | `Qwen/Qwen3-VL-2B-Instruct` | `89644892e4d85e24eaac8bacfd4f463576704203` | `Qwen3VLProcessor` | `Qwen2VLImageProcessorFast` |

The instantiated class names are recorded. A mismatch of the generation cue or of the selected color ids stops the run before any development_v4 scene is scored. A class-name alias does not by itself change the cue or the ids.

SmolVLM2 is loaded only through `vlm_iface.load.load_pinned_model`: float32, eager attention, MPS, `do_image_splitting` false. That flag is the existing session default and the Checkpoint 1.6 setting. It is not tuned on these scenes. Qwen is loaded only through `qwen_rgb.measure.load_model_after_budget` after `qwen_rgb.budget` permits a dtype. Float32 is preferred when its budget passes. BF16 is used only when float32 fails, the BF16 lines pass, and the existing MPS BF16 probe has run.

Device is MPS. A CPU result is not an MPS result.

## Manifest

`fixtures/development_v4/manifest.jsonl` sha256 `0d9d592b6700acf610453e4f4adf639fc778b4efb24530f60deb01045440d597`.

The file is 96 lines. Every row has `split` `development` and `task_family` `referent_color`. There are 48 counterfactual groups. The runner checks this hash and `fixtures/development_v4/checksums.json` before either model is loaded. A mismatch stops the run. Images are not regenerated for the pilot and are not replaced. No held-out directory is created.

## Prompt and answer prefix

The prompt is the one in `docs/checkpoint-1.7/SCORING_PROPOSAL.md`. Context and question stay separate fields. The rendered user text is three lines: the context, the question, and this fixed instruction:

`"Reply with one of these colors: red, blue, green, yellow."`

The instruction does not contain the scene label. It is the same line for every context.

The answer prefix is the chat template's generation cue after that user text. It does not depend on the color or the context.

- SmolVLM2 prefix, as a JSON string: `"<end_of_utterance>\nAssistant:"`. The cue does not end in whitespace, so each continuation is one space plus the color word.
- Qwen3-VL-2B prefix, as a JSON string: `"<|im_end|>\n<|im_start|>assistant\n"`. The cue ends in a newline, so each continuation is the bare color word.

Selected token ids, already checked without a weight load:

| Color | SmolVLM2 id | Qwen3-VL-2B id |
| --- | ---: | ---: |
| red | 2382 | 1151 |
| blue | 4461 | 12203 |
| green | 2654 | 13250 |
| yellow | 5724 | 27869 |

Each selected color is one token on these revisions. Leading-space ids are different and are not selected for Qwen. The Qwen RGB note that would have differentiated token 2518 (` red`) is not this protocol's continuation. End-of-turn ids are not part of the sum.

Before any development_v4 image is opened, the loaded processor's image-turn cue is compared with the JSON string for that model, and the selected ids are compared with the table. A difference stops that model. The cue is not searched over.

## Primary closed-vocabulary rule

Four classes: red, blue, green, yellow. For each class, \(L(c)\) is the summed teacher-forced negative log-likelihood of the full selected continuation. The decision is the lowest sum. Equal sums break by color name, so blue precedes green, red, and yellow. A finite nonzero gap is not a tie. The mean token NLL is stored and is not the decision.

On these revisions the continuation is one token, so the four sums are the negative log-probabilities of those four ids in the single logit row that predicts the first continuation token. A longer continuation, if a future tokenizer produced one, would be the sum of the teacher-forced conditionals from that row onward. The decision would still be the sum, not the mean.

Closed-vocabulary accuracy is the number of scenes whose decision equals the scene label, divided by every scene in that context. The denominator is 96. Invalid generations are not removed from it. There is no invalid bucket in the closed-vocabulary column: a tokenized row has a color.

## Secondary strict-generation diagnostic

Greedy decoding, `do_sample` false, `max_new_tokens` 16. The string is parsed with `vlm_iface.parsing.parse_generation` and `task_family="attribute"`, which is the four color names. The parser is not edited. Correct, valid-but-wrong, and invalid are a partition. Strict accuracy is correct divided by 96. The invalid rate uses the same denominator. A sentence that only contains a color stays invalid. These three counts are not added into closed-vocabulary accuracy. There is no blended rate.

## Gradient objective

The scalar a later image attack would differentiate is \(L(y)\) for the labeled color, the same sum the argmin compares. It is not the sum of the chosen color, not the mean, and not a softmax over the four sums. The derivative is with respect to canonical float RGB in \([0, 1]\), and only where a path from that tensor into the real model input exists.

Smol's path is `vlm_iface` differentiable pixels, the binding `VLMSession` already uses, and only when that binding's shape check passes. Qwen's path is `qwen_rgb` from RGB through resize, normalize, and patches, and only when the unchanged budget has permitted a load. Clean scores do not swap that differentiable tensor in for Qwen. Qwen clean scores use the official `apply_chat_template` tensors at the pinned revision. This pilot does not run the gradient, does not take a signed step, and does not perturb a pixel.

## Contexts

Every scene is scored under `neutral` and `helpful_nonrevealing`. Those two are the primary contexts. The competence gate uses only them.

`contradictory` and `answer_revealing` are scored for a model only when that model's primary pass has one row for all 192 scene-context pairs, the pass did not stop on the memory guard, and OS-available memory after the primary pass is still at least the existing 6 GiB floor. They do not enter the competence gate, and they are not a reason to change the decision rule or the scene list. If the primary pass stops, they are not started.

## Competence gate

Fixed before any development_v4 prediction:

- Closed-vocabulary accuracy under `neutral` is at least 0.65, and closed-vocabulary accuracy under `helpful_nonrevealing` is at least 0.65. Each rate is correct divided by 96. "At least 65%" means the proportion is greater than or equal to 0.65. 62/96 is below the line. 63/96 is on or above it.
- Above 80% means the proportion is strictly greater than 0.80. 76/96 is not above 80%. 77/96 is above 80%. A rate above 80% is reported, the scenes stay, and the ceiling risk is discussed. It is not a reason to drop scenes or to change this rule.
- A rate below 0.65 on either primary context is a failed gate. The scenes stay. The rule stays. The failure is the result.
- Beside the gate, report the count of independent groups eligible for H\*. That count is defined below. It is not a second accuracy threshold. This pilot does not measure development_v4 image gradients, so the pre-registered value of the count is unset. Unset is not zero and it is not the clean-both count.

## Eligible groups

The sampling unit is the counterfactual group. A group is eligible for H\* only when the group is internally consistent and at least one of its scenes passes every gate that does not look at an attack outcome:

- The scene is strict-correct under both `neutral` and `helpful_nonrevealing`.
- `ambiguity_flags` is empty.
- Neither primary context has `reveals_answer` true, and neither the question nor those context strings contains the answer word.
- The question and each primary context string are the same on both arms.
- Both primary-context image-gradient L1 norms are finite and strictly positive.
- The sign alignment of those two gradients is finite and inside \([-1, 1]\).

A missing gradient is an exclusion. It is not coded as zero. An incomplete gradient pass leaves `n_groups_eligible` unset. The clean-both counts, strict and closed-vocabulary, are separate fields. Agent C's measurement on `dev2-attribute-00-a` is not a development_v4 gradient and does not make a v4 group eligible.

H\* is not fit. No attack outcome exists, so the regression is not run.

## Pair consistency

Counterfactual pairs are the vision-dependence check. There are no perturbations. For one model and one context, each group has two arms with different labels. Closed-vocabulary pair consistency is the number of groups in which both arms' decisions equal their own labels. The same groups are also partitioned into both correct, exactly one correct, and neither correct. `decisions_differ` counts groups whose two decisions are different colors. `same_decision` counts groups whose two decisions are the same color. Strict pair consistency is the both-correct count on the strict parse and is not mixed into the closed-vocabulary count.

## Memory guards

The guards are the ones already in the tree. They are not lowered, replaced, or retried with a smaller number.

- Smol load: `vlm_iface.pins.MIN_AVAILABLE_BYTES`, 6 GiB. `load_pinned_model` refuses the load below that line.
- Qwen load: `docs/checkpoint-1.5/run_model_feasibility.py` through `qwen_rgb.budget`. Spare above weights is 5 GiB. Projected remaining floor is 2 GiB. OS floor is 6 GiB. The calibrated drop ratio is the Checkpoint 1 constant in that script. `qwen_rgb.budget.load_budget_module` refuses to run if those lines differ.
- One model is resident. The process accepts one model key. It writes a resident pid file before `from_pretrained` and deletes it on the way out. A live pid, or another `dev_pilot` or `qwen_rgb.measure` process, stops the new load.
- If the Qwen guard's single decision does not permit a load, the available bytes that entered the decision, the fresh available bytes at the stop, the required lines, and `model_from_pretrained_called: false` are written, and that model stops. There is no second attempt and no lower line.
- An out-of-memory error after a permitted load is recorded. The guard is not lowered and the load is not retried.

## What this run stores

Compact JSON only. One row per model, scene, and context: ids, the four summed NLLs, the four mean NLLs, the decision, the greedy string, and the strict partition. No weights, no logit tensors, no pixel tensors, no raw gradients. Output directory: `docs/checkpoint-1.7/development_v4/<model_key>/`.

## Reproduction

From the worktree root, after this file is the HEAD copy. Unit tests and the scene verifier do not score `fixtures/development_v4`. The scene verifier is `tests/scene_v4`, which regenerates the sample and checks the committed checksums. The two model commands are separate processes.

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

The Qwen command is started only after the Smol process has exited and the resident file is gone. Hub cache is `/Users/fedecava/Development/models/huggingface`. The interpreter is `/Users/fedecava/Development/_c0-work/venv-vlm/bin/python` for model commands and `/Users/fedecava/Development/_c0-work/venv-scene/bin/python` for the scene verifier.
