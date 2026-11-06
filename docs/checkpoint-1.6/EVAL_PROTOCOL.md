# Checkpoint 1.6 clean-evaluation protocol

**DEVELOPMENT ONLY.** This file is the record of the model, prompt, dataset, scoring method, and acceptance criteria for one clean evaluation of the new row-color scenes. It is written before that model call. It does not authorize an attack, a fit of H\*, a held-out scene, training, or a cloud GPU.

Qwen3-VL-2B-Instruct did not pass the Checkpoint 1.6 load guard. `from_pretrained` was not called. That refusal is not an out-of-memory demonstration. Qwen is not loaded in this evaluation.

## Model

SmolVLM2-500M-Video-Instruct, revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`. MPS, float32, eager attention, `do_image_splitting=False`, greedy decoding, `max_new_tokens=16`, leading-space answer tokens. This is the only model whose local MPS path has already passed a full-resolution gradient gate, in Checkpoint 1. The load starts only when the OS reports at least 6 GiB available.

## Dataset

Every scene in `fixtures/development_simple/manifest.jsonl`, sha256 `bf21cdcfeca8f5c9a05e9f53465402c48f4a3175bbd363a38ca36dbc443f2367`. 24 scenes, 12 groups, family `row_color`, all four contexts. No scene is dropped before the table exists. The contact sheet is not a model input. Checkpoint 1 and Checkpoint 1.5 fixtures are not rescored and not overwritten.

## Prompt

`context_text + "\n" + question`. No format line. Helpful text stays task guidance.

## Scoring

The official outcome is the Checkpoint 1 strict parser on the color vocabulary (`red`, `blue`, `green`, `yellow`), the same four names D4 uses for attribute and spatial items. The parser is called with that color vocabulary. The stored `task_family` remains `row_color`. Counting aliases are not used. Correct, valid-but-wrong, and invalid are a partition.

The class-loss argmin over the surfaces that parser already accepts is stored beside the official label. It is not the official outcome. Agent C's report is the reason: on the committed v2 rows, class scoring did not rescue clean accuracy, and the protocol was not chosen to raise a rate. The canonical-word argmin is stored the same way.

## Gradient feasibility

After the 96-row table is written, one scene is checked: `dev3-row-00-a`, canonical-word image gradient, `neutral` and `helpful_nonrevealing`, each repeated once. The check runs only if OS-available memory is still at least 6 GiB. A finite nonzero gradient of shape `(3, 384, 512)` is a feasibility result. It is not an attack. If the memory guard stops the check, the clean table still stands and the gradient result is "not attempted."

## Acceptance

These rules were fixed before the row-color numbers.

- The design band is 0.60 through 0.80 inclusive, on strict accuracy, in both `neutral` and `helpful_nonrevealing`.
- The task is unsuitable when strict accuracy is below 0.50 in either of those contexts. The recommendation is then not to adopt it as the primary task. The rows stay in the table.
- A rate inside the band is necessary for adoption and is not by itself an authorization of attacks.
- Failure is an acceptable result.
- Report the number of scenes, and the number of independent groups, that are strict-correct under both primary contexts. That count is not a fitted H\* coefficient.
- Also report pair correctness (both arms strict-correct), agreement of the class-loss argmin with the official label on valid rows, runtime, and memory.

H\* stays provisional and unfitted.
