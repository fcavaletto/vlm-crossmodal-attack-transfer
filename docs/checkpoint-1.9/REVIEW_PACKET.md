# Checkpoint 1.9 review packet

DEVELOPMENT ONLY. Research lead: Federico Cavaletto. Qwen3-VL-2B-Instruct was not loaded. No pixel was stepped. No H* fit. Not merged.

The protocol is commit `fce3480d158298851587a692c56b6e4eebb554a7`. The runner is commit `45c969e030af70d2d7a8f5478691271e647376dd`. The branch starts at `ebe3db8ec300303affdaebadeb280ccba439e97a`. Pull request: https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/pull/19. It is not merged. The pull request targets `checkpoint-1.8/qwen-rgb-loss`, not `main`.

## Stop

Stage A did not run. Stage B did not run. The stop is the Checkpoint 1.5 memory guard, applied by `qwen_rgb.budget` and not lowered.

OS-available memory entering the decision was 11,093,655,552 bytes. A fresh reading at the same moment was the same number. Bfloat16 needs 11,509,447,200 bytes. The shortfall is 415,791,648 bytes. `from_pretrained` was not called. The load was not retried.

| Dtype | Resident bytes | Binding line | Required available bytes | Permitted | Failed lines |
| --- | ---: | --- | ---: | --- | --- |
| float32 | 8,510,128,128 | calibrated projection | 20,871,410,752 | no | spare, calibrated projection, staging double copy |
| bfloat16 | 4,255,064,064 | calibrated projection | 11,509,447,200 | no | calibrated projection |

Bfloat16 cleared the 6 GiB OS floor (6,442,450,944 bytes), the 5 GiB spare line (9,623,773,184 bytes), and the staging double-copy line (8,510,128,128 bytes). It missed only the calibrated projection. Float32 missed three lines. MPS was available. The bfloat16 matmul probe ran and was not a model forward. The local safetensors header was the Checkpoint 1.5 count, 2,127,532,032 parameters, exclusively BF16.

The answer prefix matched `<|im_end|>\n<|im_start|>assistant\n`. The bare color ids matched red 1151, blue 12203, green 13250, yellow 27869. Token 2518 was not selected.

Wall clock was 1.24 seconds. That is the processor, the header read, and the refusal. It is not a gradient time.

## What the checkpoint was going to measure

The loss would have been the summed negative log-likelihood of the scene's own color, read from the first continuation logit. The mean over tokens is not the loss. The gradient would have been the derivative of that scalar with respect to canonical float RGB in [0, 1], through the same unclipped differentiable path as Checkpoint 1.8.

A positive gradient-sign step adds epsilon times the sign of that gradient. A negative step subtracts it. A matched random-sign step uses a fixed Rademacher field on the same support. Each proposal is projected with `qwen_rgb.constraints.project_linf`: the L-infinity ball around the clean image, intersected with [0, 1]. The official 8-bit round trip uses PyTorch half-to-even rounding. None of these functions was applied to a development image.

Epsilon is on the [0, 1] RGB scale. A budget of `k/255` moves each channel by at most k levels of an 8-bit image. `1/255` is one stored level. `0.5/255` is half a level, so the 8-bit round trip can erase it. `4/255` is 0.01569, next to the Checkpoint 1.8 finite-difference step of `1/64` = 0.015625.

Stage A would have used eight scenes, four budgets, and three directions: 32 cells, each with a positive step and two controls. The pass rule, fixed before this run, requires every budget to raise the float loss on at least 6 of 8 scenes and to beat both controls on at least 6 of 8, with the positive median strictly above 0 and strictly above both control medians. A missing change is not entered as zero. Stage B, 16 scenes of projected gradient ascent, was allowed only if that rule passed.

## Selection

Stage A rule `per-color-extreme-alignment-exploratory`. Within each labeled color, the lowest and highest Checkpoint 1.8 value of \(A(\text{neutral}, \text{helpful})\) are kept. The sample is exploratory. It cannot support an unbiased population attack-success rate. The alignments below are the stored Checkpoint 1.8 numbers the rule read. They are not attack outcomes, and they were not regressed.

| Scene | Label | Stratum | Stored alignment |
| --- | --- | --- | ---: |
| `dev4-ref-01-a` | red | lowest in color | 0.2738961986673083 |
| `dev4-ref-09-b` | red | highest in color | 0.9345366800860538 |
| `dev4-ref-25-a` | blue | lowest in color | 0.4940414745458451 |
| `dev4-ref-21-b` | blue | highest in color | 0.9522294672411216 |
| `dev4-ref-30-b` | green | lowest in color | 0.5875956154399549 |
| `dev4-ref-07-a` | green | highest in color | 0.9775420055497965 |
| `dev4-ref-17-b` | yellow | lowest in color | 0.19669284166797163 |
| `dev4-ref-45-b` | yellow | highest in color | 0.9753625290227933 |

Stage B rule `scene-id-stride-6` lists sixteen scene ids, indexes 0, 6, ..., 90 after sorting `scene_id`. It does not read alignment. Those scenes were not attacked. The list is in `docs/checkpoint-1.9/ATTACK_PROTOCOL.md`.

## Diagnostic table

No scene was forwarded through Qwen, and no scene was perturbed. The denominator of every attack count is the number of scenes this checkpoint stepped.

| Budget | Scenes stepped | Positive-sign records | Float loss increased | Beat the negative control | Beat the random control | Closed-vocabulary answer changed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.5/255 | 0/8 | 0 | 0/0 | 0/0 | 0/0 | 0/0 |
| 1/255 | 0/8 | 0 | 0/0 | 0/0 | 0/0 | 0/0 |
| 2/255 | 0/8 | 0 | 0/0 | 0/0 | 0/0 | 0/0 |
| 4/255 | 0/8 | 0 | 0/0 | 0/0 | 0/0 | 0/0 |

A cell of 0/0 is an empty denominator, not a success rate and not a failure rate. Stage B's source-flip denominator is 0/16 scenes attacked. Transfer is 0/0.

There is no illustrative pair. The precommitted figure would have been the official 8-bit image of `dev4-ref-01-a` at `4/255` beside the clean image, or the Stage B image of `dev4-ref-00-a` if Stage A had passed. Neither file was written.

Gradient hashes compared with Checkpoint 1.8: 0 recorded, 0 matches, 0 mismatches. There was no gradient to hash.

## What this does not show

This stop is not evidence that a sign step fails, and it is not evidence that the model is robust. The labeled-color loss was not evaluated at a perturbed image.

The Checkpoint 1.8 directional finite-difference check remains a failure. One of eight gate pairs was confirmed, and only at the smaller step. This checkpoint does not replace that result. It also does not add a signed-step result that would explain it.

\(A(s, t)\) is still the clean-gradient alignment from Checkpoint 1.8. It was used only to name an exploratory Stage A list. It was not regressed on an attack outcome. There is no attack outcome. H* was not fit. Held-out scenes were not opened. SmolVLM2 was not loaded.

Forty-eight eligible groups in the Checkpoint 1.8 packet remain a pre-attack count. This run does not change that count.

## Threat and open questions

No development image was modified. The attack that the protocol describes, and that this run did not execute, is an L-infinity change to a development RGB image, optimized against the labeled-color loss of this local Qwen checkpoint. It is not a claim about any deployed system.

The open questions are the ones the protocol was written to ask. Do positive-sign steps raise the labeled-color loss at 0.5/255, 1/255, 2/255, and 4/255, and do they beat a negative step and a matched random step? Does an increase on the float path survive the 8-bit round trip? If Stage A passes, does seven-step projected gradient ascent at 1/255, 2/255, or 4/255 flip a correct clean answer under neutral, and does that stored image also flip the helpful context? The bfloat16 finite-difference miss is still open. None of those questions was answered here.

## Recommendation

Do not lower the 6 GiB floor, the 5 GiB spare, the 2 GiB projection floor, or the calibrated line to make this load fit. Do not retry this saved refusal: the runner treats `stopped_budget` as terminal. Do not fit H*. Do not merge to main. Do not read the empty table as an attack-success rate.

A later attempt can use the same protocol and the same guard when the process that is about to call `from_pretrained` sees at least 11,509,447,200 bytes available for bfloat16. Until that happens, Checkpoint 1.9 has no attack result.

## Files

- `docs/checkpoint-1.9/ATTACK_PROTOCOL.md`, sha256 `2676683793d908c9f5c8af9a46867d39d0ecea18cd5dbbe54144196153c78648`
- `docs/checkpoint-1.9/attack_feasibility/summary.json`, sha256 `19779e8ba5bcbeb35e7822322ae32c0194f8a2d4d6065166c4a08847de9c2620`
- `docs/checkpoint-1.9/attack_feasibility/progress.json`, sha256 `9b6f9c1bbf763543bcd4dfe957179603bc70e4b7b1db6611877e5999b71e490b`

Checkpoint 1.7 predictions and Checkpoint 1.8 gradient records were not edited.

## Reproduction

From the worktree root. The unit tests do not load Qwen. The measurement command is the one that stopped. Starting it again does not load the model: the saved refusal is terminal.

```bash
HF_HOME=/Users/fedecava/Development/models/huggingface \
PYTHONPATH=src/attack_feasibility:src/qwen_rgb_loss:src/qwen_rgb:src/choice_eval:src/vlm_iface:src/pilot:src/dev_pilot:src/analysis:src/scene_v4 \
/Users/fedecava/Development/_c0-work/venv-vlm/bin/python -m pytest tests/attack_feasibility -q

HF_HOME=/Users/fedecava/Development/models/huggingface \
PYTHONPATH=src/attack_feasibility:src/qwen_rgb_loss:src/qwen_rgb:src/choice_eval:src/vlm_iface:src/pilot:src/dev_pilot:src/analysis:src/scene_v4 \
/Users/fedecava/Development/_c0-work/venv-vlm/bin/python -m attack_feasibility
```
