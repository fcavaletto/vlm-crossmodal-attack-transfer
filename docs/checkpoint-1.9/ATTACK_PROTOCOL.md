# Checkpoint 1.9 attack protocol

Status: fixed before any new development_v4 forward and before any perturbation from this checkpoint. Research lead: Federico Cavaletto. Label of every number this run produces: DEVELOPMENT ONLY.

This file does not replace `docs/checkpoint-1.7/DEVELOPMENT_PROTOCOL.md` or `docs/checkpoint-1.8/MEASUREMENT_PROTOCOL.md`. It does not change the decision rule, the prompts, the token ids, the model revision, or the memory lines. It does not rewrite Checkpoint 1.7 predictions or Checkpoint 1.8 gradient records.

The Checkpoint 1.8 directional finite-difference check failed. One of eight gate pairs was confirmed, and only at the smaller step. That failure stays in the record. This checkpoint does not replace it with a passing finite-difference result and does not rerun that check in order to overturn it.

No H* fit. No held-out scene. No training. No cloud GPU. No merge to main. SmolVLM2 is not loaded. Qwen3-VL-2B-Instruct only, one MPS load in the process that reaches the budget.

## Model and loss

`Qwen/Qwen3-VL-2B-Instruct`, revision `89644892e4d85e24eaac8bacfd4f463576704203`. Processor and answer prefix are the Checkpoint 1.7 lock. The answer prefix is `<|im_end|>\n<|im_start|>assistant\n`. The cue ends in a newline, so the continuation is the bare color word.

| Color | Token id |
| --- | ---: |
| red | 1151 |
| blue | 12203 |
| green | 13250 |
| yellow | 27869 |

Token 2518 (` red`) is not the continuation. A mismatch of the cue, the ids, or a loaded commit hash stops the run before a perturbation.

The scalar is the summed teacher-forced negative log-likelihood of the labeled color, read from the first continuation logit. The mean token NLL is stored and is not the step objective. On the clean backward, three formulas on that same logit row must agree within `1e-5`, the same Checkpoint 1.8 identity. If they disagree, no step is taken and Stage B does not start.

The derivative is with respect to canonical float RGB in `[0, 1]`, through `qwen_rgb.rgb_path.differentiable_model_pixels`. The backward path is not clipped. A missing, non-finite, or zero gradient is not a step. It is not entered as a zero gradient.

Device is MPS. One process calls the model load at most once, and only through `qwen_rgb.measure.load_model_after_budget` after `qwen_rgb.budget` permits a dtype. Float32 is used when its budget passes. Otherwise bfloat16 is used when its budget and the MPS probe pass. The 6 GiB OS floor is `6 * 1024**3` = 6,442,450,944 bytes. It is not `6 * 10**9`. The 5 GiB spare and the 2 GiB projection floor stay the Checkpoint 1.5 lines. If the guard refuses the load, the bytes that entered the decision and a fresh reading are recorded, and `from_pretrained` is not called. The guard is not lowered and the load is not retried.

## Feasible step

Canonical RGB is float32, channel-first, in `[0, 1]`. A proposed step is projected with `qwen_rgb.constraints.project_linf`. For a clean image already in `[0, 1]`, that function is the intersection of the L-infinity ball of radius epsilon around the clean image and the unit cube. The L-infinity clamp is applied to the delta and the result is clipped to `[0, 1]`. Epsilon is in RGB units on the `[0, 1]` scale.

`torch.sign` supplies the gradient sign. A zero component stays zero and does not move that component.

The official 8-bit round trip is `qwen_rgb.constraints.round_trip_uint8`. PyTorch `round` is half-to-even. The stored image is a multiple of `1/255`. The round trip can move a pixel by at most half a level, so the stored image can sit slightly outside the continuous ball. Both distances are recorded. The round trip is not differentiated.

## Stage A sample

Rule id: `per-color-extreme-alignment-exploratory`.

This sample is exploratory. The rule uses Checkpoint 1.8 alignment strata. It cannot support an unbiased population attack-success rate.

The rule reads `docs/checkpoint-1.8/qwen_rgb_loss/progress.json` and uses only `scene_id`, `answer`, and `alignment_neutral_to_helpful`. It does not open a gradient tensor and it does not run a model. Scenes whose stored alignment is missing or outside `[-1, 1]` stop the run before a forward. Within each labeled color, scenes are sorted by that alignment and then by `scene_id`. The lowest and the highest scene of each color are kept, in the color order red, blue, green, yellow, and within a color the lower alignment first.

On this commit those eight scenes are:

| Order | Scene | Label | Stratum | Stored \(A(\text{neutral}, \text{helpful})\) |
| --- | --- | --- | --- | ---: |
| 1 | `dev4-ref-01-a` | red | lowest in color | 0.2738961986673083 |
| 2 | `dev4-ref-09-b` | red | highest in color | 0.9345366800860538 |
| 3 | `dev4-ref-25-a` | blue | lowest in color | 0.4940414745458451 |
| 4 | `dev4-ref-21-b` | blue | highest in color | 0.9522294672411216 |
| 5 | `dev4-ref-30-b` | green | lowest in color | 0.5875956154399549 |
| 6 | `dev4-ref-07-a` | green | highest in color | 0.9775420055497965 |
| 7 | `dev4-ref-17-b` | yellow | lowest in color | 0.19669284166797163 |
| 8 | `dev4-ref-45-b` | yellow | highest in color | 0.9753625290227933 |

If the code that applies this rule returns a different id or color, the run stops before a forward.

Context for every Stage A step: `neutral`.

## Stage A steps

For each selected scene, compute the RGB gradient of the labeled-color loss at the clean image. Record L1, L2, and the float32 SHA-256. Compare that hash with the Checkpoint 1.8 neutral hash. A mismatch is reported. It does not by itself select a scene and it does not rewrite the Checkpoint 1.8 file. The gradient array is not committed.

Budgets, in this order: `0.5/255`, `1/255`, `2/255`, `4/255`.

At each budget, three projected steps from the clean image:

- Positive gradient-sign: add `epsilon * sign(gradient)`.
- Negative gradient-sign: subtract `epsilon * sign(gradient)`. This is a control.
- Matched random-sign: add `epsilon` times a Rademacher field on the same support as `sign(gradient)`. Where the gradient component is zero, the random component is zero. Elsewhere it is `+1` or `-1`. The field is drawn once per scene from `torch.Generator` seeded by the first 31 bits of SHA-256 of `checkpoint-1.9-stage-a|{scene_id}`, little-endian. The same field is scaled to every budget. This is a control.

Each proposed image is passed through `project_linf` at that epsilon. The primary perturbed loss is the labeled-color summed NLL of the projected float image on the differentiable path. The clean loss for that comparison is the loss from the clean backward, on the same path.

For every step record: original and perturbed loss and margin, the closed-vocabulary decision, whether that decision changes, the predicted first-order change `dot(gradient, projected - clean)` in float64, the observed change `perturbed loss - clean loss`, directional agreement by `qwen_rgb.finite_difference.same_sign` on those two scalars, and the exact L-infinity distance of the projected image.

Also record the official 8-bit round trip of the projected image: L-infinity against the clean image, L-infinity against the projected image, the fraction of pixels that change, and the labeled-color loss, margin, and decision of the stored image scored with the official processor pixels. The uint8 observed change uses the official-path clean loss as its baseline. The Stage A gate below does not read the uint8 change. A float increase that the round trip removes is reported as such.

The gradient array is discarded after the scene's twelve steps. Compact records only.

## Stage A stop rule

There are eight scenes and four budgets. A cell is one scene at one budget. A cell is valid only when the clean gradient is finite and nonzero and the three float observed changes are finite. An invalid cell does not count as a rise and does not count as beating a control. A budget fails if any of its eight cells is invalid. A missing observed change is not imputed as zero, and the median is not used to pass a budget that has a missing change.

The median of eight numbers is `numpy.median`: the arithmetic mean of the two central values after sorting.

A budget passes only when all of the following hold:

- At least 6 of the 8 scenes have positive-sign observed change strictly greater than 0.
- At least 6 of the 8 scenes have positive-sign observed change strictly greater than the negative-sign observed change.
- At least 6 of the 8 scenes have positive-sign observed change strictly greater than the matched random-sign observed change.
- The median positive-sign observed change is strictly greater than 0, strictly greater than the median negative-sign change, and strictly greater than the median random-sign change.

Stage A shows a meaningful advantage over both controls only when every budget in `{0.5, 1, 2, 4}/255` passes. Otherwise Stage A does not pass.

If the positive-sign steps fail that rule, stop and report. Do not start Stage B. The writeup looks for bfloat16 or implementation problems, including the Checkpoint 1.8 finite-difference failure, a gradient-hash mismatch, and a float increase that dies on the 8-bit round trip. Do not raise a budget. Do not change the loss. Do not lower the memory guard.

An out-of-memory error stops the run. The guard is not lowered and the load is not retried. Stage B does not start.

## Stage B sample

These sixteen scenes are fixed before any PGD step. Rule id: `scene-id-stride-6`. Sort `scene_id`. Take indexes `0, 6, 12, ..., 90`. The rule does not read alignment, margin, or a score. The subset is still a development feasibility sample. It is not a population attack-success rate.

| Order | Scene | Order | Scene |
| --- | --- | --- | --- |
| 1 | `dev4-ref-00-a` | 9 | `dev4-ref-24-a` |
| 2 | `dev4-ref-03-a` | 10 | `dev4-ref-27-a` |
| 3 | `dev4-ref-06-a` | 11 | `dev4-ref-30-a` |
| 4 | `dev4-ref-09-a` | 12 | `dev4-ref-33-a` |
| 5 | `dev4-ref-12-a` | 13 | `dev4-ref-36-a` |
| 6 | `dev4-ref-15-a` | 14 | `dev4-ref-39-a` |
| 7 | `dev4-ref-18-a` | 15 | `dev4-ref-42-a` |
| 8 | `dev4-ref-21-a` | 16 | `dev4-ref-45-a` |

Stage B runs only after Stage A passes. If Stage A does not pass, these scenes are not attacked.

## Stage B attack

Source context: `neutral`. The image is then evaluated under `neutral` and `helpful_nonrevealing`. Epsilon in `{1, 2, 4}/255`, in that order. Seven steps. Step length `epsilon / 4`. One deterministic init: the clean image. No restarts. No momentum.

Each step recomputes the labeled-color RGB gradient at the current projected image, on the same unclipped differentiable path, and applies `project_linf(current + (epsilon / 4) * sign(gradient), clean, epsilon)`.

The same clean gradient is reused for an FGSM control: one positive-sign step of length epsilon. A matched random-sign step of length epsilon, with seed SHA-256 of `checkpoint-1.9-stage-b|{scene_id}`, is the noise control. Same support rule as Stage A. If a backward runs out of memory, FGSM and the remaining PGD steps are not improvised on CPU, and epsilon is not raised.

After the last PGD step, record the projected float image and its official 8-bit round trip. Score both under both contexts with the same closed-vocabulary sums as the clean baseline. Record original and adversarial decisions, labeled NLL, margins, L-infinity before and after the round trip, runtime, memory, and errors.

Definitions, using the official 8-bit image as the stored attack:

- A scene is source-eligible when the clean neutral decision equals the labeled color.
- A scene is target-eligible when the clean helpful decision equals the labeled color.
- A source flip is a source-eligible scene whose adversarial neutral decision is not the labeled color.
- A target flip is a target-eligible scene whose adversarial helpful decision is not the labeled color. The image was optimized under neutral only.
- Transfer is a scene that is both source-eligible and target-eligible and that flips in both contexts.

Denominators are those eligibility counts. A scene with an incorrect clean answer stays in the table and stays out of the rate. Alignment and clean margins of the attacked scenes are described. They are not regressed, and alignment is not claimed to predict transfer.

PGD is ineffective at a budget when the uint8 source-flip count is 0 and the median uint8 labeled-NLL increase on source-eligible scenes is not strictly greater than both the FGSM median and the random median. If PGD is ineffective at every listed budget, stop. Do not add a larger epsilon and do not change the objective. The three listed budgets are still run, because "every listed budget" is otherwise unread. No budget outside the list is run.

## Illustration

The figure is chosen before the run, not by which scene moved most.

If Stage B runs, the figure is `dev4-ref-00-a` at epsilon `4/255` after PGD, official 8-bit image beside the clean image. If Stage B does not run, the figure is `dev4-ref-01-a` at budget `4/255`, positive-sign, official 8-bit image beside the clean image. A third panel may show the stored difference multiplied by 32 for display. That panel is not a second attack. The gain is printed on the figure.

## Records

Compact JSON only. No raw gradient tensors. Checkpoint 1.7 and 1.8 result files are not edited.

## Reproduction

From the worktree root. The unit tests do not load Qwen. The measurement command is the one that does, and only after this protocol file is already committed.

```bash
HF_HOME=/Users/fedecava/Development/models/huggingface \
PYTHONPATH=src/attack_feasibility:src/qwen_rgb_loss:src/qwen_rgb:src/choice_eval:src/vlm_iface:src/pilot:src/dev_pilot:src/analysis:src/scene_v4 \
/Users/fedecava/Development/_c0-work/venv-vlm/bin/python -m pytest tests/attack_feasibility -q

HF_HOME=/Users/fedecava/Development/models/huggingface \
PYTHONPATH=src/attack_feasibility:src/qwen_rgb_loss:src/qwen_rgb:src/choice_eval:src/vlm_iface:src/pilot:src/dev_pilot:src/analysis:src/scene_v4 \
/Users/fedecava/Development/_c0-work/venv-vlm/bin/python -m attack_feasibility
```

The second command is started only when no other study model process is resident. A saved budget refusal or an out-of-memory stop is not retried.
