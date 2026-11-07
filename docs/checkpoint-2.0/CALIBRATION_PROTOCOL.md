# Checkpoint 2.0 calibration protocol

Status: fixed before any perturbation from this checkpoint. Research lead: Federico Cavaletto. Label of every number this run produces: DEVELOPMENT ONLY.

This file does not replace `docs/checkpoint-1.7/DEVELOPMENT_PROTOCOL.md`, `docs/checkpoint-1.8/MEASUREMENT_PROTOCOL.md`, or `docs/checkpoint-1.9/ATTACK_PROTOCOL.md`. It does not change the decision rule, the prompts, the token ids, the model revision, or the memory lines. It does not rewrite Checkpoint 1.7 predictions, Checkpoint 1.8 gradient records, or Checkpoint 1.9 attack records.

No H* fit. No held-out scene. No training. No merge to main. SmolVLM2 is not loaded. The sample below is exploratory. It is not a population attack-success rate.

Started from Checkpoint 1.9 head `0cd2f2a8e566b6530dcfd4dd7021961b2fc17990`. The Checkpoint 1.9 protocol commit remains `fce3480d158298851587a692c56b6e4eebb554a7`.

## Model and loss

`Qwen/Qwen3-VL-2B-Instruct`, revision `89644892e4d85e24eaac8bacfd4f463576704203`. Bfloat16 on MPS. Eager attention. The processor and the answer prefix are the Checkpoint 1.7 lock. The answer prefix is `<|im_end|>\n<|im_start|>assistant\n`. The cue ends in a newline, so the continuation is the bare color word.

| Color | Token id |
| --- | ---: |
| red | 1151 |
| blue | 12203 |
| green | 13250 |
| yellow | 27869 |

Token 2518 (` red`) is not the continuation. A mismatch of the cue, the ids, or a loaded commit hash stops the run before a perturbation.

The scalar is the summed teacher-forced negative log-likelihood of the labeled color, read from the first continuation logit. The mean token NLL is not the step objective. On the clean backward, the Checkpoint 1.8 three-formula identity on that logit row has to agree within `1e-5`. If the formulas disagree, no step is taken.

The derivative is with respect to canonical float RGB in `[0, 1]`, through `qwen_rgb.rgb_path.differentiable_model_pixels`. The backward path is not clipped. A missing, non-finite, or zero clean gradient is not a step. It is not entered as a zero gradient.

Workstream A loads bfloat16 when the bfloat16 budget and the MPS probe pass. Float32 is not loaded on this workstream, including when its own budget would pass. The Checkpoint 1.5 lines are unchanged: bfloat16 requires 11,509,447,200 available bytes; float32 requires 20,871,410,752; the 6 GiB floor is `6 * 1024**3` = 6,442,450,944. The 5 GiB spare and the 2 GiB projection floor stay. If the guard refuses the load, the bytes that entered the decision and a fresh reading are recorded, and `from_pretrained` is not called. A budget refusal is terminal for that reading. The same process does not take a second reading. The guard is not lowered.

Device is MPS. A CPU forward is not recorded as an MPS result.

## Sample

Rule id: `counterfactual-group-stride-3`.

The rule reads the development_v4 manifest and uses `counterfactual_group_id` and `scene_id` only. It does not read alignment, margin, loss, a gradient, or a Checkpoint 1.9 outcome. Sort the unique group ids lexicographically. When that list has length 48, take indexes 0, 3, 6, ..., 45. Include both arms of each selected group, ordered by `scene_id`. If the group count is not 48, stop before any attack and record the count. Do not invent a replacement index list.

On this manifest the group count is 48. The sixteen groups and thirty-two scenes are:

| Order | Group | Scene | Label | Random-sign seed |
| --- | --- | --- | --- | ---: |
| 1 | `cf4-ref-00` | `dev4-ref-00-a` | red | 1342106899 |
| 2 | `cf4-ref-00` | `dev4-ref-00-b` | blue | 974576453 |
| 3 | `cf4-ref-03` | `dev4-ref-03-a` | red | 131700877 |
| 4 | `cf4-ref-03` | `dev4-ref-03-b` | blue | 79946813 |
| 5 | `cf4-ref-06` | `dev4-ref-06-a` | blue | 1250892028 |
| 6 | `cf4-ref-06` | `dev4-ref-06-b` | red | 2020182026 |
| 7 | `cf4-ref-09` | `dev4-ref-09-a` | blue | 473034008 |
| 8 | `cf4-ref-09` | `dev4-ref-09-b` | red | 771346531 |
| 9 | `cf4-ref-12` | `dev4-ref-12-a` | blue | 2074605998 |
| 10 | `cf4-ref-12` | `dev4-ref-12-b` | green | 1845135049 |
| 11 | `cf4-ref-15` | `dev4-ref-15-a` | blue | 701364960 |
| 12 | `cf4-ref-15` | `dev4-ref-15-b` | green | 2037154491 |
| 13 | `cf4-ref-18` | `dev4-ref-18-a` | green | 708532238 |
| 14 | `cf4-ref-18` | `dev4-ref-18-b` | blue | 1440290221 |
| 15 | `cf4-ref-21` | `dev4-ref-21-a` | green | 666959696 |
| 16 | `cf4-ref-21` | `dev4-ref-21-b` | blue | 578873171 |
| 17 | `cf4-ref-24` | `dev4-ref-24-a` | green | 1598181378 |
| 18 | `cf4-ref-24` | `dev4-ref-24-b` | yellow | 1679499347 |
| 19 | `cf4-ref-27` | `dev4-ref-27-a` | green | 167202381 |
| 20 | `cf4-ref-27` | `dev4-ref-27-b` | yellow | 1733500241 |
| 21 | `cf4-ref-30` | `dev4-ref-30-a` | yellow | 2054639494 |
| 22 | `cf4-ref-30` | `dev4-ref-30-b` | green | 2103363729 |
| 23 | `cf4-ref-33` | `dev4-ref-33-a` | yellow | 1664993314 |
| 24 | `cf4-ref-33` | `dev4-ref-33-b` | green | 1652986261 |
| 25 | `cf4-ref-36` | `dev4-ref-36-a` | yellow | 1445696036 |
| 26 | `cf4-ref-36` | `dev4-ref-36-b` | red | 1114441199 |
| 27 | `cf4-ref-39` | `dev4-ref-39-a` | yellow | 755236030 |
| 28 | `cf4-ref-39` | `dev4-ref-39-b` | red | 463988851 |
| 29 | `cf4-ref-42` | `dev4-ref-42-a` | red | 1529302010 |
| 30 | `cf4-ref-42` | `dev4-ref-42-b` | yellow | 659569850 |
| 31 | `cf4-ref-45` | `dev4-ref-45-a` | red | 512862953 |
| 32 | `cf4-ref-45` | `dev4-ref-45-b` | yellow | 234080788 |

If the code that applies the rule returns a different group id, scene id, or label, the run stops before a forward.

## Attack

Source context: `neutral`. Helpful text is an evaluation context. It is not a second attack. The image is optimized under neutral only.

Evaluation contexts for every recorded image: `neutral` and `helpful_nonrevealing`, after the official 8-bit round trip. The official image is `qwen_rgb.constraints.round_trip_uint8`, PyTorch half-to-even rounding onto multiples of `1/255`. The decision is the lowest summed NLL among the four colors. Attack success is conditional on a correct clean answer. A scene with an incorrect clean answer stays in the table and stays out of the rate. A missing decision, margin, or NLL stays missing. It is not filled in as a failure and it is not filled in as a success.

Epsilons: `1/255` and `2/255` only. The step is projected with `qwen_rgb.constraints.project_linf` onto the L-infinity ball around the clean image intersected with `[0, 1]`.

PGD uses `attack_feasibility.steps.pgd_update`. Seven steps. Step length `epsilon / 4`. The init is the clean image. One deterministic initialization. No restarts. No momentum. Each step recomputes the labeled-color RGB gradient at the current projected image, except the first step, which uses the clean gradient. The recorded PGD images are the official 8-bit images after steps 2, 3, 4, 5, and 7. Steps 1 and 6 are taken because the later steps depend on them. They are not separate recorded configurations.

Controls at the same epsilon, from the clean image, using the clean gradient:

- Positive sign step (FGSM): one step of length epsilon, `attack_feasibility.steps.project_signed_step` in mode `positive`.
- Matched random-sign step: one step of length epsilon. The field is a Rademacher draw on the support of `sign(gradient)`. Where the clean gradient component is zero, the random component is zero. Elsewhere it is `+1` or `-1`. The generator seed is the first 31 bits of SHA-256 of `checkpoint-2.0-calibration|{scene_id}`, little-endian, reduced with `attack_feasibility.steps.scene_seed`. One field is drawn per scene and scaled to both budgets. The integers in the sample table are those seeds. They were computed from the scene ids before any gradient.

The gradient array is discarded after the scene. Compact records only. No raw image tensors are committed.

For each scene, budget, recorded step, and control, the record holds: the clean decision, margin, and labeled NLL; the perturbed decision, margin, and labeled NLL under both evaluation contexts; source flip; target flip; transfer; the exact L-infinity distance of the projected image to the clean image before the 8-bit round trip; the exact L-infinity distance of the official image to the clean image after the round trip; runtime; memory; and errors.

## Counts

These counts describe the recorded configurations. They do not choose a further attack.

- A scene is source-clean-correct when the official clean neutral decision equals the labeled color.
- Source success is a source-clean-correct scene whose official perturbed neutral decision is not the labeled color. The source-success denominator is the number of source-clean-correct scenes with a recorded perturbed neutral decision. Scenes with a missing perturbed decision are counted apart from that denominator.
- A scene is target-clean-correct when the official clean helpful decision equals the labeled color.
- A target flip is a target-clean-correct scene whose official perturbed helpful decision is not the labeled color.
- Conditional transfer follows `analysis.metrics.conditional_transfer`. The denominator is the number of source successes whose clean helpful decision equals the labeled color and whose perturbed helpful decision was recorded. The numerator is how many of those helpful decisions are not the labeled color. Source successes whose clean helpful decision is not the labeled color are excluded and counted. They are not transfer failures. A missing helpful decision on a source success stays missing and is counted apart from the denominator.

Near the floor means the numerator is 0 or 1. Near the ceiling means the numerator is the denominator or one less. The denominator has to be at least 2; otherwise the position is undefined. Adequate source success means the source-success count is interior or near the ceiling. Variation in conditional transfer means the conditional-transfer count is interior. A configuration meets both descriptions only when both are true. Near-ceiling source success still counts as adequate. The report lists every configuration that meets both, with the counts shown. It does not rank them by alignment, margin, or gradient L1, and it does not drop a configuration to make the list shorter. If none meet both, the report says the recorded grid is at a floor or a ceiling.

The report also prints the stored Checkpoint 1.8 alignment `A(neutral, helpful)`, the stored neutral margin, the stored helpful margin, and the stored neutral gradient L1 for the attacked scenes. Those columns are descriptive. They are not a regression, and they are not an input to the sample or to the configuration list.

## Stop

The listed budgets and recorded steps are the whole grid. Finishing them stops the run. Epsilon stays inside `{1/255, 2/255}`. The PGD length stays seven. The objective stays the summed labeled-color NLL under neutral.

An out-of-memory error stops the run. The guard is not lowered, the load is not retried, and the remaining steps are not moved to CPU. A group count other than 48 stops the run before a perturbation.

## Reproduction

From the worktree root. The unit tests do not load Qwen. The measurement command is the one that does, and only after this protocol file is already committed.

```bash
HF_HOME=/Users/fedecava/Development/models/huggingface \
PYTHONPATH=src/attack_calibration:src/attack_feasibility:src/qwen_rgb_loss:src/qwen_rgb:src/choice_eval:src/vlm_iface:src/pilot:src/dev_pilot:src/analysis:src/scene_v4 \
/Users/fedecava/Development/_c0-work/venv-vlm/bin/python -m pytest tests/attack_calibration -q

HF_HOME=/Users/fedecava/Development/models/huggingface \
PYTHONPATH=src/attack_calibration:src/attack_feasibility:src/qwen_rgb_loss:src/qwen_rgb:src/choice_eval:src/vlm_iface:src/pilot:src/dev_pilot:src/analysis:src/scene_v4 \
/Users/fedecava/Development/_c0-work/venv-vlm/bin/python -m attack_calibration
```

The second command is started only when no other study model process is resident. A saved budget refusal is not reread inside the process that recorded it.
