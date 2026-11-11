# Checkpoint 1.8 measurement protocol

Status: fixed before any development_v4 gradient from this checkpoint. Research lead: Federico Cavaletto. Label of every number this run produces: DEVELOPMENT ONLY.

This file does not replace `docs/checkpoint-1.7/DEVELOPMENT_PROTOCOL.md`. It says which quantity is differentiated and which four scenes are the numerical gate. It does not change the decision rule, the prompts, the token ids, the model revision, or the memory lines.

No attack is run. No signed step is taken. H* is not fit. Alignment is not regressed on an attack outcome. Held-out images are not opened. SmolVLM2 is not loaded.

## Model

`Qwen/Qwen3-VL-2B-Instruct`, revision `89644892e4d85e24eaac8bacfd4f463576704203`. Processor and answer prefix are the Checkpoint 1.7 lock. The answer prefix is `<|im_end|>\n<|im_start|>assistant\n`. The cue ends in a newline, so the continuation is the bare color word.

| Color | Token id |
| --- | ---: |
| red | 1151 |
| blue | 12203 |
| green | 13250 |
| yellow | 27869 |

Token 2518 (` red`) is not the continuation. A mismatch of the cue, the ids, or a loaded commit hash stops the run before a development gradient.

Device is MPS. One process calls the model load at most once, and only through `qwen_rgb.measure.load_model_after_budget` after `qwen_rgb.budget` permits a dtype. The 6 GiB OS floor is `6 * 1024**3` = 6,442,450,944 bytes. It is not `6 * 10**9`. The 5 GiB spare and the 2 GiB projection floor stay the Checkpoint 1.5 lines. If the guard refuses the load, the bytes that entered the decision and a fresh reading are recorded, and `from_pretrained` is not called. The guard is not lowered and the load is not retried.

## Loss

The scalar is the summed teacher-forced negative log-likelihood of the labeled color, read from the first continuation logit. That is the Checkpoint 1.7 scoring path. The mean token NLL is stored and is not differentiated. The derivative is with respect to canonical float RGB in `[0, 1]`, through `qwen_rgb.rgb_path.differentiable_model_pixels`. The path is not clipped.

On every differentiated scene, three formulas on that same logit row must agree within `1e-5`: `choice_eval.summed_nll`, a direct negative log-softmax, and `vlm_iface.logprob.sequence_logprob` on the same row with an unused final logit. If they disagree, backward is not run and the remaining scenes are not started.

The official processor is compared with the differentiable path on every development_v4 scene. Every gap is recorded. A gap does not change the loss and does not select a scene.

The stored Checkpoint 1.7 Qwen rows are read and not rewritten. The live decision is the same argmin of summed NLL, with color-name ties. A numeric gap against those stored sums is reported. It does not change the rule.

## Four-scene gate

Rule id: `lexicographic-first-scene-per-labeled-color`.

Sort `scene_id`. For each labeled color in the order red, blue, green, yellow, take the first scene with that answer. The rule reads `scene_id` and `answer` only. On the locked manifest this is:

| Color | Scene |
| --- | --- |
| red | `dev4-ref-00-a` |
| blue | `dev4-ref-00-b` |
| green | `dev4-ref-01-b` |
| yellow | `dev4-ref-14-b` |

The gate passes only when each of these four scenes has a finite, nonzero RGB gradient under both `neutral` and `helpful_nonrevealing`. A missing gradient is not entered as zero. A zero gradient fails the gate. If the gate does not pass, the other scenes are not differentiated.

Two further checks run on those four scenes and do not, by themselves, fail the gate:

- The gradient is repeated. The repeat confirms stability when both copies are finite and nonzero and their relative L2 is at most the documented BF16 ceiling 0.5. A larger gap is a numerical failure. A non-finite or zero copy is an invalid gradient.
- A directional finite difference uses `qwen_rgb.finite_difference`: Rademacher direction, seed 17, steps `1/64` and `1/128`, and `ceiling_for_dtype`. One agreeing step confirms. A miss is a numerical failure, not an invalid gradient, when the analytic gradient is finite and nonzero.

An appended-token forward on the gate scenes is the same kind of numerical check, with the same BF16 relative ceiling. The backward itself uses the prompt-only first logit.

## Full sample and eligibility

If the gate passes, the same loss is differentiated for all 96 scenes and both primary contexts. Progress is written after each scene so a later process can resume without a different rule. Scratch arrays used only to finish a scene are not part of the commit. The commit stores compact records and SHA-256 hashes of float32 gradients.

For each finite gradient, store L1 and L2. For each scene with both gradients, store `A(neutral, helpful_nonrevealing)` and `A(helpful_nonrevealing, neutral)` from `analysis.metrics.sign_alignment`. An undefined alignment is null. It is not filled in as zero. The distribution, not only the mean, is reported for both directions and for the clean margin. The clean margin is the best other summed NLL minus the labeled summed NLL.

The eligible-group count uses `dev_pilot.summarize.eligible_groups` and the untouched Checkpoint 1.7 rows. The alignment scalar that function reads is `A(neutral, helpful_nonrevealing)`, and only when both directed alignments are finite and inside `[-1, 1]`. An incomplete pass leaves the count unset. A finished pass can still exclude a group. The count is not a coefficient. H* is not fit.

## Reproduction

From the worktree root:

```bash
HF_HOME=/Users/fedecava/Development/models/huggingface \
PYTHONPATH=src/qwen_rgb_loss:src/qwen_rgb:src/choice_eval:src/vlm_iface:src/pilot:src/dev_pilot:src/analysis:src/scene_v4 \
/Users/fedecava/Development/_c0-work/venv-vlm/bin/python -m pytest tests/qwen_rgb_loss -q

HF_HOME=/Users/fedecava/Development/models/huggingface \
PYTHONPATH=src/qwen_rgb_loss:src/qwen_rgb:src/choice_eval:src/vlm_iface:src/pilot:src/dev_pilot:src/analysis:src/scene_v4 \
/Users/fedecava/Development/_c0-work/venv-vlm/bin/python -m qwen_rgb_loss
```

The second command is the measurement. It is started only when no other study model process is resident.
