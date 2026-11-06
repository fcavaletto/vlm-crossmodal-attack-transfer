# Agent and responsibility

Agent C, model interface, branch `checkpoint-0/model-gradients`.

Owned paths: `src/vlm_iface/**`, `tests/vlm_iface/**`, `docs/models/**`, and this
report. The work is the scoring interface for Checkpoint 0: joint answer
log-probabilities, the input-image gradient of their negative, a parser that
can mark a generation invalid, and a minimal MPS feasibility run on
`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`. No attack was implemented.

# Executive summary

The interface loads SmolVLM2-500M-Video-Instruct at revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`, the commit Study 1 froze. On this MacBook Air M4 with 24 GB unified memory, PyTorch 2.15.0 ran the float32 eager model on MPS. The joint log-probability of the two-token answer "red square" was -19.2657, and the library mean token loss was exactly that value divided by 2. Gradients of `L = -log P(y | x, c)` on two synthetic images were finite, nonzero at every pixel, identical when the same image was repeated, and different across images (maximum absolute difference 4.78). One backward took about 4.4 seconds. The MPS driver cache reached 4131.7 MB. Qwen3-VL-2B-Instruct was not loaded and is not adopted: its float32 weights would be 8.51 GB, above a 6 GiB guard. Scoring and generation share a float32 preprocess that does not match the official uint8 resize (mean absolute gap about 0.008). No attack, training run, or held-out set was used.

# Work completed

- Package `vlm_iface` with `answer_logprob`, `image_loss_gradient`, and `generate`, plus an explicit `VLMSession` so tests do not download the 500M weights.
- Teacher-forced sum of token log-probabilities, checked against the Transformers mean loss on the real model.
- Differentiable float32 preprocess on CPU, shared by scoring, gradients, and generation, with a measured gap against the uint8 processor path.
- Leading-space answer tokenization, with a measured comparison against the bare token on one probe.
- Short-answer parser. Truncation, empty text, line breaks, refusals, and over-long strings are invalid. Invalid is not treated as attack success.
- Unit tests for tokenization, parsing, image conversion, preprocessing, determinism, numerical validity of the loss gradient, device resolution, and the Qwen weight guard.
- MPS feasibility script. The raw record is `docs/models/feasibility-smolvlm2-mps.json`.
- Header-only survey of `Qwen/Qwen3-VL-2B-Instruct`. The weight file was not downloaded.

Study 1 was read and not modified. Patterns reused, and remeasured, were the pinned revision, float32 eager attention, `do_image_splitting=False`, `do_rescale=False` on `[0, 1]` inputs, and leading-space answer pieces. The loss here is the sum. Study 1 used the mean.

# Key findings

Evidence from the MPS script on 6 Nov 2026. The machine was an Apple M4 MacBook Air with 25769803776 bytes of unified memory. The requested device, the resolved device, and the parameter device were all `mps`. The dtype was float32. Attention was eager.

The Hub sha of the loaded revision matched `7b375e1b73b11138ff12fe22c8f2822d8fe03467`. The safetensors header of that commit contains 507,482,304 float32 parameters and 2,029,929,216 tensor bytes.

For the prompt "What color is the central square? Reply with one word." and the answer "red square" on a synthetic red square, the joint log-probability was -19.265708923339844 over tokens `Ġred` and `Ġsquare`. The Transformers mean NLL on the same logits was 9.632854461669922. The absolute difference between that mean and `-logprob / 2` was 0.

On the same image, the leading-space log-probability of "red" was -3.3405706882476807 and the bare-token log-probability was -11.636720657348633.

Gradients with respect to the 128×128 RGB image:

| Image | Seconds | Mean abs | Max abs | Nonzero fraction |
| --- | ---: | ---: | ---: | ---: |
| Red square | 4.594 | 0.02935 | 2.349 | 1 |
| Red square, repeat | 4.357 | 0.02935 | 2.349 | 1 |
| Blue square | 4.388 | 0.04353 | 4.761 | 1 |

The repeat maximum absolute difference was 0. The red-versus-blue maximum absolute difference was 4.779532432556152. Load time was 4.717 seconds. The script wall time was 24.177 seconds, including Hub header reads, both preprocess comparisons, scoring, three gradients, and one generation.

Memory counters, which are not interchangeable:

- OS available before load: 8,435,777,536 bytes.
- After load: MPS current 1965.6 MB, MPS driver 2856.5 MB, OS available 3826.9 MB.
- After the gradients: MPS current 1965.6 MB, MPS driver 4131.7 MB, process peak RSS 602.0 MB.

The current allocation after the graph is freed is the weight storage. The driver number is a cache high-water. This PyTorch build exposes no separate MPS peak-activation counter. Process RSS does not contain that Metal cache.

Float32 versus uint8-PIL preprocess, splitting off, normalized pixel space: maximum absolute gap 0.7908 and mean 0.00789 on a 96×128 seed-0 image; maximum 0.2664 and mean 0.00885 on a 512×512 seed-0 image. A uint8 tensor matched PIL exactly on both. Shapes were `(1, 1, 3, 512, 512)`.

One greedy generation on the red square returned the string `Red.`, which the parser accepted as `red`. That is a smoke check, not a clean-accuracy result.

Qwen3-VL-2B-Instruct revision `89644892e4d85e24eaac8bacfd4f463576704203` has 2,127,532,032 BF16 parameters in a 4,255,140,312 byte file. The float32 expansion is 8,510,128,128 bytes. The survey read 76,240 header bytes. The weight tensor bytes were not downloaded.

Proposal, not a finding: keep the sum log-probability, the leading-space rule, the shared float32 preprocess, and image splitting off. Do not adopt Qwen on the evidence above.

# Evidence

- `docs/models/feasibility-smolvlm2-mps.json` is the script output. `passed` is true and `mps_validation` is `passed`.
- `docs/models/measured-environment.txt` lists the builds used: Python 3.12.14, torch 2.15.0, torchvision 0.29.2, transformers 5.22.1.
- `pytest tests/vlm_iface -q` reported `38 passed` in 4.71 seconds. Those tests do not load the 500M weights. They cover the sum-versus-mean identity, a central difference of `L = -log P` on a closed-form score, gradient flow through the real processor, exact repeat on the CPU probe, and the parser.
- The Qwen Hub directory on this machine contains the config from the header survey and does not contain `model.safetensors`.
- Study 1 sources used as lessons, not copied into this repo: `docs/freeze-config.md`, `docs/REPRODUCIBILITY.md`, `tests/test_gradient_gate.py`, and the gradient-gate implementation. Study 1 was not modified.

# Design decisions and rationale

The loss is the sum of token log-probabilities. That is the definition `L = -log P(y | x, c)` in the checkpoint contract. Study 1's mean is recoverable by dividing by `T`. The MPS run checked that division against `model.loss_function` and the difference was 0. Using the mean instead would scale every gradient by `1/T` and would change comparisons across answers of different lengths.

The answer is tokenized with a leading space. On the probe, that id was about 8 nats more probable than the bare id. The bare id remains available by setting `leading_space=False`, for a later audit, and is not the default.

Preprocessing for the gradient is the same function used to score and to generate. The alternative, differentiating a float path while evaluating through PIL, is what Study 1 did, and it leaves a pixel gap inside the loss/eval pair. The gap to PIL is still real, and it is written down rather than removed. A straight-through uint8 estimator would match the forward PIL values and would not be the derivative of those values. It was not used.

Resize stays on CPU. The installed torchvision replaces Lanczos with bicubic when the tensor is on CUDA or MPS or when torchvision is older than 0.27. Keeping the leaf on CPU avoids that second kernel. The remaining float-versus-uint8 gap is then the one in the table above.

Image splitting is off. The processor default is on, and with the longest-edge setting of 2048 it would tile a small image into many 512 crops. Study 1 froze the off setting after finding splitting expensive on this class of machine. This checkpoint did not remeasure the on setting.

The model is float32 and eager, weights frozen, `eval` mode. That matches the Study 1 freeze and keeps dropout out of the gradient. CUDA is implemented as a device option and was not timed.

The Qwen load is refused by the model-id guard and, independently, by a weight-size guard at 6 GiB of float32 parameters. The 6 GiB line is a safety policy for 24 GB unified memory, not a measured failure. The header already puts the float32 weights at 7.93 GiB before activations.

The parser is open-ended because this study's answers are not a single global vocabulary. The 4-token cap is a placeholder until those strings are frozen.

# Known limitations

The float32 path and the official uint8 path differ by a mean absolute amount of about 0.008 in normalized space, and by a maximum of 0.79 on the smaller reference image. Any later comparison that mixes this gradient with a PIL forward is not using one preprocess.

The real 500M gradient was not checked with a finite difference. The identity `g = ∇(-log P)` was checked on a tiny probe and on a closed-form score that use the same code path.

Gradient entries reached a maximum absolute value of 4.76 on the blue square. That magnitude is reported and not interpreted. It should not be read as a recommended attack step size.

There is no MPS peak-activation counter in torch 2.15.0 on this machine. The 4131.7 MB driver figure is a cache high-water after the run. Process RSS of 602 MB is not the model footprint.

The repeat match of 0 is one pair of calls on one image. It is not a general determinism proof for every prompt.

Image splitting, batches, CUDA, and Qwen inference were not measured. The single string `Red.` is not an accuracy estimate. The parser will mark a long spatial answer invalid. The chat template still builds a uint8 image in order to lay out tokens; those pixels are discarded after the shape check.

# Blockers

Blocking: none. The MPS run completed on the requested device.

Non-blocking: the float/uint8 preprocess gap has to travel with any later evaluation. The parser cap may need to change once answer strings exist. Qwen has no empirical score on this machine. CUDA has no timing.

# Decisions requested from the research lead

1. Confirm the sum, `L = -log P(y | x, c)`, rather than Study 1's mean token NLL. Recommendation: keep the sum, and store `T` beside every loss so the mean can be rebuilt. The mean matches the sum only up to a constant when all answers have equal length. Work can continue with the sum.
2. Confirm leading-space answer tokens as the default. Recommendation: keep them. On the probe they were much more probable than the bare tokens, which matches the chat template ending at `Assistant:`. Work can continue.
3. Confirm that scoring and generation stay on the float32 path, with the uint8 gap documented rather than hidden. Recommendation: keep that split of responsibilities. A straight-through uint8 path would change the meaning of the gradient. Work can continue.
4. Confirm `do_image_splitting=False` for the primary model. Recommendation: keep it off until there is a memory budget for the 2048-edge tiling. The on setting was not measured. Work can continue.
5. Qwen3-VL-2B-Instruct. Recommendation: do not adopt it at this checkpoint. The float32 weight file alone is past the guard, and there is no accuracy or gradient evidence. A later checkpoint could reconsider a lower-precision load only with an explicit memory budget. Work on SmolVLM2 can continue.
6. Parser length cap of 4 tokens and 32 characters. Recommendation: leave it until the scene answers are frozen, then set the cap from the longest frozen answer. Until then, long strings come back invalid instead of being guessed. Work can continue.
7. `max_new_tokens=16`, against Study 1's 12. Recommendation: keep 16 while answers may be short phrases. It did not affect the one-token smoke generation, which stopped on an end token. Work can continue.

# Reproduction instructions

From a clone of this branch, on the MPS machine:

```
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e 'src/vlm_iface[dev]'
python -m pip install 'torch==2.15.0' 'torchvision==0.29.2' 'transformers==5.22.1'
python -m pytest tests/vlm_iface -q
python -m vlm_iface.feasibility --device mps --output docs/models/feasibility-smolvlm2-mps.json
```

The feasibility command is also the command to run if a future environment cannot use MPS. In that case the script must not be pointed at CPU and then described as MPS. If MPS is missing, `resolve_device("mps")` raises and the script writes `mps_validation: PENDING`. This run did not take that branch.

Unit tests download the processor and tokenizer for the pinned revision. They do not download the weight file. The feasibility script does load the weights.

# Git information

- Branch: `checkpoint-0/model-gradients`
- Measurement commit: recorded in the commit that adds the pull request URL below.
- Pull request: recorded in that same commit.
- Tests: `38 passed` in 4.71 seconds with `python -m pytest tests/vlm_iface -q` on Python 3.12.14. The MPS feasibility script exited 0 with `passed: true` and `mps_validation: passed`.
