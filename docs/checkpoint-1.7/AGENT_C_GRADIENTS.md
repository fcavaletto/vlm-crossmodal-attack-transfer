# Agent C — Qwen RGB gradients

Checkpoint 1.7, section C. The branch is `checkpoint-1.7/qwen-rgb`. It starts at `004783b5604c34aa066befbcd5356bfd7bd6e7b2`, not at `main`, so the pull request also contains Checkpoints 0 through 1.6. Those earlier files were not edited. `docs/checkpoint-1.6/` was not changed. Nothing was merged.

Pull request: see the GitHub pull request opened from this branch against `main`.

## Design decisions

The pinned candidate remains `Qwen/Qwen3-VL-2B-Instruct` at `89644892e4d85e24eaac8bacfd4f463576704203`. This checkpoint does not adopt it.

The official image processor resizes an 8-bit image with bicubic interpolation and antialiasing, then fuses the 1/255 rescale into the normalization constants `(0.5, 0.5, 0.5)`. The differentiable path multiplies canonical float32 RGB by 255, without rounding to 8 bits, and calls that processor's `resize`, `rescale_and_normalize`, and `patchify`. The resize stays on CPU. `smart_resize` uses patch size 16, merge size 2, and the pinned pixel limits 65536 and 16777216. One image is repeated on the temporal axis of length 2; that repeat is the processor's `expand`, not a second frame.

The path does not clip. Clipping, an L-infinity ball, and the 8-bit round trip are the future feasible set. They are pure functions in `qwen_rgb.constraints`. No epsilon was chosen. No signed step is taken, and nothing is iterated.

The loss that would have been differentiated is the summed negative log-likelihood of the full leading-space answer sequence. For this scene that sequence is the single token ` red`. The mean over tokens is not the loss. The prompt is the stored neutral context, a newline, and the stored question. This is the prospective primary reduction. It is not a scoring rule picked because it raises accuracy, and it is not Agent A's proposal.

The pre-load budget is imported from `docs/checkpoint-1.5/run_model_feasibility.py`. The 5 GiB spare, the 2 GiB projection floor, and the 6 GiB OS floor are checked at startup and were not replaced. Float32 is preferred when its budget passes. BF16 is allowed only when float32 fails, the BF16 lines pass, and a small MPS BF16 matmul has run. The model `from_pretrained` sits behind the immediate recheck. A processor gap above `1e-3` in absolute value would also refuse the load. That tolerance was fixed before the run.

The finite-difference check, if the model had loaded, would use a deterministic Rademacher direction and RGB steps `1/64` and `1/128`. Both are at least one BF16 unit in the last place at magnitude 1 (`2^-7`). The precommitted relative-error ceiling is 0.5 for BF16 and 0.05 for float32. Those ceilings were not revised, because the check did not run.

One development scene: `dev2-attribute-00-a`, context `neutral`, fixture answer `red`. No other scene image was opened.

## Work performed

The package `src/qwen_rgb/` traces the processor, compares it with the official call, defines the future RGB constraint, and runs the guarded measurement. Tests under `tests/qwen_rgb/` do not load the model. The measurement script writes `src/qwen_rgb/records/development_rgb_gradient.json`.

The script read the local safetensors header and did not download tensor bytes. It loaded the processor and tokenizer. It compared the differentiable pixels with the official processor on the scene image. It released temporary tensors and rechecked the budget. The recheck failed. `Qwen3VLForConditionalGeneration.from_pretrained` was not called.

## Tests

From the worktree root, with the study environment:

```bash
HF_HOME=/Users/fedecava/Development/models/huggingface \
PYTHONPATH=src/qwen_rgb:src/vlm_iface:src/pilot \
/Users/fedecava/Development/_c0-work/venv-vlm/bin/python -m pytest tests/qwen_rgb -q
```

Result on this machine: 17 passed in 1.62 seconds. The tests cover the pinned preprocessor config, exact agreement on the development image, the uint8 resize gap when the size changes, a finite nonzero RGB gradient through patchify, summed NLL versus the mean, mask extension, the L-infinity and 8-bit maps, the precommitted finite-difference steps on a linear function and on a BF16 rounding staircase, and the unchanged Checkpoint 1.5 byte lines. The model call is behind the `permitted` check. The package source does not define FGSM or PGD.

## Results

Recorded run: `src/qwen_rgb/records/development_rgb_gradient.json`. Status `stopped_budget`. Wall clock 1.453 seconds. Apple M4, 25769803776 bytes of unified memory, macOS 26.6, Python 3.12.14, torch 2.15.0, transformers 5.22.1. The MPS BF16 probe multiplied an 8×8 matrix and returned a finite result. The header has 2127532032 parameters, all BF16, in a file of 4255140312 bytes. Weights were not loaded into the model class.

| Reading | OS available bytes | BF16 required | Cleared |
| --- | ---: | ---: | --- |
| Start of the saved run | 10914119680 | 11509447200 | no |
| After releasing temporary tensors | 10786570240 | 11509447200 | no |

The binding line is the calibrated projection, 11509447200 bytes. The saved start was short by 595327520 bytes. The recheck was short by 722876960 bytes. Garbage collection returned 8863744 bytes, which did not change the decision. Float32 requires 20871410752 bytes and failed the spare, projection, and staging lines. `model_load_started` is false. `model_from_pretrained_called` is false.

The chosen-answer RGB gradient was not computed. `rgb_gradient_finite` is null. The directional finite-difference check of that loss was not run.

The processor comparison did run, on `fixtures/development_v2/images/dev2-attribute-00-a.png`. The image is 384×512. `smart_resize` keeps 384×512. The grid is `[[1, 24, 32]]`. `pixel_values` are `[768, 1536]`. The two temporal copies are equal. Against the official PIL call, the uint8 tensor call, and the float 0–255 call, the maximum absolute gap is 0. The `[0, 1]` call with `do_rescale=False` differs by at most `4.470348358154297e-08`. Casting the official float32 pixels to BF16 and back differs by at most `0.0016237497329711914` (mean `0.0006181860808283091`). That cast is what the vision tower applies; it was measured without the model.

A forced bicubic resize of the same pixels to 256×320, which is not the model input, leaves a uint8-versus-float gap of `0.00795203447341919` in normalized patches. Torchvision's uint8 resize rounds. The float resize does not. This scene does not exercise that rounding, because the official resize is the identity.

The leading-space answer is token 2518, the piece ` red`. The bare word is token 1151, `red`. The loss would have used 2518. No probability comparison was used to make that choice.

A separate backward, through the sum of `pixel_values` only, is finite and nonzero at every RGB element. The gradient is the constant 4, which is the normalization scale `255/127.5` times the two temporal copies. That is the preprocessor Jacobian. It is not the gradient of the chosen-answer loss, and the record marks it as such.

An earlier process in this same session, whose JSON was overwritten by the saved run, saw 11561254912 available bytes. That reading cleared the BF16 line by 51807712 bytes. Its immediate recheck then saw 11157946368 and refused the load. `from_pretrained` was not called in that process either. The saved file is the later refusal.

## Limitations

The question this checkpoint asked of the model — whether the chosen-answer loss has a finite nonzero gradient with respect to the original RGB image — is unanswered. The finite-difference agreement under BF16 is also unanswered.

The exact processor match is for an image whose resize is the identity. A scene that `smart_resize` actually changes would carry the uint8 rounding gap. The differentiable path would then not be bit-identical to the PIL path. The `1e-3` stop would still have to be applied; the probe gap of about `0.008` is larger than that stop.

The future 8-bit round trip is detached. A stored PNG is that map, and the gradient does not include the rounding. The L-infinity projection is defined and not applied. No study epsilon exists yet.

The summed NLL uses the leading-space fixture word. If the adopted scoring sequence is different, this loss is the wrong one to differentiate.

The machine reading moves by more than the margin above the BF16 line. A later retry needs a new OS reading. It does not need a lower line.

## Unexpected findings

`tvF.resize` to the same height and width returns the same tensor. The graph still reaches RGB because the scaling and normalization are downstream arithmetic.

On this image the float path and the official PIL path match exactly. The gap that matters for a changed size is the uint8 rounding inside bicubic resize, not a difference in patch layout.

The fused `[0, 1]` normalization and the 0–255 fused normalization differ only at a few units in the last place of float32 when the resize is an identity.

Releasing Python garbage before the recheck recovered about 8.5 MiB. The budget decision did not move.

## Unresolved questions

1. Is the chosen-answer RGB gradient finite and nonzero when the unchanged BF16 budget permits a load? The recommendation is to rerun the same script when OS-available memory clears 11509447200 bytes at the recheck. Do not lower the line.
2. If that gradient is finite, does the precommitted directional check confirm it at relative error 0.5 under BF16? The steps are already fixed.
3. Should the attack path's reference image be one whose `smart_resize` changes the size, so the uint8 gap is part of the model input rather than only a probe? This scene hides that gap.
4. What L-infinity epsilon, in `[0, 1]` RGB units, should the future feasible set use? This checkpoint does not choose one.
5. If Agent A's candidate sequences differ from the single leading-space color token, the loss differentiated here has to be replaced before any attack loss is claimed.

## Reproduction

Processor tests, no model load:

```bash
HF_HOME=/Users/fedecava/Development/models/huggingface \
PYTHONPATH=src/qwen_rgb:src/vlm_iface:src/pilot \
/Users/fedecava/Development/_c0-work/venv-vlm/bin/python -m pytest tests/qwen_rgb -q
```

Guarded measurement. It rechecks the Checkpoint 1.5 budget and returns before the model load when the line fails. It does not download weights and it does not run an attack.

```bash
HF_HOME=/Users/fedecava/Development/models/huggingface \
PYTHONPATH=src/qwen_rgb:src/vlm_iface:src/pilot \
/Users/fedecava/Development/_c0-work/venv-vlm/bin/python -m qwen_rgb.measure
```

The command is the one used on this Mac. The interpreter is `/Users/fedecava/Development/_c0-work/venv-vlm/bin/python`. The Hub cache is `/Users/fedecava/Development/models/huggingface`.

## Git

- Branch: `checkpoint-1.7/qwen-rgb`
- Started from: `004783b5604c34aa066befbcd5356bfd7bd6e7b2`
- Base for the pull request: `main`
- The diff against `main` includes earlier checkpoints because this branch does not start there
- Pull request: opened from this branch; the URL is in the pull request that contains this file
- Chosen-answer RGB gradient finite: not measured. The budget stopped the load.
