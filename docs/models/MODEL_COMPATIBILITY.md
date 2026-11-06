# Model compatibility

Checkpoint 0 interface for answer log-probabilities and input-image gradients.
The numbers below come from one MPS run recorded in
[feasibility-smolvlm2-mps.json](feasibility-smolvlm2-mps.json). The package
versions of that run are in [measured-environment.txt](measured-environment.txt).

## Contract

Import `vlm_iface`.

| Function | Returns |
| --- | --- |
| `answer_logprob(image, prompt, answer_text)` | Python float, the joint log-probability of the full answer |
| `image_loss_gradient(image, prompt, answer_text)` | CPU float32 tensor, `g_c`, shape `(3, height, width)` |
| `generate(image, prompt, *, task_family)` | `{"text", "parsed_answer", "invalid"}` |

`session=` may be passed to any of the three. If it is omitted, the first call
loads the pinned checkpoint. `device="auto"` selects CUDA, then MPS, then CPU.
A measurement that must be labeled MPS has to request `device="mps"`. This
package does not relabel a CPU run.

Symbols, for an answer token sequence `y = (y_1, ..., y_T)`, image `x`, and
prompt text `c`:

```
log P(y | x, c) = sum_t log P(y_t | y_<t, x, c)
L_c(x) = -log P(y | x, c)
g_c = gradient of L_c with respect to x
```

`x` is RGB, channel-first, float32, values in `[0, 1]`. `T` is the number of
answer tokens under the rule below. `c` is the user text, not a full chat
string; the session wraps it in the SmolVLM user/assistant template with one
image. `g_c` is detached and on CPU, in that same `[0, 1]` image, not in the
normalized tensor the vision encoder consumes. A positive entry means
increasing that channel would increase `L` and decrease the answer probability.

Study 1 optimized the mean token NLL, which equals `L_c(x) / T`. The two
objectives share a minimizer when every answer has the same `T`. They do not
when answer lengths differ. This interface returns the sum because the contract
asks for the log-probability of the whole answer. `T` is available from
`VLMSession.score_details`.

## Pinned checkpoint

| Field | Value |
| --- | --- |
| Model id | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` |
| Revision | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |
| Study 1 `docs/freeze-config.md` | same revision |
| Parameters | 507,482,304, all float32 |
| Tensor bytes | 2,029,929,216 |
| Attention | eager |
| Weights | frozen, `eval` mode |
| Image splitting | off in this interface; the processor default is on |

The loader asks the Hub for that revision and refuses any other model id,
including `Qwen/Qwen3-VL-2B-Instruct`. Transformers 5.22 did not attach
`_commit_hash` to the loaded objects. The revision check that did run is the
Hub sha, plus the safetensors header of that commit.

Processor settings read from that revision, not from the class defaults in
source: longest edge 2048 before a square resize, square side 512 when
splitting is off, mean and standard deviation `(0.5, 0.5, 0.5)`. With
splitting left at the processor default, a small image is first resized toward
2048 on its long side and then tiled into 512 crops. That path was not run.

## Preprocessing

Three calls share one float32 path:

1. Quantize a copy of `x` to uint8 only to build chat-template token ids.
2. Discard the template's pixel tensor after checking that its shape matches
   the float path.
3. Run the official image processor on the float32 CHW tensor, on CPU, with
   `do_rescale=False` and `do_image_splitting=False`.

`do_rescale=False` is required. The tensor is already in `[0, 1]`; the default
would divide by 255 again. The resize stays on CPU so torchvision does not
replace Lanczos with bicubic. That replacement is what Study 1 described. It
does not apply to this run: torchvision is 0.29.2 and the tensor never leaves
CPU before the resize.

The official processor resizes uint8 pixels. Float32 Lanczos is a different
function. On the measurement machine, a uint8 tensor and a PIL image of the
same bytes matched exactly. The float path did not:

| Reference image | Max absolute gap | Mean absolute gap |
| --- | ---: | ---: |
| 96×128, seed 0 | 0.7908 | 0.00789 |
| 512×512, seed 0 | 0.2664 | 0.00885 |

Both outputs are shaped `(1, 1, 3, 512, 512)`. Masks matched. Gaps are in the
normalized pixel space, whose values lie near `[-1, 1]`.

Consequence: `answer_logprob`, `image_loss_gradient`, and `generate` see the
same pixels. A later script that scores with the raw PIL chat template does
not. The gradient is not the derivative of that uint8 function, which is not
differentiable. No straight-through estimator is used.

## Tokenization

The template ends at `Assistant:`. Answer ids are the tokenizer encoding of the
answer with a leading space, and without special tokens. On this checkpoint,
`"red"` is the bare piece `red` (id 455) and `" red"` is `Ġred` (id 2382).
`"red square"` is two pieces, `Ġred` and `Ġsquare`.

On the red-square probe, with the prompt in the feasibility file, the
leading-space log-probability was -3.341 and the bare log-probability was
-11.637. The leading-space id is the one this interface scores. That agrees
with the Study 1 lesson and with this measurement; it is not assumed for every
future prompt.

## Parser

Checkpoint 1 decision D4 replaced the Checkpoint 0 format-only parser. The
feasibility JSON was produced before that change. Its single string `Red.` is
still only a smoke check.

Normalization is case, whitespace, surrounding quotes, and terminal
punctuation. Nothing else is stripped. An `Answer:` label stays in the string
and makes it invalid. `task_family` selects the vocabulary. Attribute and
spatial questions in the current scenes ask for a color, so the vocabulary is
`red`, `blue`, `green`, and `yellow`. Counting accepts `one`, `two`, `three`,
and `four`, plus the digit aliases `1`, `2`, `3`, and `4`. A digit is stored
as the canonical word: `2` becomes `two`. The raw decoded text is kept beside it.

A recognized color or count that is not the scene answer is a valid parse.
The caller marks that row valid-but-wrong. It is not an invalid output.
Option-list echoes, more than one vocabulary item, refusals, empty strings,
line breaks, truncation, and every other string are invalid. Invalid output is
a task error and is not an attack success. `parsed_answer` is `None` when
`invalid` is true.

Greedy decoding uses `do_sample=False` and `max_new_tokens=16`.

## What the MPS run showed

Device: MPS. Parameter dtype: float32. Machine: Apple M4, 24 GiB unified
memory (`25769803776` bytes), macOS 26.6. The script requested `--device mps`
and checked `parameter.device`.

The probe images are drawn in code: a red square on black, and a blue square
on white, both 128×128. They are not dataset items.

| Check | Result |
| --- | --- |
| Two-token log-probability of `"red square"` | -19.2657 |
| Token count | 2 |
| Library mean NLL | 9.63285 |
| Absolute difference after dividing the sum by `T` | 0 |
| Gradient, red square, max abs / mean abs | 2.349 / 0.0293 |
| Gradient, blue square, max abs / mean abs | 4.761 / 0.0435 |
| Nonzero fraction, both | 1 |
| Repeat of the red square, max abs difference | 0 |
| Red versus blue, max abs difference | 4.780 |
| Load | 4.717 s |
| Gradient times (red, repeat, blue) | 4.594 s, 4.357 s, 4.388 s |
| Wall clock, whole script | 24.177 s |

Times are taken after a device synchronize, so they include the MPS kernels.

Memory, labeled by counter:

- OS available before load: 8,435,777,536 bytes.
- After load: MPS current allocation 1965.6 MB, MPS driver allocation 2856.5 MB,
  OS available 3826.9 MB.
- After the three gradients: MPS current allocation still 1965.6 MB, driver
  allocation 4131.7 MB, process peak RSS 602.0 MB.

`current_allocated` after the graph is freed matches the float32 weight size.
The driver counter is a cache high-water, not a live tensor total, and this
PyTorch build has no separate MPS peak-activation statistic. Process RSS does
not include that Metal cache. Citing 602 MB as the model footprint would be
wrong. The driver figure, 4131.7 MB, is the device number to quote, with the
caveat that it is a cache size after the run.

One greedy decode of the red square returned `"Red."`, parsed as `red`, not
invalid. That is a smoke check of `generate`. It is not a clean-accuracy
estimate.

The finite-difference check of `L = -log P` was run on the closed-form unit
test and on a tiny probe network. It was not repeated on the 500M model.

## Qwen3-VL-2B, not adopted

`Qwen/Qwen3-VL-2B-Instruct` at Hub revision
`89644892e4d85e24eaac8bacfd4f463576704203` was inspected with a range read of
the safetensors header (76,240 bytes) and the file size. The weight file was
not downloaded and `from_pretrained` was not called.

| Field | Value |
| --- | --- |
| On-disk file | 4,255,140,312 bytes |
| Stored dtype | BF16 |
| Parameters | 2,127,532,032 |
| Float32 expansion | 8,510,128,128 bytes (7.93 GiB) |
| Guard | 6 GiB of float32 weights |
| Decision | load aborted, model not adopted |

The guard is an engineering limit for this 24 GB machine. Backward activations
would sit on top of the weights, and the operating system shares the memory.
No clean-task accuracy, gradient, or runtime is claimed for Qwen.

## Not in this checkpoint

No PGD, FGSM, noise attack, training, fine-tuning, or held-out evaluation.
No batching. No CUDA timing. Image splitting was not measured. Gradient
directional alignment `A(s, t)` is not computed here.
