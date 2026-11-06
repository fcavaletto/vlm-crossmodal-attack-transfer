# Model selection

Checkpoint 1.5 decision for the study model. Every SmolVLM2 accuracy and gradient figure below is the Checkpoint 1 MPS pilot, labeled **DEVELOPMENT ONLY**. This checkpoint did not load SmolVLM2 again and did not load Qwen. No attack was run. No approved decision is revised.

The raw gate record is [`../checkpoint-1.5/feasibility-record.json`](../checkpoint-1.5/feasibility-record.json). The lab notes are [`../checkpoint-1.5/AGENT_C_REPORT.md`](../checkpoint-1.5/AGENT_C_REPORT.md). Pull request: https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/pull/7.

## Decision

Keep `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` at revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`.

The approved settings stay in force: float32 weights, eager attention, image splitting off, MPS, `max_new_tokens=16`, leading-space canonical answers, and the D4 parser. `Qwen/Qwen3-VL-2B-Instruct` is not adopted.

## SmolVLM2 and the intended tasks

The intended tasks are the three development families: name a color (attribute), name a count from one to four (counting), and name the color in a spatial relation (spatial). Each scene has four context strings. A scene can enter the proposed H\* population only when the strict parse is correct under both `neutral` and `helpful_nonrevealing` and both image gradients are finite and nonzero.

On the 24 committed development scenes the Checkpoint 1 pilot, run on this MacBook Air (Mac16,12, Apple M4, 24 GiB unified memory) under MPS, showed the following. Denominators are scenes. DEVELOPMENT ONLY.

| Context | Correct | Valid-but-wrong | Invalid | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| `neutral` | 10 | 11 | 3 | 10/24 |
| `helpful_nonrevealing` | 5 | 12 | 7 | 5/24 |
| `contradictory` | 3 | 17 | 4 | 3/24 |
| `answer_revealing` | 13 | 5 | 6 | 13/24 |

By family, each cell has 8 scenes. DEVELOPMENT ONLY.

| Family | Context | Correct | Valid-but-wrong | Invalid |
| --- | --- | ---: | ---: | ---: |
| attribute | `neutral` | 6 | 0 | 2 |
| attribute | `helpful_nonrevealing` | 1 | 0 | 7 |
| attribute | `contradictory` | 3 | 3 | 2 |
| attribute | `answer_revealing` | 5 | 0 | 3 |
| counting | `neutral` | 2 | 6 | 0 |
| counting | `helpful_nonrevealing` | 3 | 5 | 0 |
| counting | `contradictory` | 0 | 8 | 0 |
| counting | `answer_revealing` | 8 | 0 | 0 |
| spatial | `neutral` | 2 | 5 | 1 |
| spatial | `helpful_nonrevealing` | 1 | 7 | 0 |
| spatial | `contradictory` | 0 | 6 | 2 |
| spatial | `answer_revealing` | 0 | 5 | 3 |

Attribute under `neutral` is the strongest visual cell (6/8). The helpful sentence, which is task guidance and does not name the color, is followed by invalid sentences on 7/8 attribute scenes. Counting under the contradictory sentence is `four` on 8/8 scenes, and counting under the revealing sentence is correct on 8/8. Spatial accuracy is at most 2/8 in every context, and the revealing control is 0/8. Three scenes, in three groups, are clean-correct under both primary contexts: `dev-attribute-03-a`, `dev-counting-00-b`, and `dev-counting-02-b`. All 48 primary-context gradients were finite and nonzero.

SmolVLM2 can supply the image gradients the protocol defines. On this development sample it does not reliably solve the three families. That is a property of the current scenes, prompts, and parser together with this model. It is the reason the pilot asked to treat those cells as floors. It is not evidence about Qwen, and no attack result enters the choice.

The header survey in this checkpoint re-read the pinned SmolVLM2 safetensors header. The Hub sha is still `7b375e1b73b11138ff12fe22c8f2822d8fe03467`, with 507,482,304 float32 parameters. The weights were not loaded.

## Qwen3-VL-2B gates

Candidate: `Qwen/Qwen3-VL-2B-Instruct`, Hub revision `89644892e4d85e24eaac8bacfd4f463576704203`. A range read of 76,240 header bytes on this run returned 2,127,532,032 BF16 parameters, on-disk file 4,255,140,312 bytes, float32 expansion 8,510,128,128 bytes. The local snapshot does not contain `model.safetensors`. `from_pretrained` was not called.

| Gate | Result |
| --- | --- |
| 1. Weight load and memory | Aborted. OS-available memory was 7,508,770,816 bytes. Float32 requires 20,871,410,752 bytes before a load. BF16 requires 11,509,447,200 bytes. Both failed the spare line, the staging line, and the projection line. |
| 2. Minimal forward | Not started. |
| 3. Clean answers on a development subset | Not started. |
| 4. Image gradients | Not started. Gate 1 aborted. |

MPS was available. No Qwen parameter was placed on MPS, so this checkpoint has no Qwen MPS allocation, load time, or forward time. The survey wall clock was 5.572 seconds, which is the header read and the processor-config download.

The processor config would leave a 384×512 development image at 384×512, about 192 image tokens under the formula in the JSON. That calculation is not a forward pass. The binding constraint is the weight budget, and the image-size estimate did not change the abort.

There is no paired comparison. Qwen produced no scores and no generations, so the D2 prompt and the D4 parser were not applied to it.

## Standing protocol

D5 remains the model pin. The D4 parser, the sum-NLL loss, H\*, and the estimand are unchanged. Checkpoint 0's 6 GiB float32 weight cap also still refuses this Qwen file (8,510,128,128 bytes against 6,442,450,944). The loader in `src/vlm_iface` was not edited. The new budget is what refuses a native BF16 load on the memory this machine reported.

A later retry of gates 2–4 is a new measurement. It belongs after this script reports that a named dtype is inside the budget, on MPS, one model at a time. Accuracy from that retry must not be used to choose a parser or to replace SmolVLM2.
