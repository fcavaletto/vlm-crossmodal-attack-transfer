# Agent and responsibility

Agent C, model feasibility, branch `checkpoint-1.5/model-feasibility`.

Owned paths: `docs/models/MODEL_SELECTION.md`, this report, and the feasibility record under `docs/checkpoint-1.5/`. The work is the Checkpoint 1.5 question for two models. SmolVLM2-500M: can it do the intended tasks, using the Checkpoint 1 tables. Qwen3-VL-2B-Instruct: weight load and memory, then a minimal forward, then clean answers on a small development subset, then image gradients only if the earlier gates pass. No attack, no training, no cloud GPU, and no edit to Checkpoint 1 files, `PROTOCOL_CHANGELOG.md`, or `REVIEW_PACKET.md`.

# Executive summary

SmolVLM2-500M stays the study model. Qwen3-VL-2B-Instruct was not loaded. Task ability is the Checkpoint 1 MPS pilot on the 24 development scenes, labeled DEVELOPMENT ONLY. Clean accuracy was 10/24 under neutral, 5/24 under helpful_nonrevealing, 3/24 under contradictory, and 13/24 under answer_revealing. Spatial accuracy is at most 2/8 in every context, and the answer-revealing spatial cell is 0/8. Three scenes in three groups are clean-correct under both primary contexts. That pilot's full-resolution gradients were finite on every scene. This checkpoint did not load SmolVLM2 again.

The Qwen header was read again at revision `89644892e4d85e24eaac8bacfd4f463576704203`: 2,127,532,032 BF16 parameters, float32 expansion 8,510,128,128 bytes. The OS reported 7,508,770,816 bytes available. The pre-load budget requires 20,871,410,752 bytes for float32 and 11,509,447,200 bytes for BF16. Both dtypes failed, so `from_pretrained` was not called. The forward, clean-answer, and image-gradient gates did not start. There is no Qwen MPS runtime and no paired comparison. No attack, training run, or cloud GPU was used.

# Work completed

- Read `docs/checkpoint-1/PILOT_REPORT.md` and the compact files under `docs/checkpoint-1/` at commit `90c6ffe83d61514da217a20f8dfc2f202afbbe69` before any weight survey.
- Re-read the SmolVLM2 safetensors header at the pinned revision. Confirmed 507,482,304 float32 parameters. Did not call `from_pretrained`.
- Re-read the Qwen3-VL-2B header with an HTTP 206 range request of 76,240 bytes. Confirmed the Checkpoint 0 revision, parameter count, BF16 storage, and float32 expansion. The weight tensor bytes were not downloaded. After the run, `model.safetensors` was absent from the local Hub snapshot.
- Downloaded `config.json` and `preprocessor_config.json` only. Estimated the resize of a 384×512 development image from those settings. Did not run the processor on pixels.
- Applied an explicit unified-memory budget to float32 and bfloat16. The budget refuses a load unless OS-available memory covers an operating-system floor, 5 GiB of spare above the resident weights, a projection calibrated on the Checkpoint 1 SmolVLM load, and a second copy for staging. A fixture in the script requires that this same function would have allowed the Checkpoint 1 SmolVLM load.
- Wrote the decision in `docs/models/MODEL_SELECTION.md` and the machine record in `docs/checkpoint-1.5/feasibility-record.json`.

The script contains no `from_pretrained` call. The process checks its own syntax tree for that call and exits if one appears.

# Key findings

## SmolVLM2 task ability

Source: `docs/checkpoint-1/PILOT_REPORT.md` and `docs/checkpoint-1/summary.json`, MPS, float32, revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`. DEVELOPMENT ONLY. These numbers were not remeasured here.

The intended tasks are attribute, counting, and spatial questions under the four stored contexts, scored with the D4 exact-vocabulary parser. H\* would later be fit only on scenes that are clean-correct under both primary contexts. This report does not fit it.

| Context | Correct | Valid-but-wrong | Invalid |
| --- | ---: | ---: | ---: |
| `neutral` | 10 | 11 | 3 |
| `helpful_nonrevealing` | 5 | 12 | 7 |
| `contradictory` | 3 | 17 | 4 |
| `answer_revealing` | 13 | 5 | 6 |

Each family-context cell below has 8 scenes.

| Family | `neutral` | `helpful_nonrevealing` | `contradictory` | `answer_revealing` |
| --- | ---: | ---: | ---: | ---: |
| attribute | 6 | 1 | 3 | 5 |
| counting | 2 | 3 | 0 | 8 |
| spatial | 2 | 1 | 0 | 0 |

The cells are correct counts. Attribute under neutral is the only visual cell at or above one half, together with the revealing counting cell, which is the copying control (8/8). Helpful attribute text is followed by invalid sentences on 7 of 8 scenes. Every contradictory counting output is the count that sentence states. Spatial stays at or below 2/8, including 0/8 when the sentence names the answer. The clean-correct intersection of the two primary contexts is 3 scenes in 3 groups. Gradients on those primary contexts were finite and nonzero for all 24 scenes.

SmolVLM2 can produce the image gradients the study defines. On this development sample it does not reliably do the three tasks. The selection keeps the model that has those measurements. The task floors stay attached to the scenes, the prompts, and the parser.

No new development subset was run. A subset scored only with SmolVLM2 would repeat part of the 24-scene pilot. A paired subset needs Qwen generations, and gate 1 stopped before any existed.

## Qwen gates

Machine for the gate record: Mac16,12, Apple M4, `hw.memsize` 25,769,803,776, macOS 26.6 (25G72). Interpreter `/Users/fedecava/Development/_c0-work/venv-vlm/bin/python`, Python 3.12.14, torch 2.15.0, transformers 5.22.1. `torch.backends.mps.is_available()` was true. That flag is a capability check. No model tensor was allocated on MPS, and the record's `mps_model_measurement` is null.

The process printed `Redirects are currently not supported in MacOs` while importing torch, before the header survey. The pilot saw the same sentence during its load. Here the weight file was still absent after the process exited.

| Gate | Status | What was measured |
| --- | --- | --- |
| 1. Weight load and memory | `aborted` | Header survey and the OS memory snapshot below. `from_pretrained` was not called. |
| 2. Minimal forward | `not_started` | No logits and no forward runtime. |
| 3. Clean answers | `not_started` | No generations and no D4 parses. |
| 4. Image gradients | `not_started` | An earlier gate aborted. |

Header, from the range read, revision `89644892e4d85e24eaac8bacfd4f463576704203`:

| Field | Value |
| --- | --- |
| Parameters | 2,127,532,032, all BF16 |
| On-disk file | 4,255,140,312 bytes |
| Float32 expansion | 8,510,128,128 bytes |
| BF16 resident estimate | 4,255,064,064 bytes |
| Header bytes | 76,240 |
| Checkpoint 0 float32 cap (6 GiB) | refuses the float32 expansion |

OS memory at the decision, one `psutil` snapshot. "Available" includes reclaimable inactive pages. "Free" does not.

| Counter | Bytes | MiB (bytes / 1024², one decimal) |
| --- | ---: | ---: |
| Total | 25,769,803,776 | 24576.0 |
| Available | 7,508,770,816 | 7160.9 |
| Free | 243,466,240 | 232.2 |
| Active | 7,319,257,088 | 6980.2 |
| Inactive | 6,445,383,680 | 6146.8 |
| Wired | 3,651,518,464 | 3482.4 |

Available memory, 7160.9 MiB, is above the 6 GiB floor (6144.0 MiB). The abort is the three tighter lines. A load has to clear every line. The binding line is the largest one.

| Line | Float32 need | BF16 need | At 7,508,770,816 available |
| --- | ---: | ---: | --- |
| OS floor | 6,442,450,944 | 6,442,450,944 | clears |
| Spare, resident + 5 GiB | 13,878,837,248 | 9,623,773,184 | fails |
| Staging, two resident copies | 17,020,256,256 | 8,510,128,128 | fails |
| Projection from the pilot drop | 20,871,410,752 | 11,509,447,200 | fails |

The projection multiplies the resident byte count by 4324.7 / 1965.6 and then adds 2 GiB. The numerator and the denominator are the one-decimal MiB figures in Checkpoint 1 pilot section 8: available memory fell from 8074.4 MiB to 3749.7 MiB while the MPS current allocation was 1965.6 MiB. The ratio is 2.200. It is a safety extrapolation from one smaller float32 load, not a Qwen measurement. Float32 and BF16 each fail the spare line and the staging line as well, so the abort stands if that ratio is set aside.

Adding the active set to the available set gives 14,828,027,904 bytes. That sum still misses the float32 requirement. It would clear the BF16 requirement. Inactive pages are already inside the available counter, so the gap is active memory. This run left the active set alone. Another process may be using the machine, and an allocation past the available counter is the unsafe case the gate is there to avoid.

The processor file `Qwen2VLImageProcessorFast` sets `min_pixels` 65536 and `max_pixels` 16777216, patch 16, merge 2. `smart_resize(384, 512)` on those limits returns 384×512, and `(384/16)*(512/16)/(2*2)` is 192. That is an arithmetic estimate of image tokens. It is not a measured activation size, and it was not used as a reason to load.

Survey wall clock was 5.572 seconds. There is no load time to report.

# Evidence

- `docs/checkpoint-1.5/feasibility-record.json` is the script output. `action` is `load_aborted`, `passed_gate_1` is false, `from_pretrained_called` is false, `weights_downloaded` is false, `qwen_local_weight_file_present` is false, and `adopted` is false.
- `docs/checkpoint-1/PILOT_REPORT.md` sections 5, 7, and 8, plus `docs/checkpoint-1/summary.json`, are the SmolVLM2 task and gradient evidence. Commit `90c6ffe83d61514da217a20f8dfc2f202afbbe69`.
- The Qwen Hub snapshot under `HF_HOME=/Users/fedecava/Development/models/huggingface` gained the small config and preprocessor blobs. It does not contain `model.safetensors`.
- The script's fixture check passed inside the same process: the budget permits the Checkpoint 1 SmolVLM load at 8074.4 MiB available and 2,029,929,216 resident bytes, and it refuses both Qwen dtypes at 8,145,289,216 available bytes.

# Design decisions and rationale

The SmolVLM2 answer uses the existing 24-scene MPS pilot. Those tables already cover every development scene, every context, and both primary gradients. Loading the 500M model again would spend the unified-memory machine on a repeat and would not create the missing Qwen half of a comparison.

The Qwen load is refused before `from_pretrained`. On this Mac a unified-memory allocation that overruns the available counter compresses or jetsams other processes. Discovering the limit by trying the load is the failure mode the assignment says to avoid. The positive control keeps the rule from being so tight that it would have refused the SmolVLM2 load that already ran to completion, including full-resolution gradients, with about 3749.7 MiB still available afterward.

The four lines are:

- The existing 6 GiB OS-available floor (`vlm_iface.pins.MIN_AVAILABLE_BYTES`). Both dtypes cleared it on this snapshot.
- Five GiB spare above the resident weights. The pilot started with about 6 GiB of that spare. Five GiB is under that observation.
- A projection that applies the pilot's available-drop ratio and still wants 2 GiB left. Two GiB is under the 3749.7 MiB the pilot still had after loading.
- A second resident copy, for a staging buffer plus a device copy. On this machine those copies share one memory pool.

BF16 is judged on its own resident size, two bytes times the parameter count. The Checkpoint 0 cap of 6 GiB of float32 weights refuses this file and does not, by itself, answer a native-width load. The spare, staging, and projection lines refuse that native-width load on the memory measured here. `src/vlm_iface` was left unchanged, so the model-id guard and the 6 GiB cap remain in the loader.

CPU was not used. A CPU weight allocation would sit in the same unified memory, and a CPU timing would have to be labeled CPU. The gates that need a device are MPS gates, and they did not start.

The image-token estimate is recorded so a later reader can see that the development images would not be expanded by the published pixel caps. It is not a reason to skip the weight budget.

No protocol revision is proposed. A revision would need the development observation, the original decision, the revised decision, the scientific justification, whether the estimand changes, and a statement that the proposal precedes any new result. None of those are offered. D4, D5, the sum loss, H\*, and the estimand stay as Checkpoint 1 left them.

# Known limitations

The drop ratio comes from one SmolVLM2 float32 load, published to 0.1 MiB. Applied to Qwen it is an extrapolation. The spare and staging lines fail without it.

`available + active` would clear the BF16 lines and would still miss the float32 lines. That arithmetic describes the snapshot. It is not a measurement of a successful BF16 load, and the active pages were not released.

There is no Qwen log-probability, generation, parse, gradient, or MPS runtime. Nothing in this report says how Qwen would score the 24 scenes.

The 192-token figure is `smart_resize` on the processor JSON. The image processor was not called on a development PNG.

SmolVLM2 accuracy and gradient figures are the pilot's. They were not repeated today. The header sha was re-checked.

The pilot's post-load MPS current allocation was 1965.6 MiB against 2,029,929,216 tensor bytes. The budget applies the ratio to the tensor-byte resident size. That is slightly smaller than the allocator figure from the pilot.

# Blockers

Blocking for a Qwen adoption decision: the load was aborted, so gates 2–4 have no data. The safe state is "not adopted."

Not blocking for the rest of Checkpoint 1.5: SmolVLM2 remains the pinned model, and the task tables already exist. Scene redesign and answer scoring can proceed on that model. This report does not ask them to wait for a Qwen score.

# Decisions requested from the research lead

1. Keep SmolVLM2-500M at the pinned revision and the D5 settings as the only loaded study model. Recommendation: yes. The gradients exist, the task floors are already documented, and Qwen has no score on these scenes. Work on the current model can continue.
2. Leave Qwen3-VL-2B-Instruct unadopted at this checkpoint. Recommendation: yes. Gate 1 aborted and the later gates did not run.
3. A later MPS retry, if one is wanted, should rerun `docs/checkpoint-1.5/run_model_feasibility.py` when the OS reports at least 11,509,447,200 bytes available, and only then add a BF16 load that continues to a minimal forward, the same development scenes under the D2 prompt and the D4 parser, and image gradients only if those gates pass. Float32 needs 20,871,410,752 bytes available under the same function. Recommendation: defer the retry. The task floors are a scene and scoring question. They are not a reason to replace the model. Any Qwen accuracy from a retry has to be labeled DEVELOPMENT ONLY and must not be used to choose a parser. This item is a request for a future measurement, not a revision of D5.
4. Attacks stay unauthorized. This checkpoint does not choose a step size or a model from an attack.

# Reproduction instructions

From the worktree root, with the Checkpoint 1 virtualenv. The command surveys headers and writes the JSON. It does not load weights.

```bash
HF_HOME=/Users/fedecava/Development/models/huggingface \
PYTHONPATH=src/vlm_iface:src/pilot \
/Users/fedecava/Development/_c0-work/venv-vlm/bin/python \
  docs/checkpoint-1.5/run_model_feasibility.py
```

The process runs the budget fixtures before any Hub call. A fixture failure exits non-zero and writes no success record. The run recorded here exited 0. `survey_wall_seconds` in the JSON was 5.572.

`pytest` was not run. `src/` was not modified. The Checkpoint 1 verifier was not run; it rebuilds the pilot summary and does not load a model, and this checkpoint did not touch those files.

# Git

- Branch: `checkpoint-1.5/model-feasibility`
- Branch point: `90c6ffe83d61514da217a20f8dfc2f202afbbe69` on `checkpoint-1/dev-pilot`
- Base for the pull request: `main` at `65a8d8ffb2f22f3d284b64e2d481151474761c5e`
- The pull request diff against `main` contains Checkpoint 1 because this branch starts at the Checkpoint 1 head. Checkpoint 1 files were not edited.
- Pull request URL and the commit that records it: filled in the child commit after `gh pr create`.
