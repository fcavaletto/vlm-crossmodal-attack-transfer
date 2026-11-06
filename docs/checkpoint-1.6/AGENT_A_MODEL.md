# Agent and responsibility

Agent A, model and task selection, Checkpoint 1.6 of Study 2 (`vlm-crossmodal-attack-transfer`). Research lead: Federico Cavaletto.

This checkpoint asks whether `Qwen/Qwen3-VL-2B-Instruct` can replace the primary VLM after the clean-competence gate failed. The work is model and task selection only. Attacks are not authorized. No PGD, FGSM, noise, training, H\* fit, held-out scene, or paid GPU was run. `docs/checkpoint-1/` was not edited. SmolVLM2 was not loaded.

Owned paths:

- `docs/checkpoint-1.6/AGENT_A_MODEL.md`
- `docs/checkpoint-1.6/qwen_feasibility.json`
- `docs/checkpoint-1.6/run_qwen_feasibility.py`

Branch: `checkpoint-1.6/stronger-model`, started at `b2e7abff72b87b475d0abf8d037e1b4d2f9c0c06`.

# Executive summary

Qwen/Qwen3-VL-2B-Instruct is not adopted. The Checkpoint 1.5 budget was imported unchanged and applied to a new OS reading of 10,080,436,224 available bytes. The safetensors header is revision `89644892e4d85e24eaac8bacfd4f463576704203`, with 2,127,532,032 BF16 parameters. Float32 requires 20,871,410,752 available bytes and fails three lines. BF16 requires 11,509,447,200 and fails the calibrated projection by 1,429,010,976 bytes. Every line has to clear, so `from_pretrained` was not called and the weight file was not downloaded. The Checkpoint 1.5 stop was a pre-load budget refusal on a different available-byte reading. This run did not reuse that reading and did not observe an out-of-memory event. MPS is available and an 8×8 BF16 matmul ran. No Qwen parameter was allocated. The forward, the twelve development rows, and the gradient check did not start. SmolVLM2 was not loaded. A 24 GiB CUDA job is requested for a 24-scene by 4-context clean score plus one-scene gradients. The cost estimate is $0.34 to $1.79 for up to one hour. No attack was run.

# Work completed

- Re-read the Checkpoint 1.5 budget in `docs/checkpoint-1.5/run_model_feasibility.py` and imported `budget_decision` from that file. The script checks that the spare line is still 5 GiB, the projection floor is still 2 GiB, and the OS floor is still 6 GiB. The budget fixtures in that file passed in this process. The fixtures were not used as today's available-byte measurement.
- Read OS-available memory twice in this process, before the header survey and at the decision. The decision uses the second reading. The script does not open `docs/checkpoint-1.5/feasibility-record.json`.
- Resolved `Qwen3VLForConditionalGeneration`, `Qwen3VLProcessor`, and `Qwen2VLImageProcessor` from the installed transformers. The preprocessor name `Qwen2VLImageProcessorFast` aliases to `Qwen2VLImageProcessor` in transformers 5.22.1. Ran one 8×8 MPS BF16 matmul.
- Confirmed the parameter count and stored dtype with an HTTP 206 range read of the 76,240-byte safetensors header at the pinned revision. Read `config.json` and `preprocessor_config.json` only. Did not download weight tensors.
- Applied the imported budget to float32 and BF16. Neither dtype was permitted. The load function was not entered. `from_pretrained` call count is 0.
- Listed the twelve-row development subset in the runner before any generation. Confirmed those three scenes are in `fixtures/development_v2/manifest.jsonl` and that each PNG is 512×384. Did not generate.
- Wrote a CUDA planning estimate for a 24-scene by 4-context clean score plus gradients on one scene. Did not launch it.

# Key findings

The candidate remains `Qwen/Qwen3-VL-2B-Instruct` at Hub revision `89644892e4d85e24eaac8bacfd4f463576704203`. The header is an exact range read: 2,127,532,032 parameters, dtype count BF16 for every parameter, on-disk file 4,255,140,312 bytes, float32 expansion 8,510,128,128 bytes. `config.json` says `text_config.dtype` is `bfloat16` and `architectures` is `Qwen3VLForConditionalGeneration`. The header and the config agree on BF16. The local snapshot contains `config.json` and `preprocessor_config.json` only.

| Gate | Status |
| --- | --- |
| 1. OS-available memory | Measured. 10,080,436,224 bytes at the decision. |
| 2. BF16, header, and the pre-load budget | Aborted. No dtype was permitted. `from_pretrained` was not called. |
| 3. Minimal MPS forward | Not started. |
| 4. Twelve development generations | Not started. The scene list was fixed and not run. |
| 5. Canonical-word image gradients | Not started. |

Float32 resident size is 8,510,128,128 bytes. BF16 resident size is 4,255,064,064 bytes, two bytes times the parameter count. The binding line for both is the calibrated projection: resident bytes times 4324.7/1965.6, then plus 2 GiB. The ratio is the imported Checkpoint 1 constant from the SmolVLM2 pilot's one-decimal MiB figures. It is a safety extrapolation, not a Qwen measurement.

| Line | Float32 need (bytes) | BF16 need (bytes) | At 10,080,436,224 available |
| --- | ---: | ---: | --- |
| OS floor | 6,442,450,944 | 6,442,450,944 | clears |
| Spare, resident + 5 GiB | 13,878,837,248 | 9,623,773,184 | float32 fails, BF16 clears |
| Staging, two resident copies | 17,020,256,256 | 8,510,128,128 | float32 fails, BF16 clears |
| Calibrated projection | 20,871,410,752 | 11,509,447,200 | both fail |

BF16 is 1,429,010,976 bytes under its projection and 456,663,040 bytes over its spare line. The spare clearance does not authorize the load. The Checkpoint 0 float32 cap of 6,442,450,944 bytes also refuses the float32 expansion. That cap is recorded on the survey. The BF16 decision used the Checkpoint 1.5 function.

MPS is available and built. The 8×8 BF16 product was finite. `torch.mps.current_allocated_memory()` was 0 after that probe. There is no Qwen MPS allocation, load time, or forward time. Wall clock for the whole script was 4.755 seconds.

The three development PNGs are 512 wide and 384 high. The processor arithmetic in the JSON, the same `smart_resize` formula as Checkpoint 1.5, leaves that image at 384×512 and about 192 language-model image tokens. That figure is not a forward pass, and it was not used to waive the budget.

SmolVLM2 stays the comparison model through the committed Checkpoint 1.5 tables in `docs/models/MODEL_SELECTION.md` and `docs/checkpoint-1/summary.json`. Those tables are DEVELOPMENT ONLY and were not remeasured. This run produced no Qwen scores, so there is no paired subset.

# Evidence

Machine: Mac16,12, Apple M4, `hw.memsize` 25,769,803,776, macOS 26.6 (25G72). Interpreter `/Users/fedecava/Development/_c0-work/venv-vlm/bin/python`, Python 3.12.14, torch 2.15.0, transformers 5.22.1. `HF_HOME` was `/Users/fedecava/Development/models/huggingface`. `HF_HUB_OFFLINE` was unset. The budget file sha256 is `b7cd9ad2808a5bbfd37c619b3f4d7df1225f95650f60d20b63b8eb49e41981a8`.

| Item | Value |
| --- | --- |
| Device | MPS is available. No model was placed on a device. |
| Loaded dtype | None. Header dtype is BF16. Config dtype is `bfloat16`. |
| Revision | `89644892e4d85e24eaac8bacfd4f463576704203`, exact header, 76,240 bytes |
| Runtime | 4.755 s wall for the survey. No load time and no forward time. |
| `from_pretrained` | Not called. Call count 0. |
| Weight file | `model.safetensors` absent before and after. `weights_downloaded` is false. |
| MPS current allocated | 0 bytes with no model loaded |

OS memory from `psutil.virtual_memory()` in this process. Available includes reclaimable inactive pages. The decision used the second snapshot.

| Counter | At start (bytes) | At decision (bytes) | Decision, MiB |
| --- | ---: | ---: | ---: |
| Total | 25,769,803,776 | 25,769,803,776 | 24576.0 |
| Available | 10,061,201,408 | 10,080,436,224 | 9613.5 |
| Used | 11,216,207,872 | 11,193,679,872 | 10675.1 |
| Free | 2,926,624,768 | 2,815,377,408 | 2685.0 |
| Active | 7,174,717,440 | 7,241,629,696 | 6906.2 |
| Inactive | 6,769,393,664 | 6,950,895,616 | 6628.9 |
| Wired | 4,041,490,432 | 3,952,050,176 | 3769.0 |

The Checkpoint 1.5 feasibility record lists 7,508,770,816 available bytes. That field was not read by this runner and was not passed to `budget_decision`. The earlier stop was a pre-load budget refusal before any weights were loaded.

Adding active to available gives 17,322,065,920 bytes. That sum would clear the BF16 requirement and would still miss the float32 requirement. The decision function received only `available`. The sum is arithmetic on the stored counters.

The machine record is `docs/checkpoint-1.6/qwen_feasibility.json`. `action` is `load_aborted`, `stopped_at_gate` is `gate_2_preload_budget`, `adopted` is false, and `recommendation` is `needs_authorized_cuda`.

# Design decisions and rationale

The load rule is the Checkpoint 1.5 `budget_decision`, called on resident bytes from this header and on available bytes from this process. Float32 is the dtype only when the float32 budget passes. BF16 is the dtype only when float32 fails and the BF16 budget passes, and only if the small MPS BF16 matmul also ran. Neither budget passed, so the function that contains `from_pretrained` was not called.

The budget was not lowered to absorb the 1,429,010,976 byte BF16 gap. The spare line clearing on this snapshot is a real change from the tighter Checkpoint 1.5 snapshot, and it is not permission to skip the projection. The positive-control fixtures still have to pass, and they did.

The Checkpoint 0 cap of 6 GiB of float32 weights refuses this file's float32 expansion. Checkpoint 1.5 already treated that cap as a separate fact from a native-width budget. This run records the cap and still evaluates BF16 on its own resident size. Both checks refuse a load today.

The twelve-row subset is `dev2-attribute-00-a`, `dev2-counting-00-a`, and `dev2-spatial-00-a`, each with `neutral`, `helpful_nonrevealing`, `contradictory`, and `answer_revealing`. The prompt rule is context text, a newline, and the question. The official label is `vlm_iface.parsing.parse_generation`. `max_new_tokens` is 16. Those choices are in the runner above any generation call. They were not executed.

SmolVLM2 was left unloaded. A second model in unified memory would change the available-byte reading, and the committed tables already answer the comparison until Qwen has scores on the same scenes.

The CUDA request is BF16 weights on a 24 GiB GPU. Expected weight memory is the header figure, 4,255,064,064 bytes. The activation headroom is an estimate: a saved-tensor envelope of 3,489,660,928 bytes for a 256-token decoder plus a vision tower at 1,536 tokens, then doubled, then raised to an 8 GiB floor. Weights plus that floor are 12,844,998,656 bytes. The 24 GiB request leaves room because the envelope is not a profiler trace. The temporal doubling is an unread packing assumption from `temporal_patch_size` 2. The 15-to-60 minute window is the billing cap, not a measured Qwen runtime. SmolVLM2 MPS scene times were not reused as CUDA times.

# Known limitations

The drop ratio comes from one SmolVLM2 float32 MPS load. Applied to Qwen it is an extrapolation. On this snapshot the BF16 spare and staging lines clear without it, and the projection line does not.

No Qwen logit, generation, parse, gradient, or MPS model time was measured. Nothing in this report says how Qwen would score the development scenes.

The 8×8 matmul shows that this torch build can multiply small BF16 tensors on MPS. It does not show that the Qwen3-VL graph runs in BF16 on MPS.

The 192-token figure and both vision envelopes are arithmetic on the config. The image processor was not called. The larger vision sequence assumes a still image is packed twice, and that assumption was not checked.

`available + active` is not a successful BF16 load. The active pages were left in place.

The CUDA dollar range is an estimate from public hourly listings read on 6 November 2026, billed as up to one hour. It is not a quote and it was not purchased. Community capacity can be interrupted. The runner's implemented scoring list is the twelve-row subset. The 24-scene estimate is the workload the assignment asked to price, and it was not executed.

SmolVLM2 accuracy remains the committed pilot. It was not repeated.

# Blockers

Blocking for adoption: gate 2 aborted, so there is no clean-task evidence for this model. The safe state is not adopted.

Blocking for a local retry: the decision snapshot has to reach at least 11,509,447,200 available bytes, with the other lines still clearing, before this runner will call `from_pretrained`. This run did not free memory to manufacture that reading.

Not blocking for the comparison model: SmolVLM2 remains the pinned model in the committed tables. This report does not ask the other workstreams to wait on a Qwen score.

# Decisions requested from the research lead

1. Authorize one CUDA machine, 24 GiB, BF16, for a minimal forward, then the twelve-row development subset, then a 24-scene by 4-context clean score if that forward is finite, plus canonical-word gradients on `dev2-attribute-00-a` for `neutral` and `helpful_nonrevealing`. Recommendation: authorize that job. Expected weight memory is 4,255,064,064 bytes. Planning headroom is 8 GiB. Planning total is 12,844,998,656 bytes. Planning wall clock is 15 to 60 minutes. Planning cost is $0.34 to $1.79. All three planning figures are estimates. This run did not start the machine.

2. Keep Qwen unadopted on the present evidence. Recommendation: yes. A budget refusal is not a clean-competence result, and it is not a reason to declare the model unable to do the tasks.

3. Keep SmolVLM2-500M, revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`, as the comparison model until a paired development score exists. Recommendation: yes.

4. Leave attacks, H\*, and held-out scenes unauthorized on that GPU as well. The request is a clean score and a finiteness check.

# Reproduction instructions

From the worktree root. The command remeasures memory, reads the header, and writes the JSON. On this run it did not call `from_pretrained`.

```bash
HF_HOME=/Users/fedecava/Development/models/huggingface \
PYTHONPATH=src/vlm_iface:src/pilot \
/Users/fedecava/Development/_c0-work/venv-vlm/bin/python \
  docs/checkpoint-1.6/run_qwen_feasibility.py
```

Do not set `HF_HUB_OFFLINE`. The run recorded here exited 0. `wall_seconds` was 4.755. `from_pretrained_called` is false. `stopped_at_gate` is `gate_2_preload_budget`. `decision_available_bytes` is 10080436224.

# Git

- Branch: `checkpoint-1.6/stronger-model`
- Started at: `b2e7abff72b87b475d0abf8d037e1b4d2f9c0c06`
- Feasibility commit: stamped after the files are committed
- Not pushed. No pull request.
