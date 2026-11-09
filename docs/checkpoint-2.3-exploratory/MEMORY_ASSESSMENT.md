# Memory assessment: can the exploratory run proceed without closing applications? (2026-11-09)

For research lead Federico Cavaletto.

Status of what was done:
- No model has been loaded. The smoke attempt on 2026-11-09 was refused by the frozen guard (`paths_smoke/progress.json`, commit `4917f63`).
- Nothing frozen was edited:
  - the Checkpoint 1.5 budget script;
  - `qwen_rgb.budget`;
  - the 2.1 records;
  - `EXPLORATORY_PLAN.md`.

This file proposes one change for this exploratory run only, named **EMP-1**. It requires approval before use.

## 1. Verdict

The guard is a **conservative estimate, not a hard technical requirement.**

| Guard line (bf16 Qwen) | Required | Basis |
| --- | --- | --- |
| `os_floor` | 6.0 GiB | Policy choice |
| `spare_above_weights` | 3.96 + 5 = 8.96 GiB | Policy: 5 GiB above the weights |
| **`calibrated_projection`** (binding, failed) | 3.96 × 2.20 + 2 = **10.72 GiB** (11.51 GB) | Ratio 2.20 = the drop in available memory per byte of MPS tensors in **one float32 SmolVLM2 load in Checkpoint 1**. A different model and dtype. |
| `staging_double_copy` | 7.92 GiB | Assumes a full CPU copy plus an MPS copy at load |

What the four completed **Qwen bf16** runs actually used. These are committed `progress.json` files.

| Run | Available before load | Peak MPS tensors | Peak MPS driver pool | Peak process RSS | Minimum available during run |
| --- | --- | --- | --- | --- | --- |
| 1.8 | 11.93 GiB | 3.98 GiB | **5.61 GiB** | 0.59 GiB | 6.04 GiB |
| 1.9 | 14.40 GiB | 3.98 | 4.64 | 0.68 | 6.59 |
| 2.0 | 11.57 GiB | 3.98 | 4.64 | 0.59 | 7.09 |
| 2.1 | 12.54 GiB | 3.98 | 4.64 | 0.59 | 6.82 |

The measured working set is about 5.3–6.3 GiB, against the 10.7 GiB the ratio projects.

**Caveat:** those peaks were sampled between scenes, not inside a backward pass. The true peak is unmeasured. The smoke test measures it exactly (§4).

## 2. macOS and MPS behavior

Facts measured on this Mac today:
- Total memory is 24 GiB.
- Memory pressure is level 1 (normal); `memory_pressure` reports 81% free.
- psutil "available" is 9.5–10.4 GiB. It counts only free and inactive pages. Compressible memory of idle applications comes on top of that: the compressor already holds about 4.6 GB in about 2.2 GB.
- Swap: one 1 GiB swapfile, 0.5 MB used. macOS adds swapfiles on demand, and 31 GB of disk is free.
- The MPS recommended working-set limit is 17.8 GiB.
- The process physical footprint, read with `proc_pid_rusage`, **includes MPS buffers**. Tested: a 1.5 GiB MPS tensor raised it by 1.55 GiB. The lifetime maximum gives the exact process peak, including inside a backward pass.

What follows:
- **Validity.** Compression and swap are lossless. They change speed, not numbers. Memory pressure cannot alter a result as long as the numerical configuration is the same.
- **Failure modes, in order of likelihood:**
  1. **Slowdown from paging.** It is bounded by the 4.6 h time cap, which then measures fewer scenes and reports which.
  2. **An MPS allocation error.** A Python `RuntimeError`, already caught as `stopped_oom`. Records are flushed per scene.
  3. **System-wide unresponsiveness under critical pressure.** The new monitor aborts on it.
- **Not documented, and not verified:** how macOS pages Metal shared buffers that the GPU is using. That is the main reason for a monitored smoke test before any longer run.

## 3. Ways to reduce memory, and why none is needed or acceptable

| Option | Saves | Changes numerics? | Decision |
| --- | --- | --- | --- |
| CPU offload of the vision tower or some layers | about 0.8–4 GiB | **Yes.** CPU bf16 kernels differ from MPS kernels. It would be a new numerical configuration, and the frozen comparison would break. | Rejected |
| `logits_to_keep=1` (last-position logits only) | about 0.5 GiB during backward | Possibly at the bit level: a different matmul shape can choose a different kernel | Rejected. The saving is small, and it would need its own equivalence test. |
| Loading directly to MPS (`device_map`) | Load-time transient | The weights are the same, but the load path differs from the frozen call | Not needed. The "double copy" is file-backed mmap cache, which is reclaimable (peak RSS was 0.6 GiB). |
| Gradient checkpointing, quantization, fp16 | various | Quantization and fp16 do; checkpointing risks bit differences | Rejected |
| Gate the fp32 vision stage on live memory | about 0.76 GiB of weights plus activations, only when unsafe | No | **Adopted** (EMP-1) |

The plan keeps the **numerical configuration identical to Checkpoint 2.1**:
- bf16 weights on MPS and eager attention;
- the same `from_pretrained` call;
- the frozen gradient, step, projection, and scoring functions.

Only the **memory-headroom rule** changes.

## 4. Proposal: EMP-1, and a guarded one-scene smoke test

### EMP-1 (code: `src/path_probe/path_probe/memory.py`, opt-in with `--memory-policy emp1`)

**Before load**, all three must hold:
- available memory ≥ **7.79 GiB**. That is the largest observed Qwen bf16 working set (5.61 GiB driver plus 0.68 GiB RSS) plus a 1.5 GiB margin;
- macOS pressure level is 1 (normal);
- free disk ≥ 10 GiB, for swap.

The frozen guard's decision is still computed and stored next to it, and the record states that the frozen guard refused.

**During the run**, a background thread samples OS counters only (never MPS) every 0.5 s: the pressure level, available memory, swap use, and the process footprint, current and lifetime peak.

An abort is requested if any of these happens:
- the pressure is **critical** (level 4) once;
- the pressure stays at **warn** (level 2) for 120 s;
- available memory stays below 0.75 GiB for 30 s;
- swap grows by more than 6 GiB.

The abort takes effect at the next safe point: before any forward or backward call. The run then stops cleanly, saves progress, and frees the model.

If pressure is still critical 60 s after an abort was requested, the process exits (`os._exit(3)`). Records are flushed per scene, so at most one scene is lost.

The fp32 vision stage runs only if pressure is normal and available memory is ≥ 3.26 GiB (2.5 GiB plus the fp32 increase) at conversion time. Otherwise it is skipped and recorded as a limitation, as the plan already requires.

The frozen guard path is unchanged and is still the default (`--memory-policy frozen`).

### Smoke test

```bash
cd ~/Development/vlm-crossmodal-attack-transfer/repo && PYTHONPATH=$(ls -d src/*/ | tr '\n' ':') caffeinate -i ~/Development/_c0-work/venv-vlm/bin/python -m path_probe smoke --memory-policy emp1
```

- **Scope:** one scene (`dev4-ref-00-a`), both directions, PGD-3, the line scan, the controls, and the fp32 vision stage with its exact restore. About 10–15 min.
- **Output:** `paths_smoke/`, never `paths/`.

The full run proceeds only if **every** check passes:

| # | Check | Pass |
| --- | --- | --- |
| 1 | Completes without an abort or error | yes |
| 2 | Reproduction: neutral-source FGSM, random, and PGD-3 official NLL, margin, and decision, both contexts, against the frozen 2.1 record | **bit-exact** (6 of 6) |
| 3 | Determinism: the clean float-path score computed twice | difference = 0 |
| 4 | Pressure | never critical; total time at warn < 60 s |
| 5 | Swap growth | < 2 GiB |
| 6 | Process peak footprint | ≤ 7.5 GiB |
| 7 | Restore check after fp32 vision | exact |
| 8 | Scene time | ≤ 1.5× the estimate. Otherwise the cap simply measures fewer scenes, reported in advance. |

If check 2 fails, the run stops. A mismatch would mean the numerical configuration is not what it claims, and that is reported, not corrected.

### Records

- Every record of this run carries `memory_policy: EMP-1` and the frozen guard's refusal.
- `numerical_configuration` reads "identical to Checkpoint 2.1", backed by check 2.
- The plan deviation is logged here: `EXPLORATORY_PLAN.md` §0 said the frozen guard would be reused unchanged. Under EMP-1 it is computed and recorded but not binding.

## 5. If EMP-1 is not acceptable

The smallest change that runs under the **frozen** guard is about 1.2 GiB more available memory: 11.51 GB needed against 9.5–10.4 GB measured today. Closing one large application, for example Chrome or Cursor, before the run would do it.

Unattended overnight runs often meet it without closing anything: the 2.0 and 2.1 runs started with 11.6 and 12.5 GiB available.

## 6. Smoke results (2026-11-09, scene `dev4-ref-00-a`)

| Attempt | Commit | Outcome |
| --- | --- | --- |
| 1 | `4917f63`, `a2f…` | The EMP-1 preload passed, but the frozen loader received the frozen refusal and refused again. **No weights were loaded.** Fixed: EMP-1 now hands the frozen loader its own recorded permit. The loader is unchanged. |
| 2 (`paths_smoke_attempt2/`) | `ebb6f7b` | Stage B completed. Stage 3 crashed on my bug: the vision tower is `model.model.visual` in transformers 5.22. |
| 3 (`paths_smoke/`) | this commit | Completed, all stages |

Attempts 2 and 3, against the §4 criteria:

| # | Criterion | Attempt 2 (started at 10.02 GiB available; frozen guard refused) | Attempt 3 (started at 12.33 GiB; frozen guard would have permitted) | Result |
| --- | --- | --- | --- | --- |
| 1 | Completes | Stage B only | yes | pass (3) |
| 2 | Bit-exact against frozen 2.1 (clean, FGSM, random, PGD-3; both contexts) | 8 / 8 | 8 / 8 | **pass** |
| 3 | Determinism repeat | 0 | 0 | pass |
| 4 | Pressure | level 1, except one 0.5 s sample at level 2 | level 1 throughout | pass |
| 5 | Swap growth | 0 | 0 | pass |
| 6 | Peak footprint ≤ 7.5 GiB | **8.41 GiB** | **8.41 GiB** in stage B; **9.41 GiB** in fp32 vision | **FAIL** |
| 7 | fp32-vision restore exact | — | 406,957,056 elements converted and restored; clean NLL and margin identical after restore | pass |
| 8 | Scene time ≤ 1.5× the estimate | 184 s (estimate 198) | B 185 s; V32 121 s (estimate 138) | pass |

Minimum available memory was 4.32 GiB in attempt 2 and 4.08 GiB in attempt 3.

**Why criterion 6 failed.**
- The process footprint reaches 8.3–8.4 GiB during **every** backward pass, and falls to about 5.1 GiB between passes. The value is identical across attempts 2 and 3.
- EMP-1's 6.29 GiB "observed working set" came from between-scene samples in the frozen runs. Those samples missed in-backward peaks. That is the caveat §1 flagged.
- The frozen runs ran the same code, so they almost certainly peaked at the same level, unseen.
- The criterion is **not** relaxed here. Recalibrating it is a decision for the research lead (decision M1, below).

**Observation, one scene, not a result.** The bf16 and fp32-vision NLL gradients at the clean image differ a lot:
- cosine 0.51 (neutral) and 0.63 (helpful);
- sign agreement 82–84%;
- L1 ratio 2.3 and 1.6.

This is the kind of precision effect the experiment is designed to measure. Nothing should be read into a single scene.

**Decision M1 (pending).** Recalibrate EMP-1 to the measured peaks:
- the preload line becomes available ≥ 8.41 + 1.5 = **9.91 GiB**;
- criterion 6 becomes a monitor abort if the footprint exceeds **10.0 GiB**;
- the fp32-vision gate is unchanged.

Every other rule stays as written.
