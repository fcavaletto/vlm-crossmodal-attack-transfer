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
