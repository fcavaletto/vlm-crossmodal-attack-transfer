# Checkpoint 2.1 CUDA validation

DEVELOPMENT ONLY. This prepares one NVIDIA L4 on RunPod and a one-scene comparison of MPS with CUDA. The frozen attack is unchanged. No billable GPU was launched. The 96-scene attack was not run. No held-out file was opened. No sidecar from MPS or CUDA is in this commit.

MPS and CUDA measurements are separate records. A cosine, a shared sha256, or an equal loss does not make one a substitute for the other. `devices_interchangeable` is false on the plan and on every comparison report.

## Recommended GPU

One NVIDIA L4 on RunPod Secure Cloud, at **$0.59 per GPU-hour**.

The RunPod pricing page, https://www.runpod.io/pricing, says "Updated November 4, 2026". It was read on 2026-11-07. Its structured offer names the "L4 per-hour on-demand rate on Runpod Secure Cloud" at price 0.59. The same page lists the L4 at 24 GB VRAM, 50 GB RAM, and 12 vCPUs. Cirrun's index, https://cirrun.com/providers/runpod.html, checked 2026-11-07, lists "L4 Secure" at $0.59 per GPU-hour and says it reads RunPod's pricing page. The quoted price is that Secure Cloud rate.

The Community Cloud L4 rate on both pages is $0.44 per GPU-hour. The pod spec requests Secure Cloud, so $0.44 is a different listing. Cirrun's 2026-11-07 note says RunPod bills while the machine is on. A future pod would also be billed for the image pull and for any time it sat idle. Network-volume storage is separate from the GPU-hour rate. This commit did not start a pod, did not create a volume, and has no measured bill.

The L4 is the 24 GB class already used as the Checkpoint 2.0 memory floor. It is Ada Lovelace, compute capability 8.9, with native bfloat16. The cu132 wheel includes compute capability 8.6, which covers 8.9. The catalog host listing is 50 GB of RAM, so the unchanged bfloat16 OS-available line of 11,509,447,200 bytes can clear on the host. That line is host memory. The 24 GB figure is GPU memory. The comparison keeps those pools separate.

## RunPod preparation

The pod record is `docker/runpod/pod-spec.json`. `launched`, `billable_gpu_launched`, and `apiCallMade` are false, and `doNotPost` is true. Nothing in this checkout calls `runpodctl` or the RunPod API.

| Pod field | Value |
| --- | --- |
| Cloud | Secure Cloud, one GPU |
| GPU | NVIDIA L4 |
| CUDA floor | 13.2 |
| Driver floor | 580, the minimum recorded for the cu132 wheel |
| Container disk | 40 GB |
| Network volume | not created here; mount an existing Hub cache at `/models` |
| Start command | `python -m cuda_validation protocol` |
| Experiment command | `python -m cuda_validation measure --device cuda` |
| Experiment executed | false |

The image is `docker/cuda/Dockerfile`, tag `study2-cuda-readiness`. It was the Checkpoint 2.0 readiness image at `95089070ef11fc2e35bee6026f391efef36c0947`, plus `src/cuda_validation` on `PYTHONPATH`. The default command is still `preflight`. Torch 2.15.0 and torchvision 0.29.2 come from `https://download.pytorch.org/whl/cu132`. The other pins match `docs/models/measured-environment.txt`. `HF_HUB_OFFLINE=1` is set. Weights are not baked into the image.

This machine has no Docker client, so the image was not built and not pushed. A host whose driver is older than 580 fails the CUDA 13.2 wheel. The measure command stops before `from_pretrained` when the driver string is missing or below 580, when the visible name is not an L4, or when the capability is not 8.9. L40 and L40S do not satisfy the L4 name check. There is no CPU fallback and no second wheel.

The Mac runner's `prepare_environment` clears `HF_HUB_OFFLINE` and accepts only `/Users/fedecava/Development/models/huggingface`. The validation command does not call it. On the pod, `HF_HOME=/models` and `HF_HUB_OFFLINE=1` stay set. If `model.safetensors` for revision `89644892e4d85e24eaac8bacfd4f463576704203` is absent, the command stops and does not download it. On the Mac, point `HF_HOME` at the study cache that already holds that revision.

`select_dtype` is not called. It can choose float32 when that byte line clears, and its permit requires MPS. Both sides of this comparison load bfloat16. A 50 GB host may clear the float32 line of 20,871,410,752 bytes. The record stores that fact as `float32_budget_would_pass` and still does not load float32.

`budget_decision` sets `permitted` only when MPS is available. That permit is an MPS gate. The CUDA load uses the same byte lines and a CUDA bfloat16 matmul probe. The probe is marked `is_mps_probe: false`. An MPS matmul is not run in its place, and a CUDA matmul is not labeled as the MPS probe.

The memory lines are recomputed from `docs/checkpoint-1.5/run_model_feasibility.py` and refused if they move:

| Line | Bytes |
| --- | ---: |
| OS floor | 6,442,450,944 |
| Bfloat16 calibrated projection | 11,509,447,200 |
| Float32 calibrated projection | 20,871,410,752 |

The guard was not lowered.

Build and the visibility check, from a checkout, after a cache is mounted:

```bash
docker build -f docker/cuda/Dockerfile -t study2-cuda-readiness .
docker run --gpus all --rm \
  --entrypoint python \
  -v "$PWD":/opt/study \
  -v "$HF_HOME":/models:ro \
  -e HF_HOME=/models \
  -e HF_HUB_OFFLINE=1 \
  -w /opt/study \
  study2-cuda-readiness \
  -m cuda_readiness preflight --require-cuda
```

`--require-cuda` exits 2 when CUDA is not visible. A true result means the device is visible. It does not mean an image was scored. These two commands were not run here.

## The one-scene experiment

The scene is `dev4-ref-00-a`, split `development`, image `fixtures/development_v4/images/dev4-ref-00-a.png`. One scene. The command rejects any other scene id, `scene_count` other than 1, and a path component named `heldout`, `held-out`, `held_out`, or `sealed`. It also rejects an output path under `fixtures/`, `docs/checkpoint-1`, or `docs/checkpoint-2.0`.

The attack is the frozen configuration, called through the existing functions. `run_three_steps` calls `attack_feasibility.steps.pgd_update`. That update is ascent by `epsilon/4` and then `project_linf` onto the ball around the clean image. The official image is `round_trip_uint8`, half-to-even. This package does not define those functions.

| Piece | Value |
| --- | --- |
| Model | `Qwen/Qwen3-VL-2B-Instruct` |
| Revision | `89644892e4d85e24eaac8bacfd4f463576704203` |
| Weights | bfloat16, eager attention |
| RGB leaf | float32 on CPU, as `backward_at_image` already builds it |
| Source context | neutral |
| Evaluation context | `helpful_nonrevealing` is scored on the clean official image and is not a source attack |
| Epsilon | 1/255 |
| Steps | 3 |
| Step size | epsilon/4 |
| Init | clean image |
| Restarts | 0 |

`attack_feasibility` remains the MPS runner for the Checkpoint 1.9 grid: three budgets and seven steps. This experiment does not call `run_scene`. Three steps is the frozen Checkpoint 2.1 configuration. Seven remains the feasibility grid's length.

The prompt text is checked against the Checkpoint 1.7 `user_text_sha256` for this scene under neutral and `helpful_nonrevealing`. The stored Checkpoint 1.7 losses are an MPS clean run. They are not a CUDA target.

Each device writes one JSON sidecar and, when `--output` is set, a float32 gradient file next to it. The ignored directory for those files is `docs/checkpoint-2.1/cuda_validation/`.

Model outputs, for red, blue, green, and yellow:

- clean official summed NLL, chosen color, and labeled loss, under neutral
- clean differentiable summed NLL and labeled loss, under neutral
- final official summed NLL after the 8-bit round trip, under neutral
- clean official summed NLL under `helpful_nonrevealing`

Losses: the labeled-color trajectory on the differentiable neutral path. The clean loss, the loss after step 1, the loss after step 2, and the loss after step 3. Length 4.

Gradient: the clean-image float32 RGB gradient. The sidecar stores sha256, shape, L1, and L2. The `.gradient.f32` file is what `compare` reads. Cosine, relative L2, max absolute entry gap, and sign agreement are computed at compare time. Sign agreement counts a zero as its own sign. Later steps recompute the gradient inside the three-step loop; the compared array is the clean gradient.

Attack bounds, checked on each device alone:

- epsilon is 1/255, step size is epsilon/4, the start is the clean image, and restarts are 0
- after every step, the continuous L-infinity distance from the clean image is at most epsilon, with absolute tolerance 1e-6
- pixels stay in [0, 1]
- the official 8-bit image may sit outside the continuous ball by at most half a level, 0.5/255, plus that same tolerance

The other device's pixels are not part of the bound. Two legal images can differ. The Checkpoint 2.0 calibration distances on MPS are a description of that MPS run. They are not a CUDA acceptance target. The sidecar field `bound_check` is that device's own result. The pair is judged again by `compare`.

## Differences that have to be checked

The compare report copies this list into `must_check`. A finished pair still carries it.

1. Device and chip. The MPS record is the Mac GPU. The CUDA record is the L4 named above. One record is not relabeled with the other device.
2. Compute capability. The quoted CUDA card is 8.9. MPS does not report a CUDA capability. A null field is not capability 0.
3. Driver and runtime. The cu132 wheel needs an NVIDIA driver at or above 580. MPS has no NVIDIA driver and no CUDA runtime. A null CUDA version is not CUDA 0.
4. Torch build. The image pins torch 2.15.0 from the cu132 index. The Mac environment records the same version string from the measured environment. Matching version strings are different binaries. The sidecar stores `torch.__version__` from the process that ran.
5. Dtypes. Parameter dtype is bfloat16. The RGB leaf differentiated by the frozen backward is float32. Eager attention is required on both sides. sdpa or flash attention blocks the comparison. Float32 weights are not loaded.
6. TF32. CUDA exposes `allow_tf32` for matmul and for cuDNN. The frozen code does not set those flags, and this experiment does not flip them. The sidecar stores the process values. MPS has no TF32 switch. A null MPS flag is not the boolean false, and a CUDA record that omits the boolean is `blocked_precision_record`.
7. cuDNN benchmark, cuDNN deterministic, and float32 matmul precision. They are read. A CUDA boolean is not written into the MPS record.
8. Bfloat16 accumulation. MPS and CUDA do not reduce in the same order. When the torch build exposes `allow_bf16_reduced_precision_reduction`, the CUDA record stores it. There is no cutoff that turns a gradient gap into a pass. The Checkpoint 1.8 same-device BF16 repeat ceiling is a repeat check on one device. It is not applied here.
9. Memory pools. The pre-load guard is OS-available bytes, on the lines in the table above. CUDA allocated and reserved bytes are VRAM. MPS driver-allocated bytes are unified memory. The report stores the three readings under `memory` and sets `memory_quantities_are_distinct`. It does not subtract one from another. The MPS `budget_decision` permit is not a CUDA permit.
10. The 8-bit round trip. Half-to-even quantization can move a pixel by at most half a level, so each stored image is allowed 0.5/255 outside the continuous ball. The two stored images are checked separately.
11. The Mac cache path and the pod cache path. `prepare_environment` is the Mac runner's HF_HOME rule. This command leaves `HF_HUB_OFFLINE` as it found it. A weight load still has to be the pinned revision in bfloat16 on the requested device.

`numeric_values_equal` is true when the stored summed NLLs, the loss trajectory, and the gradient entries match, entry for entry, or when both gradient sha256 values match. Cosine is stored as computed. Identical entries can produce a cosine one ulp under 1, because the square roots in the denominator do not multiply back to the sum of squares. That cosine is not rounded to 1, and it is not a pass line. `identity_threshold_applied` is false.

## Reading the report

`python -m cuda_validation compare mps_scene.json cuda_scene.json` writes the report to stdout.

| Status | Meaning |
| --- | --- |
| `differences_recorded` | Both sidecars were read, the hardware fields are present, each image satisfies its own bound, and the numeric gaps are in the report |
| `bound_violation` | At least one device left the ball, the unit interval, or the half-level allowance |
| `blocked_protocol_mismatch` | Scene, revision, dtype, attention, epsilon, step count, or trajectory length disagrees with the frozen configuration |
| `blocked_precision_record` | An MPS flag was stored as a CUDA boolean, or a CUDA flag was missing |
| `incomplete_hardware_record` | A required hardware field is absent |
| `incomplete_gradient_record` | The gradient arrays were not both available, so cosine was not computed |
| `incomplete_gradient_file` | A gradient file was present and did not match its sha256 or its length |

Exit 0 means the status is `differences_recorded`. Exit 0 does not mean the numbers agree. Exit 3 means one of the other statuses. Exit 2 means the files could not be read as one MPS sidecar and one CUDA sidecar. Asking the command to treat the devices as identical raises, and a sidecar that already claims `devices_interchangeable` raises.

## What was run in this checkout

`python -m cuda_validation protocol` was run. It reported `billable_gpu_launched: false`, `runpod_api_called: false`, `model_from_pretrained_called: false`, `attacks_run: false`, `held_out_scenes_opened: false`, `executed_on_cuda: false`, `executed_on_mps: false`, and `cuda_numerical_validation: not_executed`. The attack-file hashes match `docs/checkpoint-2.0/cuda/manifest.json`. The recomputed lines are 11,509,447,200, 20,871,410,752, and 6,442,450,944.

`python -m cuda_readiness preflight` on this interpreter reported `torch_installed: false`, `cuda_available: null`, `nvidia_smi_on_path: false`, and `docker_on_path: false`. `nvidia-smi` and `docker` are absent. No CPU score was put in their place.

`python3 -m pytest tests/cuda_validation tests/cuda_readiness -q` reported 26 passed and 1 skipped. The skip is `test_frozen_pgd_update_stays_inside_the_ball`, because torch is not installed. The test that did run checks that three steps call the update with step size epsilon/4, project against the clean image, and take no restart. `measure` was not invoked on MPS or CUDA.

## Commands that were not run

MPS, on the Mac that holds the study cache:

```bash
HF_HOME=/Users/fedecava/Development/models/huggingface \
PYTHONPATH=src/cuda_validation:src/cuda_readiness:src/attack_feasibility:src/qwen_rgb:src/qwen_rgb_loss:src/choice_eval:src/vlm_iface:src/pilot:src/dev_pilot \
python -m cuda_validation measure --device mps \
  --output docs/checkpoint-2.1/cuda_validation/mps_scene.json
```

CUDA, on the L4 pod, after `preflight --require-cuda` has exited 0, and after the pod is confirmed to be an L4 with driver 580 or newer:

```bash
python -m cuda_validation measure --device cuda \
  --output docs/checkpoint-2.1/cuda_validation/cuda_scene.json
```

Comparison, on either machine, after both files exist. This command does not load Qwen:

```bash
PYTHONPATH=src/cuda_validation:src/cuda_readiness:src/qwen_rgb \
python -m cuda_validation compare \
  docs/checkpoint-2.1/cuda_validation/mps_scene.json \
  docs/checkpoint-2.1/cuda_validation/cuda_scene.json
```

Stop the pod after the CUDA sidecar is copied off. The GPU-hour rate applies until the machine is shut down.

## Deployment fixes, 2026-11-08 (before any measurement)

Added in review, before any pod, image build, or measurement. They change how the image is built and run, not the attack, the scene, the gate, or the comparison.

| Blocker | Evidence | Fix |
| --- | --- | --- |
| Weights had no provisioning path. `/models` needed a network volume that nothing created, and `HF_HUB_OFFLINE=1` blocks a download. | Pod spec `volumeInGb: 0`, `networkVolume.createdByThisCommit: false`. | A build stage downloads revision `89644892…` and checks `model.safetensors` against sha256 `7de1838c…` (the LFS blob id in the Mac cache). The final image holds it at `/models`. No network volume or paid storage. |
| The scene and prompt-hash files were not in the image. | `measure` opens `fixtures/development_v4/images/dev4-ref-00-a.png` and the Checkpoint 1.7 `predictions.jsonl`. Neither was copied. | Both are copied. |
| The start command crashed. | `cuda_validation protocol` in the image layout raised `FileNotFoundError` for `docs/checkpoint-2.0/cuda/manifest.json`. | That file and `docker/runpod/pod-spec.json` are copied. `tests/cuda_validation/test_container_layout.py` rebuilds the image tree from the COPY lines and runs `protocol` inside it. It fails on the old Dockerfile and passes on this one. |
| The pod body was not a v2 request. | It used `gpuType`, `gpuCount`, `cloudType`, `containerDiskInGb`, `volumeInGb` and had no `image`, which v2 requires. | Rewritten with v2 fields from the live `create-pod` schema: `image`, `gpu.id`, `gpu.minCudaVersion`, `disk`, `args`, `env`. |
| No way to get results off the pod, and no shutdown plan. | The container disk is wiped on stop or restart. REST v2 has no stop timer. | `docker/cuda/run_job.sh` runs preflight, `measure`, and, if an MPS sidecar was baked in, `compare` on the pod. It prints each JSON between `=== BEGIN`/`=== END` markers in the pod log, prints `JOB_DONE`, then sleeps so an exit cannot restart the job. The operator saves the log and deletes the pod. |
| No image registry or build machine. | This Mac has no Docker client. | `.github/workflows/cuda-image.yml` builds and pushes to GHCR, manual trigger only. Pin the printed digest in the pod body. GHCR packages start private: make the package public or add a registry credential. |

The image default command is still `preflight`. The job runs only through the pod's `args` override.

Still open:

- The MPS sidecar must be measured on the Mac first and placed in `docs/checkpoint-2.1/cuda_validation/` before the image is built, so that `compare` runs on the pod. The 2.4 MB gradient is too large to move reliably through the log.
- `measure` accepts only an L4 at capability 8.9 (`is_quoted_l4`). An RTX A6000 (capability 8.6, 48 GB, same $0.59 rate in the 2026-11-08 read) would be refused. Allowing it is a protocol change. It should be written down before any measurement, not after.
- Secure Cloud L4 and RTX A6000 stock was `Low` on CUDA 13.2 hosts and `Out` on 12.8 and 13.0 in the 2026-11-08 capacity read. The image needs 13.2 or newer.
