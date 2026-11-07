# On-demand GPU run guide

DEVELOPMENT ONLY. This image prepares the Checkpoint 1.9 scoring path. The default command is a preflight. It does not load Qwen, does not open a held-out scene, and does not take an attack step.

Model: `Qwen/Qwen3-VL-2B-Instruct`, revision `89644892e4d85e24eaac8bacfd4f463576704203`, bfloat16, eager attention. The answer prefix and the bare color ids stay the Checkpoint 1.7 lock, recorded in `manifest.json`. Loss, decision, pixel path, and the attack step are the files hashed there.

## Build

From the repository root. The image was not built in the checkpoint that added this file.

```bash
docker build -f docker/cuda/Dockerfile -t study2-cuda-readiness .
```

The base image is `python:3.12.14-slim-bookworm`. Torch `2.15.0` and torchvision `0.29.2` come from `https://download.pytorch.org/whl/cu132`. The other pins match `docs/models/measured-environment.txt` and are listed in `docker/cuda/constraints.txt`. PyTorch 2.15 CUDA 13.2 wheels need an NVIDIA driver at or above 580. An L4 is compute capability 8.9; the cu132 wheel includes 8.6, which covers that card. Confirm the driver on the host before expecting `torch.cuda.is_available()` to be true. Weights are not baked into the image. `HF_HUB_OFFLINE=1` is set.

Mount a cache that already contains that revision if a later authorized load needs it:

```bash
docker run --gpus all --rm \
  -e HF_HOME=/models \
  -v "$HF_HOME":/models:ro \
  study2-cuda-readiness \
  preflight --require-cuda
```

`--require-cuda` exits 2 when CUDA is not visible. A true result from that command means the device is visible. It does not mean an image was scored.

## Bookkeeping

`python -m cuda_readiness manifest` writes the development manifest hash, the protocol hash, and the hashes of the scoring files. It reads `fixtures/development_v4/manifest.jsonl` and `checksums.json`. It does not read a PNG.

`python -m cuda_readiness hash path/to/result.json` prints SHA-256 digests. A path component named `heldout`, `held-out`, `held_out`, or `sealed` is refused before the file is read.

The ledger in `cuda_readiness.ledger` stores finished unit ids and result hashes. A second write of the same unit keeps the first digest. `record_refusal` stores `stopped_budget` or `stopped_oom` with the available byte count. That process cannot record another unit or another refusal. `continue_in_new_process` copies the finished units and keeps the refusal in the history. The new ledger still has `model_from_pretrained_called` false. Nothing in this package calls `from_pretrained`.

## What this image does not start

`python -m attack_feasibility` remains the Checkpoint 1.9 MPS runner. It requests MPS, and the Checkpoint 1.5 budget decision still requires MPS. This image does not patch that runner, and it does not lower the 11,509,447,200 byte line. Do not point the container at a held-out directory.

The planning numbers, the GPU class, and the $0.9776 hourly rate are in `PLANNING_ESTIMATE.md`.
