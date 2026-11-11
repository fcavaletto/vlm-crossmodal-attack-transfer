# What was validated

DEVELOPMENT ONLY. The CUDA commands below were not executed on CUDA. This authoring machine has no NVIDIA driver (`nvidia-smi` is absent), no Docker client, and no `torch` import. `python -m cuda_readiness preflight` reported `cuda_available: null`, `torch_installed: false`, and `cuda_numerical_validation: not_executed`. No CPU score was substituted for a CUDA score.

## Checked here

- `tests/cuda_readiness` exercises the ledger, the held-out path refusal, the preflight exit, the manifest hashes, and the arithmetic that reproduces `planning_estimate.json` from `docs/checkpoint-1.9/attack_feasibility/progress.json`.
- The bfloat16 calibrated projection recomputed from `docs/checkpoint-1.5/run_model_feasibility.py` is 11,509,447,200. The float32 projection is 20,871,410,752. The 6 GiB floor is 6,442,450,944.
- Package indexes resolve the pinned names: `torch==2.15.0` and `torchvision==0.29.2` on `https://download.pytorch.org/whl/cu132` (published as `2.15.0+cu132` and `0.29.2+cu132`), and the remaining pins on PyPI, including `transformers==5.22.1`. The wheels were not installed.
- Docker Hub's tag API returns `python:3.12.14-slim-bookworm`. The image was not pulled.

## Commands for a CUDA host

These were not executed on CUDA.

```bash
docker build -f docker/cuda/Dockerfile -t study2-cuda-readiness .
docker run --gpus all --rm study2-cuda-readiness preflight --require-cuda
docker run --gpus all --rm --entrypoint python study2-cuda-readiness \
  -m pytest tests/cuda_readiness -q
```

`preflight --require-cuda` checks visibility. Pytest for this package does not load Qwen. A passing GPU preflight is still not a numerical validation of the scoring path. The Checkpoint 1.9 measurement command is MPS-only and is not part of this list.
