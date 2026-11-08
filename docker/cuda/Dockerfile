# CUDA readiness image used by the Checkpoint 2.1 RunPod validation.
# The default command is a preflight. It does not load Qwen and it does not
# run an attack. The one-scene comparison is `python -m cuda_validation`.
# Build context is the repository root:
#   docker build -f docker/cuda/Dockerfile -t study2-cuda-readiness .

# The Qwen weights are not in the image. run_job.sh downloads the pinned
# revision on the pod (docker/cuda/fetch_weights.py) and refuses to measure
# unless every file matches docker/cuda/qwen_files.sha256, which was hashed
# from the Mac study cache. This keeps third-party weights out of a public
# registry and keeps the image small enough for a standard Actions runner.
FROM python:3.12.14-slim-bookworm

ENV HF_HOME=/models \
    TRANSFORMERS_OFFLINE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    PYTHONPATH=/opt/study/src/cuda_validation:/opt/study/src/cuda_readiness:/opt/study/src/attack_feasibility:/opt/study/src/qwen_rgb_loss:/opt/study/src/qwen_rgb:/opt/study/src/choice_eval:/opt/study/src/vlm_iface:/opt/study/src/pilot:/opt/study/src/dev_pilot:/opt/study/src/analysis:/opt/study/src/scene_v4

WORKDIR /opt/study

COPY docker/cuda/requirements.txt /tmp/requirements.txt
COPY docker/cuda/constraints.txt /tmp/constraints.txt
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu132 \
        "torch==2.15.0" "torchvision==0.29.2" \
    && pip install --no-cache-dir --constraint /tmp/constraints.txt -r /tmp/requirements.txt \
    && python -c "import torch; assert torch.__version__.startswith('2.15.0'), torch.__version__"

COPY src /opt/study/src
COPY tests/cuda_readiness /opt/study/tests/cuda_readiness
COPY tests/cuda_validation /opt/study/tests/cuda_validation
COPY docs/checkpoint-1.5/run_model_feasibility.py /opt/study/docs/checkpoint-1.5/run_model_feasibility.py
COPY docs/checkpoint-1.9/ATTACK_PROTOCOL.md /opt/study/docs/checkpoint-1.9/ATTACK_PROTOCOL.md
COPY docs/checkpoint-1.9/attack_feasibility/progress.json /opt/study/docs/checkpoint-1.9/attack_feasibility/progress.json
COPY docs/checkpoint-1.9/attack_feasibility/summary.json /opt/study/docs/checkpoint-1.9/attack_feasibility/summary.json
COPY docs/models/measured-environment.txt /opt/study/docs/models/measured-environment.txt
COPY fixtures/development_v4/manifest.jsonl /opt/study/fixtures/development_v4/manifest.jsonl
COPY fixtures/development_v4/checksums.json /opt/study/fixtures/development_v4/checksums.json
# Files `cuda_validation protocol` and `measure` read. Without them the start
# command raised FileNotFoundError and `measure` could not open the scene.
COPY fixtures/development_v4/images/dev4-ref-00-a.png /opt/study/fixtures/development_v4/images/dev4-ref-00-a.png
COPY docs/checkpoint-1.7/development_v4/qwen3-vl-2b/predictions.jsonl /opt/study/docs/checkpoint-1.7/development_v4/qwen3-vl-2b/predictions.jsonl
COPY docs/checkpoint-2.0/cuda/manifest.json /opt/study/docs/checkpoint-2.0/cuda/manifest.json
COPY docker/runpod/pod-spec.json /opt/study/docker/runpod/pod-spec.json
COPY docs/checkpoint-2.1/cuda_validation/ /opt/study/docs/checkpoint-2.1/cuda_validation/
COPY docker/cuda/run_job.sh /opt/study/run_job.sh
COPY docker/cuda/fetch_weights.py /opt/study/docker/cuda/fetch_weights.py
COPY docker/cuda/self_stop.py /opt/study/docker/cuda/self_stop.py
COPY docker/cuda/qwen_files.sha256 /opt/study/docker/cuda/qwen_files.sha256

ENTRYPOINT ["python", "-m", "cuda_readiness"]
CMD ["preflight"]
