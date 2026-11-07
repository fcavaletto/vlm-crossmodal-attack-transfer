# Checkpoint 2.0 CUDA readiness image.
# The default command is a preflight. It does not load Qwen and it does not
# run an attack. Build context is the repository root:
#   docker build -f docker/cuda/Dockerfile -t study2-cuda-readiness .

FROM python:3.12.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    PYTHONPATH=/opt/study/src/cuda_readiness:/opt/study/src/attack_feasibility:/opt/study/src/qwen_rgb_loss:/opt/study/src/qwen_rgb:/opt/study/src/choice_eval:/opt/study/src/vlm_iface:/opt/study/src/pilot:/opt/study/src/dev_pilot:/opt/study/src/analysis:/opt/study/src/scene_v4

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
COPY docs/checkpoint-1.5/run_model_feasibility.py /opt/study/docs/checkpoint-1.5/run_model_feasibility.py
COPY docs/checkpoint-1.9/ATTACK_PROTOCOL.md /opt/study/docs/checkpoint-1.9/ATTACK_PROTOCOL.md
COPY docs/checkpoint-1.9/attack_feasibility/progress.json /opt/study/docs/checkpoint-1.9/attack_feasibility/progress.json
COPY docs/checkpoint-1.9/attack_feasibility/summary.json /opt/study/docs/checkpoint-1.9/attack_feasibility/summary.json
COPY docs/models/measured-environment.txt /opt/study/docs/models/measured-environment.txt
COPY fixtures/development_v4/manifest.jsonl /opt/study/fixtures/development_v4/manifest.jsonl
COPY fixtures/development_v4/checksums.json /opt/study/fixtures/development_v4/checksums.json

ENTRYPOINT ["python", "-m", "cuda_readiness"]
CMD ["preflight"]
