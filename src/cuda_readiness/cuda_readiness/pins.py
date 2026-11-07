"""Pins for the CUDA image. These match the Checkpoint 0 measured environment.

The torch index is the Linux CUDA 13.2 wheel index for that same torch
version. Nothing here changes the loss, the decision, the pixel path, or the
attack step.
"""

from __future__ import annotations

BASE_COMMIT = "0cd2f2a8e566b6530dcfd4dd7021961b2fc17990"
PROTOCOL_COMMIT = "fce3480d158298851587a692c56b6e4eebb554a7"

# docs/models/measured-environment.txt, the versions the Mac run recorded.
DEPENDENCY_PINS = {
    "python": "3.12.14",
    "torch": "2.15.0",
    "torchvision": "0.29.2",
    "transformers": "5.22.1",
    "numpy": "2.5.4",
    "pillow": "13.0.0",
    "accelerate": "1.15.0",
    "huggingface_hub": "2.4.0",
    "num2words": "0.5.14",
    "psutil": "7.2.2",
    "pytest": "9.2.0",
}

TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu132"
CUDA_WHEEL = "cu132"
# PyTorch 2.15 publishes CUDA 13.2 wheels and states that those wheels need
# an NVIDIA driver new enough for CUDA 13.2. The release note names 580.
NVIDIA_DRIVER_MINIMUM = "580"

# Checkpoint 1.5 lines. The estimate reads the budget script and refuses to
# continue if the computed lines are not these values.
BF16_REQUIRED_AVAILABLE_BYTES = 11_509_447_200
FLOAT32_REQUIRED_AVAILABLE_BYTES = 20_871_410_752
OS_FLOOR_BYTES = 6_442_450_944
BF16_WEIGHT_BYTES = 4_255_064_064

# Written sealed-set size in docs/research/RESEARCH_PROPOSAL.md section 12.
# This is not a directory listing. No held-out file is opened to obtain it.
HELD_OUT_PLANNING_SCENE_COUNT = 256
DEVELOPMENT_SCENE_COUNT = 96

# Quoted planning class. Host memory is 32 GiB so the unchanged 11.51 GB
# OS-available line can clear. The 16 GiB g6.xlarge is not the quoted class.
GPU_INSTANCE_TYPE = "g6.2xlarge"
GPU_REGION = "us-east-1"
GPU_NAME = "NVIDIA L4"
GPU_HOURLY_RATE_USD = "0.9776"
GPU_MEMORY_CLASS = "24 GB"
GPU_HOST_MEMORY_GIB = 32
GPU_VCPUS = 8

# Smaller of the published L4 listings used as a capacity floor. 22 GiB is
# 22 * 1024**3. Advertised 24 GB decimal is larger. Neither number was read
# from a CUDA allocator.
COMPARISON_FLOOR_BYTES = 22 * 1024**3

MEASURED_ENVIRONMENT_RELATIVE = "docs/models/measured-environment.txt"
PROGRESS_RELATIVE = "docs/checkpoint-1.9/attack_feasibility/progress.json"
SUMMARY_RELATIVE = "docs/checkpoint-1.9/attack_feasibility/summary.json"
BUDGET_SCRIPT_RELATIVE = "docs/checkpoint-1.5/run_model_feasibility.py"
PROTOCOL_RELATIVE = "docs/checkpoint-1.9/ATTACK_PROTOCOL.md"
MANIFEST_RELATIVE = "fixtures/development_v4/manifest.jsonl"
CHECKSUMS_RELATIVE = "fixtures/development_v4/checksums.json"

SCIENCE_FILES = (
    "src/qwen_rgb/qwen_rgb/loss.py",
    "src/qwen_rgb/qwen_rgb/constraints.py",
    "src/qwen_rgb/qwen_rgb/rgb_path.py",
    "src/attack_feasibility/attack_feasibility/score.py",
    "src/attack_feasibility/attack_feasibility/steps.py",
    "src/attack_feasibility/attack_feasibility/stage_a.py",
    "src/attack_feasibility/attack_feasibility/stage_b.py",
    "src/choice_eval/choice_eval/decision.py",
    "src/dev_pilot/dev_pilot/lock.py",
    "src/dev_pilot/dev_pilot/scoring.py",
)

BANNED_PATH_PARTS = frozenset({"heldout", "held-out", "held_out", "sealed"})


def parse_measured_environment(text: str) -> dict[str, str]:
    """Read ``name==version`` lines from the measured-environment block."""
    pins: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("`") or "==" not in line:
            continue
        name, version = line.split("==", 1)
        if name and version and " " not in name:
            pins[name] = version
    return pins
