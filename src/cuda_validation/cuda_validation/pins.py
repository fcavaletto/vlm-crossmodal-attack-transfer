"""Frozen comparison constants. These do not replace the attack code."""

from __future__ import annotations

from qwen_rgb.pins import MODEL_ID, PINNED_REVISION

MODEL_ID = MODEL_ID
REVISION = PINNED_REVISION
DTYPE = "bfloat16"
ATTN_IMPLEMENTATION = "eager"
SOURCE_CONTEXT = "neutral"
EVAL_CONTEXTS = ("neutral", "helpful_nonrevealing")

SCENE_ID = "dev4-ref-00-a"
SCENE_IMAGE_RELATIVE = "fixtures/development_v4/images/dev4-ref-00-a.png"
SCENE_SPLIT = "development"
SCENE_COUNT = 1

# PGD, epsilon 1/255, 3 steps, step size epsilon/4, clean start, no restarts.
EPSILON_LABEL = "1/255"
EPSILON = 1.0 / 255.0
STEPS = 3
STEP_SIZE_LABEL = "epsilon/4"
STEP_SIZE = EPSILON / 4.0
INIT = "clean_image"
RESTARTS = 0
RGB_LEAF_DTYPE = "torch.float32"

# Checkpoint 1.5 lines. A disagreement with the budget script is a stop.
BF16_REQUIRED_AVAILABLE_BYTES = 11_509_447_200
FLOAT32_REQUIRED_AVAILABLE_BYTES = 20_871_410_752
OS_FLOOR_BYTES = 6_442_450_944

# Half a quantization level. The stored image may sit this far outside the
# continuous L-infinity ball. 1e-6 is above the float32 noise seen on the
# Checkpoint 2.0 calibration distances and below one extra step of epsilon/4.
STORED_HALF_LEVEL = 0.5 / 255.0
BOUND_ATOL = 1e-6

BANNED_PATH_PARTS = frozenset({"heldout", "held-out", "held_out", "sealed"})
FORBIDDEN_OUTPUT_PREFIXES = (
    "docs/checkpoint-1",
    "docs/checkpoint-2.0",
    "fixtures",
)

# Quoted RunPod card. The page was read 2026-11-07. No pod was started.
GPU_NAME = "NVIDIA L4"
GPU_CLOUD = "Secure Cloud"
GPU_VRAM = "24 GB"
GPU_HOST_RAM = "50 GB"
GPU_VCPUS = 12
GPU_HOURLY_RATE_USD = "0.59"
GPU_PAGE_UPDATED = "2026-11-04"
GPU_PRICE_READ_ON = "2026-11-07"
GPU_PRICE_SOURCE = "https://www.runpod.io/pricing"
GPU_PRICE_CORROBORATION = "https://cirrun.com/providers/runpod.html"
MIN_CUDA_VERSION = "13.2"
NVIDIA_DRIVER_MINIMUM = 580
COMPUTE_CAPABILITY = "8.9"

MUST_CHECK = (
    "MPS and CUDA are different devices. A record from one is not relabeled as the other.",
    "MPS does not report a CUDA compute capability. Null is not capability 0. The quoted CUDA card is an L4 at 8.9.",
    "The cu132 wheel needs an NVIDIA driver at or above 580. MPS has no NVIDIA driver and no CUDA runtime.",
    "The same torch version string on MPS and on a cu132 wheel is not the same binary.",
    "Weights are bfloat16 and the RGB leaf is float32. select_dtype is not used, because it can select float32 and its permit requires MPS.",
    "CUDA TF32 flags are read and not flipped. MPS has no TF32 switch. Null on MPS is not the boolean false.",
    "cuDNN benchmark, deterministic, and float32 matmul precision are read. A CUDA boolean is not copied onto MPS.",
    "BF16 matmul accumulation order differs by device. No tolerance here turns a gradient difference into a pass.",
    "The pre-load guard is OS-available bytes and is not lowered. CUDA allocated bytes and MPS driver bytes are different pools.",
    "The MPS budget_decision permit requires mps_available. It is not a CUDA permit.",
    "The 8-bit round trip can leave the continuous L-infinity ball by at most half a level. Each device is checked on its own.",
    "The same-device BF16 repeat ceiling is not an MPS-versus-CUDA identity test.",
)
