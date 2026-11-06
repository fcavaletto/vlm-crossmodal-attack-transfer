"""Checkpoint 1.6 feasibility for ``Qwen/Qwen3-VL-2B-Instruct``.

The pre-load budget is the Checkpoint 1.5 function. This script does not lower
it. It remeasures OS-available memory and stops at the first gate that fails.
A failing budget returns before any ``from_pretrained`` call and before any
weight-tensor download.

Gates:

1. OS-available memory, read in this process.
2. BF16 capability, config and safetensors header, then the budget.
3. Minimal MPS forward, only if a dtype cleared gate 2.
4. Twelve development generations, only if gate 3 produced finite logits.
5. Canonical-word image gradients on one scene, only if gate 4 finished.

No PGD, FGSM, noise, training, H* fit, held-out scene, or paid GPU.

Run from the repository root:

    HF_HOME=/Users/fedecava/Development/models/huggingface \\
    PYTHONPATH=src/vlm_iface:src/pilot \\
    /Users/fedecava/Development/_c0-work/venv-vlm/bin/python \\
    docs/checkpoint-1.6/run_qwen_feasibility.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for _rel in ("src/vlm_iface", "src/pilot"):
    _path = str(ROOT / _rel)
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Fixed before any generation function. Development fixtures only.
DEVELOPMENT_SCENE_IDS: tuple[str, ...] = (
    "dev2-attribute-00-a",
    "dev2-counting-00-a",
    "dev2-spatial-00-a",
)
CONTEXTS: tuple[str, ...] = (
    "neutral",
    "helpful_nonrevealing",
    "contradictory",
    "answer_revealing",
)
GRADIENT_SCENE_ID = "dev2-attribute-00-a"
GRADIENT_CONTEXTS: tuple[str, ...] = ("neutral", "helpful_nonrevealing")
MAX_NEW_TOKENS = 16

REPO_ID = "Qwen/Qwen3-VL-2B-Instruct"
PINNED_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
REQUIRED_HF_HOME = "/Users/fedecava/Development/models/huggingface"
OUTPUT_PATH = ROOT / "docs" / "checkpoint-1.6" / "qwen_feasibility.json"
BUDGET_PATH = ROOT / "docs" / "checkpoint-1.5" / "run_model_feasibility.py"
MANIFEST_PATH = ROOT / "fixtures" / "development_v2" / "manifest.jsonl"
REPRODUCTION_COMMAND = (
    "HF_HOME=/Users/fedecava/Development/models/huggingface "
    "PYTHONPATH=src/vlm_iface:src/pilot "
    "/Users/fedecava/Development/_c0-work/venv-vlm/bin/python "
    "docs/checkpoint-1.6/run_qwen_feasibility.py"
)

# Planning constants for the CUDA envelope. They are not measurements.
PLANNING_TEXT_TOKENS = 64
PLANNING_RESIDUAL_COPIES = 8
PLANNING_MLP_MATS = 2
PLANNING_SLACK = 2
PLANNING_HEADROOM_FLOOR_BYTES = 8 * 1024**3

_FROM_PRETRAINED_CALLS = 0


def _prepare_environment() -> dict[str, Any]:
    """Point the Hub cache at the study directory and do not set offline mode."""
    offline_was_set = "HF_HUB_OFFLINE" in os.environ
    if offline_was_set:
        os.environ.pop("HF_HUB_OFFLINE", None)
    home = os.environ.get("HF_HOME")
    if home is None:
        os.environ["HF_HOME"] = REQUIRED_HF_HOME
        home = REQUIRED_HF_HOME
        home_assigned = True
    else:
        home_assigned = False
    if Path(home).resolve() != Path(REQUIRED_HF_HOME).resolve():
        raise SystemExit(f"HF_HOME is {home}, expected {REQUIRED_HF_HOME}. Refusing to use another cache.")
    return {
        "HF_HOME": home,
        "HF_HOME_assigned_by_script": home_assigned,
        "HF_HUB_OFFLINE_was_set": offline_was_set,
        "HF_HUB_OFFLINE_set_by_script": False,
    }


def _load_budget_module():
    """Import the Checkpoint 1.5 budget without running its ``main``."""
    spec = importlib.util.spec_from_file_location("checkpoint15_model_feasibility", BUDGET_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load the budget module at {BUDGET_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.SPARE_REQUIRED_BYTES != 5 * 1024**3:
        raise RuntimeError("Checkpoint 1.5 spare line is not 5 GiB. This script will not replace it.")
    if module.PROJECTED_REMAINING_FLOOR_BYTES != 2 * 1024**3:
        raise RuntimeError("Checkpoint 1.5 projection floor is not 2 GiB. This script will not replace it.")
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _memory(psutil_module) -> dict[str, int]:
    memory = psutil_module.virtual_memory()
    return {
        "total": int(memory.total),
        "available": int(memory.available),
        "used": int(memory.used),
        "free": int(memory.free),
        "active": int(getattr(memory, "active", 0)),
        "inactive": int(getattr(memory, "inactive", 0)),
        "wired": int(getattr(memory, "wired", 0)),
    }


def _machine(memory_bytes: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "platform": platform.platform(),
        "system": platform.system(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "memory_bytes": memory_bytes,
    }
    if platform.system() == "Darwin":
        summary["cpu_brand"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
        summary["hw_memsize"] = int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        )
    return summary


def _snapshot_names(revision: str) -> list[str]:
    folder = (
        Path(os.environ["HF_HOME"])
        / "hub"
        / "models--Qwen--Qwen3-VL-2B-Instruct"
        / "snapshots"
        / revision
    )
    if not folder.is_dir():
        return []
    return sorted(path.name for path in folder.iterdir())


def _large_blobs(min_bytes: int = 1_000_000) -> list[dict[str, Any]]:
    folder = Path(os.environ["HF_HOME"]) / "hub" / "models--Qwen--Qwen3-VL-2B-Instruct" / "blobs"
    if not folder.is_dir():
        return []
    found = []
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.stat().st_size >= min_bytes:
            found.append({"name": path.name, "bytes": path.stat().st_size})
    return found


def _weight_file_present(revision: str) -> bool:
    path = (
        Path(os.environ["HF_HOME"])
        / "hub"
        / "models--Qwen--Qwen3-VL-2B-Instruct"
        / "snapshots"
        / revision
        / "model.safetensors"
    )
    return path.is_file()


def _bf16_probe(torch) -> dict[str, Any]:
    """One small MPS BF16 matmul. This does not load Qwen."""
    probe: dict[str, Any] = {
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_built": bool(torch.backends.mps.is_built()),
        "torch_bfloat16": str(torch.bfloat16),
        "add_ok": False,
        "matmul_ok": False,
        "is_model_forward": False,
    }
    if not probe["mps_available"]:
        probe["error"] = "MPS is not available. A CPU matmul was not run."
        return probe
    try:
        ones = torch.zeros(4, device="mps", dtype=torch.bfloat16)
        added = (ones + torch.tensor(1, device="mps", dtype=torch.bfloat16)).float().cpu()
        probe["add_ok"] = bool(torch.isfinite(added).all()) and added.tolist() == [1.0, 1.0, 1.0, 1.0]
        matrix = torch.randn(8, 8, device="mps", dtype=torch.bfloat16)
        product = matrix @ matrix
        probe["matmul_ok"] = bool(torch.isfinite(product.float()).all()) and str(product.dtype) == "torch.bfloat16"
        probe["matmul_shape"] = [8, 8]
        del ones, added, matrix, product
        torch.mps.empty_cache()
    except Exception as exc:
        probe["error"] = f"{type(exc).__name__}: {exc}"
    return probe


def _compatibility() -> dict[str, Any]:
    """Resolve model classes from the installed transformers. Does not load weights."""
    note: dict[str, Any] = {"weights_loaded": False, "from_pretrained_called": False}
    try:
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
        from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
        import transformers

        fast = getattr(transformers, "Qwen2VLImageProcessorFast", None)
        note.update(
            {
                "model_class": Qwen3VLForConditionalGeneration.__name__,
                "processor_class": Qwen3VLProcessor.__name__,
                "image_processor_class": Qwen2VLImageProcessor.__name__,
                "fast_alias_resolves": fast is Qwen2VLImageProcessor,
                "fast_alias_name": None if fast is None else fast.__name__,
            }
        )
    except Exception as exc:
        note["error"] = f"{type(exc).__name__}: {exc}"
    return note


def _survey(survey_weights) -> dict[str, Any]:
    try:
        surveyed = survey_weights(REPO_ID, revision=PINNED_REVISION)
    except Exception as exc:
        return {
            "repo_id": REPO_ID,
            "revision": PINNED_REVISION,
            "weights_loaded": False,
            "from_pretrained_called": False,
            "survey_error": f"{type(exc).__name__}: {exc}",
        }
    surveyed["weights_loaded"] = False
    surveyed["from_pretrained_called"] = False
    return surveyed


def _dtype_decisions(budget, survey: dict[str, Any], available: int, mps_available: bool, min_available: int) -> dict[str, Any]:
    if survey.get("survey_error") or survey.get("approximate"):
        reason = survey.get("survey_error") or "The safetensors header was not an exact range read."
        refused = {
            "permitted": False,
            "from_pretrained_called": False,
            "reason": f"Refusing the load without an exact header. {reason}",
        }
        return {"float32": refused, "bfloat16": refused}
    parameter_count = int(survey["parameter_count"])
    float32_bytes = int(survey["float32_weight_bytes"])
    decisions = {
        "float32": budget.budget_decision(
            resident_bytes=float32_bytes,
            available_bytes=available,
            ratio=budget.C1_DROP_PER_CURRENT,
            min_available_bytes=min_available,
            mps_available=mps_available,
        ),
        "bfloat16": budget.budget_decision(
            resident_bytes=parameter_count * 2,
            available_bytes=available,
            ratio=budget.C1_DROP_PER_CURRENT,
            min_available_bytes=min_available,
            mps_available=mps_available,
        ),
    }
    decisions["float32"]["dtype"] = "float32"
    decisions["bfloat16"]["dtype"] = "bfloat16"
    decisions["float32"]["note"] = "Resident size is the float32 expansion of the surveyed header."
    decisions["bfloat16"]["note"] = "Resident size is two bytes times the surveyed parameter count."
    for decision in decisions.values():
        decision["from_pretrained_called"] = False
        decision["available_bytes_source"] = "psutil.virtual_memory().available at the decision snapshot"
    return decisions


def _select_dtype(decisions: dict[str, Any], bf16_ok: bool) -> dict[str, Any]:
    """Float32 if its budget passes, otherwise BF16 if its budget and the MPS probe pass."""
    if decisions["float32"].get("permitted"):
        return {"dtype": "float32", "permitted": True, "reason": "The float32 budget passed."}
    bf16 = decisions["bfloat16"]
    if bf16.get("permitted") and bf16_ok:
        return {"dtype": "bfloat16", "permitted": True, "reason": "Float32 failed. The BF16 budget passed and the MPS probe ran."}
    if bf16.get("permitted") and not bf16_ok:
        return {
            "dtype": None,
            "permitted": False,
            "reason": "The BF16 budget passed, but the MPS BF16 probe did not. from_pretrained was not called.",
        }
    return {
        "dtype": None,
        "permitted": False,
        "reason": "Neither dtype cleared the Checkpoint 1.5 pre-load budget.",
    }


def _load_manifest() -> list[dict[str, Any]]:
    rows = []
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _subset_definition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fixture labels for the twelve rows. No model output is included."""
    from PIL import Image

    by_id = {row["scene_id"]: row for row in rows}
    missing = [scene_id for scene_id in DEVELOPMENT_SCENE_IDS if scene_id not in by_id]
    if missing:
        raise RuntimeError(f"Development subset ids are not in the manifest: {missing}")
    scenes = []
    for scene_id in DEVELOPMENT_SCENE_IDS:
        row = by_id[scene_id]
        image_path = ROOT / "fixtures" / "development_v2" / row["image_path"]
        if not image_path.is_file():
            raise RuntimeError(f"Missing development image {image_path}")
        with Image.open(image_path) as handle:
            width, height = handle.size
        scenes.append(
            {
                "scene_id": scene_id,
                "task_family": row["task_family"],
                "question": row["question"],
                "fixture_answer": row["answer"],
                "image_path": str(Path("fixtures") / "development_v2" / row["image_path"]),
                "image_wh": [width, height],
                "contexts": {name: row["contexts"][name]["text"] for name in CONTEXTS},
            }
        )
    return {
        "label": "DEVELOPMENT ONLY",
        "model_outputs": None,
        "listed_in_script_before_generation": True,
        "scene_ids": list(DEVELOPMENT_SCENE_IDS),
        "contexts": list(CONTEXTS),
        "prompt_rule": "context_text + newline + question",
        "max_new_tokens": MAX_NEW_TOKENS,
        "official_parser": "checkpoint-1 strict parser, vlm_iface.parsing.parse_generation",
        "scenes": scenes,
        "note": "fixture_answer is the scene label. It is not a model generation.",
    }


def _envelope(layers: int, seq: int, hidden: int, intermediate: int, heads: int, bytes_per: int = 2) -> dict[str, int]:
    attention = layers * heads * seq * seq * bytes_per
    mlp = layers * PLANNING_MLP_MATS * seq * intermediate * bytes_per
    residual = layers * PLANNING_RESIDUAL_COPIES * seq * hidden * bytes_per
    return {
        "layers": layers,
        "sequence": seq,
        "attention_scores_bytes": attention,
        "mlp_bytes": mlp,
        "residual_bytes": residual,
        "total_bytes": attention + mlp + residual,
    }


def _cuda_proposal(survey: dict[str, Any], preprocessor: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Planning figures for a requested GPU. Nothing here was launched or timed."""
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    vision = config.get("vision_config") if isinstance(config.get("vision_config"), dict) else {}
    parameter_count = survey.get("parameter_count")
    weights: dict[str, Any] = {"source": "safetensors header" if parameter_count else "unavailable"}
    if isinstance(parameter_count, int):
        weights["bfloat16_bytes"] = parameter_count * 2
        weights["float32_bytes"] = int(survey["float32_weight_bytes"])
        weights["on_disk_bytes"] = survey.get("on_disk_bytes")
        weights["proposed_dtype"] = "bfloat16"
    image_tokens = preprocessor.get("estimated_llm_image_tokens")
    resize = preprocessor.get("estimated_resize_hw")
    patch = preprocessor.get("patch_size") or vision.get("patch_size")
    activation: dict[str, Any] = {
        "label": "ESTIMATE",
        "not_a_measurement": True,
        "formula": (
            "Per stack: attention = layers * heads * seq * seq * 2; "
            "mlp = layers * 2 * seq * intermediate * 2; "
            "residual = layers * 8 * seq * hidden * 2. "
            "Bytes are BF16. The 2 and the 8 are planning multipliers, not profiler counts."
        ),
    }
    if all(isinstance(text.get(key), int) for key in ("num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads")) and isinstance(image_tokens, int):
        decoder_seq = image_tokens + PLANNING_TEXT_TOKENS
        activation["decoder"] = _envelope(
            int(text["num_hidden_layers"]),
            decoder_seq,
            int(text["hidden_size"]),
            int(text["intermediate_size"]),
            int(text["num_attention_heads"]),
        )
        activation["decoder"]["text_tokens_assumed"] = PLANNING_TEXT_TOKENS
        activation["decoder"]["image_tokens_arithmetic"] = image_tokens
    if isinstance(resize, list) and len(resize) == 2 and isinstance(patch, int) and patch > 0 and all(
        isinstance(vision.get(key), int) for key in ("depth", "hidden_size", "intermediate_size", "num_heads", "temporal_patch_size")
    ):
        premerge = (int(resize[0]) // patch) * (int(resize[1]) // patch)
        temporal = int(vision["temporal_patch_size"])
        activation["vision_spatial_patches"] = _envelope(
            int(vision["depth"]),
            premerge,
            int(vision["hidden_size"]),
            int(vision["intermediate_size"]),
            int(vision["num_heads"]),
        )
        activation["vision_if_temporal_patch_duplicates_the_still"] = _envelope(
            int(vision["depth"]),
            premerge * temporal,
            int(vision["hidden_size"]),
            int(vision["intermediate_size"]),
            int(vision["num_heads"]),
        )
        activation["temporal_duplicate_is_unmeasured"] = True
    upper = 0
    if "decoder" in activation:
        upper += activation["decoder"]["total_bytes"]
    vision_key = "vision_if_temporal_patch_duplicates_the_still"
    if vision_key in activation:
        upper += activation[vision_key]["total_bytes"]
    elif "vision_spatial_patches" in activation:
        upper += activation["vision_spatial_patches"]["total_bytes"]
    headroom = max(upper * PLANNING_SLACK, PLANNING_HEADROOM_FLOOR_BYTES) if upper else PLANNING_HEADROOM_FLOOR_BYTES
    activation["upper_envelope_bytes"] = upper or None
    activation["slack_factor"] = PLANNING_SLACK
    activation["recommended_headroom_bytes"] = headroom
    activation["recommended_headroom_rule"] = "max(2 * upper envelope, 8 GiB)"
    weight_bytes = weights.get("bfloat16_bytes")
    return {
        "label": "ESTIMATE",
        "authorization": "requested",
        "launched": False,
        "paid_gpu": False,
        "attacks_included": False,
        "proposed_device": "cuda",
        "proposed_gpu_memory_bytes": 24 * 1024**3,
        "proposed_dtype": "bfloat16",
        "workload": (
            "24 development scenes x 4 contexts, greedy, max_new_tokens=16, "
            "plus canonical-word gradients on dev2-attribute-00-a for neutral and helpful_nonrevealing. "
            "One model. No attack, no H*, no held-out split."
        ),
        "weight_memory": weights,
        "activation_headroom": activation,
        "planning_total_bytes": None if weight_bytes is None else weight_bytes + headroom,
        "runtime": {
            "label": "ESTIMATE",
            "wall_clock": "15 to 60 minutes, including cold start and a weight download",
            "basis": (
                "Ninety-six short generations and two backwards of a 2B BF16 model are a small CUDA job. "
                "The window is the one-hour cap used for the cost range. "
                "Checkpoint 1 SmolVLM2 MPS scene times are a different model and device and are not used as this estimate."
            ),
        },
        "cost_usd": {
            "label": "ESTIMATE",
            "not_a_quote": True,
            "not_purchased": True,
            "as_of": "2026-11-06",
            "low_usd": 0.34,
            "high_usd": 1.79,
            "billing_assumption": "Up to one hour on one GPU. The job is expected to finish inside that hour.",
            "sources": [
                {
                    "gpu": "RTX 4090",
                    "vram_gb": 24,
                    "usd_per_hour": 0.34,
                    "tier": "RunPod Community",
                    "source": "https://cirrun.com/providers/runpod.html",
                    "read": "2026-11-06",
                },
                {
                    "gpu": "L4",
                    "vram_gb": 24,
                    "usd_per_hour": 0.44,
                    "tier": "RunPod Community",
                    "source": "https://cirrun.com/providers/runpod.html",
                    "read": "2026-11-06",
                },
                {
                    "gpu": "L4",
                    "vram_gb": 24,
                    "usd_per_hour": 0.59,
                    "tier": "RunPod Secure",
                    "source": "https://cirrun.com/providers/runpod.html",
                    "read": "2026-11-06",
                },
                {
                    "gpu": "RTX 4090",
                    "vram_gb": 24,
                    "usd_per_hour": 0.89,
                    "tier": "RunPod Secure",
                    "source": "https://www.runpod.io/pricing",
                    "read": "2026-11-06",
                },
                {
                    "gpu": "A10",
                    "vram_gb": 24,
                    "usd_per_hour": 1.29,
                    "tier": "Lambda on-demand",
                    "source": "https://answerfootprint.com/gpu/gpu-cloud-price-index.csv",
                    "verified": "2026-11-02",
                },
                {
                    "gpu": "A100 80GB PCIe",
                    "vram_gb": 80,
                    "usd_per_hour": 1.79,
                    "tier": "RunPod Secure",
                    "source": "https://cirrun.com/providers/runpod.html",
                    "read": "2026-11-06",
                    "note": "Upper reference if a 24 GB card is not offered. Not required by the weight size.",
                },
            ],
        },
    }


def _load_config(revision: str) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(REPO_ID, "config.json", revision=revision)
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    return config


def _not_started(reason: str) -> dict[str, Any]:
    return {"status": "not_started", "from_pretrained_called": False, "reason": reason}


def _call_from_pretrained(owner, **kwargs):
    """The only ``from_pretrained`` call site. The abort path does not reach it."""
    global _FROM_PRETRAINED_CALLS
    if not kwargs.get("_permit_token"):
        raise RuntimeError("from_pretrained refused: missing permit token.")
    kwargs.pop("_permit_token")
    _FROM_PRETRAINED_CALLS += 1
    return owner.from_pretrained(**kwargs)


def _move_inputs(inputs: dict[str, Any], device: str) -> dict[str, Any]:
    import torch

    moved = {}
    for key, value in inputs.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def _encode(processor, image, prompt: str) -> dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    encoded = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if hasattr(encoded, "to"):
        return dict(encoded)
    return dict(encoded)


def _run_permitted_path(payload: dict[str, Any], budget, min_available: int, dtype_name: str) -> None:
    """Load on MPS and walk gates 3–5. Re-checks the budget before ``from_pretrained``."""
    import torch
    from vlm_iface.devices import memory_snapshot, synchronize
    from vlm_iface.images import image_01_to_pil

    memory = _memory(__import__("psutil"))
    survey = payload["qwen_candidate"]
    resident = int(survey["float32_weight_bytes"]) if dtype_name == "float32" else int(survey["parameter_count"]) * 2
    recheck = budget.budget_decision(
        resident_bytes=resident,
        available_bytes=memory["available"],
        ratio=budget.C1_DROP_PER_CURRENT,
        min_available_bytes=min_available,
        mps_available=bool(torch.backends.mps.is_available()),
    )
    payload["budget_recheck_immediately_before_load"] = recheck
    if not recheck["permitted"]:
        payload["stopped_at_gate"] = "gate_2_preload_budget_recheck"
        payload["from_pretrained_called"] = False
        payload["gates"]["weight_load_and_memory"] = {
            "status": "aborted",
            "from_pretrained_called": False,
            "reason": "The immediate recheck failed. from_pretrained was not called.",
            "recheck": recheck,
        }
        return
    if dtype_name == "bfloat16" and not payload["bf16_support"]["matmul_ok"]:
        payload["stopped_at_gate"] = "gate_2_bf16_probe"
        payload["from_pretrained_called"] = False
        return

    torch_dtype = torch.float32 if dtype_name == "float32" else torch.bfloat16
    device = torch.device("mps")
    started = time.perf_counter()
    payload["gates"]["weight_load_and_memory"] = {
        "status": "loading",
        "from_pretrained_called": False,
    }
    try:
        from transformers import AutoProcessor
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

        processor = _call_from_pretrained(
            AutoProcessor,
            _permit_token=True,
            pretrained_model_name_or_path=REPO_ID,
            revision=PINNED_REVISION,
        )
        model = _call_from_pretrained(
            Qwen3VLForConditionalGeneration,
            _permit_token=True,
            pretrained_model_name_or_path=REPO_ID,
            revision=PINNED_REVISION,
            dtype=torch_dtype,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        synchronize(device)
    except Exception as exc:
        payload["gates"]["weight_load_and_memory"] = {
            "status": "failed",
            "from_pretrained_called": _FROM_PRETRAINED_CALLS > 0,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        payload["stopped_at_gate"] = "gate_3_load"
        payload["gates"]["minimal_forward"] = _not_started("The load raised before a forward.")
        payload["gates"]["clean_answers"] = _not_started("The load raised.")
        payload["gates"]["image_gradients"] = _not_started("The load raised.")
        return

    parameter = next(model.parameters())
    payload["load_seconds"] = round(time.perf_counter() - started, 3)
    payload["mps_model_measurement"] = {
        "device": parameter.device.type,
        "dtype": str(parameter.dtype),
        "dtype_counts": _dtype_counts(model),
        "memory_after_load": memory_snapshot(device),
    }
    payload["gates"]["weight_load_and_memory"]["status"] = "loaded"
    if parameter.device.type != "mps":
        payload["stopped_at_gate"] = "gate_3_device"
        payload["gates"]["minimal_forward"] = _not_started("Parameters are not on MPS.")
        return
    _gate_forward_and_rest(payload, model, processor, device, torch_dtype)


def _dtype_counts(model) -> dict[str, int]:
    counts: dict[str, int] = {}
    for parameter in model.parameters():
        name = str(parameter.dtype)
        counts[name] = counts.get(name, 0) + parameter.numel()
    return counts


def _gate_forward_and_rest(payload, model, processor, device, torch_dtype) -> None:
    import torch
    from PIL import Image
    from vlm_iface.devices import memory_snapshot, synchronize

    synthetic = torch.zeros(3, 64, 64, dtype=torch.float32)
    synthetic[:, 16:48, 16:48] = torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1)
    pil = _tensor_to_pil(synthetic)
    started = time.perf_counter()
    try:
        inputs = _move_inputs(_encode(processor, pil, "What color is the square? Reply with one word."), device.type)
        with torch.inference_mode():
            outputs = model(**inputs)
        finite = bool(torch.isfinite(outputs.logits).all().item())
        synchronize(device)
        payload["gates"]["minimal_forward"] = {
            "status": "passed" if finite else "failed",
            "image": "synthetic_64x64_red_square",
            "accuracy_claim": False,
            "finite_logits": finite,
            "logit_shape": list(outputs.logits.shape),
            "seconds": round(time.perf_counter() - started, 3),
            "memory": memory_snapshot(device),
            "device": device.type,
            "dtype": str(torch_dtype),
        }
        del outputs, inputs
    except Exception as exc:
        payload["gates"]["minimal_forward"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "device": device.type,
        }
        payload["stopped_at_gate"] = "gate_3_minimal_forward"
        payload["gates"]["clean_answers"] = _not_started("The minimal forward failed.")
        payload["gates"]["image_gradients"] = _not_started("An earlier gate failed.")
        return
    if not payload["gates"]["minimal_forward"]["finite_logits"]:
        payload["stopped_at_gate"] = "gate_3_minimal_forward"
        payload["gates"]["clean_answers"] = _not_started("Logits were not finite.")
        payload["gates"]["image_gradients"] = _not_started("An earlier gate failed.")
        return
    _score_subset(payload, model, processor, device)
    if payload["gates"]["clean_answers"]["status"] != "completed":
        payload["stopped_at_gate"] = "gate_4_clean_answers"
        payload["gates"]["image_gradients"] = _not_started("The development subset did not finish.")
        return
    _gradients(payload, model, processor, device)


def _tensor_to_pil(image):
    from vlm_iface.images import image_01_to_pil

    return image_01_to_pil(image)


def _score_subset(payload, model, processor, device) -> None:
    import torch
    from PIL import Image
    from pilot.prompt import user_prompt
    from vlm_iface.devices import synchronize
    from vlm_iface.parsing import parse_generation

    by_id = {scene["scene_id"]: scene for scene in payload["subset_definition"]["scenes"]}
    rows = []
    started = time.perf_counter()
    try:
        for scene_id in DEVELOPMENT_SCENE_IDS:
            scene = by_id[scene_id]
            image_path = ROOT / scene["image_path"]
            with Image.open(image_path) as handle:
                pil = handle.convert("RGB")
            for context in CONTEXTS:
                prompt = user_prompt(scene["contexts"][context], scene["question"])
                inputs = _move_inputs(_encode(processor, pil, prompt), device.type)
                eos = _end_ids(model, processor)
                with torch.inference_mode():
                    generated = model.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)
                new_ids = generated[0, inputs["input_ids"].shape[-1] :]
                token_ids = [int(token) for token in new_ids.tolist()]
                text = processor.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                truncated = not any(token in eos for token in token_ids)
                parsed = parse_generation(text, truncated=truncated, task_family=scene["task_family"])
                correct = (not parsed.invalid) and parsed.parsed_answer == scene["fixture_answer"]
                rows.append(
                    {
                        "label": "DEVELOPMENT ONLY",
                        "scene_id": scene_id,
                        "context": context,
                        "task_family": scene["task_family"],
                        "raw_output": text,
                        "parsed_answer": parsed.parsed_answer,
                        "invalid": parsed.invalid,
                        "parse_reasons": list(parsed.reasons),
                        "correct": correct,
                        "valid_wrong": (not parsed.invalid) and not correct,
                        "truncated": truncated,
                        "token_ids": token_ids,
                        "device": device.type,
                    }
                )
                del generated, inputs
        synchronize(device)
    except Exception as exc:
        payload["gates"]["clean_answers"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "rows_completed": rows,
        }
        return
    payload["development_predictions"] = rows
    payload["gates"]["clean_answers"] = {
        "status": "completed",
        "label": "DEVELOPMENT ONLY",
        "n_rows": len(rows),
        "correct": sum(bool(row["correct"]) for row in rows),
        "valid_wrong": sum(bool(row["valid_wrong"]) for row in rows),
        "invalid": sum(bool(row["invalid"]) for row in rows),
        "seconds": round(time.perf_counter() - started, 3),
        "device": device.type,
        "max_new_tokens": MAX_NEW_TOKENS,
        "official_parser": "vlm_iface.parsing.parse_generation",
    }


def _end_ids(model, processor) -> set[int]:
    found: set[int] = set()
    candidates = [
        getattr(getattr(model, "config", None), "eos_token_id", None),
        getattr(getattr(model, "generation_config", None), "eos_token_id", None),
        getattr(processor.tokenizer, "eos_token_id", None),
    ]
    for value in candidates:
        if value is None:
            continue
        if isinstance(value, int):
            found.add(value)
        else:
            found.update(int(item) for item in value)
    return found


def _gradients(payload, model, processor, device) -> None:
    """Canonical-word loss on processor pixels. Not an attack."""
    import torch
    from PIL import Image
    from pilot.prompt import user_prompt
    from vlm_iface.devices import synchronize
    from vlm_iface.logprob import sequence_logprob
    from vlm_iface.tokens import answer_token_ids

    scene = next(item for item in payload["subset_definition"]["scenes"] if item["scene_id"] == GRADIENT_SCENE_ID)
    image_path = ROOT / scene["image_path"]
    results = []
    try:
        with Image.open(image_path) as handle:
            pil = handle.convert("RGB")
        for context in GRADIENT_CONTEXTS:
            prompt = user_prompt(scene["contexts"][context], scene["question"])
            inputs = _move_inputs(_encode(processor, pil, prompt), device.type)
            answer_ids = answer_token_ids(processor.tokenizer, scene["fixture_answer"], leading_space=True)
            answer_ids = answer_ids.view(1, -1).to(device)
            prompt_ids = inputs["input_ids"]
            full_ids = torch.cat([prompt_ids, answer_ids], dim=-1)
            mask = inputs.get("attention_mask")
            if mask is not None:
                extra = torch.ones(1, answer_ids.shape[-1], dtype=mask.dtype, device=device)
                inputs["attention_mask"] = torch.cat([mask, extra], dim=-1)
            mm = inputs.get("mm_token_type_ids")
            if mm is not None:
                extra = torch.zeros(1, answer_ids.shape[-1], dtype=mm.dtype, device=device)
                inputs["mm_token_type_ids"] = torch.cat([mm, extra], dim=-1)
            pixel_values = inputs["pixel_values"].detach().requires_grad_(True)
            inputs["pixel_values"] = pixel_values
            inputs["input_ids"] = full_ids
            synchronize(device)
            started = time.perf_counter()
            outputs = model(**inputs, use_cache=False)
            logprob = sequence_logprob(outputs.logits, full_ids, int(prompt_ids.shape[-1]))
            loss = -logprob.sum()
            finite_loss = bool(torch.isfinite(loss).item())
            if not finite_loss:
                results.append({"context": context, "finite_loss": False, "gradient_computed": False})
                continue
            loss.backward()
            gradient = pixel_values.grad
            synchronize(device)
            results.append(
                {
                    "label": "DEVELOPMENT ONLY",
                    "scene_id": GRADIENT_SCENE_ID,
                    "context": context,
                    "answer": scene["fixture_answer"],
                    "loss": float(loss.detach().float().cpu()),
                    "finite_loss": True,
                    "gradient_space": "processor_pixel_values",
                    "canonical_01_rgb_gradient_computed": False,
                    "finite": bool(torch.isfinite(gradient).all().item()) if gradient is not None else False,
                    "abs_mean": None if gradient is None else float(gradient.detach().float().abs().mean().cpu()),
                    "abs_max": None if gradient is None else float(gradient.detach().float().abs().max().cpu()),
                    "nonzero": None if gradient is None else bool((gradient != 0).any().item()),
                    "shape": None if gradient is None else list(gradient.shape),
                    "seconds": round(time.perf_counter() - started, 3),
                    "attack": False,
                }
            )
            pixel_values.grad = None
            del outputs, loss, gradient, inputs
        payload["gates"]["image_gradients"] = {
            "status": "completed",
            "rows": results,
            "note": (
                "The leaf is processor pixel_values. The official processor goes through PIL, "
                "so this is not the [0, 1] RGB gradient. It is a finiteness check, not an attack."
            ),
        }
        payload["stopped_at_gate"] = None
        payload["passed_all_gates"] = all(row.get("finite") for row in results)
    except Exception as exc:
        payload["gates"]["image_gradients"] = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "rows": results,
        }
        payload["stopped_at_gate"] = "gate_5_image_gradients"


def _write(payload: dict[str, Any]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    environment = _prepare_environment()
    budget = _load_budget_module()
    import psutil
    import torch
    from vlm_iface.candidate import survey_weights
    from vlm_iface.pins import MIN_AVAILABLE_BYTES, QWEN_CANDIDATE_ID

    if QWEN_CANDIDATE_ID != REPO_ID:
        raise RuntimeError("pins.QWEN_CANDIDATE_ID does not match the repo id in this script.")
    if MIN_AVAILABLE_BYTES != 6 * 1024**3:
        raise RuntimeError("The 6 GiB OS floor changed. This script will not adopt a lower floor.")
    budget.assert_budget_fixtures(MIN_AVAILABLE_BYTES)

    memory_at_start = _memory(psutil)
    print("Gate 1: OS-available memory rechecked in this process.", flush=True)
    bf16 = _bf16_probe(torch)
    compatibility = _compatibility()
    files_before = _snapshot_names(PINNED_REVISION)
    blobs_before = _large_blobs()
    print("Gate 2: surveying the safetensors header. Tensor bytes are not downloaded.", flush=True)
    survey = _survey(survey_weights)
    preprocessor = budget._preprocessor_note(survey) if not survey.get("survey_error") else {"downloaded": False}
    config: dict[str, Any] = {}
    if survey.get("revision") and not survey.get("survey_error"):
        try:
            config = _load_config(str(survey["revision"]))
        except Exception as exc:
            config = {"error": f"{type(exc).__name__}: {exc}"}
    memory_at_decision = _memory(psutil)
    decisions = _dtype_decisions(
        budget,
        survey,
        available=memory_at_decision["available"],
        mps_available=bool(torch.backends.mps.is_available()),
        min_available=MIN_AVAILABLE_BYTES,
    )
    selection = _select_dtype(decisions, bool(bf16.get("matmul_ok")))
    subset = _subset_definition(_load_manifest())
    proposal = _cuda_proposal(survey, preprocessor, config)

    payload: dict[str, Any] = {
        "measurement_kind": "checkpoint16_qwen_feasibility",
        "label": "DEVELOPMENT ONLY",
        "checkpoint": "checkpoint-1.6",
        "attacks_run": False,
        "training_run": False,
        "cloud_gpu": False,
        "paid_gpu_launched": False,
        "held_out_opened": False,
        "h_star_fitted": False,
        "from_pretrained_called": False,
        "model_from_pretrained_called": False,
        "processor_from_pretrained_called": False,
        "weights_downloaded": False,
        "checkpoint15_feasibility_record_consulted_for_available_bytes": False,
        "reproduction_command": REPRODUCTION_COMMAND,
        "environment": environment,
        "budget_module": {
            "path": str(BUDGET_PATH.relative_to(ROOT)),
            "sha256": _sha256(BUDGET_PATH),
            "spare_required_bytes": budget.SPARE_REQUIRED_BYTES,
            "projected_remaining_floor_bytes": budget.PROJECTED_REMAINING_FLOOR_BYTES,
            "min_available_bytes": MIN_AVAILABLE_BYTES,
            "drop_per_current_allocation": budget.C1_DROP_PER_CURRENT,
            "fixtures_passed": True,
            "note": (
                "budget_decision and the fixture constants were imported from the Checkpoint 1.5 script. "
                "The live decision uses psutil available bytes from this process, not a stored available-byte field."
            ),
        },
        "machine": _machine(memory_at_decision["total"]),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
        },
        "os_memory_bytes_at_start": memory_at_start,
        "os_memory_bytes_at_decision": memory_at_decision,
        "decision_available_bytes": memory_at_decision["available"],
        "bf16_support": bf16,
        "model_compatibility": compatibility,
        "qwen_candidate": survey,
        "qwen_config_dtype": (config.get("text_config") or {}).get("dtype") if isinstance(config.get("text_config"), dict) else config.get("error"),
        "qwen_config_architectures": config.get("architectures"),
        "qwen_preprocessor": preprocessor,
        "qwen_dtype_decisions": decisions,
        "dtype_selection": selection,
        "mps_model_measurement": None,
        "mps_current_allocated_bytes_with_no_model": int(torch.mps.current_allocated_memory()) if torch.backends.mps.is_available() else None,
        "snapshot_files_before": files_before,
        "large_blobs_before": blobs_before,
        "subset_definition": subset,
        "cuda_proposal": None,
        "gates": {},
    }
    if survey.get("revision") and survey["revision"] != PINNED_REVISION:
        selection = {
            "dtype": None,
            "permitted": False,
            "reason": f"Hub sha {survey['revision']} does not match pinned revision {PINNED_REVISION}.",
        }
        payload["dtype_selection"] = selection

    if not selection["permitted"]:
        reason = "No dtype cleared the pre-load budget."
        payload["gates"] = {
            "os_memory": {"status": "measured", "available_bytes": memory_at_decision["available"]},
            "compatibility_and_budget": {
                "status": "aborted",
                "from_pretrained_called": False,
                "weights_loaded": False,
                "reason": selection["reason"],
            },
            "minimal_forward": _not_started(reason),
            "clean_answers": _not_started(reason),
            "image_gradients": _not_started(reason),
        }
        payload["action"] = "load_aborted"
        payload["adopted"] = False
        payload["recommendation"] = "needs_authorized_cuda"
        payload["stopped_at_gate"] = "gate_2_preload_budget"
        payload["passed_gate_2"] = False
        payload["cuda_proposal"] = proposal
        payload["comparison_with_smolvlm"] = (
            "Not run. Qwen produced no scores. SmolVLM2 was not loaded. "
            "The comparison model remains the committed Checkpoint 1.5 tables."
        )
    else:
        payload["gates"] = {
            "os_memory": {"status": "measured", "available_bytes": memory_at_decision["available"]},
            "compatibility_and_budget": {"status": "passed", "dtype": selection["dtype"]},
            "minimal_forward": {"status": "not_started"},
            "clean_answers": {"status": "not_started"},
            "image_gradients": {"status": "not_started"},
        }
        payload["action"] = "load_permitted"
        payload["recommendation"] = "pending_mps_gates"
        payload["stopped_at_gate"] = None
        payload["passed_gate_2"] = True
        _run_permitted_path(payload, budget, MIN_AVAILABLE_BYTES, selection["dtype"])
        if payload.get("stopped_at_gate"):
            payload["adopted"] = False
            payload["recommendation"] = "not_adopted"
            payload["action"] = "stopped"
        elif payload.get("passed_all_gates"):
            payload["adopted"] = False
            payload["recommendation"] = "not_adopted_pending_lead"
            payload["action"] = "gates_completed_not_adopted"
            payload["adoption_note"] = (
                "Gates finished on a development subset. Adoption is a lead decision. "
                "This script does not replace the pinned SmolVLM2 checkpoint."
            )

    files_after = _snapshot_names(str(survey.get("revision") or PINNED_REVISION))
    blobs_after = _large_blobs()
    weight_present = _weight_file_present(str(survey.get("revision") or PINNED_REVISION))
    payload["snapshot_files_after"] = files_after
    payload["large_blobs_after"] = blobs_after
    payload["qwen_local_weight_file_present"] = weight_present
    payload["weights_downloaded"] = blobs_after != blobs_before or (
        weight_present and "model.safetensors" not in files_before
    )
    payload["from_pretrained_call_count"] = _FROM_PRETRAINED_CALLS
    payload["processor_from_pretrained_called"] = _FROM_PRETRAINED_CALLS > 0
    payload["model_from_pretrained_called"] = _FROM_PRETRAINED_CALLS > 1
    payload["from_pretrained_called"] = _FROM_PRETRAINED_CALLS > 0
    payload["wall_seconds"] = round(time.perf_counter() - started, 3)
    _write(payload)
    if payload.get("stopped_at_gate") == "gate_2_preload_budget" and _FROM_PRETRAINED_CALLS != 0:
        raise RuntimeError("The budget abort path called from_pretrained.")
    if payload.get("stopped_at_gate") == "gate_2_preload_budget" and payload["weights_downloaded"]:
        raise RuntimeError("Weight bytes appeared even though the budget aborted the load.")
    print(json.dumps({
        "action": payload["action"],
        "stopped_at_gate": payload.get("stopped_at_gate"),
        "from_pretrained_called": payload["from_pretrained_called"],
        "from_pretrained_call_count": _FROM_PRETRAINED_CALLS,
        "decision_available_bytes": payload["decision_available_bytes"],
        "recommendation": payload["recommendation"],
        "weights_downloaded": payload["weights_downloaded"],
        "output": str(OUTPUT_PATH.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
