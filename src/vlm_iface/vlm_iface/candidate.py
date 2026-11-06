"""Hub metadata for the unapproved Qwen candidate. This module never loads weights."""

from __future__ import annotations

import json
import struct
import urllib.request
from typing import Any

from .pins import FLOAT32_WEIGHT_ABORT_BYTES

_HEADER_BYTE_CAP = 10_000_000
_FLOAT_DTYPES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2}
_OTHER_DTYPES = {"I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1, "U32": 4, "U64": 8}


def decide_load(*, parameter_count: int, float32_weight_bytes: int) -> dict[str, Any]:
    """Return whether a from_pretrained call is inside the weight-memory guard.

    The guard is about weights only. A true result is not permission to adopt a
    model, and it does not account for activation memory during backward.
    """
    if parameter_count < 0 or float32_weight_bytes < 0:
        raise ValueError("parameter counts must be non-negative")
    allowed = float32_weight_bytes <= FLOAT32_WEIGHT_ABORT_BYTES
    return {
        "load_permitted_by_weight_guard": allowed,
        "parameter_count": parameter_count,
        "float32_weight_bytes": float32_weight_bytes,
        "abort_bytes": FLOAT32_WEIGHT_ABORT_BYTES,
        "reason": (
            "float32 weight bytes are within the guard"
            if allowed
            else "float32 weight bytes exceed the guard; from_pretrained must not be called"
        ),
    }


def summarize_safetensors_header(header: dict[str, Any]) -> dict[str, Any]:
    """Count parameters and the float32 expansion of a parsed safetensors header."""
    dtype_counts: dict[str, int] = {}
    parameter_count = 0
    float32_weight_bytes = 0
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        shape = meta["shape"]
        dtype = str(meta["dtype"])
        count = 1
        for dim in shape:
            count *= int(dim)
        parameter_count += count
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + count
        if dtype in _FLOAT_DTYPES:
            float32_weight_bytes += count * 4
        elif dtype in _OTHER_DTYPES:
            float32_weight_bytes += count * _OTHER_DTYPES[dtype]
        else:
            raise ValueError(f"Unsupported safetensors dtype {dtype}")
    return {
        "parameter_count": parameter_count,
        "dtype_counts": dtype_counts,
        "float32_weight_bytes": float32_weight_bytes,
    }


def parse_safetensors_prefix(prefix: bytes) -> dict[str, Any]:
    """Parse a safetensors header from the leading bytes of a file. Does not read tensors."""
    if len(prefix) < 8:
        raise ValueError("safetensors prefix is shorter than the 8-byte header length")
    header_len = struct.unpack_from("<Q", prefix, 0)[0]
    if header_len <= 0 or header_len > _HEADER_BYTE_CAP:
        raise ValueError(f"Refusing safetensors header length {header_len}")
    if len(prefix) < 8 + header_len:
        raise ValueError("safetensors prefix does not contain the full header")
    header = json.loads(prefix[8 : 8 + header_len].decode("utf-8"))
    summary = summarize_safetensors_header(header)
    summary["header_bytes"] = int(header_len)
    return summary


def survey_weights(repo_id: str, *, revision: str | None = None) -> dict[str, Any]:
    """Read Hub file size and the safetensors header. Does not download tensor bytes.

    A range request that is not answered with HTTP 206 is rejected, so a missing
    Range implementation cannot stream the weight file into this process.
    """
    from huggingface_hub import get_hf_file_metadata, hf_hub_url, model_info

    info = model_info(repo_id, revision=revision)
    resolved = str(info.sha)
    filename = "model.safetensors"
    url = hf_hub_url(repo_id, filename, revision=resolved)
    metadata = get_hf_file_metadata(url)
    if metadata.size is None:
        raise RuntimeError(f"Hub returned no size for {repo_id}@{resolved}:{filename}")
    try:
        prefix = _read_safetensors_header_bytes(url)
        summary = parse_safetensors_prefix(prefix)
        summary["approximate"] = False
        summary["header_source"] = "range"
    except Exception as exc:
        summary = _approximate_from_config(repo_id, resolved, int(metadata.size))
        summary["header_source"] = "config_and_file_size"
        summary["header_error"] = f"{type(exc).__name__}: {exc}"
    decision = decide_load(
        parameter_count=summary["parameter_count"],
        float32_weight_bytes=summary["float32_weight_bytes"],
    )
    return {
        "repo_id": repo_id,
        "revision": resolved,
        "filename": filename,
        "on_disk_bytes": int(metadata.size),
        "gated": bool(getattr(info, "gated", False)),
        "weights_loaded": False,
        **summary,
        **decision,
    }


def _approximate_from_config(repo_id: str, revision: str, on_disk_bytes: int) -> dict[str, Any]:
    """Estimate a float32 expansion from config dtype and file size, without tensor bytes."""
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(repo_id, "config.json", revision=revision)
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    dtype = text.get("dtype") or config.get("dtype") or config.get("torch_dtype")
    if dtype in ("bfloat16", "bf16", "float16", "fp16"):
        parameter_count = on_disk_bytes // 2
        float32_weight_bytes = parameter_count * 4
    elif dtype in ("float32", "fp32", None):
        parameter_count = on_disk_bytes // 4
        float32_weight_bytes = parameter_count * 4
    else:
        raise RuntimeError(f"Cannot estimate weight memory for dtype {dtype}")
    return {
        "parameter_count": parameter_count,
        "dtype_counts": {str(dtype): parameter_count},
        "float32_weight_bytes": float32_weight_bytes,
        "approximate": True,
    }


def _read_safetensors_header_bytes(url: str) -> bytes:
    length_prefix = _read_range(url, 0, 7)
    header_len = struct.unpack("<Q", length_prefix)[0]
    if header_len <= 0 or header_len > _HEADER_BYTE_CAP:
        raise ValueError(f"Refusing safetensors header length {header_len}")
    header = _read_range(url, 8, 8 + header_len - 1)
    return length_prefix + header


def _read_range(url: str, start: int, end: int) -> bytes:
    expected = end - start + 1
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "User-Agent": "vlm-iface-checkpoint0",
        },
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        status = getattr(response, "status", None)
        if status != 206:
            raise RuntimeError(
                f"Range request for bytes {start}-{end} returned HTTP {status}, not 206. "
                "Refusing to read the body."
            )
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) != expected:
            raise RuntimeError(
                f"Range response length {content_length} does not equal {expected}. Refusing to read the body."
            )
        payload = response.read(expected)
    if len(payload) != expected:
        raise RuntimeError(f"Range read returned {len(payload)} bytes, expected {expected}")
    return payload
