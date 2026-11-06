"""Device selection and memory snapshots.

CUDA is a real code path. A snapshot from this module is an MPS result only when
the resolved device type is ``mps``. Callers must not relabel a CPU snapshot.
"""

from __future__ import annotations

import platform
import resource

import psutil
import torch


def resolve_device(name: str | None) -> torch.device:
    """Resolve ``auto``, ``cpu``, ``cuda``, or ``mps``.

    ``auto`` prefers CUDA, then MPS, then CPU. Requesting a device that is not
    available raises. There is no silent fallback.
    """
    requested = "auto" if name is None else name.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is false.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unknown device {name!r}. Use auto, cpu, cuda, or mps.")


def synchronize(device: torch.device) -> None:
    """Wait until queued work on ``device`` has finished. CPU is a no-op."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def available_bytes() -> int:
    """Bytes the OS currently reports as available. This includes reclaimable cache."""
    return int(psutil.virtual_memory().available)


def memory_snapshot(device: torch.device) -> dict[str, float | int | str | None]:
    """Point-in-time memory figures, each labeled by what it measures.

    ``peak_rss_mb`` is the process peak resident set from ``ru_maxrss``. On macOS
    that value is in bytes. It mixes the process heap with unified-memory
    allocations and is not an MPS-only number.

    MPS and CUDA keys are present only for that device. They are omitted, not
    zero, on the other devices.
    """
    raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        peak_rss_mb = raw_rss / (1024 * 1024)
    else:
        peak_rss_mb = raw_rss / 1024
    current_rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    snapshot: dict[str, float | int | str | None] = {
        "device": device.type,
        "peak_rss_mb": round(peak_rss_mb, 1),
        "current_rss_mb": round(current_rss_mb, 1),
        "os_available_mb": round(available_bytes() / (1024 * 1024), 1),
    }
    if device.type == "mps":
        snapshot["mps_driver_allocated_mb"] = round(torch.mps.driver_allocated_memory() / (1024**2), 1)
        snapshot["mps_current_allocated_mb"] = round(torch.mps.current_allocated_memory() / (1024**2), 1)
    elif device.type == "cuda":
        snapshot["cuda_allocated_mb"] = round(torch.cuda.memory_allocated(device) / (1024**2), 1)
        snapshot["cuda_reserved_mb"] = round(torch.cuda.memory_reserved(device) / (1024**2), 1)
    return snapshot
