"""Memory, disk and deadline guard for CUDA (the frozen path_probe monitor is macOS-only).

Every limit is checked at safe points between model calls. A breach raises ``GuardStop``;
the runner records it and exits. Nothing is retried and no limit is changed at runtime.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from typing import Any

import torch

GIB = 1024**3


class GuardStop(RuntimeError):
    pass


@dataclass
class CudaGuard:
    device: torch.device
    max_reserved_bytes: int = 36 * GIB
    memory_fraction: float = 0.75
    min_host_available_bytes: int = 4 * GIB
    min_free_disk_bytes: int = 5 * 10**9
    disk_path: str = "/workspace"
    deadline_epoch: float | None = None
    peaks: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # torch.cuda memory calls reject a device without an index (found on the first CUDA run,
        # 2026-11-10: "Expected a torch.device with a specified index"). The pod has one GPU.
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", 0)

    def arm(self) -> dict[str, Any]:
        torch.cuda.set_per_process_memory_fraction(self.memory_fraction, self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        total = torch.cuda.get_device_properties(self.device).total_memory
        return {"total_bytes": int(total), "memory_fraction": self.memory_fraction,
                "max_reserved_bytes": self.max_reserved_bytes, "deadline_epoch": self.deadline_epoch}

    def sample(self) -> dict[str, Any]:
        import psutil

        return {"t": time.time(), "peak_allocated": int(torch.cuda.max_memory_allocated(self.device)),
                "peak_reserved": int(torch.cuda.max_memory_reserved(self.device)),
                "host_available": int(psutil.virtual_memory().available),
                "disk_free": int(shutil.disk_usage(self.disk_path).free)}

    def check(self) -> None:
        s = self.sample()
        if s["peak_reserved"] > self.max_reserved_bytes:
            raise GuardStop(f"CUDA peak reserved {s['peak_reserved'] / GIB:.2f} GiB > {self.max_reserved_bytes / GIB:.0f} GiB")
        if s["host_available"] < self.min_host_available_bytes:
            raise GuardStop("host available memory below 4 GiB")
        if s["disk_free"] < self.min_free_disk_bytes:
            raise GuardStop("free disk below 5 GB")
        if self.deadline_epoch is not None and time.time() > self.deadline_epoch:
            raise GuardStop("wall-clock deadline reached")

    def record(self, tag: str) -> dict[str, Any]:
        s = {"tag": tag, **self.sample()}
        self.peaks.append(s)
        return s
