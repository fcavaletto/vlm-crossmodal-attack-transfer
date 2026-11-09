"""Exploratory memory policy EMP-1 and a memory-pressure monitor. EXPLORATORY ONLY.

The frozen Checkpoint 1.5 guard is not edited. Its decision is still computed and
recorded for every load. EMP-1 is an opt-in alternative for this exploratory
run only, calibrated on the four completed Qwen bf16 runs (1.8, 1.9, 2.0, 2.1)
instead of the Checkpoint 1 float32 SmolVLM2 load. See MEMORY_ASSESSMENT.md.

The monitor runs in a background thread and reads only OS counters (the macOS
memory-pressure level, available memory, and swap). It never touches MPS. The
main thread calls ``check()`` at safe points, between model calls, and an abort
raises there. If the system stays critical after an abort was requested, the
thread ends the process with ``os._exit(3)``. Records are flushed once per scene,
so a hard exit loses at most the scene in progress.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

GIB = 1024**3

# Largest observed Qwen bf16 working set: MPS driver 5.61 GiB (1.8) + RSS 0.68 GiB (1.9).
OBSERVED_QWEN_BF16_WORKING_SET_BYTES = int(5.61 * GIB + 0.68 * GIB)
PRELOAD_MARGIN_BYTES = int(1.5 * GIB)
EMP1_PRELOAD_AVAILABLE_BYTES = OBSERVED_QWEN_BF16_WORKING_SET_BYTES + PRELOAD_MARGIN_BYTES
EMP1_MIN_FREE_DISK_BYTES = 10 * GIB
EMP1_V32_AVAILABLE_BYTES = int(2.5 * GIB)

PRESSURE_NORMAL, PRESSURE_WARN, PRESSURE_CRITICAL = 1, 2, 4


@dataclass
class Limits:
    sample_seconds: float = 0.5
    warn_sustained_seconds: float = 120.0
    low_available_bytes: int = int(0.75 * GIB)
    low_available_sustained_seconds: float = 30.0
    max_swap_growth_bytes: int = 6 * GIB
    hard_exit_after_abort_seconds: float = 60.0


def _sysctl_int(name: str) -> int | None:
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        value = ctypes.c_int(0)
        size = ctypes.c_size_t(ctypes.sizeof(value))
        rc = libc.sysctlbyname(name.encode(), ctypes.byref(value), ctypes.byref(size), None, ctypes.c_size_t(0))
        return int(value.value) if rc == 0 else None
    except OSError:
        return None


def process_footprint(pid: int | None = None) -> dict[str, int] | None:
    """macOS physical footprint of a process, which includes its MPS (Metal) buffers.

    RUSAGE_INFO_V4: ri_phys_footprint at byte 72, ri_lifetime_max_phys_footprint at byte 240.
    The lifetime maximum is the true peak, including inside a backward pass.
    """
    import struct

    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c"))
        buf = (ctypes.c_uint8 * 512)()
        if libc.proc_pid_rusage(int(pid or os.getpid()), 4, buf) != 0:
            return None
        raw = bytes(buf)
        return {"current": struct.unpack_from("<Q", raw, 72)[0], "lifetime_max": struct.unpack_from("<Q", raw, 240)[0]}
    except OSError:
        return None


def os_sample() -> dict[str, Any]:
    import psutil

    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    fp = process_footprint() or {}
    return {"t": time.time(), "pressure": _sysctl_int("kern.memorystatus_vm_pressure_level"),
            "available": int(vm.available), "swap_used": int(sw.used),
            "footprint": fp.get("current"), "footprint_max": fp.get("lifetime_max")}


def emp1_preload(sample: dict[str, Any], free_disk_bytes: int) -> dict[str, Any]:
    lines = {
        "available_bytes": EMP1_PRELOAD_AVAILABLE_BYTES,
        "pressure_level_max": PRESSURE_NORMAL,
        "free_disk_bytes": EMP1_MIN_FREE_DISK_BYTES,
    }
    failed = []
    if sample["available"] < lines["available_bytes"]:
        failed.append("available_bytes")
    if sample["pressure"] is None or sample["pressure"] > PRESSURE_NORMAL:
        failed.append("pressure_level")
    if free_disk_bytes < lines["free_disk_bytes"]:
        failed.append("free_disk_bytes")
    return {"policy": "EMP-1", "lines": lines, "sample": sample, "free_disk_bytes": free_disk_bytes,
            "failed": failed, "permitted": not failed}


@dataclass
class MemoryMonitor:
    limits: Limits = field(default_factory=Limits)
    sampler: Callable[[], dict[str, Any]] = os_sample
    exit_fn: Callable[[int], None] = os._exit

    def __post_init__(self) -> None:
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.abort_reason: str | None = None
        self._abort_at: float | None = None
        self._warn_since: float | None = None
        self._low_since: float | None = None
        self.baseline: dict[str, Any] | None = None
        self.summary: dict[str, Any] = {"samples": 0, "by_level": {}, "min_available": None, "max_swap_used": None,
                                        "max_swap_growth": 0, "timeline": []}
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self.baseline = self.sampler()
        self._thread = threading.Thread(target=self._run, name="memory-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def observe(self, s: dict[str, Any]) -> None:
        """One sample. Separate from the thread so rules can be tested."""
        lim = self.limits
        with self._lock:
            sm = self.summary
            sm["samples"] += 1
            lvl = str(s["pressure"])
            sm["by_level"][lvl] = sm["by_level"].get(lvl, 0) + 1
            sm["min_available"] = s["available"] if sm["min_available"] is None else min(sm["min_available"], s["available"])
            sm["max_swap_used"] = s["swap_used"] if sm["max_swap_used"] is None else max(sm["max_swap_used"], s["swap_used"])
            growth = s["swap_used"] - (self.baseline or s)["swap_used"]
            sm["max_swap_growth"] = max(sm["max_swap_growth"], growth)
            if s.get("footprint_max") is not None:
                sm["process_peak_footprint"] = max(sm.get("process_peak_footprint") or 0, s["footprint_max"])
            if not sm["timeline"] or s["t"] - sm["timeline"][-1]["t"] >= 10:
                sm["timeline"].append(dict(s))
            now = s["t"]
            p = s["pressure"]
            if p is not None and p >= PRESSURE_CRITICAL:
                self._request(f"memory pressure critical (level {p})", now)
            if p is not None and p >= PRESSURE_WARN:
                self._warn_since = now if self._warn_since is None else self._warn_since
                if now - self._warn_since >= lim.warn_sustained_seconds:
                    self._request(f"memory pressure warn for {lim.warn_sustained_seconds:.0f} s", now)
            else:
                self._warn_since = None
            if s["available"] < lim.low_available_bytes:
                self._low_since = now if self._low_since is None else self._low_since
                if now - self._low_since >= lim.low_available_sustained_seconds:
                    self._request("available memory stayed below 0.75 GiB", now)
            else:
                self._low_since = None
            if growth > lim.max_swap_growth_bytes:
                self._request(f"swap grew by more than {lim.max_swap_growth_bytes / GIB:.0f} GiB", now)
            if (self._abort_at is not None and p is not None and p >= PRESSURE_CRITICAL
                    and now - self._abort_at >= lim.hard_exit_after_abort_seconds):
                sm["hard_exit"] = True
                self.exit_fn(3)

    def _request(self, reason: str, now: float) -> None:
        if self.abort_reason is None:
            self.abort_reason = reason
            self._abort_at = now
            self.summary["abort_reason"] = reason

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.observe(self.sampler())
            except Exception as exc:  # a failed read must not kill the run silently
                self._request(f"memory monitor failed: {type(exc).__name__}: {exc}", time.time())
            self._stop.wait(self.limits.sample_seconds)

    def check(self) -> None:
        if self.abort_reason is not None:
            from .run import Stop

            raise Stop("memory abort: " + self.abort_reason)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {k: (list(v) if isinstance(v, list) else (dict(v) if isinstance(v, dict) else v))
                    for k, v in self.summary.items()}
