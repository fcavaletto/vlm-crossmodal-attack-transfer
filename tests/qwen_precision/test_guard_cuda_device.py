"""CPU-runnable check of the CUDA guard's device handling, with torch.cuda calls replaced by fakes
that apply torch's own rule: memory calls need a device with an index."""

import torch

from qwen_precision import guard as guard_mod


def _needs_index(device):
    if isinstance(device, torch.device) and device.type == "cuda" and device.index is None:
        raise ValueError(f"Expected a torch.device with a specified index or an integer, but got:{device}")


class _Props:
    total_memory = 96 * 1024**3


def test_guard_uses_an_indexed_cuda_device(monkeypatch, tmp_path):
    calls = []

    def fake(name, ret=None):
        def f(*args, **kwargs):
            for a in list(args) + list(kwargs.values()):
                _needs_index(a)
            calls.append(name)
            return ret
        return f

    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", fake("fraction"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", fake("reset"))
    monkeypatch.setattr(torch.cuda, "get_device_properties", fake("props", _Props()))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", fake("alloc", 0))
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", fake("reserved", 0))
    g = guard_mod.CudaGuard(torch.device("cuda"), disk_path=str(tmp_path), min_free_disk_bytes=0, min_host_available_bytes=0)
    assert g.device == torch.device("cuda", 0)
    info = g.arm()
    assert info["total_bytes"] == 96 * 1024**3
    g.check()
    g.record("t")
    assert {"fraction", "reset", "props", "alloc", "reserved"} <= set(calls)
