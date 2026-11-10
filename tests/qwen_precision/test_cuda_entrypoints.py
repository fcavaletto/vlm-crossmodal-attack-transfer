"""Regression tests for the CUDA entry points of the Checkpoint 5 runner, on a machine without CUDA.

The real ``run.main()`` is called with ``--device cuda``. It goes through the real ``make_guard`` and
``CudaGuard``. ``torch.cuda`` memory calls are replaced by a shim that applies torch's own rule
(memory calls need an indexed device): the rule that broke the first GPU attempt on 2026-11-10.
Model loading and the scene computation are replaced by small deterministic stand-ins. What is
tested is the runner's control flow: preflight unit, integrity manifest, verified resume without
duplication, the G9 and throughput gates, and the end-of-run drift check.
"""

import json
import sys
import time

import pytest
import torch

from qwen_precision import guard as guard_mod
from qwen_precision import run


def _needs_index(*args, **kwargs):
    for a in list(args) + list(kwargs.values()):
        if isinstance(a, torch.device) and a.type == "cuda" and a.index is None:
            raise ValueError(f"Expected a torch.device with a specified index or an integer, but got:{a}")


class _Props:
    total_memory = 96 * 1024**3


@pytest.fixture
def cuda_shim(monkeypatch):
    def fake(ret=None):
        def f(*a, **k):
            _needs_index(*a, **k)
            return ret
        return f

    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", fake())
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", fake())
    monkeypatch.setattr(torch.cuda, "get_device_properties", fake(_Props()))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", fake(0))
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", fake(0))

    class _Disk:
        free = 10**12

    monkeypatch.setattr(guard_mod.shutil, "disk_usage", lambda p: _Disk())


SCENES = [{"scene_id": f"dev4-s{i:02d}-a", "answer": "red"} for i in range(5)]


class FakeSession:
    def __init__(self, device, dtypes):
        assert torch.device(device).type == "cuda"
        self.numerics = {"device": "cuda"}
        self.load_seconds = {d: 0.0 for d in dtypes}

    def backend(self, dtype, row, x, guard, *, float_check=True):
        class B:
            def score_official(self, x, ctx):
                return {"nll": {"red": 1.0}, "margin": 0.5}
        return B()


def fake_unit(session, dtype, row, x, check, seed_prefix):
    check()
    sid = row["scene_id"]
    rows = [{"kind": "clean", "scene_id": sid, "dtype": dtype,
             "scores": {"neutral": {"official": {"nll": {"red": 1.0}, "margin": 0.5}}}},
            {"kind": "diagnostics", "scene_id": sid, "dtype": dtype, "seconds": 0.01}]
    return rows, {"g": torch.full((3,), float(len(sid)))}


@pytest.fixture
def runner(monkeypatch, tmp_path, cuda_shim):
    monkeypatch.setattr(run, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(run, "protocol_commit", lambda root: "protocol")
    monkeypatch.setattr(run, "git", lambda root, *a: "head")
    monkeypatch.setattr(run, "scenes", lambda root: SCENES)
    monkeypatch.setattr(run, "image", lambda root, row: torch.zeros(3, 4, 4))
    monkeypatch.setattr(run, "Session", FakeSession)
    monkeypatch.setattr(run, "unit", fake_unit)
    out = tmp_path / run.OUT / "run"

    def call(*extra, deadline=None, hourly="2.49", budget="12"):
        now = time.time()
        argv = ["run", "main", "--device", "cuda", "--hourly-usd", hourly, "--pod-start-epoch", str(now),
                "--deadline-epoch", str(deadline if deadline is not None else now + 3600), "--budget-gate-usd", budget, *extra]
        monkeypatch.setattr(sys, "argv", argv)
        run.main()

    return call, out


def _records(out, d):
    return [json.loads(x) for x in (out / d / "records.jsonl").read_text().splitlines()]


def test_cuda_guard_entry_point_accepts_unindexed_device(cuda_shim):
    g, info, check = run.make_guard(torch.device("cuda"), time.time() + 60)
    assert g.device == torch.device("cuda", 0) and info["total_bytes"] == 96 * 1024**3
    check()


def test_preflight_then_resume_completes_without_duplication(runner):
    call, out = runner
    call("--max-scenes", "1")
    prog = json.loads((out / "progress.json").read_text())
    assert prog["status"].startswith("preflight-complete")
    assert "projection_after_first_scene" in prog
    man = json.loads((out / "integrity.json").read_text())
    assert man["after_scene"] == 1 and man["state"]["bf16"]["scenes"] == [SCENES[0]["scene_id"]]
    call("--resume")
    prog = json.loads((out / "progress.json").read_text())
    assert prog["status"] == "completed" and prog["drift_check_exact"] == {"bf16": True, "fp32": True}
    assert len(prog["resumes"]) == 1 and prog["resumes"][0]["at_scene"] == 1
    for d in ("bf16", "fp32"):
        sids = [r["scene_id"] for r in _records(out, d) if r["kind"] == "clean"]
        assert sids == [s["scene_id"] for s in SCENES]  # each scene once, in the frozen order
        assert len(list((out / d / "tensors").glob("*.npz"))) == len(SCENES)
    assert [u["scene"] for u in prog["units"]] == [s["scene_id"] for s in SCENES for _ in ("bf16", "fp32")]


def test_resume_refused_when_records_change(runner):
    call, out = runner
    call("--max-scenes", "1")
    with (out / "fp32" / "records.jsonl").open("a") as fh:
        fh.write(json.dumps({"kind": "clean", "scene_id": "tampered"}) + "\n")
    with pytest.raises(run.Stop, match="integrity manifest"):
        call("--resume")


def test_resume_refused_without_preflight(runner):
    call, out = runner
    with pytest.raises(run.Stop, match="no integrity manifest"):
        call("--resume")


def test_second_fresh_start_refused(runner):
    call, out = runner
    call("--max-scenes", "1")
    with pytest.raises(run.Stop, match="Single shot"):
        call("--max-scenes", "1")


def test_throughput_gate_stops_after_first_scene(runner, monkeypatch):
    call, out = runner

    def slow_unit(*a, **k):
        time.sleep(0.2)
        return fake_unit(*a, **k)

    monkeypatch.setattr(run, "unit", slow_unit)
    # first paired scene takes ~0.4 s, so 4 more need ~1.6 s: the deadline in 1.0 s passes the guard
    # for scene 1 but not the projection
    with pytest.raises(run.Stop, match="throughput gate"):
        call(deadline=time.time() + 1.0)
    assert len([r for r in _records(out, "bf16") if r["kind"] == "clean"]) == 1  # first scene kept


def test_budget_gate_at_resume(runner):
    call, out = runner
    call("--max-scenes", "1")
    with pytest.raises(run.Stop, match="G9 budget gate at resume"):
        call("--resume", budget="0.01")


class FakeBackend:
    def __init__(self, model):
        self.model = model

    def score_official(self, x, ctx):
        self.model(torch.ones(1, 2, dtype=self.model.weight.dtype))
        return {"nll": {"red": 1.0}, "margin": 0.5, "chosen": "red"}

    def score_float(self, x, ctx):
        return {"nll": float(x.double().sum())}  # linear in x, so finite differences equal the gradient

    def grad_nll(self, x, ctx):
        return torch.ones_like(x)


class FakeModelSession(FakeSession):
    def __init__(self, device, dtypes):
        super().__init__(device, dtypes)
        self.models = {d: torch.nn.Linear(2, 2).to(run.DTYPES[d] if hasattr(run, "DTYPES") else torch.float32) for d in dtypes}

    def backend(self, dtype, row, x, guard, *, float_check=True):
        guard()
        return FakeBackend(self.models[dtype])


@pytest.fixture
def cmd_runner(monkeypatch, tmp_path, cuda_shim):
    import qwen_precision.load as load_mod

    monkeypatch.setattr(run, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(run, "git", lambda root, *a: "head")
    monkeypatch.setattr(run, "scenes", lambda root: SCENES)
    monkeypatch.setattr(run, "image", lambda root, row: torch.zeros(3, 4, 4))
    monkeypatch.setattr(run, "Session", FakeModelSession)
    monkeypatch.setattr(run, "settings", lambda device: {"device": str(device)})
    monkeypatch.setattr(run, "fp32_matmul_probe", lambda device: {"passes_1e-5": True})
    monkeypatch.setattr(load_mod, "weights_facts", lambda: {"sha256": "x"})

    def call(*argv):
        monkeypatch.setattr(sys, "argv", ["run", *argv])
        run.main()

    return call, tmp_path / run.OUT / "audit"


@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
def test_audit_entry_point_on_cuda(cmd_runner, dtype):
    call, out = cmd_runner
    call("audit", "--dtype", dtype, "--device", "cuda")
    rec = json.loads((out / f"audit_{dtype}_cuda.json").read_text())
    assert rec["determinism"]["score_repeat_identical"] and rec["determinism"]["gradient_repeat_bit_identical"]
    assert "peak_reserved_gib" in rec["memory"] and "load_seconds" in rec["memory"]


def test_clean_entry_point_on_cuda(cmd_runner):
    call, out = cmd_runner
    call("clean", "--device", "cuda")
    rec = json.loads((out / "clean_cuda.json").read_text())
    assert len(rec["scores"]) == len(SCENES) * 2 * 2 and "peak_reserved_gib" in rec["memory"]


def test_fd_entry_point_on_cuda(cmd_runner):
    call, out = cmd_runner
    call("fd", "--dtype", "fp32", "--device", "cuda")
    rec = json.loads((out / "fd_fp32_cuda.json").read_text())
    assert rec["G7_pass"] is True and "peak_reserved_gib" in rec["memory"]
