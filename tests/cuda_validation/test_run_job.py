"""run_job.sh must never print JOB_DONE unless every step passed.

The job runs for real under bash, with a fake `python` and a fake `timeout`
on PATH. Nothing loads Qwen, touches a GPU, or calls RunPod.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
JOB = ROOT / "docker/cuda/run_job.sh"

FAKE_PYTHON = r'''#!{python}
import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
if args and args[0] == "-c":
    os.execv("{python}", ["{python}", *args])
def out_path():
    return Path(args[args.index("--output") + 1])
def write(payload):
    path = out_path(); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
script = Path(args[0]).name if args else ""
if script == "self_stop.py":
    print("self_stop: fake"); sys.exit(int(os.environ.get("FAKE_SELF_STOP_RC", "0")))
if script == "fetch_weights.py":
    ok = os.environ.get("FAKE_WEIGHTS", "ok") == "ok"
    write({{"all_match": ok}}); sys.exit(0 if ok else 1)
if args[:2] == ["-m", "cuda_readiness"]:
    write({{"cuda_available": True}}); sys.exit(int(os.environ.get("FAKE_PREFLIGHT_RC", "0")))
if args[:3] == ["-m", "cuda_validation", "measure"]:
    time.sleep(float(os.environ.get("FAKE_MEASURE_SLEEP", "0")))
    write({{"status": os.environ.get("FAKE_MEASURE_STATUS", "completed")}})
    if os.environ.get("FAKE_NO_GRADIENT") != "1":
        out_path().with_name("cuda_scene.gradient.f32").write_bytes(b"\0" * 16)
    sys.exit(int(os.environ.get("FAKE_MEASURE_RC", "0")))
if args[:3] == ["-m", "cuda_validation", "compare"]:
    write({{"status": os.environ.get("FAKE_COMPARE_STATUS", "differences_recorded")}})
    sys.exit(int(os.environ.get("FAKE_COMPARE_RC", "0")))
sys.exit(97)
'''

FAKE_TIMEOUT = r'''#!{python}
import subprocess, sys
args = [a for a in sys.argv[1:] if not a.startswith("--kill-after")]
limit, command = float(args[0]), args[1:]
child = subprocess.Popen(command)
try:
    sys.exit(child.wait(timeout=limit))
except subprocess.TimeoutExpired:
    child.kill(); child.wait(); sys.exit(124)
'''


def _run(tmp_path: Path, *, sidecar: bool = True, **fake: str) -> tuple[int, list[str]]:
    study = tmp_path / "study"
    out = study / "docs/checkpoint-2.1/cuda_validation"
    out.mkdir(parents=True)
    (study / "docker/cuda").mkdir(parents=True)
    if sidecar:
        (out / "mps_scene.json").write_text("{}")
        (out / "mps_scene.gradient.f32").write_bytes(b"\0" * 16)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("python", FAKE_PYTHON), ("timeout", FAKE_TIMEOUT)):
        path = bin_dir / name
        path.write_text(body.format(python=sys.executable))
        path.chmod(0o755)
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "STUDY_ROOT": str(study),
        "JOB_PYTHON": str(bin_dir / "python"),
        "JOB_EXIT_AFTER": "1",
        "JOB_DEADLINE_SECONDS": fake.pop("deadline", "60"),
        **fake,
    }
    completed = subprocess.run(["bash", str(JOB)], env=env, capture_output=True, text=True, timeout=60)
    return completed.returncode, completed.stdout.splitlines()


def _terminal(lines: list[str]) -> list[str]:
    return [line for line in lines if line.startswith(("=== JOB_DONE", "=== JOB_FAILED"))]


def test_all_steps_pass_prints_one_job_done(tmp_path):
    rc, lines = _run(tmp_path)
    assert rc == 0
    assert len(_terminal(lines)) == 1 and _terminal(lines)[0].startswith("=== JOB_DONE status=ok")
    assert "self_stop: fake" in lines


@pytest.mark.parametrize(
    ("fake", "sidecar", "expected"),
    [
        ({"FAKE_WEIGHTS": "bad"}, True, "step=weights"),
        ({"FAKE_PREFLIGHT_RC": "2"}, True, "step=preflight rc=2"),
        ({"FAKE_MEASURE_RC": "1", "FAKE_MEASURE_STATUS": "error"}, True, "step=measure rc=1"),
        ({"FAKE_MEASURE_STATUS": "stopped_unexpected_header"}, True, "reason=status_stopped_unexpected_header"),
        ({"FAKE_NO_GRADIENT": "1"}, True, "reason=gradient_file_missing"),
        ({}, False, "reason=no_mps_sidecar_in_image"),
        ({"FAKE_COMPARE_RC": "3", "FAKE_COMPARE_STATUS": "bound_violation"}, True, "step=compare rc=3"),
        ({"FAKE_COMPARE_STATUS": "blocked_protocol_mismatch"}, True, "reason=status_blocked_protocol_mismatch"),
        ({"FAKE_MEASURE_SLEEP": "5", "deadline": "2"}, True, "step=measure rc=124 reason=deadline"),
    ],
)
def test_any_failure_prints_job_failed_and_never_job_done(tmp_path, fake, sidecar, expected):
    rc, lines = _run(tmp_path, sidecar=sidecar, **fake)
    terminal = _terminal(lines)
    assert rc == 1
    assert len(terminal) == 1, terminal
    assert terminal[0].startswith("=== JOB_FAILED") and expected in terminal[0]
    assert not any("JOB_DONE" in line for line in lines)
    assert "self_stop: fake" in lines


def test_a_failed_weights_check_never_reaches_measure(tmp_path):
    _, lines = _run(tmp_path, FAKE_WEIGHTS="bad")
    assert not any(line.startswith("=== STEP measure") for line in lines)


def test_refused_self_stop_is_reported(tmp_path):
    rc, lines = _run(tmp_path, FAKE_SELF_STOP_RC="1")
    assert rc == 0
    assert any(line.startswith("=== SELF_STOP_FAILED") for line in lines)


def _fetch_module():
    spec = importlib.util.spec_from_file_location("fetch_weights", ROOT / "docker/cuda/fetch_weights.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_weight_manifest_pins_every_mac_cache_file():
    fetch = _fetch_module()
    expected = fetch.read_manifest()
    assert len(expected) == 10
    assert expected["model.safetensors"] == "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0"
    assert fetch.REVISION == "89644892e4d85e24eaac8bacfd4f463576704203"


def test_weight_check_rejects_a_changed_file(tmp_path):
    fetch = _fetch_module()
    folder = fetch.snapshot_dir(tmp_path)
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}")
    good = fetch.sha256(folder / "config.json")
    assert fetch.verify(tmp_path, {"config.json": good})["all_match"] is True
    (folder / "config.json").write_text('{"changed": 1}')
    assert fetch.verify(tmp_path, {"config.json": good})["all_match"] is False
    assert fetch.verify(tmp_path, {"missing.json": good})["all_match"] is False


def test_bash_is_available():
    assert shutil.which("bash")
