"""The image must hold every file the pod commands read.

These tests rebuild the image's /opt/study tree from the Dockerfile COPY lines
(no Docker needed) and run the start commands inside it. They do not load Qwen.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "docker/cuda/Dockerfile"
POD_SPEC = ROOT / "docker/runpod/pod-spec.json"
JOB = ROOT / "docker/cuda/run_job.sh"

# Every repository file `cuda_validation protocol` or `measure` opens.
MEASURE_READS = (
    "docs/checkpoint-1.5/run_model_feasibility.py",
    "docs/checkpoint-1.7/development_v4/qwen3-vl-2b/predictions.jsonl",
    "docs/checkpoint-1.9/ATTACK_PROTOCOL.md",
    "docs/checkpoint-2.0/cuda/manifest.json",
    "fixtures/development_v4/manifest.jsonl",
    "fixtures/development_v4/checksums.json",
    "fixtures/development_v4/images/dev4-ref-00-a.png",
)

V2_POD_FIELDS = {
    "name", "cloud", "image", "gpu", "disk", "args", "env", "ports", "mounts",
    "templateId", "dataCenterIds", "registry", "startSsh", "startJupyter", "globalNetworking",
}


def _final_stage_lines() -> list[str]:
    text = DOCKERFILE.read_text(encoding="utf-8")
    return text.split("\nFROM ")[-1].splitlines()


def _stage_image(target: Path) -> Path:
    for line in _final_stage_lines():
        match = re.match(r"COPY (\S+) (\S+)$", line.strip())
        if not match or match.group(1).startswith("--from"):
            continue
        source, destination = ROOT / match.group(1), target / match.group(2).replace("/opt/study/", "", 1)
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source, destination)
    return target


def _image_pythonpath(target: Path) -> str:
    text = DOCKERFILE.read_text(encoding="utf-8")
    entries = re.search(r"PYTHONPATH=(\S+)", text).group(1)
    return entries.replace("/opt/study", str(target))


def test_every_file_measure_reads_is_in_the_image(tmp_path):
    staged = _stage_image(tmp_path)
    missing = [relative for relative in MEASURE_READS if not (staged / relative).is_file()]
    assert missing == []


def test_protocol_start_command_runs_inside_the_image_layout(tmp_path):
    staged = _stage_image(tmp_path)
    env = {**os.environ, "PYTHONPATH": _image_pythonpath(staged), "HF_HUB_OFFLINE": "1"}
    completed = subprocess.run(
        [sys.executable, "-m", "cuda_validation", "protocol"],
        cwd=staged, env=env, capture_output=True, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    json.loads(completed.stdout)


def test_weights_are_baked_at_the_pinned_revision_and_checked():
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "89644892e4d85e24eaac8bacfd4f463576704203" in text
    assert "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0" in text
    assert "sha256sum -c" in text
    assert "COPY --from=weights /models /models" in text
    assert "HF_HOME=/models" in text


def test_pod_body_uses_v2_fields_and_runs_the_job():
    pod = json.loads(POD_SPEC.read_text(encoding="utf-8"))
    body = pod["rest"]["body"]
    assert set(body) <= V2_POD_FIELDS
    assert body["image"]
    assert body["gpu"]["id"] == "NVIDIA L4"
    assert body["gpu"]["minCudaVersion"] == "13.2"
    assert "/opt/study/run_job.sh" in body["args"]
    assert pod["doNotPost"] is True


def test_job_prints_results_and_never_exits_into_a_rerun():
    text = JOB.read_text(encoding="utf-8")
    assert "measure --device cuda" in text
    assert "=== BEGIN" in text and "=== JOB_DONE" in text
    assert text.rstrip().endswith("exec sleep infinity")
    assert os.access(JOB, os.X_OK)
