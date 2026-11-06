"""One resident model. A live pid or another scoring process blocks a second load."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .lock import OUTPUT_RELATIVE


def resident_path(root: Path) -> Path:
    return root / OUTPUT_RELATIVE / "resident.json"


def other_scoring_commands() -> list[str]:
    """Other processes that would load a study model. This process is ignored."""
    me = os.getpid()
    completed = subprocess.run(["ps", "-ax", "-o", "pid=,command="], check=True, capture_output=True, text=True)
    hits: list[str] = []
    markers = (" -m dev_pilot", "qwen_rgb.measure", "simple_eval", "run_development_v4")
    for line in completed.stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        pid_text, _, command = text.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == me:
            continue
        if any(marker in command for marker in markers):
            hits.append(text)
    return hits


def acquire(root: Path, model_key: str) -> Path:
    """Create the resident file or raise. A dead pid is removed once."""
    others = other_scoring_commands()
    if others:
        raise RuntimeError("another scoring process is already running: " + others[0])
    path = resident_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"pid": os.getpid(), "model_key": model_key}).encode("utf-8")
    for _attempt in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            if not _stale(path):
                raise RuntimeError(f"a model is already resident according to {path}") from None
            path.unlink()
            continue
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        return path
    raise RuntimeError(f"could not acquire {path}")


def release(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if payload.get("pid") == os.getpid():
        path.unlink()


def _stale(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        pid = int(payload["pid"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return True
    return False
