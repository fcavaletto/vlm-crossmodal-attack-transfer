"""Put scene_v4 on the path and generate one sample per session."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "scene_v4"))
sys.path.insert(0, str(ROOT / "src" / "scene_gen"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from scene_v4.generate import generate_development


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def generated(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("development_v4")
    generate_development(path)
    return path


@pytest.fixture(scope="session")
def records(generated: Path) -> list[dict]:
    return load_jsonl(generated / "manifest.jsonl")
