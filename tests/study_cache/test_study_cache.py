"""The opt-in model-cache launcher. No model weights, no network.

The original guards live in preserved files and are not edited. These tests pin three things:
the default refusal is unchanged, the override works only when asked for by name, and the
pinned revision does not move.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from study_cache import GUARDED_MODULES, OVERRIDE_ENV, apply_override, resolve_override

ROOT = Path(__file__).resolve().parents[2]
STUDY_CACHE = "/Users/fedecava/Development/models/huggingface"
PINNED = "89644892e4d85e24eaac8bacfd4f463576704203"


@pytest.fixture(autouse=True)
def _restore(monkeypatch: pytest.MonkeyPatch):
    """Restore the environment and every guard attribute, whatever the code under test assigns."""
    import importlib

    for name in ("HF_HOME", OVERRIDE_ENV, "HF_HUB_OFFLINE"):
        monkeypatch.setenv(name, "placeholder")
        monkeypatch.delenv(name)
    for name in GUARDED_MODULES:
        module = importlib.import_module(name)
        monkeypatch.setattr(module, "REQUIRED_HF_HOME", module.REQUIRED_HF_HOME)
    yield


def test_historical_pins_are_untouched() -> None:
    from dev_pilot import lock
    from qwen_rgb import pins

    assert pins.REQUIRED_HF_HOME == STUDY_CACHE
    assert lock.REQUIRED_HF_HOME == STUDY_CACHE
    assert pins.PINNED_REVISION == PINNED


def test_default_assigns_the_study_cache_with_the_historical_record() -> None:
    from qwen_rgb.measure import prepare_environment

    assert resolve_override(os.environ) is None
    assert prepare_environment() == {
        "HF_HOME": STUDY_CACHE,
        "HF_HOME_assigned_by_script": True,
        "HF_HUB_OFFLINE_was_set": False,
        "HF_HUB_OFFLINE_set_by_script": False,
    }


def test_default_still_refuses_another_hf_home(monkeypatch, tmp_path: Path) -> None:
    from dev_pilot.run import _prepare_cache
    from qwen_rgb.measure import prepare_environment

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    with pytest.raises(SystemExit, match="Refusing to use another cache"):
        prepare_environment()
    with pytest.raises(SystemExit):
        _prepare_cache()


def test_override_repoints_both_guards_and_is_recorded(monkeypatch, tmp_path: Path) -> None:
    from dev_pilot.run import _prepare_cache
    from qwen_rgb.measure import prepare_environment, weight_file

    monkeypatch.setenv(OVERRIDE_ENV, str(tmp_path))
    record = apply_override(resolve_override(os.environ))
    assert record == {
        "model_cache_override": str(tmp_path),
        "model_cache_default": STUDY_CACHE,
        "guards_repointed": list(GUARDED_MODULES),
    }
    environment = prepare_environment()
    assert environment["HF_HOME"] == str(tmp_path)  # the run's own record shows the cache that was read
    _prepare_cache()
    assert weight_file() == (
        tmp_path / "hub" / "models--Qwen--Qwen3-VL-2B-Instruct" / "snapshots" / PINNED / "model.safetensors"
    )


def test_override_refuses_a_conflicting_hf_home(monkeypatch, tmp_path: Path) -> None:
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv(OVERRIDE_ENV, str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(other))
    with pytest.raises(SystemExit, match="ambiguous"):
        resolve_override(os.environ)


@pytest.mark.parametrize("value", ["", "  ", "relative/cache", "/definitely/not/a/real/dir"])
def test_override_refuses_empty_relative_or_missing_paths(monkeypatch, value: str) -> None:
    monkeypatch.setenv(OVERRIDE_ENV, value)
    with pytest.raises(SystemExit):
        resolve_override(os.environ)


def _launch(tmp_path: Path, extra_env: dict[str, str]) -> subprocess.CompletedProcess:
    probe = tmp_path / "cache_probe_module.py"
    probe.write_text(
        "import sys\n"
        "from qwen_rgb.measure import prepare_environment, weight_file\n"
        "print(prepare_environment()['HF_HOME']); print(weight_file()); print(sys.argv[1:])\n",
        encoding="utf-8",
    )
    env = {key: value for key, value in os.environ.items() if key not in {"HF_HOME", OVERRIDE_ENV}}
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), env.get("PYTHONPATH", "")])
    env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-m", "study_cache", "cache_probe_module", "--flag"],
        capture_output=True, text=True, cwd=ROOT, env=env,
    )


def test_launcher_without_the_variable_runs_the_module_unchanged(tmp_path: Path) -> None:
    done = _launch(tmp_path, {})
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines()[0] == STUDY_CACHE
    assert done.stdout.splitlines()[2] == "['--flag']"
    assert "study_cache" not in done.stderr

    refused = _launch(tmp_path, {"HF_HOME": str(tmp_path)})
    assert refused.returncode != 0
    assert "Refusing to use another cache" in refused.stderr


def test_launcher_with_the_variable_uses_that_cache_and_says_so(tmp_path: Path) -> None:
    cache = tmp_path / "hf"
    cache.mkdir()
    done = _launch(tmp_path, {OVERRIDE_ENV: str(cache)})
    assert done.returncode == 0, done.stderr
    lines = done.stdout.splitlines()
    assert lines[0] == str(cache)
    assert lines[1].startswith(str(cache)) and PINNED in lines[1]
    assert OVERRIDE_ENV in done.stderr and STUDY_CACHE in done.stderr
