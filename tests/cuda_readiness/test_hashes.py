"""Hashes refuse held-out paths before reading them."""

from __future__ import annotations

from pathlib import Path

import pytest

from cuda_readiness.hashes import HeldOutPathError, hash_files, sha256_bytes, sha256_file


def test_hash_matches_sha256_and_refuses_held_out_names(tmp_path: Path):
    payload = b"development-result"
    path = tmp_path / "results" / "one.json"
    path.parent.mkdir()
    path.write_bytes(payload)
    assert sha256_file(path, root=tmp_path) == sha256_bytes(payload)
    assert hash_files([path], root=tmp_path) == {"results/one.json": sha256_bytes(payload)}
    for name in ("heldout", "held-out", "held_out", "sealed"):
        banned = tmp_path / name / "scene.png"
        banned.parent.mkdir()
        banned.write_bytes(b"do-not-read")
        with pytest.raises(HeldOutPathError):
            sha256_file(banned, root=tmp_path)
