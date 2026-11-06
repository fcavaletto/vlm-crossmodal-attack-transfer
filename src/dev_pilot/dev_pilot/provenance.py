"""Show that the protocol commit is an ancestor and contains no predictions."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from .lock import BASE_COMMIT, MANIFEST_SHA256, PREDICTION_BLOBS, PROTOCOL_RELATIVE


def protocol_provenance(root: Path) -> dict[str, object]:
    """Read git. Refuse a dirty protocol file or a protocol commit that already had predictions."""
    head = _git(root, "rev-parse", "HEAD")
    dirty = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", PROTOCOL_RELATIVE],
        cwd=root,
        check=False,
    )
    if dirty.returncode != 0:
        raise RuntimeError("DEVELOPMENT_PROTOCOL.md differs from HEAD. Commit it before scoring.")
    protocol_commit = _git(root, "log", "-1", "--format=%H", "--", PROTOCOL_RELATIVE)
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", protocol_commit, head],
        cwd=root,
        check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError(f"{protocol_commit} is not an ancestor of {head}")
    text = _git(root, "show", f"{protocol_commit}:{PROTOCOL_RELATIVE}")
    if MANIFEST_SHA256 not in text:
        raise RuntimeError("the protocol commit does not name the v4 manifest hash")
    if "0.65" not in text or "2382" not in text or "1151" not in text:
        raise RuntimeError("the protocol commit does not name the locked rule")
    present = []
    for relative in PREDICTION_BLOBS:
        probe = subprocess.run(
            ["git", "cat-file", "-e", f"{protocol_commit}:{relative}"],
            cwd=root,
            check=False,
            capture_output=True,
        )
        if probe.returncode == 0:
            present.append(relative)
    if present:
        raise RuntimeError("predictions already exist at the protocol commit: " + ", ".join(present))
    file_bytes = (root / PROTOCOL_RELATIVE).read_bytes()
    checkpoint_16 = subprocess.run(
        ["git", "diff", "--quiet", BASE_COMMIT, "--", "docs/checkpoint-1.6"],
        cwd=root,
        check=False,
    )
    return {
        "head": head,
        "protocol_commit": protocol_commit,
        "protocol_commit_is_ancestor": True,
        "protocol_sha256": hashlib.sha256(file_bytes).hexdigest(),
        "predictions_present_at_protocol_commit": False,
        "checkpoint_1_6_diff_empty": checkpoint_16.returncode == 0,
        "manifest_sha256": MANIFEST_SHA256,
    }


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()
