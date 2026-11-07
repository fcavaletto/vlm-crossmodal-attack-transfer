"""SHA-256 helpers. A held-out or sealed path is refused before it is read."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .pins import BANNED_PATH_PARTS


class HeldOutPathError(RuntimeError):
    """The caller asked to read a held-out or sealed path."""


def assert_not_held_out(relative: Path) -> None:
    parts = {part.lower() for part in relative.parts}
    if parts & BANNED_PATH_PARTS:
        raise HeldOutPathError(
            f"Refusing to open {relative}. Held-out and sealed paths stay closed."
        )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, *, root: Path) -> str:
    """Hash one file under ``root``. The path is checked before the read."""
    relative = path.resolve().relative_to(root.resolve())
    assert_not_held_out(relative)
    return sha256_bytes(path.read_bytes())


def hash_files(paths: list[Path], *, root: Path) -> dict[str, str]:
    """Map repo-relative paths to SHA-256 hex digests. Insertion order is sorted."""
    hashed: dict[str, str] = {}
    ordered = sorted(paths, key=lambda item: item.resolve().relative_to(root.resolve()).as_posix())
    for path in ordered:
        relative = path.resolve().relative_to(root.resolve())
        hashed[relative.as_posix()] = sha256_file(path, root=root)
    return hashed
