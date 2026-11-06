"""Process-stable seeds.

Python's built-in ``hash()`` is salted per process unless ``PYTHONHASHSEED`` is
fixed. Study 1 avoided that trap with an MD5 digest. The same reason still
applies here, so this module uses the same construction. The domain string is
new: development seeds must not be reusable as sealed-set seeds later.
"""

from __future__ import annotations

import hashlib

DEVELOPMENT_SEED_DOMAIN = "study2-development-v1"
SEED_MODULUS = 2_147_483_647  # 2^31 - 1, fits a signed 32-bit integer


def stable_seed_int(*parts: object, modulus: int = SEED_MODULUS) -> int:
    """Non-negative integer from the MD5 of ``parts``, joined by NUL bytes."""
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    return int(digest, 16) % modulus


def development_seed(scene_id: str) -> int:
    """Seed for one development scene. The scene graph is not sampled from it.

    The graph comes from the explicit design table. The seed is stored so a
    rerun can show that the record is a pure function of (code version, scene id)
    and so later code can derive any incidental choice from a stable integer.
    This checkpoint does not use the seed to place objects.
    """
    return stable_seed_int(DEVELOPMENT_SEED_DOMAIN, scene_id)
