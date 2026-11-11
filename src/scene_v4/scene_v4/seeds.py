"""Process-stable seeds.

The scene graph is not drawn from the seed. The integer is stored so a record
is a pure function of the generator and the scene id. The domain string is
``study2-development-v4``. It is not ``study2-development-v1``, not
``study2-development-v2``, not ``study2-development-simple``, and not a
sealed domain.
"""

from __future__ import annotations

import hashlib

DEVELOPMENT_SEED_DOMAIN = "study2-development-v4"
SEED_MODULUS = 2_147_483_647


def stable_seed_int(*parts: object, modulus: int = SEED_MODULUS) -> int:
    """Non-negative integer from the MD5 of ``parts``, joined by NUL bytes."""
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    return int(digest, 16) % modulus


def development_seed(scene_id: str) -> int:
    """Seed for one development scene. It does not place objects."""
    return stable_seed_int(DEVELOPMENT_SEED_DOMAIN, scene_id)
