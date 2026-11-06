"""Frozen identifiers for the 24 committed development scenes."""

from __future__ import annotations

MANIFEST_SHA256 = "3aea59c848aaafc76de6c70ebe77d452c5f89db7920b573161d30f9a23695453"
GATE_SCENE_ID = "dev-attribute-00-a"
IMAGE_HEIGHT = 384
IMAGE_WIDTH = 512
N_SCENES = 24
N_GROUPS = 12

EXPECTED_SCENE_IDS: tuple[str, ...] = tuple(
    f"dev-{family}-{index:02d}-{arm}"
    for family in ("attribute", "counting", "spatial")
    for index in range(4)
    for arm in ("a", "b")
)
