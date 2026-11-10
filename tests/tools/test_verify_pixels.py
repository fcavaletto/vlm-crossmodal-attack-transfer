"""The non-macOS split check compares decoded PNGs exactly. These tests show it is not loose."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("verify_reproduction", ROOT / "tools" / "verify_reproduction.py")
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def same(a: Path, b: Path) -> bool:
    fa, ma, sa, ia, xa = verify.decoded(a)
    fb, mb, sb, ib, xb = verify.decoded(b)
    return (fa, ma, sa, ia, xa.dtype, xa.shape) == (fb, mb, sb, ib, xb.dtype, xb.shape) and np.array_equal(xa, xb)


@pytest.fixture
def base(tmp_path):
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (12, 16, 3), dtype=np.uint8)
    path = tmp_path / "a.png"
    Image.fromarray(arr, "RGB").save(path, compress_level=9)
    return arr, path


def test_different_compression_same_pixels_passes(base, tmp_path):
    arr, path = base
    other = tmp_path / "b.png"
    Image.fromarray(arr, "RGB").save(other, compress_level=1)
    assert path.read_bytes() != other.read_bytes()
    assert same(path, other)


@pytest.mark.parametrize("change", ["one_level", "channel_swap", "size", "mode"])
def test_any_pixel_or_format_change_fails(base, tmp_path, change):
    arr, path = base
    other = tmp_path / "b.png"
    if change == "one_level":
        x = arr.copy(); x[5, 7, 1] ^= 1
        Image.fromarray(x, "RGB").save(other)
    elif change == "channel_swap":
        Image.fromarray(np.ascontiguousarray(arr[..., ::-1]), "RGB").save(other)
    elif change == "size":
        Image.fromarray(arr[:, :15].copy(), "RGB").save(other)
    else:
        Image.fromarray(arr, "RGB").convert("RGBA").save(other)
    assert not same(path, other)
