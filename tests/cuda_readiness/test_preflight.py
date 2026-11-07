"""Preflight reports visibility and does not relabel a missing device."""

from __future__ import annotations

import pytest

from cuda_readiness.preflight import probe, require_cuda


def test_probe_does_not_claim_a_cuda_measurement():
    report = probe()
    assert report["cuda_numerical_validation"] == "not_executed"
    assert report["attacks_run"] is False
    assert report["model_from_pretrained_called"] is False
    assert report["held_out_scenes_opened"] is False
    assert report["h_star_fitted"] is False
    if report["torch_installed"] is not True:
        assert report["cuda_available"] is None
        assert "not a CUDA measurement" in report["note"]


def test_require_cuda_exits_when_no_device_is_visible():
    with pytest.raises(SystemExit) as caught:
        require_cuda({"cuda_available": None})
    assert caught.value.code == 2
    with pytest.raises(SystemExit) as caught:
        require_cuda({"cuda_available": False})
    assert caught.value.code == 2
    require_cuda({"cuda_available": True})
