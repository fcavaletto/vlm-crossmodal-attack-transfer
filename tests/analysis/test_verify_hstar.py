"""The H* verification command rebuilds the fabricated summary."""

from __future__ import annotations

from analysis.hstar_packet import fixture_dir
from analysis.verify_hstar import verify_hstar_summary


def test_committed_hstar_summary_matches_a_fresh_rebuild():
    report = verify_hstar_summary(fixture_dir())
    assert report["ok"], report
    assert report["n_mismatches"] == 0
    assert report["numpy_version"] == "1.26.4"
