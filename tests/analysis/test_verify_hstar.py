"""The H* verification command rebuilds the fabricated summary.

The committed summary was produced with NumPy 1.26.4. The report's
``numpy_version`` is the NumPy that reran the check. Those strings are
recorded separately and are not required to match.
"""

from __future__ import annotations

import numpy as np

from analysis.hstar_packet import fixture_dir
from analysis.verify_hstar import (
    NUMPY_VERSION_THAT_PRODUCED_THE_FILE,
    verify_hstar_summary,
)


def test_committed_hstar_summary_matches_a_fresh_rebuild():
    report = verify_hstar_summary(fixture_dir())
    assert report["ok"], report
    assert report["n_mismatches"] == 0
    produced_the_file = report["numpy_version_that_produced_the_file"]
    reran_the_check = report["numpy_version_that_reran_the_check"]
    assert produced_the_file == NUMPY_VERSION_THAT_PRODUCED_THE_FILE == "1.26.4"
    assert reran_the_check == report["numpy_version"] == np.__version__
    assert isinstance(reran_the_check, str) and reran_the_check
