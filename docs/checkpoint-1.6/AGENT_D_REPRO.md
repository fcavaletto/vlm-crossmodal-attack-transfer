# Agent and responsibility

Agent D, reproducibility, Checkpoint 1.6 of `vlm-crossmodal-attack-transfer`. Research lead: Federico Cavaletto.

This workstream owns the dependency story for the checks that failed their version pins, the four tests named below, the small NumPy version record in `analysis.verify_hstar`, and this report. Attacks are not authorized. No model was loaded, no attack was run, H\* was not fit on real scenes, and nothing was trained. No cloud GPU was used. `docs/checkpoint-1/**` was not edited. Committed reference results were not rewritten.

# Executive summary

The Checkpoint 1.6 dependency story keeps two interpreters, and each pin stays what it is. The model interpreter is `/Users/fedecava/Development/_c0-work/venv-vlm`: Python 3.12.14, torch 2.15.0, transformers 5.22.1, Pillow 13.0.0, and NumPy 2.5.4. That environment was left in place. The committed development PNGs were written with Pillow 11.3.0. The H\* summary and the precision file were produced with NumPy 1.26.4.

Four tests failed on the model interpreter. Two required regenerated scene metadata to report Pillow 11.3.0. One required the H\* rebuild to report NumPy 1.26.4. One required every median interval width to match the committed precision file in the last bit. Regenerated metadata now records the Pillow that performed the regeneration. The committed metadata files must still say Pillow 11.3.0. Scene count, split, and seeds stay exact, and the committed PNG checksums stay exact. The H\* summary rebuild still requires zero numeric mismatches. It records NumPy 1.26.4 as the version that produced the file and NumPy 2.5.4 as the version that reran the check. Coverage counts, status counts, returned-interval fractions, the recommendation of 24 groups, and the "does not promise" sentence stay exact. Median interval widths may differ by at most 1e-9. On this NumPy the largest such gap was 2.22e-16. No committed fixture was rewritten. The combined suite was 231 passed and 0 failed.

# The four failures

All four were measured with `/Users/fedecava/Development/_c0-work/venv-vlm/bin/python` (Python 3.12.14, Pillow 13.0.0, NumPy 2.5.4). The pixel checksum test `test_committed_sample_matches_regeneration_and_checksums` had already passed: regenerated pixels match the committed PNGs.

1. `tests/scene_v2/test_v2_balance.py::test_metadata_describes_the_sample` loaded metadata from a fresh `scene_v2` generation and required `metadata["pillow"] == "11.3.0"`. The generator writes `PIL.__version__`, so the regeneration recorded Pillow 13.0.0.

2. `tests/scene_gen/test_balance_and_manifest.py::test_metadata_describes_the_sample` did the same for the Checkpoint 1 development sample. The committed `fixtures/development/metadata.json` still says Pillow 11.3.0. The live regeneration recorded 13.0.0.

3. `tests/analysis/test_verify_hstar.py::test_committed_hstar_summary_matches_a_fresh_rebuild` rebuilt `hstar_summary.json`. The rebuild was already `ok` with `n_mismatches` 0 on NumPy 2.5.4. The assertion `report["numpy_version"] == "1.26.4"` failed because the report string is the NumPy that reran the check.

4. `tests/analysis/test_hstar_simulation.py::test_committed_precision_file_matches_the_generator` required the rebuilt precision document to equal `precision_simulation.json` with `==`. On Python 3.12.14 and NumPy 1.26.4, coverage fractions and the recommendation of 24 groups matched the committed file, and five `median_ci_width_*` values differed in the last bit (for example 1.025924498893302 and 1.0259244988933023). On this NumPy 2.5.4 rerun the same kind of gap appears: seven of the ten median widths differ by one or two units in the last place, and the largest absolute gap is 2.22e-16. The n=6 row width that differed under NumPy 1.26.4 matches exactly under NumPy 2.5.4. Every other field matched.

# The change

- `test_metadata_describes_the_sample` in both scene suites still checks scene count, split, `seed_domain`, and `seed_modulus` on the regeneration. The regeneration's `pillow` field must equal the live `PIL.__version__`. The committed metadata file on disk must still say `"11.3.0"`, and its scene count, split, and seeds must match the regeneration. The committed PNG checksum tests are unchanged and still require Pillow 11.3.0 in `checksums.json` plus SHA-256 of the committed bytes and pixel equality.
- `verify_hstar_summary` still compares the rebuilt summary with the committed file and still fails when `n_mismatches` is not 0. The report now has `numpy_version_that_produced_the_file` (`1.26.4`) and `numpy_version_that_reran_the_check` (the live NumPy, also stored as `numpy_version`). The test requires both records and does not require the strings to be equal. `hstar.py` was not edited.
- The precision comparison is exact for coverage counts, status counts, `fraction_interval_returned`, `fraction_row_interval_returned`, the recommendation, including 24 groups, and the sentence that contains "does not promise". Only `median_ci_width_cluster` and `median_ci_width_row` may differ, by an absolute gap of at most 1e-9. `precision_simulation.json` was not edited.

# Why this is not a rewrite of evidence

The committed files are the evidence, and they were not replaced. `fixtures/development/**`, `fixtures/development_v2/**`, `src/analysis/analysis/fixtures/hstar/hstar_summary.json`, and `src/analysis/analysis/fixtures/hstar/precision_simulation.json` are unchanged. Pillow 11.3.0 remains the version recorded on the committed metadata and checksum files. NumPy 1.26.4 remains the version that produced the H\* fixtures, and the analysis package pin stays `numpy==1.26.4`. The model environment was not downgraded to those pins, and those pins were not rewritten to Pillow 13.0.0 or NumPy 2.5.4.

Pixel checksums of the committed PNGs, the H\* summary rebuild (`n_mismatches == 0`), and the development prediction checks stay exact. The 1e-9 allowance is only the last-bit median widths in the precision file. A change in a coverage count, a status count, a returned-interval fraction, or the recommendation still fails the test.

# Combined suite

Command, from the repository root, with the model interpreter:

```bash
PYTHONPATH=src/validation:src/scoring:src/vlm_iface:src/pilot:src/analysis:src/scene_v2:src/scene_gen \
  /Users/fedecava/Development/_c0-work/venv-vlm/bin/python -m pytest \
  tests/scoring tests/vlm_iface tests/pilot tests/scene_v2 tests/scene_gen tests/analysis tests/validation -q
```

Pytest summary line:

```text
231 passed, 96 warnings in 10.95s
```

0 failed. The 96 warnings are `Image.Image.getdata` deprecation notices from Pillow 13, in `tests/scene_v2/test_v2_counterfactuals.py` and `tests/scene_gen/test_shortcuts.py`. Those tests passed. They were left as they are.

The four tests:

| Test | Status |
| --- | --- |
| `tests/scene_v2/test_v2_balance.py::test_metadata_describes_the_sample` | passed |
| `tests/scene_gen/test_balance_and_manifest.py::test_metadata_describes_the_sample` | passed |
| `tests/analysis/test_verify_hstar.py::test_committed_hstar_summary_matches_a_fresh_rebuild` | passed |
| `tests/analysis/test_hstar_simulation.py::test_committed_precision_file_matches_the_generator` | passed |

`verify_hstar_summary` on this interpreter returned `ok` true, `n_mismatches` 0, `numpy_version_that_produced_the_file` `1.26.4`, and `numpy_version_that_reran_the_check` `2.5.4`.

# Reproduction

Use the model interpreter for the combined suite. Do not downgrade it to Pillow 11.3.0 or NumPy 1.26.4 in order to make a version string match.

```bash
PYTHONPATH=src/validation:src/scoring:src/vlm_iface:src/pilot:src/analysis:src/scene_v2:src/scene_gen \
  /Users/fedecava/Development/_c0-work/venv-vlm/bin/python -m pytest \
  tests/scoring tests/vlm_iface tests/pilot tests/scene_v2 tests/scene_gen tests/analysis tests/validation -q
```

The H\* summary check, which does not load a model:

```bash
PYTHONPATH=src/analysis \
  /Users/fedecava/Development/_c0-work/venv-vlm/bin/python -m analysis.verify_hstar
```

Exit 0 means `n_mismatches` is 0. `numpy_version` in that JSON is the interpreter that reran the check. `--write` would rewrite `hstar_summary.json` and is not part of the check.

An analysis environment pinned to `numpy==1.26.4`, as in `src/analysis/pyproject.toml`, remains the right place to regenerate the committed H\* fixtures if a maintainer ever does that on purpose. This checkpoint did not regenerate them.

# Git information

- Branch: `checkpoint-1.6/reproducibility`
- Start tip: `b2e7abff72b87b475d0abf8d037e1b4d2f9c0c06`
- Tests and the NumPy version record: `446ee6cec20f43ec28c70e1116019e4613bdc1c9`
- This report is the commit that follows `446ee6cec20f43ec28c70e1116019e4613bdc1c9` on the same branch.
- Not pushed. Not merged.
