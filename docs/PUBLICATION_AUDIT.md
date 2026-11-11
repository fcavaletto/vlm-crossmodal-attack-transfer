# Publication-readiness audit (2026-11-09)

Scope: the whole tree and full history of `checkpoint-2.3/exploratory-paths` relative to `main`: 113 commits, 866 files, 16.3 MB tracked. Files beyond GitHub's comparison display limit are included.

No frozen experimental file, record or historical result was changed by this audit.

## 1. What should not be published

| Check | Method | Result |
|---|---|---|
| Secrets (HF, OpenAI, GitHub, AWS and RunPod tokens; private keys; passwords) | Regex over the tree **and every patch in the history** | None found |
| Email addresses | Regex over the tree | None, except `noreply` commit trailers |
| Model weights, archives, tensors (`.safetensors`, `.bin`, `.pt`, `.npz`, `.zip`, …) | File names in every commit | None ever committed. Tensors (≈ 0.9 GB) are git-ignored and local only. |
| Large files | Largest blob in history | 1.6 MB (`paths/records.jsonl`). 318 PNGs are synthetic scenes and figures. |
| Machine-specific paths | `/Users/` in the tree | 127 lines, all under the author's username: two local cache and venv prefixes, in historical records, protocols and four source modules. No secret content. See §3. |
| Hardware identifiers | Search for hostnames and serial numbers | Only the laptop model (MacBook Air M4, `Mac16,12`), kept as reproduction metadata. No hostname or serial number. |
| Committed bytecode | History | 32 `__pycache__/*.pyc` files were committed in early checkpoints and later deleted. They contain only local source paths. They remain in history; rewriting history is not recommended. |
| Infrastructure and billing | Search for RunPod, pod IDs and IP addresses | Only public price-list references in planning documents. No account or pod identifiers. |
| Personal planning content | Read | `PORTFOLIO_PLAN.md`, `RESEARCH_STRATEGY.md` and the 2026-11-08 audit discuss applications and deadlines openly. Nothing private. Resolved in §5. |

## 2. Consistency and reproducibility

| Check | Result |
|---|---|
| Frozen 2.4 files (15) against `FROZEN.json` | All hashes match. None was modified after the freeze commit `c35b02c`. |
| Freeze before inference | GitHub received the `c35b02c` push at 16:17:36 UTC. The run's first inference sample is at 16:17:47 UTC. |
| Held-out split regeneration (`python -m scene_confirm`) | Byte-identical checksums, manifest and all PNGs, under Pillow 13.0.0 |
| Analysis from committed records only, on a **fresh clone with no tensors** | `confirm_per_group.csv`: byte-identical. `figure1_selection.svg`: byte-identical. `pilot_analysis.json`: byte-identical. `confirm_analysis.json`: identical except its `input` field, which records an absolute path. |
| README, manuscript, summary and report numbers against `confirm_analysis.json` | All headline numbers match:<br>• +6.50 [6.08, 6.92]<br>• 3% [−2%, 9%]<br>• −1.38 [−1.71, −1.06]<br>• 96/96<br>• flip rates 0.60 / 0.78 against 0.18 / 0.23<br>• transfer +5.54 / +6.64<br>• 3×, 2.8× and 0.6× ratios |
| Test suites, run per directory across all 20 directories | 538 passed, 2 failed. The 2 failures are deliberate guards; see §3. Running all of `tests/` in one pytest call fails at collection, on a duplicate test-module name. |

## 3. Discrepancies found (reported, not silently corrected)

1. **Pillow version.**
   - The README and the 2.4 report §8 said Pillow 11.3.0. The confirmation split was generated with 13.0.0, as its metadata records.
   - Docs are corrected, with a dated note in the report.
   - `src/scene_confirm/pyproject.toml` still pins `Pillow==11.3.0`. It is left unchanged and documented in the README.
2. **Hard-coded `HF_HOME`.**
   - `qwen_rgb.measure.prepare_environment()` refuses any cache path except the author's (`pins.py`), so model reruns fail on other machines.
   - Analysis-only reproduction is unaffected. Documented in the README.
   - A fix (for example an opt-in override variable) changes run-time code used by every earlier checkpoint, so it needs approval.
3. **Fixture guards.**
   - `tests/scene_gen/test_render_and_files.py:139` and `tests/scene_v2/test_v2_render.py:184` assert that `fixtures/` holds only the four development sets. Their comment says a held-out name "still fails".
   - They fail because `trackB_confirm_v1` was committed at the freeze, by design.
   - Updating them relaxes a guard, so it needs approval.
4. **Analysis JSON path.** `confirm_analysis.json` records `input` as an absolute local path. It is left as is, because it is a historical output.

## 4. Wording corrected in this audit

The 2.4 report carries a dated note for these. No number changed.
- **Subset relationship.** It is now "near-exact": up to 7 zero-gradient rounding ties per image lie outside FGSM's support. It was stated as "exactly" in the README, manuscript, summary and report.
- **"Clean decomposition / no confounds".** The comparison controls magnitude, sign and count. It does not separate individual coordinates from their spatial arrangement or interactions. The sparsity share is a ratio of mean differences, not an additive decomposition.
- **Checkpoint 3 scope.** Conclusions are restricted to the samplers and consensus rules actually tested. Untested estimators are explicitly not excluded.
- **2.4 report §9.** A remaining "not visible in the clean gradient" sentence is narrowed.

## 5. Resolutions (2026-11-09, approved by the author)

No frozen file, preserved file, record or historical result was changed. All history is kept.

| Item | Resolution |
|---|---|
| Fixture guards (§3.3) | Both tests now use an exact allowlist: the four development sets plus `trackB_confirm_v1`. The held-out split is accepted only as the hash-frozen one: its three frozen files must match `FROZEN.json`, every file must match `checksums.json`, and no unlisted file may be present. Any other directory in `fixtures/` still fails; this was checked by adding one. |
| Duplicate test-module names (§2) | Eleven test files in `tests/scene_v4`, `tests/scene_simple` and `tests/choice_eval` were renamed with a suite prefix (`git mv`, contents unchanged). A root `conftest.py` puts every `src/<package>` on the path. `python -m pytest` now runs every suite in one call. |
| Hard-coded `HF_HOME` (§3.2) | Added the opt-in `study_cache` launcher and `VLM_STUDY_MODEL_CACHE`. The original guards are **not edited**: `src/qwen_rgb` and `src/dev_pilot` are preserved paths under the Checkpoint 2.1 attack protocol (a test compares them with the protocol commit), and `dev_pilot/lock.py` is hashed in the Checkpoint 2.0 manifest. The launcher repoints the guard for one process, only when asked by name. Default behaviour, model id and pinned revision are unchanged. Not yet exercised with a full model run on a second machine. |
| Pillow pin (§3.1) | `src/scene_confirm/pyproject.toml` (not a frozen file) now pins `Pillow==13.0.0`, the version that built the split. The four development generators keep `Pillow==11.3.0`, which is what their committed checksums record. That difference is historical and documented in the README. |
| Planning documents (§1) | `docs/PORTFOLIO_PLAN.md` is removed from the tree and remains in history (last present in `fab773f`). `RESEARCH_STRATEGY.md` is kept, with a dated note marking it a historical record. |
| Continuous integration | `.github/workflows/ci.yml` runs the model-free tests and `tools/verify_reproduction.py` (frozen hashes, split regeneration, analysis outputs, Figure 1). It sets `HF_HUB_OFFLINE=1` and downloads no weights. Twenty `tests/vlm_iface` tests need the pinned processor files from a local Hub cache and skip in CI, with the reason printed. |
| Dates | This audit and the second wording note in the 2.4 report were first committed dated 2026-11-10. The work was done on 2026-11-09, as the commit timestamps show. The dates are corrected. |
| Analysis JSON path (§3.4) | Unchanged. The verifier compares every field except `input`. |

Test suites after these changes, in one call: 551 passed on the study machine (556 after the §6 pixel-comparison tests were added).

First CI run (`ce5c763`): macOS passed. Ubuntu had one failure, `test_process_footprint_reads_this_process`, which calls the macOS-only `proc_pid_rusage` syscall through the hash-frozen `path_probe/memory.py`. The test is now marked macOS-only, with the reason printed when it is skipped. No assertion was changed. The reproduction step now runs even if a test fails, so the two checks report separately.

## 6. Resolved: Linux PNG compression bytes in the held-out split (found 2026-11-09, resolved 2026-11-10)

**Status: resolved.** CI has passed on macOS and Ubuntu since commit `c3f264a` (2026-11-10). The discovery is kept below as it was recorded, followed by the approved resolution.

**As found on 2026-11-09, by the first CI run:**

- On macOS (Apple silicon, the study platform) every check passes: `python -m scene_confirm` regenerates all 100 files byte-for-byte.
- On Ubuntu 24.04 x86-64, with the same Python, numpy and Pillow 13.0.0:
  - all tests pass (529 passed, 22 skipped);
  - the frozen hashes, analysis outputs and Figure 1 are byte-identical;
  - the 97 regenerated PNGs have **identical decoded pixels** but different file bytes. `checksums.json` differs only because it records those file hashes.
- Cause, as far as diagnosed: PNG compression. The Linux Pillow wheel reports zlib 1.3; the compressed stream differs from the macOS wheel's while the image content is the same. The macOS wheel's zlib build was not compared directly.
- Consequence: the images the model would see are the same on both platforms. The earlier statement "regenerates byte-identically" was true only on the study platform. The README was corrected to say so.
- At the time the check was not changed. `tools/verify_reproduction.py` gained a diagnostic that prints what differs. A mismatch was still a failure, so the Ubuntu CI job was red (runs at `889b62b` and `e669d04`).

**Resolution (approved by the author, 2026-11-10).**
- On macOS, the study platform, the check is unchanged: all 100 files byte-identical.
- On other platforms, each regenerated PNG must decode to the same format, mode (RGB), size, metadata, dtype (uint8), shape and every pixel value as the committed one. No tolerance. The other split files, including `scene_graph_hashes.json`, which records a hash of each image's decoded pixels, must be byte-identical. `checksums.json` must list the same files, with identical entries for non-PNG files.
- On every platform the committed split files must match the committed `checksums.json`.
- Frozen hashes, analysis outputs and Figure 1 are checked as before, independently.
- `tests/tools/test_verify_pixels.py` shows the decoded comparison accepts re-compressed PNGs and rejects a one-level change in one channel of one pixel, a channel swap, a size change and a mode change.

**Current state.**
- **macOS:** byte-exact regeneration of all 100 held-out files.
- **Linux:** exact decoded-pixel comparison of the 97 PNGs, with every other split file byte-exact.
- **Everywhere:** frozen protocol hashes, analysis outputs, the per-group table and Figure 1 are checked exactly as before.
- **CI:** both jobs ("tests, frozen protocol, reproduction" on macos-15 and ubuntu-24.04) pass.

## 7. Licensing and citation metadata (2026-11-10)

Approved by the author: MIT for original code and tests; CC BY 4.0 for original synthetic data, records, figures and research documentation; the draft manuscript held back from the blanket grant. Files: `LICENSE` (MIT), `LICENSE-CC-BY-4.0.txt` (the official legal code, from `creativecommons.org/licenses/by/4.0/legalcode.txt`), and the README's License section.

**Third-party content check, before applying the licenses.** All 878 tracked files were checked.

| Item | Finding | Treatment |
|---|---|---|
| Vendored or copied upstream code | None. A search for copyright, license, "adapted from" and "copied from" markers found only notes about our own files. | Code is MIT |
| `fixtures/` (318 PNGs with manifests) | Rendered by this repository's own generators (`scene_*`) | CC BY 4.0 |
| Figures (`figure1_selection.svg`, `checkpoint-1.9/figures/illustration.png`) | Made by this repository's code from our scenes and records | CC BY 4.0 |
| Model weights (Qwen3-VL-2B-Instruct, SmolVLM2) | Not in the repository; downloaded from their publishers at pinned revisions | Not relicensed |
| Records with model outputs (for example the `predictions.jsonl` files) | Text and scores generated by those third-party models | CC BY 4.0 only to the extent we hold rights |
| `docs/research/` | Our review text, with short quotations of published work | Quotations remain their authors' |
| CLEVR | Discussed in `docs/data/`, never downloaded or included | n/a |
| Python dependencies | Listed in `requirements-ci.txt`, not included | Not relicensed |
| `docs/paper/MANUSCRIPT.md` | Draft | All rights reserved for now |

**Citation.** `CITATION.cff` (CFF 1.2.0) gives the publication title, the author, and the repository URL. It deliberately has no DOI, venue, peer-review status, version or release date. It validates with `cffconvert --validate` (cffconvert 2.0.0).

**Title consistency (resolved 2026-11-11).** The manuscript, README and citation titles differed: the manuscript read "Which Coordinates, Not How Many: Pre-registered Evidence that Iterative Attacks Beat Single-Step Attacks on a Vision-Language Model by Coordinate Selection", and the README and `CITATION.cff` read "Which coordinates, not how many: held-out evidence that 3-step PGD's advantage over FGSM on a vision-language model is coordinate selection". At the author's direction all three now use the publication title **"Which Coordinates, Not How Many: Matched-Count Evidence on the PGD–FGSM Gap in a Vision–Language Model"**. The Checkpoint 2.4 report keeps its original heading as a historical record, and frozen protocols and preregistration records are unchanged.
