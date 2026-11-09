# Publication-readiness audit (2026-11-10)

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
| Personal planning content | Read | `PORTFOLIO_PLAN.md`, `RESEARCH_STRATEGY.md` and the 2026-11-08 audit discuss applications and deadlines openly. Nothing private, but the author may prefer to move them. |

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
