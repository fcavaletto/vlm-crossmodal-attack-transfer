# Checkpoint 1.5 review packet

**DEVELOPMENT ONLY.** No attack was run. H\* was not fit. No held-out scene was generated. Qwen was not loaded. The Checkpoint 1 files under `docs/checkpoint-1/` were not edited. The diff of this branch against `main` contains Checkpoint 1 because the branch starts at `90c6ffe83d61514da217a20f8dfc2f202afbbe69`, not at `main`.

The protocol commit `e8bf93ea1b49e6a2c721431b82c1d132ca9a95bc` is the parent of `a6f3fdc91b263787390c42da0f709c8303af75a9`, which is `git_head_at_run` in `development_v2/run_meta.json`. That run is the first call that scored the v2 scenes. `validation.verify` checks that `VALIDATION_PROTOCOL.md` still matches the sha256 stored at the start of the run.

## 1. What changed and why

Four unmerged branches were merged with `--no-ff` onto `checkpoint-1.5/review-packet`. The merges did not overlap, so no scientific choice was resolved by editing a conflict.

The locked measurement is in `VALIDATION_PROTOCOL.md`. The strict parser stays official. The format line is not added. Helpful text stays task guidance. Every v2 scene and all four contexts are scored. The class-loss argmin is stored beside the strict parse. The image gradient stays the canonical-word loss. SmolVLM2 stays the only loaded model.

The scene distribution, the spatial question, and the counting label set are the Agent B revisions. Those revisions, and the decisions not to adopt the format line or the class-loss gradient, are in `PROTOCOL_CHANGELOG.md`. Each one was specified before the v2 numbers.

The instrument does not meet the pre-specified adequacy rule. That is an acceptable result. No scene was dropped.

## 2. Agent reports at their head SHAs

- Agent A, answer scoring: [AGENT_A_REPORT.md](https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/blob/7d6094855ade2ed36eabde6daf46da333603858b/docs/checkpoint-1.5/AGENT_A_REPORT.md) at `7d6094855ade2ed36eabde6daf46da333603858b`.
- Agent B, scenes: [AGENT_B_REPORT.md](https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/blob/28c50d320fb780fa7a7450abd7bdba6d18b443a7/docs/checkpoint-1.5/AGENT_B_REPORT.md) at `28c50d320fb780fa7a7450abd7bdba6d18b443a7`.
- Agent C, models: [AGENT_C_REPORT.md](https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/blob/b27cfff03c029e0d0f19052793a09cf645678dd2/docs/checkpoint-1.5/AGENT_C_REPORT.md) at `b27cfff03c029e0d0f19052793a09cf645678dd2`.
- Agent D, H\*: [AGENT_D_REPORT.md](https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/blob/be8be387fe93fd071d31543273baacfa476cb61c/docs/checkpoint-1.5/AGENT_D_REPORT.md) at `be8be387fe93fd071d31543273baacfa476cb61c`.

## 3. Branches, commits, and pull requests

| Source | Branch | Head | Pull request | Merge on this branch |
| --- | --- | --- | --- | --- |
| Answer scoring | `checkpoint-1.5/answer-scoring` | `7d6094855ade2ed36eabde6daf46da333603858b` | https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/pull/8 | `c141628` |
| Scene redesign | `checkpoint-1.5/scene-redesign` | `28c50d320fb780fa7a7450abd7bdba6d18b443a7` | https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/pull/10 | `6eac5e0` |
| Model feasibility | `checkpoint-1.5/model-feasibility` | `b27cfff03c029e0d0f19052793a09cf645678dd2` | https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/pull/7 | `8d40da2` |
| H\* estimator | `checkpoint-1.5/h-star-estimator` | `be8be387fe93fd071d31543273baacfa476cb61c` | https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/pull/9 | `a07ee8f` |

This packet is branch `checkpoint-1.5/review-packet`. The protocol commit is `e8bf93ea1b49e6a2c721431b82c1d132ca9a95bc`. The runner commit, and the git head recorded at inference, is `a6f3fdc91b263787390c42da0f709c8303af75a9`. None of these branches is merged to `main`.

## 4. Tests and verification

From the worktree, with `/Users/fedecava/Development/_c0-work/venv-vlm/bin/python` (Python 3.12.14, torch 2.15.0, NumPy 2.5.4, Pillow 13.0.0). This command does not load SmolVLM2.

```bash
PYTHONPATH=src/validation:src/scoring:src/vlm_iface:src/pilot:src/analysis:src/scene_v2:src/scene_gen \
  python -m pytest tests/scoring tests/vlm_iface tests/pilot tests/scene_v2 tests/scene_gen tests/analysis tests/validation -q
```

227 passed, 4 failed. The failures are interpreter pins, not changed assertions:

- `tests/scene_v2/test_v2_balance.py::test_metadata_describes_the_sample` and `tests/scene_gen/test_balance_and_manifest.py::test_metadata_describes_the_sample` require the running Pillow to be 11.3.0. This interpreter is 13.0.0, so regenerated metadata records 13.0.0. The companion pixel test, `test_committed_sample_matches_regeneration_and_checksums`, passed: the regenerated v2 pixels match the committed files. Those files were not regenerated for the model run.
- `tests/analysis/test_verify_hstar.py` requires `numpy_version == "1.26.4"`. This interpreter has 2.5.4.
- `tests/analysis/test_hstar_simulation.py::test_committed_precision_file_matches_the_generator` rebuilds the committed precision file. On Python 3.12.14 with NumPy 1.26.4 the coverage counts match, including 143/197, 168/200, 183/200, 190/200, and 186/200 in the source report's reading, and the recommendation remains 24 groups. The equality test still fails because five median interval widths differ in the last bit. The committed file was not rewritten. `verify_hstar` on that same NumPy passed with `n_mismatches` 0.

`tests/validation` alone: 9 passed. Those tests do not load a model.

The v2 verifier rebuilds `summary.json` from the compact rows. It does not load a model.

```bash
PYTHONPATH=src/validation:src/scoring:src/vlm_iface:src/pilot:src/analysis:src/scene_v2:src/scene_gen \
  python -m validation.verify
```

```json
{"ok": true, "n_mismatches": 0, "label": "DEVELOPMENT ONLY"}
```

The MPS command, on the author's Mac, MPS only:

```bash
HF_HOME=/Users/fedecava/Development/models/huggingface \
PYTHONPATH=src/validation:src/scoring:src/vlm_iface:src/pilot:src/analysis:src/scene_v2:src/scene_gen \
  python -m validation.run --device mps
```

Exit 0. The model was `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`, float32, eager, `do_image_splitting=False`, `max_new_tokens=16`, leading-space answer tokens. Load 4.956 s. Gate 18.231 s. Scoring 289.931 s. No gradient-pass time, because that pass did not start.

## 5. Original versus revised development performance

Checkpoint 1, cited from `docs/checkpoint-1/` and not refit. Denominator 24 scenes, or 8 in a family cell. Correct counts:

| Family | neutral | helpful_nonrevealing | contradictory | answer_revealing |
| --- | ---: | ---: | ---: | ---: |
| attribute | 6 | 1 | 3 | 5 |
| counting | 2 | 3 | 0 | 8 |
| spatial | 2 | 1 | 0 | 0 |
| all | 10 | 5 | 3 | 13 |

Three scenes in three groups were clean-correct under both primary contexts.

v2, this run, denominator 24 in each family cell and 72 overall. Cells are correct / valid-wrong / invalid. **DEVELOPMENT ONLY.**

| Family | neutral | helpful_nonrevealing | contradictory | answer_revealing |
| --- | --- | --- | --- | --- |
| attribute | 10 / 9 / 5 | 0 / 2 / 22 | 10 / 8 / 6 | 20 / 2 / 2 |
| counting | 10 / 12 / 2 | 10 / 14 / 0 | 0 / 24 / 0 | 23 / 1 / 0 |
| spatial | 6 / 18 / 0 | 12 / 12 / 0 | 7 / 9 / 8 | 2 / 13 / 9 |
| all | 26 / 39 / 7 | 22 / 28 / 22 | 17 / 41 / 14 | 45 / 16 / 11 |

Overall strict accuracy is 26/72, 22/72, 17/72, and 45/72. These scenes are not the Checkpoint 1 scenes. A scene-by-scene comparison is not defined.

Primary-context rates, which the protocol compares with the band 0.60 to 0.80: attribute 10/24 and 0/24, counting 10/24 and 10/24, spatial 6/24 and 12/24. Every family is below 0.50 in at least one primary context. The pre-specified recommendation is drop or replace for each family. The rows stay in the table.

Counterfactual pairs, 36 per context. Both arms strict-correct: neutral 4/36, helpful_nonrevealing 4/36, contradictory 4/36, answer_revealing 19/36.

The official parse matches the stored generation on 288/288 rows. Among 234 valid rows, the class-loss argmin matches the official label on 230/234. The canonical-word argmin matches on 179/234. There are 4 class-loss disagreements and 55 word-loss disagreements. The 230/234 figure is a description of this sample. It is not a reason to change the official parser.

On attribute under `helpful_nonrevealing`, 22/24 outputs are invalid with reason `not_exact`. The strings are sentences. On counting under `contradictory`, 0/24 are correct, all 24 parses are valid, and each parsed count word or its digit occurs in that scene's contradictory sentence. Those two descriptions were not used to edit the prompt.

Thirteen scenes are strict-correct under both `neutral` and `helpful_nonrevealing`: eight counting scenes in six groups, and five spatial scenes in five groups. No attribute scene is in that set. The groups are `cf2-counting-02`, `cf2-counting-04`, `cf2-counting-05`, `cf2-counting-06`, `cf2-counting-08`, `cf2-counting-09`, `cf2-spatial-00`, `cf2-spatial-03`, `cf2-spatial-05`, `cf2-spatial-06`, and `cf2-spatial-11`.

## 6. Recommended model

Keep SmolVLM2-500M at the pinned revision, MPS, float32, eager, image splitting off. Qwen3-VL-2B-Instruct stays unadopted. This run did not load it. The v2 task rates are not a reason to replace the model.

## 7. Recommended task families

None of the three families meets the design band in both primary contexts. The recommendation, from the rule fixed in the protocol, is to drop or replace attribute, counting, and spatial before a confirmatory three-family study. The full table above is the evidence. This packet does not delete a family from the file.

Attribute's helpful context is 0/24 and contributes no scene that is clean-correct under both primary contexts. A later three-family fit cannot be populated with an attribute group from this clean-correct set. That is a description of the set, not a fit.

## 8. Recommended answer-scoring approach

Keep D4 as the official label. Do not adopt the format line. Keep storing the class-loss argmin, and keep the canonical-word loss beside it. Do not switch the differentiated image gradient to the class loss until the lead accepts that revision. The agreement counts in section 5 are not the selection criterion.

## 9. Eligible counterfactual-group counts

The locked eligible-group count is the number of groups with at least one scene that is strict-correct under both primary contexts and whose canonical-word image gradients on those contexts are finite with L1 norm greater than zero, with empty ambiguity flags, non-revealing primary text, and primary strings shared inside the group.

That count was **not established**. After scoring, OS-available memory was 5583.9 MB, below the pre-specified 6 GiB guard, so the full gradient pass did not start. `n_groups_eligible` is null.

The one-scene check is the full-resolution gate on `dev2-attribute-00-a`. Both primary-context canonical-word gradients were finite, nonzero in every component, shaped `(3, 384, 512)`, and unchanged when repeated (`repeat_max_abs_diff` 0). Neutral L1 was 3373.643 and helpful L1 was 4181.158. These magnitudes are a finiteness check. They are not an attack step size. The two alignment scalars for that one scene are stored and were not regressed.

Separate from eligibility: 13 scenes in 11 groups are clean-correct under both primary contexts. Eleven is not the eligible-group count.

## 10. Proposed primary analysis and sample size

From Agent D, not from a fit on these scenes. The primary interval is the unrestricted wild cluster bootstrap, Webb six-point weights, equal-tailed percentile interval, clustered on `counterfactual_group_id`, both ordered directions kept. For three task families the estimator returns an interval only when there are at least 5 groups. On the committed fabricated grid, 24 was the smallest point whose estimated coverage of the known coefficient met the pre-specified 90 percent description. That count does not promise an interval excluding zero. The code enforces 5. It does not enforce 24. `h_star_fitted` is false.

## 11. Open scientific risks

- No family is inside the 0.60–0.80 band on both primary contexts. Failure of the instrument was an allowed result, and it is the result.
- Attribute helpful outputs are sentences. The strict parser marks 22/24 invalid. The format line was measured on the old sample and was not adopted. This run does not reopen that choice.
- Counting under the contradictory sentence is 0/24 correct, and the parsed count is in the sentence on 24/24. The wider count vocabulary did not make that cell a solved visual task.
- Spatial under `answer_revealing` is 2/24. Copying the named color is still weak.
- The clean-correct set has no attribute scene. A three-family design is not what this clean-correct set supports.
- Eleven clean-correct groups sit above the algebraic floor of 5 and below the coverage description of 24. Gradients on those scenes were not measured, so neither comparison is an eligible-group result.
- The class-loss argmin agrees with the official label more often than the word argmin does (230/234 against 179/234). Switching the gradient because of that gap would change \(A(s, t)\) after seeing this sample. The protocol does not switch it.
- PNG pixels were scored from the committed files. Pillow on this Mac is 13.0.0. The committed checksums were produced with Pillow 11.3.0. The regeneration test matched pixels.

## 12. Decision required before attack development

Attacks stay unauthorized. This packet does not choose a step size, a budget, or a model from an attack.

The lead decides whether to drop or replace the three families, whether a later development run may compute the canonical-word gradients when OS-available memory is below 6 GiB, and whether a future gradient may differentiate the class loss. Those decisions have to be written before the result that would be judged on them. H\* stays unfitted. The eligible-group count on this sample is not available until the gradient half of the inclusion rule is measured.
