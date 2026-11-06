# Checkpoint 1 development pilot

Every empirical number in this report is **DEVELOPMENT ONLY**. The run used the 24 committed development scenes. It did not fit H\*. It did not run an attack. It did not open a sealed split.

H\* stays the provisional sole primary hypothesis: gradient directional alignment between `neutral` and `helpful_nonrevealing` predicts cross-context attack transfer. These results do not test that hypothesis and do not revise it.

## 1. Executive summary

SmolVLM2-500M-Video-Instruct at revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467` ran on the author's MacBook Air M4 under PyTorch MPS. The full-resolution gate on `dev-attribute-00-a` (512×384) passed: both primary-context image gradients were finite, nonzero in every component, and unchanged when repeated. The pilot then scored all 24 scenes in all four contexts and computed `A(neutral, helpful_nonrevealing)` and `A(helpful_nonrevealing, neutral)` on the clean images.

Clean accuracy on this sample is 10/24 under `neutral`, 5/24 under `helpful_nonrevealing`, 3/24 under `contradictory`, and 13/24 under `answer_revealing` (DEVELOPMENT ONLY). The spatial family is weak in every context, including the answer-revealing control (0/8). Counting under the contradictory sentence is 0/8, and every one of those outputs is the word `four`, which is the count that sentence states. Twenty of 96 generations are full sentences. The approved parser marks every one of them invalid because the string is not exactly one vocabulary item. All 48 primary-context gradients are finite and nonzero, and all 48 alignment values are defined. Three scenes, in three counterfactual groups, are correct under both primary contexts. That is the size of the proposed H\* population on this sample. No coefficient was fit.

## 2. Integrated code versions

The branch `checkpoint-1/dev-pilot` starts at `origin/main` `65a8d8ffb2f22f3d284b64e2d481151474761c5e`. Four unmerged pull requests were merged with `--no-ff`, so their source commits remain in the history. They were not merged to `main`.

| PR | Branch | Head kept | Merge commit |
| --- | --- | --- | --- |
| [PR 1](https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/pull/1) | `checkpoint-0/research-design` | `4bde745901ee0d457d89fd073d34452678c9cb28` | `ac3d7e1` |
| [PR 2](https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/pull/2) | `checkpoint-0/model-gradients` | `920fd4a5cce4bd9f84fdf5f804fae63391d538f9` | `00ec307` |
| [PR 3](https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/pull/3) | `checkpoint-0/scene-generator` | `d738f5f13868c08824ea94abb8abfe3693d549a2` | `3b4b500` |
| [PR 4](https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/pull/4) | `checkpoint-0/analysis-plan` | `5b522752571d8d9ebcfb86c724cf6023a9997d57` | `d74cfcd` |

[PR 5](https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/pull/5), the review packet, is documentation. Its tree was not merged.

The integration commit is `591581a6cdc3cdc55eb1a9c53585b9bfd5978f44`. It applies the approved decisions without changing H\*, the prompt template, the 24 scenes, or the model revision.

- D1. The stored loss is the sum of token negative log-likelihoods of the canonical answer. Token count and mean token NLL are stored beside it.
- D2. The user message is `context_text + "\n" + question`. The session still wraps that string in the SmolVLM chat template.
- D3. `image_path` is resolved from the directory that contains `manifest.jsonl`. The loader checks the committed manifest sha256 `3aea59c848aaafc76de6c70ebe77d452c5f89db7920b573161d30f9a23695453` and refuses any other scene list.
- D4. A generation is valid only when minimal normalization yields exactly one task-vocabulary item or a documented alias. Attribute and spatial questions use the four color names. Counting uses `one` through `four`, and the digits `1` through `4` canonicalize to those words. A recognized color or count that misses the label is valid-but-wrong. Sentences, option lists, multiple labels, and refusals are invalid. Invalid output is a task error. Attack-success columns were not filled.
- D5. The model, revision, float32 weights, eager attention, `do_image_splitting=False`, leading-space answer tokens, and `max_new_tokens=16` are the approved settings. The runner refuses any device other than MPS.

Agent A's cluster-robust regression is still not implemented. Agent D's toy primary contrast was not promoted. The current helpful strings are task guidance. They were not rewritten, and they are not additional factual scene evidence. For an attribute scene the helpful sentence is "Identify the single object with the size and shape named in the question, then name its color."

## 3. Environment and hardware

The measurement machine is the author's MacBook Air, model Mac16,12, Apple M4, 10 cores, 25,769,803,776 bytes of unified memory, macOS 26.6 (build 25G72), arm64.

The pilot interpreter was `/Users/fedecava/Development/_c0-work/venv-vlm/bin/python`.

| Package | Version in the MPS run |
| --- | --- |
| Python | 3.12.14 |
| torch | 2.15.0 |
| transformers | 5.22.1 |
| numpy | 2.5.4 |
| Pillow | 13.0.0 |
| Device | mps |
| Parameter dtype | torch.float32 |
| Attention | eager |
| Image splitting | off |
| `max_new_tokens` | 16 |
| Answer tokenization | leading space |
| Model | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` |
| Revision | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |

`HF_HOME` was `/Users/fedecava/Development/models/huggingface`. The weights were already cached. Scene-generator tests were run in a second virtualenv with Pillow 11.3.0, the version pinned by that package and recorded in `fixtures/development/metadata.json`. Those tests are not MPS measurements.

## 4. Exact commands

From `/Users/fedecava/Development/_c0-work/checkpoint-1-dev-pilot`.

Scene tests, Pillow 11.3.0. Not an MPS result.

```bash
PYTHONPATH=src/scene_gen \
  /Users/fedecava/Development/_c0-work/venv-scene/bin/python -m pytest tests/scene_gen -q
```

Model, analysis, and pilot unit tests. This interpreter can see MPS. The tests do not produce the tables below.

```bash
PYTHONPATH=src/scene_gen:src/vlm_iface:src/analysis:src/pilot \
  /Users/fedecava/Development/_c0-work/venv-vlm/bin/python -m pytest \
  tests/vlm_iface tests/analysis tests/pilot -q
```

The MPS pilot. It runs the 512×384 gate first and stops without writing the 24-scene tables if that gate fails.

```bash
PYTHONPATH=src/scene_gen:src/vlm_iface:src/analysis:src/pilot \
  HF_HOME=/Users/fedecava/Development/models/huggingface \
  /Users/fedecava/Development/_c0-work/venv-vlm/bin/python -m pilot.run --device mps
```

Verification. This rebuilds `summary.json` from the compact JSONL files, `gate.json`, and `run_meta.json`. It does not load the model.

```bash
PYTHONPATH=src/scene_gen:src/vlm_iface:src/analysis:src/pilot \
  /Users/fedecava/Development/_c0-work/venv-vlm/bin/python -m pilot.verify
```

## 5. Clean performance tables

Denominator for a context is 24 scenes. A row is exactly one of correct, valid-but-wrong, or invalid. Invalid rows lower accuracy and are counted in the invalid rate. They are not folded into valid-but-wrong. DEVELOPMENT ONLY.

| Context | Correct | Valid-but-wrong | Invalid | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| `neutral` | 10 | 11 | 3 | 10/24 |
| `helpful_nonrevealing` | 5 | 12 | 7 | 5/24 |
| `contradictory` | 3 | 17 | 4 | 3/24 |
| `answer_revealing` | 13 | 5 | 6 | 13/24 |

By task family and context, each cell has 8 scenes. DEVELOPMENT ONLY.

| Family | Context | Correct | Valid-but-wrong | Invalid |
| --- | --- | ---: | ---: | ---: |
| attribute | `neutral` | 6 | 0 | 2 |
| attribute | `helpful_nonrevealing` | 1 | 0 | 7 |
| attribute | `contradictory` | 3 | 3 | 2 |
| attribute | `answer_revealing` | 5 | 0 | 3 |
| counting | `neutral` | 2 | 6 | 0 |
| counting | `helpful_nonrevealing` | 3 | 5 | 0 |
| counting | `contradictory` | 0 | 8 | 0 |
| counting | `answer_revealing` | 8 | 0 | 0 |
| spatial | `neutral` | 2 | 5 | 1 |
| spatial | `helpful_nonrevealing` | 1 | 7 | 0 |
| spatial | `contradictory` | 0 | 6 | 2 |
| spatial | `answer_revealing` | 0 | 5 | 3 |

The pooled family rows in `summary.json` use 32 scene-context rows (8 scenes × 4 contexts). Those denominators are labeled `scene-context rows` in the file. The table above is the per-context view.

Every canonical answer tokenized to one leading-space token, so on this sample the sum NLL and the mean token NLL are the same number. Both are stored. Mean sum NLL of the canonical answer, DEVELOPMENT ONLY: `neutral` 4.268, `helpful_nonrevealing` 5.116, `contradictory` 4.997, `answer_revealing` 3.832. The largest absolute gap between that mean and the Transformers library mean on the same logits was 9.54×10⁻⁷.

All 20 invalid outputs have parse reason `not_exact`. Each raw string is a sentence containing exactly one color word. Ten of those twenty sentences contain the true answer as a whole word. That count is a reading of the stored raw text. It is not a second official score. Under D4 those rows stay invalid.

Counting under `contradictory` is uniform: all 8 raw outputs are `4`, parsed as `four`, and the contradictory sentence says there are four objects of the queried color. Counting under `answer_revealing` is also uniform: all 8 raw outputs are the digit of the true count, which the alias rule marks correct. The revealing sentence contains the count word. That cell is the copying control.

## 6. Counterfactual grounding results

There are 12 pairs and 24 scenes. Within each pair the `neutral`, `helpful_nonrevealing`, and `contradictory` strings match, and the two true answers differ. The `answer_revealing` strings differ because each one names its own answer, so a change there is not a visual-edit comparison. Pair rates for "parsed answer changed" and "unchanged" use the pairs in which both arms produced a valid parse. The other rates use all 12 pairs. Scene rates use 24 scenes. DEVELOPMENT ONLY.

An answer change is a behavioral difference between the two arms. It is not treated as mechanistic visual grounding.

| Context | Texts identical | Both parsed | Parsed answer changed | Unchanged | Correct on both arms | Both arms match their own answers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `neutral` | 12/12 | 9/12 | 7/9 | 2/9 | 3/12 | 3/12 |
| `helpful_nonrevealing` | 12/12 | 8/12 | 5/8 | 3/8 | 0/12 | 0/12 |
| `contradictory` | 12/12 | 9/12 | 4/9 | 5/9 | 1/12 | 1/12 |
| `answer_revealing` | 0/12 | 8/12 | 8/8 | 0/8 | 6/12 | 6/12 |

Unchanged pairs under identical text, DEVELOPMENT ONLY:

- `neutral`: `cf-counting-00`, `cf-counting-03`
- `helpful_nonrevealing`: `cf-counting-00`, `cf-counting-01`, `cf-spatial-03`
- `contradictory`: `cf-attribute-00`, `cf-counting-00`, `cf-counting-01`, `cf-counting-02`, `cf-counting-03`

Scene-level companion, DEVELOPMENT ONLY: the pairs whose parsed answers changed cover 14/24 scenes under `neutral`, 10/24 under `helpful_nonrevealing`, 8/24 under `contradictory`, and 16/24 under `answer_revealing`. The last of those is the text-changing control.

No pair was dropped. The unchanged pairs are quality flags with `kept: true`.

## 7. Gradient diagnostics

Gradients were taken on the clean 512×384 image for `neutral` and `helpful_nonrevealing` only. Scoring, generation, and the gradient use the same float32 preprocess. No adversarial image was optimized. Raw tensors were not written to git.

On the gate scene the neutral backward took 4.216 seconds and the helpful backward took 4.424 seconds (DEVELOPMENT ONLY). Repeating each gradient changed no component: `repeat_max_abs_diff` is 0. The neutral L1 norm was 2665.861 and the helpful L1 norm was 3920.366. The L∞ norms were about 0.452 and 0.482. Nonzero fraction was 1. These magnitudes are a finiteness check. They are not an attack step size.

Across all 24 scenes, DEVELOPMENT ONLY. Norms in this table are rounded to three decimals. The unrounded values are in `summary.json`.

| Context | Finite | Zero L1 | Nonzero fraction, minimum | L1 minimum | L1 median | L1 maximum | Backward seconds, sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `neutral` | 24/24 | 0 | 1 | 2148.652 | 3894.879 | 5828.848 | 124.796 |
| `helpful_nonrevealing` | 24/24 | 0 | 1 | 1759.315 | 4256.393 | 15789.011 | 121.901 |

`A(s, t) = dot(g_t, sign(g_s)) / ||g_t||_1`. A zero target L1 norm would have been stored as null. A zero source L1 norm would have been excluded from the proposed H\* population even if the algebra returned 0. Neither case occurred.

| Direction | Defined | Undefined | Source excluded | Mean | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `A(neutral, helpful_nonrevealing)` | 24/24 | 0 | 0 | 0.566 | 0.375 | 0.731 |
| `A(helpful_nonrevealing, neutral)` | 24/24 | 0 | 0 | 0.553 | 0.381 | 0.721 |

The means are rounded to three decimals from `summary.json`. Per-scene values are in that file under `gradients.by_scene` and in `alignments.jsonl`.

The proposed H\* population is the scenes that are clean-correct under both primary contexts and have finite nonzero gradients in both. Every scene passed the gradient half of that rule. The clean-correct half leaves 3 scenes in 3 groups: `dev-attribute-03-a`, `dev-counting-00-b`, and `dev-counting-02-b` (DEVELOPMENT ONLY). `coefficient_fit` is false. `h_star_fitted` is false.

On the gate image, the float32 preprocess and the uint8 PIL preprocess have the same shape, `[1, 1, 3, 512, 512]`. The mean absolute gap was 0.000220 and the maximum was 0.113 (DEVELOPMENT ONLY; exact values are in `gate.json`). Generation and the loss use the float32 path together. The uint8 path is the comparison, and it was not used for the stored answers or gradients.

Fifteen rows are flagged `scoring_generation_disagreement`. Thirteen of them are counting rows whose raw output is a digit. The digit is canonicalized to a count word for the parse, while the loss ranks the four count words. Those two tokenizations are different, so a digit generation can be a valid parse of `one` while the lowest sum NLL among `{one, two, three, four}` is another word. The other two flags are color words, `Green` and `Blue`, whose canonical-word NLL is slightly worse than another color. Invalid rows are not also counted as this kind of disagreement.

## 8. Runtime and memory

All figures are MPS measurements on this machine. DEVELOPMENT ONLY.

| Stage | Seconds |
| --- | ---: |
| Weight load | 5.231 |
| Gate scene, including both contexts, vocabulary scores, and the repeated gradients | 23.325 |
| The other 23 scenes | 401.795 |
| Process wall clock, from the shell | 431.5 |

Memory snapshots from `vlm_iface.devices.memory_snapshot`. `peak_rss_mb` is the process peak from `ru_maxrss` and mixes the process with unified memory. It is not an MPS-only model footprint. `mps_driver_allocated_mb` is the driver cache. DEVELOPMENT ONLY.

| Snapshot | Current RSS MB | Peak RSS MB | MPS current allocated MB | MPS driver allocated MB | OS available MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before load | 205.6 | 205.6 | 0.0 | 0.5 | 8074.4 |
| After load | 88.4 | 399.4 | 1965.6 | 2856.5 | 3749.7 |
| After gate | 616.0 | 616.0 | 1965.6 | 2891.7 | 5487.5 |
| After the 24 scenes | 618.0 | 624.5 | 1965.6 | 2861.7 | 5304.5 |

OS available memory before the load was 8074.4 MB, so the machine was not empty. The current MPS allocation stayed at 1965.6 MB after the load, which matches the float32 weight resident set seen at Checkpoint 0.

## 9. Technical problems

The gate passed, and the 24-scene run exited 0. `python -m pilot.verify` reported `ok: true` and `n_mismatches: 0`.

PyTorch printed `Redirects are currently not supported in MacOs` while loading weights. The load completed and the parameter device was `mps`.

The scene-generator test `test_metadata_describes_the_sample` expects Pillow 11.3.0 because the generator records `PIL.__version__` and the package pins that version. The MPS virtualenv has Pillow 13.0.0. Under Pillow 13 the regenerated pixels still matched the committed PNGs; the metadata string was the mismatch. The scene suite was therefore run in `/Users/fedecava/Development/_c0-work/venv-scene` with Pillow 11.3.0, where 46 tests passed. The pilot read the committed PNGs with Pillow 13.0.0. It did not regenerate them.

No scene was dropped for a weak score, an invalid parse, an unchanged counterfactual, or a gradient flag. There were no non-finite gradients, no zero gradients, no undefined alignments, and no library-NLL mismatches above 10⁻³.

## 10. Scientific interpretation

These are development observations. They are not an estimate of H\*.

The neutral attribute cell, 6/8 correct, is the strongest visual cell in the sample. The same family under the helpful sentence falls to 1/8, and 7 of those 8 outputs are sentences such as "The large circle is red." The helpful string is task guidance. On this run the model often answered that guidance with a sentence instead of a bare color. D4 counts the sentence as invalid even when the sentence names the right color. The official accuracy includes that choice.

Counting behaves like a text-sensitive task on this sample. The revealing sentence, which states the true count, is followed by the correct digit on 8/8 scenes. The contradictory sentence, which states "four", is followed by `4` on 8/8 scenes. Neutral counting is 2/8. That pattern is what the copying control and the contradictory foil were built to show. It is not an attack result, and a high revealing score is not visual robustness.

Spatial accuracy is at or below 2/8 in every context, and the revealing control is 0/8. Several revealing spatial outputs name a color other than the color written in the sentence, or they repeat a sentence that names the wrong color. This family does not yet look like a task the model solves from either the picture or the revealing line.

The two alignment means, about 0.57 and 0.55, say that a one-step signed step from one primary context points somewhat with the other context's gradient on these clean images. The range across scenes is about 0.37 to 0.73. A first-order alignment is not evidence that a multi-step attack would transfer, and no attack was run. With only three scenes correct under both primary contexts, the proposed regression has three groups. That is a description of the inclusion count. It is not a fitted coefficient.

The generator still identifies the family by object count: 4 objects for attribute, 8 for counting, and 5 for spatial. That flag was already known from the scene design. The pilot did not remove it.

## 11. Limitations

The sample is 24 synthetic scenes in 12 dependent pairs. A rate with denominator 8 or 12 is a description of this generator, not a general claim about SmolVLM2.

D4's exact-match rule puts every sentence in the invalid bin. Ten of the twenty invalid sentences contain the true color as a whole word. A different parser would move those rows between invalid, valid-but-wrong, and correct. This report does not apply that different parser.

The scored string is the canonical word. Counting generations are often digits. Correctness treats the digit as an alias. The gradient and the sum NLL are the derivative and the loss of the word, not of the digit token the model emitted. The thirteen digit disagreements are that gap.

Sum NLL and mean token NLL coincide here because every canonical answer is one token. This pilot does not show how the two reductions differ when answers have different lengths.

The float32 path differs from uint8 PIL preprocessing. The gap on the gate image is small on average and reaches 0.113 in the worst component. Stored scores and generations share the float32 path.

`peak_rss_mb` is not the weight footprint. The driver allocation is a cache high-water. OS memory was partly occupied before the load.

No held-out scene was generated or scored. No second model was loaded. No attack, noise baseline, or confirmatory interval was computed.

## 12. Decisions requested before attacks

1. Keep D4's exact match, including the sentence rule, or revise it in writing before any later checkpoint. Recommendation: keep the stored labels as they are. If a sentence that contains one color word should become a valid parse, that revision has to be explicit, because it changes who is correct. Ten invalid sentences currently contain the true answer word. Adopting that reading from this pilot would spend a parser choice on development output.

2. Keep the loss on the canonical word when an attack is eventually authorized. Recommendation: yes. Digit aliases can remain a parse rule. The loss and the gradient should stay on the same string, which is the canonical word, and digit surfaces should stay visible in `raw_output`.

3. The confirmatory estimator is still required before any confirmatory evaluation. It is Agent A's regression of the transfer indicator on `A(s, t)`, with task-family indicators and clustering by `counterfactual_group_id`. It is not implemented. Recommendation: leave it unimplemented until a later checkpoint has a pre-declared population large enough to fit. Three development groups are a count, not a fit.

4. Clean accuracy is below one half for `neutral` overall (10/24), for `helpful_nonrevealing` (5/24), for `contradictory` (3/24), for every spatial cell, and for counting outside the revealing control. Recommendation: treat those cells as floors. Do not authorize an attack contrast on a cell whose development clean accuracy is below one half until the lead says otherwise. The revealing counting cell at 8/8 is a copying control, not a reason to attack that cell in the name of visual robustness.

5. Attacks remain unauthorized. This checkpoint does not choose ε, a step count, or a budget walk.

## 13. Recommended next checkpoint

Do not start an attack checkpoint from this report.

The useful next checkpoint is a decision on items 1 and 4 above, still on development scenes only, still without a sealed split, and still without PGD, FGSM, noise, or a fit of H\*. If the lead keeps the current parser and still wants an attack after reading the floors, that attack checkpoint has to be authorized separately. The budget walk in the Checkpoint 0 review packet is a candidate for that later authorization. It is not started here.

H\* should stay the provisional primary hypothesis until the lead revises it in writing. These tables are not a reason to replace it with a claim about multi-context PGD or about contradictory text.

## 14. Git branch, commit, and PR links

- Branch: `checkpoint-1/dev-pilot`
- Base: `main` at `65a8d8ffb2f22f3d284b64e2d481151474761c5e`
- Integration commit: `591581a6cdc3cdc55eb1a9c53585b9bfd5978f44`
- Source pull requests: [PR 1](https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/pull/1), [PR 2](https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/pull/2), [PR 3](https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/pull/3), [PR 4](https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/pull/4). [PR 5](https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/pull/5) was cited and not merged.
- Pilot tables and this report: the commit on this branch that adds `docs/checkpoint-1/`. The pull request to `main` records the URL. This checkpoint is not merged.

Compact files a reviewer can rebuild without the model:

- `docs/checkpoint-1/predictions.jsonl`
- `docs/checkpoint-1/gradient_stats.jsonl`
- `docs/checkpoint-1/alignments.jsonl`
- `docs/checkpoint-1/gate.json`
- `docs/checkpoint-1/run_meta.json`
- `docs/checkpoint-1/summary.json`

Verification command:

```bash
PYTHONPATH=src/scene_gen:src/vlm_iface:src/analysis:src/pilot \
  /Users/fedecava/Development/_c0-work/venv-vlm/bin/python -m pilot.verify
```

Test status before the MPS run: scene suite 46 passed in 2.22s under Pillow 11.3.0; `tests/vlm_iface` 41 passed, `tests/analysis` 41 passed, and `tests/pilot` 7 passed, together 89 passed in 4.86s. The in-run verifier then reported `n_mismatches: 0`.
