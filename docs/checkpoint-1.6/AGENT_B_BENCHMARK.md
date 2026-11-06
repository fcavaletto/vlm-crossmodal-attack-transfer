# Agent and responsibility

Agent B, benchmark workstream, Checkpoint 1.6. Branch `checkpoint-1.6/task-design`.

Owned paths: `src/scene_simple/**`, `tests/scene_simple/**`, `fixtures/development_simple/**`, `docs/checkpoint-1.6/AGENT_B_BENCHMARK.md`.

This report covers the development task and the scene generator only. It does not cover scoring, model selection, or the H\* estimator. No model was loaded. No attack was run. No held-out scene was generated. `docs/checkpoint-1/**`, `fixtures/development`, and `fixtures/development_v2` were not modified.

# Executive summary

The Checkpoint 1.5 development run missed its clean-competence band. On SmolVLM2-500M, attribute under helpful text was 0/24, counting under contradictory text was 0/24, and spatial was 6/24 and 12/24. Those numbers are the published record of that sample, in `docs/checkpoint-1.5/REVIEW_PACKET.md`. This checkpoint does not rescore them. No model was loaded. No attack was run. H\* was not fit.

The new sample is 24 scenes in 12 pairs, every row `split` `development`, one family named `row_color`. The question is "What color is the {shape} in the {top|bottom} row?" Four large shapes sit on a 4 by 2 grid. The named shape also sits in the other row in a different color, so the row is required. Another shape shares the named row in a different color, so the shape is required. Both rows hold two objects. Red, blue, green, and yellow each appear once on every scene. A pair swaps the target with a different shape in the other row, which changes the answer and keeps that inventory. Neutral text is the Checkpoint 1 sentence. Helpful text repeats the rule already in the question and names neither a row nor a color. Contradictory text names the other-row copy of the shape and is false on both arms. Layouts are written out and are not reflections of one another. Counting is not graded. These scenes have not been shown to a model.

# Work completed

- Read `docs/data/DATASET_V2_DESIGN.md`, `docs/checkpoint-1.5/AGENT_B_REPORT.md`, and the Checkpoint 1.5 review packet. The redesign avoids three recorded instrument failures: a family named by object count, an "immediately left" question whose column-mate disagreed with the graded neighbor, and helpful text that specified a different task from the question or elicited a sentence instead of a color.
- Package `scene_simple` under `src/scene_simple`. Python 3.11+, Pillow 11.3.0. No PyTorch. The design table is explicit. The seed does not place objects.
- Manifest, 24 PNGs, a 24-panel contact sheet, metadata, scene-graph hashes, and SHA-256 checksums in `fixtures/development_simple/`.
- Tests in `tests/scene_simple/` for leakage, the prompt join, pair edits, pixel locality, label balance, layout uniqueness, reflections, the sealed-split guard, refusal to write `fixtures/development` or `fixtures/development_v2`, ambiguity flags on broken inputs, and agreement of an independent reader with the manifest.
- An independent answer reader in `tests/scene_simple/graph_oracle.py`, which does not import the package, plus a hand-typed answer table.
- Confirmed `fixtures/development/manifest.jsonl` still has sha256 `3aea59c848aaafc76de6c70ebe77d452c5f89db7920b573161d30f9a23695453` and `fixtures/development_v2/manifest.jsonl` still has sha256 `b62ab1171a0aec7b2f264615a2961b6d384d00a1a4b17977d619ea27481154e1`. The new scene-graph hashes are disjoint from both older files. The 24 images are pixel-distinct from the 24 and from the 72.

# Key findings

Evidence below is read off the new scene graphs and the committed PNGs. It is not a model measurement.

- 24 scenes, 12 groups, `split` `development` on every row. The only task family is `row_color`. Counting is not a family. No question starts with "How many".
- Answers are red, blue, green, and yellow, six scenes each. Queried shapes are circle, square, and triangle, eight scenes each. Queried rows are top and bottom, twelve scenes each. Each target column, each shape-decoy column, and each row-decoy column occurs on three pairs. Six pairs stack the shape decoy in the target's column. Six do not.
- Every scene has four objects, two in each row, and one of each shape. The queried shape occurs twice, once per row, in two colors. The other object in the named row has a different color. Each of the four colors appears once. The two arms of a pair have the same color inventory.
- The pair edit swaps `o0` (the target) with `o3` (a different shape in the other row). Positions and shapes stay. On the PNGs the swap changes 14267 pixels (`cf3-row-00`, 7.26% of the 512 by 384 canvas) to 23630 pixels (`cf3-row-06`, 12.02%). Every changed pixel lies inside one of those two boxes expanded by 4 pixels.
- Rows 164 through 223 are the flat background `(236, 236, 236)` on all 24 images. Top-row shapes end near y=156 and bottom-row shapes start near y=228. The center pixel of every object is that object's palette color.
- Normalized neutral text, helpful text, contradictory text, and the question do not contain the answer. The same is true of each non-revealing string joined to the question with one newline. Only `answer_revealing` has `reveals_answer: true`. Helpful text does not contain "top", "bottom", "left", "large", "small", or a color word. It is not the Checkpoint 1 attribute sentence.
- The contradictory color is the color of the same shape in the other row. It differs from the answer on every scene. The independent reader, which does not take context text, matches the manifest on all 24 scene ids, as does the hand-typed table.
- Emitted scenes have empty `ambiguity_flags`. The flagger returns specific flags for a wrong answer, a duplicate referent, a missing shape decoy, a shape decoy in the answer color, a row decoy in the answer color, unequal row counts, a shared cell, a cramped row, a bad pixel center, an unknown shape, an unparsed question, an empty scene, a duplicate id, and a cell outside the grid.
- `development_seed("dev3-row-00-a")` is 769796342. The MD5 of the domain string, a NUL, and that scene id is `c8674e639c07ade72af52f90249e9f10`.
- Requesting split `sealed`, `test`, or `heldout` exits 2 and creates no directory. Writing to a path named `fixtures/development` or `fixtures/development_v2` exits 1 even with `--force`, and those manifests are unchanged.

The prompt join remains context text, a newline, then the question. The manifest does not store a combined prompt. The answer is not in the question.

# Evidence

Environment: macOS, CPython 3.12.14, Pillow 11.3.0, pytest 9.2.0. The interpreter was `/Users/fedecava/Development/_c0-work/venv-scene/bin/python`. This is not an Apple MPS run. No model weights were loaded.

Generator, from the repository root:

```text
PYTHONPATH=src/scene_simple python3 -m scene_simple --output fixtures/development_simple --force
```

The written metadata reports `scene_count` 24, `counterfactual_group_count` 12, `split` `development`, `pillow` `11.3.0`, `seed_domain` `study2-development-simple`, `task_families` `["row_color"]`, `model_was_run` false, and `contact_sheet_is_model_input` false.

Tests:

```text
PYTHONPATH=src/scene_simple:src/scene_gen python3 -m pytest tests/scene_simple -q
...................................................                      [100%]
51 passed in 1.70s
```

Seed check, outside the package, on the payload `study2-development-simple`, NUL, `dev3-row-00-a`:

```text
printf 'study2-development-simple\0dev3-row-00-a' | md5
c8674e639c07ade72af52f90249e9f10
```

Files a later loader needs:

- `fixtures/development_simple/manifest.jsonl`
- `fixtures/development_simple/images/*.png` (24 files)
- `fixtures/development_simple/contact_sheet.png` (review only; not a scene)
- `fixtures/development_simple/metadata.json`
- `fixtures/development_simple/checksums.json`
- `fixtures/development_simple/scene_graph_hashes.json`

`image_path` is relative to the directory that contains the manifest. The Checkpoint 1 loader still reads `fixtures/development/manifest.jsonl`. This package does not point it at the new files.

The contact sheet is 1588 by 252 pixels, 12 pair columns and 2 rows (arm A, then arm B). Captions such as `00a red` are for a human reader. Full-size review of `dev3-row-00-a`, `dev3-row-00-b`, `dev3-row-01-a`, `dev3-row-05-a`, and `dev3-row-10-b` matches the graphs: the named shape in the named row is the graded object, the same shape in the other row is a different color, and the blank band between the rows is visible.

Two existing assertions still require `fixtures/` to contain only `development` and `development_v2`. They were not edited. Pixel comparisons inside those tests passed; the directory list then failed because `development_simple` is present:

```text
tests/scene_v2/test_v2_render.py::test_committed_sample_matches_regeneration_and_checksums
tests/scene_gen/test_render_and_files.py::test_committed_sample_matches_regeneration_and_checksums
```

# Design decisions

H\* stays: image-gradient directional alignment between `neutral` and `helpful_nonrevealing` predicts cross-context adversarial error transfer. No coefficient is fit. Each paragraph below was specified before any model score on these scenes. There is no such score.

**One family, so object count cannot name one.** Development observation: on the Checkpoint 1 24, object counts were 4, 8, and 5 by family, and on the 72 a count of 4 or 5 never occurred in spatial scenes while a count of 8 never occurred in attribute scenes. Original decision: three families, with object count partly identifying the family. Revised decision: one graded family, `row_color`, 24 scenes, four objects on every scene. Justification: a picture that names the task is a bad instrument. With one family, a count has no family to select. Counting is not added back as a second family in order to vary the count. Estimand: H\* is unchanged. The scene distribution a later fit would use is not the Checkpoint 1 distribution and not the 72-scene distribution. Specified before any model score on these scenes.

**A named row, not "immediately left".** Development observation: the Checkpoint 1 spatial question did not say "same row", and a large column-mate disagreed with the nearest-left object. The 72 put "in the same row" into the question and emptied the column; spatial clean accuracy on that sample stayed at 6/24 and 12/24, and `answer_revealing` was 2/24. Original decision: grade the adjacent cell to the left of an anchor. Revised decision: grade the color of a named shape in the top row or the bottom row. The same shape is in the other row in the foil color. A different shape is in the named row in a third color. The neighbor's column is allowed to hold that other-row shape; the question does not ask who is left of whom. Justification: top and bottom are regions a person can see, separated here by a blank band, and the column-mate is the decoy that makes the row word necessary rather than a second reading of an underspecified relation. Estimand: a trial is the unique object with the named shape in the named row. H\* is unchanged. Specified before any model score on these scenes.

**Helpful text repeats the question's rule.** Development observation: the attribute helpful sentence "Identify the single object... then name its color" was followed by sentences on 22/24 scenes of the 72, which the strict parser marked invalid. The Checkpoint 1 spatial helpful sentence supplied "same row" when the question did not. Original decision: keep those helpful strings byte-identical across samples. Revised decision: one helpful sentence for this task, "Use the row named in the question and the shape named in the question. A matching shape in the other row is a different object." It does not name a row, a color, a size, or a count, and it does not say "Identify". Neutral text stays the Checkpoint 1 sentence. Justification: the oracle answers from the question and the objects alone, so helpful text cannot change the referent. It can still steer a reader away from the same-shape decoy. Contradictory text uses the foil color of that decoy, is shared by both arms, and is false for both, so copying it is not a path to a correct score. Estimand: the primary contexts remain `neutral` and `helpful_nonrevealing`, neither of which contains the answer. H\* is unchanged. Specified before any model score on these scenes.

**A two-object swap, not a one-object recolor.** Development observation: none on these scenes. The choice is about a shortcut that a one-object recolor would create, seen on the design table before any image was scored. Original decision, in the Checkpoint 1 attribute family: recolor the target only. Revised decision: swap the target's color with a different shape in the other row. Both arms then show each of the four colors once. Justification: recoloring the target from red to blue, while leaving red on another object, makes blue a singleton on that arm and makes "the color that appears twice" correct on the other arm. The swap changes the answer and keeps the inventory, so a color count cannot name the answer. The edit is two objects, 7.26% to 12.02% of the canvas, inside those objects' boxes. Estimand: the counterfactual still changes the graded color by a local pixel edit. H\* is unchanged. Specified before any model score on these scenes.

**What was left alone.** The canvas stays 512 by 384. The cell centers are the Checkpoint 1 centers, `col * 128 + 64` and `row * 192 + 96`. The radius is 60 rather than 46, so the four shapes are large and still leave the blank band. The palette triples are the Checkpoint 1 triples. CLEVR was not downloaded. "Left", "right", "above", and "below" are not questions. The scene graph is an explicit table. A rejection sampler is not in the package.

# Limitations

- There is one family. A later three-family fit has nothing here to put in a second or third family. Object count is 4 on every scene. It cannot name a family, because there is only one, and it also cannot demonstrate that several families share a count.
- Every object is large. Size is not in the question. The necessary decoy is the same shape, not a different size.
- Anchors use only the top row and the bottom row. On 6 of 12 pairs the same-shape decoy is directly above or below the target. The question names the row, so that stack is not a second graded relation. A reader who says "the upper circle" and a reader who says "the circle in the top row" agree on those pairs.
- Each color appears once. A count cannot select the answer. The answer color is a singleton, and so is every other color. Singleton status does not select the target either.
- The swap touches two objects. The largest edit is 12.02% of the canvas. That is inside the two boxes and under the 25% ceiling used for the older samples. It is larger than the older one-triangle recolors.
- Three pairs share the question "What color is the circle in the top row?", and other shape-row questions repeat as well. The shared question has more than one answer, which is the picture doing the work. Two bottom-triangle pairs also share the contradictory sentence "The triangle in the bottom row is red." That sentence is false for all four of those scenes.
- The helpful-text check is a substring test. It does not prove that no paraphrase leaks the answer.
- Whether a 500M model or a 2B model can answer these scenes is not measured. The layout is aimed at that competence. It is not evidence of it.
- PNG bytes were checksummed under Pillow 11.3.0. Another Pillow build may encode the same pixels differently.
- The drawings are not photographs. A later result on this sample is a result on this generator.
- The 24 scenes are dependent inside 12 pairs.
- `tests/scene_gen` and `tests/scene_v2` each contain one assertion that `fixtures/` has exactly two directory names. This checkpoint adds a third and does not edit those tests. The assertions fail.

# Blockers

Blocking: none for this workstream's Checkpoint 1.6 deliverable. The sample, the tests, and this note are on the branch.

Non-blocking:

- Clean accuracy on the 24 is unknown. Measuring it needs a model run, under a scoring rule that was not chosen by maximizing accuracy, labeled development only. This workstream does not select scenes by that score.
- The Checkpoint 1 loader still opens `fixtures/development`. A later development run that should use these 24 has to be pointed at `fixtures/development_simple` on purpose. This package does not retarget it.
- The two directory-list assertions named above fail until someone edits them. That edit was outside this checkpoint's owned paths.

# Decisions requested

1. Use `row_color` as the development task for a later clean-competence run, and keep `fixtures/development` and `fixtures/development_v2` as the Checkpoint 1 and Checkpoint 1.5 records? Recommendation: yes. Tradeoff: a later three-family study is not populated by this sample. The alternative is to keep authorizing work on the three families that missed the band.
2. Accept one family and a constant object count of 4? Recommendation: yes. Tradeoff: object count cannot be shown to fail as a family cue, because the cue has no second family to point at. Adding counting back in order to vary the count repeats the family that was 0/24 under contradictory text.
3. Accept the named-row question, including the six pairs where the decoy is stacked on the target? Recommendation: yes. Tradeoff: on those six, "the upper one" and "the top-row one" are the same object. The question does not say "left of". H\* does not change.
4. Accept the two-object color swap so that each color appears once on both arms? Recommendation: yes. Tradeoff: the edit is larger than a one-object recolor, and it is still inside two boxes. A one-object recolor makes a frequency shortcut on one arm.
5. Accept the new helpful sentence, and leave attacks, noise, and a fit of H\* unauthorized? Recommendation: yes. The 24 scenes have not been scored. They are not a reason to start an attack.

# Reproduction

From the repository root. Python 3.12.14, Pillow 11.3.0, pytest 9.2.0. The run below used `/Users/fedecava/Development/_c0-work/venv-scene/bin/python`. Resolver pins in `src/scene_simple/pyproject.toml`: `pillow==11.3.0`, `pytest==9.2.0`.

```bash
PYTHONPATH=src/scene_simple:src/scene_gen python3 -m pytest tests/scene_simple -q
```

Regenerate the sample only on purpose. `--force` overwrites `fixtures/development_simple`. It does not overwrite `fixtures/development` or `fixtures/development_v2`.

```bash
PYTHONPATH=src/scene_simple python3 -m scene_simple --output fixtures/development_simple --force
```

Refusing a sealed split:

```bash
PYTHONPATH=src/scene_simple python3 -m scene_simple --output /tmp/should-not-exist --split sealed
```

Exit code 2. The output directory is not created. `test` and `heldout` do the same.

Refusing the older directories, including with `--force`:

```bash
PYTHONPATH=src/scene_simple python3 -m scene_simple --output fixtures/development --force
PYTHONPATH=src/scene_simple python3 -m scene_simple --output fixtures/development_v2 --force
```

Exit code 1. Those directories are not modified.

# Git information

- Branch: `checkpoint-1.6/task-design`
- Start: `b2e7abff72b87b475d0abf8d037e1b4d2f9c0c06`
- No pull request was opened. The branch was not pushed. It was not merged.
- Generator, fixtures, tests, and the scientific text of this report: `ed24bfe33b65419e653e6cfeaf63b27499abc2a1`
- The child of that commit only records this hash. It does not change the generator, the fixtures, or the tests.
