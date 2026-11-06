# Agent and responsibility

Agent B, benchmark workstream, Checkpoint 1.7. Branch `checkpoint-1.7/scene-v4`.

Owned paths: `src/scene_v4/**`, `tests/scene_v4/**`, `fixtures/development_v4/**`, `docs/checkpoint-1.7/AGENT_B_BENCHMARK.md`.

This report covers the development scenes only. It does not cover scoring, gradients, or the H\* estimator. No model was loaded. No attack was run. No held-out scene was generated. `docs/checkpoint-1.6/**`, `fixtures/development`, `fixtures/development_v2`, and `fixtures/development_simple` were not modified.

The branch starts at `004783b5604c34aa066befbcd5356bfd7bd6e7b2` (pull request 12), not at `main`. A pull request into `main` therefore also shows the Checkpoint 1.5 and Checkpoint 1.6 commits. Those commits are the base. This workstream did not rewrite them.

# Executive summary

Checkpoint 1.7 is a prospective development protocol. It is not a rescoring of Checkpoint 1.6 and not a correction chosen to raise accuracy. No image in this sample has been shown to a model. No scene was kept or dropped because of a model output.

The sample is 96 scenes in 48 pairs. Every row has `split` `development`. The only family is `referent_color`. The question is "What color is the {large|small} {circle|square|triangle} in the {top|bottom} row?" The answer is one of red, blue, green, or yellow, read off the scene graph. Both arms use the same question and the same neutral, helpful, and contradictory sentences. The picture edit swaps the colors of the referent and one other object of a different shape. That swap changes the answer and leaves the count of every color unchanged.

Each scene fills a 4 by 2 grid with eight objects. Color counts are three, two, two, and one. The answer is one of the two colors that appear twice. The color that appears three times is never the answer. The color that appears once is the other-row copy of the referent, and it is never the answer. A reader who uses only the histogram, the object count, the shape counts, or the size counts gets the same guess on both arms, and the answers differ, so that guess is wrong on one arm of every pair. The natural frequency rules are wrong on every scene: the strict mode is wrong, and the unique singleton is wrong.

Layouts are a closed form of the pair index. They are not a rejection sample and they are not reflections of one another. The referent and its other-row copy never share a column.

# Work completed

- Read `docs/checkpoint-1.6/AGENT_B_BENCHMARK.md`, `docs/data/DATASET_V2_DESIGN.md`, and the scene packages those notes describe. The new sample keeps the Checkpoint 1 canvas, palette, and manifest keys. It does not reuse a scene graph from those samples.
- Package `scene_v4` under `src/scene_v4`. Python 3.11+, Pillow 11.3.0. No PyTorch. The design is a closed form in `design.py`. The seed does not place objects. `_validate` raises if a derived pair breaks a constraint. It does not draw a replacement.
- Manifest, 96 PNGs, a 96-panel contact sheet, metadata, scene-graph hashes, and SHA-256 checksums in `fixtures/development_v4/`.
- Tests in `tests/scene_v4/` for leakage, the prompt join, pair edits, pixel locality, label balance, layout uniqueness, reflections, the sealed-split guard, refusal to write the three older fixture directories, ambiguity flags on broken inputs, frequency and count shortcuts, and agreement of an independent reader with a hand-typed answer table.
- An independent answer reader in `tests/scene_v4/v4_graph_oracle.py`, which does not import the package.
- Confirmed the older manifests are unchanged: `fixtures/development/manifest.jsonl` sha256 `3aea59c848aaafc76de6c70ebe77d452c5f89db7920b573161d30f9a23695453`, `fixtures/development_v2/manifest.jsonl` sha256 `b62ab1171a0aec7b2f264615a2961b6d384d00a1a4b17977d619ea27481154e1`, `fixtures/development_simple/manifest.jsonl` sha256 `bf21cdcfeca8f5c9a05e9f53465402c48f4a3175bbd363a38ca36dbc443f2367`. The new scene-graph hashes are disjoint from all three. The 96 images are pixel-distinct from the 24, the 72, and the 24.

# Key findings

Evidence below is read off the new scene graphs and the committed PNGs. It is not a model measurement.

- 96 scenes, 48 groups, `split` `development` on every row. The only task family is `referent_color`. Counting is not a family. No question starts with "How many".
- Answers are red, blue, green, and yellow, 24 scenes each. Queried shapes are circle, square, and triangle, 32 scenes each. Queried sizes are large and small, 48 scenes each. Queried rows are top and bottom, 48 scenes each. There are 12 question strings. Each question occurs on 8 scenes and has each of the four colors twice.
- Each of the 12 ordered answer pairs `(arm A, arm B)` is used on 4 groups. Each unordered pair is used on 8 groups. Foil colors are 24 scenes each.
- Every scene has eight objects, four in each row, one per cell. Each row has exactly two large objects, in columns at least two apart, and two small objects. The referent and the other-row copy of its size and shape are in different columns on all 48 groups.
- Color counts on every scene are 3, 2, 2, and 1, and the two arms of a pair have the same count for every color. The answer color appears twice. The unique modal color appears three times and is not the answer. The unique singleton appears once, is the contradictory color, and is not the answer.
- The pair edit swaps `o0` (the referent) with `o4` (a different shape). Positions, shapes, and sizes stay. On the PNGs the swap changes 2745 pixels (`cf4-ref-06`, 1.40% of the 512 by 384 canvas) to 23630 pixels (`cf4-ref-25`, 12.02%). Every changed pixel lies inside one of those two bounding squares expanded by 4 pixels.
- Rows 164 through 223 are the flat background `(236, 236, 236)` on all 96 images. The center pixel of every object is that object's palette color.
- Normalized neutral text, helpful text, contradictory text, and the question do not contain the answer. The same is true of each non-revealing string joined to the question with one newline. Only `answer_revealing` has `reveals_answer: true`. Helpful text does not contain "top", "bottom", "left", "right", "large", "small", "count", "size", a shape name, or a color word.
- Dropping the size word, the shape word, or the row word leaves more than one object, and those objects are not all the answer color. The full size-shape-row triple is unique and is the answer.
- No fixed cell is the answer color on every scene. The highest cell is column 2, row 0, on 33 of 96 scenes. The leftmost object in the named row is the answer color on 32 of 96 scenes.
- The swap partner's shape is not balanced: triangle 42 scenes, circle 32, square 22. Its row is not balanced: top 54, bottom 42. Those two imbalances are recorded in `metadata.json`. They were not repaired by dropping pairs.
- Ten scenes omit one shape (four omit triangle, four omit square, two omit circle). The omitted shape is the same on both arms, so it cannot choose between the two answers. The queried shape is present twice on every scene.
- Emitted scenes have empty `ambiguity_flags`. The flagger returns specific flags for a wrong answer, a duplicate referent, a missing or extra cross-row copy, a cross-row copy in the answer color, a missing size decoy, a size decoy in the answer color, a missing row decoy, a row decoy in the answer color, unequal row counts, a shared cell, adjacent large objects, a modal color that is the referent, a singleton color that is the referent, a bad pixel center, an unknown shape, an unparsed question, an empty scene, a duplicate id, and a cell outside the grid.
- `development_seed("dev4-ref-00-a")` is 733909405. The MD5 of the domain string, a NUL, and that scene id is `2b9b4107f5f56ac5afa74e9497c0411b`.
- Requesting split `sealed`, `test`, or `heldout` exits 2 and creates no directory. An output directory named `heldout` does the same. Writing to a path named `fixtures/development`, `fixtures/development_v2`, or `fixtures/development_simple` exits 1 even with `--force`, and those manifests are unchanged.

The prompt join remains context text, a newline, then the question. The manifest does not store a combined prompt. The answer is not in the question.

# Evidence

Environment: Linux, CPython 3.12.3, Pillow 11.3.0, pytest 9.2.0. This is not an Apple MPS run. No model weights were loaded.

Generator, from the repository root:

```text
PYTHONPATH=src/scene_v4 python3 -m scene_v4 --output fixtures/development_v4 --force
```

The written metadata reports `scene_count` 96, `counterfactual_group_count` 48, `split` `development`, `pillow` `11.3.0`, `seed_domain` `study2-development-v4`, `task_families` `["referent_color"]`, `protocol_status` `prospective`, `model_was_run` false, `model_outputs_used_to_choose_scenes` false, `held_out_scenes_generated` false, and `contact_sheet_is_model_input` false.

Tests:

```text
PYTHONPATH=src/scene_v4:src/scene_gen python3 -m pytest tests/scene_v4 -q
..................................................................       [100%]
66 passed in 12.33s
```

`tests/scene_simple` was run as a check that the Checkpoint 1.6 sample still rebuilds. It was not modified. Result: 51 passed.

Seed check, outside the package, on the payload `study2-development-v4`, NUL, `dev4-ref-00-a`:

```text
printf 'study2-development-v4\0dev4-ref-00-a' | md5
2b9b4107f5f56ac5afa74e9497c0411b
```

Files a later loader needs:

- `fixtures/development_v4/manifest.jsonl`
- `fixtures/development_v4/images/*.png` (96 files)
- `fixtures/development_v4/contact_sheet.png` (review only; not a scene)
- `fixtures/development_v4/metadata.json`
- `fixtures/development_v4/checksums.json`
- `fixtures/development_v4/scene_graph_hashes.json`

`image_path` is relative to the directory that contains the manifest. The Checkpoint 1 loader still reads `fixtures/development/manifest.jsonl`. This package does not point it at the new files.

The contact sheet is 1588 by 996 pixels, 12 pair columns and 8 rows. Four bands cover pairs 00–11, 12–23, 24–35, and 36–47. Within a band the rows are arm A, then arm B. Captions such as `00a red` are for a human reader.

### One pair

Question, both arms: "What color is the small circle in the top row?"

`dev4-ref-00-a` places a small red circle at column 0, row 0. The same row also holds a large green circle and a small green square, so neither "small" nor "circle" nor "top" is enough on its own. A small yellow circle sits in the bottom row, column 1, so the rows are not a vertical stack. Green appears three times. Yellow appears once. Red and blue appear twice. The contradictory sentence, shared with arm B, says the small circle in the top row is yellow.

`dev4-ref-00-b` swaps that small circle with the large square at column 3, row 0. The referent is blue. The large square is red. Every color still has the same count. The question and the three non-revealing sentences are unchanged.

# Design decisions

H\* stays: image-gradient directional alignment between `neutral` and `helpful_nonrevealing` predicts cross-context adversarial error transfer. No coefficient is fit. Each paragraph below was specified before any model score on these scenes. There is no such score. The scenes were not chosen by maximizing accuracy.

**One family, eight objects, four colors.** Development fact, read off earlier generators and not off a score on this sample: object counts of 4, 8, and 5 named the Checkpoint 1 families before the question was read, and a one-object recolor can make the new color a singleton. Original decision in those generators: three families, or four objects with each color once. Revised decision for this protocol: one graded family, `referent_color`, 96 scenes, eight objects on every scene, color counts 3, 2, 2, 1. Justification: a picture that names the task is a bad instrument, and a color count that is flat cannot be a wrong answer because every color is tied. Here the strict mode is a distractor, the unique singleton is a distractor, and the total count is the same on every scene, so a count cannot point at one of the four labels. Counting is not added as a second family in order to vary the total. Estimand: H\* is unchanged. The scene distribution a later fit would use is not the Checkpoint 1 distribution, not the 72-scene distribution, and not the 24-scene row-color distribution. Specified before any model score on these scenes.

**Size, shape, and row, each doing work.** Development fact: a question that names only a shape, or only a relation such as "immediately left" without saying which object, can have two readings. Revised decision: the question names a size, a shape, and a row. Exactly one object has all three. Another object has that size and shape in the other row, in the foil color. Another object has that shape and row at the other size, in a color that is neither arm's answer. Another object has that size and row at another shape, in a color that is neither arm's answer. The other-row copy is never in the referent's column. The rows are separated by the empty band already used at radius 60. Justification: top and bottom are regions a person can see, and each word in the question rules out a drawn object of a different color. Estimand: a trial is the unique object with the named size, shape, and row. H\* is unchanged. Specified before any model score on these scenes.

**Helpful text repeats the question's rule.** Neutral text stays the Checkpoint 1 sentence, "Several simple objects appear on a plain background." Helpful text is "Use the object words in the question together. An object that matches only part of that description is a different object." It does not name a row, a size, a shape, a color, or a count, and it does not say "Identify". Contradictory text uses the singleton color of the other-row copy, is shared by both arms, and is false for both, so copying it is not a path to a correct score. Justification: the oracle answers from the question and the objects alone, so helpful text cannot change the referent. Estimand: the primary contexts remain `neutral` and `helpful_nonrevealing`, neither of which contains the answer. H\* is unchanged. Specified before any model score on these scenes.

**A two-object swap that keeps every color count.** The edit exchanges the referent's color with a different shape. Both arms then show the same count of each color, and the answer changes. Justification: a rule that sees only colors' frequencies, or only how many objects, shapes, or sizes are drawn, cannot be right on both arms. The stricter rules fail on every scene, not merely on one arm: the color that appears three times is not the answer, and the color that appears once is not the answer. The edit is two objects, 1.40% to 12.02% of the canvas, inside those objects' boxes. Estimand: the counterfactual still changes the graded color by a local pixel edit. H\* is unchanged. Specified before any model score on these scenes.

**A closed form, balanced where the crossing allows it.** Forty-eight is 12 descriptors times 4 columns. The descriptors are 2 sizes × 3 shapes × 2 rows. Color pairs are the three 4-cycles on the four color names, so each question has each color twice, and each ordered pair of distinct colors is the arm order of exactly four groups. The construction is the index, not a search that rejects and redraws, and not a sample weighted by a model. What the crossing does not balance is the swap partner: its shape counts are triangle 42, circle 32, square 22, and its row counts are top 54, bottom 42. Those counts are reported rather than repaired by discarding groups. Estimand: the dependence unit is the pair. H\* is unchanged. Specified before any model score on these scenes.

**What was left alone.** The canvas stays 512 by 384. Cell centers stay `col * 128 + 64` and `row * 192 + 96`. The large radius stays 60. The palette triples stay the Checkpoint 1 triples. The small radius is 28, which is new, so a small shape can sit beside a large one and still leave more than 8 pixels between the boxes. CLEVR was not downloaded. "Left", "right", "above", and "below" are not questions. No held-out image was written.

# Limitations

- There is one family. A later three-family fit has nothing here to put in a second or third family. Object count is 8 on every scene. It cannot name a family, and it also cannot show that several totals fail as cues, because the total does not vary.
- The answer colors are exactly the two colors that appear twice. A reader who already knows that rule still cannot choose which of the two without the image. A reader who picks the unique mode, or the unique singleton, is wrong on every scene. The tests check those rules. They do not check every paraphrase.
- Ten scenes omit one shape. Both arms omit the same shape. Green is not the answer on those ten scenes (red 3, blue 5, yellow 2). That is a property of 10 scenes, not a rule that separates a pair.
- The swap partner is a triangle on 42 scenes, a circle on 32, and a square on 22. It sits in the top row on 54 scenes and the bottom row on 42. The design does not balance those two margins.
- The swap touches two objects. The largest edit is 12.02% of the canvas. That is inside the two boxes and under the 25% ceiling used for the older samples.
- Every cell is occupied. An empty cell is not a factor. There is no middle row.
- The helpful-text check is a substring test. It does not prove that no paraphrase leaks the answer.
- Whether a 500M model or a 2B model can answer these scenes is not measured. The layout is a drawing with a blank band between the rows. That is not evidence of competence.
- PNG bytes were checksummed under Pillow 11.3.0. Another Pillow build may encode the same pixels differently.
- The drawings are not photographs. A later result on this sample is a result on this generator.
- The 96 scenes are dependent inside 48 pairs. This note does not say that 48 groups are enough to fit H\*.
- `tests/scene_gen/test_render_and_files.py` and `tests/scene_v2/test_v2_render.py` each assert that `fixtures/` contains only `development`, `development_simple`, and `development_v2`. This checkpoint adds `development_v4` and does not edit those tests. Pixel comparisons inside the two tests passed. The directory list then failed because `development_v4` is present.

# Blockers

Blocking: none for this workstream's Checkpoint 1.7 deliverable. The sample, the tests, and this note are on the branch.

Non-blocking:

- Clean accuracy on the 96 is unknown. Measuring it needs a model run, under a scoring rule that was not chosen by maximizing accuracy, labeled development only. This workstream does not select scenes by that score.
- The Checkpoint 1 loader still opens `fixtures/development`. A later development run that should use these 96 has to be pointed at `fixtures/development_v4` on purpose. This package does not retarget it.
- The two directory-list assertions named above fail until someone edits them. That edit is outside this workstream's owned paths.
- Whether 48 pairs can support a later estimate of H\* is not decided here. No coefficient was fit.

# Decisions requested

1. Use `referent_color` as the prospective development task, and keep `fixtures/development`, `fixtures/development_v2`, and `fixtures/development_simple` as the earlier records? Recommendation: yes. Tradeoff: a later three-family study is not populated by this sample.
2. Accept one family and a constant object count of 8, with within-scene counts 3, 2, 2, 1 so that the mode and the singleton are wrong answers? Recommendation: yes. Tradeoff: total object count cannot be shown to fail as a varying cue, because it does not vary.
3. Accept the size-shape-row question, including small objects at radius 28, and the choice that the other-row copy is never stacked on the referent? Recommendation: yes. Tradeoff: there is no stacked control in which "the upper one" and "the top-row one" are the same object.
4. Accept the two-object color swap, and accept the unbalanced swap-partner shape and row counts? Recommendation: yes. Tradeoff: those two margins are not uniform. Discarding groups to flatten them would also move the color balance.
5. Accept the new helpful sentence, and leave attacks, noise, held-out scenes, and a fit of H\* unauthorized? Recommendation: yes. The 96 scenes have not been scored. They are not a reason to start an attack.

# Reproduction

From the repository root. Python 3.12.3, Pillow 11.3.0, pytest 9.2.0, Linux. Resolver pins in `src/scene_v4/pyproject.toml`: `pillow==11.3.0`, `pytest==9.2.0`.

```bash
PYTHONPATH=src/scene_v4:src/scene_gen python3 -m pytest tests/scene_v4 -q
```

Regenerate the sample only on purpose. `--force` overwrites `fixtures/development_v4`. It does not overwrite `fixtures/development`, `fixtures/development_v2`, or `fixtures/development_simple`.

```bash
PYTHONPATH=src/scene_v4 python3 -m scene_v4 --output fixtures/development_v4 --force
```

Refusing a sealed split:

```bash
PYTHONPATH=src/scene_v4 python3 -m scene_v4 --output /tmp/should-not-exist --split sealed
```

Exit code 2. The output directory is not created. `test` and `heldout` do the same.

Refusing the older directories, including with `--force`:

```bash
PYTHONPATH=src/scene_v4 python3 -m scene_v4 --output fixtures/development --force
PYTHONPATH=src/scene_v4 python3 -m scene_v4 --output fixtures/development_v2 --force
PYTHONPATH=src/scene_v4 python3 -m scene_v4 --output fixtures/development_simple --force
```

Exit code 1. Those directories are not modified.

# Git information

- Branch: `checkpoint-1.7/scene-v4`
- Start: `004783b5604c34aa066befbcd5356bfd7bd6e7b2`
- Pull request: https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/pull/15
- The branch is not merged. The pull request targets `main` and includes earlier checkpoint commits because the branch does not start at `main`.
- Generator, fixtures, tests, and the scientific text of this report: `44aafcffc0f7d3ad8d65abe630ef35f7974aee7b`
- The child of that commit records this hash and the pull request URL. It does not change the generator, the fixtures, or the tests.
