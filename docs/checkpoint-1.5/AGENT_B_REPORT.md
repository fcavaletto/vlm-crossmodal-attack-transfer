# Agent and responsibility

Agent B, benchmark workstream, Checkpoint 1.5. Branch `checkpoint-1.5/scene-redesign`.

Owned paths: `src/scene_v2/**`, `tests/scene_v2/**`, `fixtures/development_v2/**`, `docs/data/DATASET_V2_DESIGN.md`, `docs/checkpoint-1.5/AGENT_B_REPORT.md`.

One line in `tests/scene_gen/test_render_and_files.py` now names both `fixtures/development` and `fixtures/development_v2`. The Checkpoint 1 sample is still the only content of `fixtures/development`. Any other fixture-directory name still fails that test.

This report covers the development scene generator only. It does not cover scoring, model selection, or the H\* estimator. No model was loaded. No attack was run. No held-out scene was generated.

# Executive summary

The Checkpoint 1 sample in `fixtures/development` is unchanged. This branch adds a separate development generator and 72 scenes in 36 pairs under `fixtures/development_v2`. No model was loaded, no attack was run, and no held-out image was generated.

The committed 24 scene graphs have three instrument problems. Object counts are 4, 8, and 5 by family, so the picture names the task. On 6 of 8 attribute scenes the queried shape occurs once, so the size word does no work. On all 8 spatial scenes a large object shares the neighbor's column. The nearest object to the left is the graded answer, at 128 pixels. That column-mate is 230.8 pixels away, a ratio of 1.80. The question was "immediately left." The row constraint lived in the helpful sentence and in the oracle. The generator required the two readings to disagree.

The new sample uses 12 pairs in each family. That 12 is 4 colors times 3 count transitions, and it is also 6 size-shape descriptors times 2. Counting keeps 1↔2, 2↔3, and 3↔4, including crowded scenes with true counts of three and four. Layouts are written out. They are not reflections or recolors of one template. Spatial questions include "in the same row," the neighbor's column is otherwise empty, and the next object to the left is at least 2.24 times as far. Neutral and helpful sentences are the Checkpoint 1 strings. H\* is the same sentence and is not fit. These 72 scenes have not been scored.

# Work completed

- Read `docs/checkpoint-1/PILOT_REPORT.md`, `docs/data/DATASET_DESIGN.md`, the Checkpoint 1 scene graphs, and the 24 committed images, including the contact sheet and the spatial pairs.
- Measured, from cell centers on those graphs: spatial nearest-left distance 128 on 8/8 scenes, second-left distance 230.8, ratio 1.80, and a large column-mate on 8/8. Attribute shape was unique on `dev-attribute-00`, `dev-attribute-01`, and `dev-attribute-02` (6 scenes). Object counts were exactly {attribute: 4, counting: 8, spatial: 5}.
- Package `scene_v2` under `src/scene_v2`. Python 3.11+, Pillow pinned to 11.3.0. No PyTorch. The design table is explicit. The seed does not place objects.
- Manifest, 72 PNGs, a 72-panel contact sheet, metadata, scene-graph hashes, and SHA-256 checksums in `fixtures/development_v2/`.
- Tests in `tests/scene_v2/` for leakage, shortcuts, pair edits, pixel locality, ambiguity flags on broken inputs, balance, layout uniqueness, reflections, the manifest contract, regeneration, the sealed-split guard, and refusal to write `fixtures/development`.
- An independent answer reader in `tests/scene_v2/v2_graph_oracle.py`, plus a hand-typed answer table.
- Dataset note `docs/data/DATASET_V2_DESIGN.md`.
- Confirmed `fixtures/development/manifest.jsonl` still has sha256 `3aea59c848aaafc76de6c70ebe77d452c5f89db7920b573161d30f9a23695453`, the Checkpoint 1 value. The v2 scene-graph hashes are disjoint from the Checkpoint 1 hashes. The 72 images are pixel-distinct from the 24.

# Key findings

Evidence from the Checkpoint 1 graphs, recomputed in this workstream. These are not new model measurements.

- All 8 spatial scenes: the nearest object with a smaller x-coordinate has the graded color. Center distance 128. The second object is the other-row object in the neighbor's column, distance 230.8, ratio 1.80, and that object is large.
- Attribute scenes `dev-attribute-00-a` through `dev-attribute-02-b` have one object of the queried shape. `dev-attribute-03-a` and `dev-attribute-03-b` have two circles. Size-mates were 2 on all 8, so size alone was already insufficient. Shape alone was sufficient on 6 of 8.
- Object count identifies the family on the 24.

Evidence from the new sample:

- 72 scenes, 24 in each family, 36 groups, `split` `development` on every row. Attribute and spatial colors are 6 scenes each. Each of the six size-shape descriptors is the queried object on 4 scenes in each of those families. Counting queries each color on 6 scenes. Count words are `one` 4, `two` 8, `three` 8, `four` 4. Transitions are four pairs each of 1↔2, 2↔3, and 3↔4.
- Object counts 4, 5, 6, 7, and 8 each occur in two families. No family has a single object count. Four and five objects never occur in spatial scenes. Eight objects never occur in attribute scenes.
- On all 72 scenes, normalized non-revealing text does not contain the answer or any alias. Only `answer_revealing` has `reveals_answer: true`. Neutral text and the three helpful sentences match the Checkpoint 1 strings.
- The hand-typed answer table, the independent reader, and the manifest agree on all 72 scene ids.
- Emitted scenes have empty `ambiguity_flags`. The flagger returns specific flags for a wrong answer, a duplicate referent, a shared cell, a bad pixel center, a missing left neighbor, an extra object on the left ray, a column-mate, a left object inside twice the neighbor distance, a nearest-left disagreement, a count outside the vocabulary, an unknown shape, an unparsed question, an empty scene, a duplicate id, and a cell outside the grid.
- Pair images differ by 378 to 9709 pixels (0.19% to 4.9% of the 512 by 384 canvas). Every changed pixel lies in an edited object's box expanded by 4 pixels. The smallest edit is `cf2-attribute-10`. The largest is `cf2-spatial-06`.
- Spatial neighbor distance is 128 on all 24 spatial scenes. The next left-half object is at least 286.2 pixels away. The minimum ratio is 2.236 (`dev2-spatial-00-a`). The maximum is 2.828 (`dev2-spatial-09-b`).
- `development_seed("dev2-attribute-00-a")` is 1739228704. The MD5 of the domain string, a NUL, and that scene id is `7eba8629d4011cc04d49cda8073e4679`, matching `md5sum`.
- Requesting split `sealed` exits 2 and creates no directory. Writing to a path named `fixtures/development` exits 1 even with `--force`.

Proposals, not findings: the sealed procedure in the dataset note. It was not run. Joining a context and a question with one newline remains the leakage join from Checkpoint 1. The manifest does not store a combined prompt.

# Evidence

Environment: Linux, CPython 3.12.3, Pillow 11.3.0, pytest 9.2.0. This is not an Apple MPS run. No model weights were loaded.

Generator, from the repository root:

```text
PYTHONPATH=src/scene_v2 python3 -m scene_v2 --output fixtures/development_v2 --force
```

The written metadata reports `scene_count` 72, `split` `development`, `pillow` `11.3.0`, and `seed_domain` `study2-development-v2`.

Tests:

```text
PYTHONPATH=src/scene_v2:src/scene_gen python3 -m pytest tests/scene_v2 -q
......................................................                   [100%]
54 passed in 6.82s
```

```text
PYTHONPATH=src/scene_gen python3 -m pytest tests/scene_gen -q
..............................................                           [100%]
46 passed in 1.49s
```

The v1 suite is included because one assertion now lists both development directories. It still rebuilds the 24 Checkpoint 1 images and compares pixels.

Seed check, outside the package, on the payload `study2-development-v2`, NUL, `dev2-attribute-00-a`:

```text
printf 'study2-development-v2\0dev2-attribute-00-a' | md5sum
7eba8629d4011cc04d49cda8073e4679  -
```

Files a later loader needs:

- `fixtures/development_v2/manifest.jsonl`
- `fixtures/development_v2/images/*.png` (72 files)
- `fixtures/development_v2/contact_sheet.png` (review only; not a scene)
- `fixtures/development_v2/metadata.json`
- `fixtures/development_v2/checksums.json`
- `fixtures/development_v2/scene_graph_hashes.json`

`image_path` is relative to the directory that contains the manifest. The Checkpoint 1 loader still reads `fixtures/development/manifest.jsonl`. This package does not point it at the new files.

The contact sheet is 1588 by 748 pixels, 12 pair columns and 6 rows (attribute A, attribute B, counting A, counting B, spatial A, spatial B). Full-size review of `dev2-spatial-00-a`, `dev2-spatial-00-b`, `dev2-spatial-01-a`, `dev2-counting-07-a`, and `dev2-attribute-04-a` matches the graphs: the same-row neighbor is the near object, and the far decoy is visibly farther.

# Design decisions

H\* stays: image-gradient directional alignment between `neutral` and `helpful_nonrevealing` predicts cross-context adversarial error transfer. No coefficient is fit. The paragraphs below are proposals. Each one is written before any model result on these 72 scenes, which is the result it would later be judged against.

**Sample and family cue.** Development observation: object count identifies the family on the 24, and only three groups were clean-correct under both primary contexts. Original decision: 24 scenes, with 4, 8, and 5 objects. Revised decision: keep those 24 files, and add 72 scenes whose object counts 4 through 8 each occur in two families. Justification: a picture that names the task is a bad instrument for a later attack contrast, which is the Checkpoint 1 conclusion about this benchmark. Estimand: H\* is unchanged. The scene distribution a later fit would use is not the Checkpoint 1 distribution.

**Spatial wording and the column-mate.** Development observation: spatial accuracy was at or below 2/8 in every context, and 0/8 under `answer_revealing`. On the graphs, nearest-left agrees with the oracle and the large column-mate does not. Original decision: one relation, "immediately left," with a required column-mate of a different color. Revised decision: one relation, the question says "in the same row," the neighbor's column contains only the neighbor, and the next left-half object is at least twice as far. The leftmost object stays a wrong color. Justification: the column-mate made two readings of an underspecified question disagree, on purpose. The item is unambiguous when those readings agree. The revealing-cell failure is also a failure to copy the sentence. Geometry does not repair copying. Agent A owns the parser and the output format. Estimand: a spatial trial is the same-row neighbor the oracle already defined. The items are new. H\* is unchanged.

**Counting support.** Development observation: contradictory counting produced `four` on 8/8 scenes, and the contradictory sentence said four. Original decision: true answers only `two` and `three`, foil always `four`. Revised decision: true counts `one` through `four`, with four pairs of each adjacent transition and a foil that is wrong for both arms. The 3↔4 pairs keep 7 or 8 objects. Justification: one foil word cannot be separated from a model that emits that word. Dropping 2↔3 would balance the four count words and would also drop the Checkpoint 1 contrast because the score was low. That drop is rejected. Estimand: the counting label set is wider. H\* is unchanged. Word counts are 4, 8, 8, and 4 because a +1 edit cannot balance four words and also keep the middle transition. The arithmetic is in the dataset note.

**What was made stricter.** Every attribute scene has another object of the queried shape and another of the queried size. Checkpoint 1 attribute scenes often failed the first of those. That control can lower attribute accuracy relative to the 6/8 shape-unique scenes. It is a control against a shortcut, recorded before any new score.

**What was left alone.** Neutral text and the three helpful sentences are byte-identical to Checkpoint 1. The strict parser, the canonical-word loss, and the digit alias are untouched. "Right," "above," and "below" are not added. CLEVR is not downloaded. The canvas stays 512 by 384.

The scene graph remains an explicit table. A rejection sampler is not used. `build_scenes` raises if a required cell of the factor table is missing, if a layout repeats, or if a layout is a reflection of another.

# Limitations

- Four and five objects never occur in a spatial scene. Eight objects never occur in an attribute scene. Object count still rules a family out. It does not name the family by itself.
- Count words are 4, 8, 8, and 4. The imbalance is designed. It is still an imbalance.
- The foil `one` uses "There is one {color} object." The other foils use "There are {foil} {color} objects." Each of those strings occurs with at least two answers. They are not the same sentence.
- Spatial anchors occupy only columns 2 and 3. All three rows occur. Columns 0 and 1 are never the anchor, because the leftmost decoy needs a column further left than the neighbor, and a further-left object on the anchor's row would make the relation ambiguous.
- On a spatial scene the neighbor is the only object in its column, so the column reading and the row reading agree. That is the clarity criterion. A model can be right by using either reading.
- Crowded counting scenes may still be hard. Clean accuracy was not measured. This note does not claim the one-half floor.
- The helpful-text check is a substring test. It does not prove that no paraphrase leaks the answer.
- PNG bytes were checksummed under Pillow 11.3.0 on Linux. Another Pillow build may encode the same pixels differently.
- The drawings are not photographs. A later result on this sample is a result on this generator.
- The 72 scenes are dependent inside 36 pairs.
- Making shape necessary on attribute scenes can make that family harder than the Checkpoint 1 items where one circle was the large circle. That is accepted before any score.

# Blockers

Blocking: none for this workstream's Checkpoint 1.5 deliverable. The sample, the tests, and the design note are on the branch.

Non-blocking:

- Clean accuracy on the 72 is unknown. Measuring it needs a model run on MPS, under a scoring rule Agent A proposes, labeled development only. This workstream does not select scenes by that score.
- The Checkpoint 1 pilot loader still opens `fixtures/development`. A later development run that should use the 72 has to be pointed at `fixtures/development_v2` on purpose. This package does not retarget it.
- Whether 4/5-object scenes should be added to the spatial family, at the cost of a singleton color, is a lead decision. The current rule keeps every spatial color at count two or more.

# Decisions requested

1. Use the 72 as the development sample for later checkpoints, and keep the 24 as the Checkpoint 1 record? Recommendation: yes. Tradeoff: later clean-accuracy numbers will not be comparable, scene by scene, with the pilot tables. The alternative is to keep authorizing work on an instrument whose picture names the family.
2. Accept the spatial question "in the same row" and the empty neighbor column? Recommendation: yes. Tradeoff: the column reading now agrees with the task. The leftmost decoy remains. H\* does not change. No spatial accuracy on these items exists yet.
3. Accept count words at 4, 8, 8, and 4 so that 2↔3 stays? Recommendation: yes. Tradeoff: the four words are unequal. Balancing them drops the middle transition. Dropping 2↔3 because Checkpoint 1 counting was weak is the choice this sample refuses.
4. Leave neutral and helpful text unchanged until Agent A's scoring note? Recommendation: yes. Tradeoff: the attribute helpful sentence produced invalid sentences on the pilot. Rewriting it here would spend a prompt change on this sample before a parser decision.
5. Leave attacks, noise, and a fit of H\* unauthorized? Recommendation: yes. The 72 scenes are not a reason to start an attack. They have not been scored.

# Reproduction

From the repository root. Python 3.12.3, Pillow 11.3.0, pytest 9.2.0, Linux. Resolver pins: `pillow==11.3.0`, `pytest==9.2.0`.

```bash
PYTHONPATH=src/scene_v2:src/scene_gen python3 -m pytest tests/scene_v2 -q
PYTHONPATH=src/scene_gen python3 -m pytest tests/scene_gen -q
```

Regenerate the v2 sample only on purpose. `--force` overwrites `fixtures/development_v2`. It does not overwrite `fixtures/development`.

```bash
PYTHONPATH=src/scene_v2 python3 -m scene_v2 --output fixtures/development_v2 --force
```

Refusing a sealed split:

```bash
PYTHONPATH=src/scene_v2 python3 -m scene_v2 --output /tmp/should-not-exist --split sealed
```

Exit code 2. The output directory is not created.

Refusing the Checkpoint 1 directory, including with `--force`:

```bash
PYTHONPATH=src/scene_v2 python3 -m scene_v2 --output fixtures/development --force
```

Exit code 1. That directory is not modified.

# Git information

- Branch: `checkpoint-1.5/scene-redesign`
- Start: `90c6ffe83d61514da217a20f8dfc2f202afbbe69` (`checkpoint-1/dev-pilot`)
- Pull request base: `main` at `65a8d8ffb2f22f3d284b64e2d481151474761c5e`
- The diff against `main` contains Checkpoint 1 because this branch starts there. Checkpoint 1 was not merged to `main`. This branch does not merge either.
- Generator, fixtures, tests, and the scientific sections of this report: `0c190e856ed20d6934139b12ec20001e3c1488ed`
- Pull request: https://github.com/fcavaletto/vlm-crossmodal-attack-transfer/pull/10
- Tests recorded against `0c190e856ed20d6934139b12ec20001e3c1488ed`: `tests/scene_v2` 54 passed in 6.82s; `tests/scene_gen` 46 passed in 1.49s. Together, 100 passed in 8.69s.
- The branch head is the child of `0c190e856ed20d6934139b12ec20001e3c1488ed`. That child only records this URL. It does not change the generator, the fixtures, or the tests.
