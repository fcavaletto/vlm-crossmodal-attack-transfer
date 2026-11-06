# Agent and responsibility

Agent B, benchmark workstream, Checkpoint 0. Branch `checkpoint-0/scene-generator`.

Owned paths: `src/scene_gen/**`, `tests/scene_gen/**`, `fixtures/development/**`, `docs/data/**`, `docs/checkpoint-0/AGENT_B_REPORT.md`.

This report covers the development scene generator only. It does not cover model loading, gradients, attacks, or the statistical analysis plan.

# Executive summary

The repository now has a deterministic generator and a 24-scene development sample. Each scene is a drawing on a 4 by 2 grid, with a scene graph, one question, and the four context keys from the checkpoint contract. The task families are attribute color, counting, and "immediately left." The 24 scenes form 12 pairs. A pair shares the question and the three non-revealing texts. Attribute and counting pairs change one object's color. Spatial pairs exchange two objects' cells.

`helpful_nonrevealing` does not contain the answer. That statement is a substring test under `normalize_for_leakage`, applied to the answer and its aliases, including the digits for count words. The same absence holds for the neutral sentence, the contradictory sentence, and the question. Only `answer_revealing` is marked as containing the true answer. Shortcut checks and ambiguity checks are tests, including tests that feed broken scenes to the flagger. The generator refuses every split other than `development` and does not create a directory when it refuses.

Forty-four tests passed. No model ran, no attack ran, and no held-out scene was generated. CLEVR was not downloaded. A sealed-set procedure is written in `docs/data/DATASET_DESIGN.md` and is not implemented. Twenty-four is the size at which three families and pairs of two balance on four colors. The later targets of 128 and 256 scenes do not divide evenly that way. Those counts are an open choice, not a result.

# Work completed

- Package `scene_gen` under `src/scene_gen`, Python 3.11+, dependency pinned to Pillow 11.3.0. No PyTorch.
- Explicit design table, not a random draw: `src/scene_gen/scene_gen/design.py`.
- Manifest, 24 PNGs, contact sheet, metadata, scene-graph hashes, and SHA-256 checksums in `fixtures/development/`.
- Tests in `tests/scene_gen/` for leakage, shortcuts, pair edits, ambiguity flags on broken inputs, balance, the manifest contract, pixel regeneration, determinism, and the sealed-split guard.
- An independent answer reader in `tests/scene_gen/graph_oracle.py` that does not import the package oracle, plus a hand-typed answer table.
- Dataset note `docs/data/DATASET_DESIGN.md`, including the sealed procedure and a CLEVR note that does not treat a download as done.

Study 1 was read and not modified. Patterns reused, with the reason: MD5 seeds, line-oriented manifests, checksums, an overwrite guard, original Pillow drawings, and a contact sheet. Patterns not reused: one central object, supportive text that repeats the label, a test split written in the same loop, speckle, and small rotations.

# Key findings

Evidence from this checkpoint:

- The sample contains 24 scenes, 8 in each family, 12 counterfactual groups, and `split` is `development` on every row. Answer counts are 2 scenes per color for attribute and spatial, and 4 scenes each for the count words `two` and `three`.
- On all 24 scenes, normalized `helpful_nonrevealing` text does not contain the answer or any alias. The revealing context does contain the canonical answer, and it is the only context with `reveals_answer: true`.
- The hand-typed answer table, the test reader, and the manifest agree on all 24 scene ids.
- Emitted scenes have empty `ambiguity_flags`. The flagger returns specific flags for a wrong answer, a duplicate referent, a shared cell, a bad pixel center, a missing left neighbor, an extra object on the left ray, a zero count, an unknown shape, an unparsed question, an empty scene, a duplicate id, and a cell outside the grid.
- Pair images differ by at least 378 pixels and at most 17298 pixels (8.8% of the 512 by 384 canvas). The smallest change is recoloring one small triangle (`cf-counting-01`). The largest is a spatial swap of larger shapes (`cf-spatial-03`).
- `development_seed("dev-attribute-00-a")` is 1303635806. The MD5 of the domain string, a NUL, and that scene id is `7e6e5a265ef7ba216cdbfbb784aa362f`, matching `md5sum`.
- Requesting split `sealed` through the command line exits 2 and creates no output directory.

Proposals, not findings:

- The sealed procedure in the dataset note. It was not run.
- Revising the headline 128 and 256 scene totals so each family has an even count. See decisions below.
- Joining a context and a question with a single newline and no extra label words. The leakage tests cover that join. The manifest does not store a combined prompt.

Study 1's protocol reports a development clean-accuracy floor of about 9.4% under misleading color. That number is theirs. This checkpoint did not measure clean accuracy.

# Evidence

Environment for every command below: Linux, CPython 3.12.3, Pillow 11.3.0, pytest 9.2.0. This is not an Apple MPS run. No model weights were loaded.

Generator:

```text
PYTHONPATH=src/scene_gen python3 -m scene_gen --output fixtures/development
```

The written metadata reports `scene_count` 24, `split` `development`, `pillow` `11.3.0`, and `seed_domain` `study2-development-v1`.

Tests:

```text
PYTHONPATH=src/scene_gen python3 -m pytest tests/scene_gen -q
............................................                             [100%]
44 passed in 1.54s
```

Pixel differences were counted on the committed PNGs with Pillow, pair by pair. Minimum 378 pixels (`cf-counting-01`, fraction 0.00192). Maximum 17298 pixels (`cf-spatial-03`, fraction 0.08798). The test locks a floor of 200 changed pixels and a ceiling of 25% of the canvas.

Seed check, outside the package, on the payload `study2-development-v1`, NUL, `dev-attribute-00-a`:

```text
printf 'study2-development-v1\0dev-attribute-00-a' | md5sum
7e6e5a265ef7ba216cdbfbb784aa362f  -
```

Files a later loader needs:

- `fixtures/development/manifest.jsonl`
- `fixtures/development/images/*.png` (24 files)
- `fixtures/development/contact_sheet.png` (review only; not a scene)
- `fixtures/development/metadata.json`
- `fixtures/development/checksums.json`
- `fixtures/development/scene_graph_hashes.json`

`image_path` is relative to the directory that contains the manifest.

# Design decisions and rationale

The scene graph is an explicit table. Seeds identify rows. They do not place objects. A sampled layout would have made balance and the shortcut tests harder to audit, and a rejection sampler can quietly drop the cells that fail, which changes the sample.

Counting scenes use eight objects and the color-count multiset `{2, 3, 3}` on both arms. With four objects, a true count of three is always the unique majority, so the question's color word would be unnecessary. The eight-object pattern keeps the histogram shape fixed while the queried color moves between 2 and 3. That is a proposal about shortcut control. It is not a claim that eight is the right number of objects for a model.

Helpful text is constant inside a family. Contradictory text is constant inside a pair and names a foil that is wrong for both arms. Otherwise the text would change with the answer and the pair would not be matched. `answer_revealing` is allowed to change. It is the positive control Study 1's supportive sentence was doing without the label.

Spatial anchors are only in columns 2 and 3 so that column 0 can hold a leftmost decoy on the other row. The tests check that the leftmost color is not the answer. The narrow column support is a limitation, recorded below.

No speckle and no rotation. Study 1 used both on a single object. Dots are extra countable marks, and a rotated square is easier to confuse with a diamond.

Category means shape. Material is omitted so the sample does not add a factor no question asks about.

The contact sheet shows pair 00 of each family, arm A on the top row and arm B on the bottom. It is a review figure.

# Known limitations

- Object counts are 4, 8, and 5 by family. The picture identifies the task family even before the question is read. The question also identifies it. A test locks those three numbers.
- Spatial anchors occupy only columns 2 and 3. Both of those columns occur, and both rows occur. Columns 0 and 1 are never the anchor.
- Attribute and spatial questions never target a small square or a small triangle.
- True count answers are only `two` and `three`. The contradictory word is `four`.
- Counting scenes fill every cell. Whether a model can count them cleanly is unknown. Clean accuracy was not measured.
- Constant counting statistics (maximum color count 3, second 3, minimum 2) score 4 out of 8 if someone always emits one of those numbers. That is the base rate of a balanced two-answer task. The tests show those statistics are not equal to the answer vector.
- PNG bytes were checksummed under Pillow 11.3.0 on Linux. Another Pillow build may encode the same pixels differently. Compare pixels before replacing the files.
- The drawings are not photographs and not CLEVR renders. A later result on this sample is a result on this generator.
- These 24 scenes are dependent inside 12 pairs. They are not 24 independent draws.
- The helpful-text check is a substring test. It does not prove that no paraphrase leaks the answer. It does prove the answer strings we listed are absent.

# Blockers

Blocking: none for this workstream's Checkpoint 0 deliverable. The sample, the tests, and the sealed procedure are in the branch.

Non-blocking:

- Agent C needs the image-path rule above. If the lead wants paths relative to `fixtures/` instead of `fixtures/development/`, that is a one-line change before any model run.
- The 128 / 256 totals should be settled before anyone generates more scenes. They do not block review of these 24.
- Clean-accuracy floors cannot be measured here. That measurement needs a model run, which this checkpoint forbids for this workstream.

# Decisions requested from the research lead

1. Keep the development sample at 24 until a written expansion, or add a second real layout now, up to 48? Recommendation: keep 24. A reflected copy would double-count one layout. Tradeoff: 24 is a small picture set for later smoke tests; 48 independent layouts need a new table and a new balance check. Work continues either way until a model is run.
2. Confirm `image_path` relative to `fixtures/development/`. Recommendation: yes, so the folder can move as a unit. Tradeoff: a loader that joins paths onto `fixtures/` will miss the files. The other agents can follow the rule in the dataset note.
3. Keep contradictory text identical on both arms? Recommendation: yes. Tradeoff: an arm-specific foil can push harder toward a particular wrong answer, and it also gives the text a way to tell the arms apart.
4. Keep count answers at `two` and `three` in this sample? Recommendation: yes for Checkpoint 0. Tradeoff: the model never has to say `one` or `four` as a true answer here. Widening the support belongs in the expansion, with the histogram shortcut rechecked.
5. Add "right" or "above" now? Recommendation: no. Tradeoff: one relation makes the decoy rule testable; it also limits the spatial claim to image-left on this grid.
6. Treat shape as `category` and skip material? Recommendation: yes until a question asks about material. Tradeoff: fewer factors, less like CLEVR.
7. CLEVR as a later external set only? Recommendation: yes, after a license reading of the upstream archive and a mapping onto these four contexts. Tradeoff: CLEVR is a known diagnostic set; it is also a different generator, and it was not downloaded.
8. Replace 128 development and 256 sealed with pair-friendly totals? Recommendation: 132 development (44 per family) and 252 sealed (84 per family). Tradeoff: the headline counts in the checkpoint note are not divisible by three, and an odd per-family count cannot form pairs. Keeping 128 and 256 is fine if the unequal families are written down on purpose. Do not generate either total in this checkpoint.

# Reproduction instructions

From the repository root. The run below is the one that produced the quoted pytest output. Python 3.12.3, Pillow 11.3.0, pytest 9.2.0, Linux. Resolver pins recorded from that environment: `pillow==11.3.0`, `pytest==9.2.0`, plus pytest's dependencies `iniconfig==2.3.1`, `pluggy==1.7.0`, `Pygments==2.21.0`.

```bash
PYTHONPATH=src/scene_gen python3 -m pytest tests/scene_gen -q
```

Regenerate the committed sample only on purpose. `--force` overwrites it. Without `--force`, an existing manifest exits 1.

```bash
PYTHONPATH=src/scene_gen python3 -m scene_gen --output fixtures/development --force
```

Refusing a sealed split:

```bash
PYTHONPATH=src/scene_gen python3 -m scene_gen --output /tmp/should-not-exist --split sealed
```

Exit code 2. The output directory is not created.

An editable install is the packaged form of the same command. This checkpoint's tests were run with `PYTHONPATH`, not from an installed copy:

```bash
python3 -m pip install -e 'src/scene_gen[dev]'
```

Pillow is pinned to 11.3.0 in `src/scene_gen/pyproject.toml`. Do not replace the PNGs on another platform until a pixel compare against `fixtures/development` succeeds.

# Git information

- Branch: `checkpoint-0/scene-generator`
- Base: `main` at `65a8d8ffb2f22f3d284b64e2d481151474761c5e`
- PR: recorded in the commit that follows the generator commit, and in the pull request body
- Test status: 44 passed, command and verbatim output in Reproduction instructions and Evidence. That run is the tree with the generator, fixtures, tests, and this report. A commit that only inserts the PR URL does not change the suite.

The authoritative hash of the branch head is `git rev-parse HEAD` on `checkpoint-0/scene-generator` after the pull request is opened.
