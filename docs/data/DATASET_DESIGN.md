# Development scene design

Status: Checkpoint 0 proposal. This file specifies the development sample that the generator writes, and a procedure for a later sealed set. It is not a finding about vision-language models. No model was loaded. No attack was run. No held-out scene was generated or inspected.

The sample is 24 scenes, all with `split` equal to `"development"`. The shared checkpoint note names a later target of 128 development scenes and 256 sealed scenes. Those counts are not divisible by three task families and by pairs of two. That arithmetic is an open choice at the end of this note. This checkpoint stops at the balanced 24.

## What a scene is

Each line of `fixtures/development/manifest.jsonl` is one scene. The fixtures root for paths is the directory that contains the manifest, `fixtures/development/`. Resolve an image as that directory plus `image_path`. The paths look like `images/dev-attribute-00-a.png`.

A scene has a scene graph, one question, one canonical answer, aliases, and four context strings. The graph is the authority for the answer. The picture is a drawing of the graph. Nothing in the picture is an object unless it is in the graph.

Objects sit on a 4 by 2 grid. The canvas is 512 by 384 pixels. Cell size is 128 by 192. The center of column `col` and row `row` is

```text
cx = col * 128 + 64
cy = row * 192 + 96
```

`category` is the shape: `circle`, `square`, or `triangle`. `attributes` holds `color` and `size`. Colors are `red`, `blue`, `green`, `yellow`. Sizes are `small` (radius 22) and `large` (radius 46). There is no material factor. Positions are not jittered. The seed does not choose the layout.

The seed is still stored. It is

```text
MD5("study2-development-v1" + NUL + scene_id) mod 2147483647
```

For `dev-attribute-00-a` that value is `1303635806`. The MD5 hex digest of those bytes is `7e6e5a265ef7ba216cdbfbb784aa362f`, checked with `md5sum` as well as in the test suite. Python's built-in `hash()` is not used. Study 1 had the same reason to avoid it: process-salted hashes change across runs. The domain string is new. A later sealed generator must not reuse `study2-development-v1`.

Regenerating a scene means running this package at a known commit. The design table in `src/scene_gen/scene_gen/design.py` is the scene. The seed identifies the row. It does not sample a new graph.

## Questions and ground truth

Three families. The question text is the whole question. This package does not wrap it in a chat template.

**Attribute.** "What color is the {size} {shape}?" The answer is the color of the single object with that size and shape. If zero or two objects match, the scene is rejected.

**Counting.** "How many {color} objects are there?" The answer is `two` or `three`. Aliases are the digit `2` or `3`. The word is the canonical answer. A count other than one, two, three, or four is rejected. This sample never uses one as a true answer.

**Spatial.** "What color is the object immediately left of the {size} {shape}?" "Immediately left" means the unique object in the same row and the column one step toward the left edge. An object further left on that same row makes the scene ambiguous, and the generator will not emit it. Left is image-left, which is decreasing `col`. The grid is not mirrored.

A second reader lives in `tests/scene_gen/graph_oracle.py` and does not import the package oracle. The hand-written answer table in `tests/scene_gen/test_counterfactuals.py` is a third check. All three agree on the 24 scenes.

### One attribute pair

Question, both arms: "What color is the large circle?"

Arm A (`dev-attribute-00-a`) places a large red circle at cell (0, 0). A small red square is also in the scene, so red is not the only red object. Arm B changes that circle to blue and leaves the small red square and a large blue square in place. The question, the neutral sentence, the helpful sentence, and the contradictory sentence are the same on both arms. The contradictory sentence says the large circle is green. Only the revealing sentence names red on arm A and blue on arm B.

### One counting pair

Question, both arms: "How many red objects are there?"

Both arms fill all eight cells. Arm A has two red objects. Arm B recolors one non-red object to red, so the count is three. The multiset of color counts stays `{2, 3, 3}`. A reader who ignores the word "red" and reports the largest color-count gets 3 on every counting scene, but the answers alternate between 2 and 3, so that rule is wrong on half of them. The same is true of the second-largest count and of the smallest count. Those constants are checked in `tests/scene_gen/test_shortcuts.py`.

Four objects would have made a count of three the unique majority every time. That is why counting scenes use eight objects. The choice is a shortcut control, not a claim that eight is the right number for a model.

### One spatial pair

Question, both arms: "What color is the object immediately left of the large circle?"

The anchor is a large circle in the rightmost column. The object one column to its left is red on arm A. Arm B swaps that object with a blue object that sat in column 0 on the other row. After the swap the left neighbor is blue, and the red object is now the leftmost object in the picture. The leftmost color is therefore the wrong answer on both arms. Another object shares the neighbor's column, on the other row, and is neither red nor blue.

## Matched counterfactuals

A group id such as `cf-attribute-00` has two scenes. They share the task, the question, and the three non-revealing texts. They have different answers. The picture change is one edit:

- Attribute: one object's color changes. Positions and shapes stay.
- Counting: one object's color changes. The queried count moves by 1. Shape counts and size counts stay.
- Spatial: two objects exchange cells. Each object keeps its color, shape, and size.

Tests measure the edit on the object list, and they measure the pixels. Every pair in this sample changes at least a few hundred pixels (the smallest edit recolors one small triangle) and less than 25% of the canvas, so the second arm is not a one-pixel accident and not a new picture with the same caption.

There are 12 groups. Later analysis should not treat 24 scenes as 24 independent draws. The pair is the dependence unit. Agent D owns the estimator. This note only records the dependence.

## Contexts

Keys, and only these keys:

| Key | Role in this sample |
| --- | --- |
| `neutral` | Same sentence on every scene. |
| `helpful_nonrevealing` | States the rule for the task family. Same sentence for every scene in that family. |
| `contradictory` | States a false answer. Same sentence on both arms of a pair. |
| `answer_revealing` | States the true answer. Positive control. Differs across the two arms. |

`reveals_answer` is true only for `answer_revealing`. It means "contains the true answer," not "mentions some label." The contradictory sentence mentions a wrong label and is marked false.

Neutral text: "Several simple objects appear on a plain background."

Helpful text, attribute: "Identify the single object with the size and shape named in the question, then name its color."

Helpful text, counting: "Count objects of the color named in the question. Size and shape do not matter."

Helpful text, spatial: "Use the object named in the question as the anchor. Name the color of the adjacent object toward the left edge in the same row."

The word "single" is used so the sentence does not contain the substring "one".

### Normalization: `normalize_for_leakage`

This is the leakage test, not a model parser. Agent C owns parsing of generated answers.

1. Unicode casefold.
2. Replace every run of characters that are not ASCII letters or digits with one space.
3. Strip the ends.

The test is a raw substring test on that string. It is not a whole-word test. `one` is a substring of `none`. Contexts are written so they pass the raw test. Digits are aliases for count answers, so a digit in a non-revealing sentence would fail too.

For every scene the tests check:

- The helpful sentence, the neutral sentence, the contradictory sentence, and the question do not contain the answer or any alias, before and after normalization.
- Neutral and helpful sentences contain none of `red`, `blue`, `green`, `yellow`, `one`, `two`, `three`, `four`, and no digit.
- Joining helpful text, a newline, and the question still does not contain the answer. That join is a proposal for later prompts. This package does not store a combined prompt. Extra words such as "Context:" are not part of the tested string.
- Each distinct neutral, helpful, and contradictory string occurs with at least two different answers in the sample.
- Within a pair, the non-revealing bundle is identical and the answers differ, so those strings do not identify the arm.
- The revealing sentence contains the canonical answer, and it is the only context with `reveals_answer: true`.
- The contradictory sentence contains a stated label that is not the true answer (`four` for counts; a third color otherwise).

## Scene ambiguity

`ambiguity_flags` is empty on every emitted scene. The generator raises if it is not. The same function is called on broken records in `tests/scene_gen/test_ambiguity.py`. A checker that always returned an empty list would fail those tests.

Flags the tests force:

- `answer_disagrees_with_scene`
- `non_unique_referent`
- `shared_cell`
- `overlapping_boxes`
- `position_not_derived_from_cell`
- `missing_relation_target`
- `extra_object_on_relation_ray`
- `count_out_of_vocabulary`
- `unknown_attribute`
- `unparsed_question`
- `empty_scene`
- `duplicate_object_id`
- `object_outside_canvas`

Overlap uses a gap of at least 8 pixels between the axis-aligned boxes. The box is the square of half-side equal to the radius. Triangles are drawn inside that square, so the test is conservative for them. On this grid the gap between large boxes in neighboring cells is much larger than 8 pixels.

The center pixel of every object is the palette color of that object. Corners of the canvas are the flat background `(236, 236, 236)`. There is no speckle. Study 1 drew faint dots as nuisance texture on a one-object image. Those dots would be countable marks here, so they were left out.

Palette RGB triples are the ones Study 1 used for the same four color names. That is a display choice so the names refer to the same hues. It does not import Study 1's task.

## Balance of the 24

| Family | Scenes | Answers |
| --- | --- | --- |
| attribute | 8 | red, blue, green, yellow, two scenes each |
| counting | 8 | `two` and `three`, four scenes each; each queried color on two scenes |
| spatial | 8 | red, blue, green, yellow, two scenes each |

Attribute questions use four descriptors: large circle, large square, large triangle, small circle. Small square and small triangle are not targets in this sample. Spatial anchors use the same four descriptors. Anchors sit in columns 2 and 3 only, because column 0 is kept for a leftmost decoy on the other row. Both column 2 and column 3 occur, and both rows occur. Column 0 and column 1 are not anchors.

Object counts are constant inside a family: 4 attribute, 8 counting, 5 spatial. A glance at the picture identifies the family. The question text also identifies it. The cardinality cue is still a limitation, and a test locks the three numbers so this note cannot pretend they vary.

## Rendering and the contact sheet

Drawings are original Pillow polygons. No photograph, no downloaded asset, no CLEVR image. Shapes are upright. Squares are axis-aligned. Study 1 rotated shapes slightly; a rotated square is easier to confuse with a diamond, so this sample does not rotate.

`fixtures/development/contact_sheet.png` is a review figure, 800 by 464 pixels. Columns are attribute, counting, spatial. Rows are arm A and arm B of pair 00 in each family. It is not a scene and must not be passed to a model as an input. `metadata.json` records that.

Images are PNG, RGB, compression level 9, `optimize=False`. Checksums in `checksums.json` are SHA-256 of the committed bytes, written with Pillow 11.3.0. A different Pillow build can change PNG bytes without changing pixels. Compare pixels before replacing the files. `scene_graph_hashes.json` hashes the question, the answer, and the object list. It is an exclusion list for a future sealed generator. It is not a sealed set.

## Threats

### Answer leakage

Study 1's supportive context restated the true color or shape in the answer format. The Study 1 protocol (section 2.2 and section 6) treats that as answer copying, not as evidence of visual grounding. This sample splits that idea in two. `helpful_nonrevealing` may explain the rule and must not contain the answer. `answer_revealing` contains the answer and is marked as a control. A high score under `answer_revealing` is not evidence of visual robustness.

The substring tests above are the check. They do not prove a model cannot infer the answer from a paraphrase we failed to imagine. They do prove the answer string and its aliases are absent under the stated normalization.

### Textual shortcuts

A shortcut here means a non-revealing string, or a simple image statistic that ignores the question, that lines up with the answer on every scene.

The non-revealing texts are shared inside each pair while the answers differ, so they cannot separate the arms. Across the file, every distinct non-revealing string occurs with at least two answers.

Image statistics the counting tests reject as perfect predictors: total object count, each color count, each shape count, each size count, the maximum color count, the second-largest color count, and the minimum color count. Several of those are constant (maximum 3, second 3, minimum 2, four large and four small, three circles, three squares, two triangles). A constant rule scores 4 out of 8 on a balanced two-answer task. That is the base rate. It is not a scene-wise solution. The queried color's own count does match the answer. That is the task, and the question is what names the color.

Attribute tests reject two shortcuts: the target color is never the only object of that color, and no grid cell's color matches the answer in every attribute scene where the cell is filled.

Spatial tests reject reading the leftmost object, reading a unique singleton color, and reading "the only object in the neighbor's column." The anchor's column and row are not constant. They are still a narrow support (columns 2 and 3 only).

### Label imbalance

Inside this 24, answer labels are balanced as in the table above. The balance is an assertion in `tests/scene_gen/test_balance_and_manifest.py`, not a description written after the fact. The sample does not balance every descriptor. Small square and small triangle are never the queried object. Counts other than two and three are never the true answer. Those are coverage limits of the sample, not hidden imbalance inside the labels that do occur.

### Scene ambiguity

Ground truth is a function of the graph and the question template. Ties, extra objects on the left ray, overlapping boxes, and pixel centers that do not match the cell formula are flags. Emitted scenes have none. The negative tests construct the failures.

What the flags do not measure is whether a model finds the picture clear. Crowded counting scenes fill every cell. That may lower clean accuracy later. This checkpoint does not measure clean accuracy.

### Counterfactual dependence

Arms in a pair share a layout. Pixel differences are local. An analysis that bootstraps scenes as if they were independent will overstate precision. The 12 groups are the natural cluster. Expanding the set by reflecting these 24 scenes would add dependence without a new factor. That is why this sample has no reflection replicate.

### Development and test contamination

Study 1 wrote development and test images in one generator loop. Variants 0–3 of each color and shape were development, and variants 4–5 were test, in the same manifest. The test pictures existed on disk during development work.

This package has no test split. `refuse_non_development_split` raises `SealedSplitError` for any other name, including `sealed` and `test`, and the command-line entry point returns exit code 2 before creating a directory. The repository fixture tree contains only `fixtures/development/`. Development scene-graph hashes are published so a later sealed run can reject collisions. Publishing the hashes does not publish sealed scenes, because no sealed scenes exist.

The development domain string is `study2-development-v1`. The proposed sealed domain string is `study2-sealed-v1`. The sealed master seed is not in this repository.

### Clean-accuracy floors

Study 1's frozen protocol reports a severe development floor for misleading color, about 9.4% clean accuracy, and says that low accuracy under that text must not be described as amplification. This sample does not remeasure that number and does not claim a floor of its own.

The mechanism to watch is the same kind of mechanism. The contradictory sentence states a wrong answer in a short declarative line. A model that copies the line will be wrong. Raw accuracy under `contradictory` is then a text effect. It is not, by itself, a visual-robustness result. Whether the helpful sentence avoids the Study 1 copying pattern is a hypothesis about the text, checked here by the substring tests, not a measured accuracy. Clean accuracy, invalid outputs, and additional error belong to later evaluation. They are not computed in this package.

### Generalization beyond synthetic images

These are flat geometric drawings on a gray background. A result on them is a result on this generator. It does not extend to photographs, diagrams, or other synthetic engines unless a separate comparison says so. Empty cells, constant object counts per family, and the lack of occlusion are part of the generator, not properties of natural images.

### CLEVR

CLEVR is a candidate external set only. It was not downloaded, vendored, or rendered here.

What was read, from public pages, without the dataset files:

- The generator repository `facebookresearch/clevr-dataset-gen` is archived. Its code license file is a BSD license, copyright 2017 Facebook, Inc.
- That repository's image-generation README says the combined scene JSON stores a license field that defaults to CC-BY 4.0. A default field in a writer is not a legal opinion about the released image archive.
- A Kaggle mirror labels the images CC BY 4.0. A mirror's label is not the upstream grant.
- The rendering path uses Blender, camera and light jitter, and a minimum visible-pixel check. That is a different generator from this Pillow grid, with different failure modes.

Suitability is open even after the license is settled. CLEVR questions are programs over a 3D scene. They are not our three templates, and they do not come with the four context keys. Occlusion and paraphrase are useful stresses. They also break the "exact graph, no tie" rule unless someone rewrites the checks. Recommendation: keep CLEVR out of the primary sample. Revisit it only with a written license reading of the upstream archive and a mapping onto these context keys. Do not start that by downloading the archive inside this checkpoint.

## What was carried over from Study 1, and what was not

Carried over, because the reason still applies:

- Original deterministic drawings, no external images.
- A manifest of one JSON object per line, plus SHA-256 checksums.
- Refusal to overwrite an existing manifest unless `--force` is passed.
- A small contact sheet for visual review.
- MD5-based seeds so the value does not depend on `PYTHONHASHSEED`.

Not carried over:

- One large central object, and color or shape as the only questions. The questions in this study need several objects, a count, or a relation.
- Supportive text that repeats the true label. That role is now the revealing control, and it is labeled as such.
- A test split written in the same loop as the development images.
- Random speckle and small rotations. Both fight the ambiguity checks used here.
- Treating development and test as nearby variants of the same color and shape cell.

Study 1 remains read-only. None of its images or modules are copied into this repository.

## Sealed-set procedure

This procedure is not implemented. `scene_gen` has no sampler under the sealed domain string. Running one would violate this checkpoint. The steps below are what a later freeze should do, in order.

1. Freeze the generator commit, the three question templates, the four context templates, and `normalize_for_leakage` before any sealed seed exists.
2. Decide the development set that will actually be used for design choices. If that set grows past these 24 scenes, regenerate `fixtures/development/scene_graph_hashes.json` from the final development manifest before the sealed seed is drawn. Hashes of a partial development set will not exclude the scenes added later.
3. The research lead draws one master seed from a public randomness beacon and records the value, the source, and the retrieval time in a dated note. This file does not choose a value. No example sealed scene is computed.
4. A future program, not this one, builds sealed scenes as a pure function of (`study2-sealed-v1`, the master seed, a cell id). Cell ids are an explicit factorial: task family, pair index, arm, and whatever layout factors the lead added in step 2. The program rejects any scene whose canonical graph hash is in the development exclusion file. It also rejects a scene whose question, answer, and object list match a development scene under the canonical JSON (`sort_keys`, separators `(',', ':')`).
5. Write the sealed manifest and images outside git. Do not commit them. Do not open the images in analysis notebooks, contact sheets, or debugging samples. Store the SHA-256 of the sealed manifest with the research lead. Evaluation code should take the sealed directory only when the evaluation window is explicitly authorized, the way Study 1 required a separate held-out flag after its freeze.
6. Balance targets, once the counts below are chosen: equal scenes per family, equal answer colors for attribute and spatial, equal queried colors and equal count-words for counting, groups of two, and the same non-revealing-text rule as this sample. The minimal edits stay the three edits above unless a dated amendment adds one.

Proposed sizes, not generated:

- The checkpoint text says 128 development and 256 sealed.
- 128 is not divisible by 3, and 256 is not divisible by 3. Pairing needs an even count per family. 128 and 256 cannot be split into three equal pair-friendly family sizes.
- Equal families and pairs allow 126 development scenes (42 per family, 21 pairs) or 132 (44 per family, 22 pairs). They allow 252 sealed scenes (84 per family, 42 pairs) or 258 (86 per family, 43 pairs).
- Recommendation: 132 development and 252 sealed, with these 24 scenes kept as a reviewed core and the rest added only after the layout factors are written down. Alternative: keep 128 and 256 and accept unequal families. That alternative should be an explicit amendment, not a silent drop of one scene.

No directory named `sealed` or `test` is created by the commands in this checkpoint.

## How to regenerate

From the repository root, with Python 3.11 or newer:

```bash
python3 -m pip install -e 'src/scene_gen[dev]'
python3 -m scene_gen --output fixtures/development --force
python3 -m pytest tests/scene_gen -q
```

The install needs the pinned dependency `Pillow==11.3.0`. Tests also run without install if `PYTHONPATH` includes `src/scene_gen`, because `tests/scene_gen/conftest.py` inserts that path. Pillow still has to be importable. This checkpoint's fixtures were written with CPython 3.12.3 and Pillow 11.3.0 on Linux. That is not an Apple MPS measurement. No measurement in this package is a model measurement.

`--split` accepts only `development`. Anything else exits 2 and writes nothing. A second run without `--force` exits 1.

## Open choices

These are the choices the sample had to make so the files could exist. They are reversible until a model is run. They are not results.

1. Sample size is 24, the largest size at which four colors, three families, and pairs of two balance without a second layout factor. A reflection of these scenes would look like 48 scenes and would double-count one layout.
2. `image_path` is relative to `fixtures/development/`, the manifest directory. The other reading is "relative to `fixtures/`," which would prefix `development/`.
3. Contradictory text is identical on both arms and names one foil that is wrong for both. An arm-specific foil would be a text channel that differs with the answer.
4. Count answers in this sample are only `two` and `three`. The foil word is `four`.
5. The only spatial relation is "immediately left." "Right" and "above" need their own decoy rules. A 4 by 2 grid cannot put an anchor in column 0 without losing the leftmost decoy.
6. Category means shape. Material is absent.
7. Prompt assembly is not stored. The tested proposal is helpful text, a newline, then the question, with no added label words.
8. CLEVR stays out until a license reading and a mapping study exist.
9. Headline 128 / 256 counts should be revised so each family has an even number of scenes, or the unequal split should be written down on purpose.
