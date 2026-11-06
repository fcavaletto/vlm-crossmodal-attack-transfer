# Development scene design, version 2

Status: Checkpoint 1.5 proposal. This file specifies a revised development sample. It is not a finding about vision-language models. No model was loaded. No attack was run. No held-out scene was generated or inspected. The sample does not replace `fixtures/development`.

Checkpoint 1 scored the 24 scenes in `fixtures/development`. Those files stay as they were at commit `90c6ffe83d61514da217a20f8dfc2f202afbbe69`. The pilot report in `docs/checkpoint-1/PILOT_REPORT.md` is the record of that run. This note does not rewrite those observations.

The new sample is 72 scenes, all with `split` equal to `"development"`. The generator is `scene_v2`. The fixtures root is `fixtures/development_v2/`.

## Why a new sample

Three facts about the Checkpoint 1 scenes are design facts, read off the committed scene graphs. They are not model scores.

Object counts are 4 for attribute, 8 for counting, and 5 for spatial. The picture identifies the family before the question is read. The pilot report already named this cue.

On 6 of the 8 attribute scenes the queried shape occurs once (`dev-attribute-03-a` and `dev-attribute-03-b` are the exception). The size word is then unnecessary. The color of the target is not unique, which the Checkpoint 1 tests did check. Uniqueness of shape was not checked.

On all 8 spatial scenes the nearest object with a smaller x-coordinate is the oracle's answer. Its center is 128 pixels from the anchor. The next object to the left is 230.8 pixels away, a ratio of 1.80, and it is large, and it sits in the same column as the neighbor on the other row. The question was "What color is the object immediately left of the {size} {shape}?" It did not say "same row." The helpful sentence did. The generator's own shortcut test required a second object in that column, of a different color, so "the other object in the column to the left" and "the adjacent object in the same row" disagreed on every spatial scene. A nearest-left reading agrees with the oracle. A same-column reading does not. The question does not choose between them.

Counting answers were only `two` and `three`. The contradictory word was `four` on every counting scene. The pilot then observed that contradictory counting produced the word `four` on 8/8 scenes. That is the copying control working on one foil, not a reason to delete counting. It is a reason to let the true count and the foil vary.

Clean accuracy on the 24 is not remeasured here. The pilot's floors stand as descriptions of that sample. This note does not claim the new scenes clear any floor. No image in this sample has been shown to a model.

## Revisions

Each item below is a proposal. It is written before any model result on these 72 scenes. H\* is unchanged: image-gradient directional alignment between `neutral` and `helpful_nonrevealing` predicts cross-context adversarial error transfer. Nothing in this file fits that regression. The scenes on which a later fit would be computed do change, so a later coefficient would not be a coefficient on the Checkpoint 1 24.

1. Development observation: the 24-scene picture identifies the family, the spatial question underspecifies the oracle, and counting uses one foil word. Original decision: 24 scenes, counts 4/8/5, question "immediately left" without "same row," count answers `two` and `three`. Revised decision: 72 scenes in `fixtures/development_v2`, with the controls in the rest of this note. Justification: the 24 are the wrong instrument for a later confirmatory attack, which is the Checkpoint 1 conclusion, and the defects above are properties of the generator rather than of the parser. Estimand: H\* is the same sentence. The scene distribution is not the same distribution. This proposal precedes any score on the new images.

2. Development observation: spatial accuracy was at or below 2/8 in every context, including 0/8 under `answer_revealing`. Original decision: keep one relation, "immediately left," and keep a column-mate of a different color. Revised decision: keep one relation, write "in the same row" into the question, and leave the neighbor's column otherwise empty, so the adjacent-cell reading, the same-row reading, and the nearest-left reading name the same object. The leftmost object remains a different color and is farther. Justification: the column-mate was a controlled disagreement between two readings of an underspecified question. Removing the disagreement makes the item unambiguous to a person. It does not delete the spatial family, and it does not delete the leftmost decoy. The 0/8 revealing cell is also a text-copying failure. Scene geometry does not repair that. Agent A owns scoring and output format. Estimand: a spatial trial is explicitly the same-row neighbor the oracle already scored. Items are not paired with the Checkpoint 1 items. H\* is unchanged. No new spatial accuracy is reported.

3. Development observation: contradictory counting copied `four`. Original decision: foil word `four` on every counting scene, true answers only `two` and `three`. Revised decision: true counts `one`, `two`, `three`, and `four`, with adjacent edits 1↔2, 2↔3, and 3↔4, each once per queried color. The foil is a count word that is wrong for both arms, so it is not always `four`. Justification: one foil word cannot be distinguished from a model that emits that word. Keeping 2↔3 keeps the Checkpoint 1 contrast instead of dropping it because the score was low. Estimand: the counting label set is wider. H\* is unchanged. The imbalance of the four words is stated under Label imbalance. No new counting accuracy is reported.

## What a scene is

Each line of `fixtures/development_v2/manifest.jsonl` is one scene. Resolve an image as the manifest directory plus `image_path`. The paths look like `images/dev2-attribute-00-a.png`. Scene ids use the prefix `dev2-` so they cannot be concatenated with the Checkpoint 1 ids by accident. Group ids use `cf2-`.

A scene has a scene graph, one question, one canonical answer, aliases, and four context strings. The graph is the authority for the answer. The picture is a drawing of the graph. Nothing in the picture is an object unless it is in the graph.

The record keys are the Checkpoint 1 keys: `scene_id`, `seed`, `split`, `task_family`, `counterfactual_group_id`, `question`, `answer`, `answer_aliases`, `contexts`, `objects`, `image_path`, `ambiguity_flags`.

Objects sit on a 4 by 3 grid. The canvas is still 512 by 384 pixels, the size at which Checkpoint 1 validated a full-resolution gradient on one development image. Cell size is 128 by 128. The center of column `col` and row `row` is

```text
cx = col * 128 + 64
cy = row * 128 + 64
```

`category` is the shape: `circle`, `square`, or `triangle`. `attributes` holds `color` and `size`. Colors are `red`, `blue`, `green`, `yellow`. Sizes are `small` (radius 22) and `large` (radius 46). There is no material factor. Positions are not jittered. The seed does not choose the layout.

The seed is

```text
MD5("study2-development-v2" + NUL + scene_id) mod 2147483647
```

For `dev2-attribute-00-a` that value is `1739228704`. The MD5 hex digest of those bytes is `7eba8629d4011cc04d49cda8073e4679`. Python's built-in `hash()` is not used. The domain string is not `study2-development-v1` and not a sealed domain.

Regenerating a scene means running this package at a known commit. The tables in `src/scene_v2/scene_v2/design.py` are the scenes. The seed identifies the row. It does not sample a new graph.

## Questions and ground truth

Three families. The question text is the whole question. This package does not wrap it in a chat template.

**Attribute.** "What color is the {size} {shape}?" The answer is the color of the single object with that size and shape. Every attribute scene also contains another object of that shape and another object of that size, so neither word is redundant. The answer color occurs at least twice, and the contradictory color is drawn.

**Counting.** "How many {color} objects are there?" The answer is `one`, `two`, `three`, or `four`. Aliases are the digits `1` through `4`. The word is the canonical answer. A count outside that set is rejected.

**Spatial.** "What color is the object immediately left of the {size} {shape} in the same row?" The neighbor is the unique object in the same row and the column one step toward the left edge. Left is image-left, decreasing `col`. The grid is not mirrored. The oracle returns no answer if any of the following is true: the anchor is not unique, the adjacent cell is not a single object, another object lies further left on that row, another object lies in the adjacent column, the nearest object with a smaller x-coordinate is not the neighbor, or another left-half object is closer than twice the neighbor distance. On the 72 scenes the neighbor is 128 pixels from the anchor and the next left-half object is at least 286.2 pixels away, a ratio of at least 2.236.

A second reader lives in `tests/scene_v2/v2_graph_oracle.py` and does not import the package oracle. The hand-written answer table in `tests/scene_v2/test_v2_counterfactuals.py` is a third check. All three agree on the 72 scenes.

### One attribute pair

Question, both arms: "What color is the large circle?"

Arm A (`dev2-attribute-00-a`) places a large red circle at cell (0, 0). A small blue circle, a large red square, and a small green triangle are also in the scene. The large circle is the only large circle. It is not the only circle, not the only large object, and not the only red object. Arm B changes that circle to blue. The small blue circle remains, so blue is not unique either. The contradictory color is green, which is the small triangle on both arms. The question, the neutral sentence, the helpful sentence, and the contradictory sentence are the same on both arms. The contradictory sentence says the large circle is green.

### One counting pair

Question, both arms: "How many red objects are there?"

`dev2-counting-00` has four objects. Arm A has one red object. Arm B recolors one blue object to red, so the count is two and the blue count falls by one. The sorted color-count multiset is unchanged. The contradictory sentence says there are four red objects, which is wrong for both arms. A reader who ignores the word "red" and reports the largest color-count is not correct on every counting scene. The tests check that list, and the lists for the smallest color-count, the second-largest, the total, each shape count, and each size count.

### One spatial pair

Question, both arms: "What color is the object immediately left of the large circle in the same row?"

On `dev2-spatial-00-a` the anchor is a large green circle in column 3, row 0. The object one column to its left, same row, is a small red square. A small blue triangle is the unique leftmost object, on the bottom row, and its color is not red. No other object shares the red square's column. Arm B exchanges the colors of that square and that triangle, so the neighbor is blue and the leftmost object is red. Positions, shapes, and sizes stay. The leftmost color is the wrong answer on both arms.

## Matched counterfactuals

A group id such as `cf2-attribute-00` has two scenes. They share the task, the question, and the three non-revealing texts. They have different answers. The picture change is a recoloring. Positions and shapes do not change.

- Attribute: the target object's color changes.
- Counting: one object's color changes. The queried count moves by 1. The sorted multiset of color counts stays the same.
- Spatial: the neighbor and the far partner exchange colors. Each object keeps its cell, shape, and size.

There are 36 groups. Later analysis should not treat 72 scenes as 72 independent draws. The pair is the dependence unit. Agent D owns the estimator. This note only records the dependence.

On the committed PNGs the smallest edit changes 378 pixels (`cf2-attribute-10`, one small triangle, 0.19% of the canvas) and the largest changes 9709 pixels (`cf2-spatial-06`, 4.9% of the canvas). Every changed pixel lies inside the edited objects' boxes expanded by 4 pixels. The test locks a floor of 200 changed pixels and a ceiling of 25% of the canvas.

## Why 72, and why these factors

72 = 3 families × 12 pairs × 2 arms. Twelve is the crossing that makes the families the same size:

- Counting: 4 queried colors × 3 adjacent transitions (1↔2, 2↔3, 3↔4).
- Attribute and spatial: 6 descriptors (2 sizes × 3 shapes) × 2.

Each unordered pair of colors is the answer pair of exactly two groups in attribute, and of exactly two groups in spatial, in opposite arm order. Each color is the answer on 6 attribute scenes and 6 spatial scenes. Each queried color is used on 6 counting scenes, once at each transition.

Forty-eight scenes would be 8 pairs per family. Eight is not divisible by 3 transitions and not divisible by 6 descriptors. Ninety-six scenes would be 16 pairs per family. Sixteen does not give each color one pair of each transition without dropping 2↔3 or unbalancing the words further. Seventy-two is the smallest equal-family size that keeps every adjacent count transition and every descriptor.

The layouts are written out in `design.py`. A group whose shape-and-position signature matches another group, or a horizontal reflection, a vertical reflection, or a half-turn of another group, is rejected. Reflecting the Checkpoint 1 24 is not how this sample was built, and the scene-graph hashes are disjoint from `fixtures/development/scene_graph_hashes.json`.

## Contexts

Keys, and only these keys:

| Key | Role in this sample |
| --- | --- |
| `neutral` | Same sentence on every scene. The Checkpoint 1 sentence. |
| `helpful_nonrevealing` | The Checkpoint 1 sentence for that family. |
| `contradictory` | States a false answer. Same sentence on both arms of a pair. |
| `answer_revealing` | States the true answer. Positive control. Differs across the two arms. |

`reveals_answer` is true only for `answer_revealing`.

Neutral text: "Several simple objects appear on a plain background."

Helpful text, attribute: "Identify the single object with the size and shape named in the question, then name its color."

Helpful text, counting: "Count objects of the color named in the question. Size and shape do not matter."

Helpful text, spatial: "Use the object named in the question as the anchor. Name the color of the adjacent object toward the left edge in the same row."

These four strings are byte-identical to the Checkpoint 1 generator. They were not rewritten because a model produced sentences, or because a cell was weak. A change to the helpful sentence is a prompt change. Agent A owns that decision.

Attribute contradictory text: "The {size} {shape} is {foil}."

Counting contradictory text is "There are {foil} {color} objects." when the foil is `two`, `three`, or `four`. When the foil is `one`, the sentence is "There is one {color} object." The plural frame is ungrammatical for `one`. The two frames are not the same string, and the tests require each string to occur with at least two answers anyway.

Spatial contradictory text: "The object immediately left of the {size} {shape} in the same row is {foil}."

The revealing sentences use the same frames with the true answer. A high score under `answer_revealing` is not evidence of visual robustness.

### Normalization: `normalize_for_leakage`

This is the leakage test, not a model parser. The function matches `scene_gen.normalize.normalize_for_leakage` on the strings the tests compare. The steps are the Checkpoint 1 steps.

1. Unicode casefold.
2. Replace every run of characters that are not ASCII letters or digits with one space.
3. Strip the ends.

The test is a raw substring test on that string. `one` is a substring of `none`. Contexts are written so they pass the raw test. Digits are aliases for count answers, so a digit in a non-revealing sentence would fail too.

For every scene the tests check:

- The helpful sentence, the neutral sentence, the contradictory sentence, and the question do not contain the answer or any alias, before and after normalization.
- Neutral and helpful sentences contain none of `red`, `blue`, `green`, `yellow`, `one`, `two`, `three`, `four`, and no digit.
- Joining helpful text, a newline, and the question still does not contain the answer.
- Each distinct neutral, helpful, and contradictory string occurs with at least two different answers.
- Within a pair, the non-revealing bundle is identical and the answers differ.
- The revealing sentence contains the canonical answer, and it is the only context with `reveals_answer: true`.
- The contradictory sentence contains a stated label that is not the true answer.

## Scene ambiguity

`ambiguity_flags` is empty on every emitted scene. The generator raises if it is not. The same function is called on broken records in `tests/scene_v2/test_v2_ambiguity.py`.

Flags the tests force:

- `answer_disagrees_with_scene`
- `non_unique_referent`
- `shared_cell`
- `overlapping_boxes`
- `position_not_derived_from_cell`
- `missing_relation_target`
- `extra_object_on_relation_ray`
- `column_mate_beside_relation`
- `close_left_competitor`
- `nearest_left_disagrees`
- `count_out_of_vocabulary`
- `unknown_attribute`
- `unparsed_question`
- `empty_scene`
- `duplicate_object_id`
- `object_outside_canvas`

`column_mate_beside_relation` is the Checkpoint 1 diagonal competitor. `close_left_competitor` fires when another left-half object is closer than twice the neighbor distance. `nearest_left_disagrees` fires when the nearest left-half object is not the adjacent cell. Emitted scenes have none of these.

Overlap uses a gap of at least 8 pixels between the axis-aligned boxes. The box is the square of half-side equal to the radius. On this grid the gap between large boxes in neighboring cells is 36 pixels.

The center pixel of every object is the palette color of that object. Corners of the canvas are the flat background `(236, 236, 236)`. There is no speckle and no rotation.

Palette RGB triples are the Checkpoint 1 triples for the same four color names.

## Balance of the 72

| Family | Scenes | Answers |
| --- | --- | --- |
| attribute | 24 | red, blue, green, yellow, six scenes each; each of the six size-shape descriptors on four scenes |
| counting | 24 | `one` 4, `two` 8, `three` 8, `four` 4; each queried color on six scenes |
| spatial | 24 | red, blue, green, yellow, six scenes each; each of the six descriptors on four scenes |

Object counts are not constant inside a family, and no object count belongs to only one family:

| Objects | attribute | counting | spatial |
| ---: | ---: | ---: | ---: |
| 4 | 8 | 2 | 0 |
| 5 | 6 | 4 | 0 |
| 6 | 6 | 4 | 12 |
| 7 | 4 | 8 | 10 |
| 8 | 0 | 6 | 2 |

Four or five objects is never spatial. Eight objects is never attribute. A reader can still rule a family out. A reader cannot name the family from the object count alone. That residual is a limitation, not a hidden code of the Checkpoint 1 kind.

Counting keeps a harder stratum on purpose: the 3↔4 pairs have 7 or 8 objects, and on `dev2-counting-09` and `dev2-counting-11` the arm-A count of three is neither the minimum nor the maximum color count. The 1↔2 pairs are sparser. The sparse scenes were not substituted for the crowded ones. The crowded scenes were not dropped because Checkpoint 1 counting was weak.

Spatial anchors sit in columns 2 and 3 only, so column 0 can hold a leftmost object that is not the neighbor. Both of those columns occur. All three rows occur. Column 0 and column 1 are not anchors. That is narrower than a full grid, and it is the cost of keeping a leftmost decoy without putting an object further left on the anchor's row.

## Rendering and the contact sheet

Drawings are original Pillow polygons. No photograph, no downloaded asset, no CLEVR image. Shapes are upright. Squares are axis-aligned.

`fixtures/development_v2/contact_sheet.png` is a review figure, 1588 by 748 pixels. It shows all 72 scenes. Columns are pair index 00 through 11. Rows are attribute A, attribute B, counting A, counting B, spatial A, spatial B. Captions look like `a00a red`. The sheet is not a scene and must not be passed to a model as an input. `metadata.json` records that, and `contact_sheet_is_model_input` is false.

Images are PNG, RGB, compression level 9, `optimize=False`. Checksums in `checksums.json` are SHA-256 of the committed bytes, written with Pillow 11.3.0. A different Pillow build can change PNG bytes without changing pixels. Compare pixels before replacing the files. `scene_graph_hashes.json` hashes the question, the answer, and the object list. It is an exclusion list for a future sealed generator, to be used together with the Checkpoint 1 hash file. It is not a sealed set.

## Threats

### Answer leakage

`helpful_nonrevealing` may explain the rule and must not contain the answer. `answer_revealing` contains the answer and is marked as a control. The substring tests are the check. They do not prove a model cannot infer the answer from a paraphrase. They do prove the answer string and its aliases are absent under `normalize_for_leakage`.

The helpful strings were not edited. On the Checkpoint 1 pilot, the attribute helpful sentence was often answered with a full sentence, which the strict parser marked invalid. That is a scoring fact about those 24 runs. It is not repaired by a quieter helpful sentence in this package.

### Textual shortcuts

A shortcut here means a non-revealing string, or a simple image statistic that ignores the question, that lines up with the answer on every scene.

The non-revealing texts are shared inside each pair while the answers differ. Across the file, every distinct non-revealing string occurs with at least two answers.

Image statistics the counting tests reject as perfect predictors: total object count, each color count, each shape count, each size count, the maximum color count, the second-largest color count, and the minimum color count. Several of those vary inside the counting family, which is a change from the Checkpoint 1 constants. Varying is not the same as matching the answer. The tests check inequality with the answer vector. The queried color's own count does match the answer. That is the task.

Attribute tests reject three shortcuts: the target color is the only object of that color, the target shape is the only object of that shape, and the target size is the only object of that size. They also reject a cell whose color equals the answer in every attribute scene where the cell is filled.

Spatial tests reject reading the leftmost object and reading a unique singleton color. There is no singleton color on a spatial scene: every color that appears does so at least twice, on both arms. The neighbor is the only object in its column, so "the object in the column immediately left" agrees with the task. That agreement is the clarity criterion. It is not a second path that yields the answer while the neighbor is a different object.

### Label imbalance

Attribute and spatial answer colors are balanced at six scenes each. Descriptors are balanced at four scenes each. Queried counting colors are balanced at six scenes each.

Count words are not balanced. With a +1 edit, if `a` pairs go from 1 to 2, `b` pairs from 2 to 3, and `c` pairs from 3 to 4, the scene counts are `one = a`, `two = a + b`, `three = b + c`, `four = c`. Equal words force `b = 0`. This sample sets `a = b = c = 4`, so the words are 4, 8, 8, and 4. The middle transition is the one Checkpoint 1 used. Dropping it would balance the words and would also drop that contrast. The imbalance is the recorded choice, not a count that fell out of a sampler. A sampler is not used.

The singular sentence for the foil `one` is a second textual form. It occurs with more than one true answer. It is not a label for one arm of a pair.

### Scene ambiguity

Ground truth is a function of the graph and the question template. Ties, extra objects on the left ray, a column-mate, a too-close left object, a nearest-left disagreement, overlapping boxes, and pixel centers that do not match the cell formula are flags. Emitted scenes have none. The negative tests construct the failures.

What the flags do not measure is whether a model finds the picture clear. Seven-object and eight-object counting scenes are crowded. That may lower clean accuracy later. This checkpoint does not measure clean accuracy. Those scenes stay because a low Checkpoint 1 counting score is not a reason to delete the hard end of the count range.

### Counterfactual dependence

Arms in a pair share a layout. Pixel differences are the recolored objects. An analysis that bootstraps scenes as if they were independent will overstate precision. The 36 groups are the natural cluster. The groups are not reflections of each other.

### Development and test contamination

This package has no test split. `refuse_non_development_split` raises for any other name, including `sealed`, `test`, and `heldout`, and the command-line entry point returns exit code 2 before creating a directory. `generate_development` also refuses `fixtures/development` and any directory whose metadata names the Checkpoint 1 generator, including when `--force` is passed.

The repository fixture tree contains `fixtures/development` and `fixtures/development_v2`. It does not contain a sealed or test directory. Both development hash files are published so a later sealed run can reject collisions. Publishing the hashes does not publish sealed scenes, because no sealed scenes exist.

The development domain string is `study2-development-v2`. The proposed sealed domain string is `study2-sealed-v2`. The sealed master seed is not in this repository. A later sealed generator must not reuse `study2-development-v1` or `study2-development-v2`.

### Clean-accuracy floors

Checkpoint 1 reported development floors on the 24 scenes, including clean accuracy below one half for several cells, and three scenes in three groups correct under both primary contexts. Those numbers are about that sample. This package does not remeasure them and does not claim a floor for the 72.

Whether the revised scenes are tasks a model can do is a later measurement. It has to be labeled development-only, it has to use a scoring rule that was not chosen by maximizing accuracy, and it is not a reason to have dropped a scene in this file. This checkpoint does not authorize an attack, a noise baseline, or a fit of H\*.

### Generalization beyond synthetic images

These are flat geometric drawings on a gray background. A result on them is a result on this generator. It does not extend to photographs, diagrams, or other synthetic engines unless a separate comparison says so.

### CLEVR

CLEVR stays out, for the reasons in `docs/data/DATASET_DESIGN.md`. It was not downloaded, vendored, or rendered. The recommendation is unchanged: keep it out of the primary sample until a written license reading and a mapping onto these context keys exist.

## Sealed-set procedure

This procedure is not implemented. `scene_v2` has no sampler under `study2-sealed-v2`. Running one would violate this checkpoint. The steps below are what a later freeze should do, in order.

1. Freeze the generator commit, the question templates, the context templates, and `normalize_for_leakage` before any sealed seed exists.
2. Decide which development scenes will be used for design choices. If that set is these 72, the exclusion list is the union of `fixtures/development/scene_graph_hashes.json` and `fixtures/development_v2/scene_graph_hashes.json`. Hashes of a partial development set will not exclude scenes added later.
3. The research lead draws one master seed from a public randomness beacon and records the value, the source, and the retrieval time. This file does not choose a value.
4. A future program, not this one, builds sealed scenes as a pure function of (`study2-sealed-v2`, the master seed, a cell id). It rejects any scene whose canonical graph hash is in either development exclusion file.
5. Write the sealed manifest and images outside git. Do not commit them. Do not open the images in a contact sheet.
6. Balance targets, once a size is chosen: equal scenes per family, equal answer colors for attribute and spatial, equal queried colors for counting, groups of two, and the same non-revealing-text rule as this sample.

No directory named `sealed` or `test` is created by the commands in this checkpoint.

## How to regenerate

From the repository root, with Python 3.11 or newer and Pillow 11.3.0:

```bash
PYTHONPATH=src/scene_v2 python3 -m scene_v2 --output fixtures/development_v2 --force
PYTHONPATH=src/scene_v2:src/scene_gen python3 -m pytest tests/scene_v2 -q
```

`--split` accepts only `development`. Anything else exits 2 and writes nothing. A second run without `--force` exits 1. `--output fixtures/development` exits 1 even with `--force`.

The Checkpoint 1 suite is separate and should still pass:

```bash
PYTHONPATH=src/scene_gen python3 -m pytest tests/scene_gen -q
```

No measurement in this package is a model measurement.

## What this sample does not decide

- It does not choose a parser, a loss, or a prompt. Neutral and helpful strings are unchanged.
- It does not choose a model.
- It does not fit H\*, and it does not estimate a clean-accuracy floor.
- It does not authorize PGD, FGSM, noise, or any other attack.
- It does not drop the spatial family or the crowded counting scenes because Checkpoint 1 scores were low.
- It does not add "right," "above," or "below." One repaired relation is the proposal. Adding relations would multiply a task the pilot did not show the model could do.
