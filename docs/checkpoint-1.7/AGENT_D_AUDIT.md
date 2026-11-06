# Agent and responsibility

Agent D, audit and statistics, Checkpoint 1.7 of `vlm-crossmodal-attack-transfer`. Research lead: Federico Cavaletto.

This workstream owns `src/study2_audit/**`, `tests/study2_audit/**`, this report, and `docs/checkpoint-1.7/STATISTICAL_PLAN.md`. The branch starts at `004783b5604c34aa066befbcd5356bfd7bd6e7b2`. Checkpoint 1.6 files were read and not edited. The Qwen JSON was not rewritten. No model was loaded, no attack was run, and H* was not fit on a development scene.

The diff against `main` contains Checkpoint 0 through Checkpoint 1.6 because `main` is still the initial commit. Those commits were already on the parent. This workstream adds the audit package and these two documents.

# Design decisions

The load record is split into phases. Schema `study2_audit.load_record.v1` allows one preflight phase and at most one load phase in the same document. `from_pretrained_called` lives only inside a phase. A preflight phase has call count 0. A load phase with status `loaded` has a positive call count. A derived reading is labeled `derived_reading`, carries the sha256 of the file it read, and sets `replaces_official_measurement` to false. The schema rejects a document that would overwrite the official measurement.

The original files stay the measurements. The audit pins their sha256 values and fails if the bytes change. The successful file's load-gate flag is left false, which is the defect. The derived load phase takes the document call count, which is 2, and does not copy the stale gate flag into the phase.

Reproducibility checks recompute hashes, strict parses, labels, and the stored token-id map. They load `vlm_iface.parsing` and `vlm_iface.pins` by file path so the package does not import the model session. Smol rows have no token ids. The check on those rows is the four-color class NLL, the argmin, and the strict parse of the stored string. Qwen token ids are checked for length, for the truncated flag, and for a one-to-one map with the stored string. They are not decoded.

The sampling unit is the counterfactual group. Clean-both groups and eligible groups are separate fields. The eligible count stays unset when the gradient pass is one scene. The precision study is a fabricated one-family linear probability model with one shared uniform draw per group. It refuses scene ids and group ids that look like development ids. The published three-family precision file is cited and not refit.

The decision about 96 scenes is no-go for an H* claim. A fabricated grid on which every interval excludes zero does not flip that decision, because the development packet still has no attack outcome and no eligible-group count. Building the scenes is a separate question. The decision does not block construction.

The descriptive seed is `20261108`. It is not a confirmatory seed. The Checkpoint 0 seed `20261106` and the Checkpoint 1.5 precision seed `20261107` are left on those jobs.

# Work performed

`src/study2_audit` reads the two Qwen JSON files, the row-color predictions, the manifest, the PNG checksums, the scene-graph hashes, the protocol, the budget script, the gradient gate, and the published H* precision file. It writes nothing under `docs/checkpoint-1.6`.

The Qwen audit lists every `from_pretrained_called` path. It builds an in-memory derived reading and validates that reading against the schema. The reproducibility module checks 15 conditions, including the strict parser on all 96 Smol generations and all 12 Qwen generations. The eligibility module recounts the clean-both groups. The cluster module redraws groups for the clean table. The precision module runs the fabricated grid. The decision module returns no-go.

# Tests

```bash
python3 -m pytest -q tests/study2_audit
```

27 passed. The suite covers the schema, the pinned Qwen hashes, the load-gate contradiction, the preflight file's internal agreement, the 15 reproducibility checks, the nine-group recount, the kickoff metric names, the zero-residual arithmetic, the refusal of development ids, a too-small group count, a flat outcome, an endpoint of exactly 0, seed repeatability, the documented precision grid, a no-go that survives a flattering grid, and a full audit that leaves the Qwen bytes unchanged. The package source does not import torch and does not call `from_pretrained`.

The documented grid inside the tests is the 200 by 399 run. A separate audit test uses a smaller grid so the byte-stability check does not depend on that cost. Both finished in about a second on this machine.

# Results

## The Qwen files

`docs/checkpoint-1.6/qwen_feasibility.json` sha256 `055a133be14cf7f76c20bb59880cf8f4ca0a66321dea71960a78e8ca97b21dd7`.

`docs/checkpoint-1.6/qwen_feasibility_preflight.json` sha256 `c2d8adc0a830fd58f65e553c69f84889cf37888e6381575d2a50d52976bf999c`.

Both files name revision `89644892e4d85e24eaac8bacfd4f463576704203`.

The preflight file has `from_pretrained_called` false at the top and false on every nested copy. The call count is 0. The derived load phase has status `aborted`.

The successful file has `from_pretrained_called` true, `processor_from_pretrained_called` true, `model_from_pretrained_called` true, and `from_pretrained_call_count` 2. Five nested fields are still false:

| Path | Value | Role |
| --- | --- | --- |
| `$.model_compatibility.from_pretrained_called` | false | Preflight stamp. Class resolution does not load weights. |
| `$.qwen_candidate.from_pretrained_called` | false | Preflight stamp. The survey does not load weights. |
| `$.qwen_dtype_decisions.float32.from_pretrained_called` | false | Preflight stamp. The dtype decision does not load weights. |
| `$.qwen_dtype_decisions.bfloat16.from_pretrained_called` | false | Preflight stamp. The dtype decision does not load weights. |
| `$.gates.weight_load_and_memory.from_pretrained_called` | false | Load gate. Status is `loaded`. |

The first four describe steps that did not call `from_pretrained`. They are true descriptions of those steps and false answers to the question "did the run load the model?" The fifth is inside the load object. The runner set the flag false when the load started and later changed only the status to `loaded`. The derived reading puts call count 2 on the load phase and call count 0 on the preflight phase.

The candidate object's reason still says `from_pretrained` must not be called, and `load_permitted_by_weight_guard` is false, while `dtype_selection.permitted` is true for bfloat16. The candidate object is the float32 weight-guard sentence. It was not updated after the BF16 budget cleared.

Large-blob lists before and after the successful run are equal. `weights_downloaded` is false. `qwen_local_weight_file_present` is true. The call count is 2. That combination is a load from a cache that already held the weight file.

## Reproducibility

All 15 checks matched.

The protocol sha256, the row-color manifest sha256, and the Checkpoint 1.5 budget-script sha256 match the values stored on the run. Every row-color PNG matches `fixtures/development_simple/checksums.json`. Every scene-graph hash recomputes from question, answer, and objects. Every Smol row carries revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`, which is the pin in `vlm_iface.pins`. The recorded git head `49a3945b42ad132b662178317eea0eb2bac80ddc` is an ancestor of this checkout. Qwen's recorded budget-script hash matches the file.

The 96 Smol generations reparse with the Checkpoint 1 strict parser on the color vocabulary. Labels match the manifest. Prompts are the context, a newline, and the question. Correct, valid-wrong, and invalid form a partition. The class-loss argmin matches the stored NLL surface. The summary's clean counts, the 14 scenes, the 9 groups, and the outside-band recommendation match that recomputation. No Smol row stores token ids. Tokenization of those generations is the stored four-color NLL plus the protocol's leading-space sentence. It was not re-run through a tokenizer.

The 12 Qwen generations reparse to the stored reasons. All 12 are invalid. The subset labels match the development-v2 manifest. Token-id lists have length at most 16. `truncated` is true exactly when the list has length 16. The same string maps to the same id list and the same id list maps to the same string. The non-truncated rows end with id `151645`. Truncated rows do not. The ids were not decoded.

The gradient gate is one scene, `dev3-row-00-a`, two primary contexts, finite gradients, nonzero fraction 1, shape 384 by 512, repeat difference 0.

## Nine groups, intervals, and the decision

The recount is 14 scenes, 9 groups, 5 doubletons and 4 singletons. `dev3-row-00-a` is not in the 14. The eligible-group count is unset. The groups do not establish H*. The algebra, the kickoff metrics, the ceilings, and the fabricated precision are in `STATISTICAL_PLAN.md`.

On seed `20261108` with 4000 group redraws, neutral strict accuracy 14/24 has cluster interval 0.333 to 0.792. The group rate 9/12 has interval 0.500 to 1.000. Helpful accuracy 17/24 has cluster interval 0.458 to 0.917, wider than the scene-iid interval 0.542 to 0.875.

On the fabricated one-family grid, 200 replications, 399 bootstrap draws, known coefficient 0.25, the share of intervals with lower endpoint above zero is 42/200 at 9 groups, 34/200 at 12, 64/200 at 24, and 90/200 at 48. Median cluster widths are 1.020, 0.947, 0.686, and 0.512. Coverage of 0.25 stays between 175/200 and 190/200. The decision is no-go for treating 96 scenes as support for H*. Scene construction is not blocked.

# Limitations

The derived reading is a proposal sitting in memory. It is not a second measurement, and it does not repair the committed JSON.

Smol token ids are absent, so the leading-space convention is checked only as a protocol sentence and as the four stored NLL keys. Qwen ids are not decoded, so a tokenizer change that preserved these 12 lists would pass.

The group bootstrap describes redraws of these 12 groups. It is the uncertainty of this table under that redraw. It is not a super-population claim about SmolVLM2.

The precision grid uses one dependence structure, one coefficient, one family, and 200 replications. A binomial standard error near 0.03 sits on the exclusion shares. The dip from 42/200 to 34/200 is inside that noise. The coefficient 0.25 is the old planning value, not an effect estimated from these models. The published three-family file is a different design and was not regenerated.

The audit does not score the prospective summed NLL, because that would be a model call. It records the score as prospective and leaves the official column alone.

# Unexpected findings

The load gate is the sharp defect: status `loaded` with `from_pretrained_called` false in the same object. The other four false flags are preflight stamps that share the field name. A reader who greps the file sees six answers.

The candidate reason still forbids the call after the BF16 budget has permitted it. `load_permitted_by_weight_guard` remains false.

The only gradient scene is outside the clean-both set. A full eligible-group count cannot be recovered by relabeling the nine groups, and it also cannot be recovered from the one scene that was differentiated.

At 9 groups the row-level median width (1.043) is larger than the cluster width (1.020). From 12 groups upward on this grid the cluster width is larger. A cluster interval is not automatically the wider one.

`contradictory` and `answer_revealing` both-arms rates are 0/12, and the bootstrap width is 0 because the table has no variation. The revealing strict rate is 0/24 while the class-loss argmin matches on 22/24.

The successful run reports `weights_downloaded` false while the call count is 2, with the same large blobs before and after. The weight file was already on disk.

# Unresolved questions

The lead decides whether future Qwen records must use `study2_audit.load_record.v1`, and whether the stale load-gate flag is repaired in a new file or left as the historical measurement. This workstream leaves the historical file untouched.

The lead decides whether the prospective primary score is summed teacher-forced NLL before any result is judged on it. This plan states that score and does not apply it to the row-color table.

A confirmatory seed, an eligible-group rule, and a single attack budget are still unregistered. Ninety-six scenes do not settle them. The eligible-group count on the row-color scenes stays unset until a full gradient pass exists, and H* stays unset until attack rows exist.

Whether 36 groups, the arithmetic product of 48 and 9/12, would be a useful development target is a design question. It is not an estimate of how many groups will survive the gradient gate.

# Reproduction

From the repository root:

```bash
python3 -m pytest -q tests/study2_audit
PYTHONPATH=src/study2_audit python3 -m study2_audit
```

The module prints one JSON document to stdout. Exit 0 means the pinned hashes match, the reproducibility checks match, the Qwen file was not rewritten, H* was not fit, and the decision is no-go. NumPy 1.26.4 and pytest 8.3.5 are the pins in `src/study2_audit/pyproject.toml`. This run used the system NumPy 1.26.4 and pytest 8.3.5. No GPU was used.

# Git

- Branch: `checkpoint-1.7/study-audit`
- Parent commit: `004783b5604c34aa066befbcd5356bfd7bd6e7b2`
- Head SHA: recorded in the commit that adds the pull request URL
- Pull request: recorded in the commit that adds the pull request URL
