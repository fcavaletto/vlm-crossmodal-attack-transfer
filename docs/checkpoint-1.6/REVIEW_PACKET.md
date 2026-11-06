# Checkpoint 1.6 review packet

**DEVELOPMENT ONLY.** No attack was run. H\* was not fit. No held-out scene was generated. Nothing was trained. No cloud GPU was purchased. The diff against `main` contains Checkpoint 1 and Checkpoint 1.5 because this branch starts at `b2e7abff72b87b475d0abf8d037e1b4d2f9c0c06`, not at `main`. Do not merge.

The row-color protocol is `docs/checkpoint-1.6/EVAL_PROTOCOL.md`, commit `3b4b60a0b0fb8509be6472566420b9294c9bae37`. The model call that scored those scenes recorded git head `49a3945b42ad132b662178317eea0eb2bac80ddc`. That head is a descendant of the protocol commit. The protocol file was not edited after the run.

## What the four workstreams found

| Workstream | Branch tip | Result |
| --- | --- | --- |
| A, stronger model | `f42dfa986ca96cd4e03d4b3fe89dba40768867b0` | The first reading refused the load before `from_pretrained`. After memory was free, the same budget permitted BF16. Qwen remains a candidate. |
| B, task design | `e6c0b0f9a318a32601659e421c868b94cbe61c47` | 24 development scenes, 12 pairs, one family `row_color`. No model was used to design them. |
| C, scoring | `e88f1854985675bf770fbb08f0f127f247ba3ef0` | On the committed v2 rows, class scoring does not rescue clean accuracy. The official metric stays the strict parse. |
| D, reproducibility | `c551e014126de2d50eca170117ffcabfcc0c692d` | The four Checkpoint 1.5 failures are explained without rewriting fixtures. On that branch the previous suite was 231 passed. |

Reports: `AGENT_A_MODEL.md`, `AGENT_B_BENCHMARK.md`, `AGENT_C_SCORING.md`, `AGENT_D_REPRO.md`.

## Models

The row-color table below is SmolVLM2-500M-Video-Instruct, revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`, MPS, float32, eager, image splitting off, `max_new_tokens=16`, leading-space answer tokens.

Qwen3-VL-2B-Instruct, revision `89644892e4d85e24eaac8bacfd4f463576704203`, is still a serious candidate. It is not ruled out by the earlier preflight. That reading, kept in `qwen_feasibility_preflight.json`, saw 10,080,436,224 available bytes while other processes were using the machine. Float32 needs 20,871,410,752. BF16 needs 11,509,447,200 and missed the calibrated projection by 1,429,010,976 bytes. `from_pretrained` was not called. That was a budget refusal, not an out-of-memory event, and not a finding that the model cannot run.

The budget was not lowered. With 12,188,794,880 bytes available, every BF16 line cleared. `from_pretrained` then ran twice (processor and model). The load was BF16 on MPS, eager attention, 5.253 s. All 2,127,532,032 parameters were `torch.bfloat16`. After the load, MPS current allocation was 4072.9 MB and OS-available memory was 4259.3 MB. A 64×64 synthetic forward returned finite logits in 2.011 s.

On the pre-listed 12 rows (`dev2-attribute-00-a`, `dev2-counting-00-a`, `dev2-spatial-00-a`, four contexts each), the strict parser marked 0 correct, 0 valid-wrong, and 12 invalid in 23.534 s. Every string is a sentence. Several name a color, including the revealing attribute string "Based on the image provided, the large circle is red." The strict parser rejects that sentence. This is not a demonstration that Qwen cannot see the scene, and it is not a reason to drop the candidate. It is the same sentence problem the strict parser already records for SmolVLM2.

Input gradients of the canonical-word loss were finite and nonzero for `dev2-attribute-00-a` under `neutral` (18.857 s) and `helpful_nonrevealing` (19.859 s). The leaf is processor `pixel_values`, shape `[768, 1536]`, not the `[0, 1]` RGB image. The record says so. No attack was run. Wall clock for the successful attempt was 74.653 s. An earlier permitted attempt had already downloaded the weights and then stopped on a missing record key before the forward. That key was initialized and the budget numbers were not changed. The completed record is `qwen_feasibility.json`.

A 24 GiB CUDA job remains the alternative if a larger Qwen run is wanted and local memory is again below the budget. Agent A's planning figures are 4,255,064,064 bytes of BF16 weights, 8 GiB of headroom, and an estimated $0.34 to $1.79 for up to one hour. Nothing was purchased. The safeguards were not weakened.

## Dataset, prompt, and score

Dataset: all 24 rows of `fixtures/development_simple`, sha256 `bf21cdcfeca8f5c9a05e9f53465402c48f4a3175bbd363a38ca36dbc443f2367`. Family `row_color`. Prompt: context text, a newline, the question. No format line.

Official outcome: Checkpoint 1 strict parser on the four color names. The class-loss argmin is stored and is not official. The acceptance band, fixed before these numbers, is strict accuracy from 0.60 through 0.80 in both `neutral` and `helpful_nonrevealing`. Below 0.50 in either primary context would be unsuitable. No row was deleted.

## Clean accuracy on the row-color scenes

These rows are SmolVLM2, not Qwen.

Denominator 24. Cells are correct / valid-wrong / invalid. **DEVELOPMENT ONLY.**

| Context | Strict | Class-loss argmin correct |
| --- | ---: | ---: |
| `neutral` | 14 / 3 / 7 | 17/24 |
| `helpful_nonrevealing` | 17 / 6 / 1 | 17/24 |
| `contradictory` | 2 / 0 / 22 | 18/24 |
| `answer_revealing` | 0 / 0 / 24 | 22/24 |

Strict accuracy is 14/24, 17/24, 2/24, and 0/24. Helpful text is inside the band (0.708). Neutral is 0.583, below 0.60 and above 0.50. The pre-specified recommendation is **outside the band**, not unsuitable, and not an adoption. Attacks stay unauthorized.

The class-loss argmin is also 17/24 on both primary contexts. That would sit inside the band. It is not the official outcome. Adopting it because this table looks better would spend the metric choice on this run. The v2 warning still stands: there, class scoring moved the primary totals only from 26/72 and 22/72 to 30/72 and 26/72, both below one half.

## Counterfactual pairs

Twelve pairs. Both arms strict-correct: `neutral` 5/12, `helpful_nonrevealing` 8/12, `contradictory` 0/12, `answer_revealing` 0/12.

## Scoring and generation

96 generations. 42 are valid under the strict parser. On those 42, the class-loss argmin equals the official label on 42/42. The gap is the invalid rows. Under `answer_revealing`, every output is a sentence, so the strict score is 0/24, while the class-loss argmin matches the true color on 22/24. Under `contradictory`, 22/24 are sentences. Neutral has 7 invalid rows, several of them sentences. Helpful text has one invalid row (`The answer is yellow`).

## Scenes correct under both primary contexts

14 scenes in 9 independent groups are strict-correct under both `neutral` and `helpful_nonrevealing`. The groups are `cf3-row-00`, `cf3-row-01`, `cf3-row-02`, `cf3-row-05`, `cf3-row-06`, `cf3-row-07`, `cf3-row-08`, `cf3-row-10`, and `cf3-row-11`. This is a count. It is not a fitted coefficient. H\* stays provisional.

## Memory, runtime, and gradients

Apple M4, 24 GiB unified memory. Interpreter Python 3.12.14, torch 2.15.0. Load 5.125 s. Scoring 61.696 s. Gradient check 17.843 s.

| Snapshot | OS available MB | MPS current MB | Process RSS MB |
| --- | ---: | ---: | ---: |
| Before load | 8805.4 | 0.0 | 205.9 |
| After load | 5693.5 | 1965.6 | 188.1 |
| After scoring | 7124.4 | 1965.6 | 448.5 |
| After the gradient check | 9091.8 | 2.2 | 624.2 |

The one-scene gradient check on `dev3-row-00-a` passed. Both canonical-word image gradients were finite, nonzero in every component, shaped `(3, 384, 512)`, and unchanged on repeat. Neutral took 4.789 s (L1 4246.361). Helpful took 4.378 s (L1 2881.588). These magnitudes are a finiteness check. They are not an attack step size. A full 24-scene gradient pass was not run.

## Tests

Integrated tree, model interpreter, one command covering scoring, the model interface, the pilot, both older scene packages, analysis, the Checkpoint 1.5 validator, class scoring, and the new scenes:

```bash
PYTHONPATH=src/simple_eval:src/class_score:src/scoring:src/vlm_iface:src/pilot:src/analysis:src/scene_v2:src/scene_gen:src/scene_simple:src/validation \
  python -m pytest tests/scoring tests/vlm_iface tests/pilot tests/scene_v2 tests/scene_gen tests/analysis tests/validation tests/class_score tests/scene_simple -q
```

**299 passed**, 0 failed, 120 warnings. The warnings are Pillow `Image.getdata` deprecations. Four `scene_simple` modules were renamed so they do not collide with Checkpoint 1 test modules, and the row-color oracle module was renamed for the same reason. Committed predictions, PNG checksums, and the H\* summary were not rewritten. Median interval widths in the precision file may differ by at most 1e-9; coverage counts and the recommendation of 24 groups stay exact.

## Remaining scientific risks

- The official neutral rate is just under the band. The task did not clear the gate that was written before the run.
- The revealing control is 0/24 under the strict parser because the model writes a sentence. The class loss often still ranks the true color. Changing the official metric after reading 0/24 against 22/24 is the choice this packet does not make.
- Contradictory text is followed by sentences on 22/24 scenes. Two strict successes are not evidence that the false sentence is ignored.
- Nine clean-correct groups are a development count. They are not the three-family coverage description of 24, and they are not a power calculation. H\* was not fit.
- Qwen's 12 strict scores are all invalid sentences. That does not remove it as a candidate, and it does not make it the primary model. A larger Qwen run still needs either free local memory above the unchanged budget or an authorized CUDA job.
- The drawings are this generator. A result here is not a result on photographs.

## Decision required

Attacks stay unauthorized. Do not start another checkpoint from this packet.

The lead can decide, in writing and before the result that would be judged on it, whether the official row-color outcome remains the strict parse, and whether Qwen should be scored on the new scenes under that same rule. Local BF16 inference and a processor-pixel gradient are now demonstrated. A paid GPU is not required for that demonstration and is not authorized. SmolVLM2 still has the only full row-color table. That task did not clear the pre-specified band. H\* remains provisional.
