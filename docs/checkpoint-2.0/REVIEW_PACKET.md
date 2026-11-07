# Checkpoint 2.0 review packet

DEVELOPMENT ONLY. Research lead: Federico Cavaletto. Qwen3-VL-2B-Instruct, revision `89644892e4d85e24eaac8bacfd4f463576704203`, bfloat16, eager attention. The calibration attack ran on MPS. No H* fit. No held-out scenes. No new model load. Not merged to main.

This packet integrates the two Checkpoint 2.0 workstreams. The calibration results are commit `ab8ffa27d94cf41e32cfb6e98e981f8ce5853410`. The protocol commit is `efc7d745546152a81b4cc21a021bbe641162c498`, sha256 `339d95a75de7157eaf0d2beb3e58a78a06666695c3ddc6009885e4d25ef37137`. The files under `docs/checkpoint-2.0/cuda/` are unchanged copies from `checkpoint-2.0/cuda-readiness` at `95089070ef11fc2e35bee6026f391efef36c0947`. Both workstreams start from Checkpoint 1.9 head `0cd2f2a8e566b6530dcfd4dd7021961b2fc17990`. The Checkpoint 1.9 protocol commit remains `fce3480d158298851587a692c56b6e4eebb554a7`. This pull request targets `checkpoint-1.9/attack-feasibility`.

Every count below was read from `docs/checkpoint-2.0/calibration/summary.json`. The sample is exploratory. It is not a population attack-success rate.

## Recommendations

The primary configuration for a later full-development study is PGD at epsilon 1/255 for 3 steps: 16/32 source successes and 13/16 conditional transfers, the first PGD row in the calibration grid that the precommitted rule accepts.

The secondary checks, all at epsilon 1/255, are the positive sign step (7/32 source successes, 2/7 conditional transfers), the matched random-sign step (0/32 source successes), and the same PGD trajectory at step 7 (28/32 source successes, 27/28 conditional transfers, which the report places at the transfer ceiling).

Neutral-to-helpful can be evaluated from this calibration, and helpful-to-neutral cannot, because every attack was optimized under neutral text.

Binary transfer on this sample does not have enough variation to investigate H*. Conditional transfer is 13/16 at the primary setting and 27/28 at seven steps, and H* was not fit.

On these 32 scenes the two stored clean margins rank together (Spearman 0.883), the stored neutral gradient L1 ranks opposite those margins (−0.769 and −0.653), and stored alignment is only weakly ranked with either. Transfer is not an outcome of that comparison, and the comparison does not predict attack success.

The planning estimate for the primary configuration is 2.074 hours and $2.03 for 96 development scenes, and 5.530 hours and $5.41 for a 256-scene planning count, on one NVIDIA L4 (`g6.2xlarge`, us-east-1, Linux on-demand at $0.9776 per hour), assuming a CUDA scene matches the measured MPS time. No GPU was launched, and no held-out image was opened.

## What was counted

Rule `counterfactual-group-stride-3`. The development manifest has 48 counterfactual groups. Indexes 0, 3, 6, ..., 45 give 16 groups and, with both arms, 32 scenes. The rule reads the group id and the scene id. It does not read alignment, margin, loss, or a Checkpoint 1.9 outcome. All 32 scenes finished. Status `completed`. Epsilon was not raised, and no step was added.

The source context is neutral. `helpful_nonrevealing` is an evaluation context. The stored attack is the official 8-bit image, the half-to-even round trip onto multiples of 1/255. Source success is a scene whose clean neutral decision is the labeled color and whose perturbed neutral decision is not. The source-success denominator is the number of source-clean-correct scenes with a recorded perturbed neutral decision. Conditional transfer follows that definition among source successes that were also clean-correct under helpful text: the numerator is how many of those helpful decisions leave the label. A scene already wrong on the clean helpful answer is excluded and is not a transfer failure. A missing decision stays out of the numerator and the denominator.

On every recorded configuration, 32 of 32 scenes were clean-correct under neutral and under helpful text. Exclusions are 0. Missing source decisions, missing transfer decisions, and missing target decisions are 0.

Near the floor means the numerator is 0 or 1. Near the ceiling means the numerator is the denominator or one less. A denominator below 2 is undefined. Adequate source success means the source-success count is interior or near the ceiling. Variation in conditional transfer means the conditional-transfer count is interior. The precommitted rule accepts a configuration only when both descriptions hold. The summary sets `alignment_used` to false on every row.

| Configuration | Source success | Conditional transfer | Target flips | Rule |
| --- | ---: | ---: | ---: | --- |
| `pgd\|1/255\|2` | 4/32 | 0/4 | 1/32 | source interior; transfer near the floor |
| `pgd\|1/255\|3` | 16/32 | 13/16 | 15/32 | accepted |
| `pgd\|1/255\|4` | 19/32 | 15/19 | 15/32 | accepted |
| `pgd\|1/255\|5` | 25/32 | 24/25 | 24/32 | transfer near the ceiling |
| `pgd\|1/255\|7` | 28/32 | 27/28 | 27/32 | transfer near the ceiling |
| `fgsm\|1/255\|1` | 7/32 | 2/7 | 2/32 | accepted |
| `random\|1/255\|1` | 0/32 | 0/0 | 0/32 | source near the floor; transfer undefined |
| `pgd\|2/255\|2` | 15/32 | 12/15 | 12/32 | accepted |
| `pgd\|2/255\|3` | 18/32 | 16/18 | 16/32 | accepted |
| `pgd\|2/255\|4` | 25/32 | 24/25 | 24/32 | transfer near the ceiling |
| `pgd\|2/255\|5` | 26/32 | 26/26 | 26/32 | transfer near the ceiling |
| `pgd\|2/255\|7` | 28/32 | 27/28 | 27/32 | transfer near the ceiling |
| `fgsm\|2/255\|1` | 6/32 | 2/6 | 2/32 | accepted |
| `random\|2/255\|1` | 0/32 | 0/0 | 0/32 | source near the floor; transfer undefined |

The six accepted rows are the ones with `adequate_source_and_transfer_variation` true. The calibration report lists them and does not rank them. This packet does not add an epsilon or a step to that grid.

## Primary configuration

`pgd|1/255|3`. Projected gradient ascent on the summed labeled-color negative log-likelihood, under neutral text, epsilon 1/255, three steps, step length epsilon/4, one start from the clean image, no restarts.

The calibration grid records, for each budget, PGD after steps 2, 3, 4, 5, and 7, then the positive sign step, then the matched random-sign step. Reading that order, `pgd|1/255|2` is the first PGD row and its conditional transfer is 0/4, near the floor, so the rule rejects it. `pgd|1/255|3` is the next PGD row, and the rule accepts it: source success 16/32 is interior, and conditional transfer 13/16 is interior. The selection stops at that row. `pgd|1/255|4` also meets the rule, at 19/32 and 15/19, and it is the next image on the same trajectory. It is not a second primary. The walk does not continue to epsilon 2/255 once a smaller-budget PGD row has already been accepted. The same stopping preference is the one in `docs/research/RESEARCH_PROPOSAL.md` section 9, written before this attack: freeze the mildest candidate that still leaves variance. The proposal's numerical window is not the rule used here. The rule used here is the calibration summary's pair of flags.

`pgd|1/255|7` is 28/32 and 27/28. `pgd|2/255|7` is the same pair of counts. The summary marks both transfers near the ceiling and sets `transfer_variation` false. Step 7 is not the primary. The single scene that does not transfer at `pgd|1/255|7` is `dev4-ref-21-a`.

`fgsm|1/255|1` meets the rule as well, at 7/32 and 2/7. In the recorded grid it is the control that follows the PGD steps. The full-development attack is the multi-step image, and the one-step image stays the linear check at the same epsilon. The selection reads the summary flags and the grid order. It does not read \(A(\text{neutral}, \text{helpful})\), either clean margin, or the gradient L1.

The three source successes that remain correct under helpful text are `dev4-ref-00-b`, `dev4-ref-18-b`, and `dev4-ref-33-b`. Two other scenes, `dev4-ref-03-a` and `dev4-ref-03-b`, change the helpful decision without changing the neutral decision. That is why the target-flip count is 15/32 while conditional transfer is 13/16. All 32 source-clean scenes have a recorded labeled-NLL increase; the median at this configuration is 1.484.

On all 32 scenes the L-infinity distance from the clean image is 0.002941220998764038 before the 8-bit round trip and 0.003921575844287872 after it. The before value is three steps of length epsilon/4. The after value lies within \(10^{-8}\) of 1/255 (0.00392156862745098), so the official image sits on the budget.

## Secondary checks

Three recorded configurations, all at the primary epsilon.

The positive sign step, `fgsm|1/255|1`, is the linear control. Source success is 7/32. Conditional transfer is 2/7. Both counts are interior, so the control is not a saturated copy of the primary. The median labeled-NLL increase on the 32 source-clean scenes is 0.052. The official L-infinity distance after the round trip is 0.003921575844287872 on all 32 scenes.

The matched random-sign step, `random|1/255|1`, uses the scene seed fixed in the protocol and the support of the clean gradient's sign. Source success is 0/32. Target flips are 0/32. The conditional-transfer denominator is 0, so the position is undefined. The summary marks source success near the floor. The median labeled-NLL increase is −0.000198. A step of this budget that ignores the gradient sign does not change the decision.

The step-7 image, `pgd|1/255|7`, is the saturation check. Source success is 28/32. Conditional transfer is 27/28, near the ceiling. The median labeled-NLL increase is 8.773. The check documents what the longer attack does on this sample. It is not the configuration a later study should freeze.

`pgd|1/255|4`, `pgd|2/255|2`, `pgd|2/255|3`, and `fgsm|2/255|1` remain accepted rows in the table above. They are not added to the full-development attack. At `pgd|2/255|2` the step length is half of 1/255, two steps have not filled the 2/255 ball, and the official L-infinity distance on all 32 scenes is again 0.003921575844287872. The counts for that row stay 15/32 and 12/15.

## The two directions

Neutral-to-helpful is in the records. Every configuration above was scored under `helpful_nonrevealing` after the official round trip, and the conditional-transfer denominators are those helpful decisions. Helpful-to-neutral is not in the records. The image was optimized under neutral text. No gradient was taken under helpful text, and no image was crafted there. All 32 scenes were clean-correct under the helpful prompt, with zero exclusions, so that prompt is a usable evaluation context on this sample. It was not a source context. A later study can evaluate both ordered directions only by running the helpful-source attack. This packet does not start that attack.

## Binary transfer and H*

This sample cannot support an investigation of H*.

The binary transfer count in the summary is conditional on source success. At `pgd|1/255|3` it is 13/16, so the entire zero class is three scenes. At `pgd|1/255|4` the zero class is four scenes (15/19). At `pgd|2/255|2` it is three (12/15). At `pgd|2/255|3` it is two (16/18). The FGSM rows are 2/7 and 2/6. At both seven-step settings the count is 27/28. The calibration report calls 27/28 near the ceiling, because the numerator is one less than the denominator. 27/28 = 0.964. The H* instrument marks a mean outcome near-flat below 0.05 or above 0.95, so this conditional-transfer rate is above that line. The target-flip rate at the same step is 27/32 = 0.844, which is below 0.95. Sixteen groups and a handful of non-transfers, in one direction, are not a sample on which to estimate β_A.

The target-flip column is a wider cut: it counts a helpful-decision change on every scene that was clean-correct under helpful text, without requiring a source success. At the primary that column is 15/32. Fifteen out of thirty-two is not a floor. It is not the calibration's transfer denominator, and it is not a fitted coefficient. H* was not fit. No transfer bit was regressed on alignment.

## Stored scalars

The comparison uses the 32 stored Checkpoint 1.8 values in `docs/checkpoint-2.0/calibration/checkpoint_1_8_description.json`. Alignment is \(A(\text{neutral}, \text{helpful})\). Both margins are clean margins. Gradient magnitude is the neutral gradient L1. The file marks each value as unused for the sample, unused for a configuration, and not regressed on transfer. The same flags are kept here. Source success, target flips, and conditional transfer do not enter the table.

Each median is the average of the 16th and 17th stored values after sorting. The table then rounds to the same reading precision as the calibration report: six decimals for alignment, four for the margins, and one for the gradient L1.

| Scalar | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| Alignment | 0.587596 | 0.879154 | 0.975363 |
| Neutral clean margin | 1.5000 | 8.3125 | 13.0000 |
| Helpful clean margin | 2.2500 | 7.2500 | 12.7500 |
| Neutral gradient L1 | 20.3 | 398.1 | 10168.7 |

All 32 alignments are positive. Thirteen are at least 0.90, and two are below 0.70. The smallest is 0.5875956154399549 on `dev4-ref-30-b`. The largest is 0.9753625290227933 on `dev4-ref-45-b`.

Spearman rank correlations on these 32 rows, with average ranks where values tie:

|  | Neutral margin | Helpful margin | Gradient L1 |
| --- | ---: | ---: | ---: |
| Alignment | −0.273 | −0.306 | 0.255 |
| Neutral margin |  | 0.883 | −0.769 |
| Helpful margin |  |  | −0.653 |

The two clean margins move together. The stored gradient L1 ranks in the opposite direction: large gradients sit with small margins. Alignment's rank association with the margins and with the gradient L1 is weak beside those two relationships. The rank correlations above are associations among stored clean-image scalars. They are not a model of transfer, and they do not predict attack success.

## Compute plan

This section is a planning estimate. It is not a CUDA measurement. No GPU was started. No held-out file was opened. `docs/checkpoint-2.0/cuda/VALIDATION.md` records that the CUDA commands were not executed on CUDA.

The GPU class and the rate are unchanged from `docs/checkpoint-2.0/cuda/PLANNING_ESTIMATE.md`. The class is one NVIDIA L4 on an AWS EC2 `g6.2xlarge` in `us-east-1`, Linux on-demand, at $0.9776 per hour. The instance has 8 vCPUs and 32 GiB of host memory. The L4 is the 24 GB class, and the memory note in that document uses 22 GiB as the smaller published floor. The rate excludes EBS, a public IPv4 address, data transfer, and tax. The assumption `assumed_cuda_seconds_per_mps_second = 1` is kept: a CUDA scene is assumed to take the measured MPS time. Ratios 0.5 and 2 are the same sensitivities, not new measurements.

The scaler is the Checkpoint 1.9 Stage B scene time: 7,878.226 seconds across 16 finished scenes, which is 492.389125 seconds per scene. That scene is three budgets, 1/255, 2/255, and 4/255, seven projected steps of length epsilon/4, plus one positive sign step and one matched random-sign step at each budget. The Checkpoint 1.9 finishing process records a load of 3.626 seconds. That load is added once and is not scaled. It is an MPS load time. The calibration's own load, 5.19 seconds, and its scene sum, 10,023.147 seconds over 32 scenes, describe the two-budget calibration grid with intermediate scores. They are not the scaler.

In `src/attack_feasibility/attack_feasibility/stage_b.py` the measured scene calls `backward_at_image` once on the clean image. Inside each budget the seven-step loop calls it again only when the step index is greater than zero, which is six calls. Three budgets make \(1 + 3 \times 6 = 19\) backward passes. The sign step and the random-sign step reuse the clean gradient.

The primary configuration is one budget and three PGD steps. The clean backward plus the two later steps is 3 of those 19 passes. The two one-step controls at 1/255 add no further backward. The scale on the 492.389125 seconds is 3/19. Official scoring sits inside the measured scene time and is carried by the same factor. That is a planning assumption. The ratio-2 row is the wide side of the published sensitivity.

Scene seconds equal \(492.389125 \times (3/19) \times N \times r\). Wall seconds add 3.626. Hours divide by 3600. Dollars multiply by 0.9776. Rounding is half-even, seconds and hours to three decimals and dollars to cents, matching `cuda_readiness.estimate`.

| Study | Scenes | Ratio | Wall seconds | Hours | On-demand cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Development, primary configuration | 96 | 1 | 7,467.209 | 2.074 | $2.03 |
| Development, primary configuration | 96 | 0.5 | 3,735.417 | 1.038 | $1.01 |
| Development, primary configuration | 96 | 2 | 14,930.791 | 4.147 | $4.05 |
| Held-out planning count | 256 | 1 | 19,906.513 | 5.530 | $5.41 |
| Held-out planning count | 256 | 0.5 | 9,955.069 | 2.765 | $2.70 |
| Held-out planning count | 256 | 2 | 39,809.399 | 11.058 | $10.81 |

The 96 is the `development_v4` manifest. The 256 is the sealed-set size written in `docs/research/RESEARCH_PROPOSAL.md` section 12. No sealed directory was listed or opened. A helpful-source repeat would be a second pass of the same length. Those dollars are not in the table.

Repeating the step-7 check on the same lists uses 7 of the 19 backward passes, because the trajectory continues through steps 4, 5, 6, and 7. At ratio 1 the wall time is 17,418.652 seconds, 4.839 hours, $4.73 for 96 scenes, and 46,443.695 seconds, 12.901 hours, $12.61 for 256 scenes. The calibration already has the step-7 ceiling on its 32 scenes. The 7/19 line is the cost of repeating that check. It does not make step 7 the primary.

The unscaled Stage B quote, for the three-budget seven-step grid, stays the one in the CUDA document: 47,272.982 seconds, 13.131 hours, $12.84 for 96 scenes, and 126,055.242 seconds, 35.015 hours, $34.23 for 256 scenes, at this same GPU and this same rate. The table above replaces that grid with the recommended configuration.

GPU memory is the estimate in the CUDA document. This machine's calibration reading is an MPS reading: 12,419,497,984 bytes available at the load decision, bfloat16 permitted against the unchanged line of 11,509,447,200 bytes, float32 not loaded, the 6 GiB floor still 6,442,450,944, and the lowest OS-available reading during the run 7,608,238,080. The guard was not lowered. No CUDA allocator high-water mark was read.

## What this packet leaves in place

Epsilon stays inside {1/255, 2/255}. The recorded PGD lengths stay 2, 3, 4, 5, and 7. The objective stays the summed labeled-color NLL under neutral. The memory lines stay the Checkpoint 1.5 lines. H* stays unfitted. Held-out scenes stay unopened. The branch does not merge to main.

## Files

Calibration, at `ab8ffa27d94cf41e32cfb6e98e981f8ce5853410`:

- `docs/checkpoint-2.0/CALIBRATION_PROTOCOL.md`
- `docs/checkpoint-2.0/CALIBRATION_REPORT.md`
- `docs/checkpoint-2.0/calibration/summary.json`, sha256 `bb2f9408c86c9879e23acda6f3a0533e3849e66f9db0f0270089ffae96c98574`
- `docs/checkpoint-2.0/calibration/records.jsonl`, sha256 `1669eefd11ba4703d09ec746078e7be6a52b25ff55e35a8d0b64958eb5a7122f`
- `docs/checkpoint-2.0/calibration/progress.json`, sha256 `4572ffe1463cadb47dad5b4708adfb7c603b2df8f4e6e73e5bc4bc7608cd8ca5`
- `docs/checkpoint-2.0/calibration/checkpoint_1_8_description.json`, sha256 `79dcde633a640e8c8a31ce88d3d1503cb6f76fc48b8d09625ec1890a266f9f61`

CUDA planning documents, copied unchanged from `95089070ef11fc2e35bee6026f391efef36c0947`:

- `docs/checkpoint-2.0/cuda/PLANNING_ESTIMATE.md`, sha256 `ad7100e5de3003800f54c354ab5790ca095adc44472577b78d830dc4ed8acbbc`
- `docs/checkpoint-2.0/cuda/planning_estimate.json`, sha256 `65404a678850735d4f2b255d7b7ce1e83f69c670b855d043fc8ed1e46ee677d0`
- `docs/checkpoint-2.0/cuda/RUN_GUIDE.md`, sha256 `a9e0dddbdf602deb362fae0627ac1a683421efb3c4d84501480489eefd0a7444`
- `docs/checkpoint-2.0/cuda/VALIDATION.md`, sha256 `2a715bad5c71d504792f0021b0096c37a79b198e51196dd8b3f3ec9f3bc8f594`
- `docs/checkpoint-2.0/cuda/manifest.json`, sha256 `26f57907e410363cc8745dbdc2c29833160766efd20a2698f989f877f7d11553`

The CUDA image, `src/cuda_readiness`, and its tests remain on that commit. They were not rebuilt here.
