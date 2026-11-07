# Checkpoint 2.0 calibration report

DEVELOPMENT ONLY. Research lead: Federico Cavaletto. Qwen3-VL-2B-Instruct, bfloat16, MPS. No H* fit. No held-out scenes. The sample is exploratory. It is not a population attack-success rate.

Protocol commit: `efc7d745546152a81b4cc21a021bbe641162c498`.
Protocol sha256: `339d95a75de7157eaf0d2beb3e58a78a06666695c3ddc6009885e4d25ef37137`.
Status: `completed`.
Stop reason: The listed budgets and recorded steps finished. Epsilon was not raised and no step was added.

## Sample

Rule `counterfactual-group-stride-3`. Group count on the development manifest: 48. Indexes 0, 3, 6, ..., 45 when that count is 48. Both arms. The rule reads group id and scene id. It does not read alignment, margin, loss, or a Checkpoint 1.9 outcome.

Scenes finished: 32 of 32.

## Counts

Source context is neutral. The helpful context is evaluation only. The stored attack is the official 8-bit image. Source success uses scenes whose clean neutral decision is the labeled color and whose perturbed neutral decision was recorded. Conditional transfer uses source successes that were also clean-correct under helpful text. Scenes already wrong on that clean helpful answer are excluded and counted, and they are not transfer failures. A missing decision stays out of the numerator and the denominator.

Near the floor means the numerator is 0 or 1. Near the ceiling means the numerator is the denominator or one less. A denominator below 2 is undefined. Adequate source success is an interior or near-ceiling source-success count. Variation in conditional transfer is an interior conditional-transfer count. Those labels use the counts only.

| Configuration | Source success | Source missing | Conditional transfer | Excluded | Transfer missing | Target flips | Position |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `pgd|1/255|2` | 4/32 | 0 | 0/4 | 0 | 0 | 1/32 | no; source interior; transfer near_floor |
| `pgd|1/255|3` | 16/32 | 0 | 13/16 | 0 | 0 | 15/32 | adequate source, transfer varies; source interior; transfer interior |
| `pgd|1/255|4` | 19/32 | 0 | 15/19 | 0 | 0 | 15/32 | adequate source, transfer varies; source interior; transfer interior |
| `pgd|1/255|5` | 25/32 | 0 | 24/25 | 0 | 0 | 24/32 | no; source interior; transfer near_ceiling |
| `pgd|1/255|7` | 28/32 | 0 | 27/28 | 0 | 0 | 27/32 | no; source interior; transfer near_ceiling |
| `fgsm|1/255|1` | 7/32 | 0 | 2/7 | 0 | 0 | 2/32 | adequate source, transfer varies; source interior; transfer interior |
| `random|1/255|1` | 0/32 | 0 | 0/0 | 0 | 0 | 0/32 | no; source near_floor; transfer undefined |
| `pgd|2/255|2` | 15/32 | 0 | 12/15 | 0 | 0 | 12/32 | adequate source, transfer varies; source interior; transfer interior |
| `pgd|2/255|3` | 18/32 | 0 | 16/18 | 0 | 0 | 16/32 | adequate source, transfer varies; source interior; transfer interior |
| `pgd|2/255|4` | 25/32 | 0 | 24/25 | 0 | 0 | 24/32 | no; source interior; transfer near_ceiling |
| `pgd|2/255|5` | 26/32 | 0 | 26/26 | 0 | 0 | 26/32 | no; source interior; transfer near_ceiling |
| `pgd|2/255|7` | 28/32 | 0 | 27/28 | 0 | 0 | 27/32 | no; source interior; transfer near_ceiling |
| `fgsm|2/255|1` | 6/32 | 0 | 2/6 | 0 | 0 | 2/32 | adequate source, transfer varies; source interior; transfer interior |
| `random|2/255|1` | 0/32 | 0 | 0/0 | 0 | 0 | 0/32 | no; source near_floor; transfer undefined |

## Judgment

Configurations with adequate source success and variation in conditional transfer: `pgd|1/255|3` (16/32 source success, 13/16 conditional transfer), `pgd|1/255|4` (19/32 source success, 15/19 conditional transfer), `fgsm|1/255|1` (7/32 source success, 2/7 conditional transfer), `pgd|2/255|2` (15/32 source success, 12/15 conditional transfer), `pgd|2/255|3` (18/32 source success, 16/18 conditional transfer), `fgsm|2/255|1` (6/32 source success, 2/6 conditional transfer). The list is not ranked, and alignment was not used to make it. This report does not pick one of them. Adequate source and transfer near the ceiling: `pgd|1/255|5`, `pgd|1/255|7`, `pgd|2/255|4`, `pgd|2/255|5`, `pgd|2/255|7`. Adequate source and transfer near the floor: `pgd|1/255|2`. Source near the floor: `random|1/255|1`, `random|2/255|1`. Undefined: `random|1/255|1`, `random|2/255|1`.

Epsilon stayed inside {1/255, 2/255}. The recorded PGD steps stayed 2, 3, 4, 5, and 7. The objective stayed the summed labeled-color NLL under neutral.

## Stored Checkpoint 1.8 scalars

Alignment is A(neutral, helpful). Margins are the stored clean margins. Gradient L1 is the stored neutral gradient. The JSON file holds the stored floats. This table rounds them for reading. The columns were not used to choose the sample or a configuration, and they were not regressed on transfer.

| Scene | Label | Alignment | Neutral margin | Helpful margin | Neutral gradient L1 |
| --- | --- | ---: | ---: | ---: | ---: |
| `dev4-ref-00-a` | red | 0.911022 | 8.7500 | 7.0000 | 72.2 |
| `dev4-ref-00-b` | blue | 0.938229 | 8.7500 | 8.2500 | 195.0 |
| `dev4-ref-03-a` | red | 0.877301 | 8.7500 | 5.5000 | 221.4 |
| `dev4-ref-03-b` | blue | 0.804888 | 10.2500 | 7.5000 | 102.4 |
| `dev4-ref-06-a` | blue | 0.771947 | 4.0000 | 6.0000 | 1216.8 |
| `dev4-ref-06-b` | red | 0.905600 | 3.0000 | 6.0000 | 3192.6 |
| `dev4-ref-09-a` | blue | 0.895374 | 1.5000 | 2.5000 | 10168.7 |
| `dev4-ref-09-b` | red | 0.934537 | 1.5000 | 2.2500 | 9840.8 |
| `dev4-ref-12-a` | blue | 0.900190 | 9.7500 | 8.0000 | 85.5 |
| `dev4-ref-12-b` | green | 0.758426 | 9.5000 | 9.2500 | 180.9 |
| `dev4-ref-15-a` | blue | 0.803849 | 13.0000 | 12.7500 | 20.3 |
| `dev4-ref-15-b` | green | 0.783980 | 12.5000 | 12.5000 | 69.4 |
| `dev4-ref-18-a` | green | 0.914492 | 6.2500 | 6.5000 | 327.8 |
| `dev4-ref-18-b` | blue | 0.860706 | 6.2500 | 7.2500 | 378.1 |
| `dev4-ref-21-a` | green | 0.937235 | 9.2500 | 9.7500 | 237.2 |
| `dev4-ref-21-b` | blue | 0.952229 | 7.0000 | 7.0000 | 377.5 |
| `dev4-ref-24-a` | green | 0.881007 | 9.0000 | 9.0000 | 376.0 |
| `dev4-ref-24-b` | yellow | 0.781712 | 8.0000 | 8.7500 | 460.4 |
| `dev4-ref-27-a` | green | 0.898169 | 12.2500 | 11.7500 | 805.1 |
| `dev4-ref-27-b` | yellow | 0.824435 | 10.2500 | 9.6250 | 863.7 |
| `dev4-ref-30-a` | yellow | 0.926527 | 5.2500 | 7.2500 | 889.9 |
| `dev4-ref-30-b` | green | 0.587596 | 6.5000 | 7.7500 | 643.9 |
| `dev4-ref-33-a` | yellow | 0.956037 | 3.2500 | 5.0000 | 3724.3 |
| `dev4-ref-33-b` | green | 0.780514 | 4.2500 | 5.0000 | 1447.1 |
| `dev4-ref-36-a` | yellow | 0.660850 | 3.5000 | 5.5000 | 3291.7 |
| `dev4-ref-36-b` | red | 0.932284 | 3.5000 | 3.7500 | 3074.6 |
| `dev4-ref-39-a` | yellow | 0.911181 | 7.7500 | 6.2500 | 1479.8 |
| `dev4-ref-39-b` | red | 0.786318 | 11.2500 | 8.8750 | 79.2 |
| `dev4-ref-42-a` | red | 0.703198 | 11.7500 | 10.5000 | 140.3 |
| `dev4-ref-42-b` | yellow | 0.848252 | 7.0000 | 5.7500 | 210.6 |
| `dev4-ref-45-a` | red | 0.823958 | 11.5000 | 10.7500 | 418.2 |
| `dev4-ref-45-b` | yellow | 0.975363 | 8.6250 | 6.8750 | 1561.3 |

## Memory

Available bytes entering the load decision: 12419497984. Fresh reading at the decision: 12419497984. Bfloat16 requires 11,509,447,200 available bytes. Float32 requires 20,871,410,752 and was not loaded. The 6 GiB floor is 6,442,450,944. The guard was not lowered.

Attack device: MPS. Dtype: torch.bfloat16. Scene wall-clock sum: 10023.147 seconds. This process wall clock: 10033.227 seconds. These times are from the MPS run. A CPU timing is not included.

Peak RSS mebibytes: 604.1. Peak MPS current allocation mebibytes: 4072.9. Peak MPS driver allocation mebibytes: 4749.7. Lowest OS-available bytes during the run: 7608238080.

Source context: `neutral`. Evaluation contexts: `neutral`, `helpful_nonrevealing`.

## Files

- `docs/checkpoint-2.0/calibration/checkpoint_1_8_description.json` sha256 `79dcde633a640e8c8a31ce88d3d1503cb6f76fc48b8d09625ec1890a266f9f61`
- `docs/checkpoint-2.0/calibration/progress.json` sha256 `4572ffe1463cadb47dad5b4708adfb7c603b2df8f4e6e73e5bc4bc7608cd8ca5`
- `docs/checkpoint-2.0/calibration/records.jsonl` sha256 `1669eefd11ba4703d09ec746078e7be6a52b25ff55e35a8d0b64958eb5a7122f`
- `docs/checkpoint-2.0/calibration/summary.json` sha256 `bb2f9408c86c9879e23acda6f3a0533e3849e66f9db0f0270089ffae96c98574`
