# Checkpoint 2.1 development report

DEVELOPMENT ONLY. Research lead: Federico Cavaletto. Qwen3-VL-2B-Instruct, bfloat16, MPS. The sample is all 96 development_v4 scenes.

Protocol commit: `481d7d9b9a18e832aad65c78a120b215e50010d0`.
Protocol sha256: `9a8f28d357803c66ae67924472bdbaaf60d5e92f8387ef04f2d4946c6f47bf7a`.
Attack code commit at the start of the run: `edfdf54e207189f4c6d89a3a9335ab607b73e81e`.
Attack status: `completed`.
Analysis stop reason: `effective_sample`.

H* cannot be meaningfully estimated.

## Sample

Rule `all-development-v4-scenes`. All 48 counterfactual groups, both arms. Source context is neutral. `helpful_nonrevealing` is evaluation only. The attack is PGD at epsilon 1/255 for 3 steps, step length epsilon/4, started from the clean image. FGSM at 1/255, matched random-sign at 1/255, and step 7 of the same trajectory are the pre-specified comparisons.

Scenes finished: 96 of 96.

## Counts

Source success uses scenes whose clean neutral decision is the labeled color and whose perturbed neutral decision was recorded. The numerator is a decision other than the labeled color. Conditional transfer uses source successes whose clean helpful decision is the labeled color and whose perturbed helpful decision was recorded. The numerator is a helpful decision other than the labeled color. A source success whose clean helpful decision is already wrong is excluded. A missing decision stays out of the numerator and the denominator.

| Configuration | Role | Source success | Source missing | Conditional transfer | Excluded | Transfer missing | Target attack success |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pgd|1/255|3` | primary | 55/91 | 0 | 51/55 | 0 | 0 | 62/95 |
| `fgsm|1/255|1` | comparison | 18/91 | 0 | 10/18 | 0 | 0 | 17/95 |
| `random|1/255|1` | comparison | 0/91 | 0 | unset | 0 | 0 | 0/95 |
| `pgd|1/255|7` | comparison | 84/91 | 0 | 83/84 | 0 | 0 | 87/95 |

## Group intervals

Each interval resamples the 48 counterfactual groups with replacement, 4000 draws, seed `20261109`, and NumPy's linear quantile. A zero denominator leaves the rate unset.

| Configuration | Rate | Estimate | 95% interval |
| --- | --- | ---: | --- |
| `pgd|1/255|3` | source_success | 0.604396 | [0.483516, 0.722222] |
| `pgd|1/255|3` | conditional_transfer | 0.927273 | [0.851037, 0.983333] |
| `pgd|1/255|3` | target_attack_success | 0.652632 | [0.521277, 0.763441] |
| `fgsm|1/255|1` | source_success | 0.197802 | [0.10989, 0.297872] |
| `fgsm|1/255|1` | conditional_transfer | 0.555556 | [0.277778, 0.833333] |
| `fgsm|1/255|1` | target_attack_success | 0.178947 | [0.0967742, 0.270833] |
| `random|1/255|1` | source_success | 0 | [0, 0] |
| `random|1/255|1` | conditional_transfer | unset | unset (no scene was in the denominator) |
| `random|1/255|1` | target_attack_success | 0 | [0, 0] |
| `pgd|1/255|7` | source_success | 0.923077 | [0.858696, 0.978022] |
| `pgd|1/255|7` | conditional_transfer | 0.988095 | [0.962963, 1] |
| `pgd|1/255|7` | target_attack_success | 0.915789 | [0.852632, 0.968429] |

## Effective sample

Eligible scenes: 55. Eligible conditional transfers: 51. Eligible transfer failures: 4. Eligible groups: 36.

The floor is 40 transfer failures and 40 transfers, ten events for each of the four regressors, and at least 6 counterfactual groups.

Failed checks: `effective_sample`.

Predictor exclusions among the conditional-transfer denominator: 0 (alignment 0, margins 0, gradient L1 0). A transfer failure with an undefined predictor is not one of the eligible transfer failures.

## H*

H* cannot be meaningfully estimated.
No coefficient was computed.

## Verification

Verification passed: True. Checksums, L-infinity bounds, color decisions, preserved Checkpoint 1.7–2.0 files, and the memory lines are checked again from the saved run.

The regression covariate is the stored Checkpoint 1.8 neutral gradient L1. Recomputed clean gradients compared: 96. Missing: 0. Max absolute difference: 0. Median absolute difference: 0.
Stored neutral_margin versus the recomputed clean margin: max absolute difference 0 over 96 scenes.
Stored helpful_margin versus the recomputed clean margin: max absolute difference 0 over 96 scenes.

## Memory

Available bytes entering the load decision: 13466796032. Fresh reading at the decision: 13466796032. Bfloat16 requires 11,509,447,200 available bytes. Float32 requires 20,871,410,752 and was not loaded. The 6 GiB floor is 6,442,450,944. The guard was not lowered.

Attack device: mps. Dtype: torch.bfloat16. Scene wall-clock sum: 15249.172 seconds. Recorded wall clock: 15262.352 seconds.

Peak RSS mebibytes: 606.3. Peak MPS current allocation mebibytes: 4072.9. Peak MPS driver allocation mebibytes: 4749.7. Lowest OS-available bytes during the run: 7321272320.

## Scenes

Primary decisions are the official 8-bit image after 3 PGD steps. Status is the conditional-transfer status for that image.

| Scene | Group | Label | Clean neutral | Clean helpful | PGD3 neutral | PGD3 helpful | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `dev4-ref-00-a` | `cf4-ref-00` | red | red | red | green | green | transfer |
| `dev4-ref-00-b` | `cf4-ref-00` | blue | blue | blue | green | blue | no_transfer |
| `dev4-ref-01-a` | `cf4-ref-01` | red | red | red | red | red | not_source_success |
| `dev4-ref-01-b` | `cf4-ref-01` | green | green | green | green | green | not_source_success |
| `dev4-ref-02-a` | `cf4-ref-02` | red | red | red | green | green | transfer |
| `dev4-ref-02-b` | `cf4-ref-02` | blue | blue | blue | green | green | transfer |
| `dev4-ref-03-a` | `cf4-ref-03` | red | red | red | red | yellow | not_source_success |
| `dev4-ref-03-b` | `cf4-ref-03` | blue | blue | blue | blue | yellow | not_source_success |
| `dev4-ref-04-a` | `cf4-ref-04` | red | red | red | yellow | yellow | transfer |
| `dev4-ref-04-b` | `cf4-ref-04` | green | green | green | green | green | not_source_success |
| `dev4-ref-05-a` | `cf4-ref-05` | red | red | red | yellow | yellow | transfer |
| `dev4-ref-05-b` | `cf4-ref-05` | blue | blue | blue | blue | yellow | not_source_success |
| `dev4-ref-06-a` | `cf4-ref-06` | blue | blue | blue | green | green | transfer |
| `dev4-ref-06-b` | `cf4-ref-06` | red | red | red | green | green | transfer |
| `dev4-ref-07-a` | `cf4-ref-07` | green | green | green | yellow | yellow | transfer |
| `dev4-ref-07-b` | `cf4-ref-07` | red | red | red | yellow | yellow | transfer |
| `dev4-ref-08-a` | `cf4-ref-08` | blue | blue | blue | green | green | transfer |
| `dev4-ref-08-b` | `cf4-ref-08` | red | red | red | green | green | transfer |
| `dev4-ref-09-a` | `cf4-ref-09` | blue | blue | blue | yellow | yellow | transfer |
| `dev4-ref-09-b` | `cf4-ref-09` | red | red | red | yellow | yellow | transfer |
| `dev4-ref-10-a` | `cf4-ref-10` | green | green | green | yellow | yellow | transfer |
| `dev4-ref-10-b` | `cf4-ref-10` | red | red | red | blue | blue | transfer |
| `dev4-ref-11-a` | `cf4-ref-11` | blue | blue | blue | green | yellow | transfer |
| `dev4-ref-11-b` | `cf4-ref-11` | red | red | red | yellow | yellow | transfer |
| `dev4-ref-12-a` | `cf4-ref-12` | blue | blue | blue | blue | blue | not_source_success |
| `dev4-ref-12-b` | `cf4-ref-12` | green | green | green | green | green | not_source_success |
| `dev4-ref-13-a` | `cf4-ref-13` | green | green | green | green | green | not_source_success |
| `dev4-ref-13-b` | `cf4-ref-13` | blue | blue | blue | blue | blue | not_source_success |
| `dev4-ref-14-a` | `cf4-ref-14` | blue | blue | blue | blue | blue | not_source_success |
| `dev4-ref-14-b` | `cf4-ref-14` | yellow | yellow | yellow | yellow | yellow | not_source_success |
| `dev4-ref-15-a` | `cf4-ref-15` | blue | blue | blue | blue | blue | not_source_success |
| `dev4-ref-15-b` | `cf4-ref-15` | green | green | green | green | green | not_source_success |
| `dev4-ref-16-a` | `cf4-ref-16` | green | green | green | red | red | transfer |
| `dev4-ref-16-b` | `cf4-ref-16` | blue | blue | blue | blue | red | not_source_success |
| `dev4-ref-17-a` | `cf4-ref-17` | blue | blue | blue | blue | blue | not_source_success |
| `dev4-ref-17-b` | `cf4-ref-17` | yellow | yellow | yellow | yellow | yellow | not_source_success |
| `dev4-ref-18-a` | `cf4-ref-18` | green | green | green | green | green | not_source_success |
| `dev4-ref-18-b` | `cf4-ref-18` | blue | blue | blue | red | blue | no_transfer |
| `dev4-ref-19-a` | `cf4-ref-19` | blue | blue | blue | red | red | transfer |
| `dev4-ref-19-b` | `cf4-ref-19` | green | green | green | red | red | transfer |
| `dev4-ref-20-a` | `cf4-ref-20` | yellow | green | green | green | green | not_source_clean |
| `dev4-ref-20-b` | `cf4-ref-20` | blue | blue | blue | green | green | transfer |
| `dev4-ref-21-a` | `cf4-ref-21` | green | green | green | green | green | not_source_success |
| `dev4-ref-21-b` | `cf4-ref-21` | blue | blue | blue | red | red | transfer |
| `dev4-ref-22-a` | `cf4-ref-22` | blue | blue | blue | red | red | transfer |
| `dev4-ref-22-b` | `cf4-ref-22` | green | green | green | red | red | transfer |
| `dev4-ref-23-a` | `cf4-ref-23` | yellow | yellow | yellow | green | green | transfer |
| `dev4-ref-23-b` | `cf4-ref-23` | blue | blue | blue | green | green | transfer |
| `dev4-ref-24-a` | `cf4-ref-24` | green | green | green | green | green | not_source_success |
| `dev4-ref-24-b` | `cf4-ref-24` | yellow | yellow | yellow | yellow | yellow | not_source_success |
| `dev4-ref-25-a` | `cf4-ref-25` | blue | blue | blue | blue | blue | not_source_success |
| `dev4-ref-25-b` | `cf4-ref-25` | yellow | yellow | yellow | yellow | yellow | not_source_success |
| `dev4-ref-26-a` | `cf4-ref-26` | yellow | red | yellow | red | red | not_source_clean |
| `dev4-ref-26-b` | `cf4-ref-26` | green | green | green | red | red | transfer |
| `dev4-ref-27-a` | `cf4-ref-27` | green | green | green | green | green | not_source_success |
| `dev4-ref-27-b` | `cf4-ref-27` | yellow | yellow | yellow | yellow | yellow | not_source_success |
| `dev4-ref-28-a` | `cf4-ref-28` | blue | blue | blue | green | green | transfer |
| `dev4-ref-28-b` | `cf4-ref-28` | yellow | yellow | yellow | green | green | transfer |
| `dev4-ref-29-a` | `cf4-ref-29` | yellow | yellow | yellow | red | red | transfer |
| `dev4-ref-29-b` | `cf4-ref-29` | green | green | green | green | red | not_source_success |
| `dev4-ref-30-a` | `cf4-ref-30` | yellow | yellow | yellow | red | red | transfer |
| `dev4-ref-30-b` | `cf4-ref-30` | green | green | green | green | green | not_source_success |
| `dev4-ref-31-a` | `cf4-ref-31` | yellow | green | yellow | green | green | not_source_clean |
| `dev4-ref-31-b` | `cf4-ref-31` | blue | blue | blue | green | blue | no_transfer |
| `dev4-ref-32-a` | `cf4-ref-32` | green | green | green | blue | blue | transfer |
| `dev4-ref-32-b` | `cf4-ref-32` | yellow | blue | yellow | blue | blue | not_source_clean |
| `dev4-ref-33-a` | `cf4-ref-33` | yellow | yellow | yellow | red | red | transfer |
| `dev4-ref-33-b` | `cf4-ref-33` | green | green | green | red | green | no_transfer |
| `dev4-ref-34-a` | `cf4-ref-34` | yellow | yellow | yellow | green | green | transfer |
| `dev4-ref-34-b` | `cf4-ref-34` | blue | blue | blue | green | green | transfer |
| `dev4-ref-35-a` | `cf4-ref-35` | green | green | green | blue | blue | transfer |
| `dev4-ref-35-b` | `cf4-ref-35` | yellow | blue | yellow | blue | blue | not_source_clean |
| `dev4-ref-36-a` | `cf4-ref-36` | yellow | yellow | yellow | green | green | transfer |
| `dev4-ref-36-b` | `cf4-ref-36` | red | red | red | green | green | transfer |
| `dev4-ref-37-a` | `cf4-ref-37` | yellow | yellow | yellow | yellow | yellow | not_source_success |
| `dev4-ref-37-b` | `cf4-ref-37` | red | red | red | red | red | not_source_success |
| `dev4-ref-38-a` | `cf4-ref-38` | green | green | green | green | yellow | not_source_success |
| `dev4-ref-38-b` | `cf4-ref-38` | red | red | red | yellow | yellow | transfer |
| `dev4-ref-39-a` | `cf4-ref-39` | yellow | yellow | yellow | blue | green | transfer |
| `dev4-ref-39-b` | `cf4-ref-39` | red | red | red | blue | green | transfer |
| `dev4-ref-40-a` | `cf4-ref-40` | yellow | yellow | yellow | blue | blue | transfer |
| `dev4-ref-40-b` | `cf4-ref-40` | red | red | red | blue | blue | transfer |
| `dev4-ref-41-a` | `cf4-ref-41` | green | green | green | yellow | yellow | transfer |
| `dev4-ref-41-b` | `cf4-ref-41` | red | red | red | red | red | not_source_success |
| `dev4-ref-42-a` | `cf4-ref-42` | red | red | red | red | red | not_source_success |
| `dev4-ref-42-b` | `cf4-ref-42` | yellow | yellow | yellow | green | green | transfer |
| `dev4-ref-43-a` | `cf4-ref-43` | red | red | red | blue | blue | transfer |
| `dev4-ref-43-b` | `cf4-ref-43` | yellow | yellow | yellow | blue | blue | transfer |
| `dev4-ref-44-a` | `cf4-ref-44` | red | red | red | yellow | yellow | transfer |
| `dev4-ref-44-b` | `cf4-ref-44` | green | green | green | yellow | yellow | transfer |
| `dev4-ref-45-a` | `cf4-ref-45` | red | red | red | red | red | not_source_success |
| `dev4-ref-45-b` | `cf4-ref-45` | yellow | yellow | yellow | yellow | yellow | not_source_success |
| `dev4-ref-46-a` | `cf4-ref-46` | red | red | red | red | red | not_source_success |
| `dev4-ref-46-b` | `cf4-ref-46` | yellow | yellow | yellow | blue | blue | transfer |
| `dev4-ref-47-a` | `cf4-ref-47` | red | red | red | red | yellow | not_source_success |
| `dev4-ref-47-b` | `cf4-ref-47` | green | green | green | yellow | yellow | transfer |

## Files

- `docs/checkpoint-2.1/development/progress.json` sha256 `a3c7f676173a32392bb7c7790c182332658a2718f8bf2b07408b835db44d0dc6`
- `docs/checkpoint-2.1/development/records.jsonl` sha256 `090652eb98af3eaad959b1b84046851067a03e3d3bfd3b8c7a75fefd4207a91d`
- `docs/checkpoint-2.1/development/summary.json` sha256 `b1bdf4d41673bd0d8bdc2191e60789b36c6e5fe2ffa55a7aa3d7d0a684250391`
- `docs/checkpoint-2.1/development/checkpoint_1_8_scalars.json` sha256 `03ba5ef14a1715a5260995acd0d6725aa12a69b8553e3b520bca18423a4a457e`
- `docs/checkpoint-2.1/development/analysis.json` sha256 `ba6610e8d1d22d84fcb13cb4586cdb7d2136498444bdd08bd3e0ca7ba1f85e7c`
- `docs/checkpoint-2.1/development/verification.json` sha256 `bec34d49f2c4663032e0876c7a8d65ba91b736126d2ef9e4ea668abec8c5889f`

## Reproduction

`HF_HOME=/Users/fedecava/Development/models/huggingface PYTHONPATH=src/development_attack:src/attack_calibration:src/attack_feasibility:src/qwen_rgb_loss:src/qwen_rgb:src/choice_eval:src/vlm_iface:src/pilot:src/dev_pilot:src/analysis:src/scene_v4 /Users/fedecava/Development/_c0-work/venv-vlm/bin/python -m development_attack`
