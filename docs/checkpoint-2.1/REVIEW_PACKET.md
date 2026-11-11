# Checkpoint 2.1 review packet

DEVELOPMENT ONLY. Research lead: Federico Cavaletto. Qwen3-VL-2B-Instruct, bfloat16, eager attention. The development attack ran on MPS. No H* fit. No held-out scenes. No billable GPU. Not merged to main.

This packet integrates the three Checkpoint 2.1 workstreams and then stops. The analysis protocol is commit `481d7d9b9a18e832aad65c78a120b215e50010d0`, sha256 `9a8f28d357803c66ae67924472bdbaaf60d5e92f8387ef04f2d4946c6f47bf7a`. The development report is commit `fd59f63b04296d99b94438cc2d1697a217918a39`, sha256 `1820dedd37033d3edf024056b5aff34b5f034ad02c1452554385d096de8fabc5`. The attack-code commit recorded at the start of that run is `edfdf54e207189f4c6d89a3a9335ab607b73e81e`. `docs/checkpoint-2.1/CUDA_VALIDATION.md` is an unchanged copy from `checkpoint-2.1/cuda-validation` at `3177158b641678f57f0f001e3795af1558eec428`, sha256 `5c4fa5903720a26157f707b2496f47e472a57fffa7e3cbd838a162be5fdd4f57`. Every count below is the count printed in `docs/checkpoint-2.1/DEVELOPMENT_REPORT.md`. This packet does not change those counts and does not change `docs/checkpoint-2.1/ANALYSIS_PROTOCOL.md`.

This branch is `checkpoint-2.1/review-packet`, based on `checkpoint-2.1/development-attack` at `fd59f63b04296d99b94438cc2d1697a217918a39`. Pull requests 23, 24, and 25 stay open.

## Conclusion

The frozen attack is PGD at epsilon 1/255 for 3 steps, step length epsilon/4, started from the clean image. The sample is all 96 `development_v4` scenes. The source context is neutral. Source success is 55/91. Conditional transfer from neutral to `helpful_nonrevealing` is 51/55.

Eligible scenes are 55, eligible conditional transfers are 51, and eligible transfer failures are 4, in 36 counterfactual groups. The development report states the floor as 40 transfer failures and 40 transfers, ten events for each of the four regressors, and at least 6 counterfactual groups. The 4 eligible transfer failures are below that floor. The failed check is `effective_sample`. H* was not estimated. No coefficient was computed.

The pre-specified comparisons, at the same epsilon, are FGSM (source success 18/91, conditional transfer 10/18), matched random-sign (source success 0/91, conditional transfer unset), and step 7 of the same PGD trajectory (source success 84/91, conditional transfer 83/84).

The CUDA validation document recommends one NVIDIA L4 on RunPod Secure Cloud at $0.59 per GPU-hour. That page says "Updated November 4, 2026" and was read on 2026-11-07. No billable GPU was launched.

H* is still unestimated. The one-scene MPS-versus-CUDA measurement is still unrun. Held-out scenes are still unopened. Helpful-to-neutral is still unestimated. The analysis protocol is not rewritten, and this packet does not replace it with a final freeze. No further experiment is added here.

## Frozen attack

The development report ran the configuration the analysis protocol had already locked. The protocol is the authority for that lock. The report records the run: attack status `completed`, 96 of 96 scenes finished, rule `all-development-v4-scenes`, all 48 counterfactual groups, both arms.

| Piece | Locked value |
| --- | --- |
| Model | `Qwen/Qwen3-VL-2B-Instruct`, revision `89644892e4d85e24eaac8bacfd4f463576704203` |
| Weights | bfloat16, eager attention |
| Processor and answer prefix | Checkpoint 1.7 lock. Prefix `<|im_end|>\n<|im_start|>assistant\n`. Bare color continuation. |
| Color token ids | red 1151, blue 12203, green 13250, yellow 27869. Token 2518 is not the continuation. |
| Loss | Summed negative log-likelihood of the labeled color |
| Decision | Lowest summed NLL among red, blue, green, yellow |
| Attack | PGD, epsilon `1/255`, 3 steps, step length epsilon/4 |
| Initialization | The clean image. One deterministic start. No restarts. |
| Source context | `neutral` |
| Official image | Half-to-even 8-bit round trip. Clip to `[0, 1]` and project the L-infinity ball. |
| Memory lines | bfloat16 11,509,447,200 available bytes; float32 20,871,410,752; 6 GiB floor 6,442,450,944 |

`helpful_nonrevealing` is an evaluation context. The primary step count stays 3. Epsilon stays `1/255`. The guard was not lowered.

## Counts

Source success uses scenes whose clean neutral decision is the labeled color and whose perturbed neutral decision was recorded. The numerator is a decision other than the labeled color. Conditional transfer uses source successes whose clean helpful decision is the labeled color and whose perturbed helpful decision was recorded. The numerator is a helpful decision other than the labeled color. A source success whose clean helpful decision is already wrong is excluded. A missing decision stays out of the numerator and the denominator.

| Configuration | Role | Source success | Source missing | Conditional transfer | Excluded | Transfer missing | Target attack success |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pgd|1/255|3` | primary | 55/91 | 0 | 51/55 | 0 | 0 | 62/95 |
| `fgsm|1/255|1` | comparison | 18/91 | 0 | 10/18 | 0 | 0 | 17/95 |
| `random|1/255|1` | comparison | 0/91 | 0 | unset | 0 | 0 | 0/95 |
| `pgd|1/255|7` | comparison | 84/91 | 0 | 83/84 | 0 | 0 | 87/95 |

The scene table in the development report has 96 rows. Five are `not_source_clean`, which is why the source-success denominator is 91: `dev4-ref-20-a`, `dev4-ref-26-a`, `dev4-ref-31-a`, `dev4-ref-32-b`, and `dev4-ref-35-b`. One of those, `dev4-ref-20-a`, is also wrong on the clean helpful decision, which is why the target-attack denominator is 95. That scene is not a source success, and the exclusion column is 0 on every row. Thirty-six rows are `not_source_success`. The conditional-transfer denominator is the other 55.

The four eligible transfer failures are the four `no_transfer` rows:

| Scene | Group | Label | Clean neutral | Clean helpful | PGD3 neutral | PGD3 helpful |
| --- | --- | --- | --- | --- | --- | --- |
| `dev4-ref-00-b` | `cf4-ref-00` | blue | blue | blue | green | blue |
| `dev4-ref-18-b` | `cf4-ref-18` | blue | blue | blue | red | blue |
| `dev4-ref-31-b` | `cf4-ref-31` | blue | blue | blue | green | blue |
| `dev4-ref-33-b` | `cf4-ref-33` | green | green | green | red | green |

The other 51 scenes in the conditional-transfer denominator are `transfer`.

## Group intervals

Each interval resamples the 48 counterfactual groups with replacement, 4000 draws, seed `20261109`, and NumPy's linear quantile. A zero denominator leaves the rate unset. These intervals describe the rates. The development report prints them after the effective-sample stop. They are not a coefficient on alignment.

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

The floor is 40 transfer failures and 40 transfers, ten events for each of the four regressors, and at least 6 counterfactual groups. Failed checks: `effective_sample`. The analysis stop reason is `effective_sample`.

Predictor exclusions among the conditional-transfer denominator: 0 (alignment 0, margins 0, gradient L1 0). A transfer failure with an undefined predictor is not one of the eligible transfer failures.

H* cannot be meaningfully estimated. No coefficient was computed.

Verification passed. Checksums, L-infinity bounds, color decisions, preserved Checkpoint 1.7–2.0 files, and the memory lines were checked again from the saved run. The regression covariate is the stored Checkpoint 1.8 neutral gradient L1. Recomputed clean gradients compared: 96. Missing: 0. Max absolute difference: 0. Median absolute difference: 0. Stored neutral margin versus the recomputed clean margin: max absolute difference 0 over 96 scenes. Stored helpful margin versus the recomputed clean margin: max absolute difference 0 over 96 scenes.

Available bytes entering the load decision: 13466796032. Bfloat16 requires 11,509,447,200 available bytes. Float32 requires 20,871,410,752 and was not loaded. The 6 GiB floor is 6,442,450,944. The guard was not lowered. Attack device: mps. Dtype: torch.bfloat16.

## Comparisons

FGSM, matched random-sign, and step 7 are the comparisons named in the development report and locked in the analysis protocol. The protocol keeps them at the frozen epsilon. A larger count on one of them leaves the frozen configuration in place, and the protocol does not move H* onto a comparison after the primary check has failed.

FGSM at 1/255 is the linear comparison, one positive sign step from the clean image. Source success is 18/91. Conditional transfer is 10/18. Target attack success is 17/95. Exclusions and missing decisions are 0. The group interval for conditional transfer is [0.277778, 0.833333].

Matched random-sign at 1/255 is the sign comparison. Source success is 0/91. Target attack success is 0/95. Conditional transfer is unset, because no scene was in the denominator. The group interval for source success is [0, 0].

Step 7 is the same PGD trajectory, epsilon 1/255 and step length epsilon/4, read at the official image after step 7. Source success is 84/91. Conditional transfer is 83/84. Target attack success is 87/95. Exclusions and missing decisions are 0. The group interval for conditional transfer is [0.962963, 1]. The frozen attack remains the image at step 3.

## RunPod recommendation

`docs/checkpoint-2.1/CUDA_VALIDATION.md` recommends one NVIDIA L4 on RunPod Secure Cloud, at $0.59 per GPU-hour.

The RunPod pricing page, https://www.runpod.io/pricing, says "Updated November 4, 2026". It was read on 2026-11-07. Its structured offer names the L4 per-hour on-demand rate on Runpod Secure Cloud at price 0.59. The same page lists the L4 at 24 GB VRAM, 50 GB RAM, and 12 vCPUs. Cirrun's index, https://cirrun.com/providers/runpod.html, checked 2026-11-07, lists L4 Secure at $0.59 per GPU-hour. The quoted price is that Secure Cloud rate. The Community Cloud L4 rate on both pages is $0.44 per GPU-hour. The pod spec requests Secure Cloud, so $0.44 is a different listing.

No billable GPU was launched. The validation document says so in its opening, and its pod record has `launched`, `billable_gpu_launched`, and `apiCallMade` false, with `doNotPost` true. The protocol command in that checkout reported `billable_gpu_launched: false`, `runpod_api_called: false`, `model_from_pretrained_called: false`, `attacks_run: false`, `held_out_scenes_opened: false`, `executed_on_cuda: false`, `executed_on_mps: false`, and `cuda_numerical_validation: not_executed`. That checkout did not start a pod, did not create a volume, and has no measured bill. The image was not built. `measure` was not invoked on MPS or CUDA.

The copied document describes its own checkout when it says the 96-scene attack was not run there. The development report, on `fd59f63b04296d99b94438cc2d1697a217918a39`, records that the frozen attack finished on MPS for 96 of 96 scenes. Copying the validation document does not rerun either measurement. MPS and CUDA stay separate records. `devices_interchangeable` is false on the plan and on every comparison report. The prepared comparison is one scene, `dev4-ref-00-a`. No sidecar from either device is in this commit.

## What remains unresolved

This packet stops. It does not open a held-out file, rewrite the analysis protocol, fit H*, launch a GPU, or add an experiment.

- H* remains unestimated. Eligible transfer failures are 4, below the floor of 40 transfer failures. Eligible transfers are 51 and eligible groups are 36. The recorded failed check is `effective_sample`. No coefficient was computed. The floor stays the protocol floor. The attack stays `pgd|1/255|3`.
- The MPS-versus-CUDA measurement remains unrun. `cuda_numerical_validation` is `not_executed`. No billable GPU was launched, and this packet does not launch one.
- Held-out evaluation remains unopened. The counted sample is the 96 `development_v4` scenes.
- Helpful-to-neutral remains unestimated. The recorded source is neutral, and `helpful_nonrevealing` is the evaluation context. The protocol places a helpful-source attack outside this protocol.
- A final protocol freeze remains unissued. `docs/checkpoint-2.1/ANALYSIS_PROTOCOL.md` stays the authority for the outcome, the predictors, the clustering, the rare-event rule, and the effective-sample check. This packet does not edit it.

The branch does not merge to main. Pull requests 23, 24, and 25 stay open.

## Files

Development report, at `fd59f63b04296d99b94438cc2d1697a217918a39`:

- `docs/checkpoint-2.1/DEVELOPMENT_REPORT.md`, sha256 `1820dedd37033d3edf024056b5aff34b5f034ad02c1452554385d096de8fabc5`
- `docs/checkpoint-2.1/ANALYSIS_PROTOCOL.md`, sha256 `9a8f28d357803c66ae67924472bdbaaf60d5e92f8387ef04f2d4946c6f47bf7a`
- `docs/checkpoint-2.1/development/progress.json`, sha256 `a3c7f676173a32392bb7c7790c182332658a2718f8bf2b07408b835db44d0dc6`
- `docs/checkpoint-2.1/development/records.jsonl`, sha256 `090652eb98af3eaad959b1b84046851067a03e3d3bfd3b8c7a75fefd4207a91d`
- `docs/checkpoint-2.1/development/summary.json`, sha256 `b1bdf4d41673bd0d8bdc2191e60789b36c6e5fe2ffa55a7aa3d7d0a684250391`
- `docs/checkpoint-2.1/development/checkpoint_1_8_scalars.json`, sha256 `03ba5ef14a1715a5260995acd0d6725aa12a69b8553e3b520bca18423a4a457e`
- `docs/checkpoint-2.1/development/analysis.json`, sha256 `ba6610e8d1d22d84fcb13cb4586cdb7d2136498444bdd08bd3e0ca7ba1f85e7c`
- `docs/checkpoint-2.1/development/verification.json`, sha256 `bec34d49f2c4663032e0876c7a8d65ba91b736126d2ef9e4ea668abec8c5889f`

CUDA validation document, copied unchanged from `3177158b641678f57f0f001e3795af1558eec428`:

- `docs/checkpoint-2.1/CUDA_VALIDATION.md`, sha256 `5c4fa5903720a26157f707b2496f47e472a57fffa7e3cbd838a162be5fdd4f57`

The CUDA validation code, image, and tests remain on that commit. They were not rebuilt here.
