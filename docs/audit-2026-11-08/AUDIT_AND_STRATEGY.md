# Independent audit and strategy review, 2026-11-08

Reviewer: Claude (Claude Code), for research lead Federico Cavaletto. Planning and audit only. No model weights were loaded, no attack was run, no pod was created, nothing was committed or pushed.

Audited state: `checkpoint-2.1/development-attack` at `fd59f63b04296d99b94438cc2d1697a217918a39` (PR 25), which contains the protocol commit `481d7d9b` (PR 23) and the Checkpoint 2.0 review packet `ccc5895b` (PR 22). `checkpoint-2.1/cuda-validation` at `3177158b` (PR 24) is a separate branch.

## 1. Verdict

The Checkpoint 2.1 result is reproducible from the committed artifacts, and the stopping rule was applied as written. "H* cannot be meaningfully estimated" is the correct, pre-specified conclusion. No count in the report is wrong. Three findings change how the result should be read and what comes next:

1. **Exact ties decide outcomes.** Six of 1,536 decisions are exact NLL ties, all resolved for the label by the pre-specified alphabetical rule. One of the four PGD-3 transfer failures (`dev4-ref-33-b`) and the only PGD-7 failure (`dev4-ref-21-a`) are ties. bfloat16 logits put every margin on a 1/8 grid.
2. **The variation that justified the 3-step attack was mostly in the calibration scenes.** Three of the four transfer failures are among the 32 scenes Checkpoint 2.0 used to choose the configuration. On the 64 scenes that were not used to choose it, conditional transfer is 38/39.
3. **Binary H* could not have been tested at the planned confirmatory size either.** At 128 groups (the proposal's 256 sealed scenes), the simulated chance of reaching 40 transfer failures is about 0 under any plausible prevalence.

## 2. Reproduced counts

From `docs/checkpoint-2.1/development/records.jsonl` (384 rows = 96 scenes × 4 configurations), recomputed with the protocol definitions, not read from `summary.json`:

| Configuration | Source success | Excluded | Conditional transfer | Transfer failures | Target attack success | Groups in denominator |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PGD 1/255, 3 steps | 55/91 | 0 | 51/55 | 4 | 62/95 | 36 |
| FGSM 1/255 | 18/91 | 0 | 10/18 | 8 | 17/95 | 14 |
| Random sign 1/255 | 0/91 | 0 | 0/0 (unset) | 0 | 0/95 | 0 |
| PGD 1/255, step 7 | 84/91 | 0 | 83/84 | 1 | 87/95 | 47 |

All match the report. Other checks:

| Check | Result |
| --- | --- |
| File sha256 (records, summary, analysis, progress, verification, scalars) | match the report |
| Manifest sha256 | `0d9d592b…` matches the protocol; 96 rows, 48 groups of exactly 2, all `development` / `referent_color` |
| Labels and group ids in records vs manifest | identical |
| L-infinity after the 8-bit round trip | max 0.0039215758 ≤ 1/255; no errors; no missing decisions |
| Clean accuracy vs Checkpoint 1.7 | neutral 91/96, helpful 95/96, contradictory 36/96, revealing 96/96, reproduced |
| Checkpoint 2.0 calibration | PGD-3 16/32 source, 13/16 transfer, reproduced |
| 2.0 vs 2.1 on the 32 shared scenes | perturbed decisions and margins bit-identical (MPS is deterministic here) |
| Stored 1.8 predictors | come from committed `docs/checkpoint-1.8/qwen_rgb_loss/progress.json`; alignment mean 0.83072 matches the 1.8 summary |
| Stopping rule | eligible 55, failures 4 < 40 → `effective_sample`. Transfers 51 ≥ 40, mean 0.927 inside the band, rank 5, condition 5.7e9 < 1e12. Only the pre-specified failure is recorded. No coefficient stored. |
| Tests | 560/562 pass offline with no weights. The 2 failures (`scene_gen`, `scene_v2`) are stale assertions that `fixtures/` holds only older sets; `development_v4` was added later. The v4 regeneration and checksum tests pass. |

## 3. Discrepancies and caveats (not corrected, reported)

**3.1 Ties.** The tie rule ("equal sums break by color name") is in the Checkpoint 1.7 protocol, so the counts are right. But it is not neutral: it always favors whichever tied color is earlier in the alphabet. Here every tie went to the label. Under a "tie = not the label" reading, PGD-3 source success would be 57/91 (`dev4-ref-29-b` and `dev4-ref-38-a` join, and both transfer), and conditional transfer would be 54/57 with 3 failures (`dev4-ref-33-b` becomes a transfer). The outcome is even flatter. Future protocols should score in float32 (upcast the final logits row) and state a tie rule that does not depend on the label's name.

**3.2 Calibration overlap.** The Checkpoint 2.1 sample includes the 32 calibration scenes. The protocol says the calibration is prior information, which is honest. But the 3-step setting was chosen because it left variation on those scenes. That variation shrank on new scenes (13/16 → 38/39), a selection effect. Any planning number should use the 64 non-calibration scenes: about 1 failure per 64 scenes, with wide uncertainty.

**3.3 The headline undersells the result.** "Conditional transfer 51/55" reads as "attacks mostly transfer." The fuller statement: the neutral-optimized attack causes errors under helpful text more often (62/95) than under neutral text (55/91). Seven scenes flip only under helpful text; four flip only under neutral. At this budget, the helpful context offers no protection, and the attack effect is close to context-invariant. That fits the high gradient alignment (median 0.86) but is not evidence for H*.

**3.4 Local state.** The main clone that the `_c20-work` and `_c21-work` worktrees pointed to (`…/vlm-crossmodal-attack-transfer/vlm-crossmodal-attack-transfer/.git`) no longer exists. Those folders are now plain files. Their content matches the committed heads, except `_c21-work/development-attack/docs/checkpoint-2.1/development/run.log`, which is gitignored and exists only there. Several older worktrees have uncommitted changes. None were touched. A fresh clone is at `vlm-crossmodal-attack-transfer/repo`.

## 4. Is H* testable under the current design?

Practically, no. The binary outcome needs about 40 failures, and failures are rare because the two contexts are nearly the same input: same image, two non-revealing sentences, alignment 0.83 on average. The predictor has little range, and the outcome has almost none.

`power_sim.py` (results in `power_sim_results.json`) resamples the 48 committed predictor pairs jointly, so correlations and pairing are kept. It adds a group random effect, applies the frozen gate, and judges β_A with a CR1 group-clustered interval as a cheap stand-in for the wild bootstrap. It never reads an attack outcome. Effect sizes are assumptions.

Binary design. P(gate passes) / P(gate passes and the interval excludes 0), effect OR 0.5 per SD of alignment:

| Groups (scenes) | Failure rate 2.6% (new scenes) | 7.3% (all dev) | Rate uncertain, Beta(1.5, 38.5) |
| ---: | --- | --- | --- |
| 48 (96) | 0 / 0 | 0 / 0 | — |
| 200 (400) | 0 / 0 | 0.00 / 0.00 | — |
| 500 (1000) | 0.00 / 0.00 | 0.99 / 0.98 | 0.28 / 0.27 |
| 1000 (2000) | 0.81 / 0.80 | 1 / 1 | 0.59 / 0.59 |
| 2000 (4000) | 1 / 1 | 1 / 1 | 0.86 / 0.86 |

The bottleneck is the gate, not the effect: once 40 failures exist, power is high. At ~159 s per scene on the M4 (2.1 wall clock), 1000 groups is ~88 hours of Mac time plus new scene generation and gradients. That is not achievable by 16 November, and not sensible without a CUDA run that has not been validated.

Continuous design, all source-clean-correct scenes. Power by partial R² of alignment:

| Groups | R² 0.02 | R² 0.05 | R² 0.10 |
| ---: | ---: | ---: | ---: |
| 48 | 0.34 | 0.58 | 0.86 |
| 96 | 0.49 | 0.83–0.88 | 0.99 |
| 150 | 0.69 | 0.96 | 1.00 |
| 250 | 0.87–0.90 | 1.00 | 1.00 |

(Group correlation 0.3–0.6 changed little.)

## 5. Options

| Option | Scientific value | Feasible by 16 Nov | Notes |
| --- | --- | --- | --- |
| A. Close the binary study honestly | High: a pre-registered stop, carried out as written, with full counts | Done; write-up only | Required in any plan. |
| B. New continuous endpoint, new protocol | High: answers the H* question in a form the data can support | Yes, on MPS | Must be a new named endpoint, registered before any new scene is opened. |
| C. Scale the binary design | Low per hour | No (~1000+ groups) | The table shows why. Report it as the reason for B. |
| D. Helpful → neutral direction | Medium: tests asymmetry, which 3.3 makes interesting | Yes, as a secondary in B | Same attack code with the source context swapped; adds roughly one PGD pass per scene. |
| E. Stronger models, other tasks | High long term | No, not safely | Needs CUDA validation, new clean-accuracy gates, new calibration. After the deadline. |

## 6. Recommendation

**A + B, with D as a pre-registered secondary, run on the Mac.** Not because B is more likely to "work," but because it is the only design in reach whose answer is informative either way.

1. **Write up Checkpoint 2.1 as it stands (A).** Lead with the full picture in 3.3, then the stop, then the tie and calibration caveats. That is a complete pre-registered result.
2. **Register Checkpoint 2.2 before rendering any new scene (B + D).**
   - Sample: fresh scenes from a recorded seed rule disjoint from development. The proposal's 256 sealed scenes (128 groups) fit: power ≈ 0.6–0.9 at R² 0.02 and ≈ 0.97 at R² 0.05. No held-out estimate of β_A exists yet, so registering a new endpoint before opening them stays within the proposal's §5 rules. State that explicitly.
   - Primary endpoint: change in the labeled-color NLL under helpful text after the frozen neutral PGD-3 attack, computed in float32 on the official 8-bit image. Use the NLL, not the margin, because the margin sits on a bfloat16 grid.
   - Model: Δ_helpful ~ A + M_neutral + M_helpful + log L1, with wild-cluster intervals by group. Decide in the protocol whether Δ_neutral is a covariate. It is a post-attack quantity, so pick one answer and say why.
   - Secondary: helpful-source PGD-3 evaluated under neutral text. Directional asymmetry = difference in target success between directions, with a group bootstrap.
   - Keep binary transfer as a descriptive secondary. Do not fit it.
   - Run the development pilot of the new endpoint only on the 96 development scenes, labeled development, to debug code. Do not use it to choose the endpoint after seeing it.
3. **Cost.** 256 scenes × ~160–250 s ≈ 11–18 h on the M4. That is two overnight runs, with room before the deadline. No GPU spend is needed.
4. **Defer E.** Finish the PR 24 fixes so CUDA is ready afterward. Plan the A6000 question (§7) as a protocol amendment.

What not to do: refit H* on 2.1 with fewer covariates, Firth, or a lower floor; call the 2.1 continuous NLLs a test of the new endpoint; or rerun 2.1 at another step count to get more failures.

## 7. PR 24 deployment review (fixes on local branch `checkpoint-2.1/cuda-deploy-fixes`, worktree `repo-cuda`, not pushed)

| Blocker | Status |
| --- | --- |
| Weights at `/models` had no provisioning; offline mode blocks download | Fixed: a build stage downloads the pinned revision and checks its sha256. No network volume or paid storage. |
| Scene PNG and 1.7 prompt hashes were not in the image | Fixed |
| Start command `cuda_validation protocol` crashed (missing `docs/checkpoint-2.0/cuda/manifest.json`) | Reproduced, fixed, and covered by a new layout test that fails on the old Dockerfile |
| Pod body used v1/GraphQL field names and had no `image` | Rewritten with v2 fields from the live schema |
| No result export; container disk is wiped; no stop timer | `run_job.sh` prints JSON between markers in the pod log, prints `JOB_DONE`, then sleeps; the operator deletes the pod |
| No Docker on this Mac, no registry | Manual-only GitHub Actions workflow pushing to GHCR (package must be made public, or add a registry credential) |
| Still open | The MPS sidecar must be measured locally first (loads weights; needs your OK). `measure` refuses any GPU but an L4. |

GPU choice, from a live RunPod capacity read on 2026-11-08 (Secure Cloud): L4 24 GB and RTX A6000 48 GB are both $0.59/hr, both `Low` stock on CUDA 13.2 hosts and `Out` on 12.8/13.0. At the same price the A6000 has twice the memory and about 2.5× the memory bandwidth (768 vs 300 GB/s), which usually decides speed for a 2B model at batch 1. It would also fit 7–8B VLMs in bf16 for Option E. It is Ampere (8.6), so bf16 works. The current gate refuses it. Allowing it is a protocol amendment to write before any CUDA measurement. Expected cost of the one-scene validation: under $0.50.

## 8. Corrections, 2026-11-08 (second pass)

Found while preparing the Checkpoint 2.2 draft. The sections above are left as written.

- **§1 and §3.1, ties.** There are 9 exact ties in 1,536 decisions, not 6.
  - The label won 6 and lost 3: `dev4-ref-02-a` FGSM, `dev4-ref-27-b` PGD-7, `dev4-ref-28-b` FGSM. "Every tie went to the label" is wrong.
  - The PGD-3 figures are right: 3 ties, all won by the label, and 57/91 and 54/57 under "a tie is not the label."
  - Under that rule, PGD-7 is 84/91 source and 84/84 transfer, and FGSM is 18/91 and 11/18.
  - All 9 ties are listed in `precision_check.json`.
- **§3.1, float32.** "Upcast the final logits row" would not remove the ties. The scorer already does it (`src/dev_pilot/dev_pilot/scoring.py:25`).
  - The grid is created when the final projection writes its output in bf16. Every recorded margin is a multiple of 1/8, most of 1/4.
  - Removing the grid needs the final projection computed in float32. Even then, the bf16 error inside the network remains.
  - See `precision_check.py`.
- **§4, continuous power at 128 groups.** It was interpolated as "about 0.6–0.9 at R² 0.02." The dedicated simulation (`docs/checkpoint-2.2-draft/power_continuous.py`) gives about 0.50 at R² 0.02 and about 0.89 at R² 0.05, with a wild cluster bootstrap and development-shaped errors.
- **§6.2, sample.** "The proposal's 256 sealed scenes fit" assumed a generator that can draw them. `scene_v4` is a closed-form 48-group design with no sampling, so a sampled generator is needed first. See `docs/checkpoint-2.2-draft/PROTOCOL_DRAFT.md` §4.
- **§2, tests.** The two stale assertions were corrected after confirming they are obsolete: the fixture allowlist predates `development_v4` (commit `44aafcf`). Every test directory on this branch now passes.
