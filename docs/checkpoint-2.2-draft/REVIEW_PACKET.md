# Review packet: Checkpoint 2.2 draft and pre-launch CUDA fixes (2026-11-08)

For research lead Federico Cavaletto. Planning, code, and tests only:
- No model weights were loaded, no attack was run, and no pod was created.
- No held-out scene was generated or opened. No frozen protocol was edited.
- Nothing was merged.

## 1. Summary

- **Binary study (Option A).** The 2.1 result stands as pre-registered: H\* cannot be meaningfully estimated (`effective_sample`). It should be written up as a completed stop. The audit's corrections are in §5.
- **New study.** `PROTOCOL_DRAFT.md` proposes H-ΔL, a separately named hypothesis on the continuous target-loss change. It has two co-primary models, matched random-sign as a gate, and helpful → neutral as a secondary. The sample is 128 new groups from a sampled generator, run on the Mac.
- **Two findings change the plan.**
  1. The current scene generator is a closed-form 48-group design with no sampling, so it has no sealed sample to give. A sampled generator (`scene_v5`) must be built first.
  2. The two natural model specifications are each blind to the other's mechanism, so the draft pre-registers both, under a Bonferroni split.
- **Power at 128 groups.** Good for moderate effects (partial R² ≥ 0.05: power about 0.86–0.89). Weak for small ones (R² 0.02: about 0.50). My audit's earlier "0.6–0.9 at R² 0.02" was interpolated and too optimistic.
- **CUDA.** The job can no longer report a failed measurement as done. It has a runtime cap and self-terminates. Weights are now fetched at run time with per-file hashes. No GPU launch is requested.

## 2. Estimated runtime (MacBook Air M4, MPS, bf16)

Measured bases:
- Checkpoint 1.8: 45 s per scene for clean scoring plus both clean gradients.
- Checkpoint 2.1: 159 s per scene for the attack bundle. That bundle includes the step-4–7 continuation (about 23 s) and a gradient recompute.

| Step | Scenes | Per scene | Wall clock |
| --- | ---: | --- | --- |
| Build and test `scene_v5`, generate dev-v5 (CPU) | 64 | — | about half a day of work; rendering takes seconds |
| Dev-v5 clean-accuracy check (clean scoring only) | 64 | about 5–10 s | about 10–15 min with model load |
| Development pipeline pilot (8 v4 groups) | 16 | about 190–250 s | about 1 h |
| Sealed run: clean gradients, PGD-3 neutral, FGSM, random-sign, PGD-3 helpful, float32 readout | 256 | about 190–250 s | **about 13.5–18 h** (two overnight runs, resumable) |
| Frozen analysis (B = 9,999 wild bootstrap, 2 models plus controls) | — | — | minutes |

Total Mac compute is about 15–20 hours. From an approval on 9–10 November, the sealed analysis would be ready around 12–13 November, leaving time to write before 16 November. That holds only if the generator work does not slip. If time runs short, the fallback is 64 groups, which has power 0.58 at R² 0.05. The protocol would have to name that size before rendering, not after.

GPU: not needed for 2.2. The CUDA path stays a separate validation track.

## 3. Decisions requiring your approval

Must be decided before the protocol is frozen:

| # | Decision | Recommendation | Alternative |
| --- | --- | --- | --- |
| D1 | Primary model family | Co-primary (a) additive and (b) first-order-scaled, each at one-sided 0.0125 | (a) primary at 0.025, (b) secondary. More power for additive effects; blind to the first-order mechanism. |
| D2 | Sealed data source | Build `scene_v5`: the v4 constraints with seeded sampling of descriptor, columns, cycle, arm order, and distractors; dev-v4 duplicates rejected; dev-v5 (32 groups) for a clean-accuracy check | Reuse `development_v4` as the analysis sample. Not recommended: it would be development data again. |
| D3 | Sealed size | 128 groups (256 scenes), the proposal's number | 192 groups (about +7 h, power at R² 0.02 rises to about 0.76), or 64 groups if time forces it |
| D4 | Scorer | Frozen scorer (bf16 logits, float32 softmax) primary; float32 final projection as sensitivity C4 | Float32 readout as primary: new code path in the primary measurement |
| D5 | PGD step 7 | Drop it (saves about 1.5 h; it served the binary saturation question) | Keep it as a descriptive comparison |
| D6 | Eligibility tie rule | An exact clean tie is ineligible and counted | Keep the alphabetical rule (favors labels early in the alphabet) |
| D7 | Random-sign gate | A significant positive control coefficient blocks "supported" (proposal §7) | Report the control only |
| D8 | Pre-fit floor | At least 100 groups with an eligible scene, at least 180 eligible scenes, SD(A) ≥ 0.05 | Different numbers, fixed now |
| D9 | Name | "H-ΔL", never "H\*" | — |

Execution gates, each a separate go (each loads weights on the Mac):

| # | Gate |
| --- | --- |
| G1 | Dev-v5 clean-accuracy check |
| G2 | Development pipeline pilot on 16 v4 scenes |
| G3 | Render the sealed split (after the freeze; manifest hash committed before any model run) |
| G4 | Sealed MPS run |

CUDA track, not needed for 2.2:

| # | Decision |
| --- | --- |
| C1 | Local MPS one-scene sidecar measurement (loads weights; needed before any pod run, because the job now refuses to finish without it) |
| C2 | Amend the GPU gate to allow the RTX A6000 (same $0.59/hr as the L4, 48 GB) |
| C3 | Build and publish the image (GitHub Actions, public GHCR package) |
| C4 | Create the pod (billable; worst case about $0.59 with the 60-minute operator guard) |

## 4. Pre-launch CUDA fixes (PR 27, branch `checkpoint-2.1/cuda-deploy-fixes`)

| Item | What changed | Evidence |
| --- | --- | --- |
| Failure handling | The job prints exactly one terminal line. `JOB_DONE status=ok` requires all of:<br>• a weights hash match;<br>• `preflight --require-cuda` exit 0;<br>• `measure` exit 0 with `status: completed` and a gradient file;<br>• an MPS sidecar;<br>• `compare` exit 0 with `differences_recorded`.<br>Anything else prints `JOB_FAILED step=… rc=… reason=…`. The old script printed `JOB_DONE` after any outcome, and `measure` exits 0 even on `stopped_*` statuses. | `tests/cuda_validation/test_run_job.py` runs the real script under bash with a fake `python` and `timeout`: 9 failure paths (none print `JOB_DONE`), the success path, refused self-stop, and the weights check |
| Runtime cap and cleanup | 1. Every step runs under `timeout` against `JOB_DEADLINE_MINUTES` (45).<br>2. A watchdog requests a stop 5 minutes after that.<br>3. In-pod `self_stop.py` calls REST v2 `terminate`, then `stop`, with the pod-scoped key.<br>4. If refused, the job sleeps rather than exiting and rerunning, and prints `SELF_STOP_FAILED`.<br>5. The operator-side `deadline_guard.sh <pod-id> 60` terminates that pod id. | RunPod docs: every pod gets `RUNPOD_POD_ID` and a pod-scoped `RUNPOD_API_KEY`. REST v2 has no stop timer. **Correction:** the first pass called runpodctl `--terminate-after` a cost cap. RunPod documents that the backend never enforced it and v2.12.0 removed it. |
| Weights | Not baked into the image. The pod downloads the 10 files at the pinned revision and refuses to measure unless every sha256 matches `docker/cuda/qwen_files.sha256`. Those hashes were taken from the Mac study cache, and both Mac caches are identical. | The model is Apache-2.0 and not gated, so baking was legal. But it would republish 4.3 GB of third-party weights from your public GHCR. The Hugging Face repo has no LICENSE file to ship alongside them. It would also strain a standard Actions runner's disk. Cost of fetching: a run-time dependency on Hugging Face and about 1–3 min of download (under $0.03). |
| Not verified | Whether the pod-scoped key may `terminate` or only `stop` (either ends GPU billing); actual download time | Needs a live pod |

Tests: 47/47 in `tests/cuda_validation` and `tests/cuda_readiness`.

## 5. Corrections to my own audit (PR 26), and the remaining checks

| Item | Correction or finding |
| --- | --- |
| Ties | There are **9** exact ties in 1,536 decisions, not 6. The label won 6 and lost 3 (`dev4-ref-02-a` FGSM, `dev4-ref-27-b` PGD-7, `dev4-ref-28-b` FGSM). "Every tie went to the label" was wrong. The PGD-3 numbers in the audit (3 ties, all won; 57/91 and 54/57 under "a tie is not the label") are right. Under that rule, PGD-7 is 84/91 and 84/84, and FGSM is 18/91 and 11/18. |
| Float32 | **Upcasting the logits does not address the ties.** The scorer already upcasts the bf16 logit row before `log_softmax` (`src/dev_pilot/dev_pilot/scoring.py:25`), and the ties happened anyway. All 1,536 margins lie on a bf16 grid: 391 on whole nats, 326 on 1/2, 682 on 1/4, 137 on 1/8, none finer. The rounding happens when the final projection writes its output in bf16. Upcasting afterward is bit-identical (`precision_check.py`, synthetic part). Computing the final projection in float32 from the same bf16 inputs removes the grid: 0 ties versus 1,946 in 20,000 synthetic trials. It does **not** restore the bf16 error inside the network, which only a float32 model run could measure. The audit's advice to "upcast the final logits row" is withdrawn. |
| Does the grid matter for H-ΔL? | Barely. An upper bound on the grid's rounding error is SD 0.08 nats per `ΔL`, against an outcome SD of 2.96 nats: under 0.1% of the variance. It matters for decisions near a tie and for scenes whose `ΔL` is near 0. This is why the float32 readout is a sensitivity (C4), not the primary. |
| 128-group power | Was "about 0.6–0.9 at R² 0.02." The dedicated simulation gives about 0.50 (wild bootstrap) at R² 0.02 and about 0.89 at R² 0.05. |
| Stale tests | The `scene_gen` and `scene_v2` failures are obsolete. Their allowlist predates `fixtures/development_v4`, which commit `44aafcf` (Checkpoint 1.7) added with all 96 rows `development`. `development_v4` was added to the allowlist on both branches, and the guard is still strict for any other name. Audit branch: every test directory passes (505 tests). |

## 6. Files

| File | What |
| --- | --- |
| `docs/checkpoint-2.2-draft/PROTOCOL_DRAFT.md` | The full draft protocol |
| `docs/checkpoint-2.2-draft/power_continuous.py`, `.json` | Power simulation (predictors and outcome spread only; seed 20261108) |
| `docs/audit-2026-11-08/precision_check.py`, `.json` | Grid, tie, and float32 analysis (no weights) |
| `docs/audit-2026-11-08/AUDIT_AND_STRATEGY.md` | Dated corrections section added |
| PR 27 branch: `docker/cuda/run_job.sh`, `fetch_weights.py`, `self_stop.py`, `qwen_files.sha256`, `docker/runpod/deadline_guard.sh`, `tests/cuda_validation/test_run_job.py` | CUDA job fixes |
