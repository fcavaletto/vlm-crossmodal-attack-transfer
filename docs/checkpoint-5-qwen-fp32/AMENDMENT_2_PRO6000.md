# Checkpoint 5, amendment 2: RTX PRO 6000, single-pod canary, throughput gate (2026-11-10, before any CUDA run)

Approved by the author on 2026-11-10 after the L40S attempt (amendment 1) was refused for capacity. Written before any GPU pod exists and before any CUDA result. [`PLAN.md`](PLAN.md) and the scientific parts of amendment 1 are unchanged.

## Infrastructure

| | Amendment 1 | This amendment |
|---|---|---|
| Card | L40S 48 GB | **NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB**, Secure Cloud, at most **$2.49/h**. One create attempt. |
| Absolute deadline (watchdog) | 8 h | **4.5 h** after creation, so the GPU costs at most $11.21 |
| Workload deadline (runner) | deadline − 30 min | **4 h** after creation (deadline − 30 min), leaving 30 min for export and shutdown |
| Storage | 20 GB pod volume disk | unchanged; no network volume; `0uhv9a501u` untouched |
| Watchdog | v2, stop-first | unchanged (sha256 `a9ad711e…`). A stopped pod is never deleted automatically. |

## GPU canary, in the same pod, before any scientific step

Only one deployment attempt is authorized, so the canary is the first phase of the same pod ([`infra/runpod/pipeline.sh`](../../infra/runpod/pipeline.sh)):

1. **Blackwell support.**
   - The device's `sm_XY` must be in the pinned torch build's architecture list.
   - A forward and backward kernel smoke test in bf16 and fp32 must pass: conv3d, attention, layer norm, SiLU, GELU.
   - It runs under the run's numerical controls: TF32 off, deterministic algorithms, and the fp64 matmul probe.
2. **Watchdog.**
   - The watchdog is PID 1, its script hash is recorded, and the boot count is 1.
   - The absolute deadline and the time remaining are recorded.
   - **Stop authorization probe:** the pod key sends a deliberately invalid action to `POST /v2/pods/{id}/action` and must get 400/422 (authorized, rejected as invalid), not 401/403. A real stop is not sent, because it would end the only authorized pod. The stop itself was verified on CPU pods and is exercised for real at the end of this run.
3. **Off-pod transfer.**
   - A canary bundle (environment, canary records, watchdog record) is sent with `runpodctl send`, received locally and checked against its sha256 **while the pod runs**.
   - The pod then **holds**, polling GitHub's public API every 90 s, for `infra/runpod/signals/<RUN_ID>` on the research branch.
   - It continues only on `go <sha256 of that bundle>`. Anything else, or 40 min without a signal, ends the workload, and the watchdog then stops the pod.
4. **The predefined gates**, unchanged: G1–G3 and G5–G8 before the main run; G4 and G9 after the first scene.
5. **Measured runtime of one complete paired unit**, the first scene in bf16 then fp32, recorded in `progress.json`.

## Throughput gate (new; infrastructure, not science)

After the first paired scene, the runner projects when the remaining 47 scenes would finish. If that is later than the 4-hour workload deadline, it stops before the second scene. It keeps the first scene's records and reports a throughput failure. This adds a stop condition; it changes no measurement, threshold or analysis.

## Export

- Record snapshots (no tensors) are sent after the canary, after the gates, and hourly during the main run.
- The full output folder, tensors included, is sent when the workload ends, with a SHA256SUMS manifest.
- Each transfer is received and verified locally while the pod is still running.
- The pod is terminated from outside only after the full export is verified. Restarting a stopped pod is not relied on for recovery.

## Unchanged

The protocol, the model weights (sha256 `7de1838c…`), the 48 development scenes, evaluation definitions, gates G1–G9 and their thresholds, the $12 projected-spend gate, the $20 ceiling, and the analysis script.
