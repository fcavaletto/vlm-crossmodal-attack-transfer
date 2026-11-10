# Checkpoint 5, amendment 1: hardware and storage (2026-11-10, before any CUDA run)

Approved by the author on 2026-11-10 after both EU-SE-1 attempts (A40, RTX A6000) were refused for capacity. Written before any GPU pod exists and before any CUDA result. [`PLAN.md`](PLAN.md) is not edited.

## What changes

| | PLAN.md (§8, §8.1, §8.2) | This amendment |
|---|---|---|
| Card | A40 48 GB (RTX A6000 fallback), Ampere, $0.59/h | **NVIDIA L40S 48 GB, Ada**, Secure Cloud, up to **$1.09/h**. One create attempt. No automatic fallback: the RTX PRO 6000 is **not** authorized. |
| Data center | EU-SE-1, pinned by the network volume | Any. The create names none, and the data center is recorded after creation. |
| Storage | Network volume `0uhv9a501u` at `/workspace` | A **20 GB pod volume disk** at `/workspace` plus a 30 GB container disk. No network volume is created or attached; `0uhv9a501u` is left unchanged. |
| Shutdown | Watchdog v1: delete first | **Watchdog v2** ([`infra/runpod/watchdog.sh`](../../infra/runpod/watchdog.sh)): stop first. A stopped pod is left stopped, with its disk, until its outputs are exported and hash-verified. Termination happens only if the container is still alive after both stop paths. A restarted container exports and never re-runs the workload. Tested on CPU pods: [`infra/runpod/TERMINATION_TESTS.md`](../../infra/runpod/TERMINATION_TESTS.md). |
| Absolute deadline | 8 h | **8 h**, so the GPU costs at most $8.72 at $1.09/h. The runner stops itself 30 min earlier. |
| Checkpoints | none | Record snapshots (no tensors) are offered with `runpodctl send` after the gates and every hour, so a forced termination loses at most about an hour of records. |

## What does not change

- The comparison: bf16 against fp32 on one card.
- The code paths, and the numerical controls: TF32 off, no autocast, deterministic algorithms, eager attention, the fp64 matmul probe.
- The 48 development scenes and inputs (hash-checked against `reference/input_hashes.json`).
- Gates G1–G9 and every threshold, including G4 at 36 GiB, G5 against the MPS bf16 reference, and G9 at a projected **$12**.
- The **$20** authorized compute ceiling, which includes the GPU canary.
- The analysis script, equivalence margins, bootstrap, and readings A–D.

No threshold is changed after any result.

## Known consequence

The result is a statement about Ada (L40S), not about Ampere. The CUDA kernels differ between architectures. That does not change the within-card bf16–fp32 comparison, and G5 still checks CUDA bf16 against MPS bf16 before any attack. The report names the card.
