# Checkpoint 5: alternative GPU deployment, for approval (2026-11-10)

**Nothing here has been created.** It is prepared because both approved EU-SE-1 attempts were refused for capacity (see [`DEPLOYMENT_STATUS.md`](DEPLOYMENT_STATUS.md)). Prices and stock below come from live reads made today at about 15:35 UTC, Secure Cloud, CUDA 13.2 hosts. They are re-read at create time.

The scientific protocol does not change: same commit of the runner and analysis, same 48 development scenes, same gates and thresholds (G1–G9), same $12 projected-spend gate and $20 authorized ceiling. Only the card and the storage change.

## 1. Candidates (48 GB or more, Secure Cloud, in stock today)

| Card | Memory | $/h | Where in stock | Assessment |
|---|---|---|---|---|
| **L40S** (Ada) | 48 GB | **1.09** | LOW; the read names no data center, so the create would list none and RunPod places it | **Recommended.** About 2.4× the A6000's non-tensor fp32 throughput, similar memory bandwidth. Expected no slower than the A6000 the plan was costed on. |
| **RTX PRO 6000 Server** (Blackwell) | 96 GB | 2.49 | LOW, US-NC-2 | **Fallback.** Likely the fastest per hour here, but over twice the L40S price. |
| A100 SXM | 80 GB | 1.79 | LOW, US-MD-1 | Not recommended. With TF32 disabled (a required control), fp32 matmuls run on CUDA cores at about 19.5 TFLOPS, half the A6000. The fp32 arm is the longest phase, so the run could cost $5–11 and hit the $12 gate. |
| H100 SXM | 80 GB | 3.99 | LOW, US-GA-2 | Not recommended: about 3 h fits under $12, so little margin. |
| B200 / B300 | 180 / 288 GB | 7.99 / 8.99 | LOW, EU-RO-1 / EU-NL-1 | Not recommended: cost. |
| A40, RTX A6000 (approved) | 48 GB | 0.59 | data-center list says LOW in EU-SE-1; the capacity read says the A40 is unavailable on Secure; both creates refused | Still preferred if capacity returns. |

The throughput figures are vendor peak numbers, not measurements. CUDA speed for this workload is unmeasured on every card; the plan's benchmark scene and G9 projection after the first scene still decide whether the run continues.

## 2. Cost

The plan's estimate is 2.5–6.3 pod-hours at A6000 speed (PLAN §8). I assume the L40S is no slower, and give the PRO 6000 the same hours for a worst case.

| | L40S at $1.09/h | RTX PRO 6000 at $2.49/h |
|---|---|---|
| Expected GPU compute (2.5–6.3 h) | $2.7–6.9 | $6.2–15.7 at A6000-equivalent speed; likely less |
| Hard deadline in the pod (absolute epoch) | **8 h**, so GPU at most **$8.72** | **4.5 h**, so GPU at most **$11.21** |
| Runner stops itself | 30 min before the deadline, records flushed | same |
| Export grace after the workload ends (§3) | 30 min, ≈ $0.55 | 30 min, ≈ $1.25 |
| Pod volume disk, 20 GB | under $0.01 while running; about $0.13 per day if left stopped | same |
| G9 gate | projected total ≤ $12, or the run stops | same |
| Authorized ceiling | $20 | $20 |

Network volume `0uhv9a501u` (EU-SE-1, now 50 GB after your resize, ≈ $3.50 per month) is not used, resized or deleted.

## 3. Temporary storage: what has to change

Without a network volume, the pod's own disk holds the only copy of the results until they are exported. Deleting the pod deletes them. The current watchdog deletes first, which is right with a network volume and wrong here.

1. **Storage.** A 20 GB pod volume disk at `/workspace`, plus a 30 GB container disk. `HF_HOME=/workspace/hf-cache` as before. The model (4.3 GB) is downloaded each time and its sha256 checked (G3). `pipeline.sh` needs no change: it already writes under `/workspace`.
2. **Watchdog order: stop first, delete as the last resort.** On the deadline, or 30 min after the workload ends:
   - REST v2 `stop`, then GraphQL `podStop` (already tested on CPU: 200, EXITED);
   - delete only if the pod is still running 10 min later.
   - A stopped pod ends GPU billing and keeps the volume disk. Deletion stays as the backstop, so the GPU can never run past the deadline plus about 15 min.
   - This is a code change to `infra/runpod/watchdog.sh`, so it needs **a new CPU test (about $0.01)** and the **GPU canary** before the main run.
3. **Export, with me present.** When the run ends, the pipeline writes the SHA256SUMS manifest and starts `runpodctl send` (unchanged). I run `runpodctl receive` locally and verify every hash against the manifest. Only then do I delete the pod from outside, confirm it is gone (404 and `list-pods`), and read its final billing.
4. **If this session is interrupted.** The pod stops itself at the grace or the deadline. GPU billing ends, the results stay on the stopped pod's disk (about $0.13 per day), and nothing is lost. To recover: start it again, or start it with zero GPUs if its GPU is taken, export and verify, then delete. I will not delete a pod whose outputs have not been verified locally unless you approve losing them.

## 4. Technical and scientific implications

- **Unchanged:** the comparison is bf16 against fp32 on one card, with the same code, numerical controls (TF32 off, no autocast, deterministic algorithms, eager attention), gates and thresholds.
- **Different hardware from PLAN §8.** Ada (L40S) or Blackwell (PRO 6000) instead of Ampere. CUDA kernels differ between architectures. That does not affect the within-card comparison, but the report must name the card. A plan amendment recording the card (not any threshold) is committed before the run.
- **G5 still guards the cross-device step.** CUDA bf16 must match the MPS bf16 clean margins (median ≤ 0.25, p95 ≤ 0.75, max ≤ 1.5, ≥ 94/96 chosen). If it does not, the run stops for investigation.
- **G4 stays at 36 GiB** on either card, including the 96 GB one.
- **Blackwell** needs kernels for its architecture. The pinned torch 2.15.0 build bundles CUDA 13.2, which should support it. The canary confirms this (G1) before anything else runs.
- **Placement.** With no network volume, the create does not need to name one data center, which improves the chance of getting a card. The data center is recorded after creation. Inputs are public images and a public model, so placement has no data-protection effect.

## 5. What I need from you

1. Approval of the card: **L40S at $1.09/h** first, **RTX PRO 6000 at $2.49/h** as fallback, or another choice. One create attempt per card, as before.
2. Approval of the storage change (§3) and the watchdog change, including the ≈ $0.01 CPU re-test.
3. Approval of the deadlines: 8 h (L40S), 4.5 h (PRO 6000).
