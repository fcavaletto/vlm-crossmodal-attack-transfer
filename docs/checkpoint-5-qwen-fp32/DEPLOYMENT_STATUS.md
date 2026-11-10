# Checkpoint 5 deployment status (2026-11-10, updated 15:40 UTC)

**Still blocked on GPU capacity. No GPU pod exists; no research computation has run on CUDA.** An alternative is prepared for approval in [`ALTERNATIVE_DEPLOYMENT.md`](ALTERNATIVE_DEPLOYMENT.md).

## First attempt (14:10 UTC)

| Step | State |
|---|---|
| CPU termination tests | Passed after one redesign. See [`infra/runpod/TERMINATION_TESTS.md`](../../infra/runpod/TERMINATION_TESTS.md). Every test pod was confirmed gone from outside. |
| Network volume | `0uhv9a501u` (vlm-research-eu-se-1): 30 GB, STANDARD, EU-SE-1. Created and read back. **Billing: ≈ $2.10 per month until deleted.** |
| Fresh availability, EU-SE-1, Secure, CUDA ≥ 13.2 | A40: not listed. RTX A6000: LOW, $0.59/h. |
| GPU canary | `create-pod` for RTX A6000 in EU-SE-1 with the volume refused twice: "no longer any instances available". Stock re-read between attempts (still LOW). Not retried further. |
| Other data centers with network storage and a 48 GB card (same read) | None. 32 GB RTX 5090 in EU-RO-1; H200 in EUR-IS-4, US-NC-1, US-CO-1. All are outside the approved configuration and were not used. |
| Pods on the account | none (`list-pods` empty) |
| Pod billing for the CPU tests | Not yet shown at 14:10 (0 records). Re-read at 15:40: see below. |

## Second attempt (15:30 UTC), approved as one attempt per card

The author confirmed auto-pay is disabled, approved the redesigned watchdog, and approved one create attempt each for the A40 and the RTX A6000 in EU-SE-1, with every safeguard unchanged.

| Step | State |
|---|---|
| Network volume `0uhv9a501u` | Now **50 GB** (resized by the author). Kept; not resized or deleted. ≈ $3.50 per month. |
| Stock read before the attempts | `list-data-centers`: A40 LOW and RTX A6000 LOW in EU-SE-1. `get-capacity` (Secure): A40 **unavailable** on both CUDA 12.8 and 13.2; RTX A6000 Low on 13.2, $0.59/h. The two reads disagree for the A40. |
| A40 canary create | Refused: "There are no longer any instances available with the requested specifications." |
| RTX A6000 canary create | Refused with the same message. |
| Body used | As at 14:10, with a fresh 45-min deadline (`DEADLINE_EPOCH=1794327319`), `RUN_ID=canary-20261110T1530Z`, `COMMIT=32ceb19`. Watchdog base64 checked equal to `infra/runpod/watchdog.sh` (sha256 of the encoding `d16eec8e…`). |
| Pods on the account | none (`list-pods` empty after both refusals) |
| Pod billing for the four CPU test pods | Now shown: **$0.0043 in total** (CPU only; GPU and disk $0). |
| Retries | None further, as instructed. |

## Third attempt (16:06 UTC): L40S, approved as one attempt (amendment 1)

| Step | State |
|---|---|
| Watchdog v2 (stop-first) | Implemented and tested on CPU pods; see `infra/runpod/TERMINATION_TESTS.md`. Stop-first, a stopped pod left in place, deadline, termination fallback and status read all pass. The restart-export path was refused by RunPod for host capacity, so it is a fallback only. |
| L40S stock before the attempt | `get-capacity`: **Out** on Secure for CUDA 12.8, 13.0, 13.1 and 13.2, and Out on Community. At 15:35 it read LOW. |
| GPU canary create (L40S, 20 GB pod volume disk, 1 h deadline, commit `c410e97`) | Refused: "There are no longer any instances available with the requested specifications." |
| Retries | None, as instructed. |
| Pods on the account | none (`list-pods` empty) |
| RTX PRO 6000 96 GB (not authorized) | Secure, CUDA 13.2: **Low**, **$2.49/h**, US-NC-2. With the 8 h deadline the GPU would cost at most $19.92, inside the $20 ceiling. A tighter deadline of 4.5 h caps it at $11.21. Expected: about $6–16 at A6000-equivalent speed, probably less, since the card is faster. G9 still stops the run above a projected $12. |
| Network volume `0uhv9a501u` | Unchanged (50 GB, EU-SE-1). |

## Fourth attempt (18:32 UTC): RTX PRO 6000, approved as one attempt (amendment 2)

**Outcome: the canary passed; the run then stopped at its first model step on a software bug in the CUDA memory guard. No CUDA model output exists. The approved attempt is used; no further pod was created.**

| Step | State |
|---|---|
| Stock before create | RTX PRO 6000 Blackwell Server, Secure, CUDA 13.2: LOW, $2.49/h (US-CO-1, US-NE-1) |
| Pod | `lpcwjejlq79ped`, placed in **EU-RO-1**, $2.49/h, 20 GB pod volume disk, commit `732021f`, created 18:32:25Z, deadline 23:01:29Z |
| Canary: Blackwell | sm_120 is in the build's arch list (torch 2.15.0+cu132, CUDA 13.2, cuDNN 92600). The bf16 and fp32 forward/backward kernel smoke tests pass. fp64 probe: max relative error 4.4e-7. TF32 is off, deterministic algorithms are on and autocast is off. |
| Canary: weights | sha256 `7de1838c…`, 2,127,532,032 BF16 parameters |
| Canary: watchdog | script sha256 `a9ad711e…` (the CPU-tested version), boot count 1, 16,035 s to the deadline. Stop-endpoint authorization probe: **422** (authorized, invalid action rejected). PID 1 is `docker-init`, running the watchdog's command chain; the watchdog is its foreground child. |
| Canary: off-pod transfer | Bundle sent at 18:34:30Z and received locally. sha256 `619fe4f7…` equal on both sides. Contents inspected. |
| Operator go | Signal pushed at 18:37 (`99837a9`); the pod verified it at 18:37:30Z |
| G3 input hashes | pass (96/96) |
| bf16 audit (first model step) | **Failed before loading the model**: `ValueError: Expected a torch.device with a specified index or an integer, but got:cuda`. The cause: `CudaGuard.arm()` passed `torch.device("cuda")` to `torch.cuda.set_per_process_memory_fraction`. This CUDA-only path had never run (local runs use MPS). It is not a gate failure. |
| Export | The pipeline's exit trap sent the output folder. Received locally. Every file it shares with the verified canary bundle is byte-identical. Preserved in [`attempts/full-20261110T1831Z/`](attempts/full-20261110T1831Z/) with a local hash list. **Gap:** the pod-side SHA256SUMS was written beside the folder, not inside it, so it was not sent. Fixed since. |
| Shutdown | Stopped from outside (REST action, `EXITED`) about 6 min after creation, then terminated (204). `get-pod` → 404, `list-pods` empty. The watchdog's own stop was not exercised on the GPU: the pod was stopped before its 30-min grace ended, to avoid idle GPU billing. |
| Charges | GPU pod billing not yet shown at 18:45Z. Expected ≈ $0.25 (about 6 min at $2.49/h). CPU test pods today: $0.0108. Network volume `0uhv9a501u`: $0.0156 so far today (50 GB standard). |

**Fixes after the attempt (no pod running):**
- `CudaGuard` normalizes an unindexed CUDA device to `cuda:0`. A regression test with fake `torch.cuda` calls that apply torch's index rule fails on the old guard and passes on the new one.
- The export manifest is written inside the sent folder and printed to the pod log.

## Fifth attempt (19:16 UTC): RTX PRO 6000, approved as one attempt (amendment 3)

**Outcome: RunPod's host never started the container, so nothing of ours ran. Terminated from outside about 4 min after creation. No data. The approved attempt is used; no further pod was created.**

| Step | State |
|---|---|
| Engineering before the attempt | Amendment 3 committed and pushed (`c39b9f5`): real preflight, integrity-verified resume, CUDA entry-point tests (full suite 588 passed). Watchdog unchanged (`a9ad711e…`). |
| Stock | 19:03 and 19:09: **Out** in both catalog reads, so creation was held back to avoid wasting the single attempt. 19:14: LOW, $2.49/h. |
| Pod | `ulduogqygqs4d5`, **US-PA-1**, $2.49/h, created 19:16:07Z, deadline 23:45:10Z |
| Host | System log: "create 20GB volume", image already present, then `error creating container: ... context deadline exceeded` (Docker socket timeout, 19:17:22Z). Then "container create: exit status 1" on every retry, 19:18:40Z–19:19:58Z. No container log line at all: the watchdog never started. |
| Action | Terminated from outside at about 19:20Z (204), as nothing was running and nothing could stop itself. `get-pod` → 404, `list-pods` empty. |
| Charges | Not yet shown. At most about 4 min at $2.49/h ≈ $0.17, if RunPod bills a pod whose container never started. |

**Lesson.** The in-pod safeguards depend on the container starting. When the host cannot create it, only outside monitoring can end the billing. In both GPU attempts the first container log line arrived within about 10 s of a successful start. So a pod with no container output 3 minutes after creation is treated as a host failure and terminated from outside.
