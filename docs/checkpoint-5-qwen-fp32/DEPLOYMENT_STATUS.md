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
