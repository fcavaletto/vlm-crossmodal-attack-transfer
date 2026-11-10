# Checkpoint 5, amendment 4: L40S (A40/A6000 fallback) and external startup supervision (2026-11-10, before deployment)

**Infrastructure only.** Approved by the author on 2026-11-10 after the fifth attempt, in which RunPod's host never created the container. No CUDA result exists. The scientific protocol, dtype controls, pre-registered comparisons and every numerical threshold are unchanged. If this GPU behaves differently numerically, no threshold is changed.

## Hardware

| Order | Card | Cap | Attempts |
|---|---|---|---|
| 1 | **NVIDIA L40S 48 GB** (Ada), Secure Cloud, CUDA 13.2 host | $1.09/h | one |
| 2, only if the L40S create is refused or the stock read shows it unavailable | A40 or RTX A6000 48 GB (Ampere), Secure Cloud, CUDA 13.2 host | $0.59/h | at most one more, in total |

- Not authorized: RTX PRO 6000, A100, or anything more expensive.
- Capacity, CUDA 13.2 and price are read immediately before each create. No polling loops.
- At 2026-11-10 ~19:30 UTC, `get-capacity` (Secure, CUDA 13.2) showed L40S LOW at $1.09, A40 LOW at $0.59 and A6000 LOW at $0.59.
- At most $4.91 of L40S GPU time fits in the 4.5 h deadline ($2.66 for A40/A6000).

## Unchanged from amendment 3 (commit `c39b9f5`)

- The real preflight: G3 input hashes, the bf16 and fp32 audits, clean scoring, FD, `gates.py`, and one complete paired unit through the exact main runner.
- The integrity-verified resume, with the preflight bundle exported and hash-verified before the go, and the 15-min hold.
- The guard's peak-reserved limit of **36 GiB**: on a 48 GB card that is 75% of the device, the same as the memory fraction.
- The numerical and integrity gates, G9 at a projected **$12**, the **$20** ceiling, the **4 h** workload deadline and the **4.5 h** absolute deadline.
- The stop-first watchdog v2 (sha256 `a9ad711e…`), temporary pod storage, no network volume, and checkpoints during the run.
- **Throughput is measured on the selected card** by the paired preflight unit and re-checked at resume. Nothing is extrapolated from the RTX PRO 6000.

## New: external startup supervision

The in-container watchdog protects nothing until the container has started. So, from outside the pod:

1. Within **3 min** of the create call, the container log must show the watchdog's boot line (`WATCHDOG_V2_BOOT`). It must also show its first heartbeat: `POD_KEY_PRESENT=yes`, `SELF_READ status=RUNNING`, and `BOOT_COUNT=1`.
2. If they are absent at 3 min:
   - read the system log and the pod state;
   - **terminate the pod from outside** (there is nothing to preserve before the container runs);
   - confirm 404 and an empty `list-pods`;
   - record the host status.

   That pod counts as the attempt.
3. Only after the boot line and heartbeat are seen is the in-container watchdog relied on for shutdown. Until then the operator is the only shutdown path.
4. The operator's later checks are outside safeguards on top of the watchdog: export receipt, gate review, and final termination.

## If deployment fails

No further attempts in this cycle. The failure is recorded, with a recommendation for a different provider or deployment approach.
