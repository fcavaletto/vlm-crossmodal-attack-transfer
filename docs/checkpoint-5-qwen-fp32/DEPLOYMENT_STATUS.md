# Checkpoint 5 deployment status (2026-11-10, 14:10 UTC)

**Blocked on GPU capacity. No GPU pod exists; no research computation has run on CUDA.**

| Step | State |
|---|---|
| CPU termination tests | Passed after one redesign. See [`infra/runpod/TERMINATION_TESTS.md`](../../infra/runpod/TERMINATION_TESTS.md). Every test pod was confirmed gone from outside. |
| Network volume | `0uhv9a501u` (vlm-research-eu-se-1): 30 GB, STANDARD, EU-SE-1. Created and read back. **Billing: ≈ $2.10 per month until deleted.** |
| Fresh availability, EU-SE-1, Secure, CUDA ≥ 13.2 | A40: not listed. RTX A6000: LOW, $0.59/h. |
| GPU canary | `create-pod` for RTX A6000 in EU-SE-1 with the volume refused twice: "no longer any instances available". Stock re-read between attempts (still LOW). Not retried further. |
| Other data centers with network storage and a 48 GB card (same read) | None. 32 GB RTX 5090 in EU-RO-1; H200 in EUR-IS-4, US-NC-1, US-CO-1. All are outside the approved configuration and were not used. |
| Pods on the account | none (`list-pods` empty) |
| Pod billing for the CPU tests | Not yet shown by `list-pod-billing` (0 records); expected total < $0.01. To re-check. |
