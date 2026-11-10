# Checkpoint 5: final deployment summary, for authorization (2026-11-10)

Nothing has been created yet. Every figure below comes from a read made today with RunPod's tools. Prices and stock are re-read at create time; if either differs, I stop and report before creating anything.

## What would be created, in order

| # | Resource | Configuration | Rate | Expected cost |
|---|---|---|---|---|
| 1 | Termination test pod | CPU `cpu3c`, 2 vCPU, Secure, no volume. Start command: `sleep 90; runpodctl pod remove "$RUNPOD_POD_ID"`. | $0.03/vCPU-h, so **$0.06/h** | ≈ $0.01 (about 5 min including my external check) |
| 2 | Network volume | **30 GB, STANDARD**, data center **EU-SE-1**. Mounted at `/workspace`, with `HF_HOME=/workspace/hf-cache`. | ≈ $0.07/GB-month | ≈ **$2.10 per month**, ongoing, **kept** after the run |
| 3 | GPU pod | **NVIDIA A40 48 GB, Secure, EU-SE-1** (RTX A6000 48 GB at the same price if no A40 is free). Network volume at `/workspace`, 30 GB container disk, CUDA ≥ 12.8 (hosts report 13.2). Pinned environment installed from the repository at the protocol commit. | **$0.59/h** | $1.5–3.7 expected, ≈ $5.6 with a 50% buffer |

## Automatic termination setting

- RunPod has **no provider-side terminate timer**: REST v2 has none, and `--terminate-after` was never enforced and has been removed from runpodctl. So it is not used.
- **In-pod deadline, the first thing in the GPU pod's start command:** `(sleep 28800; runpodctl pod remove "$RUNPOD_POD_ID" || runpodctl pod stop "$RUNPOD_POD_ID") &`.
  - It terminates the pod **8 h** after container start, using RunPod's preinstalled pod-scoped key. No key of ours is stored in the pod.
  - The runner stops itself at **7.5 h**, so records are flushed first.
  - Resource 1 tests whether that key can *remove* a pod before the GPU pod exists. If it cannot, I report it before creating the GPU pod. The deadline would then only *stop* the pod, which ends GPU billing.
- **External, independent of the pod.** I terminate the GPU pod from this session when the work is done. I confirm it is gone (`list-pods`) and read its final cost (`list-pod-billing`). During the run I read the pod's state and cost at every phase boundary, and at least every 30 minutes.

## Gates on the GPU pod (unchanged science)

- **Before any attack:**
  - **G1–G3:** TF32 off (fp64 probe), no autocast, dtype audit, weights sha256 `7de1838c…`, and all 96 input hashes equal to the local ones.
  - **G5:** CUDA bf16 vs MPS bf16 clean margins: median ≤ 0.25 logit, 95th percentile ≤ 0.75, maximum ≤ 1.5, and the same chosen colour on ≥ 94/96 prompts. **If CUDA bf16 differs substantially from MPS, the run stops for investigation. The thresholds are not changed.**
  - **G6:** fp32 clean accuracy ≥ 75% in each context.
  - **G7:** fp32 finite differences.
  - **G8:** determinism.
- **After the first scene:**
  - **G4:** peak reserved ≤ 36 GiB.
  - **G9:** projected total spend **≤ $12**, or the run stops and I ask.

## Maximum financial exposure

| Case | Exposure |
|---|---|
| Normal run | about $2–6 of GPU compute, plus the volume |
| Gate | the run stops if projected spend exceeds **$12** |
| In-pod deadline works (remove or stop) | GPU at most 8 h × $0.59 = **$4.72**, plus about $0.01 for the test pod |
| Authorized ceiling | **$20** of GPU compute |
| Network volume (separate, ongoing) | ≈ $2.10 per month until you delete it |
| Deadline fails *and* external termination fails | $0.59/h, about $14 per day, until the pod is terminated by hand. RunPod has no dollar cap (its only limit is $80/hour). At a zero balance, pods with a network volume are *stopped*, not terminated. |

**What you can do to tighten the last row:**
- keep **auto-pay off**, so the balance bounds spending;
- optionally set a low-balance email alert in the console.

## What I need from you

1. **Authorization** to create resources 1–3, in order, under these settings and gates.
2. **Confirmation that auto-pay is off**, or that you accept it on.
3. **An SSH public key registered in your RunPod account**, so I can copy results off the pod. Otherwise I use `runpodctl send`/`receive`.

No merge, no tag, no confirmation split and no use of `trackB_confirm_v1` are part of this.
