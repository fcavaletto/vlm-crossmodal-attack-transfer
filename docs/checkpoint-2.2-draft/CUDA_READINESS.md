# CUDA validation: readiness assessment and spending plan (2026-11-08)

No pod was created, no image was built or pushed, and no RunPod account call was made. The only RunPod reads were public-documentation searches. Code changes are on `checkpoint-2.1/cuda-deploy-fixes` (PR 27, head `691ee06`, not merged). CUDA is not needed for Checkpoint 2.2. It is a separate validation track.

## 1. Self-termination: what is now known

| Question | Answer | Source |
| --- | --- | --- |
| Does the pod know its id and have a key? | Yes. Every pod gets `RUNPOD_POD_ID` and a pod-scoped `RUNPOD_API_KEY`. | docs.runpod.io/pods/templates/environment-variables |
| Is an in-pod stop supported? | Documented: `runpodctl pod stop $RUNPOD_POD_ID` from inside the pod, including a `sleep 2h; runpodctl pod stop …` kill-switch example. | docs.runpod.io/pods/manage-pods; RunPod blog "How to Manage Funding Your Runpod Account" |
| Is an in-pod **terminate** supported? | Not documented either way. | — |
| May the pod-scoped key call REST v2 `POST /v2/pods/{id}/action`? | Not documented. The documented in-pod path is `runpodctl`, which the image does not ship. | — |
| Does stop end billing for this pod? | **Yes, completely.** The spec mounts no volume, and a stopped pod's container disk is not billed. | docs.runpod.io/pods/pricing (storage table) |
| Is there a server-side timer? | No. REST v2 has no stop or terminate field. | REST v2 spec, via the RunPod concept graph |
| Hard ceiling regardless of software | The prepaid balance. At $0 every pod stops, and pods without a network volume are terminated. Creating a pod needs one hour of credit. | docs.runpod.io/accounts-billing/billing |
| Do logs survive termination? | Not documented for pods. Serverless worker logs are documented as deleted when the worker terminates. | docs.runpod.io/serverless/development/logs |

Conclusion:
- Terminate versus stop no longer matters for cost. A stopped pod of this spec costs $0.
- The remaining uncertainty is narrower: whether the REST call made with the pod-scoped key is accepted at all. If it is refused, the job prints `SELF_STOP_FAILED` and sleeps. Billing then continues until the operator guard (60 min) or the balance stops the pod.
- The log question exposed a real defect, fixed below.

## 2. Defect found and fixed in this pass

The second pass terminated the pod right after the terminal line. The log is the **only** copy of the results, and its retention after termination is undocumented, so a fast self-terminate could have erased a successful run. Commit `691ee06` changes three things:
1. `self_stop.py` tries `stop` first, then `terminate`. Stop costs $0 here and keeps the pod record.
2. `run_job.sh` prints `=== RESULT_GRACE seconds=120` and waits before stopping itself.
3. A new `docker/runpod/capture_logs.sh <pod-id> <file>` follows the REST v2 SSE log stream (5,000-line backfill, resume on drop) and exits at the terminal line. It only reads.

Tests: 51 pass (40 in `cuda_validation`, 11 in `cuda_readiness`). New tests check the order terminal line → grace → self-stop, that self-stop tries `stop` before `terminate`, and that it never prints the key.

## 3. Safeguards, outermost first

| Layer | Bound | Fails if |
| --- | --- | --- |
| Account balance kept at **≤ $5** (decision C0) | Absolute: total spend can never exceed the balance | Auto-pay is on. Auto-pay must be **off**. |
| Operator `deadline_guard.sh <pod> 60` | Terminates that pod id at 60 min | The Mac sleeps or loses network (it uses `caffeinate`) |
| Watchdog inside the pod | Calls self-stop at 50 min | The pod-scoped key is refused |
| Per-step `timeout`, shared 45-min deadline | Kills a hung step, prints `JOB_FAILED` | — |
| Terminal-line contract | `JOB_DONE` only after every check | — (tested on 9 failure paths) |

## 4. Validation plan with an explicit spending limit (not launched)

**Spending limit: $3.00 hard, enforced by the balance (C0); planned spend about $0.30–0.60.**

| Step | Who | Cost |
| --- | --- | --- |
| 0. Turn auto-pay off. Confirm the balance is ≤ $5 and at least $0.59 (needed to create). Confirm no other pods are running. | Federico, in the console | $0 |
| 1. Local MPS one-scene sidecar (C1). Loads weights on the Mac, needed because the job refuses to finish without it. | Mac | $0 |
| 2. Build the image in GitHub Actions and publish it to GHCR (C3). Record the digest. | GitHub | $0 |
| 3. Create one L4 Secure pod from `pod-spec.json`, with the image pinned by digest, `disk` 40, and no mounts (C4). | Operator | $0.59/h |
| 4. Immediately start `capture_logs.sh <id> docs/checkpoint-2.1/cuda_validation/pod_log.jsonl` and `deadline_guard.sh <id> 60`. | Operator Mac | $0 |
| 5. Expected timeline: image pull about 3–8 min, weights download 1–3 min, measure plus compare under 10 min, grace 2 min. **About 15–25 min ≈ $0.15–0.25.** | — | — |
| 6. Read the terminal line from the saved log. Check `SELF_STOP_REQUESTED` and the HTTP codes. This answers the open key question. | Operator | — |
| 7. Terminate the pod by id from the operator machine once the log is saved and checked. Stop the guard. | Operator | ends at $0 |

Abort rules:
- If the pod is still `STARTING` after 15 min, terminate it.
- If the log shows no `JOB_START` within 20 min of `RUNNING`, terminate it.
- Never create a second pod in the same session without a new approval.

Worst case with every software layer failing except the Mac guard: 60 min ≈ $0.60. If the Mac guard also fails, the balance (≤ $5) is the ceiling.

## 5. Still unverified, and how the plan resolves it

- Whether the pod-scoped key can call the REST action route. Step 6 reads the HTTP codes. Neither outcome needs a code change, since the guard and the balance cover a refusal.
- Real pull and download times. They are recorded in the log timestamps.
- `runpodctl` flags `--stop-after`/`--terminate-after`. The RunPod concept graph lists them for `pod create`. An earlier pass found them documented as never enforced and removed in v2.12.0. The plan relies on neither.
