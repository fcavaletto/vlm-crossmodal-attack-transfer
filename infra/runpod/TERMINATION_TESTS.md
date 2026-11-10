# RunPod termination tests (2026-11-10)

Approved low-cost tests on CPU pods (`cpu3c`, 2 vCPU, Secure, $0.06/h; image `runpod/base:1.0.2-ubuntu2404` from RunPod's official "Runpod Ubuntu 24.04" template). Each check was confirmed from outside the pod, with `get-pod` and `list-pods`.

| Pod | Test | Result |
|---|---|---|
| `s7haxlv4ddn2yh` | `runpodctl` from the start command | **Failed as a mechanism.** `RUNPOD_POD_ID` and a pod-scoped `RUNPOD_API_KEY` are present at startup, but `runpodctl` is absent at first. A few seconds later an old build appears without the `pod` command: `unknown command "pod"`. So both `pod delete` and `pod stop` failed. The watchdog itself survived the workload failure and fired on time. Deleted from outside (204). |
| `y4dfj0opd75i4d` | `watchdog.sh`: workload fails after 15 s, 30 s grace | The watchdog fired at 13:58:10Z. REST `DELETE https://api.runpod.io/v2/pods/{id}` with the pod-scoped key returned **204**. Externally: `get-pod` → 404, absent from `list-pods`. |
| `42sl45fexzzs2n` | `watchdog.sh`: workload never ends, absolute deadline | Fired at the deadline (13:58:48Z). REST DELETE **204**. Externally: 404, absent from `list-pods`. |
| `v6ud3k1ufc63l5` | stop fallback alone: GraphQL `podStop` with the pod-scoped key | HTTP 200, `desiredStatus: EXITED`. Externally: status EXITED, actions `start`/`terminate`. Then **deleted from outside** (204), and `list-pods` is empty. |

**Conclusions.**
- The pod-scoped key can delete its own pod through the REST API, and the stop fallback works.
- The watchdog runs in the container's real startup environment. It stays effective when the workload fails early, and when the workload never ends.
- The deadline is an absolute epoch set at creation, so a container restart cannot extend it.
- Do not depend on the image's `runpodctl`.
- The local runpodctl 3.1.0 confirms that `pod delete` is the canonical command and `rm`/`remove` are aliases.
- Billing for these pods had not appeared in `list-pod-billing` at the time of writing; it is to be re-checked. Expected total < $0.01.

# Watchdog v2 (stop-first) tests (2026-11-10, 15:51–16:10 UTC)

Approved by the author: stop first; leave a successfully stopped pod stopped until its outputs are exported and hash-verified; terminate only if stopping fails. Same CPU flavour and image as above. Each state was confirmed from outside with `get-pod` / `list-pods`.

| Pod | Script | Test | Result |
|---|---|---|---|
| `pf16ljz0emdxe5` | v2 at `74f11a7` | Workload fails after 20 s, 30 s grace | Fired at 15:52:42Z. REST `POST /v2/pods/{id}/action {"action":"stop"}` → **200**. Externally `EXITED`, actions `start`/`terminate`. Left stopped and **still present 13+ minutes later**: nothing deletes a stopped pod. Then deleted from outside (see below). |
| `3sa8eb89ktlvqy` | v2 at `74f11a7` | Workload never ends, absolute deadline, stop requests disabled by the test hook | Fired at the deadline second (15:55:37Z). With both stops skipped, it terminated the pod itself. Externally **404**, absent from `list-pods`. |
| `htt5udikitpmm1` | v2 at `ab63252` (status-field fix) | Workload fails after 10 s, 20 s grace; then a restart | Status read now works (`SELF_READ status=RUNNING`). Fired at 16:01:04Z. REST stop **200**, then `EXITED`. **The restart was refused by RunPod**: "There are not enough free vcpu on the host machine to start this pod." Deleted from outside (204, then 404). |

**Bug found and fixed.** In `74f11a7` the status read looked for `desiredStatus`, but REST v2 names the field `status`, so the read was always empty. With an empty status, the extra wait before treating an accepted stop as failed could never trigger. Fixed in `ab63252` and re-tested on `htt5udikitpmm1`.

**Not testable on CPU.** CPU pods cannot have a pod volume disk (`persistent` mount), so the disk-based restart marker is tested in the GPU canary. On CPU, restart detection uses the late-boot rule: a boot more than `RESTART_AFTER_SECONDS` after creation.

**Restart-and-export mode is still untested on a real pod.** The restart was refused because the stopped pod's host had no free capacity. This is the documented zero-GPU risk: a stopped pod stays tied to its machine. The REST `start` action offers no zero-GPU start; the console does. **Consequence for the GPU run:** exporting from a stopped pod is a fallback, not the plan. Outputs are received and hash-verified **while the pod is still running**, through the hourly checkpoints and the 30-min grace after the workload ends. Only if that fails does the pod stop with its disk, to be recovered with a console zero-GPU start or by waiting for the host. The export-mode code path was checked locally with a stubbed API (boot marker and late-boot rules).

**End state.** Pods `pf16ljz0emdxe5` and `htt5udikitpmm1` were deleted from outside, and the watchdog terminated `3sa8eb89ktlvqy` itself. All three return 404, and `list-pods` is empty. Pod billing for these three had not appeared yet at 16:07Z. Expected: about 9–10 min each at $0.06/h, so under $0.01 each.
