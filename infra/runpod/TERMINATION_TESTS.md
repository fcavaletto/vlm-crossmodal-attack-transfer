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
