"""Ask RunPod to end this pod, from inside it. Used by run_job.sh.

Every pod gets RUNPOD_POD_ID and a pod-scoped RUNPOD_API_KEY. This tries the
REST v2 `stop` action first, then `terminate`. It prints HTTP status codes only,
never the key. Exit 0 means RunPod accepted one of the two actions.

Why stop first: the validation pod mounts no volume, and RunPod does not bill a
stopped pod's container disk, so a stopped pod costs nothing. Stopping keeps the
pod record, and its log stream is the only copy of the results. RunPod does not
document that logs outlive termination. The operator terminates the pod after
saving the log.

RunPod documents an in-pod stop (`runpodctl pod stop $RUNPOD_POD_ID`,
docs.runpod.io/pods/manage-pods). It does not document whether the pod-scoped
key may call the REST v2 action route. That is checked on the first live pod,
and the job log shows the HTTP codes.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE = "https://api.runpod.io/v2"


def _action(pod_id: str, key: str, action: str) -> int:
    request = urllib.request.Request(
        f"{BASE}/pods/{pod_id}/action",
        data=json.dumps({"action": action}).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except (urllib.error.URLError, TimeoutError, OSError):
        return -1


def main() -> int:
    pod_id = os.environ.get("RUNPOD_POD_ID")
    key = os.environ.get("RUNPOD_API_KEY")
    if not pod_id or not key:
        print("self_stop: RUNPOD_POD_ID or RUNPOD_API_KEY is not set; not on a RunPod pod")
        return 2
    for action in ("stop", "terminate"):
        status = _action(pod_id, key, action)
        print(f"self_stop: action={action} http={status}")
        if 200 <= status < 300:
            return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
