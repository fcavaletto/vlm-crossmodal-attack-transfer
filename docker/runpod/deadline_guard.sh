#!/usr/bin/env bash
# Operator-side cost cap. Run on the operator's machine right after creating
# the pod, with the pod id that the create call returned:
#
#   docker/runpod/deadline_guard.sh <pod-id> 60
#
# After the given minutes it terminates that one pod through REST v2 with the
# operator's RUNPOD_API_KEY. It never lists pods and never acts on another id.
# The machine must stay awake (macOS: it wraps itself in `caffeinate -i`).
# REST v2 has no stop timer, and runpodctl's --terminate-after was never
# enforced by the backend, so this guard and the in-pod self-stop are the caps.
set -euo pipefail

POD_ID="${1:?usage: deadline_guard.sh <pod-id> <minutes>}"
MINUTES="${2:?usage: deadline_guard.sh <pod-id> <minutes>}"
: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY to the operator key}"

if [ -z "${DEADLINE_GUARD_AWAKE:-}" ] && command -v caffeinate >/dev/null 2>&1; then
  DEADLINE_GUARD_AWAKE=1 exec caffeinate -i "$0" "$@"
fi

echo "deadline_guard: will terminate pod $POD_ID at $(date -u -v+"${MINUTES}"M +%H:%M:%SZ 2>/dev/null || date -u -d "+${MINUTES} minutes" +%H:%M:%SZ)"
sleep $((MINUTES * 60))
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST \
  "https://api.runpod.io/v2/pods/${POD_ID}/action" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" -H 'Content-Type: application/json' \
  -d '{"action":"terminate"}')
echo "deadline_guard: terminate http=$code (204 = terminated; 404 = already gone)"
