#!/usr/bin/env bash
# Operator-side log capture. Start it right after creating the pod, next to
# deadline_guard.sh:
#
#   docker/runpod/capture_logs.sh <pod-id> docs/checkpoint-2.1/cuda_validation/pod_log.jsonl
#
# The pod's log stream is the only copy of the run. This follows the REST v2 SSE
# stream (GET /v2/pods/{id}/logs, container source, 5000-line backfill), writes
# every event to the file, and exits after it sees the job's terminal line
# (=== JOB_DONE or === JOB_FAILED). On a dropped connection it resumes from the
# last event id. It only reads; it never stops or terminates anything.
set -euo pipefail

POD_ID="${1:?usage: capture_logs.sh <pod-id> <output.jsonl>}"
OUTFILE="${2:?usage: capture_logs.sh <pod-id> <output.jsonl>}"
: "${RUNPOD_API_KEY:?set RUNPOD_API_KEY to the operator key}"
LAST=""
mkdir -p "$(dirname "$OUTFILE")"
echo "capture_logs: writing $OUTFILE"
while :; do
  headers=(-H "Authorization: Bearer ${RUNPOD_API_KEY}" -H 'Accept: text/event-stream')
  [ -n "$LAST" ] && headers+=(-H "Last-Event-ID: $LAST")
  while IFS= read -r line; do
    case "$line" in
      id:*) LAST="${line#id: }"; LAST="${LAST#id:}" ;;
      data:*)
        payload="${line#data: }"; payload="${payload#data:}"
        printf '%s\n' "$payload" >> "$OUTFILE"
        case "$payload" in
          *'=== JOB_DONE'*|*'=== JOB_FAILED'*)
            echo "capture_logs: terminal line seen; keep this running until RESULT_GRACE ends if you want the self-stop lines too"
            exit 0 ;;
        esac ;;
    esac
  done < <(curl -sN "${headers[@]}" "https://api.runpod.io/v2/pods/${POD_ID}/logs?source=container&tail=5000" || true)
  echo "capture_logs: stream ended; reconnecting from ${LAST:-start} in 5 s"
  sleep 5
done
