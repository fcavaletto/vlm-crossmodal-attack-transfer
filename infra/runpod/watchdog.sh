#!/usr/bin/env bash
# Pod-side shutdown for Checkpoint 5. Runs as the container's foreground process.
#   DEADLINE_EPOCH  absolute unix time (set at pod creation, so a container restart cannot extend it)
#   GRACE_SECONDS   delay after the workload exits (success or failure) before shutting down
#   WORKLOAD        command to run in the background
# Shutdown path, in order: REST v2 DELETE, GraphQL podTerminate, GraphQL podStop.
# Uses RunPod's pod-scoped key (RUNPOD_API_KEY, injected by RunPod). The key is never printed.
set -u
echo "WATCHDOG_BOOT $(date -u +%FT%TZ) pod=${RUNPOD_POD_ID:-MISSING} deadline=${DEADLINE_EPOCH:-MISSING} grace=${GRACE_SECONDS:-MISSING}"
[ -n "${RUNPOD_API_KEY:-}" ] && echo "POD_KEY_PRESENT=yes" || echo "POD_KEY_PRESENT=no"
echo "CURL=$(command -v curl || echo MISSING)"
api_get() { curl -sS -o /dev/null -w '%{http_code}' -H "Authorization: Bearer ${RUNPOD_API_KEY}" "https://api.runpod.io/v2/pods/${RUNPOD_POD_ID}" 2>/dev/null || echo ERR; }
echo "SELF_READ_HTTP=$(api_get)"
shutdown_pod() {
  echo "WATCHDOG_FIRE reason=$1 $(date -u +%FT%TZ)"
  code=$(curl -sS -o /tmp/del.out -w '%{http_code}' -X DELETE -H "Authorization: Bearer ${RUNPOD_API_KEY}" "https://api.runpod.io/v2/pods/${RUNPOD_POD_ID}" 2>&1)
  echo "REST_DELETE_HTTP=${code}"; head -c 300 /tmp/del.out 2>/dev/null; echo
  case "$code" in 200|202|204) sleep 120; echo "STILL_RUNNING_AFTER_REST_DELETE";; esac
  code=$(curl -sS -o /tmp/gql.out -w '%{http_code}' -H 'content-type: application/json' -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    --data "{\"query\":\"mutation { podTerminate(input: {podId: \\\"${RUNPOD_POD_ID}\\\"}) }\"}" https://api.runpod.io/graphql 2>&1)
  echo "GQL_TERMINATE_HTTP=${code}"; head -c 300 /tmp/gql.out 2>/dev/null; echo
  sleep 60; echo "STILL_RUNNING_AFTER_GQL_TERMINATE"
  code=$(curl -sS -o /tmp/stop.out -w '%{http_code}' -H 'content-type: application/json' -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    --data "{\"query\":\"mutation { podStop(input: {podId: \\\"${RUNPOD_POD_ID}\\\"}) { id desiredStatus } }\"}" https://api.runpod.io/graphql 2>&1)
  echo "GQL_STOP_HTTP=${code}"; head -c 300 /tmp/stop.out 2>/dev/null; echo
  sleep 600; echo "STILL_RUNNING_AFTER_ALL_SHUTDOWN_PATHS"
}
if [ "$(date +%s)" -ge "${DEADLINE_EPOCH:-0}" ]; then shutdown_pod deadline_already_passed_at_boot; fi
bash -c "${WORKLOAD:-sleep infinity}" & WL=$!
EXIT_AT=
while :; do
  now=$(date +%s)
  if [ "$now" -ge "$DEADLINE_EPOCH" ]; then shutdown_pod deadline; fi
  if [ -z "$EXIT_AT" ] && ! kill -0 "$WL" 2>/dev/null; then wait "$WL"; echo "WORKLOAD_EXITED rc=$? $(date -u +%FT%TZ)"; EXIT_AT=$now; fi
  if [ -n "$EXIT_AT" ] && [ $((now - EXIT_AT)) -ge "${GRACE_SECONDS:-1800}" ]; then shutdown_pod workload_exit_plus_grace; fi
  sleep 5
done
