#!/usr/bin/env bash
# Pod-side shutdown for Checkpoint 5 (v2, stop-first). Runs as the container's foreground process,
# so it works without any interactive session.
#   DEADLINE_EPOCH         absolute unix time, set at pod creation (a restart cannot extend it)
#   GRACE_SECONDS          delay after the workload exits (success or failure) before shutting down
#   WORKLOAD               command to run in the background on the first boot
#   POD_START_EPOCH        unix time of the create call
#   RESTART_AFTER_SECONDS  a boot later than this after POD_START_EPOCH counts as a restart (default 2700)
#   EXPORT_WORKLOAD        command run instead of WORKLOAD after a restart (results export only)
#   EXPORT_WINDOW_SECONDS  how long a restarted container stays up for export (default 1800)
#   STOP_CONFIRM_SECONDS   wait after each stop request before calling it failed (default 300)
#   WATCHDOG_STATE_DIR     boot marker location (default /workspace/.watchdog, the pod's volume disk)
# Shutdown: stop first (REST v2 action, then GraphQL podStop). A stopped pod keeps its volume disk
# and is left stopped. Only if the container is still alive after both stops (and an extra wait when
# the platform shows the stop as accepted) is the pod terminated, to bound GPU billing. Termination
# destroys records that were not exported.
# A restarted container never re-runs the workload: it only offers the results for export, then stops.
# Uses RunPod's pod-scoped key (RUNPOD_API_KEY, injected by RunPod). The key is never printed.
# Test hook: WATCHDOG_TEST_SKIP_STOP=1 makes both stop requests no-ops, to exercise the fallback.
set -u
API=https://api.runpod.io
ID=${RUNPOD_POD_ID:-MISSING}
STATE_DIR=${WATCHDOG_STATE_DIR:-/workspace/.watchdog}
CONFIRM=${STOP_CONFIRM_SECONDS:-300}
ts() { date -u +%FT%TZ; }
echo "WATCHDOG_V2_BOOT $(ts) pod=${ID} deadline=${DEADLINE_EPOCH:-MISSING} grace=${GRACE_SECONDS:-MISSING} start=${POD_START_EPOCH:-MISSING}"
[ -n "${RUNPOD_API_KEY:-}" ] && echo "POD_KEY_PRESENT=yes" || echo "POD_KEY_PRESENT=no"
echo "CURL=$(command -v curl || echo MISSING)"
auth() { printf 'Authorization: Bearer %s' "${RUNPOD_API_KEY:-}"; }
pod_status() {  # REST v2 field "status": RUNNING, EXITED, ...
  curl -sS --max-time 30 -H "$(auth)" "$API/v2/pods/$ID" 2>/dev/null | grep -o '"status" *: *"[A-Z_]*"' | head -1 | grep -o '[A-Z_]*"$' | tr -d '"'
}
echo "SELF_READ status=$(pod_status)"

stop_rest() {
  if [ "${WATCHDOG_TEST_SKIP_STOP:-0}" = 1 ]; then echo "TEST_SKIP_STOP: REST stop not sent"; return; fi
  code=$(curl -sS --max-time 30 -o /tmp/stop_rest.out -w '%{http_code}' -X POST -H "$(auth)" -H 'content-type: application/json' \
    --data '{"action":"stop"}' "$API/v2/pods/$ID/action" 2>&1)
  echo "REST_STOP_HTTP=${code}"; head -c 300 /tmp/stop_rest.out 2>/dev/null; echo
}
stop_gql() {
  if [ "${WATCHDOG_TEST_SKIP_STOP:-0}" = 1 ]; then echo "TEST_SKIP_STOP: GraphQL stop not sent"; return; fi
  code=$(curl -sS --max-time 30 -o /tmp/stop_gql.out -w '%{http_code}' -H 'content-type: application/json' -H "$(auth)" \
    --data "{\"query\":\"mutation { podStop(input: {podId: \\\"${ID}\\\"}) { id desiredStatus } }\"}" "$API/graphql" 2>&1)
  echo "GQL_STOP_HTTP=${code}"; head -c 300 /tmp/stop_gql.out 2>/dev/null; echo
}
terminate_pod() {
  code=$(curl -sS --max-time 30 -o /tmp/del.out -w '%{http_code}' -X DELETE -H "$(auth)" "$API/v2/pods/$ID" 2>&1)
  echo "REST_DELETE_HTTP=${code}"; head -c 300 /tmp/del.out 2>/dev/null; echo
  sleep 120
  code=$(curl -sS --max-time 30 -o /tmp/term.out -w '%{http_code}' -H 'content-type: application/json' -H "$(auth)" \
    --data "{\"query\":\"mutation { podTerminate(input: {podId: \\\"${ID}\\\"}) }\"}" "$API/graphql" 2>&1)
  echo "GQL_TERMINATE_HTTP=${code}"; head -c 300 /tmp/term.out 2>/dev/null; echo
}
shutdown_pod() {
  echo "WATCHDOG_FIRE reason=$1 $(ts)"
  mkdir -p "$STATE_DIR" 2>/dev/null && echo "$1 $(ts)" >> "$STATE_DIR/fired" 2>/dev/null
  stop_rest;  sleep "$CONFIRM"; echo "STILL_RUNNING_AFTER_REST_STOP status=$(pod_status) $(ts)"
  stop_gql;   sleep "$CONFIRM"; echo "STILL_RUNNING_AFTER_GQL_STOP status=$(pod_status) $(ts)"
  if [ "$(pod_status)" = EXITED ]; then
    echo "STOP_ACCEPTED_CONTAINER_ALIVE: waiting ${CONFIRM}s more before treating the stop as failed"
    sleep "$CONFIRM"
  fi
  echo "STOP_FAILED $(ts): terminating to bound GPU billing; unexported records on the pod disk are lost"
  terminate_pod
  while :; do sleep 600; echo "STILL_RUNNING_AFTER_ALL_PATHS $(ts)"; terminate_pod; done
}

mkdir -p "$STATE_DIR" 2>/dev/null
now=$(date +%s)
PREV_BOOTS=$(cat "$STATE_DIR/boots" 2>/dev/null || echo 0)
echo $((PREV_BOOTS + 1)) > "$STATE_DIR/boots" 2>/dev/null
LATE=0; [ -n "${POD_START_EPOCH:-}" ] && LATE=$(( now - POD_START_EPOCH > ${RESTART_AFTER_SECONDS:-2700} ))
echo "BOOT_COUNT=$((PREV_BOOTS + 1)) late_boot=${LATE}"

if [ "$PREV_BOOTS" -gt 0 ] || [ "$LATE" = 1 ] || [ "$now" -ge "${DEADLINE_EPOCH:-0}" ]; then
  # Restart, late boot or deadline already passed: export only, then stop again.
  END=$(( now + ${EXPORT_WINDOW_SECONDS:-1800} ))
  echo "RESTART_EXPORT_MODE until $(date -u -d @$END +%FT%TZ 2>/dev/null || echo $END): the workload is not re-run"
  bash -c "${EXPORT_WORKLOAD:-true}" &
  while [ "$(date +%s)" -lt "$END" ]; do sleep 5; done
  shutdown_pod export_window_end
fi

bash -c "${WORKLOAD:-sleep infinity}" & WL=$!
EXIT_AT=
while :; do
  now=$(date +%s)
  if [ "$now" -ge "$DEADLINE_EPOCH" ]; then shutdown_pod deadline; fi
  if [ -z "$EXIT_AT" ] && ! kill -0 "$WL" 2>/dev/null; then wait "$WL"; echo "WORKLOAD_EXITED rc=$? $(ts)"; EXIT_AT=$now; fi
  if [ -n "$EXIT_AT" ] && [ $((now - EXIT_AT)) -ge "${GRACE_SECONDS:-1800}" ]; then shutdown_pod workload_exit_plus_grace; fi
  sleep 5
done
