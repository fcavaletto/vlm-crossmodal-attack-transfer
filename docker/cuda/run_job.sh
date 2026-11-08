#!/usr/bin/env bash
# One-shot CUDA validation job for a RunPod pod. Started through the pod's
# `args` override; the image default stays `preflight`.
#
# Results leave the pod through its log stream, between BEGIN/END markers, so
# no SSH, file transfer, or paid volume is needed. The container disk is wiped
# on stop or restart, so the log is the record of this run.
#
# The job ends with exactly one terminal line:
#   === JOB_DONE status=ok                  every step passed, compare recorded
#   === JOB_FAILED step=<s> rc=<n> reason=<r>
# A failed, stopped, or timed-out step never prints JOB_DONE.
#
# Cost guards, in order:
#   1. Every step runs under `timeout` against one deadline (JOB_DEADLINE_MINUTES).
#   2. A separate watchdog calls self_stop.py 5 minutes after that deadline,
#      even if this shell is stuck.
#   3. After the terminal line the job waits JOB_RESULT_GRACE_SECONDS (120) so the
#      operator's log capture (docker/runpod/capture_logs.sh) can read every
#      line, then calls self_stop.py (stop, then terminate). Stop comes first:
#      with no volume mounted a stopped pod costs nothing, and the pod record,
#      and with it the log, survives for the operator. RunPod does not document
#      that logs outlive termination.
#   4. If RunPod refuses both, the job sleeps instead of exiting (an exit could
#      restart the container and repeat the run) and prints SELF_STOP_FAILED.
#      The operator-side docker/runpod/deadline_guard.sh is the last guard.
set -uo pipefail

ROOT="${STUDY_ROOT:-/opt/study}"
PY="${JOB_PYTHON:-python}"
DEADLINE_MINUTES="${JOB_DEADLINE_MINUTES:-45}"
OUT="$ROOT/docs/checkpoint-2.1/cuda_validation"
TOOLS="$ROOT/docker/cuda"
cd "$ROOT" || { echo "=== JOB_FAILED step=setup rc=1 reason=no_study_root"; exit 1; }
mkdir -p "$OUT"

START=$(date +%s)
DEADLINE_SECONDS="${JOB_DEADLINE_SECONDS:-$((DEADLINE_MINUTES * 60))}"
DEADLINE=$((START + DEADLINE_SECONDS))
FINISHED=0

stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
section() { echo "=== BEGIN $1"; cat "$2" 2>/dev/null || echo "(missing)"; echo "=== END $1"; }

end_pod() {
  local grace="${JOB_RESULT_GRACE_SECONDS:-120}"
  echo "=== RESULT_GRACE seconds=$grace $(stamp): the log is the only copy of this run; capture it now"
  sleep "$grace"
  if "$PY" "$TOOLS/self_stop.py"; then
    echo "=== SELF_STOP_REQUESTED $(stamp)"
  else
    echo "=== SELF_STOP_FAILED $(stamp): delete the pod by hand now; billing continues"
  fi
  if [ "${JOB_EXIT_AFTER:-0}" = "1" ]; then  # tests only
    kill "$WATCHDOG" 2>/dev/null
    exit "$1"
  fi
  exec sleep infinity
}

finish_ok() {
  [ "$FINISHED" = 1 ] && return
  FINISHED=1
  : > "$OUT/.terminal"
  echo "=== JOB_DONE status=ok $(stamp) seconds=$(( $(date +%s) - START ))"
  end_pod 0
}

fail() {  # step rc reason
  [ "$FINISHED" = 1 ] && return
  FINISHED=1
  : > "$OUT/.terminal"
  echo "=== JOB_FAILED step=$1 rc=$2 reason=$3 $(stamp) seconds=$(( $(date +%s) - START ))"
  end_pod 1
}

trap 'fail signal 143 terminated' TERM INT

# Runs one step under the shared deadline. Sets STEP_RC.
step() {
  local name="$1"; shift
  local left=$((DEADLINE - $(date +%s)))
  if [ "$left" -le 0 ]; then fail "$name" 124 deadline; fi
  echo "=== STEP $name start $(stamp) budget_seconds=$left"
  timeout --kill-after=30 "$left" "$@"
  STEP_RC=$?
  echo "=== STEP $name rc=$STEP_RC $(stamp)"
  if [ "$STEP_RC" = 124 ] || [ "$STEP_RC" = 137 ]; then fail "$name" "$STEP_RC" deadline; fi
}

# Reads one JSON field. Prints "(unreadable)" when the file is missing or bad.
field() {
  "$PY" -c 'import json,sys
try: print(json.load(open(sys.argv[1])).get(sys.argv[2]))
except Exception: print("(unreadable)")' "$1" "$2"
}

# Watchdog: independent of the steps above, in case this shell hangs.
# It prints JOB_FAILED only if the job never printed a terminal line.
rm -f "$OUT/.terminal"
( trap 'kill "$NAP" 2>/dev/null; exit 0' TERM
  sleep $((DEADLINE_SECONDS + ${JOB_WATCHDOG_GRACE_SECONDS:-300})) & NAP=$!
  wait "$NAP"
  if [ -e "$OUT/.terminal" ]; then
    echo "=== WATCHDOG $(stamp): pod still running after the terminal line; asking RunPod again"
  else
    echo "=== JOB_FAILED step=watchdog rc=124 reason=hard_deadline $(stamp)"
  fi
  "$PY" "$TOOLS/self_stop.py" ) &
WATCHDOG=$!

echo "=== JOB_START $(stamp) git=${STUDY_GIT_COMMIT:-unknown} deadline_seconds=$DEADLINE_SECONDS"

step weights "$PY" "$TOOLS/fetch_weights.py" --output "$OUT/weights.json"
section weights.json "$OUT/weights.json"
[ "$STEP_RC" = 0 ] || fail weights "$STEP_RC" weights_missing_or_hash_mismatch

step preflight env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PY" -m cuda_readiness preflight --require-cuda --output "$OUT/preflight.json"
section preflight.json "$OUT/preflight.json"
[ "$STEP_RC" = 0 ] || fail preflight "$STEP_RC" cuda_not_available
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true

step measure env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$PY" -m cuda_validation measure --device cuda --output "$OUT/cuda_scene.json"
section cuda_scene.json "$OUT/cuda_scene.json"
sha256sum "$OUT/cuda_scene.json" "$OUT/cuda_scene.gradient.f32" 2>/dev/null || true
[ "$STEP_RC" = 0 ] || fail measure "$STEP_RC" measure_exit_nonzero
MEASURE_STATUS=$(field "$OUT/cuda_scene.json" status)
[ "$MEASURE_STATUS" = completed ] || fail measure 0 "status_$MEASURE_STATUS"
[ -s "$OUT/cuda_scene.gradient.f32" ] || fail measure 0 gradient_file_missing

if [ ! -f "$OUT/mps_scene.json" ] || [ ! -f "$OUT/mps_scene.gradient.f32" ]; then
  fail compare 0 no_mps_sidecar_in_image
fi
step compare "$PY" -m cuda_validation compare "$OUT/mps_scene.json" "$OUT/cuda_scene.json" --output "$OUT/compare.json"
section compare.json "$OUT/compare.json"
[ "$STEP_RC" = 0 ] || fail compare "$STEP_RC" compare_exit_nonzero
COMPARE_STATUS=$(field "$OUT/compare.json" status)
[ "$COMPARE_STATUS" = differences_recorded ] || fail compare 0 "status_$COMPARE_STATUS"

finish_ok
