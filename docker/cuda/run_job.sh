#!/usr/bin/env bash
# One-shot CUDA validation job for a RunPod pod. Started through the pod's
# `args` override; the image default stays `preflight`.
#
# Results leave the pod through its log stream, between BEGIN/END markers, so
# no SSH, file transfer, or paid volume is needed. The container disk is wiped
# on stop or restart, so the logs are the record of this run.
#
# After the job the container sleeps instead of exiting: an exit could restart
# the container and repeat the measurement. The operator deletes the pod as soon
# as JOB_DONE is in the logs. Billing runs until then.
set -u
cd /opt/study
OUT=/opt/study/docs/checkpoint-2.1/cuda_validation
mkdir -p "$OUT"

section() { echo "=== BEGIN $1"; cat "$2" 2>/dev/null || echo "(missing)"; echo "=== END $1"; }

echo "=== JOB_START $(date -u +%Y-%m-%dT%H:%M:%SZ) git=${STUDY_GIT_COMMIT:-unknown}"
python -m cuda_readiness preflight > "$OUT/preflight.json"; echo "preflight rc=$?"
section preflight.json "$OUT/preflight.json"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true

python -m cuda_validation measure --device cuda --output "$OUT/cuda_scene.json" > /dev/null
echo "measure rc=$?"
section cuda_scene.json "$OUT/cuda_scene.json"
sha256sum "$OUT/cuda_scene.json" "$OUT/cuda_scene.gradient.f32" 2>/dev/null || true

if [ -f "$OUT/mps_scene.json" ] && [ -f "$OUT/mps_scene.gradient.f32" ]; then
  python -m cuda_validation compare "$OUT/mps_scene.json" "$OUT/cuda_scene.json" --output "$OUT/compare.json" > /dev/null
  echo "compare rc=$?"
  section compare.json "$OUT/compare.json"
else
  echo "compare skipped: no MPS sidecar was baked into this image"
fi

echo "=== JOB_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exec sleep infinity
