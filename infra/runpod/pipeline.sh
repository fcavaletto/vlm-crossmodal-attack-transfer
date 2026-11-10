#!/usr/bin/env bash
# Checkpoint 5 GPU workload. Started in the background by watchdog.sh, which owns shutdown.
#   STAGE       canary | full
#   COMMIT      repository commit to run (the protocol commit or later)
#   RUN_ID      output folder name on the network volume
#   POD_START_EPOCH, DEADLINE_EPOCH, HOURLY_USD   for the runner's budget gate and deadline
# Outputs: /workspace/checkpoint-5/$RUN_ID on the pod's volume disk (kept while the pod is stopped,
# destroyed if it is terminated). Model cache: /workspace/hf-cache.
# Any failed step ends the workload (set -e); the watchdog then stops the pod after its grace period.
# Checkpoints: after the gates and every CHECKPOINT_SECONDS of the main run, a snapshot of the records
# (tensors excluded) is offered with `runpodctl send`; the one-time code goes to the log.
set -euo pipefail
OUT=/workspace/checkpoint-5/${RUN_ID}
mkdir -p "$OUT/audit" "$OUT/run"
exec > >(tee -a "$OUT/pipeline.log") 2>&1
status() { echo "$1" > "$OUT/STATUS"; echo "STATUS=$1 $(date -u +%FT%TZ)"; }
RPC=/workspace/bin/runpodctl   # on the volume disk, so a restarted container can still export
SEND_PID=
CKPT_LOOP=
checkpoint() {  # $1 = label. Snapshot of the records without tensors, offered for download.
  [ -x "$RPC" ] || { echo "CHECKPOINT_UNAVAILABLE $1"; return 0; }
  local snap="/workspace/checkpoint-5/${RUN_ID}.ckpt-$1.tar"
  (cd /workspace/checkpoint-5 && tar --exclude='*/tensors/*' -cf "$snap" "${RUN_ID}" && sha256sum "$(basename "$snap")" > "$snap.sha256") || { echo "CHECKPOINT_FAILED $1"; return 0; }
  [ -n "$SEND_PID" ] && kill "$SEND_PID" 2>/dev/null || true
  nohup "$RPC" send "$snap" > "$snap.send.log" 2>&1 & SEND_PID=$!
  sleep 15; echo "CHECKPOINT $1 $(cat "$snap.sha256")"; echo "CHECKPOINT_SEND_LOG_BEGIN"; head -5 "$snap.send.log"; echo "CHECKPOINT_SEND_LOG_END"
}
export_results() {  # RunPod's supported transfer: send the output folder; the one-time code goes to the log
  if [ -x "$RPC" ]; then
    (cd /workspace/checkpoint-5 && find "${RUN_ID}" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "${RUN_ID}.SHA256SUMS" || true
     echo "EXPORT_MANIFEST=/workspace/checkpoint-5/${RUN_ID}.SHA256SUMS"
     [ -n "$SEND_PID" ] && kill "$SEND_PID" 2>/dev/null || true
     nohup "$RPC" send "/workspace/checkpoint-5/${RUN_ID}" > /workspace/checkpoint-5/${RUN_ID}.send.log 2>&1 &
     sleep 15; echo "EXPORT_SEND_LOG_BEGIN"; head -5 /workspace/checkpoint-5/${RUN_ID}.send.log; echo "EXPORT_SEND_LOG_END") || true
  else
    echo "EXPORT_UNAVAILABLE: runpodctl not installed; results remain on the pod volume disk"
  fi
}
trap 'rc=$?; [ -n "$CKPT_LOOP" ] && kill "$CKPT_LOOP" 2>/dev/null; [ $rc -ne 0 ] && status "FAILED rc=$rc at line $LINENO"; export_results; exit $rc' EXIT
status "started stage=${STAGE} commit=${COMMIT}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
findmnt /workspace || true; df -h /workspace /
python3 --version
# pinned runpodctl for the export (the image's own copy is an older release without the v2+ commands)
mkdir -p /workspace/bin
curl -fsSL -o "$RPC" https://github.com/runpod/runpodctl/releases/download/v3.1.0/runpodctl-linux-amd64
echo "551d30795598a2d14cfdafd7f7bf47f5c6a313fb37a6dc7b401cd1b94b456fd4  $RPC" | sha256sum -c -
chmod +x "$RPC" && "$RPC" version

git clone --quiet https://github.com/fcavaletto/vlm-crossmodal-attack-transfer /root/repo
cd /root/repo && git checkout --quiet "$COMMIT" && git log -1 --format='HEAD=%H %s'
python3 -m venv /root/venv && . /root/venv/bin/activate
pip install --quiet --disable-pip-version-check -r requirements-ci.txt
python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('transformers', transformers.__version__)"
pip freeze > "$OUT/pip_freeze.txt"
export HF_HOME=/workspace/hf-cache PYTHONPATH=$(ls -d /root/repo/src/*/ | tr '\n' ':')
python - <<'PY'
from huggingface_hub import snapshot_download
from qwen_rgb.pins import MODEL_ID, PINNED_REVISION
print("snapshot:", snapshot_download(MODEL_ID, revision=PINNED_REVISION))
PY
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# runner outputs go to the pod volume disk
rm -rf docs/checkpoint-5-qwen-fp32/audit docs/checkpoint-5-qwen-fp32/run
ln -s "$OUT/audit" docs/checkpoint-5-qwen-fp32/audit
ln -s "$OUT/run" docs/checkpoint-5-qwen-fp32/run

if [ "$STAGE" = canary ]; then
  python - <<'PY'
import torch
from qwen_precision.numerics import configure, fp32_matmul_probe
from qwen_precision.load import weights_facts
d = torch.device("cuda")
print("numerics", configure(d)); print("probe", fp32_matmul_probe(d)); print("weights", weights_facts())
PY
  echo "canary export test file" > "$OUT/canary_export_test.txt"
  checkpoint canary
  status "canary-complete"
  exit 0
fi

python -m qwen_precision.run inputs --out "$OUT/audit/input_hashes.json" --compare docs/checkpoint-5-qwen-fp32/reference/input_hashes.json
python -m qwen_precision.run audit --dtype bf16 --device cuda
python -m qwen_precision.run audit --dtype fp32 --device cuda
python -m qwen_precision.run clean --device cuda
python -m qwen_precision.run fd --dtype fp32 --device cuda
status "gates-evaluating"
python -m qwen_precision.gates "$OUT/audit"          # exits non-zero on any failed gate: no main run
status "gates-passed; main-run"
checkpoint gates
( i=0; while sleep "${CHECKPOINT_SECONDS:-3600}"; do i=$((i+1)); checkpoint "main-$i"; done ) & CKPT_LOOP=$!
python -m qwen_precision.run main --device cuda --hourly-usd "$HOURLY_USD" \
  --pod-start-epoch "$POD_START_EPOCH" --deadline-epoch $((DEADLINE_EPOCH - 1800)) --budget-gate-usd 12
python -m qwen_precision.compare_grads "$OUT/run"
status "complete"
