#!/usr/bin/env bash
# Checkpoint 5 GPU workload. Started in the background by watchdog.sh, which owns shutdown.
#   STAGE       canary | full
#   COMMIT      repository commit to run (the protocol commit or later)
#   RUN_ID      output folder name on the network volume
#   POD_START_EPOCH, DEADLINE_EPOCH, HOURLY_USD   for the runner's budget gate and deadline
# Outputs: /workspace/checkpoint-5/$RUN_ID (network volume). Model cache: /workspace/hf-cache.
# Any failed step ends the workload (set -e); the watchdog then deletes the pod after its grace period.
set -euo pipefail
OUT=/workspace/checkpoint-5/${RUN_ID}
mkdir -p "$OUT/audit" "$OUT/run"
exec > >(tee -a "$OUT/pipeline.log") 2>&1
status() { echo "$1" > "$OUT/STATUS"; echo "STATUS=$1 $(date -u +%FT%TZ)"; }
RPC=/root/bin/runpodctl
export_results() {  # RunPod's supported transfer: send the output folder; the one-time code goes to the log
  if [ -x "$RPC" ]; then
    (cd /workspace/checkpoint-5 && find "${RUN_ID}" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "${RUN_ID}.SHA256SUMS" || true
     echo "EXPORT_MANIFEST=/workspace/checkpoint-5/${RUN_ID}.SHA256SUMS"
     nohup "$RPC" send "/workspace/checkpoint-5/${RUN_ID}" > /workspace/checkpoint-5/${RUN_ID}.send.log 2>&1 &
     sleep 15; echo "EXPORT_SEND_LOG_BEGIN"; head -5 /workspace/checkpoint-5/${RUN_ID}.send.log; echo "EXPORT_SEND_LOG_END") || true
  else
    echo "EXPORT_UNAVAILABLE: runpodctl not installed; results remain on the network volume"
  fi
}
trap 'rc=$?; [ $rc -ne 0 ] && status "FAILED rc=$rc at line $LINENO"; export_results; exit $rc' EXIT
status "started stage=${STAGE} commit=${COMMIT}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
findmnt /workspace || true; df -h /workspace /
python3 --version
# pinned runpodctl for the export (the image's own copy is an older release without the v2+ commands)
mkdir -p /root/bin
curl -fsSL -o /root/bin/runpodctl https://github.com/runpod/runpodctl/releases/download/v3.1.0/runpodctl-linux-amd64
echo "551d30795598a2d14cfdafd7f7bf47f5c6a313fb37a6dc7b401cd1b94b456fd4  /root/bin/runpodctl" | sha256sum -c -
chmod +x /root/bin/runpodctl && /root/bin/runpodctl version

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
# runner outputs go to the network volume
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
python -m qwen_precision.run main --device cuda --hourly-usd "$HOURLY_USD" \
  --pod-start-epoch "$POD_START_EPOCH" --deadline-epoch $((DEADLINE_EPOCH - 1800)) --budget-gate-usd 12
python -m qwen_precision.compare_grads "$OUT/run"
status "complete"
