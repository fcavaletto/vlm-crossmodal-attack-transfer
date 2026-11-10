#!/usr/bin/env bash
# Checkpoint 5 GPU workload. Started in the background by watchdog.sh, which owns shutdown.
#   STAGE       canary (canary only) | full (canary, operator go, gates, main run)
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
    (cd /workspace/checkpoint-5 && find "${RUN_ID}" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "${RUN_ID}/SHA256SUMS" || true
     echo "EXPORT_MANIFEST=/workspace/checkpoint-5/${RUN_ID}/SHA256SUMS (inside the sent folder)"; cat "${RUN_ID}/SHA256SUMS"
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

# ---- GPU canary (amendment 2): runs first in the same pod, before any scientific step -----------
export OUT
mkdir -p "$OUT/canary"
nvidia-smi > "$OUT/canary/nvidia-smi.txt" 2>&1 || true
python - <<'PY'
import json, os, torch
from qwen_precision.numerics import configure, fp32_matmul_probe
from qwen_precision.load import weights_facts
d = torch.device("cuda")
cap = torch.cuda.get_device_capability(0)
arch = torch.cuda.get_arch_list()
need = f"sm_{cap[0]}{cap[1]}"
rec = {"device": torch.cuda.get_device_name(0), "capability": list(cap), "arch_list": arch, "needed_arch": need,
       "arch_supported": need in arch, "torch": torch.__version__, "cuda": torch.version.cuda,
       "cudnn": torch.backends.cudnn.version()}
rec["numerics"] = configure(d)
rec["probe"] = fp32_matmul_probe(d)
rec["weights"] = weights_facts()
ker = {}
for name, dt in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
    try:  # the op families Qwen3-VL uses (conv3d patch embed, attention, norms, activations), forward and backward
        x = torch.randn(2, 3, 4, 32, 32, device=d, dtype=dt, requires_grad=True)
        conv = torch.nn.Conv3d(3, 8, (2, 4, 4), stride=(2, 4, 4)).to(d, dt)
        q = torch.randn(1, 4, 64, 32, device=d, dtype=dt, requires_grad=True)
        att = torch.softmax(q @ q.transpose(-1, -2) / 32 ** 0.5, dim=-1) @ q
        y = (conv(x).float().sum() + att.float().sum() + torch.nn.functional.layer_norm(q, (32,)).float().sum()
             + torch.nn.functional.silu(q).float().sum() + torch.nn.functional.gelu(q).float().sum())
        y.backward(); torch.cuda.synchronize()
        ker[name] = {"ok": bool(torch.isfinite(x.grad).all() and torch.isfinite(q.grad).all())}
    except Exception as e:
        ker[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
rec["kernels"] = ker
json.dump(rec, open(os.environ["OUT"] + "/canary/canary.json", "w"), indent=1, default=str)
print("CANARY", json.dumps({k: rec[k] for k in ("device", "capability", "needed_arch", "arch_supported", "kernels")}))
print("CANARY_PROBE", rec["probe"])
assert rec["arch_supported"], f"{need} not in the torch build's arch list {arch}"
assert all(v["ok"] for v in ker.values()), f"kernel smoke failed: {ker}"
PY
# watchdog: PID 1, script hash, absolute deadline, boot count, and the pod key's right to call the stop endpoint
PID1=$(tr '\0' ' ' < /proc/1/cmdline); echo "WATCHDOG_PID1=${PID1}"
case "$PID1" in *watchdog.sh*) ;; *) echo "CANARY_FAIL watchdog is not PID 1"; exit 4;; esac
sha256sum /tmp/watchdog.sh | tee "$OUT/canary/watchdog.sha256"
echo "WATCHDOG_BOOTS=$(cat /workspace/.watchdog/boots 2>/dev/null)" | tee -a "$OUT/canary/watchdog.sha256"
echo "DEADLINE_EPOCH=${DEADLINE_EPOCH} remaining_s=$((DEADLINE_EPOCH - $(date +%s)))" | tee -a "$OUT/canary/watchdog.sha256"
code=$(curl -sS --max-time 30 -o "$OUT/canary/stop_auth_probe.json" -w '%{http_code}' -X POST -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H 'content-type: application/json' --data '{"action":"canary-probe-not-an-action"}' "https://api.runpod.io/v2/pods/${RUNPOD_POD_ID}/action")
echo "STOP_ENDPOINT_AUTH_PROBE_HTTP=${code}" | tee -a "$OUT/canary/watchdog.sha256"   # 400/422: authorised, request rejected as invalid
case "$code" in 400|422) ;; *) echo "CANARY_FAIL the pod key cannot reach the stop endpoint"; exit 5;; esac
status "canary-checks-passed"
checkpoint canary
if [ "$STAGE" = canary ]; then status "canary-complete"; exit 0; fi
# hold until the operator has received the canary bundle off the running pod and verified its sha256.
# Signal: infra/runpod/signals/$RUN_ID on the research branch, "go <sha256 of the canary tar>" or "nogo <reason>".
WANT=$(cut -d' ' -f1 "/workspace/checkpoint-5/${RUN_ID}.ckpt-canary.tar.sha256")
SIGNAL="https://api.github.com/repos/fcavaletto/vlm-crossmodal-attack-transfer/contents/infra/runpod/signals/${RUN_ID}?ref=${SIGNAL_REF:-research/qwen-fp32-gpu}"
HOLD_END=$(( $(date +%s) + ${CANARY_HOLD_SECONDS:-2400} ))
status "canary-hold"
while :; do
  if body=$(curl -fsS --max-time 30 -H 'Accept: application/vnd.github.raw' "$SIGNAL" 2>/dev/null); then
    set -- $body
    if [ "${1:-}" = go ] && [ "${2:-}" = "$WANT" ]; then echo "CANARY_GO verified=$WANT"; break; fi
    echo "CANARY_NOGO_OR_MISMATCH signal='${body}' want=${WANT}"; exit 6
  fi
  [ "$(date +%s)" -ge "$HOLD_END" ] && { echo "CANARY_HOLD_TIMEOUT"; exit 7; }
  sleep 90
done
status "canary-go"

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
