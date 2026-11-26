# v5 server: drive client.py to attach + train multi-LoRA adapters and sample them, all
# over HTTP against the twinkle-server started by serve.sh.
#
# Run serve.sh first (in another shell) and wait until it prints that the gateway is up.
# This script only issues client calls; it loads no model weights locally.
#
# Env overrides (all optional):
#   TWINKLE_SERVER_URL   gateway root          (default http://localhost:8000)
#   TWINKLE_SERVER_TOKEN api key               (default EMPTY_TOKEN)
#   TWINKLE_MODEL_ID     public model name     (default Qwen/Qwen3.5-4B)
#   TWINKLE_ADAPTER_NAMES comma-separated LoRA names to train (default assistant,coder)
#   TWINKLE_MAX_STEPS    optimizer steps per adapter (default 6)
#   TWINKLE_DATA_SLICE   self-cognition rows   (default 200)
# MEASURED: client verified against server (2026-11-24)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

export TWINKLE_TRUST_REMOTE_CODE=1

# The client imports twinkle/twinkle_client only; run from a neutral cwd so the repo's
# top-level `twinkle/` package directory cannot shadow the installed one.
export TWINKLE_SERVER_URL=${TWINKLE_SERVER_URL:-http://localhost:8000}
export TWINKLE_SERVER_TOKEN=${TWINKLE_SERVER_TOKEN:-EMPTY_TOKEN}
export TWINKLE_MODEL_ID=${TWINKLE_MODEL_ID:-Qwen/Qwen3.5-4B}

# Wait for the server to be FULLY ready before firing client calls.
#
# Startup race (temporary lengthened wait, NOT an SDK retry change): the gateway
# (/api/v1/models) goes live BEFORE the launcher finishes deploying the training router
# and the sampler pool (the launcher blocks on sampler-pool initialize first). An early
# create_lora_training_client therefore hits a non-retryable 404 even though the gateway
# already answers. So beyond the gateway poll we also wait for the per-model training +
# sampler healthz endpoints, with a longer overall budget, to let those routers deploy.
python - <<'PY'
import os, time, urllib.request
url = os.environ['TWINKLE_SERVER_URL'].rstrip('/')
model = os.environ['TWINKLE_MODEL_ID']

def wait(path, attempts, label):
    for _ in range(attempts):
        try:
            urllib.request.urlopen(url + path, timeout=5)
            print(label, 'ready:', url + path)
            return
        except Exception:
            time.sleep(5)
    raise SystemExit(label + ' not ready at ' + url + path)

# Gateway first (lengthened budget: 240 x 5s = 20min).
wait('/api/v1/models', 240, 'gateway')
# Then the training router + sampler pool, which deploy AFTER the gateway is up.
wait('/api/v1/model/' + model + '/healthz', 240, 'training router')
wait('/api/v1/sampler/' + model + '/healthz', 240, 'sampler pool')
PY

cd /tmp
python "${SCRIPT_DIR}/client.py"
