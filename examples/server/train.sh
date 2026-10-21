# v5 server: drive client.py to attach + train multi-LoRA adapters and sample them, all
# over HTTP against the twinkle-server started by serve.sh.
#
# Run serve.sh first (in another shell) and wait until it prints that the gateway is up.
# This script only issues client calls; it loads no model weights locally.
#
# Env overrides (all optional):
#   TWINKLE_SERVER_URL   gateway root          (default http://localhost:8000)
#   TWINKLE_SERVER_TOKEN api key               (default EMPTY_TOKEN)
#   TWINKLE_MODEL_ID     public model name     (default Qwen/Qwen2.5-0.5B-Instruct)
#   TWINKLE_ADAPTER_NAMES comma-separated LoRA names to train (default assistant,coder)
#   TWINKLE_MAX_STEPS    optimizer steps per adapter (default 6)
#   TWINKLE_DATA_SLICE   self-cognition rows   (default 200)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

export USE_SWIFT_V5=1
export TWINKLE_TRUST_REMOTE_CODE=1

# The client imports twinkle/twinkle_client only; run from a neutral cwd so the repo's
# top-level `twinkle/` package directory cannot shadow the installed one.
export TWINKLE_SERVER_URL=${TWINKLE_SERVER_URL:-http://localhost:8000}
export TWINKLE_SERVER_TOKEN=${TWINKLE_SERVER_TOKEN:-EMPTY_TOKEN}
export TWINKLE_MODEL_ID=${TWINKLE_MODEL_ID:-Qwen/Qwen2.5-0.5B-Instruct}

# Wait for the gateway to accept connections before firing client calls.
python - <<'PY'
import os, time, urllib.request
url = os.environ['TWINKLE_SERVER_URL'].rstrip('/')
for _ in range(120):
    try:
        urllib.request.urlopen(url + '/api/v1/models', timeout=5)
        print('gateway ready:', url)
        break
    except Exception:
        time.sleep(5)
else:
    raise SystemExit('gateway not ready at ' + url)
PY

cd /tmp
python "${SCRIPT_DIR}/client.py"
