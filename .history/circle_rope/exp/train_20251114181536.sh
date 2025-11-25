#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REGISTER_PATH="${SCRIPT_DIR}/register.py"

CONFIG=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$CONFIG" ]; then
    echo "Error: --config <json_path> is required"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "Error: JSON config file not found: $CONFIG"
    exit 1
fi

ENV_PREFIX=$(python - "$CONFIG" <<'PY'
import json, sys, shlex
with open(sys.argv[1]) as f:
    cfg = json.load(f)
env = cfg.get('env', {})
pairs = []
for k, v in env.items():
    s = str(v)
    pairs.append(f"{k}={shlex.quote(s)}")
print(' '.join(pairs))
PY
)

CMD_ARGS=$(python - "$CONFIG" <<'PY'
import json, sys, shlex
with open(sys.argv[1]) as f:
    cfg = json.load(f)
args = cfg.get('args', {})
tokens = []
for k, v in args.items():
    tokens.append(f"--{k}")
    if isinstance(v, list):
        tokens.extend([str(x) for x in v])
    elif isinstance(v, dict):
        import json as _json
        tokens.append(_json.dumps(v))
    else:
        tokens.append(str(v))
print(' '.join(shlex.quote(t) for t in tokens))
PY
)

echo "Circle-RoPE Directory: ${SCRIPT_DIR}"
echo "Register Path: ${REGISTER_PATH}"
echo "JSON Config: ${CONFIG}"
echo "Env: ${ENV_PREFIX}"
echo "Args: ${CMD_ARGS}"

eval ${ENV_PREFIX} \
swift sft \
    --custom_register_path "${REGISTER_PATH}" \
    --local_repo_path "${SCRIPT_DIR}" \
    ${CMD_ARGS}

