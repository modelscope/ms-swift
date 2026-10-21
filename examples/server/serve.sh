# v5 server: start the twinkle-server that hosts a swift-loaded, multi-LoRA trainable
# model + a vLLM sampler, driven directly (v5 has no `swift server` command).
#
# KEY GOTCHA: twinkle-server's launcher does ray.init(address='auto'), which only ATTACHES
# to an existing Ray cluster and never starts one. So we must `ray start --head` first.
# This script is idempotent: it reuses a running cluster on RAY_PORT if one is up.
#
# What gets deployed is defined entirely by server_config.yaml (server + model + sampler +
# processor apps). The model app names swift.dev.model.loader:build_model_loader, so
# ms-swift MUST be importable in this env (its Ray actors inherit it) -- from the ms-swift
# root run `pip install -e .` first.
#
# Blocks in the foreground until Ctrl-C. Run client.py (via train.sh) from another shell.
set -euo pipefail
cd "$(dirname "$0")"

export USE_SWIFT_V5=1
export TWINKLE_TRUST_REMOTE_CODE=1

# 2 GPUs by default: 1 for the model app, 1 for the vLLM sampler (device_group ranks in the
# YAML). GPU0 is often busy on shared boxes, so default to 1,2. Override with GPUS=.
GPUS=${GPUS:-1,2}
RAY_PORT=${RAY_PORT:-6379}
RAY_ADDRESS=127.0.0.1:${RAY_PORT}

# Fail fast with a clear message if the toolchain or swift loader is missing.
command -v ray >/dev/null 2>&1 || { echo 'ray not installed; install server deps first' >&2; exit 1; }
command -v twinkle-server >/dev/null 2>&1 || { echo 'twinkle-server not installed; pip install -e ".[client,server]"' >&2; exit 1; }
python -c 'import swift.dev.model.loader' >/dev/null 2>&1 \
  || { echo 'ms-swift not importable; from the ms-swift root run: pip install -e .' >&2; exit 1; }

if ! ray status --address="${RAY_ADDRESS}" >/dev/null 2>&1; then
  CUDA_VISIBLE_DEVICES=${GPUS} ray start \
    --head \
    --port="${RAY_PORT}" \
    --num-gpus=2 \
    --include-dashboard=false \
    --disable-usage-stats
fi

twinkle-server check-config -c server_config.yaml
exec twinkle-server launch -c server_config.yaml
