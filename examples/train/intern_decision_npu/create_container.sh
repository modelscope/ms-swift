#!/usr/bin/env bash
# Create a fresh, isolated four-card workspace on an Ascend host.
set -euo pipefail
framework=${1:?swift or twinkle}
container=${2:?new container name}
runtime=$(realpath "${3:?new runtime directory}")
source_dir=$(realpath "${4:?framework checkout directory}")
model=$(realpath "${5:?base checkpoint directory}")
data=$(realpath "${6:?data directory containing joint and prepared}")
case "$framework" in swift|twinkle) ;; *) exit 2 ;; esac
image=quay.nju.edu.cn/ascend/ms-swift@sha256:d1b56f2d77882edb92615c45641556c8d5adaf8ed360be73f4f0978f70fa01c1
test -z "$(ls -A "$runtime")"
docker run -d --name "$container" --privileged --network host --ipc host --shm-size 32g \
  -v "$runtime:/workspace" -v "$source_dir:/workspace/framework:ro" \
  -v "$model:/models/Qwen3.5-4B:ro" -v "$data:/data:ro" \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  "$image" bash -c 'rm -f /dev/davinci4 /dev/davinci5 /dev/davinci6 /dev/davinci7; exec sleep infinity'
docker exec "$container" python -m venv --system-site-packages /workspace/.venv
if [[ "$framework" == twinkle ]]; then
  docker exec "$container" /workspace/.venv/bin/python -m pip install --no-deps peft==0.19.0
  # PYTHONPATH in run_joint.sh selects this exact checkout; distribution metadata
  # is installed without copying or modifying the read-only checkout.
  docker exec "$container" bash -c 'cp -a /workspace/framework /workspace/package-build; /workspace/.venv/bin/python -m pip install --no-deps /workspace/package-build'
fi
