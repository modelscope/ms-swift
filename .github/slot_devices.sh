#!/bin/bash
# Print this runner's `CUDA_VISIBLE_DEVICES=...` line, for appending to $GITHUB_ENV.
#
# `container.options` in a workflow is static, so every slot's container is handed all 8 GPUs; the
# slice has to be applied inside the job. RUNNER_NAME is the one per-runner fact Actions exposes to
# a container job, so the box's four runners are named `<host>-0` .. `<host>-3` and each derives its
# own devices with no lock, no shared state and nothing that can go stale -- which is what
# dockerci.sh needed 89 lines of flock for.
set -euo pipefail

slot=${RUNNER_NAME:-}
slot=${slot##*-}
case "$slot" in
'' | *[!0-9]*)
    echo "runner '${RUNNER_NAME:-<unset>}' must be named <host>-<slot index>, e.g. swift-ci-0" >&2
    exit 1
    ;;
esac

per_slot=${GPUS_PER_SLOT:-2}
first=$((slot * per_slot))
echo "CUDA_VISIBLE_DEVICES=$(seq -s, "$first" $((first + per_slot - 1)))"
