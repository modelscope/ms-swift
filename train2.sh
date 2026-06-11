#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR" || exit 1

set -a && source .deepspeed_env && set +a
. "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate swift2

HOSTS=($(awk '!/^\s*#/ && NF {print $1}' randy/hostfile))
HOSTS_STR=$(IFS=,; echo "${HOSTS[*]}")

NPROC_PER_NODE=8
NNODES=${#HOSTS[@]}
MASTER_ADDR=${HOSTS[0]}
MASTER_PORT=${MASTER_PORT:-29500}

PYTHON=$(command -v python)

CONFIG=$(
    $PYTHON randy/train.py "$@" |
    awk -F'<randy>|</randy>' '{print $2}' |
    sed 's| --| \\\n  --|g'
)

pdsh -S -R ssh -w "$HOSTS_STR" "
cd $SCRIPT_DIR || exit 1 && \
bash $(realpath randy/killer.sh) && \
set -a && source .deepspeed_env && set +a && \
$PYTHON -m torch.distributed.run \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=%n \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    $CONFIG
"
