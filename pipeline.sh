#!/bin/bash

set_env() {
    grep -q "^$1=" .deepspeed_env \
        && sed -i "s|^$1=.*|$1=$2|" .deepspeed_env \
        || echo "$1=$2" >> .deepspeed_env
}

pdsh_run() {
    hosts=$(grep -v '^\s*#' randy/hostfile | awk 'NF {print $1}' | paste -sd,)
    pdsh -S -R ssh -w "$hosts" "$@"
}

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR" || exit 1

pdsh_run "bash $(realpath randy/killer.sh)"

# ./train.sh randy

./train.sh randy/openbee/llava_4b_1.yaml
./train.sh randy/openbee/llava_4b_2.yaml
./train.sh randy/openbee/llava_4b_3.yaml
./train.sh randy/openbee/llava_4b_4.yaml

pdsh_run "/nas_train/app.e0016372/tools/train.sh"
