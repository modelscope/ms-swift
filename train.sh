#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd $SCRIPT_DIR

export CONDA_DIR="$WORK_DIR/miniforge3"
. $CONDA_DIR/etc/profile.d/conda.sh
conda activate swift

outputs=$(python randy/train.py "$@")
command=$(echo "$outputs" | awk -F'<randy>|</randy>' '{print $2}')

command="deepspeed $command"
# command="deepspeed --hostfile randy/hostfile $command"

eval $command
