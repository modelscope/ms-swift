#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd $SCRIPT_DIR

export CONDA_DIR="$WORK_DIR/miniforge3"
. $CONDA_DIR/etc/profile.d/conda.sh
conda activate swift

export CUDA_VISIBLE_DEVICES=$(python << EOF
import os
import torch
n_gpus = torch.cuda.device_count()
print(os.environ.get(
    'CUDA_VISIBLE_DEVICES',
    ','.join(map(str, range(n_gpus)))
))
EOF
)

export NPROC_PER_NODE=$(python << EOF
import os
print(len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
EOF
)

outputs=$(python randy/train.py "$@")
command=$(echo "$outputs" | awk -F'<randy>|</randy>' '{print $2}')

command="swift $command"

eval $command
