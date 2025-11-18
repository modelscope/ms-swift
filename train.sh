#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd $SCRIPT_DIR

export CONDA_DIR="/nas_train/$USER/miniforge3"
source $CONDA_DIR/etc/profile.d/conda.sh
conda activate swift

# export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA="mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_8,mlx5_9,mlx5_10,mlx5_11"
export NCCL_SOCKET_IFNAME="enp24s0np0,enp25s0np0,enp66s0np0,ens5np0,ens7np0,ens8np0,ens10np0,ens11np0"
export GLOO_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
export TP_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME

outputs=$(python randy/train.py "$@")
command=$(echo "$outputs" | awk -F'<randy>|</randy>' '{print $2}')

deepspeed $command
# deepspeed --hostfile randy/hostfile $command
