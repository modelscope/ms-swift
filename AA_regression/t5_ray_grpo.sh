#!/bin/bash
# Regression smoke test: Megatron Ray GRPO (colocate) + off-policy metrics
cd /mnt/nas2/hujinghan.hjh/swift
export PATH=/mnt/nas2/anaconda3/envs/hjh_h20/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1
megatron rlhf --use_ray true --config AA_regression/t5_ray_grpo.yaml
