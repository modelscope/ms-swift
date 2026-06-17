#!/bin/bash
# Regression smoke test: Megatron Ray GKD (colocate train+rollout, colocated teacher)
cd /mnt/nas2/hujinghan.hjh/swift
export PATH=/mnt/nas2/anaconda3/envs/hjh_h20/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1
megatron rlhf --use_ray true --config AA_regression/t6_ray_gkd.yaml
