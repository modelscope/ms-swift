#!/bin/bash
# DeepSeek-V4.1 full-parameter SFT (Megatron backend).
#
# Install Megatron-LM with DeepSeek-V4.1 support (CSA2 + Engram + HybridModel), then ms-swift
# and the mcore-bridge:
#   pip install "git+https://github.com/tastelikefeet/Megatron-LM.git@dsv41-pr7224-engram-local"
#   pip install -e .              # ms-swift (run from the repo root)
#   pip install -e mcore-bridge   # DeepSeek-V4.1 Megatron bridge/loader
#
# --model is the standard model id, downloaded from the hub on first run.
#
# Engram is driven by the model config (engram_layer_ids), not a CLI flag. Under full-parameter
# SFT it stays trainable and participates in the backward pass.
# DeepSeek-V4.1 does not support TP/SP: keep tensor_model_parallel_size 1, no --sequence_parallel.
#
#   bash examples/models/deepseek_v41/megatron_sft.sh          # GPU 0
#   GPUS=3 bash examples/models/deepseek_v41/megatron_sft.sh   # choose a GPU
set -e
cd "$(git rev-parse --show-toplevel)"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES=${GPUS:-0} \
megatron sft \
    --model deepseek-ai/DeepSeek-V4.1-Flash \
    --dataset 'AI-ModelScope/LaTeX_OCR:human_handwrite#2000' \
    --split_dataset_ratio 0 \
    --tuner_type full \
    --context_parallel_size 1 \
    --tensor_model_parallel_size 1 \
    --expert_model_parallel_size 1 \
    --expert_tensor_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --micro_batch_size 1 \
    --global_batch_size 2 \
    --max_length 512 \
    --train_iters 100 \
    --lr 1e-4 \
    --min_lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --bf16 true \
    --finetune true \
    --recompute_granularity none \
    --masked_softmax_fusion false \
    --attention_backend flash \
    --logging_steps 1 \
    --eval_iters 0 \
    --save_steps 100 \
    --no_save_optim true \
    --no_save_rng true \
    --save_safetensors true \
    --dataloader_num_workers 1 \
    --dataset_num_proc 1 \
    --output_dir megatron_output/dsv41-sft
