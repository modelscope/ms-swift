#!/bin/bash
# DeepSeek-V4.1 full-parameter GRPO (Megatron + vLLM colocate).
#
# Install Megatron-LM with DeepSeek-V4.1 support (CSA2 + Engram + HybridModel), then ms-swift
# and the mcore-bridge:
#   pip install "git+https://github.com/tastelikefeet/Megatron-LM.git@dsv41-pr7224-engram-local"
#   pip install -e .              # ms-swift (run from the repo root)
#   pip install -e mcore-bridge   # DeepSeek-V4.1 Megatron bridge/loader
# The rollout side needs a vLLM with native DeepSeek-V4.1 support. Native support is now merged
# into vLLM main (DeepseekV41ForCausalLM), so build from source off main, e.g.:
#   pip install "git+https://github.com/vllm-project/vllm.git@main"
# The official vllm/vllm-openai:deepseekv41-flash image also works.
#
# --model below is a local 4-layer random-weight checkpoint for a fast smoke test; replace it
# with a real model such as deepseek-ai/DeepSeek-V4.1-Flash for actual training.
#
# Engram is driven by the model config (engram_layer_ids), not a CLI flag. Under full-parameter
# GRPO it is frozen (requires_grad=False, skipped in the vLLM weight sync) but still runs in the
# forward pass. DeepSeek-V4.1 does not support TP/SP: keep tensor_model_parallel_size 1, no
# --sequence_parallel (--vllm_tensor_parallel_size is the independent inference-side TP).
#
#   bash examples/models/deepseek_v41/megatron_grpo.sh          # GPU 0
#   GPUS=3 bash examples/models/deepseek_v41/megatron_grpo.sh   # choose a GPU
set -e
cd "$(git rev-parse --show-toplevel)"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES=${GPUS:-0} \
megatron rlhf \
    --rlhf_type grpo \
    --model .temp/dsv41_tiny_sft \
    --dataset 'open-r1/DAPO-Math-17k-Processed' \
    --split_dataset_ratio 0 \
    --tuner_type full \
    --context_parallel_size 1 \
    --tensor_model_parallel_size 1 \
    --expert_model_parallel_size 1 \
    --expert_tensor_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --num_generations 2 \
    --steps_per_generation 1 \
    --global_batch_size 2 \
    --micro_batch_size 1 \
    --max_length 256 \
    --max_completion_length 128 \
    --train_iters 100 \
    --reward_funcs accuracy format \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.25 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len 512 \
    --vllm_enforce_eager true \
    --lr 1e-5 \
    --min_lr 1e-6 \
    --lr_warmup_fraction 0.1 \
    --bf16 true \
    --beta 0.00 \
    --epsilon 0.2 \
    --loss_type grpo \
    --importance_sampling_level token \
    --temperature 1.0 \
    --sleep_level 2 \
    --offload_model true \
    --offload_bridge false \
    --offload_optimizer true \
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
    --log_completions true \
    --output_dir megatron_output/dsv41-grpo
