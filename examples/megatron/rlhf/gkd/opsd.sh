# OPSD Fixed Teacher Mode (Self-Distillation) - Megatron
# Paper: Self-Distilled Reasoner (arXiv:2601.18734)
# Teacher = student model itself (self-distillation, no separate teacher loaded)
# Dataset: open-r1/OpenThoughts-114k-math
# Model: Qwen3-4B
#
# Hyperparameters aligned with paper's run_opsd.sh:
#   lr=2e-5, temp=1.2, beta=0.5, lmbda=1, effective batch=32

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
megatron rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen3-4B \
    --external_plugins examples/train/rlhf/opsd/opsd_plugin.py \
    --dataset 'open-r1/OpenThoughts-114k-math' \
    --lmbda 1.0 \
    --beta 0.5 \
    --temperature 1.2 \
    --sft_alpha 0 \
    --torch_dtype bfloat16 \
    --micro_batch_size 1 \
    --global_batch_size 32 \
    --max_steps 1000 \
    --lr 2e-5 \
    --save_steps 100 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --max_length 8192 \
    --max_completion_length 2048 \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --attention_backend flash \
    --recompute_granularity selective \
    --finetune \
    --no_save_optim \
    --no_save_rng
