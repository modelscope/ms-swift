# v5 models / qwen3_next: MoE LoRA SFT of Qwen3-Next-80B-A3B on the Megatron backend.
#
# model_type = `qwen3_next` (swift/dev/model/loader/qwen.py:762; Qwen3NextForCausalLM, template
# `qwen3_nothinking`, is_moe). Siblings: `qwen3_next_thinking` (Qwen3-Next-80B-A3B-Thinking, template
# qwen3_thinking) and `qwen3_next_coder` (Qwen3-Coder-Next, template qwen3_coder) -- all resolved from
# --model. Qwen3-Next is a large sparse MoE (80B total, 3B active) with GatedDeltaNet linear attention;
# the FP8 training path decouples in_proj via --linear_decoupled_in_proj (see ../../../docs Megatron
# Command-line-parameters). This uses `swift sft --backend megatron` with expert parallelism.
#
# A transformers-backend LoRA on a single node is also possible for the 80B-A3B (3B active) with heavy
# recompute/offload; the megatron layout below is the recommended path for a sparse model this size.
#
# REFERENCE-ONLY: not smoke-run here. Qwen3-Next-80B-A3B needs a multi-GPU Megatron cluster (expert +
# pipeline parallel) and mcore-bridge; it is far beyond a single box. model_type is verified against
# swift/dev/model/loader/qwen.py; the Megatron MoE flag set mirrors examples/train/megatron/moe/train.sh.
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NNODES=${NNODES:-1} \
NODE_RANK=${NODE_RANK:-0} \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
USE_SWIFT_V5=1 \
swift sft \
    --backend megatron \
    --model Qwen/Qwen3-Next-80B-A3B-Instruct \
    --task_type causal_lm \
    --dataset 'swift/self-cognition#600' \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --expert_model_parallel_size 8 \
    --moe_token_dispatcher_type alltoall \
    --moe_grouped_gemm true \
    --moe_permute_fusion true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --attention_backend flash \
    --lr 1e-4 \
    --min_lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --num_train_epochs 3 \
    --max_length 4096 \
    --logging_steps 5 \
    --save_steps 100 \
    --save_safetensors true \
    --output_dir output \
    --report_to tensorboard
