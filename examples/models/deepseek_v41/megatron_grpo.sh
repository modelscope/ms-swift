# DeepSeek-V4.1-Flash LoRA GRPO (Megatron + vLLM colocate).
#
# Requirements / notes:
# - Versions -- this is the combination DeepSeek-V4.1 was validated on:
#   * Megatron-LM must be a source checkout, not a megatron-core wheel: v4.1 needs the hybrid
#     stack, Engram and CSA2 code that no release carries. Validated baseline is the dev branch
#     with NVIDIA/Megatron-LM PR #7224 merged in (megatron-core 0.19.0; dev was at 0cd11658f):
#         git clone -b dev https://github.com/NVIDIA/Megatron-LM.git && cd Megatron-LM
#         git fetch origin pull/7224/head:pr7224 && git merge --no-edit pr7224
#     Put it on PYTHONPATH so it shadows any installed megatron-core:
#         export PYTHONPATH=/abs/path/to/Megatron-LM:$PYTHONPATH
#   * ms-swift >= 4.6.0 and mcore-bridge >= 1.7.0, both source installs -- DeepSeek-V4.1 landed
#     after the latest release of either, and requirements/megatron.txt still pins mcore-bridge
#     >= 1.6.3, which predates it:
#         pip install -e /path/to/ms-swift -e /path/to/mcore-bridge
#     The Megatron-LM checkout above is the training side only; the rollout side needs the vLLM
#     build below, which is a separate install.
# - Real DeepSeek-V4.1-Flash is a ~700B MoE model and needs an H200-class multi-node cluster.
#   The rollout side relies on a vLLM build with native DeepSeek-V4.1 support (use the official
#   vllm/vllm-openai:deepseekv41-flash image, there is no pip wheel yet). Numbers below are a
#   starting point, adjust EP/PP/DP and node count to your hardware.
# - Parallelism support (training): CP / PP / EP / DP / VPP are supported. TP and SP are NOT
#   supported yet for DeepSeek-V4.1, so keep --tensor_model_parallel_size 1 and do NOT pass
#   --sequence_parallel. NOTE: --vllm_tensor_parallel_size is the vLLM inference-side TP and is
#   independent of the (unsupported) Megatron training TP -- it is fine to use it for rollout.
#   To use context parallelism (--context_parallel_size > 1), DSv4's hybrid attention requires
#   contiguous CP over packed (THD) inputs, so also add:
#       --cp_partition_mode contiguous --packing true --sequence_packing_scheduler default_dynamic_cp
# - LoRA weight sync exports adapter tensors only (peft format), so the ~183 GiB Engram tables
#   are never touched: they stay in the rollout engine from the base checkpoint. Keep
#   --merge_lora false so the adapter is synced separately instead of merged back.
#   (For full-parameter GRPO the Engram tables are auto-frozen and skipped during weight sync;
#   see examples/megatron/grpo/deepseek_v41_colocate_full.sh.)
# - Keep --bf16: FP4 checkpoint weights are dequantized to BF16 on load; do not pass --fp4.
SYSTEM_PROMPT="Please reason step by step, and put your final answer within \\boxed{}."

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
megatron rlhf \
    --rlhf_type grpo \
    --model deepseek-ai/DeepSeek-V4.1-Flash \
    --dataset open-r1/DAPO-Math-17k-Processed \
    --system "$SYSTEM_PROMPT" \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --bf16 true \
    --context_parallel_size 1 \
    --tensor_model_parallel_size 1 \
    --expert_model_parallel_size 8 \
    --expert_tensor_parallel_size 1 \
    --pipeline_model_parallel_size 4 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --num_train_epochs 1 \
    --global_batch_size 16 \
    --micro_batch_size 1 \
    --steps_per_generation 1 \
    --num_generations 8 \
    --reward_funcs accuracy format \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_tensor_parallel_size 8 \
    --vllm_max_model_len 16384 \
    --max_length 8192 \
    --max_completion_length 8192 \
    --lr 5e-5 \
    --beta 0.00 \
    --importance_sampling_level sequence \
    --epsilon 3e-4 \
    --epsilon_high 4e-4 \
    --overlong_filter true \
    --loss_type grpo \
    --temperature 1.0 \
    --sleep_level 2 \
    --offload_model true \
    --offload_optimizer true \
    --offload_bridge false \
    --save_safetensors true \
    --merge_lora false \
    --recompute_granularity selective \
    --padding_free true \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --attention_backend flash \
    --logging_steps 1 \
    --log_completions true \
    --output_dir output
