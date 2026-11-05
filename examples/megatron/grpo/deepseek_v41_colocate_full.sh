# DeepSeek-V4.1-Flash full-parameter GRPO (Megatron + vLLM colocate).
#
# Requirements / notes:
# - Install the project's Megatron-LM with DeepSeek-V4.1 support (CSA2 + Engram + HybridModel),
#   then ms-swift and the mcore-bridge:
#     pip install "git+https://github.com/tastelikefeet/Megatron-LM.git@dsv41-pr7224-engram-local"
#     pip install -e .              # ms-swift (run from the repo root)
#     pip install -e mcore-bridge   # DeepSeek-V4.1 Megatron bridge/loader
# - Real DeepSeek-V4.1-Flash needs an H200-class cluster; the rollout side relies on a
#   vLLM build with native DeepSeek-V4.1 support (use the official
#   vllm/vllm-openai:deepseekv41-flash image, there is no pip wheel yet).
# - Engram tables are auto-frozen under full-parameter GRPO (requires_grad=False, excluded
#   from the optimizer). They stay resident in the rollout engine from the base checkpoint
#   and are skipped during per-step weight sync (a full ~183 GiB resync is infeasible).
# - Adjust EP/PP and node count to your hardware; the values below are a starting point.
# - Parallelism support (training): CP / PP / EP / DP / VPP are supported. TP and SP are NOT
#   supported yet for DeepSeek-V4.1, so keep --tensor_model_parallel_size 1 and do NOT pass
#   --sequence_parallel. --vllm_tensor_parallel_size is the vLLM inference-side TP and is
#   independent of the (unsupported) Megatron training TP.
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
megatron rlhf \
    --rlhf_type grpo \
    --model deepseek-ai/DeepSeek-V4.1-Flash \
    --save_safetensors true \
    --context_parallel_size 1 \
    --tensor_model_parallel_size 1 \
    --expert_model_parallel_size 8 \
    --pipeline_model_parallel_size 4 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --dataset open-r1/DAPO-Math-17k-Processed \
    --num_train_epochs 1 \
    --global_batch_size 8 \
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
    --tuner_type full \
    --lr 1e-6 \
    --bf16 true \
    --beta 0.00 \
    --importance_sampling_level sequence \
    --epsilon 3e-4 \
    --epsilon_high 4e-4 \
    --dynamic_sample false \
    --overlong_filter true \
    --loss_type grpo \
    --sleep_level 2 \
    --offload_model true \
    --offload_bridge false \
    --offload_optimizer true \
    --optimizer_cpu_offload true \
    --use_precision_aware_optimizer \
    --logging_steps 1 \
    --recompute_granularity selective \
    --finetune \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim \
    --no_save_rng \
    --attention_backend flash \
    --temperature 1.0 \
    --padding_free true \
    --log_completions true \
    --report_to wandb
