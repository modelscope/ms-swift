# Multi-turn GRPO with the FrozenLake env.
# Env lives in frozen_lake_plugin.py (loaded via --external_plugins);
# with --use_gym_env true, the env's total_reward is consumed directly — no reward_funcs needed.

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
MASTER_PORT=29600 \
megatron rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3.5-2B \
    --save_safetensors true \
    --context_parallel_size 1 \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --dataset 'examples/megatron/grpo/multi_turn/frozen_lake.jsonl#1024' \
    --load_from_cache_file false \
    --num_train_epochs 3 \
    --global_batch_size 64 \
    --micro_batch_size 2 \
    --steps_per_generation 4 \
    --num_generations 8 \
    --external_plugins examples/megatron/grpo/multi_turn/frozen_lake_plugin.py \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_max_model_len 4096 \
    --max_length 4096 \
    --max_completion_length 1024 \
    --multi_turn_scheduler gym_scheduler \
    --gym_env frozen_lake \
    --use_gym_env true \
    --max_turns 10 \
    --tuner_type full \
    --lr 1e-6 \
    --bf16 true \
    --beta 0.001 \
    --importance_sampling_level token \
    --epsilon 0.2 \
    --epsilon_high 0.2 \
    --dynamic_sample false \
    --overlong_filter true \
    --loss_type grpo \
    --sleep_level 2 \
    --offload_model true \
    --offload_bridge false \
    --offload_optimizer true \
    --logging_steps 1 \
    --recompute_granularity selective \
    --finetune \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --no_save_optim \
    --no_save_rng \
    --attention_backend flash \
    --temperature 1.0 \
    --top_p 1.0 \
    --top_k 80 \
    --padding_free true \
    --log_completions true \
    --train_iters 300 \
    --eval_steps 1000 \
    --save_steps 1000
