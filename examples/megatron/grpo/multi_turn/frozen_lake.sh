# Multi-turn GRPO with the FrozenLake env.
# Env lives in frozen_lake_plugin.py (loaded via --external_plugins);
# with --use_gym_env true, the env's total_reward is consumed directly — no reward_funcs needed.
# To prevent excessively long generations, max_completion_length is capped at 512 (per turn);
# since prompts are short, max_length (first 9 turns + prompt) is capped at 6120.
# vllm_max_model_len = max_length + last-turn length = 6632
# reward improves from 0.2 → 0.6 within 120 steps: https://github.com/modelscope/ms-swift/pull/9405

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
megatron rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3.5-2B \
    --enable_thinking false \
    --save_safetensors true \
    --context_parallel_size 1 \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --dataset 'examples/megatron/grpo/multi_turn/frozen_lake.jsonl#1024' \
    --load_from_cache_file false \
    --train_iters 120 \
    --global_batch_size 64 \
    --micro_batch_size 1 \
    --steps_per_generation 4 \
    --num_generations 8 \
    --external_plugins examples/megatron/grpo/multi_turn/frozen_lake_plugin.py \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_max_model_len 6632 \
    --max_length 6120 \
    --max_completion_length 512 \
    --multi_turn_scheduler gym_scheduler \
    --gym_env frozen_lake \
    --use_gym_env true \
    --max_turns 10 \
    --tuner_type lora \
    --lr 5e-5 \
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
    --report_to tensorboard swanlab \
    --eval_steps 1000 \
    --save_steps 1000
