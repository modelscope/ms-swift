# GRPO for text-to-SQL with an execution reward on BIRD (SQLite).
# 1. Download the BIRD train set and databases from https://bird-bench.github.io/
# 2. python examples/train/grpo/plugin/text2sql/prepare_bird.py \
#        --input train/train.json --db_root train/train_databases --output bird_train.jsonl
# 3. Run this script. Evaluate on BIRD Mini-Dev, which is disjoint from train.
export TEXT2SQL_DB_ROOT=train/train_databases
export TEXT2SQL_TIMEOUT=30
export TEXT2SQL_SKIP_EMPTY_GOLD=1

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3-4B \
    --external_plugins examples/train/grpo/plugin/text2sql/text2sql_plugin.py \
    --reward_funcs text2sql_exec text2sql_format \
    --reward_weights 1.0 0.1 \
    --dataset bird_train.jsonl \
    --enable_thinking false \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.4 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len 10240 \
    --sleep_level 1 \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --max_length 8192 \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-6 \
    --lr_scheduler_type cosine \
    --save_steps 50 \
    --save_total_limit 10 \
    --logging_steps 1 \
    --warmup_ratio 0.0 \
    --dataloader_num_workers 4 \
    --num_generations 8 \
    --temperature 1.0 \
    --deepspeed zero2 \
    --log_completions true \
    --report_to tensorboard \
    --max_grad_norm 1.0 \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --scale_rewards none
