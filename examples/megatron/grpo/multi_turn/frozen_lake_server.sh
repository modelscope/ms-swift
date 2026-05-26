# server mode version of frozen_lake.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift rollout \
    --model Qwen/Qwen3.5-2B \
    --enable_thinking false \
    --vllm_data_parallel_size 4 \
    --multi_turn_scheduler gym_scheduler \
    --gym_env frozen_lake \
    --use_gym_env true \
    --max_turns 10 \
    --max_length 6120 \
    --vllm_max_model_len 6632 \
    --external_plugins examples/megatron/grpo/multi_turn/frozen_lake_plugin.py > frozen_lake_rollout.log 2>&1 &

# Wait for rollout server to be ready
echo "Waiting for rollout server to start..."
until curl -s http://127.0.0.1:8000/health/ > /dev/null 2>&1; do
    sleep 10
done
echo "Rollout server is ready!"

CUDA_VISIBLE_DEVICES=4,5,6,7 \
NPROC_PER_NODE=4 \
megatron rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen3.5-2B \
    --save_safetensors true \
    --context_parallel_size 1 \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --dataset 'examples/megatron/grpo/multi_turn/frozen_lake.jsonl#1024' \
    --load_from_cache_file false \
    --global_batch_size 32 \
    --micro_batch_size 1 \
    --steps_per_generation 4 \
    --num_generations 8 \
    --external_plugins examples/megatron/grpo/multi_turn/frozen_lake_plugin.py \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --max_length 6120 \
    --max_completion_length 512 \
    --enable_thinking false \
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
    --report_to swanlab \
    --swanlab_project ms-swift-grpo-frozen-lake \
    --swanlab_exp_name nonray-cap8192-thinkingoff \
    --train_iters ${TRAIN_ITERS:-120} \
    --eval_steps 1000 \
    --save_steps 1000 \
    "$@"
