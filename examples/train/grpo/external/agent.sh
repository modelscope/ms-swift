# wandb result link: https://wandb.ai/tastelikefeet/tastelikefeet?nw=nwuseryuzezyz
# model link: https://www.modelscope.cn/models/swift/Qwen2-7B-Agent-GRPO
# WANDB_API_KEY=xxx \

# CUDA_VISIBLE_DEVICES=7 \
# swift rollout \
#     --model Qwen/Qwen2.5-7B

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
NPROC_PER_NODE=7 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B \
    --train_type full \
    --dataset LLM-Research/xlam-function-calling-60k:grpo \
    --load_from_cache_file true \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --max_length 2048 \
    --per_device_train_batch_size 7 \
    --per_device_eval_batch_size 7 \
    --eval_steps 2000 \
    --save_steps 2000 \
    --learning_rate 1e-6 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --max_completion_length 1024 \
    --reward_funcs toolbench react_format \
    --num_generations 49 \
    --deepspeed zero3 \
    --temperature 1.0 \
    --stop_words Observation: \
    --agent_template react_grpo \
    --top_p 0.85 \
    --top_k 50 \
    --log_completions true \
    --report_to wandb
