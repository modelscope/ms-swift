# This script require main branch ms-swift
# This script is intended solely as a Tool Calling training example
# The calculator tool implemented here can perform only basic arithmetic operations and may not be able to solve all math problems in the dataset.
# Before running this script, please run the following `swift rollout` script first

# CUDA_VISIBLE_DEVICES=0 \
# swift rollout \
#     --model Qwen/Qwen2.5-7B-Instruct \
#     --vllm_use_async_engine true \
#     --external_plugins examples/train/grpo/plugin/plugin.py \
#     --multi_turn_scheduler tool_call_scheduler \
#     --vllm_max_model_len 8192 \
#     --vllm_gpu_memory_utilization 0.8 \
#     --max_turns 5

SYSTEM_PROMPT='
Answer the following questions as best you can. You have access to the following tools:

calculator
Purpose: Perform basic arithmetic (+, -, *, /, parentheses) and return the result as text.
Input (single string): the math expression to evaluate, e.g. "2*(3+4)".
Only digits, spaces, and the characters +-*/(). are allowed.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [calculator]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question, the answer should be written as \(\boxed{<answer>}\), e.g. \(\boxed{10}\)

Begin!
'

CUDA_VISIBLE_DEVICES=1,2,3 \
NPROC_PER_NODE=3 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --reward_funcs accuracy \
    --train_type full \
    --torch_dtype bfloat16 \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --dataset 'AI-MO/NuminaMath-TIR#1000' \
    --load_from_cache_file true \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 4 \
    --temperature 0.9 \
    --system "$SYSTEM_PROMPT" \
    --log_completions true \
    --deepspeed zero3 \
    --stop_words "Observation:" \
    --report_to swanlab tensorboard
