# 4 * 50GiB
# Currently, it only supports the case where the model and reward_model use the same template/tokenizer.
# Currently, multimodal model PPO is not supported.

# pip install "deepspeed==0.14.*"
nproc_per_node=4

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=$nproc_per_node \
swift rlhf \
    --rlhf_type ppo \
    --model LLM-Research/Meta-Llama-3.1-8B-Instruct \
    --reward_model 'AI-ModelScope/Skywork-Reward-Llama-3.1-8B-v0.2' \
    --train_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#20000' 'AI-ModelScope/alpaca-gpt4-data-en#20000' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero2 \
    --response_length 512 \
    --temperature 0.7 \
    --dataset_num_proc 4 \
    --save_only_model true
