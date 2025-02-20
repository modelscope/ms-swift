# pip install lmdeploy==0.6.4
# Replace three files:
# 1. https://github.com/tastelikefeet/lmdeploy/blob/feat/reload_state_dict_064/lmdeploy/messages.py
# 2. https://github.com/tastelikefeet/lmdeploy/blob/feat/reload_state_dict_064/lmdeploy/turbomind/turbomind.py
# 3. https://github.com/tastelikefeet/lmdeploy/blob/feat/reload_state_dict_064/lmdeploy/turbomind/deploy/loader.py

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=3 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --reward_funcs accuracy format \
    --use_lmdeploy true \
    --lmdeploy_session_len 2048 \
    --lmdeploy_cache_max_entry_count 0.8 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset 'AI-MO/NuminaMath-TIR#5000' \
    --max_completion_length 2048 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 2 \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --num_generations 3 \
    --temperature 0.9 \
    --system 'examples/train/grpo/prompt.txt' \
    --deepspeed zero3 \
    --log_completions true
