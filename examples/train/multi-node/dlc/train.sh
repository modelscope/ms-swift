# For more information, visit: https://www.aliyun.com/activity/bigdata/pai-dlc
# https://help.aliyun.com/zh/pai/user-guide/general-environment-variables
NNODES=$WORLD_SIZE \
NODE_RANK=$RANK \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type full \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#20000' \
              'AI-ModelScope/alpaca-gpt4-data-en#20000' \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero2
