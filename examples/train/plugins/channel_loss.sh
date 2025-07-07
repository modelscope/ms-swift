# use loss_type channel_loss
# channels specifies the channels included in the dataset
# data should have 'channel' field
# eg.
# {"channel": "chat",
#   "messages": [
#     {"role": "system", "content": "You are a helpful assistant"},
#     {"role": "user", "content": "What color do you like?"},
#     {"role": "assistant", "content": "I like blue."}
#   ]}
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --dataset '/path/to/channel_dataset' \
    --split_dataset_ratio 0.01 \
    --train_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 8 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 1 \
    --max_length 512 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --loss_type channel_loss \
    --channels 'chat' 'math' 'code'
