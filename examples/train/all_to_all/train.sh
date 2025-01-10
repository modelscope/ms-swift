# 70 GiB * 2
nproc_per_node=2
NPROC_PER_NODE=$nproc_per_node \
CUDA_VISIBLE_DEVICES=0,2 \
max_position_embeddings=10240 \
image_area=518400 \
swift sft \
    --model BAAI/Emu3-Gen \
    --train_type lora \
    --dataset 'swift/TextCaps#40' \
    --loss_scale react \
    --tools_prompt react_zh \
    --torch_dtype bfloat16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 4 \
    --warmup_ratio 0.03 \
    --eval_steps 500 \
    --save_steps 500 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 1024 \
    --weight_decay 0.1 \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}'
