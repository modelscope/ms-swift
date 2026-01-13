NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
    --model Qwen/Qwen3-0.6B \
    --train_type full \
    --dataset 'swift_shuf_19k_data.jsonl' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 16 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 16384 \
    --output_dir swift_output/Qwen3-0.6B/eaft \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --enable_eaft_loss true \
    --eaft_alpha 1.0 \
    --deepspeed zero3 \
    --report_to tensorboard \
    --logging_dir tensorboard/swift_output/Qwen3-0.6B/eaft \
