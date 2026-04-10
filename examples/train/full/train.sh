# 76GiB
CUDA_VISIBLE_DEVICES=0 \
MASTER_PORT=29905 \
swift sft \
    --model Qwen/Qwen3-1.7B \
    --tuner_type full \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#5000' \
              'AI-ModelScope/alpaca-gpt4-data-en#5000' \
              'swift/self-cognition#5000' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 16 \
    --packing true \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot \
    --attn_impl flash_attn
