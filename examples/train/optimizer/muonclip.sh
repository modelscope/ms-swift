nproc_per_node=1 \
CUDA_VISIBLE_DEVICES=0 \
MASTER_PORT=29501 \
NPROC_PER_NODE=$nproc_per_node
swift sft \
    --model Qwen/Qwen2.5-1.5B-Instruct  \
    --tuner_type lora  \
    --optimizer muonclip \
    --optim_args "qk_clip_threshold=10000" \
    --dataset swift/self-cognition \
    --torch_dtype bfloat16 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 2e-4 \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 5 \
    --logging_steps 1 \
    --max_length 2048 \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.1 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot \
    --dataset_num_proc 4
