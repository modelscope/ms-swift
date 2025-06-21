
export teacher_model='Qwen/Qwen3-8B'

NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift infer \
    --model $teacher_model \
    --infer_backend vllm \
    --val_dataset 'AI-ModelScope/alpaca-gpt4-data-en#5000' 'AI-ModelScope/alpaca-gpt4-data-zh#5000' \
    --gpu_memory_utilization 0.9 \
    --max_model_len 8192 \
    --max_new_tokens 2048 \
    --write_batch_size 10000 \
    --result_path new_dataset.jsonl


# 4 * 67GiB, 2.50s/it
# You need to additionally add sft_loss, because tokens like '<think>' have not been trained.
NPROC_PER_NODE=4 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen3-8B-Base \
    --teacher_model $teacher_model \
    --train_type full \
    --dataset 'new_dataset.jsonl' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --save_only_model true \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --deepspeed zero3 \
    --packing true \
    --attn_impl flash_attn \
    --sft_alpha 0.1 \
    --lmbda 0
