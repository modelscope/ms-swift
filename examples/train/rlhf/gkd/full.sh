# 4 * 66GiB, 46s/it
NPROC_PER_NODE=4 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift rlhf \
    --rlhf_type gkd \
    --model Qwen/Qwen2.5-7B \
    --teacher_model Qwen/Qwen2.5-14B-Instruct \
    --train_type full \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-en#2000' 'AI-ModelScope/alpaca-gpt4-data-zh#2000' \
    --seq_kd true \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --max_completion_length 512 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --save_only_model true \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --deepspeed zero3 \
    --attn_impl flash_attn
