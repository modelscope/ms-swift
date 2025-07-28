# Env: 4 * A100
# GPU Memory: 4 * 52GiB, Training Speed 4s/it
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift rlhf \
    --rlhf_type dpo \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type full \
    --dataset hjh0119/shareAI-Llama3-DPO-zh-en-emoji \
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
    --warmup_ratio 0.05 \
    --save_only_model true \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --deepspeed zero3 \
    --attn_impl flash_attn \
    --padding_free true \
    --sequence_parallel_size 2
