# 4 * 80GiB
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model ZhipuAI/GLM-4-9B-0414 \
    --train_type full \
    --dataset AI-ModelScope/function-calling-chatml \
    --split_dataset_ratio 0.01 \
    --agent_template hermes \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 2 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 8192 \
    --save_only_model true \
    --packing true \
    --deepspeed zero3 \
    --use_liger_kernel true \
    --output_dir output \
    --warmup_ratio 0.05 \
    --attn_impl flash_attn \
    --dataloader_num_workers 4 \
    --dataset_num_proc 16
