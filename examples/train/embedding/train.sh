nproc_per_node=2

CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model Qwen/Qwen2.5-7B \
    --train_type full \
    --dataset 'sentence-transformers/stsb' \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps $(expr 64 / $nproc_per_node) \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4
