# Due to the absence of a multi-modal open-source dataset for kto,
# we will use a pure text kto dataset as an example here.
nproc_per_node=2

CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=$nproc_per_node \
MAX_PIXELS=1003520 \
swift rlhf \
    --rlhf_type kto \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset 'AI-ModelScope/ultrafeedback-binarized-preferences-cleaned-kto#10000' \
    --split_dataset_ratio 0.01 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --deepspeed zero2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4
