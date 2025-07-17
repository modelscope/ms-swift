# with padding_free: 4 * 53GiB, 3.55s/it
# without padding_free: 4 * 62GiB 4.41s/it
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
MAX_PIXELS=1003520 \
swift rlhf \
    --rlhf_type dpo \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset 'swift/RLAIF-V-Dataset#20000' \
    --split_dataset_ratio 0.01 \
    --train_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-5 \
    --freeze_vit true \
    --gradient_accumulation_steps 1 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --deepspeed zero3 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --attn_impl flash_attn \
    --save_only_model true \
    --padding_free true
