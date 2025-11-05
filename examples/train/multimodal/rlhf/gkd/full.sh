# 4 * 45GiB, 10.29s/it
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MASTER_PORT=29501 \
NPROC_PER_NODE=4 \
swift rlhf \
    --rlhf_type gkd \
    --model OpenGVLab/InternVL3-2B-Pretrained \
    --teacher_model OpenGVLab/InternVL3-8B \
    --dataset 'modelscope/coco_2014_caption:validation#2000' \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --train_type full \
    --seq_kd true \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-5 \
    --freeze_vit true \
    --gradient_accumulation_steps 1 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --deepspeed zero2 \
    --attn_impl flash_attn \
    --logging_steps 5 \
    --max_length 4096 \
    --max_completion_length 512 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --save_only_model true
