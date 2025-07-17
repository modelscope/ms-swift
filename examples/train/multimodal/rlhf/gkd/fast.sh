
export teacher_model='OpenGVLab/InternVL3-8B'

NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift infer \
    --model $teacher_model \
    --infer_backend vllm \
    --val_dataset 'modelscope/coco_2014_caption:validation#5000' \
    --gpu_memory_utilization 0.9 \
    --max_model_len 8192 \
    --max_new_tokens 2048 \
    --write_batch_size 1000 \
    --result_path new_coco_dataset.jsonl


# 4 * 42GiB, 3.05s/it
NPROC_PER_NODE=4 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift rlhf \
    --rlhf_type gkd \
    --model OpenGVLab/InternVL3-2B-Pretrained \
    --teacher_model $teacher_model \
    --train_type full \
    --dataset 'new_coco_dataset.jsonl' \
    --split_dataset_ratio 0.01 \
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
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --save_only_model true \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --deepspeed zero2 \
    --padding_free true \
    --attn_impl flash_attn \
    --lmbda 0
