# 4 * 22GiB
# vit/merger lr 1e-5; llm lora lr 1e-4
# Note: not support resume_from_checkpoint (only support resume_only_model)
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MAX_PIXELS=1003520 \
swift sft \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset 'AI-ModelScope/coco#20000' \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --train_type custom \
    --external_plugins 'examples/train/multimodal/lora_llm_full_vit/custom_plugin.py' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --vit_lr 1e-5 \
    --aligner_lr 1e-5 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --gradient_accumulation_steps 4 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --deepspeed zero2 \
    --save_only_model true
