# 4 * 36GB
# Multimodal packing currently only supports qwen2_vl, qwen2_5_vl, qwen2_5_omni, internvl2_5/3
# Efficiency: With packing: 10 minutes; Without packing: >=1 hour
# For local datasets, it is recommended to use streaming: `--streaming true` (save memory)
# You can also use padding_free to avoid the space/time cost caused by multi-modal packing:
# https://github.com/modelscope/ms-swift/blob/main/examples/train/padding_free/sft.sh

NPROC_PER_NODE=4 \
MAX_PIXELS=1003520 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --train_type lora \
    --dataset 'AI-ModelScope/LaTeX_OCR#20000' \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --packing true \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 1 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --deepspeed zero2
