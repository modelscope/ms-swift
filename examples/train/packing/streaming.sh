# 4 * 36GB
# A demo using the Hugging Face dataset
# The first model weights will be saved around step 70.
NPROC_PER_NODE=4 \
MAX_PIXELS=1003520 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
HF_ENDPOINT=https://hf-mirror.com \
swift sft \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --train_type lora \
    --dataset 'HF::linxy/LaTeX_OCR:full#20000' \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --streaming true \
    --shuffle_buffer_size 1000 \
    --packing true \
    --save_strategy epoch \
    --max_steps 1000 \
    --max_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 1 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 1 \
    --dataset_num_proc 8 \
    --deepspeed zero2
