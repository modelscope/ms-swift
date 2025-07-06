# 4 * 32GB
# Multimodal packing currently only supports qwen2_vl, qwen2_5_vl, qwen2_5_omni, internvl2_5/3
# A demo for four modalities that can be run directly
# For local datasets, it is recommended to use streaming: `--streaming true` (save memory)
NPROC_PER_NODE=4 \
ENABLE_AUDIO_OUTPUT=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
MAX_PIXELS=1003520 \
swift sft \
    --model Qwen/Qwen2.5-Omni-7B \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#10000' \
              'AI-ModelScope/LaTeX_OCR#2000' \
              'speech_asr/speech_asr_aishell1_trainsets:validation#2000' \
    --split_dataset_ratio 0.01 \
    --train_type lora \
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
    --freeze_vit true \
    --gradient_accumulation_steps 1 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --deepspeed zero2
