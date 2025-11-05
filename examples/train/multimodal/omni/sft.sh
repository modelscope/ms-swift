# 4*35GB
# A demo for four modalities that can be run directly
nproc_per_node=4

# If using zero3, please set `ENABLE_AUDIO_OUTPUT=0`.
CUDA_VISIBLE_DEVICES=0,1,2,3 \
ENABLE_AUDIO_OUTPUT=1 \
NPROC_PER_NODE=$nproc_per_node \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
MAX_PIXELS=1003520 \
swift sft \
    --model Qwen/Qwen2.5-Omni-7B \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#2000' \
              'AI-ModelScope/LaTeX_OCR:human_handwrite#2000' \
              'speech_asr/speech_asr_aishell1_trainsets:validation#2000' \
              'swift/VideoChatGPT:all#2000' \
    --load_from_cache_file true \
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
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero2
