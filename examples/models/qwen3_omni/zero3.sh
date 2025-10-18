# 2 * 60GiB
# mcore shell: https://github.com/modelscope/ms-swift/blob/main/examples/megatron/multimodal/omni/moe.sh
MAX_PIXELS=1003520 \
NPROC_PER_NODE=2 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
    --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#10000' \
              'AI-ModelScope/LaTeX_OCR:human_handwrite#5000' \
              'swift/VideoChatGPT:Generic#2000' \
              'speech_asr/speech_asr_aishell1_trainsets:validation#5000' \
    --split_dataset_ratio 0.01 \
    --load_from_cache_file true \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --attn_impl flash_attn \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --padding_free true \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing true \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataset_num_proc 4 \
    --deepspeed zero3 \
    --dataloader_num_workers 4
