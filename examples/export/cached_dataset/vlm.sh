OMP_NUM_THREADS=14 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
swift export \
    --model Qwen/Qwen2.5-Omni-7B \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#10000' \
              'AI-ModelScope/LaTeX_OCR:human_handwrite#5000' \
              'speech_asr/speech_asr_aishell1_trainsets:validation#5000' \
    --max_length 4096 \
    --split_dataset_ratio 0.01 \
    --dataset_num_proc 16 \
    --to_cached_dataset true \
    --lazy_tokenize false \
    --output_dir ./qwen2_5_omni_cached_dataset

# 4 * 70GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
NPROC_PER_NODE=4 \
ENABLE_AUDIO_OUTPUT=0 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model Qwen/Qwen2.5-Omni-7B \
    --train_type full \
    --cached_dataset './qwen2_5_omni_cached_dataset' \
    --num_train_epochs 1 \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --packing true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --eval_steps 200 \
    --save_steps 200 \
    --logging_steps 5 \
    --max_length 4096 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --save_total_limit 2 \
    --save_only_model true \
    --output_dir output/Qwen2.5-Omni-7B \
    --deepspeed zero2 \
    --use_liger_kernel true \
    --attn_impl flash_attn

# Use the validation set
CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
ENABLE_AUDIO_OUTPUT=0 \
swift infer \
    --model output/Qwen2.5-Omni-7B/vx-xxx/checkpoint-xxx \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#10000' \
              'AI-ModelScope/LaTeX_OCR:human_handwrite#5000' \
              'speech_asr/speech_asr_aishell1_trainsets:validation#5000' \
    --max_length 4096 \
    --split_dataset_ratio 0.01 \
    --attn_impl flash_attn \
    --stream true \
    --temperature 0 \
    --max_new_tokens 512
