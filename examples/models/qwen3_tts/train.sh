# Execution order: prepare_data.py -> train.sh -> infer.py
# The dataset can also be set to `--dataset qsdong/Qwen3-1.7-TTS-SFT-Furina`,
# but the preprocessing will affect training speed.

SPEAKER_NAME='speaker_test' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
    --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --dataset 'tts_data.parquet' \
    --split_dataset_ratio 0.01 \
    --load_from_cache_file true \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing true \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --ddp_find_unused_parameters true \
    --dataset_num_proc 1 \
    --dataloader_num_workers 4
