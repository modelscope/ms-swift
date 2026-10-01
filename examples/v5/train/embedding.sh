# v5 embedding training (task_type=embedding) with the InfoNCE objective on a single GPU.
#
# `--task_type embedding` dispatches to run_embedding, which pools the last token into a sentence
# vector; `--loss infonce` selects the contrastive InfoNCE objective (others: cosine_similarity,
# contrastive, online_contrastive). Each row carries an anchor `messages` plus `positive_messages` and
# `negative_messages` -- each of those a LIST OF conversations (one per candidate), not a single one.
# Embedding has no classification head, so LoRA trains cleanly here.
#
# The dataset is the tiny bundled sample; for a real run point --dataset at an embedding set such as
# `sentence-transformers/stsb` and use a dedicated embedding checkpoint (e.g. Qwen/Qwen3-Embedding-0.6B).
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift sft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --task_type embedding \
    --loss infonce \
    --torch_dtype bfloat16 \
    --dataset examples/v5/train/data/embedding.jsonl \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler cosine \
    --warmup_ratio 0.05 \
    --max_length 512 \
    --logging_steps 1 \
    --save_steps 100 \
    --output_dir output \
    --report_to tensorboard
