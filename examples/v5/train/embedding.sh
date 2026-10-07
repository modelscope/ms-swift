# v5 embedding training (task_type=embedding) with the InfoNCE objective on a single GPU.
#
# `--task_type embedding` dispatches to run_embedding, which pools the last token into a sentence
# vector; `--loss infonce` selects the contrastive InfoNCE objective (others: cosine_similarity,
# contrastive, online_contrastive). Each row carries an anchor `messages` plus `positive_messages` and
# `negative_messages` -- each of those a LIST OF conversations (one per candidate), not a single one.
# Embedding has no classification head, so LoRA trains cleanly here.
#
# HOW TO READ THIS RUN: the InfoNCE loss alone is an opaque scalar, so run_embedding registers twinkle's
# EmbeddingMetric -- watch the tensorboard `pos_sim` (anchor-to-positive cosine, should RISE) and `neg_sim`
# (anchor vs in-batch negatives, should stay flat or FALL); a widening pos_sim - neg_sim gap is the embedding
# model converging. (`accuracy` is auto-registered but is a no-op here -- embeddings carry no class label.)
#
# Dataset: `stsb:positive#2000` is a REAL hub set -- the STS-B sentence-similarity pairs, filtered to the
# genuinely-similar ones (score >= 0.75) the `positive` subset keeps, so each anchor's partner is a TRUE
# positive. InfoNCE needs no explicit negatives: it contrasts each anchor against the OTHER rows of the batch
# (in-batch negatives, all-gathered across ranks). Reference it by the REGISTERED name (`stsb:positive`), not
# the raw `sentence-transformers/stsb` id, so the InfoNCE preprocessor emits `positive_messages`. The 0.5B LM
# keeps the example light; for a production embedder use a dedicated checkpoint (e.g. Qwen/Qwen3-Embedding-0.6B).
#
# ATTENTION BACKEND IS LOAD-BEARING -- keep `--attn_impl flash_attention_2`. Left unset, attn_impl=None and
# transformers picks SDPA, which on a current torch/cuDNN build resolves to the cuDNN attention kernel whose
# bf16 BACKWARD returns NaN: the run trains cleanly for ~15 steps, then grad_norm goes non-finite (it drops out
# of the log line entirely -- the tracker's coercion discards non-finite scalars) and every subsequent step is
# loss=nan. Verified: default SDPA/cuDNN -> nan by step ~18; flash_attention_2 -> 40+ steps clean, loss 1.17->0.02.
# `--attn_impl eager` also avoids the cuDNN kernel if flash-attn is unavailable. NOTE the token: the dev
# (transformers) backend passes attn_impl straight through to HF, so it must be `flash_attention_2`, NOT legacy
# swift's `flash_attn` (which HF rejects). The reference qwen3_emb.sh pins flash attention for the same reason.
#
# BATCH SIZE IS THE NEGATIVE COUNT. `--per_device_train_batch_size 8` gives each anchor 7 in-batch negatives
# (the reference qwen3_emb.sh on this same dataset uses 8); a batch of 2 leaves only ONE negative per anchor,
# which makes the contrastive task trivially separable and starves the objective. Keep the batch >= 8 and
# --learning_rate at 5e-5 (both the reference values).
# OFFLINE fallback (no hub): the bundled `examples/v5/train/data/embedding.jsonl`.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift sft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --task_type embedding \
    --loss infonce \
    --attn_impl flash_attention_2 \
    --torch_dtype bfloat16 \
    --dataset 'stsb:positive#2000' \
    --tuner lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --lr_scheduler cosine \
    --warmup_ratio 0.05 \
    --max_length 512 \
    --logging_steps 1 \
    --save_steps 100 \
    --save_total_limit 2 \
    --output_dir output \
    --report_to tensorboard
