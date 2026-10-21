# v5 trainable_tokens tuner: train ONLY chosen embedding rows, freezing the rest of the model.
#
# `--tuner trainable_tokens` builds a peft TrainableTokens adapter that unfreezes just the embedding rows
# listed in `--trainable_token_indices` (int token ids) and trains nothing else. It is the standard way to
# learn embeddings for NEWLY ADDED special tokens: pass --new_special_tokens to extend the vocabulary
# (dev resizes the embedding), then list the new ids -- they start at the pre-extension tokenizer length
# (for Qwen2.5-0.5B-Instruct that is 151665, so the first added token is 151665, the next 151666, ...).
# --tuner trainable_tokens REQUIRES --trainable_token_indices (dev raises without it).
#
# Trainable tokens also ride alongside LoRA: `--tuner lora --trainable_token_indices ...` adapts the model
# with LoRA while additionally learning those embedding rows -- the usual production pattern when you add
# domain tokens and still want a full adaptation. Standalone (shown here) is for when the base is meant to
# stay frozen and only the new-token embeddings should move.
#
# This runnable demo points the indices at a handful of frequent EXISTING token ids of
# Qwen2.5-0.5B-Instruct (you=9330, are=546, the=1782, assistant=77091, What=3838, I=40, a=64, to=983) so
# gradients flow and the loss moves without needing a dataset that contains brand-new tokens; swap in your
# own new-special-token ids for the real workflow.
CUDA_VISIBLE_DEVICES=0 \
USE_SWIFT_V5=1 \
swift sft \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --task_type causal_lm \
    --torch_dtype bfloat16 \
    --dataset 'swift/self-cognition#600' \
    --tuner trainable_tokens \
    --trainable_token_indices 9330 546 1782 77091 3838 40 64 983 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-3 \
    --lr_scheduler cosine \
    --warmup_ratio 0.05 \
    --max_length 2048 \
    --logging_steps 5 \
    --save_steps 100 \
    --save_total_limit 2 \
    --split_dataset_ratio 0.01 \
    --output_dir output \
    --report_to tensorboard
