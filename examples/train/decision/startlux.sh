# Typed-decision (System-1) fine-tuning of StartLux-Decision v1.1 AutoThink.
#
# StartLux scores ONE question per forward (each question is its own prompt holding the whole state),
# so StartLuxPreprocessor fans a record out into one row per question. The read-out takes the option
# LETTERS' lm_head logits (A, B, ...) at the last prompt position -- the prompt is rendered with the
# checkpoint's own chat template, thinking off, so it ends with `<think>\n\n</think>` exactly as the
# model was trained -- and softmax(z / T_kind) over the question's own options, with the per-kind
# temperatures fitted in the release's decision_config.json. `--model_type startlux` loads the merged
# Qwen3.5 hybrid checkpoint (vision tower + causal attention) and attaches that read-out as
# `scoring_head`.
#
# This covers the one-pass read-out (the official `think_mode: never`); the AutoThink gate/thought
# loop is inference orchestration in the official startlux-decide package and stays out of scope.
# `--loss_type distill_rps` is KL(target || model) over each question's options (+ RPS on `score`),
# so `answers` may be one-hot gold or `targets` full soft distributions. StartLux ships a merged
# backbone, so a fresh `--tuner_type lora --target_modules all-linear` is correct. Records are never
# truncated (the read-out sits at the last token), so keep --max_length generous. Packing /
# padding_free is not supported (records would attend to each other).
#
# sh examples/train/decision/startlux.sh
MODELSCOPE_ENDPOINT=https://www.modelscope.ai \
USE_HF=0 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --external_plugins examples/train/decision/dataset.py \
    --model StartLuxAI/StartLux-Decision-v1.1-27B-AutoThink \
    --model_type startlux \
    --task_type decision \
    --loss_type distill_rps \
    --dataset startlux_decision \
    --tuner_type lora \
    --target_modules all-linear \
    --freeze_vit true \
    --torch_dtype bfloat16 \
    --attn_impl sdpa \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --logging_steps 1 \
    --save_steps 50 \
    --save_total_limit 2 \
    --max_length 8192 \
    --split_dataset_ratio 0 \
    --output_dir output \
    --dataloader_num_workers 1
