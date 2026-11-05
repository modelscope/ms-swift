# Typed-decision (System-1) fine-tuning of THX-02.
#
# THX-02 reads ALL of a record's questions jointly in ONE noncausal forward: every question ends with a
# readout marker, and a linear readout over the answer codes (A, B, ...) is applied to the last hidden
# state at each marker, then divided by the calibrated temperature. `--model_type thx02` loads the Qwen3.5
# hybrid backbone (vision + gated-deltanet / full attention) from THX-02's own config.json, switches its
# full-attention layers to bidirectional attention, and attaches the readout as `scoring_head`.
#
# `--loss_type distill_rps` is KL(target || model) over each question's options (+ RPS on `score`), so
# `targets` may be one-hot gold answers or full soft distributions (e.g. a teacher's probabilities).
# The readout is re-activated for joint training AUTOMATICALLY by ScoringTrainer and saved beside the
# adapter. THX-02 ships a merged backbone, so a fresh `--tuner_type lora --target_modules all-linear` is
# correct. Packing / padding_free is not supported (records would attend to each other).
#
# sh examples/train/decision/thx02.sh
MODELSCOPE_ENDPOINT=https://www.modelscope.ai \
USE_HF=0 \
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --external_plugins examples/train/decision/dataset.py \
    --model HAL-X/THX-02 \
    --model_type thx02 \
    --task_type decision \
    --loss_type distill_rps \
    --dataset thx02_decision \
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
