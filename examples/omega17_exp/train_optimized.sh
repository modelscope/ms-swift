#!/bin/bash
# Omega17Exp Optimized Training Script
#
# Usage:
#   chmod +x train_optimized.sh
#   ./train_optimized.sh

set -e

# ============================================================
# CONFIGURATION - Modify these values
# ============================================================
MODEL_PATH="../model"
OUTPUT_DIR="../output"
DATASET="tatsu-lab/alpaca"  # Or your custom dataset

# Training parameters (optimized for speed)
BATCH_SIZE=2                 # Increase if GPU memory allows
GRAD_ACCUM=8                 # Effective batch = BATCH_SIZE * GRAD_ACCUM = 16
MAX_LENGTH=512
LORA_RANK=16
LORA_ALPHA=32
EPOCHS=1

# ============================================================
# DO NOT MODIFY BELOW
# ============================================================

echo "============================================================"
echo "OMEGA17EXP OPTIMIZED TRAINING"
echo "============================================================"
echo ""
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET"
echo "Batch size: $BATCH_SIZE x $GRAD_ACCUM = $((BATCH_SIZE * GRAD_ACCUM)) effective"
echo "LoRA: rank=$LORA_RANK, alpha=$LORA_ALPHA"
echo "============================================================"
echo ""

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

# Run training
swift sft \
    --model $MODEL_PATH \
    --model_type omega17_exp \
    --dataset $DATASET \
    --use_hf true \
    --train_type lora \
    --quant_method bnb \
    --quant_bits 4 \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --target_modules q_proj k_proj v_proj o_proj \
    --output_dir $OUTPUT_DIR \
    --max_length $MAX_LENGTH \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --gradient_checkpointing true \
    --optim paged_adamw_8bit \
    --learning_rate 2e-4 \
    --warmup_ratio 0.03 \
    --num_train_epochs $EPOCHS \
    --save_strategy epoch \
    --logging_steps 10 \
    --bf16 true

echo ""
echo "============================================================"
echo "TRAINING COMPLETE!"
echo "============================================================"
echo "Output saved to: $OUTPUT_DIR"
