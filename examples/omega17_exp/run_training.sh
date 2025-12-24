#!/bin/bash
# ============================================================
# Omega17Exp LoRA SFT Training Script for RunPod
# Model: arpitsh018/omega17exp-prod-v1.1
# MS-SWIFT 3.x
# ============================================================

# IMPORTANT: Set your HuggingFace token before running
# export HF_TOKEN=your_token_here

set -e

echo "============================================================"
echo "OMEGA17EXP LORA SFT TRAINING"
echo "============================================================"

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "❌ ERROR: HF_TOKEN environment variable is not set!"
    echo "   Run: export HF_TOKEN=your_token_here"
    exit 1
fi

echo "✅ HF_TOKEN is set"

# Install dependencies (ORDER MATTERS!)
echo ""
echo "📦 Installing dependencies..."
# 1. Install ms-swift first
pip install ms-swift[llm] -q
# 2. Install custom transformers AFTER (overwrites huggingface-hub to required version)
pip install transformers-usf-om-vl-exp-v0 --force-reinstall -q
# 3. Other dependencies
pip install accelerate bitsandbytes peft datasets -q

# 4. Apply patches to fix compatibility issues (REQUIRED!)
echo ""
echo "🔧 Applying environment patches..."
python setup_environment.py

# Download model if not exists
MODEL_DIR="./model"
if [ ! -d "$MODEL_DIR" ] || [ ! -f "$MODEL_DIR/config.json" ]; then
    echo ""
    echo "📥 Downloading model from HuggingFace..."
    python download_model.py --output_dir $MODEL_DIR
else
    echo ""
    echo "✅ Model already downloaded at $MODEL_DIR"
fi

# Register the custom model
echo ""
echo "📦 Registering Omega17Exp model..."
python -c "import register_omega17; print('✅ Model registered')"

# Run training using MS-SWIFT CLI
echo ""
echo "🚀 Starting LoRA SFT training..."
echo "============================================================"

swift sft \
    --model $MODEL_DIR \
    --model_type omega17_exp \
    --dataset alpaca-en \
    --train_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --output_dir ./output/omega17_lora \
    --max_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing true \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --logging_steps 10 \
    --save_steps 500

echo ""
echo "============================================================"
echo "✅ Training completed!"
echo "   Output: ./output/omega17_lora"
echo "============================================================"
