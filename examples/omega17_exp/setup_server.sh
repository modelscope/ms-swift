#!/bin/bash
# Omega17Exp Training Server Setup Script
# 
# This script sets up everything needed to fine-tune Omega17Exp on a new server.
# Uses MS-SWIFT from source with integrated Omega17Exp support.
#
# Usage:
#   chmod +x setup_server.sh
#   ./setup_server.sh

set -e

echo "============================================================"
echo "OMEGA17EXP TRAINING SERVER SETUP"
echo "============================================================"

# Configuration
WORK_DIR="/workspace/omega17_finetune"
MODEL_ID="arpitsh018/omega17exp-prod-v1.1"

# Create working directory
echo ""
echo "[1/6] Creating working directory..."
mkdir -p $WORK_DIR
cd $WORK_DIR

# Clone MS-SWIFT with Omega17Exp integration
echo ""
echo "[2/6] Setting up MS-SWIFT with Omega17Exp..."
if [ ! -d "ms-swift" ]; then
    # Option A: Clone from your repo (uncomment and modify URL)
    # git clone https://github.com/YOUR_REPO/usf-ms-swift.git ms-swift
    
    # Option B: Copy from upload (if you uploaded the code)
    if [ -d "/workspace/usf-ms-swift" ]; then
        cp -r /workspace/usf-ms-swift ms-swift
    else
        echo "ERROR: Please upload the usf-ms-swift code to /workspace/usf-ms-swift"
        echo "Or modify this script to clone from your git repository"
        exit 1
    fi
fi

# Install MS-SWIFT from source
echo ""
echo "[3/6] Installing MS-SWIFT from source..."
cd ms-swift
pip install -e ".[llm]" --quiet

# Install dependencies (standard transformers for Qwen3Moe-like performance)
echo ""
echo "[4/6] Installing dependencies..."
pip install transformers>=4.51 accelerate bitsandbytes peft datasets --quiet

# Set HuggingFace token
echo ""
echo "[5/6] Setting up HuggingFace token..."
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set. Set it with:"
    echo "  export HF_TOKEN=your_token_here"
fi

# Download model
echo ""
echo "[6/6] Downloading Omega17Exp model..."
cd $WORK_DIR
python -c "
from huggingface_hub import snapshot_download
import os

token = os.environ.get('HF_TOKEN')
if not token:
    print('ERROR: Set HF_TOKEN environment variable')
    exit(1)

print('Downloading model...')
snapshot_download(
    repo_id='$MODEL_ID',
    local_dir='./model',
    token=token
)
print('Download complete!')
"

echo ""
echo "============================================================"
echo "SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "Model location: $WORK_DIR/model"
echo "MS-SWIFT location: $WORK_DIR/ms-swift"
echo ""
echo "To start training, run:"
echo ""
echo "cd $WORK_DIR/ms-swift"
echo "swift sft \\"
echo "    --model ../model \\"
echo "    --model_type omega17_exp \\"
echo "    --dataset tatsu-lab/alpaca \\"
echo "    --use_hf true \\"
echo "    --train_type lora \\"
echo "    --quant_method bnb \\"
echo "    --quant_bits 4 \\"
echo "    --lora_rank 16 \\"
echo "    --lora_alpha 32 \\"
echo "    --target_modules q_proj k_proj v_proj o_proj \\"
echo "    --output_dir ../output \\"
echo "    --max_length 512 \\"
echo "    --per_device_train_batch_size 2 \\"
echo "    --gradient_accumulation_steps 8 \\"
echo "    --gradient_checkpointing true \\"
echo "    --optim paged_adamw_8bit \\"
echo "    --num_train_epochs 1"
echo ""
