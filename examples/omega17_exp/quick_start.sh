#!/bin/bash
# ============================================================
# OMEGA17EXP QUICK START - SINGLE COMMAND SETUP & TRAINING
# ============================================================
#
# Usage:
#   1. Upload usf-ms-swift folder to /workspace/
#   2. Set your HuggingFace token below
#   3. Run: bash /workspace/usf-ms-swift/examples/omega17_exp/quick_start.sh
#
# ============================================================

set -e

# ============================================================
# CONFIGURATION - MODIFY THESE VALUES
# ============================================================
HF_TOKEN="${HF_TOKEN:-}"  # Set via environment or modify here
DATASET="tatsu-lab/alpaca"
NUM_EPOCHS=1
LORA_RANK=16
MAX_LENGTH=512
BATCH_SIZE=2
GRAD_ACCUM=8

# ============================================================
# AUTO-DETECT PATHS
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWIFT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_DIR="$SCRIPT_DIR/model"
OUTPUT_DIR="$SWIFT_DIR/output_$(date +%Y%m%d_%H%M%S)"

# ============================================================
# COLORS FOR OUTPUT
# ============================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${GREEN}[STEP]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# ============================================================
# PRE-FLIGHT CHECKS
# ============================================================
echo "============================================================"
echo "OMEGA17EXP QUICK START"
echo "============================================================"
echo ""

# Check HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    print_error "HF_TOKEN not set!"
    echo ""
    echo "Set it with one of these methods:"
    echo "  Option 1: export HF_TOKEN=hf_your_token_here"
    echo "  Option 2: Edit this script and set HF_TOKEN variable"
    echo ""
    exit 1
fi
export HF_TOKEN

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    print_warning "nvidia-smi not found. Make sure CUDA is installed."
else
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

# ============================================================
# STEP 1: INSTALL DEPENDENCIES
# ============================================================
print_step "Installing MS-SWIFT from source..."
cd "$SWIFT_DIR"
pip install -e ".[llm]" -q

print_step "Installing required packages..."
pip install transformers>=4.51 accelerate bitsandbytes>=0.46.1 peft datasets -q

# ============================================================
# STEP 2: RUN ENVIRONMENT PATCHES
# ============================================================
print_step "Running environment patches..."
cd "$SCRIPT_DIR"
python setup_environment.py

# ============================================================
# STEP 3: DOWNLOAD MODEL (if not exists)
# ============================================================
if [ ! -f "$MODEL_DIR/config.json" ]; then
    print_step "Downloading Omega17Exp model..."
    python download_model.py --output_dir "$MODEL_DIR"
else
    print_step "Model already exists at $MODEL_DIR"
fi

# ============================================================
# STEP 4: START TRAINING
# ============================================================
echo ""
echo "============================================================"
echo "STARTING TRAINING"
echo "============================================================"
echo "Model: $MODEL_DIR"
echo "Dataset: $DATASET"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $NUM_EPOCHS"
echo "LoRA Rank: $LORA_RANK"
echo "Batch: $BATCH_SIZE x $GRAD_ACCUM = $((BATCH_SIZE * GRAD_ACCUM)) effective"
echo "============================================================"
echo ""

cd "$SWIFT_DIR"

swift sft \
    --model "$MODEL_DIR" \
    --model_type omega17_exp \
    --dataset "$DATASET" \
    --use_hf true \
    --train_type lora \
    --quant_method bnb \
    --quant_bits 4 \
    --lora_rank $LORA_RANK \
    --lora_alpha $((LORA_RANK * 2)) \
    --target_modules q_proj k_proj v_proj o_proj \
    --output_dir "$OUTPUT_DIR" \
    --max_length $MAX_LENGTH \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --gradient_checkpointing true \
    --optim paged_adamw_8bit \
    --learning_rate 2e-4 \
    --warmup_ratio 0.03 \
    --num_train_epochs $NUM_EPOCHS \
    --logging_steps 10 \
    --save_strategy epoch \
    --bf16 true

echo ""
echo "============================================================"
echo "TRAINING COMPLETE!"
echo "============================================================"
echo "Output saved to: $OUTPUT_DIR"
echo ""
