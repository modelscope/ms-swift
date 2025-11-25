#!/bin/bash
# Circle-RoPE Training Launcher Script
# This script simplifies the process of launching Circle-RoPE training

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
NUM_GPUS=4
CONFIG="examples/circle_rope/full_sft_zero2.yaml"
MODEL_PATH=""
DATASET=""
MASTER_PORT=29500

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --port)
            MASTER_PORT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --num_gpus NUM     Number of GPUs to use (default: 4)"
            echo "  --config PATH      Path to config file (default: examples/circle_rope/full_sft_zero2.yaml)"
            echo "  --model PATH       Override model path in config"
            echo "  --dataset NAME     Override dataset in config"
            echo "  --port PORT        Master port for distributed training (default: 29500)"
            echo "  --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  # LoRA training (single GPU)"
            echo "  $0 --num_gpus 1 --config examples/circle_rope/sft.yaml --model /path/to/model"
            echo ""
            echo "  # Full training with ZeRO-2 (4 GPUs)"
            echo "  $0 --num_gpus 4 --config examples/circle_rope/full_sft_zero2.yaml"
            echo ""
            echo "  # Full training with ZeRO-3 (8 GPUs)"
            echo "  $0 --num_gpus 8 --config examples/circle_rope/full_sft_zero3.yaml"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate config file exists
if [ ! -f "$CONFIG" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG${NC}"
    exit 1
fi

# Build command
CMD_ARGS="--config $CONFIG"
if [ -n "$MODEL_PATH" ]; then
    CMD_ARGS="$CMD_ARGS --model $MODEL_PATH"
fi
if [ -n "$DATASET" ]; then
    CMD_ARGS="$CMD_ARGS --dataset $DATASET"
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Circle-RoPE Training Configuration${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Number of GPUs: ${YELLOW}$NUM_GPUS${NC}"
echo -e "Config file:    ${YELLOW}$CONFIG${NC}"
echo -e "Master port:    ${YELLOW}$MASTER_PORT${NC}"
if [ -n "$MODEL_PATH" ]; then
    echo -e "Model path:     ${YELLOW}$MODEL_PATH${NC}"
fi
if [ -n "$DATASET" ]; then
    echo -e "Dataset:        ${YELLOW}$DATASET${NC}"
fi
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if circle_rope module is accessible
echo -e "${YELLOW}Checking circle_rope module...${NC}"
python -c "import sys; sys.path.insert(0, '.'); from circle_rope import register_circle_rope; register_circle_rope()" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Circle-RoPE module found${NC}"
else
    echo -e "${YELLOW}⚠ Circle-RoPE module not found in current directory${NC}"
    echo -e "${YELLOW}  Make sure circle_rope folder is copied to your model directory${NC}"
fi
echo ""

# Select launcher based on number of GPUs
if [ $NUM_GPUS -eq 1 ]; then
    echo -e "${GREEN}Launching single-GPU training...${NC}"
    swift sft $CMD_ARGS
else
    echo -e "${GREEN}Launching multi-GPU training with $NUM_GPUS GPUs...${NC}"

    # Use deepspeed for better ZeRO support
    if grep -q "zero3" "$CONFIG"; then
        echo -e "${YELLOW}Using DeepSpeed launcher (ZeRO-3 detected)${NC}"
        deepspeed --num_gpus=$NUM_GPUS --master_port=$MASTER_PORT \
            $(which swift) sft $CMD_ARGS
    else
        echo -e "${YELLOW}Using torchrun launcher${NC}"
        NPROC_PER_NODE=$NUM_GPUS \
        torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
            $(which swift) sft $CMD_ARGS
    fi
fi
