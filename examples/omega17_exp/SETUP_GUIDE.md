# Omega17Exp Fine-Tuning Setup Guide

**One-time setup that works on any new GPU server.**

---

## Prerequisites

- GPU with 24GB+ VRAM (e.g., RTX 4090, A100, H100)
- CUDA 11.8+ installed
- Python 3.10+
- HuggingFace account with access to `arpitsh018/omega17exp-prod-v1.1`

---

## Quick Start (Copy-Paste Ready)

### Step 1: Upload Code to Server

Upload the entire `usf-ms-swift` folder to `/workspace/` on your server.

```bash
# On your local machine (example with scp)
scp -r /path/to/usf-ms-swift user@server:/workspace/
```

### Step 2: Run Complete Setup (Single Command Block)

SSH into your server and run this entire block:

```bash
# ============================================================
# OMEGA17EXP COMPLETE SETUP - COPY THIS ENTIRE BLOCK
# ============================================================

# Set your HuggingFace token (REQUIRED - replace with your token)
export HF_TOKEN="hf_your_token_here"

# Go to working directory
cd /workspace/usf-ms-swift

# ============================================================
# INSTALL ORDER IS CRITICAL - DO NOT CHANGE
# ============================================================

# Step A: Install MS-SWIFT from source FIRST
pip install -e ".[llm]"

# Step B: Install custom transformers fork AFTER ms-swift
# This MUST come after ms-swift to override huggingface-hub version
pip install transformers-usf-om-vl-exp-v0 --force-reinstall

# Step C: Install other dependencies
pip install accelerate bitsandbytes>=0.46.1 peft datasets

# Step D: Run environment patches (fixes all compatibility issues)
cd examples/omega17_exp
python setup_environment.py

# Step E: Download model
python download_model.py --output_dir ./model

echo "============================================================"
echo "SETUP COMPLETE! Ready to train."
echo "============================================================"
```

> ⚠️ **IMPORTANT**: The install order matters!
> 1. MS-SWIFT first (sets up base dependencies)
> 2. Custom transformers fork AFTER (overrides huggingface-hub)
> 3. Other packages
> 4. Run patches

### Step 3: Start Training (Single Command)

```bash
cd /workspace/usf-ms-swift

swift sft \
    --model examples/omega17_exp/model \
    --model_type omega17_exp \
    --dataset tatsu-lab/alpaca \
    --use_hf true \
    --train_type lora \
    --quant_method bnb \
    --quant_bits 4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules q_proj k_proj v_proj o_proj \
    --output_dir ./output \
    --max_length 512 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing true \
    --optim paged_adamw_8bit \
    --learning_rate 2e-4 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_strategy epoch
```

---

## All-in-One Script

Create and run this script for fully automated setup:

```bash
cat << 'EOF' > /workspace/run_omega17_training.sh
#!/bin/bash
set -e

# ============================================================
# CONFIGURATION - MODIFY THESE
# ============================================================
export HF_TOKEN="hf_your_token_here"  # <-- PUT YOUR TOKEN HERE
SWIFT_DIR="/workspace/usf-ms-swift"
MODEL_DIR="$SWIFT_DIR/examples/omega17_exp/model"
OUTPUT_DIR="$SWIFT_DIR/output"

# ============================================================
# SETUP (runs once)
# ============================================================
echo "Starting Omega17Exp setup..."

cd $SWIFT_DIR

# Install dependencies
pip install -e ".[llm]" -q
pip install transformers>=4.51 accelerate bitsandbytes>=0.46.1 peft datasets -q

# Run patches
cd examples/omega17_exp
python setup_environment.py

# Download model if not exists
if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "Downloading model..."
    python download_model.py --output_dir ./model
fi

# ============================================================
# TRAINING
# ============================================================
echo "Starting training..."
cd $SWIFT_DIR

swift sft \
    --model $MODEL_DIR \
    --model_type omega17_exp \
    --dataset tatsu-lab/alpaca \
    --use_hf true \
    --train_type lora \
    --quant_method bnb \
    --quant_bits 4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules q_proj k_proj v_proj o_proj \
    --output_dir $OUTPUT_DIR \
    --max_length 512 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing true \
    --optim paged_adamw_8bit \
    --learning_rate 2e-4 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_strategy epoch

echo "Training complete! Output: $OUTPUT_DIR"
EOF

chmod +x /workspace/run_omega17_training.sh
```

Then run:
```bash
/workspace/run_omega17_training.sh
```

---

## What setup_environment.py Fixes

The script automatically patches these issues:

| Patch | Issue | Fix |
|-------|-------|-----|
| 1 | BACKENDS_MAPPING missing 'tf' | Adds dummy tf backend |
| 2 | transformers.deepspeed missing | Creates stub module |
| 3 | ROPE_INIT_FUNCTIONS missing 'default' | Adds default key |
| 4 | swift train_args.py logging_dir bug | Patches attribute error |
| 5 | omega17_exp not registered | Registers model in swift |
| 6 | @check_model_inputs decorator error | Removes decorator |
| 7 | bitsandbytes missing | Installs for QLoRA |
| 8 | Omega17Exp native registration | Registers in transformers |

---

## Custom Dataset Training

### Option A: HuggingFace Dataset
```bash
swift sft \
    --model examples/omega17_exp/model \
    --model_type omega17_exp \
    --dataset your-org/your-dataset \
    --use_hf true \
    # ... rest of parameters
```

### Option B: Local JSON/JSONL File
```bash
swift sft \
    --model examples/omega17_exp/model \
    --model_type omega17_exp \
    --dataset /path/to/your/data.jsonl \
    # ... rest of parameters
```

Dataset format (JSONL):
```json
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}
{"messages": [{"role": "user", "content": "How are you?"}, {"role": "assistant", "content": "I'm doing well!"}]}
```

---

## Multi-GPU Training

```bash
# 4 GPUs with DeepSpeed ZeRO-2
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model examples/omega17_exp/model \
    --model_type omega17_exp \
    --dataset tatsu-lab/alpaca \
    --use_hf true \
    --train_type lora \
    --deepspeed examples/omega17_exp/ds_config_zero2.json \
    --quant_method bnb \
    --quant_bits 4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules q_proj k_proj v_proj o_proj \
    --output_dir ./output \
    --max_length 512 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --num_train_epochs 1
```

---

## Troubleshooting

### Error: "Model not found"
```bash
# Check if model downloaded correctly
ls -la examples/omega17_exp/model/
# Should see: config.json, model.safetensors, tokenizer files, etc.
```

### Error: "CUDA out of memory"
```bash
# Reduce batch size
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 16 \

# Or reduce sequence length
--max_length 256 \
```

### Error: "Module not found: transformers.deepspeed"
```bash
# Re-run patches
cd examples/omega17_exp
python setup_environment.py
```

### Error: "omega17_exp not registered"
```bash
# Re-run patches
python setup_environment.py
```

### Training too slow
```bash
# Increase batch size if memory allows
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4 \

# Use multi-GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 swift sft ...
```

---

## Expected Training Time

| GPUs | Batch | Dataset | Time Estimate |
|------|-------|---------|---------------|
| 1x A100 80GB | 4 | 52k samples | ~8-12 hours |
| 1x RTX 4090 24GB | 2 | 52k samples | ~15-20 hours |
| 4x A100 80GB | 4 | 52k samples | ~2-3 hours |

---

## Files in This Directory

| File | Purpose |
|------|---------|
| `SETUP_GUIDE.md` | This guide |
| `setup_environment.py` | Patches all compatibility issues |
| `download_model.py` | Downloads model from HuggingFace |
| `train_optimized.sh` | Training script |
| `ds_config_zero2.json` | DeepSpeed config for multi-GPU |

---

## Model Information

| Property | Value |
|----------|-------|
| **Model ID** | `arpitsh018/omega17exp-prod-v1.1` |
| **Architecture** | Omega17ExpForCausalLM (MoE) |
| **Parameters** | ~30B total, ~4B active |
| **Layers** | 48 |
| **Experts** | 128 (8 active per token) |
| **Context Length** | 262,144 tokens |
| **Template** | ChatML |

---

## Support

If you encounter issues:
1. Check the Troubleshooting section above
2. Ensure HF_TOKEN is set correctly
3. Verify GPU has enough memory
4. Re-run `python setup_environment.py`
