# Omega17Exp LoRA SFT Fine-tuning Guide

Complete guide for fine-tuning **Omega17ExpForCausalLM** using LoRA with MS-SWIFT 3.x on RunPod.

---

## Table of Contents

1. [Model Information](#model-information)
2. [Files Overview](#files-overview)
3. [Step-by-Step Setup (RunPod)](#step-by-step-setup-runpod)
4. [What setup_environment.py Fixes](#what-setup_environmentpy-fixes)
5. [Dataset Options](#dataset-options)
6. [Training Commands](#training-commands)
7. [Training Configurations](#training-configurations)
8. [Inference](#inference)
9. [Merge LoRA Weights](#merge-lora-weights)
10. [Troubleshooting](#troubleshooting)

---

## Model Information

| Property | Value |
|----------|-------|
| **Model ID** | `arpitsh018/omega17exp-prod-v1.1` |
| **Type** | Private (requires HuggingFace token) |
| **Architecture** | Omega17ExpForCausalLM (MoE) |
| **Hidden Size** | 2048 |
| **Layers** | 48 |
| **Experts** | 128 total, 8 active per token |
| **Context Length** | 262,144 tokens |
| **Vocab Size** | 151,936 |

---

## Files Overview

| File | Description |
|------|-------------|
| `setup_environment.py` | **REQUIRED** - Patches all compatibility issues (run once after install) |
| `register_omega17.py` | Registers custom model with MS-SWIFT (used by setup_environment.py) |
| `download_model.py` | Downloads model from HuggingFace with token support |
| `train_lora_sft.py` | Python training script (alternative to CLI) |
| `run_training.sh` | One-click training script |
| `ds_config_zero2.json` | DeepSpeed ZeRO-2 config for multi-GPU |

---

## Step-by-Step Setup (RunPod)

### Step 1: Create Working Directory

```bash
mkdir -p /workspace/finetune
cd /workspace/finetune
```

### Step 2: Upload Files

Upload these files to `/workspace/finetune/`:
- `setup_environment.py`
- `download_model.py`
- `ds_config_zero2.json` (optional, for multi-GPU)

### Step 3: Install Dependencies (ORDER MATTERS!)

```bash
# 1. Install MS-SWIFT first (this sets up base dependencies)
pip install ms-swift[llm]

# 2. Install custom transformers fork AFTER ms-swift
# This MUST be done after ms-swift to override huggingface-hub version
pip install transformers-usf-om-vl-exp-v0 --force-reinstall

# 3. Additional dependencies
pip install accelerate bitsandbytes peft datasets
```

> ⚠️ **IMPORTANT**: The installation order matters! Installing `transformers-usf-om-vl-exp-v0` AFTER `ms-swift` ensures the correct `huggingface-hub` version.

### Step 4: Run Setup Script (REQUIRED!)

```bash
python setup_environment.py
```

**Expected Output:**
```
============================================================
OMEGA17EXP ENVIRONMENT SETUP
============================================================

🔧 Applying patches...

1. Patching BACKENDS_MAPPING (tf backend)...
   ✅ Patched BACKENDS_MAPPING

2. Patching deepspeed module (peft compatibility)...
   ✅ deepspeed module created successfully

3. Patching ROPE_INIT_FUNCTIONS (add 'default' key)...
   ✅ ROPE_INIT_FUNCTIONS patched successfully

4. Patching swift train_args.py (logging_dir fix)...
   ✅ Patched train_args.py (logging_dir fix)

5. Registering omega17_exp model in swift...
   ✅ Created omega17.py in swift
   ✅ Updated swift __init__.py to import omega17

6. Removing @check_model_inputs decorator from model files...
   ✅ Removed @check_model_inputs from ./model/modeling_omega17_exp.py

7. Installing bitsandbytes for 4-bit quantization (QLoRA)...
   ✅ bitsandbytes installed successfully

🔍 Verifying imports...
   ✅ transformers imports OK
   ✅ peft imports OK
   ✅ swift imports OK
   ✅ omega17_exp model registered in swift

============================================================
✅ SETUP COMPLETE! Environment is ready.
============================================================
```

### Step 5: Set HuggingFace Token

```bash
export HF_TOKEN=your_huggingface_token_here
```

### Step 6: Download Model

```bash
python download_model.py --output_dir ./model
```

This downloads the model from `arpitsh018/omega17exp-prod-v1.1` to `./model/`.

### Step 7: Run Training

See [Training Commands](#training-commands) below.

---

## What setup_environment.py Fixes

The `setup_environment.py` script applies **7 patches** to fix compatibility issues:

### Patch 1: BACKENDS_MAPPING (tf backend)
- **File**: `transformers/utils/import_utils.py`
- **Issue**: Custom transformers fork missing 'tf' backend causes import errors
- **Fix**: Adds dummy `is_tf_available` function and 'tf' key to `BACKENDS_MAPPING`

### Patch 2: deepspeed module (peft compatibility)
- **File**: `transformers/deepspeed.py`
- **Issue**: PEFT tries to import `transformers.deepspeed` which doesn't exist in the custom fork
- **Fix**: Creates a dummy `deepspeed.py` module with stub functions

### Patch 3: ROPE_INIT_FUNCTIONS (default key)
- **File**: `transformers/modeling_rope_utils.py`
- **Issue**: Model uses `rope_type='default'` but transformers only has `['linear', 'dynamic', 'yarn', 'longrope', 'llama3']`
- **Fix**: Adds `_compute_default_rope_parameters` function and registers it as 'default'

### Patch 4: swift train_args.py (logging_dir fix)
- **File**: `swift/llm/argument/train_args.py`
- **Issue**: `AttributeError: 'TrainArguments' object has no attribute 'logging_dir'`
- **Fix**: Changes `if self.logging_dir` to `if getattr(self, "logging_dir", None)`

### Patch 5: omega17_exp model registration
- **File**: `swift/llm/model/model/omega17.py` + `__init__.py`
- **Issue**: MS-SWIFT doesn't know about `omega17_exp` model type
- **Fix**: Creates `omega17.py` with model registration and imports it

### Patch 6: @check_model_inputs decorator
- **File**: `./model/modeling_omega17_exp.py` (and HuggingFace cache)
- **Issue**: `TypeError: got an unexpected keyword argument 'input_ids'` during training
- **Fix**: Removes `@check_model_inputs` decorator and its import from model files

### Patch 7: bitsandbytes installation
- **Package**: `bitsandbytes>=0.46.1`
- **Issue**: 4-bit quantization (QLoRA) requires bitsandbytes
- **Fix**: Auto-installs or upgrades bitsandbytes for QLoRA support

---

## Dataset Options

### Option 1: HuggingFace Datasets (Recommended)

Use `--use_hf true` to load datasets from HuggingFace Hub:

```bash
swift sft \
    --model ./model \
    --model_type omega17_exp \
    --dataset tatsu-lab/alpaca \
    --use_hf true \
    --train_type lora \
    --output_dir ./output
```

**Popular HuggingFace Datasets:**
- `tatsu-lab/alpaca` - General instruction following
- `databricks/dolly-15k` - Instruction following
- `Open-Orca/OpenOrca` - Large instruction dataset
- `timdettmers/openassistant-guanaco` - Conversational

### Option 2: Local JSONL File

Create a JSONL file with your training data:

**Messages format (recommended):**
```json
{"messages": [{"role": "user", "content": "What is AI?"}, {"role": "assistant", "content": "AI is artificial intelligence..."}]}
{"messages": [{"role": "user", "content": "Explain machine learning"}, {"role": "assistant", "content": "Machine learning is..."}]}
```

**Query/Response format:**
```json
{"query": "What is the capital of France?", "response": "The capital of France is Paris."}
{"query": "Who wrote Romeo and Juliet?", "response": "William Shakespeare wrote Romeo and Juliet."}
```

**Use local file:**
```bash
swift sft \
    --model ./model \
    --model_type omega17_exp \
    --dataset /path/to/your/data.jsonl \
    --train_type lora \
    --output_dir ./output
```

### Option 3: ModelScope Datasets

Without `--use_hf`, datasets load from ModelScope (may not have all datasets):

```bash
swift sft \
    --model ./model \
    --model_type omega17_exp \
    --dataset ms-agent-for-agentfabric \
    --train_type lora \
    --output_dir ./output
```

---

## Training Commands

### Basic Training (HuggingFace Dataset)

```bash
swift sft \
    --model ./model \
    --model_type omega17_exp \
    --dataset tatsu-lab/alpaca \
    --use_hf true \
    --train_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --output_dir ./output \
    --max_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing true \
    --num_train_epochs 3 \
    --learning_rate 1e-4 \
    --logging_steps 10 \
    --save_steps 500
```

### Training with Local Dataset

```bash
swift sft \
    --model ./model \
    --model_type omega17_exp \
    --dataset /workspace/data/train.jsonl \
    --train_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --output_dir ./output \
    --max_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing true \
    --num_train_epochs 3
```

### Important Flags

| Flag | Description |
|------|-------------|
| `--model` | Path to model directory |
| `--model_type` | Must be `omega17_exp` |
| `--dataset` | Dataset name or path to JSONL file |
| `--use_hf true` | Load dataset from HuggingFace Hub |
| `--train_type lora` | Use LoRA fine-tuning |
| `--lora_rank` | LoRA rank (8, 16, 32, 64, 128) |
| `--lora_alpha` | LoRA alpha (usually 2x rank) |
| `--gradient_checkpointing true` | Reduce memory usage |
| `--per_device_train_batch_size` | Batch size per GPU |
| `--gradient_accumulation_steps` | Accumulate gradients |
| `--quant_method bnb` | Use bitsandbytes quantization |
| `--quant_bits 4` | 4-bit quantization (QLoRA) |

> ⚠️ **Note**: Do NOT use `--dtype`. Use `--torch_dtype` if needed, or omit to auto-detect (bfloat16).
> 
> 💡 **Recommended**: Use `--quant_method bnb --quant_bits 4` for QLoRA to reduce VRAM usage significantly.

---

## Training Configurations

### Memory-Efficient (QLoRA) - A100 40GB / RTX 4090

```bash
swift sft \
    --model ./model \
    --model_type omega17_exp \
    --dataset tatsu-lab/alpaca \
    --use_hf true \
    --train_type lora \
    --quant_method bnb \
    --quant_bits 4 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --output_dir ./output \
    --max_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing true \
    --num_train_epochs 3
```

### Standard QLoRA - A100 80GB / H100

```bash
swift sft \
    --model ./model \
    --model_type omega17_exp \
    --dataset tatsu-lab/alpaca \
    --use_hf true \
    --train_type lora \
    --quant_method bnb \
    --quant_bits 4 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --output_dir ./output \
    --max_length 2048 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing true \
    --num_train_epochs 3
```

### Multi-GPU with DeepSpeed ZeRO-2

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 swift sft \
    --model ./model \
    --model_type omega17_exp \
    --dataset tatsu-lab/alpaca \
    --use_hf true \
    --train_type lora \
    --lora_rank 128 \
    --lora_alpha 256 \
    --output_dir ./output \
    --max_length 4096 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --deepspeed ds_config_zero2.json \
    --num_train_epochs 3
```

---

## Inference

After training, run inference with the LoRA adapter:

```bash
swift infer \
    --model ./model \
    --model_type omega17_exp \
    --adapters ./output/checkpoint-xxx \
    --max_new_tokens 512 \
    --stream true
```

**Interactive chat:**
```bash
swift infer \
    --model ./model \
    --model_type omega17_exp \
    --adapters ./output/checkpoint-xxx \
    --infer_backend pt \
    --max_new_tokens 1024
```

---

## Merge LoRA Weights

To merge LoRA adapter into base model for deployment:

```bash
swift export \
    --model ./model \
    --model_type omega17_exp \
    --adapters ./output/checkpoint-xxx \
    --merge_lora true \
    --output_dir ./merged_model
```

---

## Troubleshooting

### Dataset Not Found (404 Error)

**Error:** `Dataset alpaca-en load failed: ... HTTPError: <Response [404]>`

**Solution:** Use `--use_hf true` with HuggingFace datasets:
```bash
--dataset tatsu-lab/alpaca --use_hf true
```

### Out of Memory (OOM)

**Solutions:**
1. Reduce batch size: `--per_device_train_batch_size 1`
2. Reduce sequence length: `--max_length 1024`
3. Enable gradient checkpointing: `--gradient_checkpointing true`
4. Use QLoRA: `--quant_bits 4`
5. Use DeepSpeed: `--deepspeed ds_config_zero2.json`

### Model Type Not Found

**Error:** `model_type: 'omega17_exp' not in ...`

**Solution:** Run `python setup_environment.py` to register the model.

### ROPE KeyError 'default'

**Error:** `KeyError: 'default'` in ROPE_INIT_FUNCTIONS

**Solution:** Run `python setup_environment.py` - this adds the 'default' key.

### logging_dir AttributeError

**Error:** `AttributeError: 'TrainArguments' object has no attribute 'logging_dir'`

**Solution:** Run `python setup_environment.py` - this patches train_args.py.

### BACKENDS_MAPPING KeyError 'tf'

**Error:** `KeyError: 'tf'` in BACKENDS_MAPPING

**Solution:** Run `python setup_environment.py` - this patches import_utils.py.

### Slow Training

**Tips:**
1. Use bfloat16 (auto-detected)
2. Increase batch size if memory allows
3. Use multi-GPU with DeepSpeed
4. Reduce logging frequency: `--logging_steps 50`

---

## Quick Reference

### Complete Setup Commands (Copy-Paste Ready)

```bash
# 1. Setup
cd /workspace
mkdir finetune && cd finetune

# 2. Install (ORDER MATTERS!)
pip install ms-swift[llm]
pip install transformers-usf-om-vl-exp-v0 --force-reinstall
pip install accelerate bitsandbytes peft datasets

# 3. Upload setup_environment.py and download_model.py, then:
python setup_environment.py

# 4. Download model
export HF_TOKEN=your_token_here
python download_model.py --output_dir ./model

# 5. Train (QLoRA 4-bit)
swift sft \
    --model ./model \
    --model_type omega17_exp \
    --dataset tatsu-lab/alpaca \
    --use_hf true \
    --train_type lora \
    --quant_method bnb \
    --quant_bits 4 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --output_dir ./output \
    --max_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --gradient_checkpointing true \
    --num_train_epochs 3
```

---

## Support

- **MS-SWIFT Documentation**: https://swift.readthedocs.io/
- **HuggingFace Datasets**: https://huggingface.co/datasets
