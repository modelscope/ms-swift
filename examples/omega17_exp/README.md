# Omega17Exp LoRA SFT Fine-tuning Guide

This guide provides complete instructions for fine-tuning the **Omega17ExpForCausalLM** model using LoRA (Low-Rank Adaptation) with MS-SWIFT on RunPod or similar GPU environments.

## Model Specifications

| Parameter | Value |
|-----------|-------|
| Architecture | Omega17ExpForCausalLM (MoE) |
| Hidden Size | 2048 |
| Layers | 48 |
| Experts | 128 total, 8 per token |
| Context Length | 262,144 tokens |
| Vocab Size | 151,936 |
| Attention Heads | 32 (4 KV heads) |
| Intermediate Size | 6144 (MoE: 768) |

## Custom Model Files Required

Your model directory must contain these custom files:
- `config.json` - Model configuration
- `tokenizer_config.json` - Tokenizer configuration
- `tokenization_omega17.py` - Custom Omega17Tokenizer class
- `configuration_omega17_exp.py` - Custom Omega17ExpConfig class
- `modeling_omega17_exp.py` - Custom Omega17ExpForCausalLM class
- Model weights (`.safetensors` or `.bin` files)

## Quick Start

### 1. Install Dependencies

```bash
# Install custom transformers fork (REQUIRED)
pip install transformers-usf-om-vl-exp-v0

# Install MS-SWIFT
pip install ms-swift[llm]

# Additional dependencies
pip install accelerate bitsandbytes peft datasets deepspeed
```

### 2. Run Training

**Option A: Using Python Script**
```bash
python train_lora_sft.py \
    --model_path /path/to/omega17-exp \
    --dataset alpaca-en \
    --output_dir ./output/omega17_lora \
    --lora_rank 64 \
    --batch_size 1 \
    --gradient_accumulation_steps 16
```

**Option B: Using MS-SWIFT CLI**
```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model /path/to/omega17-exp \
    --model_type omega17_exp \
    --dataset alpaca-en \
    --train_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --output_dir ./output/omega17_lora \
    --max_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing true
```

**Option C: Using Jupyter Notebook**
```bash
jupyter notebook omega17_lora_sft_finetune.ipynb
```

## RunPod Setup

### Step 1: Launch RunPod Instance

1. Go to [RunPod.io](https://runpod.io)
2. Select a GPU template:
   - **Recommended**: A100 80GB or H100 80GB
   - **Minimum**: A100 40GB (with QLoRA)
3. Choose PyTorch template with CUDA 12.x

### Step 2: Setup Environment

```bash
# SSH into your RunPod instance or use the terminal

# Clone the repository (if using git)
cd /workspace

# Install dependencies
pip install transformers-usf-om-vl-exp-v0
pip install ms-swift[llm]
pip install accelerate bitsandbytes peft datasets

# Upload your model to /workspace/model
# Upload your dataset to /workspace/data
```

### Step 3: Prepare Dataset

Create a JSONL file with your training data:

```json
{"messages": [{"role": "user", "content": "What is AI?"}, {"role": "assistant", "content": "AI is..."}]}
{"messages": [{"role": "user", "content": "Explain ML"}, {"role": "assistant", "content": "ML is..."}]}
```

Or use query/response format:
```json
{"query": "What is the capital of France?", "response": "The capital of France is Paris."}
```

### Step 4: Run Training

```bash
# Single GPU
python train_lora_sft.py \
    --model_path /workspace/model \
    --dataset /workspace/data/train.jsonl \
    --output_dir /workspace/output \
    --lora_rank 64 \
    --batch_size 1 \
    --max_length 2048

# Multi-GPU with DeepSpeed
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_lora_sft.py \
    --model_path /workspace/model \
    --dataset /workspace/data/train.jsonl \
    --output_dir /workspace/output \
    --deepspeed ds_config_zero2.json
```

## Training Configurations

### Memory-Efficient (QLoRA) - A100 40GB
```bash
python train_lora_sft.py \
    --model_path /workspace/model \
    --dataset your_dataset.jsonl \
    --use_qlora \
    --quant_bits 4 \
    --lora_rank 32 \
    --batch_size 1 \
    --max_length 1024 \
    --gradient_checkpointing
```

### Standard LoRA - A100 80GB
```bash
python train_lora_sft.py \
    --model_path /workspace/model \
    --dataset your_dataset.jsonl \
    --lora_rank 64 \
    --lora_alpha 128 \
    --batch_size 2 \
    --max_length 2048 \
    --gradient_checkpointing
```

### High Performance - Multi-GPU
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_lora_sft.py \
    --model_path /workspace/model \
    --dataset your_dataset.jsonl \
    --lora_rank 128 \
    --batch_size 4 \
    --max_length 4096 \
    --deepspeed ds_config_zero2.json
```

## LoRA Target Modules

For Omega17Exp (llama-style architecture):

```python
# Attention layers (recommended)
["q_proj", "k_proj", "v_proj", "o_proj"]

# Attention + MLP (more parameters)
["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# All linear layers
["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"]
```

## Inference

After training, use the LoRA adapter for inference:

```bash
swift infer \
    --model /workspace/model \
    --model_type omega17_exp \
    --adapters /workspace/output/checkpoint-xxx \
    --torch_dtype bfloat16 \
    --max_new_tokens 512 \
    --stream true
```

## Merge LoRA Weights

To merge LoRA adapter into base model:

```bash
swift export \
    --model /workspace/model \
    --model_type omega17_exp \
    --adapters /workspace/output/checkpoint-xxx \
    --merge_lora true \
    --output_dir /workspace/merged_model
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce `--batch_size` to 1
- Reduce `--max_length` to 1024 or 512
- Enable `--gradient_checkpointing`
- Use `--use_qlora` for 4-bit quantization
- Use DeepSpeed ZeRO-2 or ZeRO-3

### Model Not Found
- Ensure `transformers-usf-om-vl-exp-v0` is installed
- Check model path contains: `config.json`, `modeling_omega17_exp.py`, `configuration_omega17_exp.py`

### Slow Training
- Use `--torch_dtype bfloat16`
- Increase batch size if memory allows
- Use multi-GPU with DeepSpeed

### Dataset Errors
- Ensure JSONL format (one JSON per line)
- Check JSON structure matches expected format
- Validate with: `python -c "import json; [json.loads(l) for l in open('data.jsonl')]"`

## Files

```
examples/omega17_exp/
├── README.md                          # This file
├── train_lora_sft.py                  # Python training script
├── ds_config_zero2.json               # DeepSpeed ZeRO-2 config
└── ../notebooks/
    └── omega17_lora_sft_finetune.ipynb # Jupyter notebook
```

## Support

- MS-SWIFT Documentation: https://swift.readthedocs.io/
- GitHub Issues: Report bugs and feature requests
