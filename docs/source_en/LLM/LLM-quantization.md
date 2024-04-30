# LLM Quantization Documentation
Swift supports using AWQ and GPTQ techniques to quantize models. These two quantization techniques support VLLM inference acceleration, and the quantized models also support QLORA fine-tuning.

## Table of Contents
- [Environment Preparation](#environment-preparation)
- [Original Model](#original-model)
- [Fine-tuned Model](#fine-tuned-model)
- [Pushing Models](#pushing-models)

## Environment Preparation
GPU devices: A10, 3090, V100, A100 are all supported.
```bash
# Install ms-swift
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'

# Using AWQ quantization:
# AutoAWQ and CUDA versions have a corresponding relationship, please select the version according to `https://github.com/casper-hansen/AutoAWQ`
pip install autoawq -U

# Using GPTQ quantization:
# Auto_GPTQ and CUDA versions have a corresponding relationship, please select the version according to `https://github.com/PanQiWei/AutoGPTQ#quick-installation`
pip install auto_gptq -U

# Environment alignment (usually not needed. If you encounter errors, you can run the code below, the repository uses the latest environment for testing)
pip install -r requirements/framework.txt -U
pip install -r requirements/llm.txt -U
```

## Original Model
Here we demonstrate AWQ and GPTQ quantization on the qwen1half-7b-chat model.
```bash
# AWQ-INT4 quantization (takes about 18 minutes using A100, memory usage: 13GB)
# If OOM occurs during quantization, you can appropriately reduce `--quant_n_samples` (default 256) and `--quant_seqlen` (default 2048).
# GPTQ-INT4 quantization (takes about 20 minutes using A100, memory usage: 7GB)

# AWQ: Use `alpaca-zh alpaca-en` as the quantization dataset
CUDA_VISIBLE_DEVICES=0 swift export \
    --model_type qwen1half-7b-chat --quant_bits 4 \
    --dataset alpaca-zh alpaca-en --quant_method awq

# GPTQ: Use `alpaca-zh alpaca-en` as the quantization dataset
# For GPTQ quantization, please first refer to this issue: https://github.com/AutoGPTQ/AutoGPTQ/issues/439
OMP_NUM_THREADS=14 CUDA_VISIBLE_DEVICES=0 swift export \
    --model_type qwen1half-7b-chat --quant_bits 4 \
    --dataset alpaca-zh alpaca-en --quant_method gptq

# AWQ: Use custom quantization dataset (don't use the `--custom_val_dataset_path` parameter)
# Same for GPTQ
CUDA_VISIBLE_DEVICES=0 swift export \
    --model_type qwen1half-7b-chat --quant_bits 4 \
    --custom_train_dataset_path xxx.jsonl \
    --quant_method awq

# Inference using swift quantized model
# AWQ
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model_type qwen1half-7b-chat \
    --model_id_or_path qwen1half-7b-chat-awq-int4
# GPTQ
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model_type qwen1half-7b-chat \
    --model_id_or_path qwen1half-7b-chat-gptq-int4

# Inference using original model
CUDA_VISIBLE_DEVICES=0 swift infer --model_type qwen1half-7b-chat
```

**Comparison of quantization effects**:

The comparison shows inference results from the AWQ-INT4 model, GPTQ-INT4 model, and the original unquantized model. The quantized models maintain high quality output while enabling faster inference speeds.

## Fine-tuned Model

Assume you fine-tuned qwen1half-4b-chat using LoRA, and the model weights directory is: `output/qwen1half-4b-chat/vx-xxx/checkpoint-xxx`.

Here we only introduce using the AWQ technique to quantize the fine-tuned model. Using GPTQ for quantization would be similar.

**Merge-LoRA & Quantization**
```shell
# Use `alpaca-zh alpaca-en` as the quantization dataset
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir 'output/qwen1half-4b-chat/vx-xxx/checkpoint-xxx' \
    --merge_lora true --quant_bits 4 \
    --dataset alpaca-zh alpaca-en --quant_method awq

# Use the dataset from fine-tuning as the quantization dataset
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir 'output/qwen1half-4b-chat/vx-xxx/checkpoint-xxx' \
    --merge_lora true --quant_bits 4 \
    --load_dataset_config true --quant_method awq
```

**Inference using quantized model**
```shell
# AWQ/GPTQ quantized models support VLLM inference acceleration. They also support model deployment.
CUDA_VISIBLE_DEVICES=0 swift infer --ckpt_dir 'output/qwen1half-4b-chat/vx-xxx/checkpoint-xxx-merged-awq-int4'
```

**Deploying the quantized model**

Server:

```shell
CUDA_VISIBLE_DEVICES=0 swift deploy --ckpt_dir 'output/qwen1half-4b-chat/vx-xxx/checkpoint-xxx-merged-awq-int4'
```

Testing:
```shell
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "qwen1half-4b-chat",
"messages": [{"role": "user", "content": "How to fall asleep at night?"}],
"max_tokens": 256,
"temperature": 0
}'
```

## Pushing Models
Assume you fine-tuned qwen1half-4b-chat using LoRA, and the model weights directory is: `output/qwen1half-4b-chat/vx-xxx/checkpoint-xxx`.

```shell
# Push the original quantized model
CUDA_VISIBLE_DEVICES=0 swift export \
    --model_type qwen1half-7b-chat \
    --model_id_or_path qwen1half-7b-chat-gptq-int4 \
    --push_to_hub true \
    --hub_model_id qwen1half-7b-chat-gptq-int4 \
    --hub_token '<your-sdk-token>'

# Push LoRA incremental model
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir output/qwen1half-4b-chat/vx-xxx/checkpoint-xxx \
    --push_to_hub true \
    --hub_model_id qwen1half-4b-chat-lora \
    --hub_token '<your-sdk-token>'

# Push merged model
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir output/qwen1half-4b-chat/vx-xxx/checkpoint-xxx \
    --push_to_hub true \
    --hub_model_id qwen1half-4b-chat-lora \
    --hub_token '<your-sdk-token>' \
    --merge_lora true

# Push quantized model
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir output/qwen1half-4b-chat/vx-xxx/checkpoint-xxx \
    --push_to_hub true \
    --hub_model_id qwen1half-4b-chat-lora \
    --hub_token '<your-sdk-token>' \
    --merge_lora true \
    --quant_bits 4
```
