# LLM Quantization Documentation
Swift supports model quantization using the techniques of awq, gptq, bnb, hqq, eetq. Among these, awq and gptq quantization techniques support inference acceleration for vllm, and the quantized models support fine-tuning with qlora.
Note The effect of quantization varies under different commands:
- During sft lora training, quantization specified for `qlora` is used to reduce the memory required for training.
- In export, quantization is specified to quantize the model and save it.
- In infer, quantization is specified for model quantization and inference.

bnb, hqq, and eetq do not require calibration data and offer fast quantization speed. They are used in sft lora training and inference by specifying `--quant_method bnb/hqq/eetq`.

awq and gptq require calibration data and are used in export by specifying `--quant_method awq/gptq`.

## Table of Contents
- [Environment Preparation](#environment-preparation)
- [Qlora](#qlora)
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

## QLora
In the sft lora training, specify `--quant_method` and `--quantization_bit` to execute qlora, which significantly reduces the GPU memory required for training.

```bash
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type qwen1half-7b-chat \
    --sft_type lora \
    --dataset alpaca-zh#5000 \
    --quant_method hqq \
    --quantization_bit 4 \

CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type qwen1half-7b-chat \
    --sft_type lora \
    --dataset alpaca-zh#5000 \
    --quant_method eetq \
    --dtype fp16 \

CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type qwen1half-7b-chat \
    --sft_type lora \
    --dataset alpaca-zh#5000 \
    --quant_method bnb \
    --quantization_bit 4 \
    --dtype fp16 \
```
**Note**
- hqq supports more customizable parameters, such as specifying different quantization configurations for different network layers. For details, please see [Command Line Arguments](https://github.com/modelscope/swift/blob/main/docs/source_en/LLM/Command-line-parameters.md).
- eetq quantization uses 8-bit quantization, and there's no need to specify quantization_bit. Currently, bf16 is not supported; you need to specify dtype as fp16.
- Currently, eetq's qlora speed is relatively slow; it is recommended to use hqq instead. For reference, see the [issue](https://github.com/NetEase-FuXi/EETQ/issues/17).

## Original Model
Use bnb, hqq, and eetq for model quantization and inference.
```bash
CUDA_VISIBLE_DEVICES=0 swift infer \
    --model_type qwen1half-7b-chat \
    --quant_method bnb \
    --quantization_bit 4

CUDA_VISIBLE_DEVICES=0 swift infer \
    --model_type qwen1half-7b-chat \
    --quant_method hqq \
    --quantization_bit 4

CUDA_VISIBLE_DEVICES=0 swift infer \
    --model_type qwen1half-7b-chat \
    --quant_method eetq \
    --dtype fp16
```

Here we demonstrate AWQ and GPTQ quantization on the qwen1half-7b-chat model.
```bash
# AWQ-INT4 quantization (takes about 18 minutes using A100, memory usage: 13GB)
# If OOM occurs during quantization, you can appropriately reduce `--quant_n_samples` (default 256) and `--quant_seqlen` (default 2048).
# GPTQ-INT4 quantization (takes about 20 minutes using A100, memory usage: 7GB)

# AWQ: Use `alpaca-zh alpaca-en sharegpt-gpt4-mini` as the quantization dataset
CUDA_VISIBLE_DEVICES=0 swift export \
    --model_type qwen1half-7b-chat --quant_bits 4 \
    --dataset alpaca-zh alpaca-en sharegpt-gpt4-mini --quant_method awq

# GPTQ: Use `alpaca-zh alpaca-en sharegpt-gpt4-mini` as the quantization dataset
# For GPTQ quantization, please first refer to this issue: https://github.com/AutoGPTQ/AutoGPTQ/issues/439
OMP_NUM_THREADS=14 CUDA_VISIBLE_DEVICES=0 swift export \
    --model_type qwen1half-7b-chat --quant_bits 4 \
    --dataset alpaca-zh alpaca-en sharegpt-gpt4-mini --quant_method gptq

# AWQ: Use custom quantization dataset
# Same for GPTQ
CUDA_VISIBLE_DEVICES=0 swift export \
    --model_type qwen1half-7b-chat --quant_bits 4 \
    --dataset xxx.jsonl \
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

## Fine-tuned Model

Assume you fine-tuned qwen1half-4b-chat using LoRA, and the model weights directory is: `output/qwen1half-4b-chat/vx-xxx/checkpoint-xxx`.

Here we only introduce using the AWQ technique to quantize the fine-tuned model. Using GPTQ for quantization would be similar.

**Merge-LoRA & Quantization**
```shell
# Use `alpaca-zh alpaca-en sharegpt-gpt4-mini` as the quantization dataset
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir 'output/qwen1half-4b-chat/vx-xxx/checkpoint-xxx' \
    --merge_lora true --quant_bits 4 \
    --dataset alpaca-zh alpaca-en sharegpt-gpt4-mini --quant_method awq

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
