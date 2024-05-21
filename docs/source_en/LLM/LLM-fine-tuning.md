# LLM Fine-tuning Documentation
## Table of Contents
- [Environment Preparation](#environment-preparation)
- [Fine-tuning](#fine-tuning)
- [DPO](#dpo)
- [Merge LoRA](#merge-lora)
- [Quantization](#quantization)
- [Inference](#inference)
- [Web-UI](#web-ui)

## Environment Preparation
GPU devices: A10, 3090, V100, A100 are all suitable.
```bash
# Install ms-swift
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'

# If you want to use deepspeed.
pip install deepspeed -U

# If you want to use qlora training based on auto_gptq. (Recommended, better than bnb)
# Models supporting auto_gptq: `https://github.com/modelscope/swift/blob/main/docs/source/LLM/supported-models-and-datasets.md#models`
# auto_gptq and cuda versions are related, please choose the version according to `https://github.com/PanQiWei/AutoGPTQ#quick-installation`
pip install auto_gptq -U

# If you want to use bnb-based qlora training.
pip install bitsandbytes -U

# Align environment (usually not necessary to run. If you encounter errors, you can run the following code, the repository is tested with the latest environment)
pip install -r requirements/framework.txt -U
pip install -r requirements/llm.txt -U
```

## Fine-Tuning
If you want to fine-tune and infer using the interface, you can check [Web-ui Documentation](../GetStarted/Web-ui.md).

### Using Python
```python
# Experimental environment: A10, 3090, V100, ...
# 20GB GPU memory
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch

from swift.llm import (
    DatasetName, InferArguments, ModelType, SftArguments,
    infer_main, sft_main, app_ui_main
)

model_type = ModelType.qwen_7b_chat
sft_args = SftArguments(
    model_type=model_type,
    dataset=[f'{DatasetName.blossom_math_zh}#2000'],
    output_dir='output')
result = sft_main(sft_args)
best_model_checkpoint = result['best_model_checkpoint']
print(f'best_model_checkpoint: {best_model_checkpoint}')
torch.cuda.empty_cache()

infer_args = InferArguments(
    ckpt_dir=best_model_checkpoint,
    load_dataset_config=True)
# merge_lora(infer_args, device_map='cpu')
result = infer_main(infer_args)
torch.cuda.empty_cache()

app_ui_main(infer_args)
```

### Using CLI
```bash
# Experimental environment: A10, 3090, V100, ...
# 20GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_id_or_path qwen/Qwen-7B-Chat \
    --dataset AI-ModelScope/blossom-math-v2 \
    --output_dir output \

# Using your own dataset
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_id_or_path qwen/Qwen-7B-Chat \
    --dataset chatml.jsonl \
    --output_dir output \

# Using DDP
# Experimental environment: 2 * 3090
# 2 * 23GB GPU memory
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
swift sft \
    --model_id_or_path qwen/Qwen-7B-Chat \
    --dataset AI-ModelScope/blossom-math-v2 \
    --output_dir output \

# Multi-machine multi-card
# node0
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NNODES=2 \
NODE_RANK=0 \
MASTER_ADDR=127.0.0.1 \
NPROC_PER_NODE=4 \
swift sft \
    --model_id_or_path qwen/Qwen-7B-Chat \
    --dataset AI-ModelScope/blossom-math-v2 \
    --output_dir output \
# node1
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NNODES=2 \
NODE_RANK=1 \
MASTER_ADDR=xxx.xxx.xxx.xxx \
NPROC_PER_NODE=4 \
swift sft \
    --model_id_or_path qwen/Qwen-7B-Chat \
    --dataset AI-ModelScope/blossom-math-v2 \
    --output_dir output \
```

### More sh Scripts

More sh scripts can be viewed [here](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts)

```bash
# Scripts need to be executed in this directory
cd examples/pytorch/llm
```

**Tips**:

- We default to setting `--gradient_checkpointing true` during training to **save memory**, which may slightly reduce training speed.
- If you want to use quantization parameters `--quantization_bit 4`, you need to first install [bnb](https://github.com/TimDettmers/bitsandbytes): `pip install bitsandbytes -U`. This will reduce memory usage but usually slows down the training speed.
- If you want to use quantization based on **auto_gptq**, you need to install the corresponding cuda version of [auto_gptq](https://github.com/PanQiWei/AutoGPTQ): `pip install auto_gptq -U`.
  > Models that can use auto_gptq can be viewed in [LLM Supported Models](Supported-models-datasets.md#models). It is recommended to use auto_gptq instead of bnb.
- If you want to use deepspeed, you need `pip install deepspeed -U`. Using deepspeed can **save memory**, but may slightly reduce training speed.
- If your training involves **knowledge editing**, such as: [Self-aware Fine-tuning](Self-cognition-best-practice.md), you need to add LoRA to MLP as well, otherwise, the results might be poor. You can simply pass the argument `--lora_target_modules ALL` to add lora to all linear(qkvo, mlp), **this is usually the best result**.
- If you are using older GPUs like **V100**, you need to set `--dtype AUTO` or `--dtype fp16`, as they do not support bf16.
- If your machine has high-performance graphics cards like A100 and the model supports flash-attn, it is recommended to install [**flash-attn**](https://github.com/Dao-AILab/flash-attention), which will speed up training and inference as well as reduce memory usage (A10, 3090, V100, etc. graphics cards do not support training with flash-attn). Models that support flash-attn can be viewed in [LLM Supported Models](Supported-models-datasets.md#models)
- If you are doing **second pre-training** or **multi-turn dialogue**, you can refer to [Customization and Extension](Customization.md#Registering-Datasets)
- If you need to train **offline**, please use `--model_id_or_path <model_dir>` and set `--check_model_is_latest false`. For specific parameter meanings, please check [Command-line Parameters](Command-line-parameters.md).
- If you want to push weights to the ModelScope Hub during training, you need to set `--push_to_hub true`.
- If you want to merge LoRA weights and save them during inference, you need to set `--merge_lora true`. **It is not recommended to merge** for models trained with qlora, as this will result in precision loss. Therefore **it is not recommended to fine-tune** with qlora, as the deployment ecology is not good.


**Note**:

- Due to the legacy name issue, scripts ending with `xxx_ds` mean: training using deepspeed zero2. (e.g. `full_ddp_ds`).
- In addition to the scripts listed below, other scripts may not be maintained.


If you want to **customize scripts**, you can refer to the following scripts for modification: (The following scripts will be **regularly maintained**)

- full: [qwen1half-7b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen1half_7b_chat/full) (A100), [qwen-7b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_7b_chat/full_mp) (2*A100)
- full+ddp+zero2: [qwen-7b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_7b_chat/full_ddp_zero2) (4*A100)
- full+ddp+zero3: [qwen-14b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_14b_chat/full_ddp_zero3) (4*A100)
- lora: [chatglm3-6b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/chatglm3_6b/lora) (3090), [baichuan2-13b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/baichuan2_13b_chat/lora_mp) (2*3090), [yi-34b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/yi_34b_chat/lora) (A100), [qwen-72b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_72b_chat/lora_mp) (2*A100)
- lora+ddp: [chatglm3-6b](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/chatglm3_6b/lora_ddp) (2*3090)
- lora+ddp+zero3: [qwen-14b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_14b_chat/lora_ddp_zero3) (4*3090), [qwen-72b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_72b_chat/lora_ddp_zero3) (4*A100)
- qlora(gptq-int4): [qwen-7b-chat-int4](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_7b_chat_int4/qlora) (3090)
- qlora(gptq-int8): [qwen1half-7b-chat-int8](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen1half_7b_chat_int8/qlora) (3090)
- qlora(bnb-int4): [qwen-7b-chat](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/scripts/qwen_7b_chat/qlora) (3090)

## DPO
If you want to use DPO for human-aligned fine-tuning, you can check the [DPO Fine-Tuning Documentation](DPO.md).

## ORPO
If you want to use ORPO for human-aligned fine-tuning, you can check the [ORPO Fine-Tuning Documentation](ORPO.md).

## Merge LoRA
Tip: **Currently**, merging LoRA is not supported for bnb and auto_gptq quantized models, as this would result in significant accuracy loss.
```bash
# If you need quantization, you can specify `--quant_bits 4`.
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx' --merge_lora true
```

## Quantization

For quantization of the fine-tuned model, you can check [LLM Quantization Documentation](LLM-quantization.md#fine-tuned-model)

## Inference
If you want to use VLLM for accelerated inference, you can check [VLLM Inference Acceleration and Deployment](VLLM-inference-acceleration-and-deployment.md)

### Original Model
**Single sample inference** can be checked in [LLM Inference Documentation](LLM-inference.md)

Using **Dataset** for evaluation:
```bash
CUDA_VISIBLE_DEVICES=0 swift infer --model_id_or_path qwen/Qwen-7B-Chat --dataset AI-ModelScope/blossom-math-v2
```
### Fine-tuned Model
**Single sample inference**:

Inference using LoRA **incremental** weights:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type
)
from swift.tuners import Swift

ckpt_dir = 'vx-xxx/checkpoint-100'
model_type = ModelType.qwen_7b_chat
template_type = get_default_template_type(model_type)

model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'})

model = Swift.from_pretrained(model, ckpt_dir, inference_mode=True)
template = get_template(template_type, tokenizer)
query = 'xxxxxx'
response, history = inference(model, template, query)
print(f'response: {response}')
print(f'history: {history}')
```

Inference using LoRA **merged** weights:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type
)

ckpt_dir = 'vx-xxx/checkpoint-100-merged'
model_type = ModelType.qwen_7b_chat
template_type = get_default_template_type(model_type)

model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'},
                                       model_id_or_path=ckpt_dir)

template = get_template(template_type, tokenizer)
query = 'xxxxxx'
response, history = inference(model, template, query)
print(f'response: {response}')
print(f'history: {history}')
```

Using **Dataset** for evaluation:
```bash
# Direct inference
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx' \
    --load_dataset_config true \

# Merge LoRA incremental weights and infer
# If you need quantization, you can specify `--quant_bits 4`.
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx' --merge_lora true

CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx-merged' --load_dataset_config true
```

**Manual** evaluation:
```bash
# Direct inference
CUDA_VISIBLE_DEVICES=0 swift infer --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx'

# Merge LoRA incremental weights and infer
# If you need quantization, you can specify `--quant_bits 4`.
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx' --merge_lora true

CUDA_VISIBLE_DEVICES=0 swift infer --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx-merged'
```

## Web-UI
If you want to deploy VLLM and provide **API** interface, you can check [VLLM Inference Acceleration and Deployment](VLLM-inference-acceleration-and-deployment.md)

### Original Model
Using the original model's web-ui can be viewed in [LLM Inference Documentation](LLM-inference.md#Web-UI)

### Fine-tuned Model
```bash
# Directly use app-ui
CUDA_VISIBLE_DEVICES=0 swift app-ui --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx'

# Merge LoRA incremental weights and use app-ui
# If you need quantization, you can specify `--quant_bits 4`.
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx' --merge_lora true

CUDA_VISIBLE_DEVICES=0 swift app-ui --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx-merged'
```
