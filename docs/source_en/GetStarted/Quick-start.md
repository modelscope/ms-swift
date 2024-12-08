# Quick Start

SWIFT is an integrated framework that encompasses model training, inference deployment, evaluation, and quantization, allowing model developers to meet various needs around their models in one-stop within the SWIFT framework. Currently, the main capabilities of SWIFT include:

- Model Types: Supporting training and post-training for large models ranging from pure text models, multi-modal large models, to All-to-All fully modal models.
- Dataset Types: Covering pure text datasets, multi-modal datasets, and text-to-image datasets, suitable for different tasks.
- Task Types: Besides general generative tasks, it supports training for classification tasks.
- Lightweight Fine-tuning: Supports various lightweight fine-tuning methods such as LoRA, QLoRA, DoRA, ReFT, LLaMAPro, Adapter, SCEdit, GaLore, and Liger-Kernel.
- Training stages: Covering the entire stages of pre-training, fine-tuning, and human alignment.
- Training Parallelism: Covers single machine single card, single machine multiple card device mapping, distributed data parallelism (DDP), multi-machine multi-card, DeepSpeed, FSDP, PAI DLC, and supports training for models based on the Megatron architecture.
  - Extra support for [TorchAcc](https://github.imc.re/AlibabaPAI/torchacc) training acceleration.
  - Extra support for sequence parallelism based on [XTuner](https://github.com/InternLM/xtuner).
- Inference Deployment: Supports inference deployment on multiple frameworks such as PyTorch, vLLM, LmDeploy, which can be directly applied in Docker images or Kubernetes environments.
- Evaluation: Supports pure text and multi-modal evaluation capabilities based on the EvalScope framework, and allows for customized evaluation.
- Export: Supports quantization methods like awq, gptq, bnb, and operations for merging lora and llamapro.
- User Interface: Supports interface operations based on the Gradio framework and allows for the deployment of single model applications in space or demo environments.
- Plug-in System: Supports customizable definitions for loss, metrics, trainer, loss-scale, callback, optimizer, etc., making it easier for users to customize the training process.

## Installation

Installing SWIFT is straightforward. Please refer to the [installation documentation](./SWIFT-installation.md).

## Some Key Concepts

### Model Type

In SWIFT 3.0, the model_type is different from that in 2.0. The model_type in 3.x refers to a collection of models that share the following identical characteristics:
1. The same model architecture, such as the typical LLaMA structure.
2. The same template, such as those using the chatml format.
3. The same model loading method, like using get_model_tokenizer_flash_attn.

When all three points are identical, the models are classified into one group, and the type of this group is referred to as model_type.

## Usage Example

Comprehensive usage examples can be found in the [examples](https://github.com/modelscope/ms-swift/tree/main/examples) section. Below are some basic examples:

Command line method for LoRA training.
```shell
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset AI-ModelScope/alpaca-gpt4-data-zh#500 \
              AI-ModelScope/alpaca-gpt4-data-en#500 \
              swift/self-cognition#500 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot
```

You can check examples of training using code at [examples/notebook](https://github.com/modelscope/ms-swift/tree/main/examples/notebook).

Inference and deployment using the command line
```shell
# Inference
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model Qwen/Qwen2.5-7B-Instruct \
    --infer_backend pt
```

```shell
# Deployment
CUDA_VISIBLE_DEVICES=0 \
swift deploy \
    --model Qwen/Qwen2-7B-Instruct \
    --infer_backend pt
```

```python
# Client-side deployment code
from openai import OpenAI

client = OpenAI(
    api_key='EMPTY',
    base_url='http://localhost:8000/v1',
)
model_type = client.models.list().data[0].id
print(f'model_type: {model_type}')

query = 'Where is the capital of Zhejiang?'
messages = [{'role': 'user', 'content': query}]
resp = client.chat.completions.create(model=model_type, messages=messages, seed=42)
response = resp.choices[0].message.content
print(f'query: {query}')
print(f'response: {response}')

# Streaming
messages.append({'role': 'assistant', 'content': response})
query = 'What delicious food is there?'
messages.append({'role': 'user', 'content': query})
stream_resp = client.chat.completions.create(model=model_type, messages=messages, stream=True, seed=42)

print(f'query: {query}')
print('response: ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].delta.content, end='', flush=True)
print()
```

## Evaluation
```shell
swift eval \
  --model Qwen/Qwen2-7B-Instruct \
  --eval_limit 10 \
  --eval_dataset gsm8k
```

## Quantization
```shell
swift export \
  --model Qwen/Qwen2-7B-Instruct \
  --quant_method bnb \
  --quant_bits 8
```
