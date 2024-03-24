# VLLM Inference Acceleration and Deployment

## Table of Contents
- [Environment Preparation](#environment-preparation)
- [Inference Acceleration](#inference-acceleration)
- [Web-UI Acceleration](#web-ui-acceleration)
- [Deployment](#deployment)

## Environment Preparation
GPU devices: A10, 3090, V100, A100 are all supported.
```bash
# Set pip global mirror (speeds up downloads)
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# Install ms-swift
pip install ms-swift[llm] -U

# vllm version corresponds to cuda version, please select version according to `https://docs.vllm.ai/en/latest/getting_started/installation.html`
pip install vllm -U
pip install openai -U

# Environment alignment (usually not needed. If you get errors, you can run the code below, the repo uses the latest environment for testing)
pip install -r requirements/framework.txt -U
pip install -r requirements/llm.txt -U
```

## Inference Acceleration
vllm does not support bnb quantized models. The models supported by vllm can be found in [Supported Models](Supported-models-datasets.md#Models).

### qwen-7b-chat
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    ModelType, get_vllm_engine, get_default_template_type,
    get_template, inference_vllm
)

model_type = ModelType.qwen_7b_chat
llm_engine = get_vllm_engine(model_type)
template_type = get_default_template_type(model_type)
template = get_template(template_type, llm_engine.hf_tokenizer)
# Similar to `transformers.GenerationConfig` interface
llm_engine.generation_config.max_new_tokens = 256

request_list = [{'query': 'Hello!'}, {'query': 'Where is the capital of Zhejiang?'} ]
resp_list = inference_vllm(llm_engine, template, request_list)
for request, resp in zip(request_list, resp_list):
    print(f"query: {request['query']}")
    print(f"response: {resp['response']}")

history1 = resp_list[1]['history']
request_list = [{'query': 'What delicious food is there', 'history': history1}]
resp_list = inference_vllm(llm_engine, template, request_list)
for request, resp in zip(request_list, resp_list):
    print(f"query: {request['query']}")
    print(f"response: {resp['response']}")
    print(f"history: {resp['history']}")

"""Out[0]
query: Hello!
response: Hello! I'm happy to be of service. Is there anything I can help you with?
query: Where is the capital of Zhejiang?
response: Hangzhou is the capital of Zhejiang Province.
query: What delicious food is there
response: Hangzhou is a city of gastronomy, with many famous dishes and snacks such as West Lake Vinegar Fish, Dongpo Pork, Beggar's Chicken, etc. In addition, Hangzhou has many snack shops where you can taste a variety of local delicacies.
history: [('Where is the capital of Zhejiang?', 'Hangzhou is the capital of Zhejiang Province.'), ('What delicious food is there', "Hangzhou is a city of gastronomy, with many famous dishes and snacks such as West Lake Vinegar Fish, Dongpo Pork, Beggar's Chicken, etc. In addition, Hangzhou has many snack shops where you can taste a variety of local delicacies.")]
"""
```

### Streaming Output
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    ModelType, get_vllm_engine, get_default_template_type,
    get_template, inference_stream_vllm
)

model_type = ModelType.qwen_7b_chat
llm_engine = get_vllm_engine(model_type)
template_type = get_default_template_type(model_type)
template = get_template(template_type, llm_engine.hf_tokenizer)
# Similar to `transformers.GenerationConfig` interface
llm_engine.generation_config.max_new_tokens = 256

request_list = [{'query': 'Hello!'}, {'query': 'Where is the capital of Zhejiang?'}]
gen = inference_stream_vllm(llm_engine, template, request_list)
query_list = [request['query'] for request in request_list]
print(f"query_list: {query_list}")
for resp_list in gen:
    response_list = [resp['response'] for resp in resp_list]
    print(f'response_list: {response_list}')

history1 = resp_list[1]['history']
request_list = [{'query': 'What delicious food is there', 'history': history1}]
gen = inference_stream_vllm(llm_engine, template, request_list)
query = request_list[0]['query']
print(f"query: {query}")
for resp_list in gen:
    response = resp_list[0]['response']
    print(f'response: {response}')

history = resp_list[0]['history']
print(f'history: {history}')

"""Out[0]
query_list: ['Hello!', 'Where is the capital of Zhejiang?']
...
response_list: ['Hello! I'm happy to be of service. Is there anything I can help you with?', 'Hangzhou is the capital of Zhejiang Province.']
query: What delicious food is there
...
response: Hangzhou is a city of gastronomy, with many famous dishes and snacks such as West Lake Vinegar Fish, Dongpo Pork, Beggar's Chicken, etc. In addition, Hangzhou has many snack shops where you can taste a variety of local delicacies.
history: [('Where is the capital of Zhejiang?', 'Hangzhou is the capital of Zhejiang Province.'), ('What delicious food is there', "Hangzhou is a city of gastronomy, with many famous dishes and snacks such as West Lake Vinegar Fish, Dongpo Pork, Beggar's Chicken, etc. In addition, Hangzhou has many snack shops where you can taste a variety of local delicacies.")]
"""
```

### chatglm3
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    ModelType, get_vllm_engine, get_default_template_type,
    get_template, inference_vllm
)

model_type = ModelType.chatglm3_6b
llm_engine = get_vllm_engine(model_type)
template_type = get_default_template_type(model_type)
template = get_template(template_type, llm_engine.hf_tokenizer)
# Similar to `transformers.GenerationConfig` interface
llm_engine.generation_config.max_new_tokens = 256

request_list = [{'query': 'Hello!'}, {'query': 'Where is the capital of Zhejiang?'}]
resp_list = inference_vllm(llm_engine, template, request_list)
for request, resp in zip(request_list, resp_list):
    print(f"query: {request['query']}")
    print(f"response: {resp['response']}")

history1 = resp_list[1]['history']
request_list = [{'query': 'What delicious food is there', 'history': history1}]
resp_list = inference_vllm(llm_engine, template, request_list)
for request, resp in zip(request_list, resp_list):
    print(f"query: {request['query']}")
    print(f"response: {resp['response']}")
    print(f"history: {resp['history']}")

"""Out[0]
query: Hello!
response: Hello, I am an AI assistant. I'm very pleased to serve you! Do you have any questions I can help you answer?
query: Where is the capital of Zhejiang?
response: The capital of Zhejiang is Hangzhou.
query: What delicious food is there
response: Zhejiang has many delicious foods, some of the most famous ones include Longjing Shrimp from Hangzhou, Dongpo Pork, West Lake Vinegar Fish, Beggar's Chicken, etc. In addition, Zhejiang also has many specialty snacks and pastries, such as Tang Yuan and Nian Gao from Ningbo, stir-fried crab and Wenzhou meatballs from Wenzhou, etc.
history: [('Where is the capital of Zhejiang?', 'The capital of Zhejiang is Hangzhou.'), ('What delicious food is there', 'Zhejiang has many delicious foods, some of the most famous ones include Longjing Shrimp from Hangzhou, Dongpo Pork, West Lake Vinegar Fish, Beggar's Chicken, etc. In addition, Zhejiang also has many specialty snacks and pastries, such as Tang Yuan and Nian Gao from Ningbo, stir-fried crab and Wenzhou meatballs from Wenzhou, etc.')]
"""
```

### Using CLI
```bash
# qwen
CUDA_VISIBLE_DEVICES=0 swift infer --model_type qwen-7b-chat --infer_backend vllm
# yi
CUDA_VISIBLE_DEVICES=0 swift infer --model_type yi-6b-chat --infer_backend vllm
# gptq
CUDA_VISIBLE_DEVICES=0 swift infer --model_type qwen1half-7b-chat-int4 --infer_backend vllm
```

### Fine-tuned Models

**Single sample inference**:

For models fine-tuned using LoRA, you need to first [merge-lora](LLM-fine-tuning.md#merge-lora) to generate a complete checkpoint directory.

Models fine-tuned with full parameters can seamlessly use VLLM for inference acceleration.
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    ModelType, get_vllm_engine, get_default_template_type,
    get_template, inference_vllm
)
from swift.tuners import Swift

ckpt_dir = 'vx-xxx/checkpoint-100-merged'
model_type = ModelType.qwen_7b_chat
template_type = get_default_template_type(model_type)

llm_engine = get_vllm_engine(model_type, model_id_or_path=ckpt_dir)
tokenizer = llm_engine.hf_tokenizer
template = get_template(template_type, tokenizer)
query = 'Hello'
resp = inference_vllm(llm_engine, template, [{'query': query}])[0]
print(f"response: {resp['response']}")
print(f"history: {resp['history']}")
```

**Using CLI**:
```bash
# merge LoRA incremental weights and use vllm for inference acceleration
# If you need quantization, you can specify `--quant_bits 4`.
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx' --merge_lora true

# Evaluate using dataset
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx-merged' \
    --infer_backend vllm \
    --load_dataset_config true \

# Manual evaluation
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx-merged' \
    --infer_backend vllm \
```

## Web-UI Acceleration

### Original Models
```bash
CUDA_VISIBLE_DEVICES=0 swift app-ui --model_type qwen-7b-chat --infer_backend vllm
```

### Fine-tuned Models
```bash
# merge LoRA incremental weights and use vllm as backend to build app-ui
# If you need quantization, you can specify `--quant_bits 4`.
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx' --merge_lora true

CUDA_VISIBLE_DEVICES=0 swift app-ui --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx-merged' --infer_backend vllm
```

## Deployment
Swift uses VLLM as the inference backend and is compatible with the OpenAI API style.

For server deployment command line arguments, refer to: [deploy command line arguments](Command-line-parameters.md#deploy-Parameters).

For OpenAI API arguments on the client side, refer to: https://platform.openai.com/docs/api-reference/introduction.

### Original Models
#### qwen-7b-chat

**Server side:**
```bash
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen-7b-chat
# Multi-GPU deployment
RAY_memory_monitor_refresh_ms=0 CUDA_VISIBLE_DEVICES=0,1,2,3 swift deploy --model_type qwen-7b-chat --tensor_parallel_size 4
```

**Client side:**

Test:
```bash
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "qwen-7b-chat",
"messages": [{"role": "user", "content": "What to do if I can't fall asleep at night?"}],
"max_tokens": 256,
"temperature": 0
}'
```

Using swift:
```python
from swift.llm import get_model_list_client, XRequestConfig, inference_client

model_list = get_model_list_client()
model_type = model_list.data[0].id
print(f'model_type: {model_type}')

query = 'Where is the capital of Zhejiang?'
request_config = XRequestConfig(seed=42)
resp = inference_client(model_type, query, request_config=request_config)
response = resp.choices[0].message.content
print(f'query: {query}')
print(f'response: {response}')

history = [(query, response)]
query = 'What delicious food is there?'
request_config = XRequestConfig(stream=True, seed=42)
stream_resp = inference_client(model_type, query, history, request_config=request_config)
print(f'query: {query}')
print('response: ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].delta.content, end='', flush=True)
print()

"""Out[0]
model_type: qwen-7b-chat
query: Where is the capital of Zhejiang?
response: The capital of Zhejiang Province is Hangzhou.
query: What delicious food is there?
response: Hangzhou has many delicious foods, such as West Lake Vinegar Fish, Dongpo Pork, Longjing Shrimp, Beggar's Chicken, etc. In addition, Hangzhou also has many specialty snacks, such as West Lake Lotus Root Powder, Hangzhou Xiao Long Bao, Hangzhou You Tiao, etc.
"""
```

Using openai:
```python
from openai import OpenAI
client = OpenAI(
    api_key='EMPTY',
    base_url='http://localhost:8000/v1',
)
model_type = client.models.list().data[0].id
print(f'model_type: {model_type}')

query = 'Where is the capital of Zhejiang?'
messages = [{
    'role': 'user',
    'content': query
}]
resp = client.chat.completions.create(
    model=model_type,
    messages=messages,
    seed=42)
response = resp.choices[0].message.content
print(f'query: {query}')
print(f'response: {response}')

# Streaming
messages.append({'role': 'assistant', 'content': response})
query = 'What delicious food is there?'
messages.append({'role': 'user', 'content': query})
stream_resp = client.chat.completions.create(
    model=model_type,
    messages=messages,
    stream=True,
    seed=42)

print(f'query: {query}')
print('response: ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].delta.content, end='', flush=True)
print()

"""Out[0]
model_type: qwen-7b-chat
query: Where is the capital of Zhejiang?
response: The capital of Zhejiang Province is Hangzhou.
query: What delicious food is there?
response: Hangzhou has many delicious foods, such as West Lake Vinegar Fish, Dongpo Pork, Longjing Shrimp, Beggar's Chicken, etc. In addition, Hangzhou also has many specialty snacks, such as West Lake Lotus Root Powder, Hangzhou Xiao Long Bao, Hangzhou You Tiao, etc.
"""
```

#### qwen-7b

**Server side:**
```bash
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen-7b
# Multi-GPU deployment
RAY_memory_monitor_refresh_ms=0 CUDA_VISIBLE_DEVICES=0,1,2,3 swift deploy --model_type qwen-7b --tensor_parallel_size 4
```

**Client side:**

Test:
```bash
curl http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "qwen-7b",
"prompt": "Zhejiang -> Hangzhou\nAnhui -> Hefei\nSichuan ->",
"max_tokens": 32,
"temperature": 0.1,
"seed": 42
}'
```

Using swift:
```python
from swift.llm import get_model_list_client, XRequestConfig, inference_client

model_list = get_model_list_client()
model_type = model_list.data[0].id
print(f'model_type: {model_type}')

query = 'Zhejiang -> Hangzhou\nAnhui -> Hefei\nSichuan ->'
request_config = XRequestConfig(max_tokens=32, temperature=0.1, seed=42)
resp = inference_client(model_type, query, request_config=request_config)
response = resp.choices[0].text
print(f'query: {query}')
print(f'response: {response}')

request_config.stream = True
stream_resp = inference_client(model_type, query, request_config=request_config)
print(f'query: {query}')
print('response: ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].text, end='', flush=True)
print()

"""Out[0]
model_type: qwen-7b
query: Zhejiang -> Hangzhou
Anhui -> Hefei
Sichuan ->
response:  Chengdu
Guangdong -> Guangzhou
Jiangsu -> Nanjing
Zhejiang -> Hangzhou
Anhui -> Hefei
Sichuan -> Chengdu

query: Zhejiang -> Hangzhou
Anhui -> Hefei
Sichuan ->
response:  Chengdu
Guangdong -> Guangzhou
Jiangsu -> Nanjing
Zhejiang -> Hangzhou
Anhui -> Hefei
Sichuan -> Chengdu
"""
```

Using openai:
```python
from openai import OpenAI
client = OpenAI(
    api_key='EMPTY',
    base_url='http://localhost:8000/v1',
)
model_type = client.models.list().data[0].id
print(f'model_type: {model_type}')

query = 'Zhejiang -> Hangzhou\nAnhui -> Hefei\nSichuan ->'
kwargs = {'model': model_type, 'prompt': query, 'seed': 42, 'temperature': 0.1, 'max_tokens': 32}

resp = client.completions.create(**kwargs)
response = resp.choices[0].text
print(f'query: {query}')
print(f'response: {response}')

# Streaming
stream_resp = client.completions.create(stream=True, **kwargs)
response = resp.choices[0].text
print(f'query: {query}')
print('response: ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].text, end='', flush=True)
print()

"""Out[0]
model_type: qwen-7b
query: Zhejiang -> Hangzhou
Anhui -> Hefei
Sichuan ->
response:  Chengdu
Guangdong -> Guangzhou
Jiangsu -> Nanjing
Zhejiang -> Hangzhou
Anhui -> Hefei
Sichuan -> Chengdu

query: Zhejiang -> Hangzhou
Anhui -> Hefei
Sichuan ->
response:  Chengdu
Guangdong -> Guangzhou
Jiangsu -> Nanjing
Zhejiang -> Hangzhou
Anhui -> Hefei
Sichuan -> Chengdu
"""
```

### Fine-tuned Models
Server side:
```bash
# merge LoRA incremental weights and deploy
# If you need quantization, you can specify `--quant_bits 4`.
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx' --merge_lora true

CUDA_VISIBLE_DEVICES=0 swift deploy --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx-merged'
```

The example code for the client side is the same as the original models.
