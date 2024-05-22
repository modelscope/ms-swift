
# VLLM推理加速与部署

## 目录
- [环境准备](#环境准备)
- [推理加速](#推理加速)
- [Web-UI加速](#web-ui加速)
- [部署](#部署)
- [VLLM & LoRA](#vllm--lora)


## 环境准备
GPU设备: A10, 3090, V100, A100均可.
```bash
# 设置pip全局镜像 (加速下载)
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# 安装ms-swift
pip install 'ms-swift[llm]' -U

# vllm与cuda版本有对应关系，请按照`https://docs.vllm.ai/en/latest/getting_started/installation.html`选择版本
pip install vllm -U
pip install openai -U

# 环境对齐 (通常不需要运行. 如果你运行错误, 可以跑下面的代码, 仓库使用最新环境测试)
pip install -r requirements/framework.txt  -U
pip install -r requirements/llm.txt  -U
```

## 推理加速
vllm不支持bnb量化的模型. vllm支持的模型可以查看[支持的模型](支持的模型和数据集.md#模型).

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
# 与`transformers.GenerationConfig`类似的接口
llm_engine.generation_config.max_new_tokens = 256

request_list = [{'query': '你好!'}, {'query': '浙江的省会在哪？'}]
resp_list = inference_vllm(llm_engine, template, request_list)
for request, resp in zip(request_list, resp_list):
    print(f"query: {request['query']}")
    print(f"response: {resp['response']}")

history1 = resp_list[1]['history']
request_list = [{'query': '这有什么好吃的', 'history': history1}]
resp_list = inference_vllm(llm_engine, template, request_list)
for request, resp in zip(request_list, resp_list):
    print(f"query: {request['query']}")
    print(f"response: {resp['response']}")
    print(f"history: {resp['history']}")

"""Out[0]
query: 你好!
response: 你好！很高兴为你服务。有什么我可以帮助你的吗？
query: 浙江的省会在哪？
response: 浙江省会是杭州市。
query: 这有什么好吃的
response: 杭州是一个美食之城，拥有许多著名的菜肴和小吃，例如西湖醋鱼、东坡肉、叫化童子鸡等。此外，杭州还有许多小吃店，可以品尝到各种各样的本地美食。
history: [('浙江的省会在哪？', '浙江省会是杭州市。'), ('这有什么好吃的', '杭州是一个美食之城，拥有许多著名的菜肴和小吃，例如西湖醋鱼、东坡肉、叫化童子鸡等。此外，杭州还有许多小吃店，可以品尝到各种各样的本地美食。')]
"""
```

### 流式输出
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
# 与`transformers.GenerationConfig`类似的接口
llm_engine.generation_config.max_new_tokens = 256

request_list = [{'query': '你好!'}, {'query': '浙江的省会在哪？'}]
gen = inference_stream_vllm(llm_engine, template, request_list)
query_list = [request['query'] for request in request_list]
print(f"query_list: {query_list}")
for resp_list in gen:
    response_list = [resp['response'] for resp in resp_list]
    print(f'response_list: {response_list}')

history1 = resp_list[1]['history']
request_list = [{'query': '这有什么好吃的', 'history': history1}]
gen = inference_stream_vllm(llm_engine, template, request_list)
query = request_list[0]['query']
print(f"query: {query}")
for resp_list in gen:
    response = resp_list[0]['response']
    print(f'response: {response}')

history = resp_list[0]['history']
print(f'history: {history}')

"""Out[0]
query_list: ['你好!', '浙江的省会在哪？']
...
response_list: ['你好！很高兴为你服务。有什么我可以帮助你的吗？', '浙江省会是杭州市。']
query: 这有什么好吃的
...
response: 杭州是一个美食之城，拥有许多著名的菜肴和小吃，例如西湖醋鱼、东坡肉、叫化童子鸡等。此外，杭州还有许多小吃店，可以品尝到各种各样的本地美食。
history: [('浙江的省会在哪？', '浙江省会是杭州市。'), ('这有什么好吃的', '杭州是一个美食之城，拥有许多著名的菜肴和小吃，例如西湖醋鱼、东坡肉、叫化童子鸡等。此外，杭州还有许多小吃店，可以品尝到各种各样的本地美食。')]
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
# 与`transformers.GenerationConfig`类似的接口
llm_engine.generation_config.max_new_tokens = 256

request_list = [{'query': '你好!'}, {'query': '浙江的省会在哪？'}]
resp_list = inference_vllm(llm_engine, template, request_list)
for request, resp in zip(request_list, resp_list):
    print(f"query: {request['query']}")
    print(f"response: {resp['response']}")

history1 = resp_list[1]['history']
request_list = [{'query': '这有什么好吃的', 'history': history1}]
resp_list = inference_vllm(llm_engine, template, request_list)
for request, resp in zip(request_list, resp_list):
    print(f"query: {request['query']}")
    print(f"response: {resp['response']}")
    print(f"history: {resp['history']}")

"""Out[0]
query: 你好!
response: 您好，我是人工智能助手。很高兴为您服务！请问有什么问题我可以帮您解答？
query: 浙江的省会在哪？
response: 浙江的省会是杭州。
query: 这有什么好吃的
response: 浙江有很多美食,其中一些非常有名的包括杭州的龙井虾仁、东坡肉、西湖醋鱼、叫化童子鸡等。另外,浙江还有很多特色小吃和糕点,比如宁波的汤团、年糕,温州的炒螃蟹、温州肉圆等。
history: [('浙江的省会在哪？', '浙江的省会是杭州。'), ('这有什么好吃的', '浙江有很多美食,其中一些非常有名的包括杭州的龙井虾仁、东坡肉、西湖醋鱼、叫化童子鸡等。另外,浙江还有很多特色小吃和糕点,比如宁波的汤团、年糕,温州的炒螃蟹、温州肉圆等。')]
"""
```

### 使用CLI
```bash
# qwen
CUDA_VISIBLE_DEVICES=0 swift infer --model_type qwen-7b-chat --infer_backend vllm
# yi
CUDA_VISIBLE_DEVICES=0 swift infer --model_type yi-6b-chat --infer_backend vllm
# gptq
CUDA_VISIBLE_DEVICES=0 swift infer --model_type qwen1half-7b-chat-int4 --infer_backend vllm
```

### 微调后的模型

**单样本推理**:

使用LoRA进行微调的模型你需要先[merge-lora](LLM微调文档.md#merge-lora), 产生完整的checkpoint目录.

使用全参数微调的模型可以无缝使用VLLM进行推理加速.
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    ModelType, get_vllm_engine, get_default_template_type,
    get_template, inference_vllm
)

ckpt_dir = 'vx-xxx/checkpoint-100-merged'
model_type = ModelType.qwen_7b_chat
template_type = get_default_template_type(model_type)

llm_engine = get_vllm_engine(model_type, model_id_or_path=ckpt_dir)
tokenizer = llm_engine.hf_tokenizer
template = get_template(template_type, tokenizer)
query = '你好'
resp = inference_vllm(llm_engine, template, [{'query': query}])[0]
print(f"response: {resp['response']}")
print(f"history: {resp['history']}")
```

**使用CLI**:
```bash
# merge LoRA增量权重并使用vllm进行推理加速
# 如果你需要量化, 可以指定`--quant_bits 4`.
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx' --merge_lora true

# 使用数据集评估
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx-merged' \
    --infer_backend vllm \
    --load_dataset_config true \

# 人工评估
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx-merged' \
    --infer_backend vllm \
```

## Web-UI加速

### 原始模型
```bash
CUDA_VISIBLE_DEVICES=0 swift app-ui --model_type qwen-7b-chat --infer_backend vllm
```

### 微调后模型
```bash
# merge LoRA增量权重并使用vllm作为backend构建app-ui
# 如果你需要量化, 可以指定`--quant_bits 4`.
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx' --merge_lora true

CUDA_VISIBLE_DEVICES=0 swift app-ui --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx-merged' --infer_backend vllm
```

## 部署
swift使用VLLM作为推理后端, 并兼容openai的API样式.

服务端的部署命令行参数可以参考: [deploy命令行参数](命令行参数.md#deploy-参数).

客户端的openai的API参数可以参考: https://platform.openai.com/docs/api-reference/introduction.

### 原始模型
#### qwen-7b-chat

**服务端:**
```bash
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen-7b-chat
# 多卡部署
RAY_memory_monitor_refresh_ms=0 CUDA_VISIBLE_DEVICES=0,1,2,3 swift deploy --model_type qwen-7b-chat --tensor_parallel_size 4
```

**客户端:**

测试:
```bash
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "qwen-7b-chat",
"messages": [{"role": "user", "content": "晚上睡不着觉怎么办？"}],
"max_tokens": 256,
"temperature": 0
}'
```

使用swift:
```python
from swift.llm import get_model_list_client, XRequestConfig, inference_client

model_list = get_model_list_client()
model_type = model_list.data[0].id
print(f'model_type: {model_type}')

query = '浙江的省会在哪里?'
request_config = XRequestConfig(seed=42)
resp = inference_client(model_type, query, request_config=request_config)
response = resp.choices[0].message.content
print(f'query: {query}')
print(f'response: {response}')

history = [(query, response)]
query = '这有什么好吃的?'
request_config = XRequestConfig(stream=True, seed=42)
stream_resp = inference_client(model_type, query, history, request_config=request_config)
print(f'query: {query}')
print('response: ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].delta.content, end='', flush=True)
print()

"""Out[0]
model_type: qwen-7b-chat
query: 浙江的省会在哪里?
response: 浙江省的省会是杭州市。
query: 这有什么好吃的?
response: 杭州有许多美食，例如西湖醋鱼、东坡肉、龙井虾仁、叫化童子鸡等。此外，杭州还有许多特色小吃，如西湖藕粉、杭州小笼包、杭州油条等。
"""
```

使用openai:
```python
from openai import OpenAI
client = OpenAI(
    api_key='EMPTY',
    base_url='http://localhost:8000/v1',
)
model_type = client.models.list().data[0].id
print(f'model_type: {model_type}')

query = '浙江的省会在哪里?'
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

# 流式
messages.append({'role': 'assistant', 'content': response})
query = '这有什么好吃的?'
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
query: 浙江的省会在哪里?
response: 浙江省的省会是杭州市。
query: 这有什么好吃的?
response: 杭州有许多美食，例如西湖醋鱼、东坡肉、龙井虾仁、叫化童子鸡等。此外，杭州还有许多特色小吃，如西湖藕粉、杭州小笼包、杭州油条等。
"""
```

#### qwen-7b

**服务端:**
```bash
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen-7b
# 多卡部署
RAY_memory_monitor_refresh_ms=0 CUDA_VISIBLE_DEVICES=0,1,2,3 swift deploy --model_type qwen-7b --tensor_parallel_size 4
```

**客户端:**

测试:
```bash
curl http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "qwen-7b",
"prompt": "浙江 -> 杭州\n安徽 -> 合肥\n四川 ->",
"max_tokens": 32,
"temperature": 0.1,
"seed": 42
}'
```

使用swift:
```python
from swift.llm import get_model_list_client, XRequestConfig, inference_client

model_list = get_model_list_client()
model_type = model_list.data[0].id
print(f'model_type: {model_type}')

query = '浙江 -> 杭州\n安徽 -> 合肥\n四川 ->'
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
query: 浙江 -> 杭州
安徽 -> 合肥
四川 ->
response:  成都
广东 -> 广州
江苏 -> 南京
浙江 -> 杭州
安徽 -> 合肥
四川 -> 成都

query: 浙江 -> 杭州
安徽 -> 合肥
四川 ->
response:  成都
广东 -> 广州
江苏 -> 南京
浙江 -> 杭州
安徽 -> 合肥
四川 -> 成都
"""
```

使用openai:
```python
from openai import OpenAI
client = OpenAI(
    api_key='EMPTY',
    base_url='http://localhost:8000/v1',
)
model_type = client.models.list().data[0].id
print(f'model_type: {model_type}')

query = '浙江 -> 杭州\n安徽 -> 合肥\n四川 ->'
kwargs = {'model': model_type, 'prompt': query, 'seed': 42, 'temperature': 0.1, 'max_tokens': 32}

resp = client.completions.create(**kwargs)
response = resp.choices[0].text
print(f'query: {query}')
print(f'response: {response}')

# 流式
stream_resp = client.completions.create(stream=True, **kwargs)
response = resp.choices[0].text
print(f'query: {query}')
print('response: ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].text, end='', flush=True)
print()

"""Out[0]
model_type: qwen-7b
query: 浙江 -> 杭州
安徽 -> 合肥
四川 ->
response:  成都
广东 -> 广州
江苏 -> 南京
浙江 -> 杭州
安徽 -> 合肥
四川 -> 成都

query: 浙江 -> 杭州
安徽 -> 合肥
四川 ->
response:  成都
广东 -> 广州
江苏 -> 南京
浙江 -> 杭州
安徽 -> 合肥
四川 -> 成都
"""
```

### 微调后模型
服务端:
```bash
# merge LoRA增量权重并部署
# 如果你需要量化, 可以指定`--quant_bits 4`.
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx' --merge_lora true

CUDA_VISIBLE_DEVICES=0 swift deploy --ckpt_dir 'xxx/vx-xxx/checkpoint-xxx-merged'
```

客户端示例代码同原始模型.

### 多LoRA部署

目前pt方式部署模型已经支持`peft>=0.10.0`进行多LoRA部署，具体方法为：

- 确保部署时`merge_lora`为`False`
- 使用`--lora_modules`参数,  可以查看[命令行文档](命令行参数.md)
- 推理时指定lora tuner的名字到模型字段

举例：

```shell
# 假设从llama3-8b-instruct训练了一个名字叫卡卡罗特的LoRA模型
# 服务端
swift deploy --ckpt_dir /mnt/ckpt-1000 --infer_backend pt --lora_modules my_tuner=/mnt/my-tuner
# 会加载起来两个tuner，一个是`/mnt/ckpt-1000`的`default-lora`，一个是`/mnt/my-tuner`的`my_tuner`

# 客户端
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
"model": "my-tuner",
"messages": [{"role": "user", "content": "who are you?"}],
"max_tokens": 256,
"temperature": 0
}'
# resp: 我是卡卡罗特...
# 如果指定mode='llama3-8b-instruct'，则返回I'm llama3...，即原模型的返回值
```

> [!NOTE]
>
> `--ckpt_dir`参数如果是个lora路径，则原来的default会被加载到default-lora的tuner上，其他的tuner需要通过`lora_modules`自行加载

## VLLM & LoRA

VLLM & LoRA支持的模型可以查看: https://docs.vllm.ai/en/latest/models/supported_models.html

### 准备LoRA
```shell
# Experimental environment: 4 * A100
# 4 * 30GB GPU memory
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
swift sft \
    --model_type llama2-7b-chat \
    --dataset self-cognition#500 sharegpt-gpt4-mini#1000 \
    --logging_steps 5 \
    --max_length 4096 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.4 \
    --output_dir output \
    --lora_target_modules ALL \
    --model_name 小黄 'Xiao Huang' \
    --model_author 魔搭 ModelScope \
```

将lora从swift格式转换成peft格式:
```shell
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir output/llama2-7b-chat/vx-xxx/checkpoint-xxx \
    --to_peft_format true
```


### VLLM推理加速

推理:
```shell
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/llama2-7b-chat/vx-xxx/checkpoint-xxx-peft \
    --infer_backend vllm \
    --vllm_enable_lora true
```

运行结果:
```python
"""
<<< who are you?
I am an artificial intelligence language model developed by ModelScope. I am designed to assist and communicate with users in a helpful and respectful manner. I can answer questions, provide information, and engage in conversation. How can I help you?
"""
```

单样本推理:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from swift.llm import (
    ModelType, get_vllm_engine, get_default_template_type,
    get_template, inference_stream_vllm, LoRARequest, inference_vllm
)

lora_checkpoint = 'output/llama2-7b-chat/vx-xxx/checkpoint-xxx-peft'
lora_request = LoRARequest('default-lora', 1, lora_checkpoint)

model_type = ModelType.llama2_7b_chat
llm_engine = get_vllm_engine(model_type, torch.float16, enable_lora=True,
                             max_loras=1, max_lora_rank=16)
template_type = get_default_template_type(model_type)
template = get_template(template_type, llm_engine.hf_tokenizer)
# 与`transformers.GenerationConfig`类似的接口
llm_engine.generation_config.max_new_tokens = 256

# use lora
request_list = [{'query': 'who are you?'}]
query = request_list[0]['query']
resp_list = inference_vllm(llm_engine, template, request_list, lora_request=lora_request)
response = resp_list[0]['response']
print(f'query: {query}')
print(f'response: {response}')

# no lora
gen = inference_stream_vllm(llm_engine, template, request_list)
query = request_list[0]['query']
print(f'query: {query}\nresponse: ', end='')
print_idx = 0
for resp_list in gen:
    response = resp_list[0]['response']
    print(response[print_idx:], end='', flush=True)
    print_idx = len(response)
print()
"""
query: who are you?
response: I am an artificial intelligence language model developed by ModelScope. I can understand and respond to text-based questions and prompts, and provide information and assistance on a wide range of topics.
query: who are you?
response:  Hello! I'm just an AI assistant, here to help you with any questions or tasks you may have. I'm designed to be helpful, respectful, and honest in my responses, and I strive to provide socially unbiased and positive answers. I'm not a human, but a machine learning model trained on a large dataset of text to generate responses to a wide range of questions and prompts. I'm here to help you in any way I can, while always ensuring that my answers are safe and respectful. Is there anything specific you'd like to know or discuss?
"""
```


### 部署

**服务端**:
```shell
CUDA_VISIBLE_DEVICES=0 swift deploy \
    --ckpt_dir output/llama2-7b-chat/vx-xxx/checkpoint-xxx-peft \
    --infer_backend vllm \
    --vllm_enable_lora true
```

**客户端**:

测试:
```bash
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "default-lora",
"messages": [{"role": "user", "content": "who are you?"}],
"max_tokens": 256,
"temperature": 0
}'

curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "llama2-7b-chat",
"messages": [{"role": "user", "content": "who are you?"}],
"max_tokens": 256,
"temperature": 0
}'
```

输出:
```python
"""
{"model":"default-lora","choices":[{"index":0,"message":{"role":"assistant","content":"I am an artificial intelligence language model developed by ModelScope. I am designed to assist and communicate with users in a helpful, respectful, and honest manner. I can answer questions, provide information, and engage in conversation. How can I assist you?"},"finish_reason":"stop"}],"usage":{"prompt_tokens":141,"completion_tokens":53,"total_tokens":194},"id":"chatcmpl-fb95932dcdab4ce68f4be49c9946b306","object":"chat.completion","created":1710820459}

{"model":"llama2-7b-chat","choices":[{"index":0,"message":{"role":"assistant","content":" Hello! I'm just an AI assistant, here to help you with any questions or concerns you may have. I'm designed to provide helpful, respectful, and honest responses, while ensuring that my answers are socially unbiased and positive in nature. I'm not capable of providing harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, and I will always do my best to explain why I cannot answer a question if it does not make sense or is not factually coherent. If I don't know the answer to a question, I will not provide false information. My goal is to assist and provide accurate information to the best of my abilities. Is there anything else I can help you with?"},"finish_reason":"stop"}],"usage":{"prompt_tokens":141,"completion_tokens":163,"total_tokens":304},"id":"chatcmpl-d867a3a52bb7451588d4f73e1df4ba95","object":"chat.completion","created":1710820557}
"""
```

使用openai:
```python
from openai import OpenAI
client = OpenAI(
    api_key='EMPTY',
    base_url='http://localhost:8000/v1',
)
model_type_list = [model.id for model in client.models.list().data]
print(f'model_type_list: {model_type_list}')

query = 'who are you?'
messages = [{
    'role': 'user',
    'content': query
}]
resp = client.chat.completions.create(
    model='default-lora',
    messages=messages,
    seed=42)
response = resp.choices[0].message.content
print(f'query: {query}')
print(f'response: {response}')

# 流式
stream_resp = client.chat.completions.create(
    model='llama2-7b-chat',
    messages=messages,
    stream=True,
    seed=42)

print(f'query: {query}')
print('response: ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].delta.content, end='', flush=True)
print()

"""Out[0]
model_type_list: ['llama2-7b-chat', 'default-lora']
query: who are you?
response: I am an artificial intelligence language model developed by ModelScope. I am designed to assist and communicate with users in a helpful, respectful, and honest manner. I can answer questions, provide information, and engage in conversation. How can I assist you?
query: who are you?
response:  Hello! I'm just an AI assistant, here to help you with any questions or concerns you may have. I'm designed to provide helpful, respectful, and honest responses, while ensuring that my answers are socially unbiased and positive in nature. I'm not capable of providing harmful, unethical, racist, sexist, toxic, dangerous, or illegal content, and I will always do my best to explain why I cannot answer a question if it does not make sense or is not factually coherent. If I don't know the answer to a question, I will not provide false information. Is there anything else I can help you with?
"""
```
