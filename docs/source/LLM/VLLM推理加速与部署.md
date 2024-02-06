
# VLLM推理加速与部署

## 目录
- [环境准备](#环境准备)
- [推理加速](#推理加速)
- [Web-UI加速](#web-ui加速)
- [部署](#部署)

## 环境准备
GPU设备: A10, 3090, V100, A100均可.
```bash
# 设置pip全局镜像
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# 安装ms-swift
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e .[llm]

# vllm与cuda版本有对应关系，请按照`https://docs.vllm.ai/en/latest/getting_started/installation.html`选择版本
pip install vllm -U
pip install openai -U

# 环境对齐 (如果你运行错误, 可以跑下面的代码, 仓库使用最新环境测试)
pip install -r requirements/framework.txt  -U
pip install -r requirements/llm.txt  -U
```

## 推理加速
vllm不支持bnb和auto_gptq量化的模型. vllm支持的模型可以查看[支持的模型](./支持的模型和数据集.md#模型).

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
```

### 微调后的模型

**单样本推理**:

使用LoRA进行微调的模型你需要先[merge-lora](./LLM微调文档.md#merge-lora), 产生完整的checkpoint目录.

使用全参数微调的模型可以无缝使用VLLM进行推理加速.
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    ModelType, get_vllm_engine, get_default_template_type,
    get_template, inference_vllm
)
from swift.tuners import Swift

model_dir = 'vx_xxx/checkpoint-100-merged'
model_type = ModelType.qwen_7b_chat
template_type = get_default_template_type(model_type)

llm_engine = get_vllm_engine(model_type, model_dir=model_dir)
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
swift merge-lora --ckpt_dir 'xxx/vx_xxx/checkpoint-xxx'

# 使用数据集评估
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --ckpt_dir 'xxx/vx_xxx/checkpoint-xxx-merged' \
    --infer_backend vllm \
    --load_dataset_config true \

# 人工评估
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --ckpt_dir 'xxx/vx_xxx/checkpoint-xxx-merged' \
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
swift merge-lora --ckpt_dir 'xxx/vx_xxx/checkpoint-xxx'
CUDA_VISIBLE_DEVICES=0 swift app-ui --ckpt_dir 'xxx/vx_xxx/checkpoint-xxx-merged' --infer_backend vllm
```

## 部署
swift使用VLLM作为推理后端, 并兼容openai的API样式.

服务端的部署命令行参数可以参考: [deploy命令行参数](命令行参数.md#deploy-命令行参数).

客户端的openai的API参数可以参考: https://platform.openai.com/docs/api-reference/introduction.

### 原始模型
#### qwen-7b-chat

**服务端:**
```bash
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen-7b-chat
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
swift merge-lora --ckpt_dir 'xxx/vx_xxx/checkpoint-xxx'
CUDA_VISIBLE_DEVICES=0 swift deploy --ckpt_dir 'xxx/vx_xxx/checkpoint-xxx-merged'
```

客户端示例代码同原始模型.
