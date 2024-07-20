# LmDeploy推理加速与部署

## 目录
- [环境准备](#环境准备)
- [推理加速](#推理加速)
- [部署](#部署)
- [多模态](#多模态)

## 环境准备
GPU设备: A10, 3090, V100, A100均可.
```bash
# 设置pip全局镜像 (加速下载)
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# 安装ms-swift
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'

pip install lmdeploy
```

## 推理加速

### 使用python

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    ModelType, get_lmdeploy_engine, get_default_template_type,
    get_template, inference_lmdeploy, inference_stream_lmdeploy
)

model_type = ModelType.qwen_7b_chat
lmdeploy_engine = get_lmdeploy_engine(model_type)
template_type = get_default_template_type(model_type)
template = get_template(template_type, lmdeploy_engine.hf_tokenizer)
# 与`transformers.GenerationConfig`类似的接口
lmdeploy_engine.generation_config.max_new_tokens = 256
generation_info = {}

request_list = [{'query': '你好!'}, {'query': '浙江的省会在哪？'}]
resp_list = inference_lmdeploy(lmdeploy_engine, template, request_list, generation_info=generation_info)
for request, resp in zip(request_list, resp_list):
    print(f"query: {request['query']}")
    print(f"response: {resp['response']}")
print(generation_info)

# stream
history1 = resp_list[1]['history']
request_list = [{'query': '这有什么好吃的', 'history': history1}]
gen = inference_stream_lmdeploy(lmdeploy_engine, template, request_list, generation_info=generation_info)
query = request_list[0]['query']
print_idx = 0
print(f'query: {query}\nresponse: ', end='')
for resp_list in gen:
    resp = resp_list[0]
    response = resp['response']
    delta = response[print_idx:]
    print(delta, end='', flush=True)
    print_idx = len(response)
print()

history = resp_list[0]['history']
print(f'history: {history}')
print(generation_info)
"""
query: 你好!
response: 你好！有什么我能帮助你的吗？
query: 浙江的省会在哪？
response: 浙江省会是杭州市。
{'num_prompt_tokens': 46, 'num_generated_tokens': 13, 'num_samples': 2, 'runtime': 0.2037766759749502, 'samples/s': 9.81466593480922, 'tokens/s': 63.79532857625993}
query: 这有什么好吃的
response: 杭州有许多美食，比如西湖醋鱼、东坡肉、龙井虾仁、油炸臭豆腐等，都是当地非常有名的传统名菜。此外，当地的点心也非常有特色，比如桂花糕、马蹄酥、绿豆糕等。
history: [['浙江的省会在哪？', '浙江省会是杭州市。'], ['这有什么好吃的', '杭州有许多美食，比如西湖醋鱼、东坡肉、龙井虾仁、油炸臭豆腐等，都是当地非常有名的传统名菜。此外，当地的点心也非常有特色，比如桂花糕、马蹄酥、绿豆糕等。']]
{'num_prompt_tokens': 44, 'num_generated_tokens': 53, 'num_samples': 1, 'runtime': 0.6306625790311955, 'samples/s': 1.5856339558566632, 'tokens/s': 84.03859966040315}
"""
```

**TP:**

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

from swift.llm import (
    ModelType, get_lmdeploy_engine, get_default_template_type,
    get_template, inference_lmdeploy, inference_stream_lmdeploy
)

model_type = ModelType.qwen_7b_chat
lmdeploy_engine = get_lmdeploy_engine(model_type, tp=2)
template_type = get_default_template_type(model_type)
template = get_template(template_type, lmdeploy_engine.hf_tokenizer)
# 与`transformers.GenerationConfig`类似的接口
lmdeploy_engine.generation_config.max_new_tokens = 256
generation_info = {}

request_list = [{'query': '你好!'}, {'query': '浙江的省会在哪？'}]
resp_list = inference_lmdeploy(lmdeploy_engine, template, request_list, generation_info=generation_info)
for request, resp in zip(request_list, resp_list):
    print(f"query: {request['query']}")
    print(f"response: {resp['response']}")
print(generation_info)

# stream
history1 = resp_list[1]['history']
request_list = [{'query': '这有什么好吃的', 'history': history1}]
gen = inference_stream_lmdeploy(lmdeploy_engine, template, request_list, generation_info=generation_info)
query = request_list[0]['query']
print_idx = 0
print(f'query: {query}\nresponse: ', end='')
for resp_list in gen:
    resp = resp_list[0]
    response = resp['response']
    delta = response[print_idx:]
    print(delta, end='', flush=True)
    print_idx = len(response)
print()

history = resp_list[0]['history']
print(f'history: {history}')
print(generation_info)
"""
query: 你好!
response: 你好！有什么我能帮助你的吗？
query: 浙江的省会在哪？
response: 浙江省会是杭州市。
{'num_prompt_tokens': 46, 'num_generated_tokens': 13, 'num_samples': 2, 'runtime': 0.2080078640137799, 'samples/s': 9.61502109298861, 'tokens/s': 62.497637104425955}
query: 这有什么好吃的
response: 杭州有许多美食，比如西湖醋鱼、东坡肉、龙井虾仁、油焖笋等等。杭州的特色小吃也很有风味，比如桂花糕、叫花鸡、油爆虾等。此外，杭州还有许多美味的甜品，如月饼、麻薯、绿豆糕等。
history: [['浙江的省会在哪？', '浙江省会是杭州市。'], ['这有什么好吃的', '杭州有许多美食，比如西湖醋鱼、东坡肉、龙井虾仁、油焖笋等等。杭州的特色小吃也很有风味，比如桂花糕、叫花鸡、油爆虾等。此外，杭州还有许多美味的甜品，如月饼、麻薯、绿豆糕等。']]
{'num_prompt_tokens': 44, 'num_generated_tokens': 64, 'num_samples': 1, 'runtime': 0.5715192809584551, 'samples/s': 1.7497222461558426, 'tokens/s': 111.98222375397393}
"""
```


### 使用CLI
敬请期待...

## 部署
敬请期待...

## 多模态
敬请期待...
