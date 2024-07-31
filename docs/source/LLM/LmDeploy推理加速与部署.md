# LmDeploy推理加速与部署
lmdeploy github: [https://github.com/InternLM/lmdeploy](https://github.com/InternLM/lmdeploy).

支持lmdeploy推理加速的模型可以查看[支持的模型](支持的模型和数据集.md#模型).

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

if __name__ == '__main__':
    model_type = ModelType.qwen2_7b_instruct
    lmdeploy_engine = get_lmdeploy_engine(model_type, tp=2)
    template_type = get_default_template_type(model_type)
    template = get_template(template_type, lmdeploy_engine.hf_tokenizer)
    # 与`transformers.GenerationConfig`类似的接口
    lmdeploy_engine.generation_config.max_new_tokens = 1024
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
response: 你好！有什么我可以帮助你的吗？
query: 浙江的省会在哪？
response: 浙江省的省会是杭州市。
{'num_prompt_tokens': 46, 'num_generated_tokens': 15, 'num_samples': 2, 'runtime': 0.18026001192629337, 'samples/s': 11.095084143330586, 'tokens/s': 83.2131310749794}
query: 这有什么好吃的
response: 浙江省，简称“浙”，位于中国东南沿海长江三角洲地区，是一个美食资源丰富的地区。这里不仅有传统的江南菜系，还融合了海洋文化的特色，形成了独特的饮食文化。以下是一些浙江的著名美食：

1. **西湖醋鱼**：一道源自杭州的传统名菜，选用鲜活的草鱼，肉质细嫩，酸甜适口，是来杭州必尝的佳肴。

2. **东坡肉**：也是源于杭州的一道经典菜肴，以五花肉为主料，经过长时间的慢炖，肉质酥软，味道浓郁。

3. **龙井虾仁**：以杭州龙井茶为原料，搭配新鲜的虾仁，色香味俱佳，是将茶文化和美食完美结合的佳作。

4. **宁波汤圆**：宁波的汤圆以皮薄馅多、甜而不腻著称，有芝麻、豆沙等多种口味，是宁波地区的传统小吃。

5. **海鲜大餐**：浙江沿海城市如宁波、舟山等地，海鲜种类丰富，可以品尝到各种新鲜的海产，如东海三鲜（黄鱼、带鱼、小黄鱼）、虾蟹等。

6. **绍兴酒**：绍兴不仅是著名的黄酒产地，还有其他多种酒类，如女儿红、加饭酒等，口感醇厚，是佐餐或品饮的好选择。

7. **衢州烤饼**：在衢州地区非常有名的小吃，外皮酥脆，内里松软，通常会夹上肉末、葱花等配料。

8. **台州海鲜面**：台州的海鲜面以其丰富的海鲜和独特的调味方式闻名，面条滑爽，海鲜鲜美。

这些只是浙江美食中的一部分，每个地方都有其独特的风味和特色小吃，值得一一尝试。
history: [['浙江的省会在哪？', '浙江省的省会是杭州市。'], ['这有什么好吃的', '浙江省，简称“浙”，位于中国东南沿海长江三角洲地区，是一个美食资源丰富的地区。这里不仅有传统的江南菜系，还融合了海洋文化的特色，形成了独特的饮食文化。以下是一些浙江的著名美食：\n\n1. **西湖醋鱼**：一道源自杭州的传统名菜，选用鲜活的草鱼，肉质细嫩，酸甜适口，是来杭州必尝的佳肴。\n\n2. **东坡肉**：也是源于杭州的一道经典菜肴，以五花肉为主料，经过长时间的慢炖，肉质酥软，味道浓郁。\n\n3. **龙井虾仁**：以杭州龙井茶为原料，搭配新鲜的虾仁，色香味俱佳，是将茶文化和美食完美结合的佳作。\n\n4. **宁波汤圆**：宁波的汤圆以皮薄馅多、甜而不腻著称，有芝麻、豆沙等多种口味，是宁波地区的传统小吃。\n\n5. **海鲜大餐**：浙江沿海城市如宁波、舟山等地，海鲜种类丰富，可以品尝到各种新鲜的海产，如东海三鲜（黄鱼、带鱼、小黄鱼）、虾蟹等。\n\n6. **绍兴酒**：绍兴不仅是著名的黄酒产地，还有其他多种酒类，如女儿红、加饭酒等，口感醇厚，是佐餐或品饮的好选择。\n\n7. **衢州烤饼**：在衢州地区非常有名的小吃，外皮酥脆，内里松软，通常会夹上肉末、葱花等配料。\n\n8. **台州海鲜面**：台州的海鲜面以其丰富的海鲜和独特的调味方式闻名，面条滑爽，海鲜鲜美。\n\n这些只是浙江美食中的一部分，每个地方都有其独特的风味和特色小吃，值得一一尝试。']]
{'num_prompt_tokens': 46, 'num_generated_tokens': 384, 'num_samples': 1, 'runtime': 2.7036479230737314, 'samples/s': 0.36987064457087926, 'tokens/s': 142.03032751521764}
"""
```

### 使用CLI
```bash
CUDA_VISIBLE_DEVICES=0 swift infer --model_type qwen2-7b-instruct --infer_backend lmdeploy
# TP
CUDA_VISIBLE_DEVICES=0,1 swift infer --model_type qwen2-7b-instruct --infer_backend lmdeploy --tp 2

CUDA_VISIBLE_DEVICES=0,1 swift infer --model_type qwen2-72b-instruct --infer_backend lmdeploy --tp 2
```

## 部署
```bash
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen2-7b-instruct --infer_backend lmdeploy
# TP
CUDA_VISIBLE_DEVICES=0,1 swift deploy --model_type qwen2-7b-instruct --infer_backend lmdeploy --tp 2

CUDA_VISIBLE_DEVICES=0,1 swift deploy --model_type qwen2-72b-instruct --infer_backend lmdeploy --tp 2
```

客户端调用方式可以查看: [vLLM推理加速与部署文档](VLLM推理加速与部署.md#部署)

## 多模态
查看[这里](../Multi-Modal/LmDeploy推理加速文档.md)
