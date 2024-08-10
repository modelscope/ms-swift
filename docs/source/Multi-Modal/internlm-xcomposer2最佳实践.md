
# Internlm-Xcomposer2 & Internlm-Xcomposer2.5 最佳实践

本篇文档涉及的模型如下:

- [internlm-xcomposer2-7b-chat](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-xcomposer2-7b/summary)
- [internlm-xcomposer2_5-7b-chat](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-xcomposer2d5-7b/summary)

以下实践以`internlm-xcomposer2-7b-chat`为例，你也可以通过指定`--model_type`切换为其他模型.

## 目录
- [环境准备](#环境准备)
- [推理](#推理)
- [微调](#微调)
- [微调后推理](#微调后推理)


## 环境准备
```shell
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'
```

## 推理

推理internlm-xcomposer2-7b-chat:
```shell
# Experimental environment: A10, 3090, V100, ...
# 21GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift infer --model_type internlm-xcomposer2-7b-chat
```

输出: (支持传入本地路径或URL)
```python
"""
<<< 你是谁？
我是浦语·灵笔，一个由上海人工智能实验室开发的语言模型。我能理解并流畅地使用英语和中文与你对话。
--------------------------------------------------
<<< <img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png</img><img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png</img>这两张图片有什么区别
这两张图片没有直接的关联，它们分别展示了两个不同的场景。第一幅图是一张卡通画，描绘了一群羊在草地上，背景是蓝天和山脉。第二幅图则是一张猫的照片，猫正看着镜头，背景模糊不清。
--------------------------------------------------
<<< clear
<<< <img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png</img>图中有几只羊
图中有4只羊
--------------------------------------------------
<<< <img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/math.png</img>计算结果是多少
1452 + 45304 = 46756
--------------------------------------------------
<<< <img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/poem.png</img>根据图片中的内容写首诗
夜色苍茫月影斜，
湖面平静如明镜。
小舟轻荡波光里，
灯火微摇映水乡。
"""
```

示例图片如下:

cat:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png" width="250" style="display: inline-block;">

animal:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png" width="250" style="display: inline-block;">

math:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/math.png" width="250" style="display: inline-block;">

poem:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/poem.png" width="250" style="display: inline-block;">

ocr:

<img src="https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/ocr.png" width="250" style="display: inline-block;">

**单样本推理**

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type, inference_stream
)
from swift.utils import seed_everything
import torch

model_type = ModelType.internlm_xcomposer2_7b_chat
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')

model, tokenizer = get_model_tokenizer(model_type, torch.float16,
                                       model_kwargs={'device_map': 'auto'})
model.generation_config.max_new_tokens = 256
template = get_template(template_type, tokenizer)
seed_everything(42)

query = """<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png</img>距离各城市多远？"""
response, history = inference(model, template, query)
print(f'query: {query}')
print(f'response: {response}')

# 流式
query = '距离最远的城市是哪？'
gen = inference_stream(model, template, query, history)
print_idx = 0
print(f'query: {query}\nresponse: ', end='')
for response, history in gen:
    delta = response[print_idx:]
    print(delta, end='', flush=True)
    print_idx = len(response)
print()
print(f'history: {history}')
"""
query: <img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png</img>距离各城市多远？
response: 马鞍山距离阳江62公里，广州距离广州293公里。
query: 距离最远的城市是哪？
response: 距离最远的城市是广州，距离广州293公里。
history: [['<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png</img>距离各城市多远？', '马鞍山距离阳江62公里，广州距离广州293公里。'], ['距离最远的城市是哪？', '距离最远的城市是广州，距离广州293公里。']]
"""
```

示例图片如下:

road:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png" width="250" style="display: inline-block;">


## 微调
多模态大模型微调通常使用**自定义数据集**进行微调. 这里展示可直接运行的demo:

```shell
# Experimental environment: A10, 3090, V100, ...
# 21GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type internlm-xcomposer2-7b-chat \
    --dataset coco-en-mini \
```

[自定义数据集](../LLM/自定义与拓展.md#-推荐命令行参数的形式)支持json, jsonl样式, 以下是自定义数据集的例子:

(支持多轮对话, 支持每轮对话含多张图片或不含图片, 支持传入本地路径或URL. 该模型不支持merge-lora)

```json
[
    {"conversations": [
        {"from": "user", "value": "<img>img_path</img>11111"},
        {"from": "assistant", "value": "22222"}
    ]},
    {"conversations": [
        {"from": "user", "value": "<img>img_path</img><img>img_path2</img><img>img_path3</img>aaaaa"},
        {"from": "assistant", "value": "bbbbb"},
        {"from": "user", "value": "<img>img_path</img>ccccc"},
        {"from": "assistant", "value": "ddddd"}
    ]},
    {"conversations": [
        {"from": "user", "value": "AAAAA"},
        {"from": "assistant", "value": "BBBBB"},
        {"from": "user", "value": "CCCCC"},
        {"from": "assistant", "value": "DDDDD"}
    ]}
]
```


## 微调后推理
```shell
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/internlm-xcomposer2-7b-chat/vx-xxx/checkpoint-xxx \
    --load_dataset_config true \
```
