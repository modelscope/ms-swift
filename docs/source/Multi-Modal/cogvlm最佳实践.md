
# CogVLM 最佳实践

## 目录
- [环境准备](#环境准备)
- [推理](#推理)
- [微调](#微调)
- [微调后推理](#微调后推理)


## 环境准备
```shell
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e .[llm]
```

## 推理

推理[cogvlm-17b-instruct](https://modelscope.cn/models/ZhipuAI/cogvlm-chat/summary):
```shell
# Experimental environment: A100
# 38GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift infer --model_type cogvlm-17b-instruct
```

输出: (支持传入本地路径或URL)
```python
"""
<<< Describe this image.
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png
This image showcases a close-up of a young kitten. The kitten has a mix of white and gray fur, with striking blue eyes. The fur appears soft and fluffy, and the kitten seems to be in a relaxed position, possibly resting or lounging.
--------------------------------------------------
<<< How many sheep are in the picture?
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png
There are four sheep in the picture.
--------------------------------------------------
<<< What is the calculation result?
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/math.png
The calculation result is '1452+45304=146544'.
--------------------------------------------------
<<< Write a poem based on the content of the picture.
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/poem.png
In a realm where night and day intertwine,
A boat floats gently, on water so fine.
Glowing orbs dance, in the starry sky,
While the forest whispers, secrets it holds.
A journey of wonder, in the embrace of the night,
Where dreams take flight, and spirits ignite.
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

model_type = ModelType.cogvlm_17b_instruct
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')

model, tokenizer = get_model_tokenizer(model_type, torch.float16,
                                       model_kwargs={'device_map': 'auto'})
model.generation_config.max_new_tokens = 256
template = get_template(template_type, tokenizer)
seed_everything(42)

images = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png']
query = 'How far is it from each city?'
response, _ = inference(model, template, query, images=images)
print(f'query: {query}')
print(f'response: {response}')

# 流式
query = 'Which city is the farthest?'
images = images
gen = inference_stream(model, template, query, images=images)
print_idx = 0
print(f'query: {query}\nresponse: ', end='')
for response, _ in gen:
    delta = response[print_idx:]
    print(delta, end='', flush=True)
    print_idx = len(response)
print()
"""
query: How far is it from each city?
response: From Mata, it is 14 km; from Yangjiang, it is 62 km; and from Guangzhou, it is 293 km.
query: Which city is the farthest?
response: Guangzhou is the farthest city with a distance of 293 km.
"""
```

示例图片如下:

road:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png" width="250" style="display: inline-block;">


## 微调
多模态大模型微调通常使用**自定义数据集**进行微调. 这里展示可直接运行的demo:

(默认对语言和视觉模型的qkv进行lora微调. 如果你想对所有linear都进行微调, 可以指定`--lora_target_modules ALL`)
```shell
# Experimental environment: A100
# 50GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type cogvlm-17b-instruct \
    --dataset coco-mini-en-2 \
```

[自定义数据集](../LLM/自定义与拓展.md#-推荐命令行参数的形式)支持json, jsonl样式, 以下是自定义数据集的例子:

(只支持单轮对话, 且必须包含一张图片, 支持传入本地路径或URL)

```jsonl
{"query": "55555", "response": "66666", "images": ["image_path"]}
{"query": "eeeee", "response": "fffff", "images": ["image_path"]}
{"query": "EEEEE", "response": "FFFFF", "images": ["image_path"]}
```


## 微调后推理
直接推理:
```shell
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/cogvlm-17b-instruct/vx-xxx/checkpoint-xxx \
    --load_dataset_config true \
```

**merge-lora**并推理:
```shell
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir output/cogvlm-17b-instruct/vx-xxx/checkpoint-xxx \
    --merge_lora true

CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/cogvlm-17b-instruct/vx-xxx/checkpoint-xxx-merged \
    --load_dataset_config true
```
