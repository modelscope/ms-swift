以下是对上述内容的翻译:

# CogVLM 最佳实践

## 目录
- [环境准备](#环境准备) 
- [推理](#推理)
- [微调](#微调)
- [微调后推理](#微调后推理)

## 环境准备
```shell
pip install ms-swift[llm] -U
```

## 推理

推理 [cogvlm-17b-instruct](https://modelscope.cn/models/ZhipuAI/cogvlm-chat/summary):
```shell 
# 实验环境:A100
# 38GB GPU内存
CUDA_VISIBLE_DEVICES=0 swift infer --model_type cogvlm-17b-instruct
```

输出:(支持传入本地路径或URL)
```python
"""
<<< 描述这张图片。
输入媒体路径或URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png
这张图片展示了一只小猫的特写。这只小猫有白色和灰色的毛发,有着引人注目的蓝色眼睛。毛发看起来柔软蓬松,小猫似乎处于放松的姿态,可能在休息或懒洋洋地躺着。
--------------------------------------------------  
<<< 图片中有多少只羊?
输入媒体路径或URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png
图片中有四只羊。
--------------------------------------------------
<<< 计算结果是什么?
输入媒体路径或URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/math.png
计算结果是'1452+45304=146544'。  
--------------------------------------------------
<<< 根据图片内容写一首诗。
输入媒体路径或URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/poem.png 
在昼夜交织的境界,
轻舟漂荡,水面如镜。 
光球在星空中舞动,
森林低语,藏着秘密。
夜的怀抱里,一段奇妙的旅程, 
梦想腾飞,灵魂激荡。
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
query = '每个城市的距离分别是多少?'
response, _ = inference(model, template, query, images=images)
print(f'query: {query}')  
print(f'response: {response}')

# 流式
query = '哪个城市距离最远?'  
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
query: 每个城市的距离分别是多少?
response: 从Mata开始,距离14公里;从阳江开始,距离62公里;从广州开始,距离293公里。 
query: 哪个城市距离最远?
response: 广州距离最远,为293公里。
"""
```

示例图片如下:

road:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png" width="250" style="display: inline-block;">


## 微调
多模态大模型微调通常使用**自定义数据集**进行微调。这里展示可直接运行的demo:

(默认对语言和视觉模型的qkv进行lora微调。如果你想对所有linear都进行微调,可以指定`--lora_target_modules ALL`)  
```shell
# 实验环境:A100 
# 50GB GPU内存
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type cogvlm-17b-instruct \
    --dataset coco-mini-en-2 \  
```

[自定义数据集](../LLM/自定义与拓展.md#-推荐命令行参数的形式)支持json, jsonl样式,以下是自定义数据集的例子:

(只支持单轮对话,且必须包含一张图片,支持传入本地路径或URL)

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