
# MiniCPM-V-2 最佳实践

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

推理[minicpm-v-2](https://modelscope.cn/models/OpenBMB/MiniCPM-V-2/summary):
```shell
# Experimental environment: A10, 3090, V100, ...
# 10GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift infer --model_type minicpm-v-v2
```

输出: (支持传入本地路径或URL)
```python
"""
<<< 描述这张图片
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png
 这张图片展示了一只小猫的特写，它的毛色主要是黑白相间，带有一些浅色条纹，可能是灰色或白色。小猫的眼睛是蓝色的，瞳孔呈黑色，与它的毛色形成鲜明对比。它的耳朵竖立着，尖端是白色的，看起来警觉而好奇。小猫的鼻子是黑色的，嘴巴微微张开，露出牙齿，表明它可能在微笑或发出声音。背景模糊不清，但似乎是柔和的绿色，可能是室内环境，比如房间或房间的一部分。小猫的表情和姿势传达出一种顽皮和可爱爱的感觉。
--------------------------------------------------
<<< clear
<<< 图中有几只羊？
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png
 这幅图描绘了一群羊在草地上。总共有四只羊，它们都长着白色的毛和棕色的角。这些羊的大小各不相同，其中一只看起来比其他三只要小一些。它们站在一片郁郁葱葱的草地上，背景是起伏的山脉，天空中飘着几朵云。这幅图像的风格是卡通化的，羊的面部表情和身体特征都夸夸张化了。
--------------------------------------------------
<<< clear
<<< 计算结果是多少
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/math.png
 计算结果是1452 + 4530 =  5982。
--------------------------------------------------
<<< clear
<<< 根据图片中的内容写首诗
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/poem.png
 这幅图片描绘了一个宁静的夜晚场景，一艘小船漂浮在宁静的湖面上。船身呈棕色，看起来像是木质结构，船头有桅杆，顶部有一盏灯，可能是为了导航或照明。船身周围散布着一些小火苗，给画面增添了温暖的光芒。湖面反射着星星和灯光，营造出一种宁静而梦幻的氛围。背景中，树木繁茂，呈现出深绿色，暗示着森林或丛林的环境。天空呈现出渐变的粉色和紫色，暗示着日出或日落。整体氛围宁静而略带带神秘感。
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

model_type = ModelType.minicpm_v_v2
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')

model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16,
                                       model_kwargs={'device_map': 'auto'})
model.generation_config.max_new_tokens = 256
template = get_template(template_type, tokenizer)
seed_everything(42)

images = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png']
query = '距离各城市多远？'
response, history = inference(model, template, query, images=images)
print(f'query: {query}')
print(f'response: {response}')

# 流式
query = '距离最远的城市是哪？'
gen = inference_stream(model, template, query, history, images=images)
print_idx = 0
print(f'query: {query}\nresponse: ', end='')
for response, history in gen:
    delta = response[print_idx:]
    print(delta, end='', flush=True)
    print_idx = len(response)
print()
print(f'history: {history}')
"""
query: 距离最远的城市是哪？
response: 距离最远的城市是广州，距离离为293公里。
history: [['距离各城市多远？', ' 马踏到马塔14公里，到阳江62公里，到广州293公里。'], ['距离最远的城市是哪？', ' 距离最远的城市是广州，距离为293公里。']]
"""
```

示例图片如下:

road:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png" width="250" style="display: inline-block;">


## 微调
多模态大模型微调通常使用**自定义数据集**进行微调. 这里展示可直接运行的demo:

(默认只对LLM部分的qkv进行lora微调. 如果你想对所有linear含vision模型部分都进行微调, 可以指定`--lora_target_modules ALL`. 支持全参数微调.)
```shell
# Experimental environment: A10, 3090, V100, ...
# 10GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type minicpm-v-v2 \
    --dataset coco-mini-en-2 \
```

[自定义数据集](../LLM/自定义与拓展.md#-推荐命令行参数的形式)支持json, jsonl样式, 以下是自定义数据集的例子:

(支持多轮对话, 但总的轮次对话只能包含一张图片, 支持传入本地路径或URL)

```jsonl
{"query": "55555", "response": "66666", "images": ["image_path"]}
{"query": "eeeee", "response": "fffff", "history": [], "images": ["image_path"]}
{"query": "EEEEE", "response": "FFFFF", "history": [["AAAAA", "BBBBB"], ["CCCCC", "DDDDD"]], "images": ["image_path"]}
```


## 微调后推理
直接推理:
```shell
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/minicpm-v-v2/vx-xxx/checkpoint-xxx \
    --load_dataset_config true \
```

**merge-lora**并推理:
```shell
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir output/minicpm-v-v2/vx-xxx/checkpoint-xxx \
    --merge_lora true

CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/minicpm-v-v2/vx-xxx/checkpoint-xxx-merged \
    --load_dataset_config true
```
