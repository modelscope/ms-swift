
# MiniCPM-V-2 最佳实践

## 目录
- [环境准备](#环境准备)
- [推理](#推理)
- [微调](#微调)
- [微调后推理](#微调后推理)


## 环境准备
```shell
# 请使用"ms-swift>=2.2"或者main分支.
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'
```

## 推理

推理[minicpm-v-v2-chat](https://modelscope.cn/models/OpenBMB/MiniCPM-V-2/summary):
```shell
# Experimental environment: A10, 3090, V100, ...
# 10GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift infer --model_type minicpm-v-v2-chat
```

输出: (支持传入本地路径或URL)
```python
"""
<<< 描述这张图片
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png
这张图片展示了一只小猫的特写，它的毛色主要是黑白相间，带有一些浅色条纹，可能暗示着虎斑猫品种。小猫的眼睛是蓝色的，瞳孔看起来是黑色的，给人一种深邃和好奇的感觉。它的耳朵竖立着，尖端是白色的，与毛色相匹配。小猫的鼻子是黑色的，嘴巴微微张开，露出牙齿，表明它可能在微笑或嬉戏。背景模糊，但似乎是室内环境，可能是地板或墙壁，颜色柔和，与小猫的毛色相融合。图片中的风格化效果使小猫看起来像一幅绘画或插图，而不是一张真实的照片。
--------------------------------------------------
<<< clear
<<< 图中有几只羊？
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png
这幅图片描绘了一群羊在草地上。总共有四只羊，它们都长着白色的毛和棕色的角。这些羊看起来大小不一，其中一只看起来比另外三只要小一些。它们站在一片郁郁葱葱的绿草中，背景是起伏的山丘和天空。这幅图片的风格是卡通化的，羊的面部特征和身体特征都非常夸张。
--------------------------------------------------
<<< clear
<<< 计算结果是多少
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/math.png
计算结果是1452 + 45304 = 46756。
--------------------------------------------------
<<< clear
<<< 根据图片中的内容写首诗
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/poem.png
这幅图片描绘了一个宁静的夜晚场景，一艘船漂浮在水面之上。船看起来是一艘小木船，船头有一个桅杆，上面挂着一个灯笼，发出温暖的光芒。船身涂成深棕色，与水面形成鲜明对比。水面反射着星星和船只的灯光，营造出一种宁静而梦幻的氛围。背景中，树木繁茂，树叶呈现出金色和绿色，暗示着可能是黄昏或黎明时分。天空布满星星，给整个场景增添了神秘感。整体氛围宁静而幽静，让人联想到一个童话般的场景。
--------------------------------------------------
<<< clear
<<< 对图片进行OCR
Input a media path or URL <<< https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/ocr.png
图片中的文字内容为：“SWIFT支持250+LLM和35+LLM（多模态大模型）的训练、推理和评估部署。开发者可以直接将我们的Research和生产环境中应用的模型框架应用到自己的Research和生产环境中。我们提供了完整的Adapters库以支持最新的训练技术，如NEFtune、Lora、LMA-PRO等，这个适配器库可以脱壳脚本直接在自己的流程中使用。为方便不熟悉深度学习用户的使用，我们提供了配套的深度学习课程和最佳实践新手入门门。此外，我们还在拓展其他强大的能力，目前我们支持了AnimateDiff的全参数LORA训练。SWIFT有丰富的文档体系，如有使用问题请查看这里。”
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

model_type = ModelType.minicpm_v_v2_chat
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
query: 距离各城市多远？
response:  马踏到马塔14公里，到阳江62公里，到广州293公里。
query: 距离最远的城市是哪？
response: 距离最远的城市是广州，距离为293公里。
history: [['距离各城市多远？', ' 马踏到马塔14公里，到阳江62公里，到广州293公里。'], ['距离最远的城市是哪？', '距离最远的城市是广州，距离为293公里。']]
"""
```

示例图片如下:

road:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png" width="250" style="display: inline-block;">


## 微调
多模态大模型微调通常使用**自定义数据集**进行微调. 这里展示可直接运行的demo:

```shell
# Experimental environment: A10, 3090, V100, ...
# 10GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type minicpm-v-v2-chat \
    --dataset coco-en-2-mini \
```

[自定义数据集](../Instruction/自定义与拓展.md#-推荐命令行参数的形式)支持json, jsonl样式, 以下是自定义数据集的例子:

(支持多轮对话, 但总的轮次对话只能包含一张图片, 支持传入本地路径或URL)

```jsonl
{"query": "55555", "response": "66666", "images": ["image_path"]}
{"query": "eeeee", "response": "fffff", "history": [], "images": ["image_path"]}
{"query": "EEEEE", "response": "FFFFF", "history": [["query1", "response1"], ["query2", "response2"]], "images": ["image_path"]}
```


## 微调后推理
直接推理:
```shell
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/minicpm-v-v2-chat/vx-xxx/checkpoint-xxx \
    --load_dataset_config true \
```

**merge-lora**并推理:
```shell
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir output/minicpm-v-v2-chat/vx-xxx/checkpoint-xxx \
    --merge_lora true

CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/minicpm-v-v2-chat/vx-xxx/checkpoint-xxx-merged \
    --load_dataset_config true
```
