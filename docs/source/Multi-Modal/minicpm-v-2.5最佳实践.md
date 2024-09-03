
# MiniCPM-V-2.5 最佳实践

MiniCPM-V-2.6 最佳实践: [https://github.com/modelscope/ms-swift/issues/1613](https://github.com/modelscope/ms-swift/issues/1613)

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
模型链接:
- minicpm-v-v2_5-chat: [https://modelscope.cn/models/OpenBMB/MiniCPM-Llama3-V-2_5/summary](https://modelscope.cn/models/OpenBMB/MiniCPM-Llama3-V-2_5/summary)


## 推理

推理 minicpm-v-v2_5-chat:
```shell
# Experimental environment: A10, 3090, V100, ...
# 20GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift infer --model_type minicpm-v-v2_5-chat
```

输出: (支持传入本地路径或URL)
```python
"""
<<< 描述这张图片
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png
这张图片展示了一只年轻的猫咪的特写，可能是一只小猫，具有明显的特征。它的毛发主要是白色的，带有灰色和黑色的条纹和斑点，这是虎斑猫的典型特征。小猫的眼睛是蓝色的，瞳孔是圆形的，给人一种好奇和专注的表情。它的耳朵尖尖的，竖立着，显示出警觉性。小猫的鼻子是粉红色的，鼻孔是可见的。背景模糊不清，突出了小猫的特征。整体的色调柔和，重点放在小猫的毛发和眼睛上。
--------------------------------------------------
<<< clear
<<< 图中有几只羊？
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png
图中有四只羊。
--------------------------------------------------
<<< clear
<<< 计算结果是多少
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/math.png
计算结果是1452 + 4530 = 5982。
--------------------------------------------------
<<< clear
<<< 根据图片中的内容写首诗
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/poem.png
在宁静的夜晚，船只航行，
在星光闪烁的水面上，
一只熊猫乘风破浪，
在夜空的映衬下。
船上灯火通明，照亮了前方的道路，
在宁静的水面上投下温暖的光芒，
熊猫坐在船头，享受着旅程，
在这宁静的夜晚中，享受着旅程。
星星在上方闪烁，点缀着天空，
在这宁静的夜晚中，创造出一幅美丽的画面，
船只在水面上轻轻摇晃，
在这宁静的夜晚中，创造出一幅美丽的画面。
--------------------------------------------------
<<< clear
<<< 对图片进行OCR
Input a media path or URL <<< https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/ocr.png
标题: SWIFT支持多模态大模型的训练、推理、评测和部署

正文:
开发者可以直接将我们的框架应用到自己的Research和生产环境中，实现模型训练评测到应用的完整链路。我们除支持了PEFT提供的轻量训练方案外，也提供了一个完整的Adapters库以支持最新的训练技术，如NEFTune、LoRA+、LLaMA-PRO等，这个适配器库可以脱离训练脚本直接使用在自己的自定流程中。

为方便不熟悉深度学习的用户使用，我们提供了一个Gradio的web-ui用于控制训练和推理，并提供了配套的深度学习课程和最佳实践供新手入门。此外，我们也在拓展其他模态的能力，目前我们支持了AnimateDiff的全参数训练和LoRA训练。

SWIFT具有丰富的文档体系，如有使用问题请请查看这里。

可以在Huggingface space和ModelScope创空间中体验SWIFT web-ui功能了。
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

model_type = ModelType.minicpm_v_v2_5_chat
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
response: 马踏到阳江的距离是62公里，阳江到广州的距离是293公里。
query: 距离最远的城市是哪？
response: 距离最远的城市是广州，到广州的距离为293公里。
history: [['距离各城市多远？', '马踏到阳江的距离是62公里，阳江到广州的距离是293公里。'], ['距离最远的城市是哪？', '距离最远的城市是广州，到广州的距离为293公里。']]
"""
```

示例图片如下:

road:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png" width="250" style="display: inline-block;">


## 微调
多模态大模型微调通常使用**自定义数据集**进行微调. 这里展示可直接运行的demo:

```shell
# Experimental environment: 3090
# 20GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type minicpm-v-v2_5-chat \
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
    --ckpt_dir output/minicpm-v-v2_5-chat/vx-xxx/checkpoint-xxx \
    --load_dataset_config true \
```

**merge-lora**并推理:
```shell
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir output/minicpm-v-v2_5-chat/vx-xxx/checkpoint-xxx \
    --merge_lora true

CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/minicpm-v-v2_5-chat/vx-xxx/checkpoint-xxx-merged \
    --load_dataset_config true
```
