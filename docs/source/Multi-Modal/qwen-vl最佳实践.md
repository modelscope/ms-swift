
# Qwen-VL 最佳实践

## 目录
- [环境准备](#环境准备)
- [推理](#推理)
- [微调](#微调)
- [微调后推理](#微调后推理)


## 环境准备
```shell
pip install 'ms-swift[llm]' -U
```

## 推理

推理[qwen-vl-chat](https://modelscope.cn/models/qwen/Qwen-VL-Chat/summary):
```shell
# Experimental environment: 3090
# 24GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift infer --model_type qwen-vl-chat
```

输出: (支持传入本地路径或URL)
```python
"""
<<< 你是谁？
我是通义千问，由阿里云开发的AI助手。我被设计用来回答各种问题、提供信息和与用户进行对话。有什么我可以帮助你的吗？
--------------------------------------------------
<<< <img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png</img><img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png</img>这两张图片有什么区别
这两张图片的主要区别在于内容和主题。
第一张图片是一张卡通插画，画面中是一只公羊或山羊在绿色的草地上，配以群山和白云的背景，整体呈现出自然和动物的主题。
第二张图片也是一张卡通插画，画面中是一只小猫，有条纹的毛发和蓝色的眼睛，整体呈现出可爱和动物的主题。
--------------------------------------------------
<<< <img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png</img>图中有几只羊
图中有一家四口的羊，一共四只。
--------------------------------------------------
<<< <img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/math.png</img>计算结果是多少
1452 + 45304 = 46756
--------------------------------------------------
<<< clear
<<< <img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/poem.png</img>根据图片中的内容写首诗
月光如水洒河中，孤舟一灯独自空。
两岸青山倒影美，星河灿烂天空宏。
--------------------------------------------------
<<< clear
<<< <img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/ocr.png</img>对图片进行OCR
SWIFT支持250+ LLM和35+ MLLM（多模态大模型）的训练、推理、评测和部署。开发者可以直接将我们的框架应用到自己的Research和生产环境中，实现模型训练评测到应用的完整链路。我们除了支持PEPT提供的轻量训练方案外，也提供了一个完整的Adapters库以支持最新的训练技术，如NEFTune、LoRA+、LLaMa-PRO等，这个适配器库可以脱离训练脚本直接使用在自己的自定流程中。
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

model_type = ModelType.qwen_vl_chat
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
response: 马路边距离马路边14公里；阳江边距离马路边62公里；广州边距离马路边293公里。
query: 距离最远的城市是哪？
response: 距离最远的城市是广州，距离马路边293公里。
history: [['<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png</img>距离各城市多远？', '马路边距离马路边14公里；阳江边距离马路边62公里；广州边距离马路边293公里。'], ['距离最远的城市是哪？', '距离最远的城市是广州，距离马路边293公里。']]
"""
```

示例图片如下:

road:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png" width="250" style="display: inline-block;">


## 微调
多模态大模型微调通常使用**自定义数据集**进行微调. 这里展示可直接运行的demo:

LoRA微调:

```shell
# Experimental environment: 3090
# 23GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type qwen-vl-chat \
    --dataset coco-en-mini \
```

全参数微调:
```shell
# Experimental environment: 4 * A100
# 4 * 70 GPU memory
NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES=0,1,2,3 swift sft \
    --model_type qwen-vl-chat \
    --dataset coco-en-mini \
    --sft_type full \
```

**Qwen-VL**模型支持grounding任务的训练，数据参考下面的格式：
```jsonl
{"query": "Find <bbox>", "response": "<ref-object>", "images": ["/coco2014/train2014/COCO_train2014_000000001507.jpg"], "objects": "[{\"caption\": \"guy in red\", \"bbox\": [138, 136, 235, 359], \"bbox_type\": \"real\", \"image\": 0}]" }
# mapping to multiple bboxes
{"query": "Find <ref-object>", "response": "<bbox>", "images": ["/coco2014/train2014/COCO_train2014_000000001507.jpg"], "objects": "[{\"caption\": \"guy in red\", \"bbox\": [[138, 136, 235, 359],[1,2,3,4]], \"bbox_type\": \"real\", \"image\": 0}]" }
# 或者使用<img></img>标签
{"query": "<img>/coco2014/train2014/COCO_train2014_000000001507.jpg</img>Find <bbox>", "response": "<ref-object>", "objects": "[{\"caption\": \"guy in red\", \"bbox\": [138, 136, 235, 359], \"bbox_type\": \"real\", \"image\": 0}]" }
{"query": "<img>/coco2014/train2014/COCO_train2014_000000001507.jpg</img>Find <ref-object>", "response": "<bbox>", "objects": "[{\"caption\": \"guy in red\", \"bbox\": [138, 136, 235, 359], \"bbox_type\": \"real\", \"image\": 0}]" }
```
上述objects字段中包含了一个json string，其中有四个字段：
    - caption bbox对应的物体描述
    - bbox 坐标 建议给四个整数（而非float型），分别是x_min,y_min,x_max,y_max四个值
    - bbox_type: bbox类型 目前支持三种：real/norm_1000/norm_1，分别代表实际像素值坐标/千分位比例坐标/归一化比例坐标
    - image: bbox对应的图片是第几张, 索引从0开始
上述格式会被转换为Qwen-VL可识别的格式，具体来说：
```jsonl
{"query": "<img>/coco2014/train2014/COCO_train2014_000000001507.jpg</img>Find <ref>the man</ref>", "response": "<box>(200,200),(600,600)</box>"}
```
也可以直接传入上述格式，但是注意坐标请使用千分位坐标。

[自定义数据集](../Instruction/自定义与拓展.md#-推荐命令行参数的形式)支持json, jsonl样式, 以下是自定义数据集的例子:

(支持多轮对话, 支持每轮对话含多张图片或不含图片, 支持传入本地路径或URL)

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
直接推理:
```shell
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/qwen-vl-chat/vx-xxx/checkpoint-xxx \
    --load_dataset_config true \
```

**merge-lora**并推理:
```shell
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir output/qwen-vl-chat/vx-xxx/checkpoint-xxx \
    --merge_lora true

CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/qwen-vl-chat/vx-xxx/checkpoint-xxx-merged \
    --load_dataset_config true
```
