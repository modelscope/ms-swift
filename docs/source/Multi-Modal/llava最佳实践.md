# Llava 最佳实践
本篇文档涉及的模型如下:

- [llava1_5-7b-instruct](https://modelscope.cn/models/swift/llava-1.5-7b-hf)
- [llava1_5-13b-instruct](https://modelscope.cn/models/swift/llava-1.5-13b-hf)
- [llava1_6-mistral-7b-instruct](https://modelscope.cn/models/swift/llava-v1.6-mistral-7b-hf)
- [llava1_6-vicuna-7b-instruct](https://modelscope.cn/models/swift/llava-v1.6-vicuna-7b-hf)
- [llava1_6-vicuna-13b-instruct](https://modelscope.cn/models/swift/llava-v1.6-vicuna-13b-hf)
- [llava1_6-yi-34b-instruct](https://modelscope.cn/models/swift/llava-v1.6-34b-hf)
- [llava-next-72b](https://modelscope.cn/models/AI-Modelscope/llava-next-72b)
- [llava-next-110b](https://modelscope.cn/models/AI-Modelscope/llava-next-110b)


其中, 前6个llava-hf模型支持vllm推理加速, 具体可以参考[vLLM推理加速文档](vLLM推理加速文档.md). 以下实践以`llava1_6-mistral-7b-instruct`为例，你也可以通过指定`--model_type`切换为其他模型.

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
```shell
# Experimental environment: A100
# 20GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift infer --model_type llava1_6-mistral-7b-instruct

# 70GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift infer --model_type llava1_6-yi-34b-instruct

# 4*20GB GPU memory
CUDA_VISIBLE_DEVICES=0,1,2,3 swift infer --model_type llava1_6-yi-34b-instruct
```

输出: (支持传入本地路径或URL)
```python
"""
<<< who are you
Input a media path or URL <<<
I am a language model, specifically a transformer model, trained to generate text based on the input it receives. I do not have personal experiences or emotions, and I do not have a physical form. I exist purely as a software program that can process and generate text.
--------------------------------------------------
<<< <image>Describe this image.
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png
The image shows a close-up of a kitten with a soft, blurred background that suggests a natural, outdoor setting. The kitten has a mix of white and gray fur with darker stripes, typical of a tabby pattern. Its eyes are wide open, with a striking blue color that contrasts with the kitten's fur. The kitten's nose is small and pink, and its whiskers are long and white, adding to the kitten's cute and innocent appearance. The lighting in the image is soft and diffused, creating a gentle and warm atmosphere. The focus is sharp on the kitten's face, while the rest of the image is slightly out of focus, which draws attention to the kitten's features.
--------------------------------------------------
<<< <image>How many sheep are in the picture?
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png
There are four sheep in the picture.
--------------------------------------------------
<<< clear
<<< <image>What is the calculation result?
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/math.png
The calculation result is 1452 + 453004 = 453006.
--------------------------------------------------
<<< <image>Write a poem based on the content of the picture.
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/poem.png
In the quiet of the night,
A solitary boat takes flight,
Across the water's gentle swell,
Underneath the stars that softly fell.

The boat, a vessel of the night,
Carries but one, a lone delight,
A solitary figure, lost in thought,
In the tranquil calm, they find a wraith.

The stars above, like diamonds bright,
Reflect upon the water's surface light,
Creating a path for the boat's journey,
Guiding through the night with a gentle purity.

The boat, a silent sentinel,
In the stillness, it gently swells,
A vessel of peace and calm,
In the quiet of the night, it carries on.

The figure on board, a soul at ease,
In the serene embrace of nature's peace,
They sail through the night,
Under the watchful eyes of the stars' light.

The boat, a symbol of solitude,
In the vast expanse of the universe's beauty,
A lone journey, a solitary quest,
In the quiet of the night, it finds its rest.
--------------------------------------------------
<<< <image>Perform OCR on the image.
Input a media path or URL <<< https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/ocr_en.png
The text in the image is as follows:

INTRODUCTION

SWIFT supports training, inference, evaluation and deployment of 250+ LLMs (multimodal large models). Developers can directly apply our framework to their own research and production environments to realize the complete workflow from model training and evaluation to application. In addition, SWIFT provides a complete Adapters library to support the latest training techniques such as NLP, Vision, etc. This adapter library can be used directly in your own custom workflow without our training scripts.

To facilitate use by users unfamiliar with deep learning, we provide a Grado web-ui for controlling training and inference, as well as accompanying deep learning courses and best practices for beginners.

SWIFT has rich documentation for users, please check here.

SWIFT is web-ui available both on Huggingface space and ModelScope studio, please feel free to try!
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

ocr_en:

<img src="https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/ocr_en.png" width="250" style="display: inline-block;">

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

model_type = 'llava1_6-mistral-7b-instruct'
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')

model, tokenizer = get_model_tokenizer(model_type, torch.float16,
                                       model_kwargs={'device_map': 'auto'})
model.generation_config.max_new_tokens = 256
template = get_template(template_type, tokenizer)
seed_everything(42)

images = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png']
query = '<image>How far is it from each city?'
response, _ = inference(model, template, query, images=images)
print(f'query: {query}')
print(f'response: {response}')

# 流式
query = 'Which city is the farthest?'
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
response: The image shows a road sign indicating the distances to three cities: Mata, Yangjiang, and Guangzhou. The distances are given in kilometers.

- Mata is 14 kilometers away.
- Yangjiang is 62 kilometers away.
- Guangzhou is 293 kilometers away.

Please note that these distances are as the crow flies and do not take into account the actual driving distance due to road conditions, traffic, or other factors.
query: Which city is the farthest?
response: The farthest city listed on the sign is Mata, which is 14 kilometers away.
"""
```

示例图片如下:

road:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png" width="250" style="display: inline-block;">


## 微调
多模态大模型微调通常使用**自定义数据集**进行微调. 这里展示可直接运行的demo:

LoRA微调:

```shell
# Experimental environment: A10, 3090, V100...
# 21GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type llava1_6-mistral-7b-instruct \
    --dataset coco-en-2-mini \

# Experimental environment: 2*A100...
# 2*45GB GPU memory
CUDA_VISIBLE_DEVICES=0,1 swift sft \
    --model_type llava1_6-yi-34b-instruct \
    --dataset coco-en-2-mini \
```

全参数微调:
```shell
# Experimental environment: 4 * A100
# 4 * 70 GPU memory
NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 swift sft \
    --model_type llava1_6-mistral-7b-instruct \
    --dataset coco-en-2-mini \
    --sft_type full \
    --deepspeed default-zero2

# 8 * 50 GPU memory
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 swift sft \
    --model_type llava1_6-yi-34b-instruct \
    --dataset coco-en-2-mini \
    --sft_type full \
```


[自定义数据集](../Instruction/自定义与拓展.md#-推荐命令行参数的形式)支持json, jsonl样式, 以下是自定义数据集的例子:

(支持多轮对话, 支持每轮对话含多张图片或不含图片, 支持传入本地路径或URL)

```jsonl
{"query": "<image>55555", "response": "66666", "images": ["image_path"]}
{"query": "<image>eeeee", "response": "fffff", "images": ["image_path"]}
{"query": "<image>EEEEE", "response": "FFFFF", "images": ["image_path"]}
```

## 微调后推理
直接推理:
```shell
model_type="llava1_6-mistral-7b-instruct"

CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/${model_type}/vx-xxx/checkpoint-xxx \
    --load_dataset_config true
```

**merge-lora**并推理:
```shell
model_type="llava1_6-mistral-7b-instruct"

CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir "output/${model_type}/vx-xxx/checkpoint-xxx" \
    --merge_lora true
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir "output/${model_type}/vx-xxx/checkpoint-xxx-merged" \
    --load_dataset_config true
```
