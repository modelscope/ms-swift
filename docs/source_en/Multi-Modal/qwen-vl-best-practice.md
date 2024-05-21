# Qwen-VL Best Practices

## Table of Contents
- [Environment Setup](#environment-setup)
- [Inference](#inference)
- [Fine-tuning](#fine-tuning)
- [Inference after Fine-tuning](#inference-after-fine-tuning)

## Environment Setup
```shell
pip install 'ms-swift[llm]' -U
```

## Inference

Infer using [qwen-vl-chat](https://modelscope.cn/models/qwen/Qwen-VL-Chat/summary):
```shell
# Experimental environment: 3090
# 24GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift infer --model_type qwen-vl-chat
```

Output: (supports passing in local paths or URLs)
```python
"""
<<< multi-line
[INFO:swift] End multi-line input with `#`.
[INFO:swift] Input `single-line` to switch to single-line input mode.
<<<[M] Who are you?#
I am Tongyi Qianwen, an AI assistant developed by Alibaba Cloud. I am designed to answer various questions, provide information and converse with users. Is there anything I can help you with?
--------------------------------------------------
<<<[M] Picture 1:<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png</img>
Picture 2:<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png</img>
What are the differences between these two pictures#
The two pictures are similar in that they are both illustrations about animals, but the animals are different.
The first picture shows sheep, while the second picture shows a kitten.
--------------------------------------------------
<<<[M] Picture 1:<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png</img>
How many sheep are in the picture#
There are four sheep in the picture.
--------------------------------------------------
<<<[M] Picture 1:<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/math.png</img>
What is the calculation result#
1452 + 45304 = 46756
--------------------------------------------------
<<<[M] Picture 1:<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/poem.png</img>
Write a poem based on the content in the picture#
Starlight sparkling on the lake surface, a lone boat's shadow still as if asleep.
The man holds up a lantern to illuminate the valley, with a kitten accompanying by his side.
"""
```

Sample images are as follows:

cat:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png" width="250" style="display: inline-block;">

animal:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png" width="250" style="display: inline-block;">

math:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/math.png" width="250" style="display: inline-block;">

poem:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/poem.png" width="250" style="display: inline-block;">

**Single Sample Inference**

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

query = """Picture 1:<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png</img>
How far is it to each city?"""
response, history = inference(model, template, query)
print(f'query: {query}')
print(f'response: {response}')

# Streaming
query = 'Which city is the farthest away?'
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
query: Picture 1:<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png</img>
How far is it to each city?
response: Malu边 is 14 km away from Malu; Yangjiang边 is 62 km away from Malu; Guangzhou边 is 293 km away from Malu.
query: Which city is the farthest away?
response: The farthest city is Guangzhou, 293 km away from Malu.
history: [['Picture 1:<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png</img>\nHow far is it to each city?', 'Malu边 is 14 km away from Malu; Yangjiang边 is 62 km away from Malu; Guangzhou边 is 293 km away from Malu.'], ['Which city is the farthest away?', 'The farthest city is Guangzhou, 293 km away from Malu.']]
"""
```

Sample image is as follows:

road:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png" width="250" style="display: inline-block;">


## Fine-tuning
Multimodal large model fine-tuning usually uses **custom datasets**. Here is a demo that can be run directly:

LoRA fine-tuning:

(By default, only the qkv part of the LLM is lora fine-tuned. If you want to fine-tune all linear modules including the vision model, you can specify `--lora_target_modules ALL`)
```shell
# Experimental environment: 3090
# 23GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type qwen-vl-chat \
    --dataset coco-en-mini \
```

Full parameter fine-tuning:
```shell
# Experimental environment: 2 * A100
# 2 * 55 GPU memory
CUDA_VISIBLE_DEVICES=0,1 swift sft \
    --model_type qwen-vl-chat \
    --dataset coco-en-mini \
    --sft_type full \
```

[Custom datasets](../LLM/Customization.md#-Recommended-Command-line-arguments)  support json and jsonl formats. Here is an example of a custom dataset:

(Supports multi-turn dialogues, where each turn can contain multiple images or no images, and supports passing in local paths or URLs)

```json
[
    {"conversations": [
        {"from": "user", "value": "Picture 1:<img>img_path</img>\n11111"},
        {"from": "assistant", "value": "22222"}
    ]},
    {"conversations": [
        {"from": "user", "value": "Picture 1:<img>img_path</img>\nPicture 2:<img>img_path2</img>\nPicture 3:<img>img_path3</img>\naaaaa"},
        {"from": "assistant", "value": "bbbbb"},
        {"from": "user", "value": "Picture 1:<img>img_path</img>\nccccc"},
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


## Inference after Fine-tuning
Direct inference:
```shell
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/qwen-vl-chat/vx-xxx/checkpoint-xxx \
    --load_dataset_config true \
```

**merge-lora** and infer:
```shell
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir output/qwen-vl-chat/vx-xxx/checkpoint-xxx \
    --merge_lora true

CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/qwen-vl-chat/vx-xxx/checkpoint-xxx-merged \
    --load_dataset_config true
```
