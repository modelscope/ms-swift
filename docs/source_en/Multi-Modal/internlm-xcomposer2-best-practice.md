# Internlm-Xcomposer2 Best Practices

## Table of Contents
- [Environment Preparation](#environment-preparation)
- [Inference](#inference)
- [Fine-tuning](#fine-tuning)
- [Inference After Fine-tuning](#inference-after-fine-tuning)

## Environment Preparation
```shell
pip install 'ms-swift[llm]' -U
```

## Inference

Inference for [internlm-xcomposer2-7b-chat](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-xcomposer2-7b/summary):
```shell
# Experimental environment: A10, 3090, V100, ...
# 21GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift infer --model_type internlm-xcomposer2-7b-chat
```

Output: (supports passing local path or URL)
```python
"""
<<< Who are you?
 I am your assistant, a language-based artificial intelligence model that can answer your questions.
--------------------------------------------------
<<< <img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png</img><img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png</img>What's the difference between these two images?
 These two images are different. The first one is a picture of sheep, and the second one is a picture of a cat.
--------------------------------------------------
<<< <img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png</img>How many sheep are there in the picture?
 There are 4 sheep in the picture
--------------------------------------------------
<<< <img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/math.png</img>What is the calculation result?
 The calculation result is 1452+45304=46756
--------------------------------------------------
<<< <img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/poem.png</img>Write a poem based on the content in the picture
 Ripples glisten on the lake's surface, a lone boat drifts.
On the boat, a light illuminates the night,
Speckles of stars reflected in the water.

In the distance, mountains shrouded in mist and clouds,
The starry night sky twinkling endlessly.
The lake is like a mirror, reflections clear,
The little boat passing through, like a poem, like a painting.
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

model_type = ModelType.internlm_xcomposer2_7b_chat
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')

model, tokenizer = get_model_tokenizer(model_type, torch.float16,
                                       model_kwargs={'device_map': 'auto'})
model.generation_config.max_new_tokens = 256
template = get_template(template_type, tokenizer)
seed_everything(42)

query = """<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png</img>How far is it to each city?"""
response, history = inference(model, template, query)
print(f'query: {query}')
print(f'response: {response}')

# Streaming
query = 'Which city is the farthest?'
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
query: <img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png</img>How far is it to each city?
response:  The distance from Ma'anshan to Yangjiang is 62 kilometers, and the distance from Guangzhou to Guangzhou is 293 kilometers.
query: Which city is the farthest?
response: The farthest city is Guangzhou, with a distance of 293 kilometers from Guangzhou.
history: [['<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png</img>How far is it to each city?', ' The distance from Ma'anshan to Yangjiang is 62 kilometers, and the distance from Guangzhou to Guangzhou is 293 kilometers.'], ['Which city is the farthest?', ' The farthest city is Guangzhou, with a distance of 293 kilometers from Guangzhou.']]
"""
```

Sample image is as follows:

road:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png" width="250" style="display: inline-block;">


## Fine-tuning
Fine-tuning of multimodal large models usually uses **custom datasets**. Here's a demo that can be run directly:

(By default, only the qkv part of the LLM is fine-tuned using Lora. `--lora_target_modules ALL` is not supported. Full-parameter fine-tuning is supported.)
```shell
# Experimental environment: A10, 3090, V100, ...
# 21GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type internlm-xcomposer2-7b-chat \
    --dataset coco-en-mini \
```

[Custom datasets](../LLM/Customization.md#-Recommended-Command-line-arguments)  support json and jsonl formats. Here's an example of a custom dataset:

(Supports multi-turn conversations, each turn can contain multiple images or no images, supports passing local paths or URLs. This model does not support merge-lora)

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


## Inference After Fine-tuning
```shell
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/internlm-xcomposer2-7b-chat/vx-xxx/checkpoint-xxx \
    --load_dataset_config true \
```
