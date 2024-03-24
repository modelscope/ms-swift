Here is the English translation of the text:

# Deepseek-VL Best Practices

## Table of Contents
- [Environment Preparation](#environment-preparation) 
- [Inference](#inference)
- [Fine-tuning](#fine-tuning)
- [Inference After Fine-tuning](#inference-after-fine-tuning)

## Environment Preparation
```shell
pip install ms-swift[llm] -U 
```

## Inference

Inference for [deepseek-vl-7b-chat](https://www.modelscope.cn/models/deepseek-ai/deepseek-vl-7b-chat/summary):

```shell
# Experimental environment: A100
# 30GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift infer --model_type deepseek-vl-7b-chat

# If you want to run it on 3090, you can infer the 1.3b model 
CUDA_VISIBLE_DEVICES=0 swift infer --model_type deepseek-vl-1_3b-chat
```

7b model effect demonstration: (supports passing local paths or URLs)
```python 
"""
<<< Describe this kind of picture
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png
This picture captures the charming scene of a little kitten with its eyes wide open, full of curiosity. The kitten's fur is a mix of white and gray, giving it an almost ethereal appearance. Its ears are pointed and alertly pointing upward, while its nose is a soft pink. The kitten's eyes are a striking blue, full of innocence and curiosity. The kitten is comfortably sitting on a piece of white fabric, beautifully contrasting with its gray and white fur. The background is blurred, focusing people's attention on the kitten's face, highlighting the fine details of its features. This picture exudes a sense of warmth and softness, capturing the purity and charm of the kitten.
--------------------------------------------------  
<<< How many sheep are there in the picture?
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png 
There are four sheep in the picture.
--------------------------------------------------
<<< What is the calculation result
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/math.png
The result of adding 1452 and 45304 is 1452 + 45304 = 46756.
--------------------------------------------------
<<< Write a poem based on the content in the picture
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/poem.png
Starlight sprinkled on the tranquil lake surface,
A lone boat gently swaying in the night breeze.
Flickering lights accompany the stars,
Shimmering waves reflect the mountain shadows.

Green bamboo gently brushes the dense night colors,
The Milky Way hangs upside down in the clear sky and water.
Fishing lights dot the dream,
One boat, one person, talking about the stars.
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


**Single sample inference**

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type, inference_stream
)
from swift.utils import seed_everything
import torch

model_type = ModelType.deepseek_vl_7b_chat  
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')

model, tokenizer = get_model_tokenizer(model_type, torch.float16,
                                       model_kwargs={'device_map': 'auto'})
model.generation_config.max_new_tokens = 256
template = get_template(template_type, tokenizer)
seed_everything(42)

images = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png']
query = 'How far from each city?'  
response, history = inference(model, template, query, images=images)
print(f'query: {query}')
print(f'response: {response}')

# Streaming  
query = 'Which city is the farthest?'
images = images * 2
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
query: How far from each city?
response: This sign shows the distances from the current location to the following cities:

- Mata: 14 kilometers
- Yangjiang: 62 kilometers  
- Guangzhou: 293 kilometers

This information is provided based on the sign in the image.  
query: Which city is the farthest?
response: The farthest city is Guangzhou. According to the sign, the distance from the current location to Guangzhou is 293 kilometers.
history: [['How far from each city?', 'This sign shows the distances from the current location to the following cities:\n\n- Mata: 14 kilometers\n- Yangjiang: 62 kilometers\n- Guangzhou: 293 kilometers\n\nThis information is provided based on the sign in the image.'], ['Which city is the farthest?', 'The farthest city is Guangzhou. According to the sign, the distance from the current location to Guangzhou is 293 kilometers.']]
"""
```

Sample image is as follows:

road:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png" width="250" style="display: inline-block;">


## Fine-tuning
Multi-modal large model fine-tuning usually uses **custom datasets**. Here is a runnable demo:

LoRA fine-tuning:

(By default, only lora fine-tuning is performed on the qkv part of the LLM. If you want to fine-tune all linear parts including the vision model, you can specify `--lora_target_modules ALL`)
```shell  
# Experimental environment: A10, 3090, V100
# 20GB GPU memory  
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type deepseek-vl-7b-chat \
    --dataset coco-mini-en-2 \
```

Full parameter fine-tuning:
```shell
# Experimental environment: 4 * A100  
# 4 * 70GB GPU memory
NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 swift sft \
    --model_type deepseek-vl-7b-chat \
    --dataset coco-mini-en-2 \
    --train_dataset_sample -1 \  
    --sft_type full \
    --use_flash_attn true \
    --deepspeed default-zero2
```

[Custom dataset](../LLM/Customization and Extension.md#-Recommended form of command line parameters) supports json, jsonl styles. The following is an example of a custom dataset:

(Supports multi-turn conversations, each turn must include an image, and supports passing local paths or URLs)

```jsonl
{"query": "55555", "response": "66666", "images": ["image_path"]} 
{"query": "eeeee", "response": "fffff", "history": [], "images": ["image_path"]}
{"query": "EEEEE", "response": "FFFFF", "history": [["AAAAA", "BBBBB"], ["CCCCC", "DDDDD"]], "images": ["image_path", "image_path2", "image_path3"]}
```


## Inference After Fine-tuning
Direct inference:
```shell
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/deepseek-vl-7b-chat/vx-xxx/checkpoint-xxx \
    --load_dataset_config true \
```

**merge-lora** and inference:  
```shell
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir output/deepseek-vl-7b-chat/vx-xxx/checkpoint-xxx \
    --merge_lora true

CUDA_VISIBLE_DEVICES=0 swift infer \  
    --ckpt_dir output/deepseek-vl-7b-chat/vx-xxx/checkpoint-xxx-merged \
    --load_dataset_config true
```