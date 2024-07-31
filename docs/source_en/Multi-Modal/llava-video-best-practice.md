# Llava Video Best Practice

## Table of Contents
- [Environment Setup](#environment-setup)
- [Inference](#inference)
- [Fine-tuning](#fine-tuning)
- [Inference after Fine-tuning](#inference-after-fine-tuning)

## Environment Setup
```shell
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'
```

## Inference
```shell
# Experimental environment: A10
# 20GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift infer --model_type llava-next-video-7b-instruct
```

Output: (supports passing in local path or URL)
```python
"""
<<< who are you
I am Vicuna, a language model trained by researchers from Large Model Systems Organization (LMSYS).
--------------------------------------------------
<<< clear
<<< <video>Describe this video.
Input a video path or URL <<< https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4
In the video, a young child is seen sitting on a bed, engrossed in reading a book. The child is wearing glasses and appears to be enjoying the book. The bed is covered with a white blanket, and there are some toys scattered around the room. The child's focus on the book suggests that they are deeply immersed in the story. The room appears to be a comfortable and cozy space, with the child's playful demeanor adding to the overall warmth of the scene.
--------------------------------------------------
<<< clear
<<< <video>Describe this video.
Input a video path or URL <<< https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/fire.mp4
In the video, we see a person's hands holding a bag of chips. The person is standing in front of a fire pit, which is surrounded by a wooden fence. The fire pit is filled with wood, and there is a small fire burning in it. The person is holding the bag of chips over the fire pit, and we can see the flames from the fire reflected on the bag. The person then opens the bag and throws the chips onto the fire, causing them to sizzle and pop as they land on the burning wood. The sound of the chips hitting the fire can be heard clearly in the video. Overall, the video captures a simple yet satisfying moment of someone enjoying a snack while surrounded by the warmth and light of a fire pit.
--------------------------------------------------
<<< clear
<<< <image>Describe this image.
Input an image path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png
This is a close-up photograph of a kitten with a soft, blurred background. The kitten has a light brown coat with darker brown stripes and patches, typical of a calico pattern. Its eyes are wide open, and its nose is pink, which is common for young kittens. The kitten's whiskers are visible, and its ears are perked up, suggesting alertness. The image has a shallow depth of field, with the kitten in focus and the background out of focus, creating a bokeh effect.
--------------------------------------------------
<<< clear
<<< <image>How many sheep are in the picture?
Input an image path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png
There are four sheep in the picture.
"""
```

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

model_type = 'llava-next-video-7b-instruct'
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')

model, tokenizer = get_model_tokenizer(model_type, torch.float16,
                                       model_kwargs={'device_map': 'auto'})
model.generation_config.max_new_tokens = 1024
template = get_template(template_type, tokenizer)
seed_everything(42)

videos = ['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4']
query = '<video>Describe this video.'
response, _ = inference(model, template, query, videos=videos)
print(f'query: {query}')
print(f'response: {response}')

# Streaming
images = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png']
query = '<image>How many sheep are in the picture?'
gen = inference_stream(model, template, query, images=images)
print_idx = 0
print(f'query: {query}\nresponse: ', end='')
for response, _ in gen:
    delta = response[print_idx:]
    print(delta, end='', flush=True)
    print_idx = len(response)
print()

"""
query: <video>Describe this video.
response: In the video, a young child is seen sitting on a bed, engrossed in reading a book. The child is wearing a pair of glasses, which adds a touch of innocence to the scene. The child's focus is entirely on the book, indicating a sense of curiosity and interest in the content. The bed, covered with a white blanket, provides a cozy and comfortable setting for the child's reading session. The video captures a simple yet beautiful moment of a child's learning and exploration.
query: <image>How many sheep are in the picture?
response: There are four sheep in the picture.
"""
```


## Fine-tuning
Multimodal large model fine-tuning usually uses **custom datasets** for fine-tuning. Here is a demo that can be run directly:

LoRA fine-tuning:

(By default, only the qkv of the LLM part is fine-tuned using LoRA. If you want to fine-tune all linear layers including the vision model part, you can specify `--lora_target_modules ALL`.)
```shell
# Experimental environment: A10, 3090, V100...
# 21GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type llava-next-video-7b-instruct \
    --dataset video-chatgpt \
```

[Custom datasets](../LLM/Customization.md#-Recommended-Command-line-arguments) support json, jsonl formats. Here is an example of a custom dataset:

(Each round of conversation needs to include a video/image or not include a video/image, supports local path or URL input.)

```jsonl
{"query": "55555", "response": "66666", "videos": ["video_path"]}
{"query": "eeeee", "response": "fffff", "videos": ["video_path"]}
{"query": "EEEEE", "response": "FFFFF", "images": ["image_path"]}
```


## Inference after Fine-tuning
Direct inference:
```shell
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/llava-next-video-7b-instruct/vx-xxx/checkpoint-xxx \
    --load_dataset_config true
```

**merge-lora** and inference:
```shell
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir "output/llava-next-video-7b-instruct/vx-xxx/checkpoint-xxx" \
    --merge_lora true

CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir "output/llava-next-video-7b-instruct/vx-xxx/checkpoint-xxx-merged" \
    --load_dataset_config true
```
