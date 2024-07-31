
# CogVLM2 Video 最佳实践

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

# https://github.com/facebookresearch/pytorchvideo/issues/258
# https://github.com/dmlc/decord/issues/177
pip install decord pytorchvideo
```

模型链接:
- cogvlm2-video-13b-chat: [https://modelscope.cn/models/ZhipuAI/cogvlm2-video-llama3-chat](https://modelscope.cn/models/ZhipuAI/cogvlm2-video-llama3-chat)


## 推理

推理cogvlm2-video-13b-chat:
```shell
# Experimental environment: A100
# 28GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift infer --model_type cogvlm2-video-13b-chat
```

输出: (支持传入本地路径或URL)
```python
"""
<<< 描述这段视频
Input a video path or URL <<< https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4
In the video, a young child is seen sitting on a bed and reading a book. The child is wearing glasses and is dressed in a light blue top and pink pants. The room appears to be a bedroom with a crib in the background. The child is engrossed in the book, and the scene is captured in a series of frames showing the child's interaction with the book.
--------------------------------------------------
<<< clear
<<< Describe this video.
Input a video path or URL <<< https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/fire.mp4
In the video, a person is seen lighting a fire in a backyard setting. They start by holding a piece of food and then proceed to light a match to the food. The fire is then ignited, and the person continues to light more pieces of food, including a bag of chips and a piece of wood. The fire is seen burning brightly, and the person is seen standing over the fire, possibly enjoying the warmth. The video captures the process of starting a fire and the person's interaction with the flames, creating a cozy and inviting atmosphere.
--------------------------------------------------
<<< clear
<<< who are you
Input a video path or URL <<<
I am a person named John.
"""
```

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

model_type = ModelType.cogvlm2_video_13b_chat
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')

model, tokenizer = get_model_tokenizer(model_type, torch.float16,
                                       model_kwargs={'device_map': 'auto'})
model.generation_config.max_new_tokens = 256
template = get_template(template_type, tokenizer)
seed_everything(42)

videos = ['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4']
query = '描述这段视频'
response, history = inference(model, template, query, videos=videos)
print(f'query: {query}')
print(f'response: {response}')

# 流式
query = 'Describe this video.'
videos = ['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/fire.mp4']
gen = inference_stream(model, template, query, history, videos=videos)
print_idx = 0
print(f'query: {query}\nresponse: ', end='')
for response, _ in gen:
    delta = response[print_idx:]
    print(delta, end='', flush=True)
    print_idx = len(response)
print()

"""
query: 描述这段视频
response: The video depicts a young child sitting on a bed and reading a book. The child is wearing glasses and is seen in various positions, such as sitting on the bed, sitting on a couch, and sitting on a bed with a blanket. The child's attire changes from a light blue top and pink pants to a light blue top and pink leggings. The room has a cozy and warm atmosphere with soft lighting, and there are personal items scattered around, such as a crib, a television, and a white garment.
query: Describe this video.
response: The video shows a person lighting a fire in a backyard setting. The person is seen holding a piece of food and a lighter, and then lighting the food on fire. The fire is then used to light other pieces of wood, and the person is seen standing over the fire, holding a bag of food. The video captures the process of starting a fire and the person's interaction with the fire.
"""
```


## 微调
多模态大模型微调通常使用**自定义数据集**进行微调. 这里展示可直接运行的demo:

(默认对LLM的qkv进行lora微调. 如果你想对所有linear都进行微调, 可以指定`--lora_target_modules ALL`)
```shell
# Experimental environment: A100
# 40GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type cogvlm2-video-13b-chat \
    --dataset video-chatgpt \
    --num_train_epochs 3 \

# ZeRO2
# Experimental environment: 4 * A100
# 4 * 40GB GPU memory
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 swift sft \
    --model_type cogvlm2-video-13b-chat \
    --dataset video-chatgpt \
    --num_train_epochs 3 \
    --deepspeed default-zero2
```

[自定义数据集](../LLM/自定义与拓展.md#-推荐命令行参数的形式)支持json, jsonl样式, 以下是自定义数据集的例子:

(支持多轮对话, 但总的轮次对话只能包含一张图片, 支持传入本地路径或URL)

```jsonl
{"query": "55555", "response": "66666", "videos": ["video_path"]}
{"query": "eeeee", "response": "fffff", "history": [], "videos": ["video_path"]}
{"query": "EEEEE", "response": "FFFFF", "history": [["query1", "response1"], ["query2", "response2"]], "videos": ["video_path"]}
```


## 微调后推理
直接推理:
```shell
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/cogvlm2-video-13b-chat/vx-xxx/checkpoint-xxx \
    --load_dataset_config true \
```

**merge-lora**并推理:
```shell
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir output/cogvlm2-video-13b-chat/vx-xxx/checkpoint-xxx \
    --merge_lora true

CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/cogvlm2-video-13b-chat/vx-xxx/checkpoint-xxx-merged \
    --load_dataset_config true
```
