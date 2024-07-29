

# Llava Video 最佳实践

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
# Experimental environment: A10
# 20GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift infer --model_type llava-next-video-7b-instruct
```

输出: (支持传入本地路径或URL)
```python
"""
<<< 你是谁
我是 Assistant，一个大型语言模型。 我被训练来回答各种问题，包括提供信息、提供建议、提供帮助等等。 我可以回答你关于各种话题的问题，但如果你有具体问题，请告诉我，我会尽力回答。
--------------------------------------------------
<<< clear
<<< <video>描述这段视频
Input a video path or URL <<< https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4
这段视频展示了一个小孩在躺在床上，正在玩一本书。她穿着粉色的玩具裤和绿色的玩具裙，穿着眼镜。她的手在书上摸索，她的脸上带着微笑，看起来很开心。她的头发是金色的，整个场景充满了温馨和轻松的氛围。
--------------------------------------------------
<<< clear
<<< <video>Describe this video.
Input a video path or URL <<< https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/fire.mp4
In the video, a person is seen holding a bag of chips and a lighter. The person then proceeds to light the chips on fire, creating a small fire. The fire is contained within the bag, and the person appears to be enjoying the fire as they watch it burn. The video is a simple yet intriguing display of pyromania, where the person is fascinated by the fire and enjoys watching it burn. The use of the bag as a container for the fire adds an element of danger to the scene, as it could potentially cause the fire to spread or cause injury. Overall, the video is a brief yet captivating display of pyromania and the allure of fire.
--------------------------------------------------
<<< clear
<<< <image>描述这张图片
Input an image path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png
这张图片是一张照片，显示了一只充满活力和可爱的猫咪。它的头部和脸部呈现出细腻的白色和柔和的灰色斑点，给人一种非常可爱的感觉。猫咪的眼睛非常大，充满了生机和好奇，它们的色彩是深蓝色，与猫咪的眼睛通常的颜色相反。猫咪的耳朵看起来很小，即使它们是很大的猫咪，也很常见。它的身体看起来很健康，毛发柔软而光滑，呈现出一种非常柔和的外观。
--------------------------------------------------
<<< clear
<<< <image>图中有几只羊
Input an image path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png
在这张图中，有四只羊。
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

model_type = 'llava-next-video-7b-instruct'
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')

model, tokenizer = get_model_tokenizer(model_type, torch.float16,
                                       model_kwargs={'device_map': 'auto'})
model.generation_config.max_new_tokens = 1024
template = get_template(template_type, tokenizer)
seed_everything(42)

videos = ['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4']
query = '<video>描述这段视频'
response, _ = inference(model, template, query, videos=videos)
print(f'query: {query}')
print(f'response: {response}')

# 流式
images = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png']
query = '<image>图中有几只羊'
gen = inference_stream(model, template, query, images=images)
print_idx = 0
print(f'query: {query}\nresponse: ', end='')
for response, _ in gen:
    delta = response[print_idx:]
    print(delta, end='', flush=True)
    print_idx = len(response)
print()

"""
query: <video>描述这段视频
response: 这段视频展示了一个小孩在床上享受一本书的愉悦。她穿着一件简单的纱衣，头戴着眼镜，手轻轻地摸索着书页。她的表情充满了兴奋和惊喜，她的眼睛时不时地眨眼地看着书页，仿佛在探索一个新的世界。她的姿势和动作都充满了轻松和自然，让人感觉到她在享受这个简单而美好的时刻。
query: <image>图中有几只羊
response: 在这张图像中，有四只羊。
"""
```


## 微调
多模态大模型微调通常使用**自定义数据集**进行微调. 这里展示可直接运行的demo:

LoRA微调:

(默认只对LLM部分的qkv进行lora微调. 如果你想对所有linear含vision模型部分都进行微调, 可以指定`--lora_target_modules ALL`.)
```shell
# Experimental environment: A10, 3090, V100...
# 21GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type llava-next-video-7b-instruct \
    --dataset video-chatgpt \
```

[自定义数据集](../LLM/自定义与拓展.md#-推荐命令行参数的形式)支持json, jsonl样式, 以下是自定义数据集的例子:

(每轮对话需包含一段视频/图片或不含视频/图片, 支持传入本地路径或URL)

```jsonl
{"query": "55555", "response": "66666", "videos": ["video_path"]}
{"query": "eeeee", "response": "fffff", "videos": ["video_path"]}
{"query": "EEEEE", "response": "FFFFF", "images": ["image_path"]}
```

## 微调后推理
直接推理:
```shell
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/llava-next-video-7b-instruct/vx-xxx/checkpoint-xxx \
    --load_dataset_config true
```

**merge-lora**并推理:
```shell
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir "output/llava-next-video-7b-instruct/vx-xxx/checkpoint-xxx" \
    --merge_lora true

CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir "output/llava-next-video-7b-instruct/vx-xxx/checkpoint-xxx-merged" \
    --load_dataset_config true
```
