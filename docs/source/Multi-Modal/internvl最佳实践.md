
# InternVL 最佳实践

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
pip install Pillow
```

## 推理

推理[internvl-chat-v1.5](https://www.modelscope.cn/models/AI-ModelScope/InternVL-Chat-V1-5/summary)和[internvl-chat-v1.5-int8](https://www.modelscope.cn/models/AI-ModelScope/InternVL-Chat-V1-5-int8/summary)

下面教程以`internvl-chat-v1.5`为例，你可以修改`--model_type internvl-chat-v1_5-int8`来选择int8版本的模型，使用`mini-internvl-chat-2b-v1_5`或
`mini-internvl-chat-4b-v1_5`来使用Mini-Internvl

**注意**
- 如果要使用本地模型文件，加上参数 `--model_id_or_path /path/to/model`
- 如果你的GPU不支持flash attention, 使用参数`--use_flash_attn false`。且对于int8模型，推理时需要指定`dtype --bf16`, 否则可能会出现乱码
- 模型本身config中的max_length较小，为2048，可以设置`--max_length`来修改
- 可以使用参数`--gradient_checkpoting true`减少显存占用
- InternVL系列模型的**训练**只支持带有图片的数据集

```shell
# Experimental environment: A100
# 55GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift infer --model_type internvl-chat-v1_5 --dtype bf16 --max_length 4096

# 2*30GB GPU memory
CUDA_VISIBLE_DEVICES=0,1 swift infer --model_type internvl-chat-v1_5 --dtype bf16 --max_length 4096
```

输出: (支持传入本地路径或URL)
```python
"""
<<< 你是谁
Input a media path or URL <<<
我是一个人工智能助手，旨在通过自然语言处理和机器学习技术来帮助用户解决问题和完成任务。
--------------------------------------------------
<<< clear
<<< 描述这张图片
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png
这张图片是一只小猫咪的特写照片。这只小猫咪有着蓝灰色的眼睛和白色的毛发，上面有灰色和黑色的条纹。它的耳朵是尖的，眼睛睁得大大的，看起来非常可爱和好奇。背景是模糊的，无法分辨具体的环境，但看起来像是在室内，有柔和的光线。
--------------------------------------------------
<<< clear
<<< 图中有几只羊
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png
图中有四只羊。
--------------------------------------------------
<<< clear
<<< 计算结果是多少?
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/math.png
将两个数相加，得到：
1452 + 45304 = 46766
因此，1452 + 45304 = 46766。
--------------------------------------------------
<<< clear
<<< 根据图片中的内容写首诗
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/poem.png
夜色笼罩水面，
小舟轻摇入画帘。
星辉闪烁如珠串，
月色朦胧似轻烟。

树影婆娑映水面，
静谧宁和心自安。
夜深人静思无限，
唯有舟影伴我眠。
--------------------------------------------------
<<< clear
<<< 对图片进行OCR
Input a media path or URL <<< https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/ocr.png
图中所有文字：
简介
SWIFT支持250＋LLM和35＋MLLM（多模态大模型）的训练、推
理、评测和部署。开发者可以直接将我们的框架应用到自己的Research和
生产环境中，实现模型训练评测到应用的完整链路。我们除支持
PEFT提供的轻量训练方案外，也提供了一个完整的Adapters库以支持
最新的训练技术，如NEFTune、LoRA+、LLaMA-PRO等，这个适配
器库可以脱离训练脚本直接使用在自已的自定义流程中。
为了方便不熟悉深度学习的用户使用，我们提供了一个Gradio的web-ui
于控制训练和推理，并提供了配套的深度学习课程和最佳实践供新手入
此外，我们也正在拓展其他模态的能力，目前我们支持了AnimateDiff的全参
数训练和LoRA训练。
SWIFT具有丰富的文档体系，如有使用问题请查看这里：
可以在Huggingface space和ModelScope创空间中体验SWIFT web-
ui功能了。
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

model_type = ModelType.internvl_chat_v1_5
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')
model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16,
                                       model_kwargs={'device_map': 'auto'})

# for GPUs that do not support flash attention
# model, tokenizer = get_model_tokenizer(model_type, torch.float16,
#                                        model_kwargs={'device_map': 'auto'},
#                                        use_flash_attn = False)

model.generation_config.max_new_tokens = 256
template = get_template(template_type, tokenizer)
seed_everything(42)

images = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png']
query = '距离各城市多远？'
response, history = inference(model, template, query, images=images) # chat with image
print(f'query: {query}')
print(f'response: {response}')

# 流式
query = '距离最远的城市是哪？'
gen = inference_stream(model, template, query, history) # chat without image
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
response: 这张图片显示的是一个路标，上面标示了三个目的地及其距离：

- 马踏（Mata）：14公里
- 阳江（Yangjiang）：62公里
- 广州（Guangzhou）：293公里

这些距离是按照路标上的指示来计算的。
query: 距离最远的城市是哪？
response: 根据这张图片，距离最远的城市是广州（Guangzhou），距离为293公里。
history: [['距离各城市多远？', '这张图片显示的是一个路标，上面标示了三个目的地及其距离：\n\n- 马踏（Mata）：14公里\n- 阳江（Yangjiang）：62公里\n- 广州（Guangzhou）：293公里\n\n这些距离是按照路标上的指示来计算的。 '], ['距离最远的城市是哪？', '根据这张图片，距离最远的城市是广州（Guangzhou），距离为293公里。 ']]
"""
```

示例图片如下:

road:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png" width="250" style="display: inline-block;">


## 微调
多模态大模型微调通常使用**自定义数据集**进行微调. 这里展示可直接运行的demo:

LoRA微调:

**注意**
- 默认只对LLM部分的qkv进行lora微调. 如果你想对所有linear含vision模型部分都进行微调, 可以指定`--lora_target_modules ALL`.
- 如果你的GPU不支持flash attention, 使用参数`--use_flash_attn false`

```shell
# Experimental environment: A100
# 80GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type internvl-chat-v1_5 \
    --dataset coco-en-2-mini \
    --max_length 4096

# device_map
# Experimental environment: 2*A100...
# 2*43GB GPU memory
CUDA_VISIBLE_DEVICES=0,1 swift sft \
    --model_type  internvl-chat-v1_5 \
    --dataset coco-en-2-mini \
    --max_length 4096

# ddp + deepspeed-zero2
# Experimental environment: 2*A100...
# 2*80GB GPU memory
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 swift sft \
    --model_type  internvl-chat-v1_5 \
    --dataset coco-en-2-mini \
    --max_length 4096 \
    --deepspeed default-zero2
```

全参数微调:
```bash
# Experimental environment: 4 * A100
# device map
# 4 * 72 GPU memory
CUDA_VISIBLE_DEVICES=0,1,2,3 swift sft \
    --model_type internvl-chat-v1_5 \
    --dataset coco-en-2-mini \
    --max_length 4096 \
    --sft_type full \
```


[自定义数据集](../LLM/自定义与拓展.md#-推荐命令行参数的形式)支持json, jsonl样式, 以下是自定义数据集的例子:

(只支持单轮对话, 每轮对话必须包含一张图片, 支持传入本地路径或URL)

```jsonl
{"query": "55555", "response": "66666", "images": ["image_path"]}
{"query": "eeeee", "response": "fffff", "images": ["image_path"]}
{"query": "EEEEE", "response": "FFFFF", "images": ["image_path"]}
```

## 微调后推理
直接推理:
```shell
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/internvl-chat-v1_5/vx-xxx/checkpoint-xxx \
    --load_dataset_config true \
    --max_length 4096
```

**merge-lora**并推理:
```shell
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir "output/internvl-chat-v1_5/vx-xxx/checkpoint-xxx" \
    --merge_lora true

CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir "output/internvl-chat-v1_5/vx-xxx/checkpoint-xxx-merged" \
    --load_dataset_config true \
    --max_length 4096

# device map
CUDA_VISIBLE_DEVICES=0,1 swift infer \
    --ckpt_dir "output/internvl-chat-v1_5/vx-xxx/checkpoint-xxx-merged" \
    --load_dataset_config true \
    --max_length 4096
```
