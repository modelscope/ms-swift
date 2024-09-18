
# InternVL 最佳实践
本篇文档涉及的模型如下:

- [internvl-chat-v1_5](https://www.modelscope.cn/models/AI-ModelScope/InternVL-Chat-V1-5/summary)
- [internvl-chat-v1_5-int8](https://www.modelscope.cn/models/AI-ModelScope/InternVL-Chat-V1-5-int8/summary)
- [mini-internvl-chat-2b-v1_5](https://www.modelscope.cn/models/OpenGVLab/Mini-InternVL-Chat-2B-V1-5)
- [mini-internvl-chat-4b-v1_5](https://www.modelscope.cn/models/OpenGVLab/Mini-InternVL-Chat-4B-V1-5)
- [internvl2-1b](https://www.modelscope.cn/models/OpenGVLab/InternVL2-1B)
- [internvl2-2b](https://www.modelscope.cn/models/OpenGVLab/InternVL2-2B)
- [internvl2-4b](https://www.modelscope.cn/models/OpenGVLab/InternVL2-4B)
- [internvl2-8b](https://www.modelscope.cn/models/OpenGVLab/InternVL2-8B)
- [internvl2-26b](https://www.modelscope.cn/models/OpenGVLab/InternVL2-26B)
- [internvl2-40b](https://www.modelscope.cn/models/OpenGVLab/InternVL2-40B)
- [internvl2-llama3-76b](https://www.modelscope.cn/models/OpenGVLab/InternVL2-Llama3-76B)


以下实践以`internvl-chat-v1_5`为例，你也可以通过指定`--model_type`切换为其他模型.

**FAQ**

1. **模型显示 `The request model does not exist!`**

这种情况通常发生在尝试使用mini-internvl或InternVL2模型, 原因是modelscope上相应模型是申请制。解决这个问题，你需要登录modelscope, 并前往相应的模型页面进行**申请下载**, 申请成功后可以通过以下任意一种方式获取模型：
- 使用`snap_download`将模型下载到本地(在模型文件中的模型下载中有相应代码), 然后使用`--model_id_or_path`指定本地模型文件路径
- 在[modelscope账号主页](https://www.modelscope.cn/my/myaccesstoken)获取账号的SDK token, 使用参数`--hub_token`或者环境变量`MODELSCOPE_API_TOKEN`指定

也可以设置环境变量`USE_HF`, 从hugging face处下载模型

2. **多卡运行模型时, 为什么不同卡的分布不均匀, 导致OOM?**

transformers的auto device map算法对多模态模型支持不友好, 这可能导致不同 GPU 卡之间的显存分配不均匀。
- 可以通过参数`--device_max_memory`设置每张卡的显存使用, 比如四卡环境, 可以设置`--device_max_memory 15GB 15GB 15GB 15GB`
- 或者通过`--device_map_config`显式指定device map

3. **InternVL2模型与前系列(InternVL-V1.5和Mini-InternVL)模型的区别**

- InternVL2模型支持多轮多图推理和训练, 即多轮对话带有图片, 且单轮中支持文字图片交错,具体参考[自定义数据集](#自定义数据集)和推理的InternVL2部分。前系列模型支持多轮对话, 但只能有单轮带有图片
- InternVL2模型支持视频输入, 具体格式参考[自定义数据集](#自定义数据集)


## 目录
- [环境准备](#环境准备)
- [推理](#推理)
- [微调](#微调)
- [自定义数据集](#自定义数据集)
- [微调后推理](#微调后推理)


## 环境准备
```shell
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'
pip install Pillow
```

## 推理


**注意**
- 如果要使用本地模型文件，加上参数 `--model_id_or_path /path/to/model`
- 如果你的GPU不支持flash attention, 使用参数`--use_flash_attn false`。且对于int8模型，推理时需要指定`dtype --bf16`, 否则可能会出现乱码
- 模型本身config中的max_length较小，为2048，可以设置`--max_length`来修改
- 可以使用参数`--gradient_checkpoting true`减少显存占用

```shell
# Experimental environment: A100
# 55GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift infer --model_type internvl-chat-v1_5 --dtype bf16 --max_length 4096

# 2*30GB GPU memory
CUDA_VISIBLE_DEVICES=0,1 swift infer --model_type internvl-chat-v1_5 --dtype bf16 --max_length 4096
```

输出: (支持传入本地路径或URL)
```
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
门。
此外，我们也正在拓展其他模态的能力，目前我们支持了AnimateDiff的全
参数训练和LoRA训练。
SWIFT具有丰富的文档体系，如有使用问题请查看这里：
可以在Huggingface space和ModelScope创空间中体验SWIFT web-
ui功能了。
"""
```

对于**InternVL2**系列模型, 支持多轮多图推理, 且单轮中支持图像文本交错, 需要在输入通过`<image>`确定图像位置
多轮多图示例
```shell
CUDA_VISIBLE_DEVICES=0 swift infer --model_type internvl2-2b
```

```
<<< <image>描述图片
Input an image path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png
这是一只非常可爱的幼猫。它的毛色主要是白色，脸部和耳朵上有黑色和灰色的条纹。猫的眼睛大而明亮，瞳孔是蓝色的，显得非常清澈和无辜。它的耳朵竖立着，耳朵尖端是浅色的。这只猫看起来非常活泼，眼神中透露出一种好奇和友善的气质。背景模糊，可能是为了突出这只猫本身。整体画面给人一种温馨和可爱的感觉。
--------------------------------------------------
<<< <image>描述与上一张图片的差距
Input an image path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png
这张图片与上一张图片相比，有以下几个显著的差距：

1. **动物种类不同**：
   - 上一张图片中的动物是猫，而这张图片中的动物是羊。

2. **动物外观和特征不同**：
   - 猫有明显的竖立耳朵和蓝色的眼睛，而羊有竖立的长角和白色的羊毛。

3. **背景和环境不同**：
   - 上一张图片背景是模糊的，天空中有一些云朵，而这张图片背景是清晰的，背景中有绿色的山丘和蓝天白云。

4. **动物表情和姿态不同**：
   - 猫的表情显得非常警觉和好奇，而羊的表情显得非常温和和友好。

5. **整体风格和风格不同**：
   - 上一张图片有一种更加写实和细腻的绘画风格，而这张图片则有一种卡通和简化的风格。

这些差异使得这张图片与上一张图片在视觉上存在显著的差异。
--------------------------------------------------
<<< 两张图片的共同点是什么
两张图片的共同点如下：

1. **动物种类不同**：
   - 两张图片中的动物都是羊。

2. **背景和环境不同**：
   - 两张图片中的背景都是自然景观，包括绿色的山丘和蓝天白云。

3. **动物表情和姿态不同**：
   - 猫和羊的表情和姿态都不同，但都显得非常可爱和友好。

4. **整体风格和风格不同**：
   - 两张图片在风格上有所不同，但都具有卡通和简化的特点。

这些共同点使得两张图片在视觉上存在显著的差异，但它们都展示了可爱的动物形象。
--------------------------------------------------
<<< clear
<<< <video>Describe this video.
Input a video path or URL <<< https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4
In the video, a young child is seen sitting on a bed, engrossed in reading a book. The child is wearing a light blue shirt and dark glasses, and appears to be very focused on the book. The bed has a floral patterned cover, and there is a white blanket spread over it. The child's legs are crossed, and they are holding the book with both hands. The book is open, and the child is reading it with great interest.

As the child continues to read, they occasionally glance at the camera, seemingly curious about who is watching them. The child's expression is one of concentration and enjoyment, as they seem to be fully immersed in the story. The camera captures the child's face and the book, providing a clear view of their actions.

In the background, there is a glimpse of a room with a white wall and a wooden door. There is also a chair visible in the background, and a small table with a lamp on it. The room appears to be a bedroom, and the child seems to be in a comfortable and cozy environment.

The child's actions are repetitive, as they continue to read the book with great enthusiasm. The camera captures their movements and expressions, providing a detailed view of their reading experience. The child's focus and dedication to the book are evident, and the video conveys a sense of innocence and curiosity.

Overall, the video captures a heartwarming moment of a young child reading a book, showcasing their love for books and the joy of reading. The setting is simple and cozy, with a focus on the child's engagement with the book. The video is a delightful portrayal of childhood innocence and the simple pleasures of reading.
--------------------------------------------------
<<< clear
<<< image1: <img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png</img> image2: <img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png</img> What is the difference bewteen the two images?
The two images are of the same kitten, but the first image is a close-up shot, while the second image is a more distant, artistic illustration. The close-up image captures the kitten in detail, showing its fur, eyes, and facial features in sharp focus. In contrast, the artistic illustration is more abstract and stylized, with a blurred background and a different color palette. The distant illustration gives the kitten a more whimsical and dreamy appearance, while the close-up image emphasizes the kitten's realism and detail.
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
# os.environ['MODELSCOPE_API_TOKEN'] = 'Your API Token' # If the message "The request model does not exist!" appears.

from swift.llm import (
    get_model_tokenizer, get_template, inference,
    get_default_template_type, inference_stream
)
from swift.utils import seed_everything
import torch


model_type = "internvl-chat-v1_5"
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
query = '距离各城市多远'
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
query: 距离各城市多远
response: 根据图片，距离各城市如下：

- 马踏：14公里
- 阳江：62公里
- 广州：293公里

请注意，这些距离可能不是最新的，因为道路建设和交通状况可能会影响实际距离。
query: 距离最远的城市是哪？
response: 根据图片，距离最远的城市是广州，距离为293公里。
history: [['距离各城市多远', '根据图片，距离各城市如下：\n\n- 马踏：14公里\n- 阳江：62公里\n- 广州：293公里\n\n请注意，这些距离可能不是最新的，因为道路建设和交通状况可能会影响实际距离。 '], ['距离最远的城市是哪？', '根据图片，距离最远的城市是广州，距离为293公里。 ']]
"""
```

示例图片如下:

road:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png" width="250" style="display: inline-block;">


## 微调
多模态大模型微调通常使用**自定义数据集**进行微调. 这里展示可直接运行的demo:

LoRA微调:

**注意**
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

## 自定义数据集
[自定义数据集](../Instruction/自定义与拓展.md#-推荐命令行参数的形式)支持json, jsonl样式, 以下是自定义数据集的例子:

(支持多轮对话, 图片支持传入本地路径或URL, 多张图片用逗号','分割)

```jsonl
{"query": "55555", "response": "66666", "images": ["image_path"]}
{"query": "eeeee", "response": "fffff", "history": [], "images": ["image_path1", "image_path2"]}
{"query": "EEEEE", "response": "FFFFF", "history": [["query1", "response1"], ["query2", "response2"]], "images": ["image_path"]}
```

(支持纯文本数据)
```jsonl
{"query": "55555", "response": "66666"}
{"query": "eeeee", "response": "fffff", "history": []}
{"query": "EEEEE", "response": "FFFFF", "history": [["query1", "response1"], ["query2", "response2"]]}
```

**InternVL2**模型除了以上数据格式外, 还支持多图多轮训练, 使用tag `<image>` 标明图片在对话中的位置, 如果数据集中没有tag `<image>`, 默认放在最后一轮query的开头
```jsonl
{"query": "Image-1: <image>\nImage-2: <image>\nDescribe the two images in detail.", "response": "xxxxxxxxx", "history": [["<image>Describe the image", "xxxxxxx"], ["CCCCC", "DDDDD"]], "images": ["image_path1", "image_path2", "image_path3"]}
```
或者用`<img>image_path</img>` 表示图像路径和图像位置

```jsonl
{"query": "Image-1: <img>img_path</img>\n Image-2: <img>img_path2</img>\n Describe the two images in detail.", "response": "xxxxxxxxx", "history": [["<img>img_path3</img> Describe the image", "xxxxxxx"], ["CCCCC", "DDDDD"]], }
```

**InternVL2**模型支持视频数据集训练, 无需标明tag
```jsonl
{"query": "Describe this video in detail. Don't repeat", "response": "xxxxxxxxx", "history": [], "videos": ["video_path"]}
```

**InternVL2**模型支持grounding任务的训练，数据参考下面的格式：
```jsonl
{"query": "Find <bbox>", "response": "<ref-object>", "images": ["/coco2014/train2014/COCO_train2014_000000001507.jpg"], "objects": "[{\"caption\": \"guy in red\", \"bbox\": [138, 136, 235, 359], \"bbox_type\": \"real\", \"image\": 0}]" }
# mapping to multiple bboxes
{"query": "Find <ref-object>", "response": "<bbox>", "images": ["/coco2014/train2014/COCO_train2014_000000001507.jpg"], "objects": "[{\"caption\": \"guy in red\", \"bbox\": [[138, 136, 235, 359], [1,2,3,4]], \"bbox_type\": \"real\", \"image\": 0}]" }
```
上述objects字段中包含了一个json string，其中有四个字段：
    a. caption bbox对应的物体描述
    b. bbox 坐标 建议给四个整数（而非float型），分别是x_min,y_min,x_max,y_max四个值
    c. bbox_type: bbox类型 目前支持三种：real/norm_1000/norm_1，分别代表实际像素值坐标/千分位比例坐标/归一化比例坐标
    d. image: bbox对应的图片是第几张, 索引从0开始
上述格式会被转换为InternVL2可识别的格式，具体来说：
```jsonl
{"query": "Find <ref>the man</ref>", "response": "<box> [[200, 200, 600, 600]] </box>", "images": ["image_path1"]}
```
也可以直接传入上述格式，但是注意坐标请使用千分位坐标。

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
