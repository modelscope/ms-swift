# InternVL Best Practice
The document corresponds to the following models:

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

The following practice takes `internvl-chat-v1_5` as an example, and you can also switch to other models by specifying `--model_type`.

**FAQ**
1. **Model shows `The request model does not exist!`**

This issue often arises when attempting to use the mini-internvl or InternVL2 models, as the corresponding models on modelscope are subject to an application process. To resolve this, you need to log in to modelscope and go to the respective model page to apply for download. After approval, you can obtain the model through either of the following methods:
- Use `snap_download` to download the model locally (the relevant code is available in the model download section of the model file), and then specify the local model file path using `--model_id_or_path`.
- Obtain the SDK token for your account from the [modelscope account homepage](https://www.modelscope.cn/my/myaccesstoken), and specify it using the `--hub_token` parameter or the `MODELSCOPE_API_TOKEN` environment variable.

2. **Why is the distribution uneven across multiple GPU cards when running models, leading to OOM?**

The auto device map algorithm in transformers is not friendly to multi-modal models, which may result in uneven memory allocation across different GPU cards.

- You can set the memory usage for each card using the `--device_max_memory parameter`, for example, in a four-card environment, you can set `--device_max_memory 15GB 15GB 15GB 15GB`.
- Alternatively, you can explicitly specify the device map using `--device_map_config_path`.

3. **Differences between the InternVL2 model and its predecessors (InternVL-V1.5 and Mini-InternVL)**

- The InternVL2 model supports multi-turn multi-image inference and training, meaning multi-turn conversations with images, and supports text and images interleaved within a single turn. For details, refer to [Custom Dataset](#custom-dataset) and InternVL2 part in Inference section. The predecessors models supported multi-turn conversations but could only have images in a single turn.
- The InternVL2 model supports video input. For specific formats, refer to [Custom Dataset](#custom-dataset).

## Table of Contents
- [Environment Setup](#environment-setup)
- [Inference](#inference)
- [Fine-tuning](#fine-tuning)
- [Custom Dataset](#custom-dataset)
- [Inference after Fine-tuning](#inference-after-fine-tuning)

## Environment Setup
```shell
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'
pip install Pillow
```

## Inference

**Note**
- If you want to use a local model file, add the argument --model_id_or_path /path/to/model.
- If your GPU does not support flash attention, use the argument --use_flash_attn false. And for int8 models, it is necessary to specify `dtype --bf16` during inference, otherwise the output may be garbled.
- The model's configuration specifies a relatively small max_length of 2048, which can be modified by setting `--max_length`.
- Memory consumption can be reduced by using the parameter `--gradient_checkpointing true`.

```shell
# Experimental environment: A100
# 55GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift infer --model_type internvl-chat-v1_5 --dtype bf16 --max_length 4096

# 2*30GB GPU memory
CUDA_VISIBLE_DEVICES=0,1 swift infer --model_type internvl-chat-v1_5 --dtype bf16 --max_length 4096
```

Output: (supports passing in local path or URL)
```python
"""
<<< Describe this image.
Input a media path or URL <<<  http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png
This is a high-resolution image of a kitten. The kitten has striking blue eyes and a fluffy white and grey coat. The fur pattern suggests that it may be a Maine Coon or a similar breed. The kitten's ears are perked up, and it has a curious and innocent expression. The background is blurred, which brings the focus to the kitten's face.
--------------------------------------------------
<<< clear
<<< How many sheep are in the picture?
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png
There are four sheep in the picture.
--------------------------------------------------
<<< clear
<<< What is the calculation result?
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/math.png
The calculation result is 59,856.
--------------------------------------------------
<<< clear
<<< Write a poem based on the content of the picture.
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/poem.png
Token indices sequence length is longer than the specified maximum sequence length for this model (5142 > 4096). Running this sequence through the model will result in indexing errors
In the still of the night,
A lone boat sails on the light.
The stars above, a twinkling sight,
Reflecting in the water's might.

The trees stand tall, a silent guard,
Their leaves rustling in the yard.
The boatman's lantern, a beacon bright,
Guiding him through the night.

The river flows, a gentle stream,
Carrying the boatman's dream.
His journey long, his heart serene,
In the beauty of the scene.

The stars above, a guiding light,
Leading him through the night.
The boatman's journey, a tale to tell,
Of courage, hope, and love as well.
"""
```
For the **InternVL2** series models, multi-turn multi-image inference is supported, and within a single turn, images and text can be interleaved. The position of the images needs to be determined by <image> in the input.

Example of multi-turn multi-image input:
```shell
CUDA_VISIBLE_DEVICES=0 swift infer --model_type internvl2-2b
```

```
<<< <image>describe the image
Input an image path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png
The image depicts a kitten with striking blue eyes and a mix of white and black fur. The kitten's fur appears soft and fluffy, and its ears are large and pointed. The eyes of the kitten are particularly expressive, giving it a curious and endearing appearance. The background is blurred, drawing attention to the kitten's detailed features. The overall composition of the image highlights the kitten's innocence and charm.
--------------------------------------------------
<<< <image>What is the difference from the last picture
Input an image path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png
The difference between the last picture and the one in the image is primarily in the facial expressions and the overall demeanor of the sheep.

In the first picture, the sheep have a calm and curious expression, with large, expressive eyes and a gentle demeanor. The background is blurred, emphasizing the sheep's features.

In the second picture, the sheep have a more serious and focused expression. Their eyes are slightly narrowed, and their ears are perked up, giving them a more alert and attentive appearance. The background remains the same, maintaining the focus on the sheep.

In the third picture, the sheep have a more neutral expression. Their eyes are slightly narrowed, and their ears are perked up, but their overall demeanor is still calm and attentive. The background remains the same, ensuring the focus remains on the sheep.

In the fourth picture, the sheep have a more neutral and calm expression. Their eyes are slightly narrowed, and their ears are perked up, but their overall demeanor is still calm and attentive. The background remains the same, maintaining the focus on the sheep.

Overall, the main difference is the facial expressions and the overall demeanor of the sheep, with the second picture showing a more alert and focused look compared to the other three pictures.
--------------------------------------------------
<<< What the two pictures have in common
The two pictures have several similarities:

1. **Foreground Sheep**: All four sheep in the images are in the foreground, standing on a grassy field. This creates a sense of unity and focus on the sheep.

2. **Expression and Demeanor**: Each sheep has a distinct expression and demeanor. The first picture shows a curious and calm sheep, the second shows a more alert and focused sheep, the third shows a neutral and calm sheep, and the fourth shows a neutral and calm sheep. Despite the differences in expressions, all four sheep maintain a calm demeanor.

3. **Background**: The background remains consistent in all four pictures. The background features a blurred green field with mountains in the distance, which helps to keep the focus on the sheep.

4. **Art Style**: The art style is consistent across all four pictures. The sheep are depicted in a cartoonish and friendly manner, with large eyes and expressive faces.

5. **Overall Composition**: The composition of the images is similar, with the sheep standing in the foreground and the background featuring a blurred natural landscape.

These similarities create a cohesive and engaging visual experience, despite the differences in expressions and demeanor.
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

Example images are as follows:

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
query = 'How far is it from each city?'
response, history = inference(model, template, query, images=images)  # chat with image
print(f'query: {query}')
print(f'response: {response}')

# 流式
query = 'Which city is the farthest?'
gen = inference_stream(model, template, query, history)  # chat withoud image
print_idx = 0
print(f'query: {query}\nresponse: ', end='')
for response, history in gen:
    delta = response[print_idx:]
    print(delta, end='', flush=True)
    print_idx = len(response)
print()
print(f'history: {history}')
"""
query: How far is it from each city?
response: The distances from the location of the sign to each city are as follows:

- Mata: 14 kilometers
- Yangjiang: 62 kilometers
- Guangzhou: 293 kilometers

These distances are indicated on the road sign in the image.
query: Which city is the farthest?
response: The city that is farthest from the location of the sign is Guangzhou, which is 293 kilometers away.
history: [['How far is it from each city?', 'The distances from the location of the sign to each city are as follows:\n\n- Mata: 14 kilometers\n- Yangjiang: 62 kilometers\n- Guangzhou: 293 kilometers\n\nThese distances are indicated on the road sign in the image. '], ['Which city is the farthest?', 'The city that is farthest from the location of the sign is Guangzhou, which is 293 kilometers away. ']]
"""
```



Example image is as follows:

road:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/road.png" width="250" style="display: inline-block;">


## Fine-tuning
Multimodal large model fine-tuning usually uses **custom datasets** for fine-tuning. Here is a demo that can be run directly:

LoRA fine-tuning:

**note**
- If your GPU does not support flash attention, use the argument --use_flash_attn false.

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

Full parameter fine-tuning:
```shell
# Experimental environment: 4 * A100
# device map
# 4 * 72 GPU memory
CUDA_VISIBLE_DEVICES=0,1,2,3 swift sft \
    --model_type internvl-chat-v1_5 \
    --dataset coco-en-2-mini \
    --sft_type full \
    --max_length 4096
```

### Custom Dataset
[Custom datasets](../LLM/Customization.md#-Recommended-Command-line-arguments) support json, jsonl formats. Here is an example of a custom dataset:

Supports multi-turn conversations, Images support for local path or URL input, multiple images separated by commas ','

```jsonl
{"query": "55555", "response": "66666", "images": ["image_path"]}
{"query": "eeeee", "response": "fffff", "history": [], "images": ["image_path1", "image_path2"]}
{"query": "EEEEE", "response": "FFFFF", "history": [["query1", "response1"], ["query2", "response2"]], "images": ["image_path"]}
```

(Supports data without images)
```jsonl
{"query": "55555", "response": "66666"}
{"query": "eeeee", "response": "fffff", "history": []}
{"query": "EEEEE", "response": "FFFFF", "history": [["query1", "response1"], ["query2", "response2"]]}
```

In addition to the above data formats, the **InternVL2** model also supports multi-image multi-turn training. It uses the tag `<image>` to indicate the position of images in the conversation. If the tag `<image>` is not present in the dataset, the images are placed at the beginning of the last round's query by default.
```jsonl
{"query": "Image-1: <image>\nImage-2: <image>\nDescribe the two images in detail.", "response": "xxxxxxxxx", "history": [["<image>Describe the image", "xxxxxxx"], ["CCCCC", "DDDDD"]], "images": ["image_path1", "image_path2", "image_path3"]}
```
Alternatively, use `<img>image_path</img>` to represent the image path and image location.

```jsonl
{"query": "Image-1: <img>img_path</img>\n Image-2: <img>img_path2</img>\n Describe the two images in detail.", "response": "xxxxxxxxx", "history": [["<img>img_path3</img> Describe the image", "xxxxxxx"], ["CCCCC", "DDDDD"]], }
```

The **InternVL2** model supports training with video datasets without the need to specify a tag.
```jsonl
{"query": "Describe this video in detail. Don't repeat", "response": "xxxxxxxxx", "history": [], "videos": ["video_path"]}
```

The **InternVL2** model supports training for grounding tasks, with data referenced in the following format:
```jsonl
{"query": "Find <bbox>", "response": "<ref-object>", "images": ["/coco2014/train2014/COCO_train2014_000000001507.jpg"], "objects": "[{\"caption\": \"guy in red\", \"bbox\": [138, 136, 235, 359], \"bbox_type\": \"real\", \"image\": 0}]" }
{"query": "Find <ref-object>", "response": "<bbox>", "images": ["/coco2014/train2014/COCO_train2014_000000001507.jpg"], "objects": "[{\"caption\": \"guy in red\", \"bbox\": [138, 136, 235, 359], \"bbox_type\": \"real\", \"image\": 0}]" }
```
The `objects` field contains a JSON string with four fields:
  1. **caption**: Description of the object corresponding to the bounding box.
  2. **bbox**: Coordinates suggested as four integers (instead of floats), representing the values `x_min`, `y_min`, `x_max`, and `y_max`.
  3. **bbox_type**: Type of bounding box. Currently, three types are supported: `real` / `norm_1000` / `norm_1`, representing actual pixel value coordinates / thousandth-scale coordinates / normalized coordinates.
  4. **image**: The index of the corresponding image, starting from 0.

This format will be converted to a format recognizable by InternVL2, specifically:
```json
{"query": "Find <ref>the man</ref>", "response": "<box> [[200, 200, 600, 600]] </box>", "images": ["image_path1"]}
```
You can also directly input the above format, but please ensure that the coordinates use thousandth-scale coordinates.

## Inference after Fine-tuning
Direct inference:
```shell
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/internvl-chat-v1_5/vx-xxx/checkpoint-xxx \
    --load_dataset_config true \
    --max_length 4096
```

**merge-lora** and inference:
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
