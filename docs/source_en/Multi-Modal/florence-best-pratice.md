
# Florence Best Practice

The document corresponds to the following models

| model | model_type |
|-------|------------|
| [Florence-2-base](https://www.modelscope.cn/models/AI-ModelScope/Florence-2-base) | florence-2-base |
| [Florence-2-base-ft](https://www.modelscope.cn/models/AI-ModelScope/Florence-2-base-ft) | florence-2-base-ft |
| [Florence-2-large](https://www.modelscope.cn/models/AI-ModelScope/Florence-2-large) | florence-2-large |
| [Florence-2-large-ft](https://www.modelscope.cn/models/AI-ModelScope/Florence-2-large-ft) | florence-2-large-ft |

The following practices take `florence-2-large-ft` as an example. You can also switch to other models by specifying the `--model_type`.

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

```shell
# 2.4GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift infer --model_type florence-2-large-ft --max_new_tokens 1024 --stream false
```

**Note**
- If you want to use local model files, add the parameter `--model_id_or_path /path/to/model`
- To use Flash Attention, include the parameter `--use_flash_attn true`, and specify `--dtype` as fp16 or bf16 (the model defaults to fp32).
- The Florence series models have built-in prompts for some vision tasks. You can check the corresponding mappings in `swift.llm.utils.template.FlorenceTemplate`. More prompts can be found on the Modelscope/Hugging Face model detail pages.
- The Florence series models do not support Chinese.
- The Florence series models do not support system prompts and history.


Output: (supports passing in local path or URL)
```python
"""
<<< Describe the image
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png
{'Describe the image': 'A grey and white kitten with blue eyes.'}
<<< <OD>
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png
{'Locate the objects with category name in the image.': 'shelf<loc_264><loc_173><loc_572><loc_748><loc_755><loc_274><loc_966><loc_737><loc_46><loc_335><loc_261><loc_763><loc_555><loc_360><loc_760><loc_756>'}
--------------------------------------------------
<<< <CAPTION>
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png
{'What does the image describe?': 'A cartoon picture of four sheep standing in a field.'}
--------------------------------------------------
<<< <DETAILED_CAPTION>
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png
{'Describe in detail what is shown in the image.': 'In the image is animated. In the image there are sheeps. At the bottom of the image on the ground there is grass. In background there are hills. At top of the images there are clouds.'}
--------------------------------------------------
<<< <MORE_DETAILED_CAPTION>
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png
{'Describe with a paragraph what is shown in the image.': 'Four sheep are standing in a field. They are all white and fluffy. They have horns on their heads. There are mountains behind them. There is grass and weeds on the ground in front of them. '}
--------------------------------------------------
<<< <DENSE_REGION_CAPTION>
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png
{'Locate the objects in the image, with their descriptions.': 'cartoon sheep illustration<loc_265><loc_175><loc_572><loc_748>cartoon ram illustration<loc_755><loc_275><loc_966><loc_737>cartoon white sheep illustration<loc_44><loc_335><loc_262><loc_764>cartoon goat illustration<loc_555><loc_361><loc_762><loc_756>'}
--------------------------------------------------
<<< <REGION_PROPOSAL>
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png
{'Locate the region proposals in the image.': '<loc_45><loc_176><loc_967><loc_761><loc_266><loc_175><loc_570><loc_749><loc_757><loc_274><loc_966><loc_738><loc_46><loc_334><loc_261><loc_763><loc_556><loc_361><loc_760><loc_756>'}
--------------------------------------------------
<<< <CAPTION_TO_PHRASE_GROUNDING>the sheeps
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png
{'Locate the phrases in the caption: the sheeps': 'thethe sheeps<loc_45><loc_175><loc_967><loc_764><loc_266><loc_176><loc_572><loc_749><loc_756><loc_275><loc_965><loc_739><loc_46><loc_335><loc_261><loc_765><loc_557><loc_361><loc_760><loc_758>'}
```

Example images are as follows:

cat:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png" width="250" style="display: inline-block;">

animal:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png" width="250" style="display: inline-block;">

**Python Inference**

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference,
    get_default_template_type, inference_stream
)
from swift.utils import seed_everything

model_type = "florence-2-large-ft"
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')
model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': "cuda:0"})

model.generation_config.max_new_tokens = 1024
template = get_template(template_type, tokenizer)
seed_everything(42)

images = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png']
query = 'Describe the image'
response, history = inference(model, template, query, images=images)
print(f'query: {query}')
print(f'response: {response}')
'''
query: Describe the image
response: {'Describe the image': 'Four sheep standing in a field with mountains in the background.'}
'''
```

Multimodal large model fine-tuning usually uses **custom datasets** for fine-tuning. Here is a demo that can be run directly:

LoRA fine-tuning:
```shell
# Experimental environment: 4090
# 6.6GB GPU memory

# caption task
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type florence-2-large-ft \
    --dataset coco-en-2-mini \
    --lora_target_modules ALL

# grounding task
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type florence-2-large-ft \
    --dataset refcoco-unofficial-grounding \
    --lora_target_modules ALL
```

Full parameter fine-tuning:
```bash
# Experimental environment: 4090
# 11 GPU memory
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type florence-2-large-ft \
    --dataset coco-en-2-mini \
    --sft_type full

```

[Custom datasets](../Instruction/Customization.md#-Recommended-Command-line-arguments) support json, jsonl formats. Here is an example of a custom dataset:


**Caption/VQA** task
```jsonl
{"query": "55555", "response": "66666", "images": ["image_path"]}
{"query": "eeeee", "response": "fffff", "images": ["image_path"]}
{"query": "EEEEE", "response": "FFFFF", "images": ["image_path"]}
```

**grounding** task

Currently, two types of custom grounding tasks are supported:

1. For tasks asking about the target for a given bounding box, specify `<bbox>` in the query, `<ref-object>` in the response, and provide the target and bounding box details in objects.
2. For tasks asking about the bounding box for a given target, specify `<ref-object>` in the query, `<bbox>` in the response, and provide the target and bounding box details in objects.
```jsonl
{"query": "Find <bbox>", "response": "<ref-object>", "images": ["/coco2014/train2014/COCO_train2014_000000001507.jpg"], "objects": "[{\"caption\": \"guy in red\", \"bbox\": [138, 136, 235, 359], \"bbox_type\": \"real\", \"image\": 0}]" }
# mapping to multiple bboxes
{"query": "Find <ref-object>", "response": "<bbox>", "images": ["/coco2014/train2014/COCO_train2014_000000001507.jpg"], "objects": "[{\"caption\": \"guy in red\", \"bbox\": [[138, 136, 235, 359],[1,2,3,4]], \"bbox_type\": \"real\", \"image\": 0}]" }
```
The `objects` field contains a JSON string with four fields:
  1. `caption`: Description of the object corresponding to the bounding box (bbox)
  2. `bbox`: Coordinates of the bounding box. It is recommended to provide four integers (rather than float values), specifically `x_min`, `y_min`, `x_max`, and `y_max`.
  3. `bbox_type`: Type of the bounding box. Currently, three types are supported: `real`, `norm_1000`, and `norm_1`, which respectively represent actual pixel value coordinates, thousandth ratio coordinates, and normalized ratio coordinates.
  4. `image`: The index of the image corresponding to the bounding box. The index starts from 0.


Let me know if you need further assistance!

## Inference after Fine-tuning
Direct inference:
```shell
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/florence-2-large-ft/vx-xxx/checkpoint-xxx \
    --stream false \
    --max_new_tokens 1024
```

**merge-lora** and inference:
```shell
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir "output/florence-2-large-ft/vx-xxx/checkpoint-xxx" \
    --stream false \
    --max_new_tokens 1024 \
    --merge_lora true

CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir "output/florence-2-large-ft/vx-xxx/checkpoint-xxx-merged" \
    --stream false \
    --max_new_tokens 1024 \
```
