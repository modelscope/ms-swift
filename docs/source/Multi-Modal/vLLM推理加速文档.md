# vLLM推理加速文档
ms-swift已接入了vLLM对多模态模型进行推理加速. 支持的模型可以查看[支持的模型和数据集](../LLM/支持的模型和数据集.md#多模态大模型).

## 目录
- [环境准备](#环境准备)
- [推理加速](#推理加速)
- [部署](#部署)


## 环境准备
```bash
# 设置pip全局镜像 (加速下载)
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# 安装ms-swift
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'

# vllm与cuda版本有对应关系，请按照`https://docs.vllm.ai/en/latest/getting_started/installation.html`选择版本
pip install "vllm>=0.5.1"
pip install openai -U
```


## 推理加速

使用python:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    ModelType, get_vllm_engine, get_default_template_type,
    get_template, inference_vllm
)

# 'minicpm-v-v2_5-chat', 'minicpm-v-v2_6-chat', 'internvl2-1b', 'internvl2-4b', 'phi3-vision-128k-instruct'
model_type = ModelType.llava1_6_mistral_7b_instruct
model_id_or_path = None
llm_engine = get_vllm_engine(model_type, model_id_or_path=model_id_or_path)
template_type = get_default_template_type(model_type)
template = get_template(template_type, llm_engine.hf_tokenizer)

llm_engine.generation_config.max_new_tokens = 1024

images = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png']
request_list = [{'query': 'who are you'}, {'query': 'Describe this image.', 'images': images}]
resp_list = inference_vllm(llm_engine, template, request_list)
for request, resp in zip(request_list, resp_list):
    print(f"query: {request['query']}")
    print(f"response: {resp['response']}")

history1 = resp_list[1]['history']
images.append(None)
request_list = [{'query': 'Is the creature in the picture a dog?', 'history': history1, 'images': images}]
resp_list = inference_vllm(llm_engine, template, request_list)
for request, resp in zip(request_list, resp_list):
    print(f"query: {request['query']}")
    print(f"response: {resp['response']}")
    print(f"history: {resp['history']}")

"""
query: who are you
response: Hello! I am an AI language model, designed to assist users with information and provide helpful prompts and suggestions. As an artificial intelligence, I do not have personal experiences, so I don't have a personality or individuality. Instead, my purpose is to provide accurate, useful information to users like you. Is there anything specific you would like help with or any other questions you have?
query: Describe this image.
response: The image features a close-up of a kitten's face. The kitten has striking blue eyes, which are open and appear to be looking towards the camera. Its fur exhibits a mix of black and white stripes with black markings around its eyes. The fur texture is soft and dense with whiskers adorning the sides of its face, adding to its feline charm. The background is blurred with hints of green and white, which creates a bokeh effect, keeping the focus on the kitten's face. The image exudes a sense of innocence and curiosity typically associated with young felines.
query: Is the creature in the picture a dog?
response: No, the creature in the picture is a kitten, which is a young cat, not a dog. The presence of distinct feline features such as stripes, whiskers, and the appearance of blue eyes confirms this.
history: [['Describe this image.', "The image features a close-up of a kitten's face. The kitten has striking blue eyes, which are open and appear to be looking towards the camera. Its fur exhibits a mix of black and white stripes with black markings around its eyes. The fur texture is soft and dense with whiskers adorning the sides of its face, adding to its feline charm. The background is blurred with hints of green and white, which creates a bokeh effect, keeping the focus on the kitten's face. The image exudes a sense of innocence and curiosity typically associated with young felines. "], ['Is the creature in the picture a dog?', 'No, the creature in the picture is a kitten, which is a young cat, not a dog. The presence of distinct feline features such as stripes, whiskers, and the appearance of blue eyes confirms this. ']]
"""
```


batch处理:
```python
# vllm>=0.5.4
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_vllm_engine, get_template, inference_vllm, ModelType,
    get_default_template_type, inference_stream_vllm
)
from swift.utils import seed_everything
import torch

model_type = ModelType.minicpm_v_v2_6_chat
model_id_or_path = None
vllm_engine = get_vllm_engine(model_type, torch.bfloat16, model_id_or_path=model_id_or_path,
                              max_model_len=8192)

tokenizer = vllm_engine.hf_tokenizer
vllm_engine.generation_config.max_new_tokens = 256
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')
template = get_template(template_type, tokenizer)
seed_everything(42)

query = '<image>描述这张图片'
images = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png']
generation_info = {}
request_list = [{'query': query, 'images': images} for _ in range(100)]
resp_list = inference_vllm(vllm_engine, template, request_list, generation_info=generation_info, use_tqdm=True)
print(f'query: {query}')
print(f'response: {resp_list[0]["response"]}')
print(generation_info)

# 流式
generation_info = {}
gen = inference_stream_vllm(vllm_engine, template, request_list, generation_info=generation_info)
print_idx = 0
print(f'query: {query}\nresponse: ', end='')
# only show first
for resp_list in gen:
    resp = resp_list[0]
    if resp is None:
        continue
    response = resp['response']
    delta = response[print_idx:]
    print(delta, end='', flush=True)
    print_idx = len(response)
print()
print(generation_info)
"""
100%|██████████████████████████████████████████████████████████████████████████████| 100/100 [00:01<00:00, 91.47it/s]
100%|██████████████████████████████████████████████████████████████████████████████| 100/100 [00:22<00:00,  4.48it/s]
query: <image>描述这张图片
response: 这张图片展示了一只小猫咪的特写，可能是美国短毛猫品种，因为其花纹和毛发质地。猫咪有着引人注目的蓝色眼睛，这是其外貌中非常突出的特征。它皮毛上有着独特的黑色条纹，从面颊延伸至头顶，暗示着一种有条纹的花纹图案。它的耳朵小而尖，内侧是粉色的。猫咪的胡须细长而突出，围绕在它的下颌两侧和眼睛周围。猫咪坐着，用一种表达丰富的方式直视着，嘴巴微微张开，露出粉红色的内唇。背景模糊，柔和的光线增强了猫咪的特征。
{'num_prompt_tokens': 2700, 'num_generated_tokens': 14734, 'num_samples': 100, 'runtime': 23.53027338697575, 'samples/s': 4.249844375176322, 'tokens/s': 626.1720702384794}
query: <image>描述这张图片
response: 这张图片展示了一只小猫的特写，可能是一只幼年猫，在模糊的背景中，集中注意力在猫的表情上。这只猫长着一身白色与黑色条纹相间的毛皮，带有微妙的灰褐色。它的眼睛大而圆，具有高度的反光度，表明它们可能含有异色瞳，即一只眼睛是蓝色的，另一只是绿色的，但这只猫两只眼睛都是绿色的。睫毛清晰可见，增添了一种生动的表情。猫的耳朵竖立着，内部呈粉红色，边缘有浅色的阴影，显示出柔软的毛发。胡须又长又明显，突显了小猫的脸部形状。这个品种的猫看起来是一个常见品种，毛皮图案和眼睛颜色表明它可能是一只虎斑猫。光线柔和，产生一种天鹅绒般的效果，突出了猫绒毛的质感。
{'num_prompt_tokens': 2700, 'num_generated_tokens': 14986, 'num_samples': 100, 'runtime': 23.375922130944673, 'samples/s': 4.277906105257837, 'tokens/s': 641.0870089339394}
"""
```

使用CLI:
```shell
# 多模态模型必须显式指定`--infer_backend vllm`
CUDA_VISIBLE_DEVICES=0 swift infer --model_type llava1_6-vicuna-7b-instruct --infer_backend vllm

# 对数据集进行批量推理
CUDA_VISIBLE_DEVICES=0 swift infer --model_type llava1_6-vicuna-7b-instruct --infer_backend vllm \
    --val_dataset coco-en-2-mini#100

# TP:
CUDA_VISIBLE_DEVICES=0,1 swift infer --model_type internvl2-1b \
    --infer_backend vllm --tensor_parallel_size 2
```

```python
"""
<<< How many sheep are in the picture?
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png
There are four sheep in the picture.
--------------------------------------------------
<<< Perform OCR on the image.
Input a media path or URL <<< https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/ocr_en.png
The image contains text that appears to be an introduction or description of a software or service called SWIFT. Here is the transcribed text:

introduction
SWIFT supports training, inference, evaluation and deployment of 250+ LLMs and 35 MLMs (multimodal large models). Developers can directly apply their own research and production environments to realize the complete workflow from model training and evaluation to application. In addition, we provide a complete Adapters Library to support the latest training techniques such as PEFT, we also provide a Gradio web-ui for controlling training and inference, as well as accompanying deep learning courses and best practices for beginners.

Additionally, we are expanding capabilities for other modalities. Currently, we support full-paraphrase training and LORA training for AnimatedDiff.

SWIFT web-ui is available both on HuggingFace space and ModelScope studio.

Please feel free to try.

Please note that the text is a mix of English and what appears to be a programming or technical language, and some words or phrases might not be fully transcribed due to the complexity of the text.
--------------------------------------------------
<<< who are you
Input a media path or URL <<<
I'm a language model called Vicuna, and I was trained by researchers from Large Model Systems Organization (LMSYS).
"""
```


## 部署

**服务端:**
```shell
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type llava1_6-vicuna-13b-instruct --infer_backend vllm

# TP:
CUDA_VISIBLE_DEVICES=0,1 swift deploy --model_type internvl2-1b \
    --infer_backend vllm --tensor_parallel_size 2
```

**客户端:**

测试:
```bash
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "llava1_6-vicuna-13b-instruct",
"messages": [{"role": "user", "content": "Describe this image."}],
"temperature": 0,
"images": ["http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png"]
}'
```

使用ms-swift:
```python
import asyncio
from swift.llm import get_model_list_client, XRequestConfig, inference_client_async

model_list = get_model_list_client()
model_type = model_list.data[0].id
print(f'model_type: {model_type}')
request_config = XRequestConfig(seed=42)

query = '<image>Describe this image.'
images = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png']
tasks = [inference_client_async(model_type, query, images=images, request_config=request_config) for _ in range(100)]
async def _batch_run(tasks):
    return await asyncio.gather(*tasks)

resp_list = asyncio.run(_batch_run(tasks))
print(f'query: {query}')
print(f'response0: {resp_list[0].choices[0].message.content}')
print(f'response1: {resp_list[1].choices[0].message.content}')

query = '<image>How many sheep are in the picture?'
images = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png']

async def _stream():
    global query
    request_config = XRequestConfig(seed=42, stream=True)
    stream_resp = await inference_client_async(model_type, query, images=images, request_config=request_config)
    print(f'query: {query}')
    print('response: ', end='')
    async for chunk in stream_resp:
        print(chunk.choices[0].delta.content, end='', flush=True)
    print()

asyncio.run(_stream())
"""
model_type: llava1_6-vicuna-13b-instruct
query: <image>Describe this image.
response0: The image captures a moment of tranquility featuring a kitten. The kitten, with its fur a mix of gray and white, is the main subject of the image. It's sitting on a surface that appears to be a table or a similar flat surface. The kitten's eyes, a striking shade of blue, are wide open, giving it a curious and alert expression. Its ears, also gray and white, are perked up, suggesting it's attentive to its surroundings. The background is blurred, drawing focus to the kitten, and it's a soft, muted color that doesn't distract from the main subject. The overall image gives a sense of calm and innocence.
response1: The image captures a moment of tranquility featuring a kitten. The kitten, with its fur a mix of gray and white, is the main subject of the image. It's sitting on a surface that appears to be a table or a similar flat surface. The kitten's eyes, a striking shade of blue, are wide open, giving it a curious and alert expression. Its ears, also gray and white, are perked up, suggesting it's attentive to its surroundings. The background is blurred, drawing focus to the kitten, and it's a soft, muted color that doesn't distract from the main subject. The overall image gives a sense of calm and innocence.
query: <image>How many sheep are in the picture?
response: There are four sheep in the picture.
"""
```


使用openai:
```python
from openai import OpenAI
client = OpenAI(
    api_key='EMPTY',
    base_url='http://localhost:8000/v1',
)
model_type = client.models.list().data[0].id
print(f'model_type: {model_type}')

# use base64
# import base64
# with open('cat.png', 'rb') as f:
#     img_base64 = base64.b64encode(f.read()).decode('utf-8')
# image_url = f'data:image/jpeg;base64,{img_base64}'

# use local_path
# from swift.llm import convert_to_base64
# image_url = convert_to_base64(images=['cat.png'])['images'][0]
# image_url = f'data:image/jpeg;base64,{image_url}'

# use url
image_url = 'http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png'

query = 'Describe this image.'
messages = [{
    'role': 'user',
    'content': [
        {'type': 'image_url', 'image_url': {'url': image_url}},
        {'type': 'text', 'text': query},
    ]
}]

resp = client.chat.completions.create(
    model=model_type,
    messages=messages,
    temperature=0)
response = resp.choices[0].message.content
print(f'query: {query}')
print(f'response: {response}')

# 流式
query = 'How many sheep are in the picture?'
messages = [{
    'role': 'user',
    'content': [
        {'type': 'image_url', 'image_url': {'url': 'http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png'}},
        {'type': 'text', 'text': query},
    ]
}]
stream_resp = client.chat.completions.create(
    model=model_type,
    messages=messages,
    stream=True,
    temperature=0)

print(f'query: {query}')
print('response: ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].delta.content, end='', flush=True)
print()
"""
model_type: llava1_6-vicuna-13b-instruct
query: Describe this image.
response: The image captures a moment of tranquility featuring a kitten. The kitten, with its fur a mix of gray and white, is the main subject of the image. It's sitting on a surface that appears to be a table or a similar flat surface. The kitten's eyes, a striking shade of blue, are wide open, giving it a curious and alert expression. Its ears, also gray and white, are perked up, suggesting it's attentive to its surroundings. The background is blurred, drawing focus to the kitten, and it's a soft, muted color that doesn't distract from the main subject. The overall image gives a sense of calm and innocence.
query: How many sheep are in the picture?
response: There are four sheep in the picture.
"""
```

更多客户端使用方法可以查看[MLLM部署文档](MLLM部署文档.md#yi-vl-6b-chat)
