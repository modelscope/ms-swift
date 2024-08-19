# vLLM Inference Acceleration Documentation

ms-swift has integrated vLLM for accelerating inference of multimodal models. Check out the supported models in [Supported Models and Datasets Documentation](../LLM/Supported-models-datasets.md#MLLM).

## Table of Contents
- [Environment Setup](#environment-setup)
- [Inference Acceleration](#inference-acceleration)
- [Deployment](#deployment)

## Environment Setup
```bash
# Set pip global mirror (speeds up downloads)
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# Install ms-swift
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'

# vllm version corresponds to cuda version, please select version according to `https://docs.vllm.ai/en/latest/getting_started/installation.html`
pip install "vllm>=0.5.1"
pip install openai -U
```

## Inference Acceleration

Using python:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    ModelType, get_vllm_engine, get_default_template_type,
    get_template, inference_vllm
)

# 'minicpm-v-v2_5-chat', 'minicpm-v-v2_6-chat', 'internvl2-1b', 'internvl2-4b', 'phi3-vision-128k-instruct'
model_type = ModelType.llava1_6_mistral_7b_instruct
llm_engine = get_vllm_engine(model_type)
template_type = get_default_template_type(model_type)
template = get_template(template_type, llm_engine.hf_tokenizer)
# Interface similar to `transformers.GenerationConfig`
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

Batch processin:
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

query = '<image>Describe this image.'
images = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png']
generation_info = {}
request_list = [{'query': query, 'images': images} for _ in range(100)]
resp_list = inference_vllm(vllm_engine, template, request_list, generation_info=generation_info, use_tqdm=True)
print(f'query: {query}')
print(f'response: {resp_list[0]["response"]}')
print(generation_info)

# streaming
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
100%|███████████████████████████████████████████████████████████████████| 100/100 [00:05<00:00, 17.80it/s]
100%|███████████████████████████████████████████████████████████████████| 100/100 [00:22<00:00,  4.53it/s]
query: <image>Describe this image.
response: The image features a close-up of a kitten that appears to be a young domestic cat. Its large, expressive eyes are striking, and the fur pattern is a mix of striped and spotted markings, which is common in certain breeds like the Nebelung. Kitten's eyes are typically blue at birth, turning to their permanent color within four to six months. The kitten's curious and attentive gaze could suggest it is alert to its surroundings and possibly interested in something outside the frame of the image. The soft focus and warm lighting contribute to a cozy and inviting atmosphere, which is often associated with young animals and can invoke feelings of warmth and affection in viewers.
{'num_prompt_tokens': 2800, 'num_generated_tokens': 12569, 'num_samples': 100, 'runtime': 27.816649557033088, 'samples/s': 3.5949692573495526, 'tokens/s': 451.85168595626527}
query: <image>Describe this image.
response: The image features a close-up of a kitten, likely a young maine coon, characterized by its distinctive facial markings and large, expressive eyes. Maine coons are known for their robust stature and friendly demeanor, traits that this kitten also seems to exhibit. The blurred background suggests that the focus is entirely on the kitten, enhancing its cuteness and making it the central subject of the photograph. This kind of image is often used to elicit feelings of affection and to highlight the charm and innocence of young animals. It's a simple yet powerful image that could be used for themes such as pet adoption, animal welfare, or simply as an adorable piece for pet enthusiasts.
{'num_prompt_tokens': 2800, 'num_generated_tokens': 12275, 'num_samples': 100, 'runtime': 40.04483833198901, 'samples/s': 2.4972007421020606, 'tokens/s': 306.53139109302793}
"""
```


Using CLI:
```shell
# Multimodal models must explicitly specify `--infer_backend vllm`.
CUDA_VISIBLE_DEVICES=0 swift infer --model_type llava1_6-vicuna-7b-instruct --infer_backend vllm

# Batch inference on the dataset
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


## Deployment

**Server**:
```shell
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type llava1_6-vicuna-13b-instruct --infer_backend vllm

# TP:
CUDA_VISIBLE_DEVICES=0,1 swift deploy --model_type internvl2-1b \
    --infer_backend vllm --tensor_parallel_size 2
```

**Client**:

Test:
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

Using ms-swift:
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

Using OpenAI
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

You can check out more client usage methods in the [MLLM Deployment Documentation](mutlimodal-deployment.md#yi-vl-6b-chat).
