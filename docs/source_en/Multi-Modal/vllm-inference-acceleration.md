# vLLM Inference Acceleration Documentation

ms-swift has integrated vLLM for accelerating inference of multimodal models. Check out the supported models in [Supported Models and Datasets Documentation](../LLM/Supported-models-datasets.md). It's worth noting that using vLLM for acceleration may have a slight impact on the inference quality, so please ensure you can tolerate this loss in order to boost the inference speed.

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
pip install "vllm>=0.5"
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

Using CLI:
```shell
# Multimodal models must explicitly specify `--infer_backend vllm`.
CUDA_VISIBLE_DEVICES=0 swift infer --model_type llava1_6-vicuna-7b-instruct --infer_backend vllm

# Batch inference on the dataset
CUDA_VISIBLE_DEVICES=0 swift infer --model_type llava1_6-vicuna-7b-instruct --infer_backend vllm \
    --val_dataset coco-en-2-mini#100
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

### Llava Series

**Server**:
```shell
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type llava1_6-vicuna-13b-instruct --infer_backend vllm
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
# images = [img_base64]

# use local_path
# from swift.llm import convert_to_base64
# images = ['cat.png']
# images = convert_to_base64(images=images)['images']

# use url
images = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png']

query = 'Describe this image.'
messages = [{
    'role': 'user',
    'content': query
}]
resp = client.chat.completions.create(
    model=model_type,
    messages=messages,
    temperature=0,
    extra_body={'images': images})
response = resp.choices[0].message.content
print(f'query: {query}')
print(f'response: {response}')

# Streaming
images = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png']
query = 'How many sheep are in the picture?'
messages = [{
    'role': 'user',
    'content': query
}]
stream_resp = client.chat.completions.create(
    model=model_type,
    messages=messages,
    stream=True,
    temperature=0,
    extra_body={'images': images})

print(f'query: {query}')
print('response: ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].delta.content, end='', flush=True)
print()
"""
model_type: llava1_6-vicuna-13b-instruct
query: Describe this image.
response: The image shows a close-up of a person's hands playing a guitar. The hands are positioned on the neck of the guitar, with the fingers pressing down on the strings to produce music. The guitar has a wooden body and a glossy finish, and the strings are clearly visible. The background is blurred, but it appears to be an indoor setting with warm lighting. The focus of the image is on the hands and the guitar, with the background serving to highlight the subject. There are no visible texts or distinctive brands in the image. The style of the image is a realistic photograph with a shallow depth of field, which is a common technique in portrait photography to draw attention to the subject.
query: How many sheep are in the picture?
response: There are two sheep in the picture.
"""
```

You can check out more client usage methods in the [MLLM Deployment Documentation](mutlimodal-deployment.md#yi-vl-6b-chat).

### phi3-vision

**Server**:
```shell
# vllm>=0.5.1 or build from source
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type phi3-vision-128k-instruct --infer_backend vllm --max_model_len 8192
```

**Client**:

Test:
```bash
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "phi3-vision-128k-instruct",
"messages": [{"role": "user", "content": "<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png</img><img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png</img>What is the difference between these two pictures?"}],
"temperature": 0
}'
```

Using openai:
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
# images = [img_base64]

# use local_path
# from swift.llm import convert_to_base64
# images = ['cat.png']
# images = convert_to_base64(images=images)['images']

# use url

query = '<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png</img><img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png</img>What is the difference between these two pictures?'
messages = [{
    'role': 'user',
    'content': query
}]
resp = client.chat.completions.create(
    model=model_type,
    messages=messages,
    temperature=0)
response = resp.choices[0].message.content
print(f'query: {query}')
print(f'response: {response}')

# Streaming
query = 'How many sheep are in the picture?'
messages = [{
    'role': 'user',
    'content': query
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
model_type: phi3-vision-128k-instruct
query: <img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png</img><img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png</img>What is the difference between these two pictures?
response: The first picture shows a group of four sheep standing in a field, while the second picture is a close-up of a kitten with big eyes. The main difference between these two pictures is the subjects and the setting. The first image features animals typically found in a pastoral or rural environment, whereas the second image focuses on a small domestic animal, a kitten, which is usually found indoors. Additionally, the first picture has a more peaceful and serene atmosphere, while the second image has a more intimate and detailed view of the kitten.
query: How many sheep are in the picture?
response: There are three sheep in the picture.
"""
```
