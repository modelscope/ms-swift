# LmDeploy Inference Acceleration and Deployment
lmdeploy github: [https://github.com/InternLM/lmdeploy](https://github.com/InternLM/lmdeploy).

MLLM that support inference acceleration using lmdeploy can be found at [Supported Models](../LLM/Supported-models-datasets.md#MLLM).

## Table of Contents
- [Environment Preparation](#environment-preparation)
- [Inference Acceleration](#inference-acceleration)
- [Deployment](#deployment)

## Environment Preparation
GPU devices: A10, 3090, V100, A100 are all supported.
```bash
# Set pip global mirror (speeds up downloads)
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# Install ms-swift
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'

pip install lmdeploy
```

## Inference Acceleration

### Using Python

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# from swift.hub import HubApi
# _api = HubApi()
# _api.login('<your-sdk-token>')  # https://modelscope.cn/my/myaccesstoken

from swift.llm import (
    ModelType, get_lmdeploy_engine, get_default_template_type,
    get_template, inference_lmdeploy, inference_stream_lmdeploy
)

model_type = ModelType.internvl2_2b
lmdeploy_engine = get_lmdeploy_engine(model_type)
template_type = get_default_template_type(model_type)
template = get_template(template_type, lmdeploy_engine.hf_tokenizer)
# An interface similar to transformers.GenerationConfig
lmdeploy_engine.generation_config.max_new_tokens = 256
generation_info = {}

request_list = [{'query': '<image>Describe the image.', 'images': ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png']},
                {'query': 'who are you?'},
                {'query': (
                    '<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png</img>'
                    '<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png</img>'
                    'What is the difference bewteen the two images?'
                )}]
resp_list = inference_lmdeploy(lmdeploy_engine, template, request_list, generation_info=generation_info)
for request, resp in zip(request_list, resp_list):
    print(f"query: {request['query']}")
    print(f"response: {resp['response']}")
print(generation_info)

# stream
history0 = resp_list[0]['history']
request_list = [{'query': 'How many sheep are there?', 'history': history0, 'images': ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png']}]
gen = inference_stream_lmdeploy(lmdeploy_engine, template, request_list, generation_info=generation_info)
query = request_list[0]['query']
print_idx = 0
print(f'query: {query}\nresponse: ', end='')
for resp_list in gen:
    resp = resp_list[0]
    response = resp['response']
    delta = response[print_idx:]
    print(delta, end='', flush=True)
    print_idx = len(response)
print()

history = resp_list[0]['history']
print(f'history: {history}')
print(generation_info)
"""
query: <image>Describe the image.
response: The image features four sheep standing in a grassy meadow against a backdrop of mountains. The animals are lined up straight, with their large, floppy ears, white, woolly coats, and big, expressive black eyes. The background includes a sky with some fluffy clouds and subtle shades of green and blue. Each sheep has a different facial expression and hat, adding a playful and friendly touch to the overall scene.
query: who are you?
response: I am an AI assistant whose name is InternVL, developed jointly by Shanghai AI Lab and SenseTime.
query: <img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png</img><img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png</img>What is the difference bewteen the two images?
response: In the first image, the sheep are standing in front of lush green mountains. In the second image, some of their wool is dyed green instead of white.
{'num_prompt_tokens': 8090, 'num_generated_tokens': 140, 'num_samples': 3, 'runtime': 1.55301758996211, 'samples/s': 1.9317231301116127, 'tokens/s': 90.14707940520859}
query: How many sheep are there?
response: There are four sheep in the image.
history: [['<image>Describe the image.', 'The image features four sheep standing in a grassy meadow against a backdrop of mountains. The animals are lined up straight, with their large, floppy ears, white, woolly coats, and big, expressive black eyes. The background includes a sky with some fluffy clouds and subtle shades of green and blue. Each sheep has a different facial expression and hat, adding a playful and friendly touch to the overall scene.'], ['How many sheep are there?', 'There are four sheep in the image.']]
{'num_prompt_tokens': 3479, 'num_generated_tokens': 8, 'num_samples': 1, 'runtime': 0.6162854079157114, 'samples/s': 1.6226248214800645, 'tokens/s': 12.980998571840516}
"""
```


**TP:**

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

from swift.llm import (
    ModelType, get_lmdeploy_engine, get_default_template_type,
    get_template, inference_lmdeploy, inference_stream_lmdeploy
)

if __name__ == '__main__':
    model_type = ModelType.glm4v_9b_chat
    lmdeploy_engine = get_lmdeploy_engine(model_type, tp=2)
    template_type = get_default_template_type(model_type)
    template = get_template(template_type, lmdeploy_engine.hf_tokenizer)
    # An interface similar to transformers.GenerationConfig
    lmdeploy_engine.generation_config.max_new_tokens = 256
    generation_info = {}

    request_list = [{'query': '<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png</img>Describe the image.'},
                    {'query': '<image>Describe the image.', 'images': ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png']},
                    {'query': 'who are you?'}]
    resp_list = inference_lmdeploy(lmdeploy_engine, template, request_list, generation_info=generation_info)
    for request, resp in zip(request_list, resp_list):
        print(f"query: {request['query']}")
        print(f"response: {resp['response']}")
    print(generation_info)

    # stream
    history0 = resp_list[0]['history']
    request_list = [{'query': 'How many sheep are there?', 'history': history0}]
    gen = inference_stream_lmdeploy(lmdeploy_engine, template, request_list, generation_info=generation_info)
    query = request_list[0]['query']
    print_idx = 0
    print(f'query: {query}\nresponse: ', end='')
    for resp_list in gen:
        resp = resp_list[0]
        response = resp['response']
        delta = response[print_idx:]
        print(delta, end='', flush=True)
        print_idx = len(response)
    print()

    history = resp_list[0]['history']
    print(f'history: {history}')
    print(generation_info)
"""
query: <img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png</img>Describe the image.
response: The image features a charming illustration of a group of sheep in a pastoral setting. The sheep are depicted with a friendly and somewhat cartoonish design, with their fluffy wool suggesting they are well-cared for animals. They are standing on a lush green field, with the grass appearing soft and inviting, and there are hints of yellow wildflowers sprinkled throughout, adding to the idyllic scene.

The sheep are positioned in a way that suggests a family or group dynamic. They are standing in a row, with the sheep on the left appearing to be of smaller stature, likely indicating they are younger, and the sheep on the right have a more mature appearance. The sheep in the middle of the image is the most prominent and has a fluffy, puffy white wool, which stands out against the green background, giving it a sense of importance or leadership within the group.

The background of the image is a tranquil and pastoral landscape. There are gentle undulations of hills that suggest a meadow, and the hills are a rich shade of green, blending into the horizon where the sky meets the earth. The sky is a soft, clear blue, with a few wispy, light clouds scattered across it, contributing to the peaceful atmosphere of the scene. The sunlight appears to be coming
query: <image>Describe the image.
response: The image features a charming illustration of a group of sheep in a pastoral setting. The sheep are depicted with a friendly and somewhat cartoonish design, with their fluffy wool suggesting they are well-cared for animals. They are standing on a lush green field, with the grass appearing soft and inviting, and there are hints of yellow wildflowers sprinkled throughout, adding to the idyllic scene.

The sheep are positioned in a way that suggests a family or group dynamic. They are standing in a row, with the sheep on the left appearing to be of smaller stature, likely indicating they are younger, and the sheep on the right have a more mature appearance. The sheep in the middle of the image is the most prominent and has a fluffy, puffy white wool, which stands out against the green background, giving it a sense of importance or leadership within the group.

The background of the image is a tranquil and pastoral landscape. There are gentle undulations of hills that suggest a meadow, and the hills are a rich shade of green, blending into the horizon where the sky meets the earth. The sky is a soft, clear blue, with a few wispy, light clouds scattered across it, contributing to the peaceful atmosphere of the scene. The sunlight appears to be coming
query: who are you?
response: I am an AI assistant named ChatGLM（智谱清言）, which is developed based on the language model trained by Zhipu AI in 2023. My job is to provide appropriate answers and support to users' questions and requests.
{'num_prompt_tokens': 3231, 'num_generated_tokens': 563, 'num_samples': 3, 'runtime': 14.152525326004252, 'samples/s': 0.21197630323174302, 'tokens/s': 39.78088623982377}
query: How many sheep are there?
response: There are four sheep in the image. From left to right, the first sheep has a smaller body and wool, the second one is larger with a fluffy wool, the third one also appears to have a fluffy wool, and the last sheep on the right has a similar fluffy appearance as the second one. Each sheep has a unique expression and stance, which gives the image a sense of liveliness and individuality.
history: [['<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png</img>Describe the image.', 'The image features a charming illustration of a group of sheep in a pastoral setting. The sheep are depicted with a friendly and somewhat cartoonish design, with their fluffy wool suggesting they are well-cared for animals. They are standing on a lush green field, with the grass appearing soft and inviting, and there are hints of yellow wildflowers sprinkled throughout, adding to the idyllic scene.\n\nThe sheep are positioned in a way that suggests a family or group dynamic. They are standing in a row, with the sheep on the left appearing to be of smaller stature, likely indicating they are younger, and the sheep on the right have a more mature appearance. The sheep in the middle of the image is the most prominent and has a fluffy, puffy white wool, which stands out against the green background, giving it a sense of importance or leadership within the group.\n\nThe background of the image is a tranquil and pastoral landscape. There are gentle undulations of hills that suggest a meadow, and the hills are a rich shade of green, blending into the horizon where the sky meets the earth. The sky is a soft, clear blue, with a few wispy, light clouds scattered across it, contributing to the peaceful atmosphere of the scene. The sunlight appears to be coming'], ['How many sheep are there?', 'There are four sheep in the image. From left to right, the first sheep has a smaller body and wool, the second one is larger with a fluffy wool, the third one also appears to have a fluffy wool, and the last sheep on the right has a similar fluffy appearance as the second one. Each sheep has a unique expression and stance, which gives the image a sense of liveliness and individuality.']]
{'num_prompt_tokens': 1876, 'num_generated_tokens': 83, 'num_samples': 1, 'runtime': 4.516964272013865, 'samples/s': 0.22138762668453765, 'tokens/s': 18.375173014816625}
"""
```

### Using CLI
```bash
CUDA_VISIBLE_DEVICES=0 swift infer --model_type deepseek-vl-1_3b-chat --infer_backend lmdeploy

CUDA_VISIBLE_DEVICES=0 swift infer --model_type internvl2-2b --infer_backend lmdeploy

# TP
CUDA_VISIBLE_DEVICES=0,1 swift infer --model_type qwen-vl-chat \
    --infer_backend lmdeploy --tp 2

CUDA_VISIBLE_DEVICES=0,1 swift infer --model_type internlm-xcomposer2_5-7b-chat \
    --infer_backend lmdeploy --tp 2
```

## Deployment
```bash
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type deepseek-vl-1_3b-chat --infer_backend lmdeploy

CUDA_VISIBLE_DEVICES=0 swift deploy --model_type internvl2-2b --infer_backend lmdeploy

# TP
CUDA_VISIBLE_DEVICES=0,1 swift deploy --model_type qwen-vl-chat \
    --infer_backend lmdeploy --tp 2

CUDA_VISIBLE_DEVICES=0,1 swift deploy --model_type internlm-xcomposer2_5-7b-chat \
    --infer_backend lmdeploy --tp 2
```

The method for client invocation can be found in: [MLLM Deployment Documentation](mutlimodal-deployment.md), [vLLM Inference Acceleration Documentation](vllm-inference-acceleration.md#deployment).
