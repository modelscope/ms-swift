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

# There is a correspondence between lmdeploy and CUDA versions. Please follow the installation instructions at `https://github.com/InternLM/lmdeploy#installation`.
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

# ModelType.qwen_vl_chat, ModelType.deepseek_vl_1_3b_chat
# ModelType.internlm_xcomposer2_5_7b_chat, ModelType.minicpm_v_v2_5_chat
model_type = ModelType.internvl2_2b
lmdeploy_engine = get_lmdeploy_engine(model_type)
template_type = get_default_template_type(model_type)
template = get_template(template_type, lmdeploy_engine.hf_tokenizer)
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
request_list = [{'query': '<video>Describe the video.', 'videos': ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4']}]
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
print(generation_info)

"""
query: <image>Describe the image.
response: The image depicts four sheep standing in a grassy field against a backdrop of a gentle mountain and a slightly clouded sky. The sheep appear cute and friendly, with sheep faces that have large, friendly eyes and rosy cheeks. Each sheep has a unique coloration pattern; for instance, the sheep on the far left is predominantly white with brown wool around the snout and horns, while the other three have primarily white wool but with different color patterns on their snouts, tails, and horns. The overall mood of the image seems calm and serene.
query: who are you?
response: I am an AI assistant whose name is InternVL, developed jointly by Shanghai AI Lab and SenseTime.
query: <img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png</img><img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png</img>What is the difference bewteen the two images?
response: I'm unable to identify or compare images. However, if this image were to be classified based on its design or layout, it might demonstrate:

- A change in the display order or arrangement of elements within the image.
- An evolution in artistic style or technique.
- Different elements added or cut out to create a variation.

I'd need more specific details to make an accurate comparison.
{'num_prompt_tokens': 8099, 'num_generated_tokens': 212, 'num_samples': 3, 'runtime': 4.134621603996493, 'samples/s': 0.7255803039146855, 'tokens/s': 51.27434147663778}
query: <video>Describe the video.
response: The video features a young child sitting on a bed wearing a tank top and glasses. The child looks at some papers which are spread out in front of them. The child plays with the papers, taking off the glasses one eye at a time, and then puts them back on. After removing, reinserting, and replacing them, the child looks down and moves them around. The child continues to play with the papers and moves around them. The child seems to enjoy playing with the documents as they engage in this activity with the papers spread before them. The video portrays a sense of the child's curiosity and enthusiasm as they explore the objects around them. The child's interactions with the papers, with one eye and then one hand, show a playful yet methodical approach to engaging with the setting and materials.
{'num_prompt_tokens': 6250, 'num_generated_tokens': 164, 'num_samples': 1, 'runtime': 2.783833138004411, 'samples/s': 0.3592169323470477, 'tokens/s': 58.91157690491582}
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

**Server**:

```bash
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type deepseek-vl-1_3b-chat --infer_backend lmdeploy

CUDA_VISIBLE_DEVICES=0 swift deploy --model_type internvl2-2b --infer_backend lmdeploy

# TP
CUDA_VISIBLE_DEVICES=0,1 swift deploy --model_type qwen-vl-chat \
    --infer_backend lmdeploy --tp 2

CUDA_VISIBLE_DEVICES=0,1 swift deploy --model_type internlm-xcomposer2_5-7b-chat \
    --infer_backend lmdeploy --tp 2
```

**Client**:

This section introduces a demonstration of client calls to internvl2-2b:

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
# with open('baby.mp4', 'rb') as f:
#     vid_base64 = base64.b64encode(f.read()).decode('utf-8')
# video_url = f'data:video/mp4;base64,{vid_base64}'

# use local_path
# from swift.llm import convert_to_base64
# video_url = convert_to_base64(images=['baby.mp4'])['images'][0]
# video_url = f'data:video/mp4;base64,{video_url}'

# use url
video_url = 'https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4'

query = 'Describe this video.'
messages = [{
    'role': 'user',
    'content': [
        {'type': 'video_url', 'video_url': {'url': video_url}},
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

# Streaming
query = 'How many sheep are in the picture?'
image_url = 'http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png'
messages = [{
    'role': 'user',
    'content': [
        {'type': 'image_url', 'image_url': {'url': image_url}},
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
model_type: internvl2-2b
query: Describe this video.
response: The video features a young child, who appears to be a toddler, sitting on a bed and reading a book. The child is wearing a light blue shirt and dark glasses, and is engrossed in the book. The bed has a floral-patterned bedspread, and there is a white blanket on the bed. In the background, there is a wooden crib with a pink blanket and a white blanket on the bed. The room appears to be a bedroom, and there is a television on the wall, which is turned off. The child is holding the book with both hands and appears to be reading it with great interest. The child's face is illuminated by the light from the book, and the glasses reflect the light, making the child's eyes visible. The child's hair is light-colored, and it is neatly pulled back. The video captures the child's concentration and the peacefulness of the moment, as the child is absorbed in the book. The overall atmosphere of the video is calm and serene, with the child's focus on the book and the peacefulness of the room.
query: How many sheep are in the picture?
response: There are four sheep in the picture.
"""
```
The method for client invocation can be found in: [MLLM Deployment Documentation](mutlimodal-deployment.md).
