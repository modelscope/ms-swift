# Mutlimoda LLM Deployment
For inference acceleration and deployment of MLLM, you can refer to the [LmDeploy Inference Acceleration Documentation](LmDeploy-inference-acceleration.md) and the [vLLM Inference Acceleration Documentation](vllm-inference-acceleration.md).

## Table of Contents
- [Environment Setup](#environment-setup)
- [qwen-vl-chat](#qwen-vl-chat)
- [yi-vl-6b-chat](#yi-vl-6b-chat)
- [minicpm-v-v2_5-chat](#minicpm-v-v2_5-chat)
- [Audio and Video Modalities](#Audio-and-Video-Modalities)

## Environment Setup
```shell
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'
```

Here are several examples of models (we chose smaller-sized models for convenience in experimentation). I believe you can find the patterns for deployment and invocation, so I won't elaborate further.

## qwen-vl-chat

**Server**:
```bash
# Using the original model
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen-vl-chat

# Using the fine-tuned LoRA
CUDA_VISIBLE_DEVICES=0 swift deploy --ckpt_dir output/qwen-vl-chat/vx-xxx/checkpoint-xxx

# Using the fine-tuned Merge LoRA model
CUDA_VISIBLE_DEVICES=0 swift deploy --ckpt_dir output/qwen-vl-chat/vx-xxx/checkpoint-xxx-merged
```

**Client**:

Test:
```bash
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "qwen-vl-chat",
"messages": [{"role": "user", "content": "<img>https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/rose.jpg</img>What kind of flower is in the picture and how many are there?"}],
"max_tokens": 256,
"temperature": 0
}'
```

Using swift:
```python
from swift.llm import get_model_list_client, XRequestConfig, inference_client

model_list = get_model_list_client()
model_type = model_list.data[0].id
print(f'model_type: {model_type}')

# use base64
# import base64
# with open('rose.jpg', 'rb') as f:
#     img_base64 = base64.b64encode(f.read()).decode('utf-8')
# query = f'<img>{img_base64}</img>What kind of flower is in the picture and how many are there?'

# use local_path
# query = '<img>rose.jpg</img>What kind of flower is in the picture and how many are there?'

# use url
query = '<img>https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/rose.jpg</img>What kind of flower is in the picture and how many are there?'

request_config = XRequestConfig(seed=42)
resp = inference_client(model_type, query, request_config=request_config)
response = resp.choices[0].message.content
print(f'query: {query}')
print(f'response: {response}')

history = [(query, response)]
query = 'Box out the flowers in the picture.'
request_config = XRequestConfig(stream=True, seed=42)
stream_resp = inference_client(model_type, query, history, request_config=request_config)
print(f'query: {query}')
print('response: ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].delta.content, end='', flush=True)
print()

"""
model_type: qwen-vl-chat
query: <img>https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/rose.jpg</img>What kind of flower is in the picture and how many are there?
response: There are three roses in the picture.
query: Box out the flowers in the picture.
response: <ref> flowers</ref><box>(33,448),(360,979)</box>
"""
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
# with open('rose.jpg', 'rb') as f:
#     img_base64 = base64.b64encode(f.read()).decode('utf-8')
# image_url = f'data:image/jpeg;base64,{img_base64}'

# use local_path
# from swift.llm import convert_to_base64
# image_url = convert_to_base64(images=['rose.jpg'])['images'][0]
# image_url = f'data:image/jpeg;base64,{image_url}'

# use url
image_url = 'https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/rose.jpg'

query = 'What kind of flower is in the picture and how many are there?'
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
    seed=42)
response = resp.choices[0].message.content
print(f'query: {query}')
print(f'response: {response}')

# Streaming
messages.append({'role': 'assistant', 'content': response})
query = 'Box out the flowers in the picture.'
messages.append({'role': 'user', 'content': query})
stream_resp = client.chat.completions.create(
    model=model_type,
    messages=messages,
    stream=True,
    seed=42)

print(f'query: {query}')
print('response: ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].delta.content, end='', flush=True)
print()

"""Out[0]
model_type: qwen-vl-chat
query: What kind of flower is in the picture and how many are there?
response: There are three roses in the picture.
query: Box out the flowers in the picture.
response: <ref> flowers</ref><box>(33,448),(360,979)</box>
"""
```

## yi-vl-6b-chat

**Server side:**
```bash
# Using the original model
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type yi-vl-6b-chat

# Using the fine-tuned LoRA
CUDA_VISIBLE_DEVICES=0 swift deploy --ckpt_dir output/yi-vl-6b-chat/vx-xxx/checkpoint-xxx

# Using the fine-tuned Merge LoRA model
CUDA_VISIBLE_DEVICES=0 swift deploy --ckpt_dir output/yi-vl-6b-chat/vx-xxx/checkpoint-xxx-merged
```

**Client side:**

Test:
```bash
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "yi-vl-6b-chat",
"messages": [{"role": "user", "content": "Describe this image."}],
"seed": 42,
"images": ["http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png"]
}'
```

Using swift:
```python
from swift.llm import get_model_list_client, XRequestConfig, inference_client

model_list = get_model_list_client()
model_type = model_list.data[0].id
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

query = '<image>Describe this image.'
request_config = XRequestConfig(seed=42)
resp = inference_client(model_type, query, images=images, request_config=request_config)
response = resp.choices[0].message.content
print(f'query: {query}')
print(f'response: {response}')

history = [(query, response)]
query = '<image>How many sheep are in the picture?'
images.append('http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png')
request_config = XRequestConfig(stream=True, seed=42)
stream_resp = inference_client(model_type, query, history, images=images, request_config=request_config)
print(f'query: {query}')
print('response: ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].delta.content, end='', flush=True)
print()

"""
model_type: yi-vl-6b-chat
query: <image>Describe this image.
response: The image captures a moment of tranquility featuring a gray and white kitten. The kitten, with its eyes wide open, is the main subject of the image. Its nose is pink, adding a touch of color to its gray and white fur. The kitten is sitting on a white surface, which contrasts with its gray and white fur. The background is blurred, drawing focus to the kitten. The image does not contain any text. The kitten's position relative to the background suggests it is in the foreground of the image. The image does not contain any other objects or creatures. The kitten appears to be alone in the image. The image does not contain any action, but the kitten's wide-open eyes give it a curious and alert appearance. The image does not contain any aesthetic descriptions. The image is a simple yet captivating portrait of a gray and white kitten.
query: <image>How many sheep are in the picture?
response: There are four sheep in the picture.
"""
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
    seed=42)
response = resp.choices[0].message.content
print(f'query: {query}')
print(f'response: {response}')

# Streaming
messages.append({'role': 'assistant', 'content': response})
query = 'How many sheep are in the picture?'
messages.append({'role': 'user', 'content': [
    {'type': 'image_url', 'image_url': {'url': 'http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png'}},
    {'type': 'text', 'text': query},
]})
stream_resp = client.chat.completions.create(
    model=model_type,
    messages=messages,
    stream=True,
    seed=42)

print(f'query: {query}')
print('response: ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].delta.content, end='', flush=True)
print()

"""
model_type: yi-vl-6b-chat
query: Describe this image.
response: The image captures a moment of tranquility featuring a gray and white kitten. The kitten, with its eyes wide open, is the main subject of the image. Its nose is pink, adding a touch of color to its gray and white fur. The kitten is sitting on a white surface, which contrasts with its gray and white fur. The background is blurred, drawing focus to the kitten. The image does not contain any text. The kitten's position relative to the background suggests it is in the foreground of the image. The image does not contain any other objects or creatures. The kitten appears to be alone in the image. The image does not contain any action, but the kitten's wide-open eyes give it a curious and alert appearance. The image does not contain any aesthetic descriptions. The image is a simple yet captivating portrait of a gray and white kitten.
query: How many sheep are in the picture?
response: There are four sheep in the picture.
"""
```

## minicpm-v-v2_5-chat

**Server side:**
```bash
# Using the original model
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type minicpm-v-v2_5-chat

# Using the fine-tuned LoRA
CUDA_VISIBLE_DEVICES=0 swift deploy --ckpt_dir output/minicpm-v-v2_5-chat/vx-xxx/checkpoint-xxx

# Using the fine-tuned Merge LoRA model
CUDA_VISIBLE_DEVICES=0 swift deploy --ckpt_dir output/minicpm-v-v2_5-chat/vx-xxx/checkpoint-xxx-merged
```

**Client side:**

Test:
```bash
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "minicpm-v-v2_5-chat",
"messages": [{"role": "user", "content": "Describe this image."}],
"temperature": 0,
"images": ["http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png"]
}'
```

Using swift:
```python
from swift.llm import get_model_list_client, XRequestConfig, inference_client

model_list = get_model_list_client()
model_type = model_list.data[0].id
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

query = '<image>Describe this image.'
request_config = XRequestConfig(temperature=0)
resp = inference_client(model_type, query, images=images, request_config=request_config)
response = resp.choices[0].message.content
print(f'query: {query}')
print(f'response: {response}')

history = [(query, response)]
query = 'How was this picture generated?'
request_config = XRequestConfig(stream=True, temperature=0)
stream_resp = inference_client(model_type, query, history, images=images, request_config=request_config)
print(f'query: {query}')
print('response: ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].delta.content, end='', flush=True)
print()

"""
model_type: minicpm-v-v2_5-chat
query: <image>Describe this image.
response: The image is a digital painting of a kitten, which is the main subject. The kitten's fur is rendered with a mix of gray, black, and white, giving it a realistic appearance. Its eyes are wide open, and the expression is one of curiosity or alertness. The background is blurred, which brings the focus entirely on the kitten. The painting style is detailed and lifelike, capturing the essence of a young feline's innocent and playful nature. The image does not convey any specific context or background story beyond the depiction of the kitten itself.
query: How was this picture generated?
response: This picture was generated using digital art techniques. The artist likely used a software program to create the image, manipulating pixels and colors to achieve the detailed and lifelike representation of the kitten. Digital art allows for a high degree of control over the final product, enabling artists to create intricate details and textures that might be difficult to achieve with traditional media.
"""
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

# Streaming
messages.append({'role': 'assistant', 'content': response})
query = 'How was this picture generated?'
messages.append({'role': 'user', 'content': query})
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
model_type: minicpm-v-v2_5-chat
query: Describe this image.
response: The image is a digital painting of a kitten, which is the main subject. The kitten's fur is rendered with a mix of gray, black, and white, giving it a realistic appearance. Its eyes are wide open, and the expression is one of curiosity or alertness. The background is blurred, which brings the focus entirely on the kitten. The painting style is detailed and lifelike, capturing the essence of a young feline's innocent and playful nature. The image does not convey any specific context or background story beyond the depiction of the kitten itself.
query: How was this picture generated?
response: This picture was generated using digital art techniques. The artist likely used a software program to create the image, manipulating pixels and colors to achieve the detailed and lifelike representation of the kitten. Digital art allows for a high degree of control over the final product, enabling artists to create intricate details and textures that might be difficult to achieve with traditional media.
"""
```

## Audio and Video Modalities

### qwen2-audio-7b-instruct

**Server**:
```bash
# pip install transformers>=4.45
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen2-audio-7b-instruct
# ...
```

**Client**:

Using swift:
```python
from swift.llm import get_model_list_client, XRequestConfig, inference_client

model_list = get_model_list_client()
model_type = model_list.data[0].id
print(f'model_type: {model_type}')

# use base64
# import base64
# with open('weather.wav', 'rb') as f:
#     aud_base64 = base64.b64encode(f.read()).decode('utf-8')
# audios = [aud_base64]

# use local_path
# from swift.llm import convert_to_base64
# audios = ['weather.wav']
# audios = convert_to_base64(images=audios)['images']

# use url
audios = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/weather.wav']

query = '<audio>What did this speech say'
request_config = XRequestConfig(temperature=0)
resp = inference_client(model_type, query, audios=audios, request_config=request_config)
response = resp.choices[0].message.content
print(f'query: {query}')
print(f'response: {response}')

history = [(query, response)]
query = 'The gender of this speech is male.'
request_config = XRequestConfig(stream=True, temperature=0)
stream_resp = inference_client(model_type, query, history, audios=audios, request_config=request_config)
print(f'query: {query}')
print('response: ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].delta.content, end='', flush=True)
print()

"""
model_type: qwen2-audio-7b-instruct
query: <audio>What did this speech say
response: The original content of this audio is: '今天天气真好呀'
query: The gender of this speech is male.
response: The speaker is male.
"""
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
# with open('weather.wav', 'rb') as f:
#     aud_base64 = base64.b64encode(f.read()).decode('utf-8')
# audio_url = f'data:audio/wav;base64,{aud_base64}'

# use local_path
# from swift.llm import convert_to_base64
# audio_url = convert_to_base64(images=['weather.wav'])['images'][0]
# audio_url = f'data:audio/wav;base64,{audio_url}'

# use url
audio_url = 'http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/weather.wav'

query = 'What did this speech say'
messages = [{
    'role': 'user',
    'content': [
        {'type': 'audio_url', 'audio_url': {'url': audio_url}},
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
messages.append({'role': 'assistant', 'content': response})
query = 'The gender of this speech is male.'
messages.append({'role': 'user', 'content': query})
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
model_type: qwen2-audio-7b-instruct
query: What did this speech say
response: The original content of this audio is: '今天天气真好呀'
query: The gender of this speech is male.
response: The speaker is male.
"""
```

### internvl2-2b

**Server**:
```bash
# or 'minicpm-v-v2_6-chat'
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type internvl2-2b
# ...
```

**Client**:

Using swift:
```python
from swift.llm import get_model_list_client, XRequestConfig, inference_client

model_list = get_model_list_client()
model_type = model_list.data[0].id
print(f'model_type: {model_type}')

# use base64
# import base64
# with open('baby.mp4', 'rb') as f:
#     vid_base64 = base64.b64encode(f.read()).decode('utf-8')
# videos = [vid_base64]

# use local_path
# from swift.llm import convert_to_base64
# videos = ['baby.mp4']
# videos = convert_to_base64(images=videos)['images']

# use url
videos = ['https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4']

query = '<video>Describe this video.'
request_config = XRequestConfig(temperature=0)
resp = inference_client(model_type, query, videos=videos, request_config=request_config)
response = resp.choices[0].message.content
print(f'query: {query}')
print(f'response: {response}')

images = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png']
query = '<image>How many sheep are in the picture?'
request_config = XRequestConfig(stream=True, temperature=0)
stream_resp = inference_client(model_type, query, images=images, request_config=request_config)
print(f'query: {query}')
print('response: ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].delta.content, end='', flush=True)
print()
"""
model_type: internvl2-2b
query: <video>Describe this video.
response:  The video features a young child, who appears to be a toddler, sitting on a bed and reading a book. The child is wearing a light blue shirt and dark glasses, and is engrossed in the book. The child's attention is focused on the pages, and they seem to be enjoying the story. The bed has a floral patterned cover, and there is a white blanket on the bed. In the background, there is a wooden crib with a white sheet and a few other items, including a white towel and a black and white striped garment. The room appears to be a bedroom, and there is a window visible in the background. The child's hair is light-colored, and they are wearing a pair of dark-framed glasses. The video captures the child's peaceful and focused demeanor as they read the book. The overall atmosphere of the video is calm and serene, with the child's concentration on the book being the main focus.
query: <image>How many sheep are in the picture?
response: There are four sheep in the picture.
"""
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
response:  The video features a young child, who appears to be a toddler, sitting on a bed and reading a book. The child is wearing a light blue shirt and dark glasses, and is engrossed in the book. The child's attention is focused on the pages, and they seem to be enjoying the story. The bed has a floral patterned cover, and there is a white blanket on the bed. In the background, there is a wooden crib with a white sheet and a few other items, including a white towel and a black and white striped garment. The room appears to be a bedroom, and there is a window visible in the background. The child's hair is light-colored, and they are wearing a pair of dark-framed glasses. The video captures the child's peaceful and focused demeanor as they read the book. The overall atmosphere of the video is calm and serene, with the child's concentration on the book being the main focus.
query: How many sheep are in the picture?
response: There are four sheep in the picture.
"""
```
