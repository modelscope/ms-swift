# MLLM部署文档
对MLLM进行推理加速和部署可以查看[lmdeploy推理加速文档](LmDeploy推理加速文档.md)和[vLLM推理加速文档](vLLM推理加速文档.md).

## 目录
- [环境准备](#环境准备)
- [qwen-vl-chat](#qwen-vl-chat)
- [yi-vl-6b-chat](#yi-vl-6b-chat)
- [minicpm-v-v2_5-chat](#minicpm-v-v2_5-chat)
- [语音与视频模态](#语音与视频模态)

## 环境准备
```shell
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'
```

以下我们给出了若干模型的例子（选择了尺寸较小的模型来方便实验），相信聪明的你可以从中找到部署与调用的规律，我就不多介绍啦。

## qwen-vl-chat

**服务端:**
```bash
# 使用原始模型
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen-vl-chat

# 使用微调后的LoRA
CUDA_VISIBLE_DEVICES=0 swift deploy --ckpt_dir output/qwen-vl-chat/vx-xxx/checkpoint-xxx

# 使用微调后Merge LoRA的模型
CUDA_VISIBLE_DEVICES=0 swift deploy --ckpt_dir output/qwen-vl-chat/vx-xxx/checkpoint-xxx-merged
```

**客户端:**

测试:
```bash
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "qwen-vl-chat",
"messages": [{"role": "user", "content": "<img>https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/rose.jpg</img>图中是什么花，有几只？"}],
"max_tokens": 256,
"temperature": 0
}'
```

使用swift:
```python
from swift.llm import get_model_list_client, XRequestConfig, inference_client

model_list = get_model_list_client()
model_type = model_list.data[0].id
print(f'model_type: {model_type}')

# use base64
# import base64
# with open('rose.jpg', 'rb') as f:
#     img_base64 = base64.b64encode(f.read()).decode('utf-8')
# query = f'<img>{img_base64}</img>图中是什么花，有几只？'

# use local_path
# query = '<img>rose.jpg</img>图中是什么花，有几只？'

# use url
query = '<img>https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/rose.jpg</img>图中是什么花，有几只？'

request_config = XRequestConfig(seed=42)
resp = inference_client(model_type, query, request_config=request_config)
response = resp.choices[0].message.content
print(f'query: {query}')
print(f'response: {response}')

history = [(query, response)]
query = '框出图中的花'
request_config = XRequestConfig(stream=True, seed=42)
stream_resp = inference_client(model_type, query, history, request_config=request_config)
print(f'query: {query}')
print('response: ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].delta.content, end='', flush=True)
print()

"""
model_type: qwen-vl-chat
query: <img>rose.jpg</img>图中是什么花，有几只？
response: 图中是三朵红玫瑰花。
query: 框出图中的花
response: <ref>花</ref><box>(34,449),(368,981)</box><box>(342,456),(670,917)</box><box>(585,508),(859,977)</box>
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
# with open('rose.jpg', 'rb') as f:
#     img_base64 = base64.b64encode(f.read()).decode('utf-8')
# image_url = f'data:image/jpeg;base64,{img_base64}'

# use local_path
# from swift.llm import convert_to_base64
# image_url = convert_to_base64(images=['rose.jpg'])['images'][0]
# image_url = f'data:image/jpeg;base64,{image_url}'

# use url
image_url = 'https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/rose.jpg'

query = '图中是什么花，有几只？'
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

# 流式
messages.append({'role': 'assistant', 'content': response})
query = '框出图中的花'
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
query: <img>https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/rose.jpg</img>图中是什么花，有几只？
response: 图中是三朵红玫瑰花。
query: 框出图中的花
response: <ref>花</ref><box>(34,449),(368,981)</box><box>(342,456),(670,917)</box><box>(585,508),(859,977)</box>
"""
```

## yi-vl-6b-chat

**服务端:**
```bash
# 使用原始模型
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type yi-vl-6b-chat

# 使用微调后的LoRA
CUDA_VISIBLE_DEVICES=0 swift deploy --ckpt_dir output/yi-vl-6b-chat/vx-xxx/checkpoint-xxx

# 使用微调后Merge LoRA的模型
CUDA_VISIBLE_DEVICES=0 swift deploy --ckpt_dir output/yi-vl-6b-chat/vx-xxx/checkpoint-xxx-merged
```

**客户端:**

测试:
```bash
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "yi-vl-6b-chat",
"messages": [{"role": "user", "content": "描述这张图片"}],
"temperature": 0,
"images": ["http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png"]
}'
```

使用swift:
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

query = '<image>描述这张图片'
request_config = XRequestConfig(temperature=0)
resp = inference_client(model_type, query, images=images, request_config=request_config)
response = resp.choices[0].message.content
print(f'query: {query}')
print(f'response: {response}')

history = [(query, response)]
query = '<image>图中有几只羊'
images.append('http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png')
request_config = XRequestConfig(stream=True, temperature=0)
stream_resp = inference_client(model_type, query, history, images=images, request_config=request_config)
print(f'query: {query}')
print('response: ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].delta.content, end='', flush=True)
print()

"""
model_type: yi-vl-6b-chat
query: <image>描述这张图片
response: 图片显示一只小猫坐在地板上,眼睛睁开,凝视着摄像机。小猫看起来很可爱,有灰色和白色的毛皮,以及蓝色的眼睛。小猫似乎正在看摄像机,可能被吸引到它正在拍摄它的照片或视频。
query: <image>图中有几只羊
response: 图中有四只羊.
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

query = '描述这张图片'
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
messages.append({'role': 'assistant', 'content': response})
query = '图中有几只羊'
messages.append({'role': 'user', 'content': [
    {'type': 'image_url', 'image_url': {'url': 'http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png'}},
    {'type': 'text', 'text': query},
]})
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
model_type: yi-vl-6b-chat
query: 描述这张图片
response: 图片显示一只小猫坐在地板上,眼睛睁开,凝视着摄像机。小猫看起来很可爱,有灰色和白色的毛皮,以及蓝色的眼睛。小猫似乎正在看摄像机,可能被吸引到它正在拍摄它的照片或视频。
query: 图中有几只羊
response: 图中有四只羊.
"""
```

## minicpm-v-v2_5-chat

**服务端:**
```bash
# 使用原始模型
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type minicpm-v-v2_5-chat

# 使用微调后的LoRA
CUDA_VISIBLE_DEVICES=0 swift deploy --ckpt_dir output/minicpm-v-v2_5-chat/vx-xxx/checkpoint-xxx

# 使用微调后Merge LoRA的模型
CUDA_VISIBLE_DEVICES=0 swift deploy --ckpt_dir output/minicpm-v-v2_5-chat/vx-xxx/checkpoint-xxx-merged
```

**客户端:**

测试:
```bash
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "minicpm-v-v2_5-chat",
"messages": [{"role": "user", "content": "描述这张图片"}],
"temperature": 0,
"images": ["http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png"]
}'
```

使用swift:
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

query = '<image>描述这张图片'
request_config = XRequestConfig(temperature=0)
resp = inference_client(model_type, query, images=images, request_config=request_config)
response = resp.choices[0].message.content
print(f'query: {query}')
print(f'response: {response}')

history = [(query, response)]
query = '这张图是如何产生的？'
request_config = XRequestConfig(stream=True, temperature=0)
stream_resp = inference_client(model_type, query, history, images=images, request_config=request_config)
print(f'query: {query}')
print('response: ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].delta.content, end='', flush=True)
print()

"""
model_type: minicpm-v-v2_5-chat
query: <image>描述这张图片
response: 这张图片展示了一只年轻的猫咪的特写，可能是一只小猫，具有明显的特征。它的毛皮主要是白色的，带有灰色和黑色的条纹，尤其是在脸部周围。小猫的眼睛很大，呈蓝色，给人一种好奇和迷人的表情。耳朵尖尖，竖立着，显示出警觉性。背景模糊不清，突出了小猫作为图片的主题。整体的色调柔和，猫咪的毛皮与背景的柔和色调形成对比。
query: 这张图是如何产生的？
response: 这张图片看起来是用数字绘画技术创作的。艺术家使用数字绘图工具来模仿毛皮的纹理和颜色，眼睛的反射，以及整体的柔和感。这种技术使艺术家能够精确地控制细节和色彩，创造出逼真的猫咪形象。
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

query = '描述这张图片'
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
messages.append({'role': 'assistant', 'content': response})
query = '这张图是如何产生的？'
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
query: 描述这张图片
response: 这张图片展示了一只年轻的猫咪的特写，可能是一只小猫，具有明显的特征。它的毛皮主要是白色的，带有灰色和黑色的条纹，尤其是在脸部周围。小猫的眼睛很大，呈蓝色，给人一种好奇和迷人的表情。耳朵尖尖，竖立着，显示出警觉性。背景模糊不清，突出了小猫作为图片的主题。整体的色调柔和，猫咪的毛皮与背景的柔和色调形成对比。
query: 这张图是如何产生的？
response: 这张图片看起来是用数字绘画技术创作的。艺术家使用数字绘图工具来模仿毛皮的纹理和颜色，眼睛的反射，以及整体的柔和感。这种技术使艺术家能够精确地控制细节和色彩，创造出逼真的猫咪形象。
"""
```

## 语音与视频模态

### qwen2-audio-7b-instruct

**服务端:**
```bash
# pip install transformers>=4.45
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen2-audio-7b-instruct
# ...
```

**客户端:**


使用swift:
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

query = '<audio>这段语音说了什么'
request_config = XRequestConfig(temperature=0)
resp = inference_client(model_type, query, audios=audios, request_config=request_config)
response = resp.choices[0].message.content
print(f'query: {query}')
print(f'response: {response}')

history = [(query, response)]
query = '这段语音是男生还是女生'
request_config = XRequestConfig(stream=True, temperature=0)
stream_resp = inference_client(model_type, query, history, audios=audios, request_config=request_config)
print(f'query: {query}')
print('response: ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].delta.content, end='', flush=True)
print()

"""
model_type: qwen2-audio-7b-instruct
query: <audio>这段语音说了什么
response: 这段语音说的是:'今天天气真好呀'
query: 这段语音是男生还是女生
response: 男声。
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
# with open('weather.wav', 'rb') as f:
#     aud_base64 = base64.b64encode(f.read()).decode('utf-8')
# audio_url = f'data:audio/wav;base64,{aud_base64}'

# use local_path
# from swift.llm import convert_to_base64
# audio_url = convert_to_base64(images=['weather.wav'])['images'][0]
# audio_url = f'data:audio/wav;base64,{audio_url}'

# use url
audio_url = 'http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/weather.wav'

query = '这段语音说了什么'
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

# 流式
messages.append({'role': 'assistant', 'content': response})
query = '这段语音是男生还是女生'
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
query: 这段语音说了什么
response: 这段语音说的是:'今天天气真好呀'
query: 这段语音是男生还是女生
response: 男声。
"""
```

### internvl2-2b

**服务端:**
```bash
# or 'minicpm-v-v2_6-chat'
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type internvl2-2b
# ...
```

**客户端:**


使用swift:
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

query = '<video>描述这段视频'
request_config = XRequestConfig(temperature=0)
resp = inference_client(model_type, query, videos=videos, request_config=request_config)
response = resp.choices[0].message.content
print(f'query: {query}')
print(f'response: {response}')

images = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png']
query = '<image>图中有几只羊'
request_config = XRequestConfig(stream=True, temperature=0)
stream_resp = inference_client(model_type, query, images=images, request_config=request_config)
print(f'query: {query}')
print('response: ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].delta.content, end='', flush=True)
print()
"""
model_type: internvl2-2b
query: <video>描述这段视频
response:  这段视频展示了一个小女孩坐在床上，专注地阅读一本书。她戴着一副黑框眼镜，穿着浅绿色的无袖上衣，坐在一张铺有花纹床单的床上。她的动作非常专注，时而翻页，时而用手指轻轻拨动书页，似乎在享受阅读的乐趣。

视频中，小女孩的身边放着一个白色的枕头，床的旁边可以看到一些衣物和杂物，包括一条白色的毛巾和几件衣物。背景中隐约可以看到一个木制的婴儿床，以及一些家居装饰，如墙上的画框和墙上的装饰品。

整个场景显得温馨而舒适，小女孩的专注和认真阅读的样子，让人感受到一种宁静和专注的氛围。视频通过展示小女孩的阅读过程，传递出一种热爱阅读和享受阅读的美好情感。
query: <image>图中有几只羊
response: 图中有四只羊。
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
# with open('baby.mp4', 'rb') as f:
#     vid_base64 = base64.b64encode(f.read()).decode('utf-8')
# video_url = f'data:video/mp4;base64,{vid_base64}'

# use local_path
# from swift.llm import convert_to_base64
# video_url = convert_to_base64(images=['baby.mp4'])['images'][0]
# video_url = f'data:video/mp4;base64,{video_url}'

# use url
video_url = 'https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4'

query = '描述这段视频'
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

# 流式
query = '图中有几只羊'
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
query: 描述这段视频
response:  这段视频展示了一个小女孩坐在床上，专注地阅读一本书。她戴着一副黑框眼镜，穿着浅绿色的无袖上衣，坐在一张铺有花纹床单的床上。她的动作非常专注，时而翻页，时而用手指轻轻拨动书页，似乎在享受阅读的乐趣。

视频中，小女孩的身边放着一个白色的枕头，床的旁边可以看到一些衣物和杂物，包括一条白色的毛巾和几件衣物。背景中隐约可以看到一个木制的婴儿床，以及一些家居装饰，如墙上的画框和墙上的装饰品。

整个场景显得温馨而舒适，小女孩的专注和认真阅读的样子，让人感受到一种宁静和专注的氛围。视频通过展示小女孩的阅读过程，传递出一种热爱阅读和享受阅读的美好情感。
query: 图中有几只羊
response: 图中有四只羊。
"""
```
