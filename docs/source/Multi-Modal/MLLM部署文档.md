# MLLM部署文档
对MLLM进行推理加速和部署可以查看[lmdeploy推理加速文档](LmDeploy推理加速文档.md)和[vLLM推理加速文档](vLLM推理加速文档.md).

## 目录
- [环境准备](#环境准备)
- [qwen-vl-chat](#qwen-vl-chat)
- [yi-vl-6b-chat](#yi-vl-6b-chat)
- [minicpm-v-v2_5-chat](#minicpm-v-v2_5-chat)
- [qwen-vl](#qwen-vl)

## 环境准备
```shell
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'

pip install vllm
```

以下我们给出了4个模型的例子（选择了尺寸较小的模型来方便实验），分别为qwen-vl-chat、qwen-vl、yi-vl-6b-chat和minicpm-v-v2_5-chat。从这些例子中，你可以发现三种不同类型MLLM：一轮对话可以包含多张图片（或不含图片）、一轮对话只能包含一张图片、整个对话围绕一张图片的部署和调用方式差异，以及MLLM中chat模型与base模型的部署和调用方式差异。

如果使用qwen-audio-chat, 请在qwen-vl-chat示例的基础上将`<img>`改为`<audio>`标签即可.

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
"messages": [{"role": "user", "content": "Picture 1:<img>https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/rose.jpg</img>\n图中是什么花，有几只？"}],
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
# query = f"""Picture 1:<img>{img_base64}</img>
# 图中是什么花，有几只？"""

# use local_path
# query = """Picture 1:<img>rose.jpg</img>
# 图中是什么花，有几只？"""

# use url
query = """Picture 1:<img>https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/rose.jpg</img>
图中是什么花，有几只？"""

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
query: Picture 1:<img>rose.jpg</img>
图中是什么花，有几只？
response: 图中是三朵红玫瑰花。
query: 框出图中的花
response: <ref>花</ref><box>(34,449),(368,981)</box><box>(342,456),(670,917)</box><box>(585,508),(859,977)</box>
"""
```

使用openai（方案1）:
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
# query = f"""Picture 1:<img>{img_base64}</img>
# 图中是什么花，有几只？"""

# use local_path
# from swift.llm import convert_to_base64
# query = """Picture 1:<img>rose.jpg</img>
# 图中是什么花，有几只？"""
# query = convert_to_base64(prompt=query)['prompt']

# use url
query = """Picture 1:<img>https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/rose.jpg</img>
图中是什么花，有几只？"""

messages = [{
    'role': 'user',
    'content': query
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
query: Picture 1:<img>https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/rose.jpg</img>
图中是什么花，有几只？
response: 图中是三朵红玫瑰花。
query: 框出图中的花
response: <ref>花</ref><box>(34,449),(368,981)</box><box>(342,456),(670,917)</box><box>(585,508),(859,977)</box>
"""
```

使用openai（方案2）:
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
        {'type': 'text', 'text': 'Picture 1:'},
        {'type': 'image_url', 'image_url': {'url': image_url}},
        {'type': 'text', 'text': f'\n{query}'},
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
query: Picture 1:<img>https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/rose.jpg</img>
图中是什么花，有几只？
response: 图中是三朵红玫瑰花。
query: 框出图中的花
response: <ref>花</ref><box>(34,449),(368,981)</box><box>(342,456),(670,917)</box><box>(585,508),(859,977)</box>
"""

"""Out[0]
model_type: qwen-vl-chat
query: Picture 1:<img>https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/rose.jpg</img>
图中是什么花，有几只？
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

query = '描述这张图片'
request_config = XRequestConfig(temperature=0)
resp = inference_client(model_type, query, images=images, request_config=request_config)
response = resp.choices[0].message.content
print(f'query: {query}')
print(f'response: {response}')

history = [(query, response)]
query = '图中有几只羊'
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
query: 描述这张图片
response: 图片显示一只小猫坐在地板上,眼睛睁开,凝视着摄像机。小猫看起来很可爱,有灰色和白色的毛皮,以及蓝色的眼睛。小猫似乎正在看摄像机,可能被吸引到它正在被拍摄。
query: 图中有几只羊
response: 图中有四只羊。
"""
```

使用openai（方案1）:
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

query = '描述这张图片'
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

# 流式
messages.append({'role': 'assistant', 'content': response})
query = '图中有几只羊'
images.append('http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png')
messages.append({'role': 'user', 'content': query})
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
model_type: yi-vl-6b-chat
query: 描述这张图片
response: 图片显示一只小猫坐在地板上,眼睛睁开,凝视着摄像机。小猫看起来很可爱,有灰色和白色的毛皮,以及蓝色的眼睛。小猫似乎正在看摄像机,可能被吸引到它正在被拍摄。
query: 图中有几只羊
response: 图中有四只羊。
"""
```

使用openai（方案2）:
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
        {'type': 'text', 'text': query},
        {'type': 'image_url', 'image_url': {'url': image_url}},
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
    {'type': 'text', 'text': query},
    {'type': 'image_url', 'image_url': {'url': 'http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png'}}
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
response: 图片显示一只小猫坐在地板上,眼睛睁开,凝视着摄像机。小猫看起来很可爱,有灰色和白色的毛皮,以及蓝色的眼睛。小猫似乎正在看摄像机,可能被吸引到它正在被拍摄。
query: 图中有几只羊
response: 图中有四只羊。
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

query = '描述这张图片'
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
query: 描述这张图片
response: 这张图片展示了一只年轻的猫咪的特写，可能是一只小猫，具有明显的特征。它的毛皮主要是白色的，带有灰色和黑色的条纹，尤其是在脸部周围。小猫的眼睛很大，呈蓝色，给人一种好奇和迷人的表情。耳朵尖尖，竖立着，显示出警觉。背景模糊不清，突出了小猫的特征。整体的色调柔和，猫咪的毛皮与背景的柔和色调形成对比。
query: 这张图是如何产生的？
response: 这张图片看起来是用数字绘画技术创作的。艺术家使用数字绘图工具来模仿毛皮的纹理和颜色，眼睛的反射，以及整体的柔和感。这种技术使艺术家能够精确地控制细节和色彩，创造出逼真的猫咪形象。
"""
```

使用openai（方案1）:
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

query = '描述这张图片'
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

# 流式
messages.append({'role': 'assistant', 'content': response})
query = '这张图是如何产生的？'
messages.append({'role': 'user', 'content': query})
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
model_type: minicpm-v-v2_5-chat
query: 描述这张图片
response: 这张图片展示了一只年轻的猫咪的特写，可能是一只小猫，具有明显的特征。它的毛皮主要是白色的，带有灰色和黑色的条纹，尤其是在脸部周围。小猫的眼睛很大，呈蓝色，给人一种好奇和迷人的表情。耳朵尖尖，竖立着，显示出警觉。背景模糊不清，突出了小猫的特征。整体的色调柔和，猫咪的毛皮与背景的柔和色调形成对比。
query: 这张图是如何产生的？
response: 这张图片看起来是用数字绘画技术创作的。艺术家使用数字绘图工具来模仿毛皮的纹理和颜色，眼睛的反射，以及整体的柔和感。这种技术使艺术家能够精确地控制细节和色彩，创造出逼真的猫咪形象。
"""
```

使用openai（方案2）:
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
        {'type': 'text', 'text': query},
        {'type': 'image_url', 'image_url': {'url': image_url}},
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
response: 这张图片展示了一只年轻的猫咪的特写，可能是一只小猫，具有明显的特征。它的毛皮主要是白色的，带有灰色和黑色的条纹，尤其是在脸部周围。小猫的眼睛很大，呈蓝色，给人一种好奇和迷人的表情。耳朵尖尖，竖立着，显示出警觉。背景模糊不清，突出了小猫的特征。整体的色调柔和，猫咪的毛皮与背景的柔和色调形成对比。
query: 这张图是如何产生的？
response: 这张图片看起来是用数字绘画技术创作的。艺术家使用数字绘图工具来模仿毛皮的纹理和颜色，眼睛的反射，以及整体的柔和感。这种技术使艺术家能够精确地控制细节和色彩，创造出逼真的猫咪形象。
"""
```

## qwen-vl

**服务端:**
```bash
# 使用原始模型
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen-vl

# 使用微调后的LoRA
CUDA_VISIBLE_DEVICES=0 swift deploy --ckpt_dir output/qwen-vl/vx-xxx/checkpoint-xxx

# 使用微调后Merge LoRA的模型
CUDA_VISIBLE_DEVICES=0 swift deploy --ckpt_dir output/qwen-vl/vx-xxx/checkpoint-xxx-merged
```

**客户端:**

测试:
```bash
curl http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "qwen-vl",
"prompt": "Picture 1:<img>https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/rose.jpg</img>\n这是一朵",
"max_tokens": 32,
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
# query = f"""Picture 1:<img>{img_base64}</img>
# 这是一朵"""

# use local_path
# query = """Picture 1:<img>rose.jpg</img>
# 这是一朵"""

# use url
query = """Picture 1:<img>https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/rose.jpg</img>
这是一朵"""

request_config = XRequestConfig(seed=42, max_tokens=32)
resp = inference_client(model_type, query, request_config=request_config)
response = resp.choices[0].text
print(f'query: {query}')
print(f'response: {response}')

request_config = XRequestConfig(stream=True, seed=42, max_tokens=32)
stream_resp = inference_client(model_type, query, request_config=request_config)
print(f'query: {query}')
print('response: ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].text, end='', flush=True)
print()

"""
model_type: qwen-vl
query: Picture 1:<img>https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/rose.jpg</img>
这是一朵
response: 玫瑰花的图片,希望你喜欢
query: Picture 1:<img>https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/rose.jpg</img>
这是一朵
response: 玫瑰花的图片,希望你喜欢
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
# query = f"""Picture 1:<img>{img_base64}</img>
# 这是一朵"""

# use local_path
# from swift.llm import convert_to_base64
# query = """Picture 1:<img>rose.jpg</img>
# 这是一朵"""
# query = convert_to_base64(prompt=query)['prompt']

# use url
query = """Picture 1:<img>https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/rose.jpg</img>
这是一朵"""

resp = client.completions.create(
    model=model_type,
    prompt=query,
    seed=42)
response = resp.choices[0].text
print(f'query: {query}')
print(f'response: {response}')

# 流式
stream_resp = client.completions.create(
    model=model_type,
    prompt=query,
    stream=True,
    seed=42)

print(f'query: {query}')
print('response: ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].text, end='', flush=True)
print()

"""Out[0]
model_type: qwen-vl
query: Picture 1:<img>https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/rose.jpg</img>
这是一朵
response: 玫瑰花的图片,希望你喜欢
query: Picture 1:<img>https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/rose.jpg</img>
这是一朵
response: 玫瑰花的图片,希望你喜欢
"""
```
