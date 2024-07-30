# Mutlimoda LLM Deployment
For inference acceleration and deployment of MLLM, you can refer to the [LmDeploy Inference Acceleration Documentation](LmDeploy-inference-acceleration.md) and the [vLLM Inference Acceleration Documentation](vllm-inference-acceleration.md).

## Table of Contents
- [Environment Setup](#environment-setup)
- [qwen-vl-chat](#qwen-vl-chat)
- [yi-vl-6b-chat](#yi-vl-6b-chat)
- [minicpm-v-v2_5-chat](#minicpm-v-v2_5-chat)
- [qwen-vl](#qwen-vl)

## Environment Setup
```shell
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'

pip install vllm
```

Here we provide examples of four models (selecting smaller-sized models to facilitate experiments): qwen-vl-chat, qwen-vl, yi-vl-6b-chat, and minicpm-v-v2_5-chat. From these examples, you can identify three different types of MLLMs: a single round of dialogue can contain multiple images (or no images), a single round of dialogue can only contain one image, and the way the entire dialogue revolves around an image and the differences in deployment and invocation methods, as well as the differences between the chat and base models within MLLMs.

If you're using qwen-audio-chat, simply replace the `<img>` tag with `<audio>` based on the qwen-vl-chat example.

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
"messages": [{"role": "user", "content": "Picture 1:<img>https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/rose.jpg</img>\nWhat kind of flower is in the picture and how many are there?"}],
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
# query = f"""Picture 1:<img>{img_base64}</img>
# What kind of flower is in the picture and how many are there?"""

# use local_path
# query = """Picture 1:<img>rose.jpg</img>
# What kind of flower is in the picture and how many are there?"""

# use url
query = """Picture 1:<img>https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/rose.jpg</img>
What kind of flower is in the picture and how many are there?"""

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
query: Picture 1:<img>https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/rose.jpg</img>
What kind of flower is in the picture and how many are there?
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
# query = f"""Picture 1:<img>{img_base64}</img>
# What kind of flower is in the picture and how many are there?"""

# use local_path
# from swift.llm import convert_to_base64
# query = """Picture 1:<img>rose.jpg</img>
# What kind of flower is in the picture and how many are there?"""
# query = convert_to_base64(prompt=query)['prompt']

# use url
query = """Picture 1:<img>https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/rose.jpg</img>
What kind of flower is in the picture and how many are there?"""

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
query: Picture 1:<img>https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/rose.jpg</img>
What kind of flower is in the picture and how many are there?
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

query = 'Describe this image.'
request_config = XRequestConfig(seed=42)
resp = inference_client(model_type, query, images=images, request_config=request_config)
response = resp.choices[0].message.content
print(f'query: {query}')
print(f'response: {response}')

history = [(query, response)]
query = 'How many sheep are in the picture?'
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
query: Describe this image.
response: The image captures a moment of tranquility featuring a gray and white kitten. The kitten, with its eyes wide open, is the main subject of the image. Its nose is pink, adding a touch of color to its gray and white fur. The kitten is sitting on a white surface, which contrasts with its gray and white fur. The background is blurred, drawing focus to the kitten. The image does not contain any text. The kitten's position relative to the background suggests it is in the foreground of the image. The image does not contain any other objects or creatures. The kitten appears to be alone in the image. The image does not contain any action, but the kitten's wide-open eyes give a sense of curiosity and alertness. The image does not contain any aesthetic descriptions. The image is a simple yet captivating portrait of a gray and white kitten.
query: How many sheep are in the picture?
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
    seed=42,
    extra_body={'images': images})
response = resp.choices[0].message.content
print(f'query: {query}')
print(f'response: {response}')

# Streaming
messages.append({'role': 'assistant', 'content': response})
query = 'How many sheep are in the picture?'
images.append('http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png')
messages.append({'role': 'user', 'content': query})
stream_resp = client.chat.completions.create(
    model=model_type,
    messages=messages,
    stream=True,
    seed=42,
    extra_body={'images': images})

print(f'query: {query}')
print('response: ', end='')
for chunk in stream_resp:
    print(chunk.choices[0].delta.content, end='', flush=True)
print()

"""
model_type: yi-vl-6b-chat
query: Describe this image.
response: The image captures a moment of tranquility featuring a gray and white kitten. The kitten, with its eyes wide open, is the main subject of the image. Its nose is pink, adding a touch of color to its gray and white fur. The kitten is sitting on a white surface, which contrasts with its gray and white fur. The background is blurred, drawing focus to the kitten. The image does not contain any text. The kitten's position relative to the background suggests it is in the foreground of the image. The image does not contain any other objects or creatures. The kitten appears to be alone in the image. The image does not contain any action, but the kitten's wide-open eyes give a sense of curiosity and alertness. The image does not contain any aesthetic descriptions. The image is a simple yet captivating portrait of a gray and white kitten.
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

query = 'Describe this image.'
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
query: Describe this image.
response: The image is a digital painting of a kitten, which is the main subject. The kitten's fur is rendered with a mix of gray, black, and white, giving it a realistic appearance. Its eyes are wide open, and the expression is one of curiosity or alertness. The background is blurred, which brings the focus entirely on the kitten. The painting style is detailed and lifelike, capturing the essence of a young feline's innocent and playful nature. The image does not convey any specific context or background story beyond the depiction of the kitten itself.
query: How was this picture generated?
response: This picture was generated using digital art techniques. The artist likely used a software program to create the image, manipulating pixels and colors to achieve the detailed and lifelike representation of the kitten. Digital art allows for a high degree of control over the final product, enabling artists to fine-tune details and create realistic textures and shading.
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
messages.append({'role': 'assistant', 'content': response})
query = 'How was this picture generated?'
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
query: Describe this image.
response: The image is a digital painting of a kitten, which is the main subject. The kitten's fur is rendered with a mix of gray, black, and white, giving it a realistic appearance. Its eyes are wide open, and the expression is one of curiosity or alertness. The background is blurred, which brings the focus entirely on the kitten. The painting style is detailed and lifelike, capturing the essence of a young feline's innocent and playful nature. The image does not convey any specific context or background story beyond the depiction of the kitten itself.
query: How was this picture generated?
response: This picture was generated using digital art techniques. The artist likely used a software program to create the image, manipulating pixels and colors to achieve the detailed and lifelike representation of the kitten. Digital art allows for a high degree of control over the final product, enabling artists to fine-tune details and create realistic textures and shading.
"""
```

## qwen-vl

**Server side:**
```bash
# Using the original model
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type qwen-vl

# Using the fine-tuned LoRA
CUDA_VISIBLE_DEVICES=0 swift deploy --ckpt_dir output/qwen-vl/vx-xxx/checkpoint-xxx

# Using the fine-tuned Merge LoRA model
CUDA_VISIBLE_DEVICES=0 swift deploy --ckpt_dir output/qwen-vl/vx-xxx/checkpoint-xxx-merged
```

**Client side:**

Test:
```bash
curl http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "qwen-vl",
"prompt": "Picture 1:<img>https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/rose.jpg</img>\nThis is a",
"max_tokens": 32,
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
# query = f"""Picture 1:<img>{img_base64}</img>
# This is a"""

# use local_path
# query = """Picture 1:<img>rose.jpg</img>
# This is a"""

# use url
query = """Picture 1:<img>https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/rose.jpg</img>
This is a"""

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
This is a
response:  picture of a bouquet of roses.
query: Picture 1:<img>https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/rose.jpg</img>
This is a
response: picture of a bouquet of roses.
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
# query = f"""Picture 1:<img>{img_base64}</img>
# This is a"""

# use local_path
# from swift.llm import convert_to_base64
# query = """Picture 1:<img>rose.jpg</img>
# This is a"""
# query = convert_to_base64(prompt=query)['prompt']

# use url
query = """Picture 1:<img>https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/rose.jpg</img>
This is a"""

resp = client.completions.create(
    model=model_type,
    prompt=query,
    seed=42)
response = resp.choices[0].text
print(f'query: {query}')
print(f'response: {response}')

# Streaming
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
This is a
response:  picture of a bouquet of roses.
query: Picture 1:<img>https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/rose.jpg</img>
This is a
response: picture of a bouquet of roses.
"""
```
