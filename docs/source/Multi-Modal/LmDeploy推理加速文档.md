# LmDeploy推理加速与部署
lmdeploy github: [https://github.com/InternLM/lmdeploy](https://github.com/InternLM/lmdeploy).

支持lmdeploy推理加速的多模态模型可以查看[支持的模型](../LLM/支持的模型和数据集.md#多模态大模型).

## 目录
- [环境准备](#环境准备)
- [推理加速](#推理加速)
- [部署](#部署)

## 环境准备
GPU设备: A10, 3090, V100, A100均可.
```bash
# 设置pip全局镜像 (加速下载)
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# 安装ms-swift
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'

# lmdeploy与cuda版本有对应关系，请按照`https://github.com/InternLM/lmdeploy#installation`进行安装
pip install lmdeploy
```

## 推理加速

### 使用python

[OpenGVLab/InternVL2-2B](https://modelscope.cn/models/OpenGVLab/InternVL2-2B/summary)

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
lmdeploy_engine.generation_config.max_new_tokens = 256
generation_info = {}

request_list = [{'query': '<image>描述图片', 'images': ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png']},
                {'query': '你是谁？'},
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
request_list = [{'query': '<video>描述视频', 'videos': ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/baby.mp4']}]
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
query: <image>描述图片
response: 这张图片展示了四只卡通风格的羊，它们并排排列在一片绿色的草地上。草地上似乎还有轻微的阴影，显示了光源从左上方照射下来的效果。

从左到右，第一只羊头上顶着一团厚厚的羊毛，第二只羊稍微有点低，眼睛大大的，看起来非常友善；第三只羊和第四只羊头高且耳朵竖起，它们看起来似乎更威严和独立。

背景是一片绿色、蓝色的天空中有着一些白色的云朵，远处的山峦线条明显，使得整个画面充满自然的美感。

总体来说，这幅画作表达了一种和谐、宁静的自然氛围，四只羊的形象也很生动可爱，似乎传递出一种温柔、质朴的感觉。
query: 你是谁？
response: 我是InternVL，是由上海人工智能实验室的通用视觉团队（OpenGVLab）和商汤科技联合开发的模型。
query: <img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png</img><img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png</img>What is the difference bewteen the two images?
response: I'm unable to identify or compare images. However, if this image were to be classified based on its design or layout, it might demonstrate:

- A change in the display order or arrangement of elements within the image.
- An evolution in artistic style or technique.
- Different elements added or cut out to create a variation.

I'd need more specific details to make an accurate comparison.
{'num_prompt_tokens': 8095, 'num_generated_tokens': 253, 'num_samples': 3, 'runtime': 4.090330162958708, 'samples/s': 0.7334371262172084, 'tokens/s': 61.8531976443179}
query: <video>描述视频
response: 这个视频展示了一个小女孩在房间里读书的场景。镜头从一个特定的角度捕捉到小女孩专注于书本的情况。这个小女孩是金发，她穿着蓝色的无袖上衣，还戴着一副黑色的眼镜。小女孩的注意力集中在手中的一个白色封皮的书上，她轻轻翻开书页，显示出对书的兴趣和热爱。她的手偶尔会抚摸和翻看书页，表现出一种探索和专注的态度。

背景中可以看到一个木制的婴儿床，房间的地面铺满了温馨的米色毯子。房间的装饰温暖而家庭化，有一个带灯的台灯在床边，旁边似乎还放了一些玩具。

视频中，小女孩的右侧，她的左手上可以看到另一本书。这本书已经翻开到一页，她将目光转向这本书，用手指轻轻触碰到书本。她的身体稍微向前倾，显示出一种沉浸在阅读中的状态。

整个视频的色调温暖而柔和，背景中的物品和她的穿着形成了和谐的家庭氛围，给人一种舒适和温馨的感觉。小女孩的举止显得非常自然和放松，书页的翻动似乎也传递了她对这个故事的着迷。她用双手轻轻翻动书页的动作显得非常可爱和童真。

总之，这个视频完美捕捉到一个小女孩在家庭环境中享受阅读乐趣的瞬间
{'num_prompt_tokens': 6247, 'num_generated_tokens': 257, 'num_samples': 1, 'runtime': 3.0897628950187936, 'samples/s': 0.32364943006214636, 'tokens/s': 83.17790352597162}
"""
```

[Shanghai_AI_Laboratory/internlm-xcomposer2d5-7b](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-xcomposer2d5-7b)

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    ModelType, get_lmdeploy_engine, get_default_template_type,
    get_template, inference_lmdeploy, inference_stream_lmdeploy
)

# ModelType.qwen_vl_chat, ModelType.deepseek_vl_1_3b_chat, ModelType.minicpm_v_v2_5_chat
model_type = ModelType.internlm_xcomposer2_5_7b_chat
lmdeploy_engine = get_lmdeploy_engine(model_type)
template_type = get_default_template_type(model_type)
template = get_template(template_type, lmdeploy_engine.hf_tokenizer)
# 与`transformers.GenerationConfig`类似的接口
lmdeploy_engine.generation_config.max_new_tokens = 256
generation_info = {}

request_list = [{'query': '<image>描述图片', 'images': ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png']},
               ]
resp_list = inference_lmdeploy(lmdeploy_engine, template, request_list, generation_info=generation_info)
for request, resp in zip(request_list, resp_list):
    print(f"query: {request['query']}")
    print(f"response: {resp['response']}")
print(generation_info)

# stream
history0 = resp_list[0]['history']
request_list = [{'query': '有几只羊', 'history': history0, 'images': ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png']}]
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
query: <image>描述图片
response: 在图片中，有四只卡通风格的羊站在一片翠绿的草地中间。这些羊以简洁而不失真挚的形象出现，它们的躯干由白色和棕色的形状组成，而四肢则是纯粹的黑色。头部设计简洁，白色与棕色的搭配与整体协调一致。图中有四只羊，最突出的是一只最大的羊，它似乎处于图片中央，可能是画面的焦点。另外三只羊环绕在它的周围，形成一种对称感。这些羊们没有穿上任何衣物，它们在阳光下显得格外耀眼。天空是明亮的蓝色，背景中的山峰柔和地与天空相接，形成了一种宁静的田园景象。
{'num_prompt_tokens': 2206, 'num_generated_tokens': 132, 'num_samples': 1, 'runtime': 2.793646134901792, 'samples/s': 0.3579551423878365, 'tokens/s': 47.25007879519442}
query: 有几只羊
response: 图片中一共有四只羊。
history: [['<image>描述图片', '在图片中，有四只卡通风格的羊站在一片翠绿的草地中间。这些羊以简洁而不失真挚的形象出现，它们的躯干由白色和棕色的形状组成，而四肢则是纯粹的黑色。头部设计简洁，白色与棕色的搭配与整体协调一致。图中有四只羊，最突出的是一只最大的羊，它似乎处于图片中央，可能是画面的焦点。另外三只羊环绕在它的周围，形成一种对称感。这些羊们没有穿上任何衣物，它们在阳光下显得格外耀眼。天空是明亮的蓝色，背景中的山峰柔和地与天空相接，形成了一种宁静的田园景象。'], ['有几只羊', '图片中一共有四只羊。']]
{'num_prompt_tokens': 2352, 'num_generated_tokens': 6, 'num_samples': 1, 'runtime': 0.635085433954373, 'samples/s': 1.5745913014781, 'tokens/s': 9.447547808868599}
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
    # 与`transformers.GenerationConfig`类似的接口
    lmdeploy_engine.generation_config.max_new_tokens = 256
    generation_info = {}

    request_list = [{'query': '<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png</img>描述图片'},
                    {'query': '<image>描述图片', 'images': ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png']},
                    {'query': '你是谁？'}]
    resp_list = inference_lmdeploy(lmdeploy_engine, template, request_list, generation_info=generation_info)
    for request, resp in zip(request_list, resp_list):
        print(f"query: {request['query']}")
        print(f"response: {resp['response']}")
    print(generation_info)

    # stream
    history0 = resp_list[0]['history']
    request_list = [{'query': '有几只羊', 'history': history0}]
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
query: <img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png</img>描述图片
response: 这张图片展示了一群羊站在草地上。具体来说，图片中共有四只羊，它们的身体都是白色的，头部也是白色的，但是它们的耳朵颜色和脸部的细节各不相同。从左到右，第一只羊的耳朵是棕色的，脸部是白色的，鼻子是粉色的；第二只羊的耳朵也是棕色的，脸部是白色的，鼻子是粉色的；第三只羊的耳朵是棕色的，脸部是白色的，鼻子是粉色的；第四只羊的耳朵是棕色的，脸部是白色的，鼻子是粉色的。四只羊站在绿色的草地上，草地呈现出不同的绿色阴影，显示出草地的起伏。在图片的背景中，可以看到蓝色的天空和几朵白云，以及连绵起伏的山脉。
query: <image>描述图片
response: 这张图片展示了一群羊站在草地上。具体来说，图片中共有四只羊，它们的身体都是白色的，头部也是白色的，但是它们的耳朵颜色和脸部的细节各不相同。从左到右，第一只羊的耳朵是棕色的，脸部是白色的，鼻子是粉色的；第二只羊的耳朵也是棕色的，脸部是白色的，鼻子是粉色的；第三只羊的耳朵是棕色的，脸部是白色的，鼻子是粉色的；第四只羊的耳朵是棕色的，脸部是白色的，鼻子是粉色的。四只羊站在绿色的草地上，草地呈现出不同的绿色阴影，显示出草地的起伏。在图片的背景中，可以看到蓝色的天空和几朵白云，以及连绵起伏的山脉。
query: 你是谁？
response: 我是人工智能助手智谱清言（ChatGLM），是基于智谱 AI 公司于 2023 年训练的语言模型开发的。我的任务是针对用户的问题和要求提供适当的答复和支持。
{'num_prompt_tokens': 3226, 'num_generated_tokens': 352, 'num_samples': 3, 'runtime': 9.829129087971523, 'samples/s': 0.3052152406535462, 'tokens/s': 35.81192157001609}
query: 有几只羊
response: 图中共有四只羊。
history: [['<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png</img>描述图片', '这张图片展示了一群羊站在草地上。具体来说，图片中共有四只羊，它们的身体都是白色的，头部也是白色的，但是它们的耳朵颜色和脸部的细节各不相同。从左到右，第一只羊的耳朵是棕色的，脸部是白色的，鼻子是粉色的；第二只羊的耳朵也是棕色的，脸部是白色的，鼻子是粉色的；第三只羊的耳朵是棕色的，脸部是白色的，鼻子是粉色的；第四只羊的耳朵是棕色的，脸部是白色的，鼻子是粉色的。四只羊站在绿色的草地上，草地呈现出不同的绿色阴影，显示出草地的起伏。在图片的背景中，可以看到蓝色的天空和几朵白云，以及连绵起伏的山脉。'], ['有几只羊', '图中共有四只羊。']]
{'num_prompt_tokens': 1772, 'num_generated_tokens': 7, 'num_samples': 1, 'runtime': 1.6001809199806303, 'samples/s': 0.6249293361228834, 'tokens/s': 4.374505352860184}
"""
```


### 使用CLI
```bash
CUDA_VISIBLE_DEVICES=0 swift infer --model_type deepseek-vl-1_3b-chat --infer_backend lmdeploy

CUDA_VISIBLE_DEVICES=0 swift infer --model_type internvl2-2b --infer_backend lmdeploy

# TP
CUDA_VISIBLE_DEVICES=0,1 swift infer --model_type qwen-vl-chat \
    --infer_backend lmdeploy --tp 2

CUDA_VISIBLE_DEVICES=0,1 swift infer --model_type internlm-xcomposer2_5-7b-chat \
    --infer_backend lmdeploy --tp 2
```

## 部署

**服务端:**

```bash
CUDA_VISIBLE_DEVICES=0 swift deploy --model_type deepseek-vl-1_3b-chat --infer_backend lmdeploy

CUDA_VISIBLE_DEVICES=0 swift deploy --model_type internvl2-2b --infer_backend lmdeploy

# TP
CUDA_VISIBLE_DEVICES=0,1 swift deploy --model_type qwen-vl-chat \
    --infer_backend lmdeploy --tp 2

CUDA_VISIBLE_DEVICES=0,1 swift deploy --model_type internlm-xcomposer2_5-7b-chat \
    --infer_backend lmdeploy --tp 2
```

**客户端:**

这里介绍对internvl2-2b进行客户端调用的展示:

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
response: 这段视频展示了一个小女孩坐在床上，专注地阅读一本书。她戴着一副黑框眼镜，穿着浅绿色的无袖上衣，头发梳成马尾辫。视频中，小女孩的注意力完全集中在书本上，她用双手捧着书，时而翻页，时而抬头看向镜头。

背景中可以看到一个木制的婴儿床，床上铺着花纹的床单，旁边还有一些衣物和玩具。房间的墙壁上挂着一些装饰品，显得温馨而舒适。

视频中，小女孩的动作非常自然，她时而翻页，时而用手指轻轻拨动书页，显得非常专注和投入。她的表情平静而专注，似乎完全沉浸在书中的内容中。

整个视频给人一种温馨、宁静的感觉，小女孩的专注和认真让人感到非常温暖。视频中的每一个细节都展示了小女孩的纯真和好奇心，让人不禁想要和她一起探索书中的世界。
query: 图中有几只羊
response: 图中有四只羊。
"""
```

更多客户端调用方式可以查看: [MLLM部署文档](MLLM部署文档.md).
