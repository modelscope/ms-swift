# LmDeploy推理加速与部署
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

pip install lmdeploy
```

## 推理加速

### 使用python

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
# 与`transformers.GenerationConfig`类似的接口
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
response: 这是一幅以卡通风格绘制的四只绵羊的画面。图中绵羊们站在一片绿色的草地中，背景是多山的风景，有蓝天、白云和几只鸟在飞翔。绵羊们的毛发主要是白色并点缀着一些浅棕色，它们头上都有绒毛。这些绵羊有着大大的眼睛和耳朵，造型可爱，整体画面给人一种和平与田园的感觉。
query: 你是谁？
response: 我是InternVL，是由上海人工智能实验室的通用视觉团队（OpenGVLab）和商汤科技联合开发的模型。
query: <img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png</img><img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png</img>What is the difference bewteen the two images?
response: In the first image, the sheep are standing in front of lush green mountains. In the second image, some of their wool is dyed green instead of white.
{'num_prompt_tokens': 8086, 'num_generated_tokens': 135, 'num_samples': 3, 'runtime': 1.5491395709104836, 'samples/s': 1.93655888490202, 'tokens/s': 87.1451498205909}
query: 有几只羊
response: 图片中总共有四只绵羊。
history: [['<image>描述图片', '这是一幅以卡通风格绘制的四只绵羊的画面。图中绵羊们站在一片绿色的草地中，背景是多山的风景，有蓝天、白云和几只鸟在飞翔。绵羊们的毛发主要是白色并点缀着一些浅棕色，它们头上都有绒毛。这些绵羊有着大大的眼睛和耳朵，造型可爱，整体画面给人一种和平与田园的感觉。'], ['有几只羊', '图片中总共有四只绵羊。']]
{'num_prompt_tokens': 3470, 'num_generated_tokens': 8, 'num_samples': 1, 'runtime': 0.6616258070571348, 'samples/s': 1.5114283471618646, 'tokens/s': 12.091426777294917}
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
敬请期待...

## 部署
敬请期待...
