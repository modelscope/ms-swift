
# InternVL 最佳实践

本篇文档对应的模型

| model | model_type |
|-------|------------|
| [Florence-2-base](https://www.modelscope.cn/models/AI-ModelScope/Florence-2-base) | florence-2-base |
| [Florence-2-base-ft](https://www.modelscope.cn/models/AI-ModelScope/Florence-2-base-ft) | florence-2-base-ft |
| [Florence-2-large](https://www.modelscope.cn/models/AI-ModelScope/Florence-2-large) | florence-2-large |
| [Florence-2-large-ft](https://www.modelscope.cn/models/AI-ModelScope/Florence-2-large-ft) | florence-2-large-ft |


## 目录
- [环境准备](#环境准备)
- [推理](#推理)
- [微调](#微调)
- [微调后推理](#微调后推理)

## 环境准备
```shell
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'
```

## 推理
下面的教程以[Florence-2-large-ft](https://www.modelscope.cn/models/AI-ModelScope/Florence-2-large-ft)为例, 你可以通过切换model_type使用其他florence系列模型

**注意**
- 如果要使用本地模型文件，加上参数 `--model_id_or_path /path/to/model`
- florence系列模型内置了一些视觉任务的prompt, 对应的映射可以查看`swift.llm.utils.template.FlorenceTemplate`, 更多prompt可以查看 Modelscope/Hugging Face 的模型详情页
- Florence模型不具备中文能力
- Florence模型暂时不支持流式输出

```shell
# 4.4GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift infer --model_type florence-2-large-ft  --dtype fp32 --max_new_tokens 1024 --stream False
```

输出: (支持传入本地路径或URL)
```python
"""
<<< Describe the image
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png
{'Describe the image': 'A grey and white kitten with blue eyes.'}
<<< <OD>
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png
{'Locate the objects with category name in the image.': 'shelf<loc_264><loc_173><loc_572><loc_748><loc_755><loc_274><loc_966><loc_737><loc_46><loc_335><loc_261><loc_763><loc_555><loc_360><loc_760><loc_756>'}
--------------------------------------------------
<<< <CAPTION>
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png
{'What does the image describe?': 'A cartoon picture of four sheep standing in a field.'}
--------------------------------------------------
<<< <DETAILED_CAPTION>
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png
{'Describe in detail what is shown in the image.': 'In the image is animated. In the image there are sheeps. At the bottom of the image on the ground there is grass. In background there are hills. At top of the images there are clouds.'}
--------------------------------------------------
<<< <MORE_DETAILED_CAPTION>
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png
{'Describe with a paragraph what is shown in the image.': 'Four sheep are standing in a field. They are all white and fluffy. They have horns on their heads. There are mountains behind them. There is grass and weeds on the ground in front of them. '}
--------------------------------------------------
<<< <DENSE_REGION_CAPTION>
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png
{'Locate the objects in the image, with their descriptions.': 'cartoon sheep illustration<loc_265><loc_175><loc_572><loc_748>cartoon ram illustration<loc_755><loc_275><loc_966><loc_737>cartoon white sheep illustration<loc_44><loc_335><loc_262><loc_764>cartoon goat illustration<loc_555><loc_361><loc_762><loc_756>'}
--------------------------------------------------
<<< <REGION_PROPOSAL>
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png
{'Locate the region proposals in the image.': '<loc_45><loc_176><loc_967><loc_761><loc_266><loc_175><loc_570><loc_749><loc_757><loc_274><loc_966><loc_738><loc_46><loc_334><loc_261><loc_763><loc_556><loc_361><loc_760><loc_756>'}
--------------------------------------------------
<<< <CAPTION_TO_PHRASE_GROUNDING>the sheeps
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png
{'Locate the phrases in the caption: the sheeps': 'thethe sheeps<loc_45><loc_175><loc_967><loc_764><loc_266><loc_176><loc_572><loc_749><loc_756><loc_275><loc_965><loc_739><loc_46><loc_335><loc_261><loc_765><loc_557><loc_361><loc_760><loc_758>'}
```
示例图片如下:

cat:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png" width="250" style="display: inline-block;">

animal:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png" width="250" style="display: inline-block;">

**Python 推理**

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference,
    get_default_template_type, inference_stream
)
from swift.utils import seed_everything
import torch

model_type = "florence-2-large-ft"
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')
model, tokenizer = get_model_tokenizer(model_type, torch.float32, model_kwargs={'device_map': None})

model.generation_config.max_new_tokens = 1024
template = get_template(template_type, tokenizer)
seed_everything(42)

images = ['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png']
query = 'Describe the image'
response, history = inference(model, template, query, images=images)
print(f'query: {query}')
print(f'response: {response}')
'''
query: Describe the image
response: {'Describe the image': 'Four sheep standing in a field with mountains in the background.'}
'''
```