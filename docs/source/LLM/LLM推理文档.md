# LLM推理文档
如果你要使用vllm进行推理加速, 可以查看[VLLM推理加速与部署](VLLM推理加速与部署.md#推理加速)

## 目录
- [环境准备](#环境准备)
- [推理](#推理)
- [Web-UI](#web-ui)

## 环境准备
GPU设备: A10, 3090, V100, A100均可.
```bash
# 设置pip全局镜像 (加速下载)
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# 安装ms-swift
pip install 'ms-swift[llm]' -U

# 如果你想要使用基于auto_gptq的模型进行推理.
# 使用auto_gptq的模型: `https://github.com/modelscope/swift/blob/main/docs/source/LLM/支持的模型和数据集.md#模型`
# auto_gptq和cuda版本有对应关系，请按照`https://github.com/PanQiWei/AutoGPTQ#quick-installation`选择版本
pip install auto_gptq -U

# 环境对齐 (通常不需要运行. 如果你运行错误, 可以跑下面的代码, 仓库使用最新环境测试)
pip install -r requirements/framework.txt  -U
pip install -r requirements/llm.txt  -U
```

## 推理
### qwen-7b-chat
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,
)
from swift.utils import seed_everything

model_type = ModelType.qwen_7b_chat
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')  # template_type: qwen


kwargs = {}
# kwargs['use_flash_attn'] = True  # 使用flash_attn

model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'}, **kwargs)
# 修改max_new_tokens
model.generation_config.max_new_tokens = 128

template = get_template(template_type, tokenizer)
seed_everything(42)
query = '浙江的省会在哪里？'
response, history = inference(model, template, query)
print(f'query: {query}')
print(f'response: {response}')
query = '这有什么好吃的？'
response, history = inference(model, template, query, history)
print(f'query: {query}')
print(f'response: {response}')
print(f'history: {history}')

"""Out[0]
query: 浙江的省会在哪里？
response: 浙江省的省会是杭州。
query: 这有什么好吃的？
response: 杭州市有很多著名的美食，例如西湖醋鱼、龙井虾仁、糖醋排骨、毛血旺等。此外，还有杭州特色的点心，如桂花糕、荷花酥、艾窝窝等。
history: [('浙江的省会在哪里？', '浙江省的省会是杭州。'), ('这有什么好吃的？', '杭州市有很多著名的美食，例如西湖醋鱼、龙井虾仁、糖醋排骨、毛血旺等。此外，还有杭州特色的点心，如桂花糕、荷花酥、艾窝窝等。')]
"""

# 流式输出对话模板
inference(model, template, '第一个问题是什么', history, verbose=True, stream=True)
"""Out[1]
[PROMPT]<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
浙江的省会在哪里？<|im_end|>
<|im_start|>assistant
浙江省的省会是杭州。<|im_end|>
<|im_start|>user
这有什么好吃的？<|im_end|>
<|im_start|>assistant
杭州市有很多著名的美食，例如西湖醋鱼、龙井虾仁、糖醋排骨、毛血旺等。此外，还有杭州特色的点心，如桂花糕、荷花酥、艾窝窝等。<|im_end|>
<|im_start|>user
第一个问题是什么<|im_end|>
<|im_start|>assistant
[OUTPUT]你的第一个问题是“浙江的省会在哪里？”<|im_end|>
"""
```

### qwen-7b-chat-int4
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,
)
from swift.utils import seed_everything

model_type = ModelType.qwen_7b_chat_int4
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')  # template_type: qwen

model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'})

template = get_template(template_type, tokenizer)
seed_everything(42)
query = '浙江的省会在哪里？'
response, history = inference(model, template, query)
print(f'query: {query}')
print(f'response: {response}')
query = '这有什么好吃的？'
response, history = inference(model, template, query, history)
print(f'query: {query}')
print(f'response: {response}')
print(f'history: {history}')

"""Out[0]
query: 浙江的省会在哪里？
response: 浙江省的省会是杭州。
query: 这有什么好吃的？
response: 杭州有很多著名的美食，例如西湖醋鱼、东坡肉、宋嫂鱼羹、叫化鸡等。此外，还有杭州特色的点心，如桂花糖藕、酒酿圆子、麻婆豆腐等等。
history: [('浙江的省会在哪里？', '浙江省的省会是杭州。'), ('这有什么好吃的？', '杭州有很多著名的美食，例如西湖醋鱼、东坡肉、宋嫂鱼羹、叫化鸡等。此外，还有杭州特色的点心，如桂花糖藕、酒酿圆子、麻婆豆腐等等。')]
"""
```

### qwen-7b
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,
)
from swift.utils import seed_everything

model_type = ModelType.qwen_7b
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')  # template_type: default-generation

model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'})
model.generation_config.max_new_tokens = 64
template = get_template(template_type, tokenizer)
seed_everything(42)
query = '浙江 -> 杭州\n安徽 -> 合肥\n四川 ->'
response, history = inference(model, template, query)
print(f'query: {query}')
print(f'response: {response}')
"""Out[0]
query: 浙江 -> 杭州
安徽 -> 合肥
四川 ->
response:  成都
山东 -> 济南
福建 -> 福州
重庆 -> 重庆
广东 -> 广州
北京 -> 北京
浙江 -> 杭州
安徽 -> 合肥
四川 -> 成都
山东 -> 济南
福建 -> 福州
重庆
"""
```

### 流式输出
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference_stream, ModelType, get_default_template_type,
)
from swift.utils import seed_everything

model_type = ModelType.qwen_7b_chat
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')  # template_type: qwen

model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'})

template = get_template(template_type, tokenizer)
seed_everything(42)
query = '浙江的省会在哪里？'
gen = inference_stream(model, template, query)
print(f'query: {query}')
for response, history in gen:
    print(f'response: {response}')
query = '这有什么好吃的？'
gen = inference_stream(model, template, query, history)
print(f'query: {query}')
for response, history in gen:
    print(f'response: {response}')
print(f'history: {history}')

"""Out[0]
query: 浙江的省会在哪里？
...
response: 浙江省的省会是杭州。
query: 这有什么好吃的？
...
response: 杭州市有很多著名的美食，例如西湖醋鱼、龙井虾仁、糖醋排骨、毛血旺等。此外，还有杭州特色的点心，如桂花糕、荷花酥、艾窝窝等。
history: [('浙江的省会在哪里？', '浙江省的省会是杭州。'), ('这有什么好吃的？', '杭州市有很多著名的美食，例如西湖醋鱼、龙井虾仁、糖醋排骨、毛血旺等。此外，还有杭州特色的点心，如桂花糕、荷花酥、艾窝窝等。')]
"""
```

### qwen-vl-chat
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,
)
from swift.utils import seed_everything

model_type = ModelType.qwen_vl_chat
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')  # template_type: qwen

model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'})

template = get_template(template_type, tokenizer)
seed_everything(42)
query = tokenizer.from_list_format([
    {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'},
    {'text': '这是什么'},
])
response, history = inference(model, template, query)
print(f'query: {query}')
print(f'response: {response}')
query = '输出击掌的检测框'
response, history = inference(model, template, query, history)
print(f'query: {query}')
print(f'response: {response}')
print(f'history: {history}')
image = tokenizer.draw_bbox_on_latest_picture(response, history)
image.save('output_chat.jpg')
"""
query: Picture 1:<img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>
这是什么
response: 图中是一名女子在沙滩上和狗玩耍，旁边的狗是一只拉布拉多犬，它们处于沙滩上。
query: 输出击掌的检测框
response: <ref>击掌</ref><box>(523,513),(584,605)</box>
history: [('Picture 1:<img>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg</img>\n这是什么', '图中是一名女子在沙滩上和狗玩耍，旁边的狗是一只拉布拉多犬，它们处于沙滩上。'), ('输出击掌的检测框', '<ref>击掌</ref><box>(523,513),(584,605)</box>')]
"""
```

### qwen-audio-chat
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,
)
from swift.utils import seed_everything

model_type = ModelType.qwen_audio_chat
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')  # template_type: qwen

model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'})

template = get_template(template_type, tokenizer)

seed_everything(42)
query = tokenizer.from_list_format([
    {'audio': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/1272-128104-0000.flac'},
    {'text': 'what does the person say?'},
])
response, history = inference(model, template, query)
print(f'query: {query}')
print(f'response: {response}')
query = 'Find the start time and end time of the word "middle classes'
response, history = inference(model, template, query, history)
print(f'query: {query}')
print(f'response: {response}')
print(f'history: {history}')
"""Out[0]
query: Audio 1:<audio>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/1272-128104-0000.flac</audio>
what does the person say?
response: The person says: "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel".
query: Find the start time and end time of the word "middle classes
response: The word "middle classes" starts at <|2.33|> seconds and ends at <|3.26|> seconds.
history: [('Audio 1:<audio>https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/1272-128104-0000.flac</audio>\nwhat does the person say?', 'The person says: "mister quilter is the apostle of the middle classes and we are glad to welcome his gospel".'), ('Find the start time and end time of the word "middle classes', 'The word "middle classes" starts at <|2.33|> seconds and ends at <|3.26|> seconds.')]
"""
```

### chatglm3
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,
)
from swift.utils import seed_everything

model_type = ModelType.chatglm3_6b
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')  # template_type: chatglm3

model, tokenizer = get_model_tokenizer(model_type, model_kwargs={'device_map': 'auto'})
model.generation_config.max_new_tokens = 128

template = get_template(template_type, tokenizer)
seed_everything(42)
query = '浙江的省会在哪里？'
response, history = inference(model, template, query)
print(f'query: {query}')
print(f'response: {response}')
query = '这有什么好吃的？'
response, history = inference(model, template, query, history)
print(f'query: {query}')
print(f'response: {response}')
print(f'history: {history}')

"""Out[0]
response: 浙江有很多美食,以下是一些著名的:

1. 杭州小笼包:这是杭州著名的传统小吃,外皮薄而有韧性,内馅鲜美多汁。

2. 西湖醋鱼:这是杭州的名菜之一,用草鱼煮熟后,淋上特制的糟汁和醋,味道鲜美。

3. 浙江炖鸡:这是浙江省传统的名菜之一,用鸡肉加上姜、葱、酱油等调料慢慢炖煮而成,味道浓郁。

4. 油爆双脆:这是浙江省传统的糕点之一,外皮酥脆,内馅香甜
history: [('浙江的省会在哪里？', '浙江的省会是杭州。'), ('这有什么好吃的？', '浙江有很多美食,以下是一些著名的:\n\n1. 杭州小笼包:这是杭州著名的传统小吃,外皮薄而有韧性,内馅鲜美多汁。\n\n2. 西湖醋鱼:这是杭州的名菜之一,用草鱼煮熟后,淋上特制的糟汁和醋,味道鲜美。\n\n3. 浙江炖鸡:这是浙江省传统的名菜之一,用鸡肉加上姜、葱、酱油等调料慢慢炖煮而成,味道浓郁。\n\n4. 油爆双脆:这是浙江省传统的糕点之一,外皮酥脆,内馅香甜')]
"""
```


### bnb量化
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,
)
from swift.utils import seed_everything
from modelscope import BitsAndBytesConfig
import torch

model_type = ModelType.chatglm3_6b
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')  # template_type: chatglm3

torch_dtype = torch.bfloat16
quantization_config = BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True)
model, tokenizer = get_model_tokenizer(model_type, torch_dtype, {'device_map': 'auto',
                                      'quantization_config': quantization_config})
model.generation_config.max_new_tokens = 128
template = get_template(template_type, tokenizer)
seed_everything(42)
query = '浙江的省会在哪里？'
response, history = inference(model, template, query)
print(f'query: {query}')
print(f'response: {response}')
query = '这有什么好吃的？'
response, history = inference(model, template, query, history)
print(f'query: {query}')
print(f'response: {response}')
print(f'history: {history}')

"""Out[0]
query: 浙江的省会在哪里？
response: 浙江的省会是杭州。
query: 这有什么好吃的？
response: 浙江有很多美食,以下是一些著名的:

1. 杭州小笼包:这是杭州著名的传统小吃,外皮薄而有韧性,内馅鲜美多汁。

2. 浙江粽子:浙江粽子有多种口味,如咸蛋黄肉粽、豆沙粽等,其中以杭州粽子最为著名。

3. 油爆虾:这是浙江海鲜中的代表之一,用热油爆炒虾仁,口感鲜嫩。

4. 椒盐土豆丝:这是浙江传统的素菜之一,用土豆丝和椒盐一起炒制,口感清爽。

history: [('浙江的省会在哪里？', '浙江的省会是杭州。'), ('这有什么好吃的？', '浙江有很多美食,以下是一些著名的:\n\n1. 杭州小笼包:这是杭州著名的传统小吃,外皮薄而有韧性,内馅鲜美多汁。\n\n2. 浙江粽子:浙江粽子有多种口味,如咸蛋黄肉粽、豆沙粽等,其中以杭州粽子最为著名。\n\n3. 油爆虾:这是浙江海鲜中的代表之一,用热油爆炒虾仁,口感鲜嫩。\n\n4. 椒盐土豆丝:这是浙江传统的素菜之一,用土豆丝和椒盐一起炒制,口感清爽。\n')]
"""
```

### 使用CLI
```bash
# qwen
CUDA_VISIBLE_DEVICES=0 swift infer --model_type qwen-7b-chat
# yi
CUDA_VISIBLE_DEVICES=0 swift infer --model_type yi-6b-chat
```

### 微调后模型
如果你要使用微调后模型进行推理, 可以查看[LLM微调文档](LLM微调文档.md#微调后模型)


## Web-UI
### qwen-7b-chat
使用CLI:
```bash
CUDA_VISIBLE_DEVICES=0 swift app-ui --model_type qwen-7b-chat
```

使用python:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import AppUIArguments, ModelType, app_ui_main

app_ui_args = AppUIArguments(model_type=ModelType.qwen_7b_chat)
app_ui_main(app_ui_args)
```

使用bnb量化:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import AppUIArguments, ModelType, app_ui_main

app_ui_args = AppUIArguments(model_type=ModelType.qwen_7b_chat, quantization_bit=4)
app_ui_main(app_ui_args)
```

### qwen-7b
使用CLI:
```bash
CUDA_VISIBLE_DEVICES=0 swift app-ui --model_type qwen-7b
```

使用python:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import AppUIArguments, ModelType, app_ui_main

app_ui_args = AppUIArguments(model_type=ModelType.qwen_7b)
app_ui_main(app_ui_args)
```

### 微调后模型
使用微调后模型的web-ui可以查看[LLM微调文档](LLM微调文档.md#微调后模型)
