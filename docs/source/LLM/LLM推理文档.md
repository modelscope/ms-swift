# LLM推理文档
## 环境准备
GPU设备: A10, 3090, V100, A100均可.
```bash
# 设置pip全局镜像
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# 安装ms-swift
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e .
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
import torch

model_type = ModelType.qwen_7b_chat
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')  # template_type: chatml


kwargs = {}
# kwargs['use_flash_attn'] = True  # 使用flash_attn

model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16, {'device_map': 'auto'}, **kwargs)
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
you are a helpful assistant!<|im_end|>
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

### qwen-7b
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,
)
from swift.utils import seed_everything
import torch

model_type = ModelType.qwen_7b
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')  # template_type: default-generation

model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16, {'device_map': 'auto'})
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
import torch

model_type = ModelType.qwen_7b_chat
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')  # template_type: chatml

model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16, {'device_map': 'auto'})
model.generation_config.max_new_tokens = 128

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

### 量化
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type,
)
from swift.utils import seed_everything
from modelscope import BitsAndBytesConfig
import torch

model_type = ModelType.qwen_7b_chat
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')  # template_type: chatml

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
response: 浙江省会是杭州。
query: 这有什么好吃的？
response: 浙江有许多美食，比如西湖醋鱼、龙井虾仁、东坡肉、梅干菜烧肉等，这些都是浙江地区非常有名的食物。此外，浙江还盛产海鲜，如螃蟹、海螺、贝壳类和各种鱼类。
history: [('浙江的省会在哪里？', '浙江省会是杭州。'), ('这有什么好吃的？', '浙江有许多美食，比如西湖醋鱼、龙井虾仁、东坡肉、梅干菜烧肉等，这些都是浙江地区非常有名的食物。此外，浙江还盛产海鲜，如螃蟹、海螺、贝壳类和各种鱼类。')]
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
import torch

model_type = ModelType.chatglm3_6b
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')  # template_type: chatglm3

model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16, {'device_map': 'auto'})
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
response: 浙江有很多美食,以下是一些著名的:

1. 杭州小笼包:这是杭州著名的传统小吃,外皮薄而有韧性,内馅鲜美多汁。

2. 西湖醋鱼:这是杭州的名菜之一,用草鱼煮熟后,淋上特制的糟汁和醋,味道鲜美。

3. 浙江炖鸡:这是浙江省传统的名菜之一,用鸡肉加上姜、葱、酱油等调料慢慢炖煮而成,味道浓郁。

4. 油爆双脆:这是浙江省传统的糕点之一,外皮酥脆,内馅香甜
history: [('浙江的省会在哪里？', '浙江的省会是杭州。'), ('这有什么好吃的？', '浙江有很多美食,以下是一些著名的:\n\n1. 杭州小笼包:这是杭州著名的传统小吃,外皮薄而有韧性,内馅鲜美多汁。\n\n2. 西湖醋鱼:这是杭州的名菜之一,用草鱼煮熟后,淋上特制的糟汁和醋,味道鲜美。\n\n3. 浙江炖鸡:这是浙江省传统的名菜之一,用鸡肉加上姜、葱、酱油等调料慢慢炖煮而成,味道浓郁。\n\n4. 油爆双脆:这是浙江省传统的糕点之一,外皮酥脆,内馅香甜')]
"""
```

## Web-UI
### qwen-7b-chat
使用CLI
```bash
CUDA_VISIBLE_DEVICES=0 swift web-ui --model_id_or_path qwen/Qwen-7B-Chat
```

使用python
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm.run import web_ui_main
from swift.llm import InferArguments, ModelType

infer_args = InferArguments(model_type=ModelType.qwen_7b_chat)
web_ui_main(infer_args)
```

使用量化
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm.run import web_ui_main
from swift.llm import InferArguments, ModelType

infer_args = InferArguments(model_type=ModelType.qwen_7b_chat, quantization_bit=4)
web_ui_main(infer_args)
```

### qwen-7b
使用CLI
```bash
swift web-ui --model_id_or_path qwen/Qwen-7B
```

使用python
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm.run import web_ui_main
from swift.llm import InferArguments, ModelType

infer_args = InferArguments(model_type=ModelType.qwen_7b)
web_ui_main(infer_args)
```
