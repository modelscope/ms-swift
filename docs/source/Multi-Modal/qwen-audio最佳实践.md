# Qwen-Audio 最佳实践

Qwen2-Audio的最佳实践可以查看: [https://github.com/modelscope/ms-swift/issues/1653](https://github.com/modelscope/ms-swift/issues/1653)

## 目录
- [环境准备](#环境准备)
- [推理](#推理)
- [微调](#微调)
- [微调后推理](#微调后推理)


## 环境准备
```shell
pip install 'ms-swift[llm]' -U
```

## 推理

推理[qwen-audio-chat](https://modelscope.cn/models/qwen/Qwen-Audio-Chat/summary):
```shell
# Experimental environment: A10, 3090, V100...
# 21GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift infer --model_type qwen-audio-chat
```

输出: (支持传入本地路径或URL)
```python
"""
<<< 你是谁？
我是来自达摩院的大规模语言模型，我叫通义千问。
--------------------------------------------------
<<< <audio>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/music.wav</audio>这是首什么样的音乐
这是一首风格是Pop的音乐。
--------------------------------------------------
<<< <audio>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/weather.wav</audio>这段语音说了什么
这段语音中说了中文："今天天气真好呀"。
--------------------------------------------------
<<< 这段语音是男生还是女生
根据音色判断，这段语音是男性。
"""
```

**单样本推理**

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type, inference_stream
)
from swift.utils import seed_everything
import torch

model_type = ModelType.qwen_audio_chat
template_type = get_default_template_type(model_type)
print(f'template_type: {template_type}')

model, tokenizer = get_model_tokenizer(model_type, torch.float16,
                                       model_kwargs={'device_map': 'auto'})
model.generation_config.max_new_tokens = 256
template = get_template(template_type, tokenizer)
seed_everything(42)

query = '<audio>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/weather.wav</audio>这段语音说了什么'
response, history = inference(model, template, query)
print(f'query: {query}')
print(f'response: {response}')

# 流式
query = '这段语音是男生还是女生'
gen = inference_stream(model, template, query, history)
print_idx = 0
print(f'query: {query}\nresponse: ', end='')
for response, history in gen:
    delta = response[print_idx:]
    print(delta, end='', flush=True)
    print_idx = len(response)
print()
print(f'history: {history}')
"""
query: <audio>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/weather.wav</audio>这段语音说了什么
response: 这段语音说了中文："今天天气真好呀"。
query: 这段语音是男生还是女生
response: 根据音色判断，这段语音是男性。
history: [['<audio>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/weather.wav</audio>这段语音说了什么', '这段语音说了中文："今天天气真好呀"。'], ['这段语音是男生还是女生', '根据音色判断，这段语音是男性。']]
"""
```


## 微调
多模态大模型微调通常使用**自定义数据集**进行微调. 这里展示可直接运行的demo:

LoRA微调:

```shell
# Experimental environment: A10, 3090, V100...
# 22GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type qwen-audio-chat \
    --dataset aishell1-mini-zh \
```

全参数微调:
```shell
# MP
# Experimental environment: 2 * A100
# 2 * 50 GPU memory
CUDA_VISIBLE_DEVICES=0,1 swift sft \
    --model_type qwen-audio-chat \
    --dataset aishell1-mini-zh \
    --sft_type full \

# ZeRO2
# Experimental environment: 4 * A100
# 4 * 80 GPU memory
NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 swift sft \
    --model_type qwen-audio-chat \
    --dataset aishell1-mini-zh \
    --sft_type full \
    --use_flash_attn true \
    --deepspeed default-zero2
```

[自定义数据集](../Instruction/自定义与拓展.md#-推荐命令行参数的形式)支持json, jsonl样式, 以下是自定义数据集的例子:

(支持多轮对话, 支持每轮对话含多段语音或不含语音, 支持传入本地路径或URL)

```json
[
    {"conversations": [
        {"from": "user", "value": "<audio>audio_path</audio>11111"},
        {"from": "assistant", "value": "22222"}
    ]},
    {"conversations": [
        {"from": "user", "value": "<audio>audio_path</audio><audio>audio_path2</audio><audio>audio_path3</audio>aaaaa"},
        {"from": "assistant", "value": "bbbbb"},
        {"from": "user", "value": "<audio>audio_path</audio>ccccc"},
        {"from": "assistant", "value": "ddddd"}
    ]},
    {"conversations": [
        {"from": "user", "value": "AAAAA"},
        {"from": "assistant", "value": "BBBBB"},
        {"from": "user", "value": "CCCCC"},
        {"from": "assistant", "value": "DDDDD"}
    ]}
]
```


## 微调后推理
直接推理:
```shell
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/qwen-audio-chat/vx-xxx/checkpoint-xxx \
    --load_dataset_config true \
```

**merge-lora**并推理:
```shell
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir output/qwen-audio-chat/vx-xxx/checkpoint-xxx \
    --merge_lora true

CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/qwen-audio-chat/vx-xxx/checkpoint-xxx-merged \
    --load_dataset_config true
```
