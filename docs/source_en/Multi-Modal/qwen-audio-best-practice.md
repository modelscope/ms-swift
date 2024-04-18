# Qwen-Audio Best Practices

## Table of Contents
- [Environment Setup](#environment-setup)
- [Inference](#inference)
- [Fine-tuning](#fine-tuning)
- [Inference After Fine-tuning](#inference-after-fine-tuning)

## Environment Setup
```shell
pip install 'ms-swift[llm]' -U
```

## Inference

Inference with [qwen-audio-chat](https://modelscope.cn/models/qwen/Qwen-Audio-Chat/summary):
```shell
# Experimental environment: A10, 3090, V100...
# 21GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift infer --model_type qwen-audio-chat
```

Output: (supports passing local path or URL)
```python
"""
<<< multi-line
[INFO:swift] End multi-line input with `#`.
[INFO:swift] Input `single-line` to switch to single-line input mode.
<<<[M] Who are you?#
I am a large language model from DAMO Academy, my name is Tongyee Qianwen.
--------------------------------------------------
<<<[M] Audio 1:<audio>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/music.wav</audio>
What kind of music is this?#
This is electronic, experimental pop style music.
--------------------------------------------------
<<<[M] Audio 1:<audio>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/weather.wav</audio>
What did this speech say?#
This speech said in Chinese: "The weather is really nice today".
--------------------------------------------------
<<<[M] Is this speech male or female?#
Based on the timbre, this speech is male.
"""
```

**Single-sample Inference**

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

query = """Audio 1:<audio>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/weather.wav</audio>
What did this speech say"""
response, history = inference(model, template, query)
print(f'query: {query}')
print(f'response: {response}')

# Streaming
query = 'Is this speech male or female'
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
query: Audio 1:<audio>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/weather.wav</audio>
What did this speech say
response: This speech said in Chinese: "The weather is really nice today".
query: Is this speech male or female
response: Based on the timbre, this speech is male.
history: [['Audio 1:<audio>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/weather.wav</audio>\nWhat did this speech say',
'This speech said in Chinese: "The weather is really nice today".'], ['Is this speech male or female', 'Based on the timbre, this speech is male.']]
"""
```

## Fine-tuning
Multimodal large model fine-tuning usually uses **custom datasets** for fine-tuning. Here shows a demo that can be run directly:

LoRA fine-tuning:

(By default, only the qkv of the LLM part is lora fine-tuned. If you want to fine-tune all linear including the audio model part, you can specify `--lora_target_modules ALL`)
```shell
# Experimental environment: A10, 3090, V100...
# 22GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type qwen-audio-chat \
    --dataset aishell1-mini-zh \
```

Full-parameter fine-tuning:
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
# 2 * 80 GPU memory
NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3 swift sft \
    --model_type qwen-audio-chat \
    --dataset aishell1-mini-zh \
    --sft_type full \
    --use_flash_attn true \
    --deepspeed default-zero2
```

[Custom datasets](../LLM/Customization.md#-Recommended-Command-line-arguments)  supports json, jsonl styles, the following is an example of a custom dataset:

(Supports multi-turn conversations, supports each turn of conversation containing multiple or no audio segments, supports passing local paths or URLs)

```json
[
    {"conversations": [
        {"from": "user", "value": "Audio 1:<audio>audio_path</audio>\n11111"},
        {"from": "assistant", "value": "22222"}
    ]},
    {"conversations": [
        {"from": "user", "value": "Audio 1:<audio>audio_path</audio>\nAudio 2:<audio>audio_path2</audio>\nAudio 3: <audio>audio_path3</audio>\naaaaa"},
        {"from": "assistant", "value": "bbbbb"},
        {"from": "user", "value": "Audio 1:<audio>audio_path</audio>\nccccc"},
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

## Inference After Fine-tuning
Direct inference:
```shell
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/qwen-audio-chat/vx-xxx/checkpoint-xxx \
    --load_dataset_config true \
```

**merge-lora** and inference:
```shell
CUDA_VISIBLE_DEVICES=0 swift export \
    --ckpt_dir output/qwen-audio-chat/vx-xxx/checkpoint-xxx \
    --merge_lora true

CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/qwen-audio-chat/vx-xxx/checkpoint-xxx-merged \
    --load_dataset_config true
```
