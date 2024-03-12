# Qwen-Audio 最佳实践

## 目录
- [环境准备](#环境准备)
- [推理](#推理)
- [微调](#微调)
- [微调后推理](#微调后推理)


## 环境准备
```shell
pip instal ms-swift[llm] -U
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
<<< multi-line
[INFO:swift] End multi-line input with `#`.
[INFO:swift] Input `single-line` to switch to single-line input mode.
<<<[M] 你是谁？#
我是来自达摩院的大规模语言模型，我叫通义千问。
--------------------------------------------------
<<<[M] Audio 1:<audio>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/music.wav</audio>
这是首什么样的音乐#
这是电子、实验流行风格的音乐。
--------------------------------------------------
<<<[M] Audio 1:<audio>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/weather.wav</audio>
这段语音说了什么#
这段语音中说了中文："今天天气真好呀"。
--------------------------------------------------
<<<[M] 这段语音是男生还是女生#
根据音色判断，这段语音是男性。
"""
```


## 微调
多模态大模型微调通常使用**自定义数据集**进行微调. 这里展示可直接运行的demo:

```shell
# Experimental environment: A10, 3090, V100...
# 22GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type qwen-audio-chat \
    --dataset aishell1-mini-zh \
```

[自定义数据集](../LLM/自定义与拓展.md#-推荐命令行参数的形式)支持json, jsonl样式, 以下是自定义数据集的例子:

(支持多轮对话, 支持每轮对话含多段语音或不含语音, 支持传入本地路径或URL)

```json
[
    {"conversations": [
        {"from": "user", "value": "Audio 1:<audio>audio_path</audio>\n11111"},
        {"from": "assistant", "value": "22222"}
    ]},
    {"conversations": [
        {"from": "user", "value": "Audio 1:<audio>audio_path</audio>\nAudio 2:<audio>audio_path2</audio>\nAudio 3:<audio>audio_path3</audio>\naaaaa"},
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


## 微调后推理

```shell
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/qwen-audio-chat/vx-xxx/checkpoint-xxx \
    --load_dataset_config true \
```
