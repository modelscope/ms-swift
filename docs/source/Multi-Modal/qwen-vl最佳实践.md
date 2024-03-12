
# Qwen-VL 最佳实践

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

推理[qwen-vl-chat](https://modelscope.cn/models/qwen/Qwen-VL-Chat/summary):
```shell
# Experimental environment: 3090
# 24GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift infer --model_type qwen-vl-chat
```

输出: (支持传入本地路径或URL)
```python
"""
<<< multi-line
[INFO:swift] End multi-line input with `#`.
[INFO:swift] Input `single-line` to switch to single-line input mode.
<<<[M] 你是谁？#
我是通义千问，由阿里云开发的AI助手。我被设计用来回答各种问题、提供信息和与用户进行对话。有什么我可以帮助你的吗？
--------------------------------------------------
<<<[M] Picture 1:<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png</img>
Picture 2:<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png</img>
这两张图片有什么区别#
两张图片的相同点是它们都是关于动物的插画，但是它们的动物不同。
第一张图片中的动物是绵羊，而第二张图片中的动物是小猫。
--------------------------------------------------
<<<[M] Picture 1:<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png</img>
图中有几只羊#
图中有四只羊。
--------------------------------------------------
<<<[M] Picture 1:<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/math.png</img>
计算结果是多少#
1452 + 45304 = 46756
--------------------------------------------------
<<<[M] Picture 1:<img>http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/poem.png</img>
根据图片中的内容写首诗#
湖面星光点点闪，孤舟独影静如眠。男子举灯照山谷，小猫陪伴在身边。
"""
```

示例图片如下:

cat:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png" width="250" style="display: inline-block;">

animal:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png" width="250" style="display: inline-block;">

math:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/math.png" width="250" style="display: inline-block;">

poem:

<img src="http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/poem.png" width="250" style="display: inline-block;">


## 微调
多模态大模型微调通常使用**自定义数据集**进行微调. 这里展示可直接运行的demo:

```shell
# Experimental environment: 3090
# 23GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type qwen-vl-chat \
    --dataset coco-mini-en \
```

[自定义数据集](../LLM/自定义与拓展.md#-推荐命令行参数的形式)支持json, jsonl样式, 以下是自定义数据集的例子:

(支持多轮对话, 支持每轮对话含多张图片或不含图片, 支持传入本地路径或URL)

```json
[
    {"conversations": [
        {"from": "user", "value": "Picture 1:<img>img_path</img>\n11111"},
        {"from": "assistant", "value": "22222"}
    ]},
    {"conversations": [
        {"from": "user", "value": "Picture 1:<img>img_path</img>\nPicture 2:<img>img_path2</img>\nPicture 3:<img>img_path3</img>\naaaaa"},
        {"from": "assistant", "value": "bbbbb"},
        {"from": "user", "value": "Picture 1:<img>img_path</img>\nccccc"},
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
    --ckpt_dir output/qwen-vl-chat/vx-xxx/checkpoint-xxx \
    --load_dataset_config true \
```
