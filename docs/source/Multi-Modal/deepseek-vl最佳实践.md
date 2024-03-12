
# Deepseek-VL 最佳实践

## 目录
- [环境准备](#环境准备)
- [推理](#推理)
- [微调](#微调)
- [微调后推理](#微调后推理)


## 环境准备
```shell
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e .[llm]
```

## 推理

推理[deepseek-vl-7b-chat](https://www.modelscope.cn/models/deepseek-ai/deepseek-vl-7b-chat/summary):

```shell
# Experimental environment: A100
# 30GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift infer --model_type deepseek-vl-7b-chat

# 如果你想在3090上运行, 可以推理1.3b模型
CUDA_VISIBLE_DEVICES=0 swift infer --model_type deepseek-vl-1_3b-chat
```

7b模型效果展示: (支持传入本地路径或URL)
```python
"""
<<< 描述这种图片
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png
这幅图片捕捉到了一只小猫咪迷人的场景，它的眼睛睁得大大的，带着好奇心。小猫的毛发是白色和灰色的混合，给它一种几乎是空灵的外观。它的耳朵尖尖的，警觉地向上指，而它的鼻子是柔和的粉红色。小猫的眼睛是醒目的蓝色，充满了天真和好奇。小猫舒适地坐在一块白色的布料上，与它的灰色和白色毛发形成了美丽的对比。背景模糊，使人们的焦点集中在小猫的脸上，突出了它特征的精细细节。这幅图片散发出一种温暖和柔软的感觉，捕捉到了小猫的纯真和魅力。
--------------------------------------------------
<<< 图中有几只羊
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png
图片中有一只羊。
--------------------------------------------------
<<< 计算结果是多少
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/math.png
计算结果是1452 + 45304 = 46756。
--------------------------------------------------
<<< 根据图片中的内容写首诗
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/poem.png
星辉洒落湖面静，
独舟轻摇夜风中。
灯火摇曳星光里，
心随波涛逐梦行。
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
# Experimental environment: A10, 3090, V100
# 20GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type deepseek-vl-7b-chat \
    --dataset coco-mini-en-2 \
```

自定义数据集支持json, jsonl样式, 以下是自定义数据集的例子:

(支持多轮对话, 每轮对话必须包含一张图片, 支持传入本地路径或URL)

```jsonl
{"query": "55555", "response": "66666", "images": ["image_path"]}
{"query": "eeeee", "response": "fffff", "history": [], "images": ["image_path"]}
{"query": "EEEEE", "response": "FFFFF", "history": [["AAAAA", "BBBBB"], ["CCCCC", "DDDDD"]], "images": ["image_path", "image_path2", "image_path3"]}
```


## 微调后推理

```shell
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/deepseek-vl-7b-chat/vx-xxx/checkpoint-xxx \
    --load_dataset_config true \
```
