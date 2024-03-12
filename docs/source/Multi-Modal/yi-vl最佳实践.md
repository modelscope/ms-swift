
# Yi-VL 最佳实践

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

推理[yi-vl-6b-chat](https://modelscope.cn/models/01ai/Yi-VL-6B/summary):
```shell
# Experimental environment: A10, 3090, V100...
# 18GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift infer --model_type yi-vl-6b-chat
```

输出: (支持传入本地路径或URL)
```python
"""
<<< 描述这种图片
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png
图片显示一只小猫坐在地板上,眼睛睁开,凝视着摄像机。小猫看起来很可爱,有灰色和白色的毛皮,以及蓝色的眼睛。它似乎正在看摄像机,可能对周围环境很好奇。
--------------------------------------------------
<<< 图中有几只羊
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png
图中有四只羊.
--------------------------------------------------
<<< 计算结果是多少
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/math.png
1452 + 45304 = ?
--------------------------------------------------
<<< 根据图片中的内容写首诗
Input a media path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/poem.png
星光闪烁,照亮了宁静的夜晚,
在湖上,一艘小船静静地漂浮着,
船上有灯,像星星一样闪耀,
船上有两个人,在星光下静静地坐着。

森林的影子在湖边摇曳,
在星光下,它显得神秘而美丽。
船在湖中,在星光下,
在星光下,它显得神秘而美丽。

星光闪烁,照亮了宁静的夜晚,
在湖上,一艘小船静静地漂浮着,
船上有灯,像星星一样闪耀,
船上有两个人,在星光下静静地坐着。
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
# Experimental environment: A10, 3090, V100...
# 19GB GPU memory
CUDA_VISIBLE_DEVICES=0 swift sft \
    --model_type yi-vl-6b-chat \
    --dataset coco-mini-en-2 \
```

[自定义数据集](../LLM/自定义与拓展.md#-推荐命令行参数的形式)支持json, jsonl样式, 以下是自定义数据集的例子:

(支持多轮对话, 每轮对话必须包含一张图片, 支持传入本地路径或URL)

```jsonl
{"query": "55555", "response": "66666", "images": ["image_path"]}
{"query": "eeeee", "response": "fffff", "history": [], "images": ["image_path"]}
{"query": "EEEEE", "response": "FFFFF", "history": [["AAAAA", "BBBBB"], ["CCCCC", "DDDDD"]], "images": ["image_path", "image_path2", "image_path3"]}
```


## 微调后推理

```shell
CUDA_VISIBLE_DEVICES=0 swift infer \
    --ckpt_dir output/yi-vl-6b-chat/vx-xxx/checkpoint-xxx \
    --load_dataset_config true \
```
