<h1 align="center">微调稳定扩散模型例子</h1>

<p align="center">
<a href="https://modelscope.cn/home">魔搭社区</a>
<br>
        中文&nbsp ｜ &nbsp<a href="README.md">English</a>
</p>


## 特性
1. 支持[LoRA](https://arxiv.org/abs/2106.09685)方法微调稳定扩散模型。
2. 支持[LoRA](https://arxiv.org/abs/2106.09685)方法微调XL版本的稳定扩散模型。

## 环境准备
```bash
pip install -r requirements.txt
```

## 训练和推理
```bash
# 克隆代码库并进入代码目录
git clone https://github.com/modelscope/swift.git

# LoRA方法微调和推理稳定扩散模型
bash examples/pytorch/stable_diffusion/run_train_lora.sh

# LoRA方法微调和推理XL版本的稳定扩散模型
bash examples/pytorch/stable_diffusion/run_train_lora_xl.sh
```

## 数据集拓展
示例中使用的数据集[buptwq/lora-stable-diffusion-finetune](https://www.modelscope.cn/datasets/buptwq/lora-stable-diffusion-finetune/summary)来自[ModelScope Hub](https://www.modelscope.cn/my/overview)，您可以在ModelScope Hub选择其他数据集，用被选择的数据集ID来修改`train_dataset_name`参数。

除此之外，您也可以使用本地数据集。请用本地数据集路径修改`train_dataset_name`参数，请注意在本地数据集路径中应该包含一个`train.csv`文件用来映射图片和文本提示词。`train.csv`文件请参照以下的格式：
```
Text,Target:FILE
[提示词], [图片路径]
......
```
下面是一个 `train.csv` 文件的例子:
```
Text,Target:FILE
a dog,target/00.jpg
a dog,target/01.jpg
a dog,target/02.jpg
a dog,target/03.jpg
a dog,target/04.jpg
```
