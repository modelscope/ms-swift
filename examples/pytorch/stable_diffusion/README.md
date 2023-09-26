<h1 align="center">Stable Diffusion Example</h1>

<p align="center">
<a href="https://modelscope.cn/home">Modelscope Hub</a>
<br>
        <a href="README_CN.md">中文</a>&nbsp ｜ &nbspEnglish
</p>

## Features
1. Support Stable Diffusion [LoRA](https://arxiv.org/abs/2106.09685) method.
2. Support Stable Diffusion XL [LoRA](https://arxiv.org/abs/2106.09685) method.

## Prepare the Environment
```bash
pip install -r requirements.txt
```

## Train and Inference
```bash
# Clone the repository and enter the code directory.
git clone https://github.com/modelscope/swift.git

# Stable Diffusion LoRA
bash examples/pytorch/stable_diffusion/run_train_lora.sh

# Stable Diffusion XL LoRA
bash examples/pytorch/stable_diffusion/run_train_lora_xl.sh
```

## Extend Datasets
The [buptwq/lora-stable-diffusion-finetune](https://www.modelscope.cn/datasets/buptwq/lora-stable-diffusion-finetune/summary) dataset used in the example is from [ModelScope Hub](https://www.modelscope.cn/my/overview), you can replace different datasets ID by modifying the `train_dataset_name` parameter.
In addition, you can also use local datasets. Fill in the path of the dataset file in `train_dataset_name` parameter, which needs to include a `train.csv` file to map image files and text prompts. Please organize it into the following format:
```
Text,Target:FILE
[prompt], [image dir]
......
```

Here is an example of `train.csv` file:
```
Text,Target:FILE
a dog,target/00.jpg
a dog,target/01.jpg
a dog,target/02.jpg
a dog,target/03.jpg
a dog,target/04.jpg
```
