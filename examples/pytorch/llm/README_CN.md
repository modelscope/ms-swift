# 大模型微调的例子
<p align="center">
<a href="https://modelscope.cn/home">魔搭社区</a>
<br>
        中文&nbsp ｜ &nbsp<a href="examples/pytorch/llm/README.md">English</a>
</p>


## 特性
1. 支持的sft方法: lora, qlora, 全参数微调, ...
2. 支持的模型: **qwen-7b**, baichuan-7b, baichuan-13b, chatglm2-6b, llama2-7b, llama2-13b, llama2-70b, openbuddy-llama2-13b, ...
3. 支持的特性: 模型量化, DDP, 模型并行(device_map), gradient checkpoint, 梯度累加, 支持推送modelscope hub, 支持自定义数据集, 兼容notebook, tensorboard, warmup, lr scheduler, 断点续训, ...
4. 支持的数据集: alpaca-en(gpt4), alpaca-zh(gpt4), finance-en, multi-alpaca-all, multi-alpaca-ar, multi-alpaca-de, multi-alpaca-es, multi-alpaca-fr, multi-alpaca-id, multi-alpaca-ja, multi-alpaca-ko, multi-alpaca-pt, multi-alpaca-ru, multi-alpaca-th, multi-alpaca-vi, code-en, instinwild-en, instinwild-zh


## 准备实验环境
```bash
# 请注意修改cuda的版本
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

pip install sentencepiece charset_normalizer cpm_kernels tiktoken -U
pip install matplotlib tqdm tensorboard -U
pip install transformers datasets -U
pip install accelerate transformers_stream_generator -U

# 推荐从源码安装swift和modelscope, 这具有更多的特性和更快的bug修复
git clone https://github.com/modelscope/swift.git
cd swift
pip install -r requirements.txt
pip install .
# modelscope类似...(git clone ...)
# 当然, 你也可以从pypi上下载
pip install ms-swift modelscope -U
```

## 微调和推理
```bash
git clone https://github.com/modelscope/swift.git
cd swift/examples/pytorch/llm

# 微调(qlora)+推理 qwen-7b
bash script/qlora_qwen_7b/sft.sh
bash script/qlora_qwen_7b/infer.sh

# 微调(qlora+ddp)+推理 qwen-7b
bash script/qlora_ddp_qwen_7b/sft.sh
bash script/qlora_ddp_qwen_7b/infer.sh

# 微调(full)+推理 qwen-7b
bash script/full_qwen_7b/sft.sh
bash script/full_qwen_7b/infer.sh
```

## 拓展数据集
1. 如果你想要拓展模型, 你可以修改`utils/models.py`文件中的`MODEL_MAPPING`. `model_id`可以指定为本地路径, 这种情况下, `revision`参数不起作用.
2. 如果你想要拓展或使用自定义数据集, 你可以修改`utils/datasets.py`文件中的`DATASET_MAPPING`. 你需要自定义`get_*_dataset`函数, 并返回包含`instruction`, `output`两列的数据集.

## TODO
1. 支持 多轮对话
2. RLHF
3. 支持更多的模型: Qwen-7B-Chat (使用相同的prompt)
4. 支持更多的数据集
5. 指标与评估
6. ...