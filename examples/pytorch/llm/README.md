
<h1 align="center">LLM SFT Example</h1>

<p align="center">
<img src="https://img.shields.io/badge/python-%E2%89%A53.8-5be.svg">
<img src="https://img.shields.io/badge/pytorch-%E2%89%A51.12%20%7C%20%E2%89%A52.0-orange.svg">
<img src="https://img.shields.io/badge/modelscope-%E2%89%A51.8.1-5D91D4.svg">
<img src="https://img.shields.io/badge/ms--swift-%E2%89%A51.0.0-6FEBB9.svg">
</p>

<p align="center">
<a href="https://modelscope.cn/home">Modelscope Hub</a>
<br>
        <a href="README_CN.md">中文</a>&nbsp ｜ &nbspEnglish
</p>

## Features
1. supported sft method: lora, qlora, full, ...
2. supported models: **qwen-7b**, baichuan-7b, baichuan-13b, chatglm2-6b, llama2-7b, llama2-13b, llama2-70b, openbuddy-llama2-13b, ...
3. supported feature: quantization, ddp, model parallelism(device map), gradient checkpoint, gradient accumulation steps, push to modelscope hub, custom datasets, notebook compatibility, tensorboard, warmup, lr scheduler, resume from ckpt, ...
4. supported datasets: alpaca-en(gpt4), alpaca-zh(gpt4), finance-en, multi-alpaca-all, multi-alpaca-ar, multi-alpaca-de, multi-alpaca-es, multi-alpaca-fr, multi-alpaca-id, multi-alpaca-ja, multi-alpaca-ko, multi-alpaca-pt, multi-alpaca-ru, multi-alpaca-th, multi-alpaca-vi, code-en, instinwild-en, instinwild-zh

## Prepare the Environment
```bash
# Please note the cuda version
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

pip install sentencepiece charset_normalizer cpm_kernels tiktoken -U
pip install matplotlib tqdm tensorboard -U
pip install transformers datasets -U
pip install accelerate transformers_stream_generator -U

# Recommended installation from source code for faster bug fixes
git clone https://github.com/modelscope/swift.git
cd swift
pip install -r requirements.txt
pip install .
# same as modelscope...(git clone ...)
# You can also install it from pypi
pip install ms-swift modelscope -U
```

## Run SFT and Inference
```bash
git clone https://github.com/modelscope/swift.git
cd swift/examples/pytorch/llm

# sft(qlora) and infer qwen-7b
bash script/qlora_qwen_7b/sft.sh
bash script/qlora_qwen_7b/infer.sh

# sft(qlora+ddp) and infer qwen-7b
bash script/qlora_ddp_qwen_7b/sft.sh
bash script/qlora_ddp_qwen_7b/infer.sh

# sft(full) and infer qwen-7b
bash script/full_qwen_7b/sft.sh
bash script/full_qwen_7b/infer.sh
```

## Extend Datasets
1. If you need to extend the model, you can modify the `MODEL_MAPPING` in `utils/models.py`. `model_id` can be specified as a local path. In this case, `revision` doesn't work.
2. If you need to extend or customize the dataset, you can modify the `DATASET_MAPPING` in `utils/datasets.py`. You need to customize the `get_*_dataset` function, which returns a dataset with two columns: `instruction`, `output`.


## TODO
1. Support multi-round
2. RLHF
3. more models: Qwen-7B-Chat (use same prompt)
4. more datasets
5. metrics
6. ...
