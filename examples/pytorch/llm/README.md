
<h1 align="center">LLM SFT Example</h1>

<p align="center">
<img src="https://img.shields.io/badge/python-%E2%89%A53.8-5be.svg">
<img src="https://img.shields.io/badge/pytorch-%E2%89%A51.12%20%7C%20%E2%89%A52.0-orange.svg">
<a href="https://github.com/modelscope/modelscope/"><img src="https://img.shields.io/badge/modelscope-%E2%89%A51.8.4-5D91D4.svg"></a>
<a href="https://github.com/modelscope/swift/"><img src="https://img.shields.io/badge/ms--swift-%E2%89%A51.0.0-6FEBB9.svg"></a>
</p>

<p align="center">
<a href="https://modelscope.cn/home">Modelscope Hub</a>
<br>
        <a href="README_CN.md">中文</a>&nbsp ｜ &nbspEnglish
</p>

## Features
1. supported sft method: [lora](https://arxiv.org/abs/2106.09685), [qlora](https://arxiv.org/abs/2305.14314), full(full parameter fine tuning), ...
2. supported models: qwen-7b, [qwen-7b-chat](https://github.com/QwenLM/Qwen-7B), qwen-vl, [qwen-vl-chat](https://github.com/QwenLM/Qwen-VL), baichuan-7b, baichuan-13b, baichuan-13b-chat, chatglm2-6b, chatglm2-6b-32k, llama2-7b, llama2-7b-chat, llama2-13b, llama2-13b-chat, llama2-70b, llama2-70b-chat, openbuddy-llama2-13b, openbuddy-llama-65b, polylm-13b
3. supported feature: quantization, ddp, model parallelism(device map), gradient checkpoint, gradient accumulation steps, push to modelscope hub, custom datasets, multimodal and agent sft, mutli-round chat, ...
4. supported datasets: alpaca-en(gpt4), alpaca-zh(gpt4), finance-en, multi-alpaca-all, code-en, instinwild-en, instinwild-zh, cot-en, cot-zh, coco-en, firefly-all-zh, poetry-zh, damo-agent-mini-zh, damo-agent-zh
5. supported templates: chatml(qwen), baichuan, chatglm2, llama, openbuddy_llama, default

## Prepare the Environment
Experimental environment: A10, 3090, A100, ... (V100 does not support bf16, quantization)
```bash
# Installing miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh

# Setting up a conda virtual environment
conda create --name ms-sft python=3.10
conda activate ms-sft

# Setting up a global pip mirror for faster downloads
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

pip install torch torchvision torchaudio -U
pip install sentencepiece charset_normalizer cpm_kernels tiktoken -U
pip install matplotlib scikit-learn tqdm tensorboard -U
pip install transformers datasets -U
pip install accelerate transformers_stream_generator -U

pip install ms-swift modelscope -U
# Recommended installation from source code for faster bug fixes
git clone https://github.com/modelscope/swift.git
cd swift
pip install -r requirements.txt
pip install .
# same as modelscope...(git clone ...)
```

## Run SFT and Inference
```bash
# Clone the repository and enter the code directory.
git clone https://github.com/modelscope/swift.git
cd swift/examples/pytorch/llm

# sft(qlora) and infer qwen-7b, Requires 16GB VRAM.
# If you want to use quantification, you need to `pip install bitsandbytes`
# If you want to push weights into modelscope hub during training, you need to set '--push_to_hub true'
bash scripts/qwen_7b_chat/qlora/sft.sh
bash scripts/qwen_7b_chat/qlora/infer.sh

# sft(qlora+ddp) and infer qwen-7b, Requires 4*16GB VRAM.
bash scripts/qwen_7b_chat/qlora_ddp/sft.sh
bash scripts/qwen_7b_chat/qlora_ddp/infer.sh

# sft(lora+ddp) and infer qwen-7b, Requires 4*22GB VRAM.
bash scripts/qwen_7b_chat/lora_ddp/sft.sh
bash scripts/qwen_7b_chat/lora_ddp/infer.sh

# sft(full) and infer qwen-7b, Requires 95GB VRAM.
bash scripts/qwen_7b_chat/full/sft.sh
bash scripts/qwen_7b_chat/full/infer.sh

# For more scripts, please see `scripts/` folder
```

## Extend Datasets
1. If you need to extend the model, you can modify the `MODEL_MAPPING` in `utils/model.py`. `model_id` can be specified as a local path. In this case, `revision` doesn't work.
2. If you need to extend or customize the dataset, you can modify the `DATASET_MAPPING` in `utils/dataset.py`. You need to customize the `get_*_dataset` function, which returns a dataset with two columns: `query`, `response`.
3. If you need to extend the template, you can modify the `TEMPLATE_MAPPING` in `utils/preprocess.py`.
