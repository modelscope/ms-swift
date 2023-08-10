# LLM Example
1. supported sft method: lora, qlora, full, ...
2. supported models: **qwen-7b**, baichuan-7b, baichuan-13b, chatglm2-6b, llama2-7b, llama2-13b, llama2-70b, openbuddy-llama2-13b, ...
3. supported feature: quantization, ddp, model parallelism(device map), gradient checkpoint, gradient accumulation steps, push to modelscope hub, custom datasets, notebook compatibility, tensorboard, warmup, lr_scheduler, easy to extend models, resume from ckpt, custom prompt, ...
4. supported datasets: alpaca-en(gpt4), alpaca-zh(gpt4), finance-en, multi-alpaca-all, multi-alpaca-ar, multi-alpaca-de, multi-alpaca-es, multi-alpaca-fr, multi-alpaca-id, multi-alpaca-ja, multi-alpaca-ko, multi-alpaca-pt, multi-alpaca-ru', multi-alpaca-th, multi-alpaca-vi, code-en, instinwild-zh, instinwild-en
5. todo: metrics(ROUGE, BELU), multi-round, RLHF, more models and datasets, ...

## Prepare the environment
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install sentencepiece charset_normalizer cpm_kernels tiktoken -U
pip install matplotlib scikit-learn -U
pip install transformers datasets -U
pip install tqdm tensorboard torchmetrics -U
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

## Run sft and inference
```bash
git clone https://github.com/modelscope/swift.git
cd swift/examples/pytorch/llm

# sft(qlora) and infer qwen-7b
bash script/qlora_qwen_7b//sft.sh
bash script/qlora_qwen_7b//infer.sh

# sft(qlora+ddp) and infer qwen-7b
bash script/qlora_ddp_qwen_7b//sft.sh
bash script/qlora_ddp_qwen_7b//infer.sh

# sft(full) and infer qwen-7b
bash script/full_qwen_7b/sft.sh
bash script/full_qwen_7b/infer.sh
```

## Extend models and datasets
1. If you need to extend or customize the model, you can modify the `MODEL_MAPPING` in `utils/models.py`. model_id can be specified as a local path. In this case, 'revision' doesn't work.
2. If you need to extend or customize the dataset, you can modify the `DATASET_MAPPING` in `utils/dataset.py`. You need to customize the `get_*_dataset` function, which returns a dataset with two columns: `instruction`, `output`.
