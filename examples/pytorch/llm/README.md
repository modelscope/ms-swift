# LLM Example
1. supported feature: quantization, ddp, model parallelism(device map), gradient checkpoint, gradient accumulation steps, warmup, lr_scheduler, push to modelscope hub, easy to extend models and datasets, tensorboard(and plot tb-like images after training), notebook compatibility, resume from ckpt, custom prompt, ...
2. supported models: baichuan-7b, baichuan-13b, chatglm2-6b, llama2-7b, llama2-13b, **llama2-70b**(when quantization_bit=4, only 44GB of memory is required), openbuddy-llama2-13b, **qwen-7b**, ...
3. supported datasets: alpaca-en(gpt4), alpaca-zh(gpt4), finance-en, ...
4. supported sft method: lora, qlora, full, ...
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
# same as modelscope...
# You can also install it from pypi
pip install ms-swift modelscope -U
```

## Run sft and inference
```bash
git clone https://github.com/modelscope/swift.git
cd swift/examples/pytorch/llm
# sft
bash run_sft.sh
# inference
bash run_infer.sh
```

## Extend models and datasets
1. If you need to extend or customize the model, you can modify the `MODEL_MAPPING` in `utils/models.py`. model_id can be specified as a local path. In this case, 'revision' doesn't work.
2. If you need to extend or customize the dataset, you can modify the `DATASET_MAPPING` in `utils/dataset.py`. You need to customize the `get_*_dataset` function, which returns a dataset with two columns: `instruction`, `output`.
