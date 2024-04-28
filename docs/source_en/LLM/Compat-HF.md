# HuggingFace Eco-compatibility
By default, we use models and datasets from [ModelScope](https://modelscope.cn/my/overview) for fine-tuning and inference. However, considering that overseas users are more familiar with the [HuggingFace](https://huggingface.co/) ecosystem, we have made it compatible with HuggingFace.

To enable HuggingFace compatibility, you need to set the environment variable `USE_HF=1`. Supported HuggingFace models and datasets can be found in the [Supported Models and Datasets](Supported-models-datasets.md). Note that some datasets are only supported in the ModelScope environment.

Here is an example inference script for qwen1.5-7b-chat:
```shell
# Experimental Environment: A10, 3090, V100
USE_HF=1 CUDA_VISIBLE_DEVICES=0 swift infer --model_type qwen1half-7b-chat
```

Fine-tuning script:
```shell
# Experimental Environment: 2 * A100
# GPU Memory Requirement: 2 * 30GB
USE_HF=1 \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
swift sft \
    --model_type qwen1half-7b-chat \
    --dataset blossom-math-zh \
    --num_train_epochs 5 \
    --sft_type lora \
    --output_dir output \
```

Please refer to other documents for inference after fine-tuning, and deployment .
