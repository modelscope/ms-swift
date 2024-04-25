# HuggingFace生态兼容
默认我们会使用[ModelScope](https://modelscope.cn/my/overview)中的模型和数据集进行微调和推理。但是考虑到海外用户更熟悉[HuggingFace](https://huggingface.co/)生态，这里对其进行兼容。

你需要设置环境变量`USE_HF=1`，支持的HuggingFace模型和数据集可以参考[支持的模型和数据集](支持的模型和数据集.md)，部分数据集只支持在ModelScope环境下使用。

以下是对`qwen1.5-7b-chat`的推理脚本:
```shell
# Experimental Environment: A10, 3090, V100
USE_HF=1 CUDA_VISIBLE_DEVICES=0 swift infer --model_type qwen1half-7b-chat
```

微调脚本:
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

微调后推理与部署等内容参考其他文档.
