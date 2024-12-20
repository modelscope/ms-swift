# Quick Start

ms-swift is a comprehensive training and deployment framework for large language models and multimodal large models, provided by the ModelScope Community. It currently supports the training (CPT, SFT, RLHF), inference, evaluation, quantization, and deployment of over 400 LLM and over 100 MLLM. Model developers can fulfill all kinds of needs related to large models in a single platform within the ms-swift framework. The main capabilities of ms-swift include:

- 🍎 Model Types: Supports the full process from training to deployment of over 400 text-based large models and over 100 multimodal large models, including All-to-All all-modality models.
- Dataset Types: Comes with more than 150 pre-built datasets for pre-training, fine-tuning, human alignment, multimodal, and supports custom datasets.
- Hardware Support: Compatible with CPU, RTX series, T4/V100, A10/A100/H100, Ascend NPU, and others.
- 🍊 Lightweight Training: Supports lightweight fine-tuning methods like LoRA, QLoRA, DoRA, LoRA+, ReFT, RS-LoRA, LLaMAPro, Adapter, GaLore, Q-Galore, LISA, UnSloth, Liger-Kernel, and more.
- Distributed Training: Supports distributed data parallel (DDP), simple model parallelism via device_map, DeepSpeed ZeRO2 ZeRO3, FSDP, and other distributed training technologies.
- Quantization Training: Provides training for quantized models like BNB, AWQ, GPTQ, AQLM, HQQ, EETQ.
- RLHF Training: Supports human alignment training methods like DPO, CPO, SimPO, ORPO, KTO, RM for both text-based and multimodal large models.
- 🍓 Multimodal Training: Capable of training models for different modalities such as images, videos, and audios; supports tasks like VQA (Visual Question Answering), Captioning, OCR (Optical Character Recognition), and Grounding.
- Interface-driven Training: Offers training, inference, evaluation, and quantization capabilities through an interface, enabling a complete workflow for large models.
- Plugins and Extensions: Allows customization and extension of models and datasets, and supports customizations for components like loss, metric, trainer, loss-scale, callback, optimizer, etc.
- 🍉 Toolbox Capabilities: Offers not only training support for large models and multi-modal large models but also covers the entire process of inference, evaluation, quantization, and deployment.
- Inference Acceleration: Supports inference acceleration engines like PyTorch, vLLM, LmDeploy, and provides OpenAI interface, accelerating inference, deployment, and evaluation modules.
- Model Evaluation: Uses EvalScope as the evaluation backend and supports evaluation of text-based and multimodal models with over 100 evaluation datasets.
- Model Quantization: Supports the export of quantized models in AWQ, GPTQ, and BNB formats, which can be accelerated using vLLM/LmDeploy for inference and support continued training.

## Installation

For the installation of ms-swift, please refer to the [installation documentation](./SWIFT-installation.md).

## Usage Example

10 minutes of self-cognition fine-tuning of Qwen2.5-7B-Instruct on a single 3090 GPU:

```shell
# 22GB
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset AI-ModelScope/alpaca-gpt4-data-zh#500 \
              AI-ModelScope/alpaca-gpt4-data-en#500 \
              swift/self-cognition#500 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot
```

After training is complete, use the following command to perform inference with the trained weights. The `--adapters` option should be replaced with the last checkpoint folder generated from the training. Since the adapters folder contains the parameter files from the training, there is no need to specify `--model` or `--system` separately.

```shell
# Using an interactive command line for inference.
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --stream true

# merge-lora and use vLLM for inference acceleration
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --stream true \
    --merge_lora true \
    --infer_backend vllm \
    --max_model_len 8192
```

> [!TIP]
> More examples can be found here: [examples](https://github.com/modelscope/ms-swift/tree/main/examples).
>
> Examples of training and inference using Python can be found in the [notebook](https://github.com/modelscope/ms-swift/tree/main/examples/notebook).
