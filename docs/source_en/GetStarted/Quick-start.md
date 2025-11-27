# Quick Start

🍲 **ms-swift** is a large model and multimodal large model fine-tuning and deployment framework provided by the ModelScope community. It now supports training (pre-training, fine-tuning, human alignment), inference, evaluation, quantization, and deployment for 600+ text-only large models and 300+ multimodal large models. Large models include: Qwen3, Qwen3-Next, InternLM3, GLM4.5, Mistral, DeepSeek-R1, Llama4, etc. Multimodal large models include: Qwen3-VL, Qwen3-Omni, Llava, InternVL3.5, MiniCPM-V-4, Ovis2.5, GLM4.5-V, DeepSeek-VL2, etc.

🍔 In addition, ms-swift integrates the latest training technologies, including Megatron parallelism techniques such as TP, PP, CP, EP to accelerate training, as well as numerous GRPO algorithm family reinforcement learning algorithms including: GRPO, DAPO, GSPO, SAPO, CISPO, RLOO, Reinforce++, etc. to enhance model intelligence. ms-swift supports a wide range of training tasks, including preference learning algorithms such as DPO, KTO, RM, CPO, SimPO, ORPO, as well as Embedding, Reranker, and sequence classification tasks. ms-swift provides full-pipeline support for large model training, including acceleration for inference, evaluation, and deployment modules using vLLM, SGLang, and LMDeploy, as well as model quantization using GPTQ, AWQ, BNB, and FP8 technologies.

**Why Choose ms-swift?**

- 🍎 **Model Types**: Supports **600+ text-only large models**, **300+ multimodal large models**, and All-to-All full modality models from training to deployment full pipeline, with Day-0 support for popular models.
- **Dataset Types**: Built-in 150+ datasets for pre-training, fine-tuning, human alignment, multimodal and various other tasks, with support for custom datasets. Users only need to prepare datasets for one-click training.
- **Hardware Support**: Supports A10/A100/H100, RTX series, T4/V100, CPU, MPS, and domestic hardware Ascend NPU, etc.
- **Lightweight Training**: Supports lightweight fine-tuning methods such as LoRA, QLoRA, DoRA, LoRA+, LLaMAPro, LongLoRA, LoRA-GA, ReFT, RS-LoRA, Adapter, LISA, etc.
- **Quantized Training**: Supports training on BNB, AWQ, GPTQ, AQLM, HQQ, EETQ quantized models, requiring only 9GB training resources for 7B models.
- **Memory Optimization**: GaLore, Q-Galore, UnSloth, Liger-Kernel, Flash-Attention 2/3, and **Ulysses and Ring-Attention sequence parallelism techniques** support, reducing memory consumption for long-text training.
- **Distributed Training**: Supports distributed data parallelism (DDP), device_map simple model parallelism, DeepSpeed ZeRO2 ZeRO3, FSDP/FSDP2, and Megatron distributed training technologies.
- 🍓 **Multimodal Training**: Supports multimodal packing technology to improve training speed by 100%+, supports mixed modality data training with text, images, video and audio, and supports independent control of vit/aligner/llm.
- **Agent Training**: Supports Agent templates, allowing one dataset to be used for training different models.
- 🍊 **Training Tasks**: Supports pre-training and instruction fine-tuning, as well as training tasks such as DPO, GKD, KTO, RM, CPO, SimPO, ORPO, and supports **Embedding/Reranker** and sequence classification tasks.
- 🥥 **Megatron Parallelism**: Provides TP/PP/SP/CP/ETP/EP/VPP parallel strategies, **MoE model acceleration up to 10x**. Supports full-parameter and LoRA training methods for 250+ text-only large models and 100+ multimodal large models. Supports CPT/SFT/GRPO/DPO/KTO/RM training tasks.
- 🍉 **Reinforcement Learning**: Built-in **rich GRPO family algorithms**, including GRPO, DAPO, GSPO, SAPO, CISPO, CHORD, RLOO, Reinforce++, etc. Supports synchronous and asynchronous vLLM engine inference acceleration, with extensible reward functions, multi-turn inference Schedulers, and environments through plugins.
- **Full-Pipeline Capabilities**: Covers the entire workflow of training, inference, evaluation, quantization, and deployment.
- **UI Training**: Provides Web-UI interface for training, inference, evaluation, and quantization, completing the full pipeline for large models.
- **Inference Acceleration**: Supports PyTorch, vLLM, SGLang, and LmDeploy inference acceleration engines, providing OpenAI interfaces for accelerating inference, deployment, and evaluation modules.
- **Model Evaluation**: Uses EvalScope as the evaluation backend, supporting 100+ evaluation datasets for evaluating text-only and multimodal models.
- **Model Quantization**: Supports quantization export for AWQ, GPTQ, FP8, and BNB. Exported models support inference acceleration using vLLM/SGLang/LmDeploy.


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
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
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

Tips:

- If you want to train with a custom dataset, you can refer to [this guide](../Customization/Custom-dataset.md) to organize your dataset format and specify `--dataset <dataset_path>`.
- The `--model_author` and `--model_name` parameters are only effective when the dataset includes `swift/self-cognition`.
- To train with a different model, simply modify `--model <model_id/model_path>`.
- By default, ModelScope is used for downloading models and datasets. If you want to use HuggingFace, simply specify `--use_hf true`.

After training is complete, use the following command to infer with the trained weights:

- Here, `--adapters` should be replaced with the last checkpoint folder generated during training. Since the adapters folder contains the training parameter file `args.json`, there is no need to specify `--model`, `--system` separately; Swift will automatically read these parameters. To disable this behavior, you can set `--load_args false`.

```shell
# Using an interactive command line for inference.
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048

# merge-lora and use vLLM for inference acceleration
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --stream true \
    --merge_lora true \
    --infer_backend vllm \
    --vllm_max_model_len 8192 \
    --temperature 0 \
    --max_new_tokens 2048
```

Finally, use the following command to push the model to ModelScope:

```shell
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --adapters output/vx-xxx/checkpoint-xxx \
    --push_to_hub true \
    --hub_model_id '<your-model-id>' \
    --hub_token '<your-sdk-token>' \
    --use_hf false
```

## Learn More
- More Shell scripts: [https://github.com/modelscope/ms-swift/tree/main/examples](https://github.com/modelscope/ms-swift/tree/main/examples)
- Using Python: [https://github.com/modelscope/ms-swift/blob/main/examples/notebook/qwen2_5-self-cognition/self-cognition-sft.ipynb](https://github.com/modelscope/ms-swift/blob/main/examples/notebook/qwen2_5-self-cognition/self-cognition-sft.ipynb)
