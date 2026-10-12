# Moore Threads MUSA Support

ms-swift supports Moore Threads GPUs through [torch_musa](https://github.com/MooreThreads/torch_musa) and [torchada](https://github.com/MooreThreads/torchada). torchada redirects the `torch.cuda.*` APIs to `torch.musa`, and swift imports it automatically when both torch_musa and torchada are installed, so most training/inference scripts run on MUSA without modification.

This document uses **Qwen3.5-4B** as an example to demonstrate LoRA and full-parameter fine-tuning, inference, and LoRA merging with ms-swift on MTT S5000.

## 1. Environment Setup

### 1.1 Base Environment

Pull the Moore Threads training image (which ships torch_musa, DeepSpeed, flash-attention, MT-TransformerEngine, etc.) and start a container with the following command:

```bash
IMAGE_NAME=sh-harbor.mthreads.com/mcctest/training-suite:v2.1.7-rc4
docker pull ${IMAGE_NAME}

CONTAINER_NAME=swift_test
docker run -itd --privileged --network=host \
    --env MTHREADS_VISIBLE_DEVICES=all \
    --shm-size=80g \
    -v /data:/data \
    --name ${CONTAINER_NAME} \
    ${IMAGE_NAME} \
    /bin/bash

docker exec -it ${CONTAINER_NAME} /bin/bash
```

Software versions used for verification in this document:

| Component | Version |
|---|---|
| GPU / Driver | MTT S5000 (80GB) × 8 / Driver 3.3.8-server |
| MUSA Toolkits | 4.3.8 |
| python | 3.10.12 |
| torch / torch_musa | 2.7.1 / 2.7.1.post1 |
| torchada | 0.1.90 |
| transformers | 5.16.1 |
| peft | 0.18.1 |
| deepspeed | 0.19.3 (MUSA build shipped in the image) |

### 1.2 Install ms-swift

Qwen3.5 requires `transformers>=5.2.0`, while the image ships an older transformers (4.57). To avoid breaking the environment that other components in the image (vLLM, sglang, etc.) depend on, we recommend creating a virtual environment that inherits the system packages and installing ms-swift there:

```bash
# Reuse torch/torch_musa/deepspeed from the image; only override the packages that need upgrading in the venv
python -m venv --without-pip --system-site-packages /home/venv_swift
source /home/venv_swift/bin/activate

git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
python -m pip install -e . -r requirements/musa.txt qwen_vl_utils decord
# The image ships torchao 0.9.0 (required by sglang); peft>=0.19 requires torchao>=0.16, otherwise LoRA fails
python -m pip install peft==0.18.1
```

### 1.3 Environment Check

- Confirm that torch_musa detects the GPUs:

```bash
python -c "import torch, torch_musa; print(torch.musa.is_available(), torch.musa.device_count())"
# output: True 8
```

- Confirm that swift imports correctly (torchada is enabled automatically):

```bash
python -c "import swift, torch; print(swift.__version__, torch.cuda.device_count())"
# output: 4.6.0.dev0 8
```

- Check GPU utilization and memory usage: `mthreads-gmi`

```
---------------------------------------------------------------------
    mthreads-gmi:2.3.3           Driver Version:3.3.8-server
---------------------------------------------------------------------
ID   Name                 |PCIe                |%GPU  Mem
     Device Type   Perf   |Pcie Lane Width     |Temp  MPC Capable
                                               |      ECC Mode
+-------------------------------------------------------------------+
0    MTT S5000            |00000000:18:00.0    |0%    0MiB(81920MiB)
     Physical      P0     |16x(16x)            |47C   YES
                                               |      On-die, EDC
+-------------------------------------------------------------------+
...
```

## 2. Examples

Notes:
- Use `MUSA_VISIBLE_DEVICES` to select the visible GPUs; for multi-GPU training set `NPROC_PER_NODE`, and swift launches the processes via torchrun automatically.
- `--model Qwen/Qwen3.5-4B` below downloads the model from ModelScope automatically; it can also be replaced by a local model directory.
- The datasets follow the ms-swift examples (alpaca Chinese/English data plus self-cognition data); `--model_author`/`--model_name` are used for self-cognition training.

### 2.1 Single-GPU LoRA Fine-tuning of Qwen3.5-4B

```bash
PYTORCH_MUSA_ALLOC_CONF='expandable_segments:True' \
MUSA_VISIBLE_DEVICES=0 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=12 \
swift sft \
    --model Qwen/Qwen3.5-4B \
    --tuner_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --model_author swift \
    --model_name swift-robot \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 4 \
    --group_by_length true \
    --output_dir output/Qwen3.5-4B-lora \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --warmup_ratio 0.05 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4
```

The single GPU uses about 15GiB of memory, and 1 epoch (94 steps) takes about 9 minutes.

### 2.2 Multi-GPU Full-parameter Fine-tuning of Qwen3.5-4B (DeepSpeed ZeRO-3)

The image ships a MUSA build of DeepSpeed, so `--deepspeed zero3` can be used directly:

```bash
PYTORCH_MUSA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=4 \
MUSA_VISIBLE_DEVICES=0,1,2,3 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=12 \
swift sft \
    --model Qwen/Qwen3.5-4B \
    --tuner_type full \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --model_author swift \
    --model_name swift-robot \
    --load_from_cache_file true \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --freeze_vit true \
    --freeze_aligner true \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing true \
    --group_by_length true \
    --output_dir output/Qwen3.5-4B-full \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --save_only_model true \
    --logging_steps 5 \
    --max_length 2048 \
    --warmup_ratio 0.05 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --deepspeed zero3
```

With 4-GPU ZeRO-3, each GPU uses about 26.5GiB of memory, and 1 epoch (93 steps) takes about 7 minutes. To use 8 GPUs, set `NPROC_PER_NODE=8` and `MUSA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`.

### 2.3 Inference and LoRA Merging

Inference with the LoRA weights (transformers backend):

```bash
MUSA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/Qwen3.5-4B-lora/vx-xxx/checkpoint-xxx \
    --infer_backend transformers \
    --enable_thinking false \
    --stream true \
    --max_new_tokens 2048
```

Inference with the full-parameter checkpoint:

```bash
MUSA_VISIBLE_DEVICES=0 \
swift infer \
    --model output/Qwen3.5-4B-full/vx-xxx/checkpoint-xxx \
    --infer_backend transformers \
    --enable_thinking false \
    --stream true \
    --max_new_tokens 2048
```

Inference results after self-cognition training:

```
[QUERY] 你是谁？
[RESPONSE] 我是一个由swift开发的人工智能助手，名为swift-robot。我被设计用来回答您的问题，提供信息，并进行有趣的对话。如果您有任何问题或需要帮助，请随时向我提问。

[QUERY] who are you?
[RESPONSE] I am an artificial intelligence assistant named swift-robot, trained by swift. My purpose is to provide assistance, answer questions, and engage in conversation with users. ...
```

Merge the LoRA weights (written to `checkpoint-xxx-merged`):

```bash
MUSA_VISIBLE_DEVICES=0 \
swift export \
    --adapters output/Qwen3.5-4B-lora/vx-xxx/checkpoint-xxx \
    --merge_lora true
```
