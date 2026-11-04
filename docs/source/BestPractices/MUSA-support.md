# 摩尔线程 MUSA 支持

ms-swift 通过 [torch_musa](https://github.com/MooreThreads/torch_musa) 与 [torchada](https://github.com/MooreThreads/torchada) 支持摩尔线程（Moore Threads）GPU。torchada 会把 `torch.cuda.*` 接口重定向到 `torch.musa`，swift 在检测到 torch_musa 与 torchada 均已安装时会自动导入它，因此绝大多数训练/推理脚本无需修改即可在 MUSA 上运行。

本文以 **Qwen3.5-4B** 为例，演示在 MTT S5000 上使用 ms-swift 进行 LoRA 与全参数微调、推理和 LoRA 合并。

## 1. 环境配置

### 1.1 基础环境

拉取摩尔线程训练镜像（已包含 torch_musa、DeepSpeed、flash-attention、MT-TransformerEngine 等组件），并参照以下命令启动容器：

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

本文验证所用的软件版本：

| 组件 | 版本 |
|---|---|
| GPU / 驱动 | MTT S5000 (80GB) × 8 / Driver 3.3.8-server |
| MUSA Toolkits | 4.3.8 |
| python | 3.10.12 |
| torch / torch_musa | 2.7.1 / 2.7.1.post1 |
| torchada | 0.1.90 |
| transformers | 5.16.1 |
| peft | 0.18.1 |
| deepspeed | 0.19.3（镜像内置 MUSA 版本） |

### 1.2 安装 ms-swift

Qwen3.5 需要 `transformers>=5.2.0`，而镜像内置的 transformers 版本较低（4.57）。为了不破坏镜像中其他组件（vLLM、sglang 等）依赖的环境，建议创建一个继承系统包的虚拟环境，在其中安装 ms-swift：

```bash
# 复用镜像中的 torch/torch_musa/deepspeed，仅在 venv 中覆盖需要升级的包
python -m venv --without-pip --system-site-packages /home/venv_swift
source /home/venv_swift/bin/activate

git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
python -m pip install -e . -r requirements/musa.txt qwen_vl_utils decord
# 镜像中的 torchao 为 0.9.0（sglang 依赖），peft>=0.19 要求 torchao>=0.16，否则 LoRA 会报错
python -m pip install peft==0.18.1
```

### 1.3 环境检查

- 确认 torch_musa 正确识别 GPU：

```bash
python -c "import torch, torch_musa; print(torch.musa.is_available(), torch.musa.device_count())"
# output: True 8
```

- 确认 swift 能正常导入（会自动启用 torchada）：

```bash
python -c "import swift, torch; print(swift.__version__, torch.cuda.device_count())"
# output: 4.6.0.dev0 8
```

- 查看 GPU 利用率及显存占用：`mthreads-gmi`

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

## 2. 运行示例

说明：
- 使用 `MUSA_VISIBLE_DEVICES` 指定可见的 GPU；多卡训练时设置 `NPROC_PER_NODE`，swift 会自动通过 torchrun 拉起多进程。
- 下文中的 `--model Qwen/Qwen3.5-4B` 会自动从 ModelScope 下载模型，也可以替换为本地模型目录。
- 数据集沿用 ms-swift 示例中的 alpaca 中英文数据与自我认知数据，`--model_author`/`--model_name` 用于自我认知训练。

### 2.1 单卡 LoRA 微调 Qwen3.5-4B

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
单卡显存占用约 15GiB，1 个 epoch（94 step）耗时约 9 分钟。

### 2.2 多卡全参数微调 Qwen3.5-4B（DeepSpeed ZeRO-3）

镜像中内置了适配 MUSA 的 DeepSpeed，可直接使用 `--deepspeed zero3`：

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

4 卡 ZeRO-3 每卡显存占用约 26.5GiB，1 个 epoch（93 step）耗时约 7 分钟。如需使用 8 卡，设置 `NPROC_PER_NODE=8` 与 `MUSA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` 即可。

### 2.3 推理与 LoRA 合并

使用 LoRA 权重进行推理（transformers 后端）：

```bash
MUSA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/Qwen3.5-4B-lora/vx-xxx/checkpoint-xxx \
    --infer_backend transformers \
    --enable_thinking false \
    --stream true \
    --max_new_tokens 2048
```

推理全参数训练后的权重：

```bash
MUSA_VISIBLE_DEVICES=0 \
swift infer \
    --model output/Qwen3.5-4B-full/vx-xxx/checkpoint-xxx \
    --infer_backend transformers \
    --enable_thinking false \
    --stream true \
    --max_new_tokens 2048
```

自我认知训练后的推理效果：

```
[QUERY] 你是谁？
[RESPONSE] 我是一个由swift开发的人工智能助手，名为swift-robot。我被设计用来回答您的问题，提供信息，并进行有趣的对话。如果您有任何问题或需要帮助，请随时向我提问。

[QUERY] who are you?
[RESPONSE] I am an artificial intelligence assistant named swift-robot, trained by swift. My purpose is to provide assistance, answer questions, and engage in conversation with users. ...
```

合并 LoRA 权重（输出到 `checkpoint-xxx-merged`）：

```bash
MUSA_VISIBLE_DEVICES=0 \
swift export \
    --adapters output/Qwen3.5-4B-lora/vx-xxx/checkpoint-xxx \
    --merge_lora true
```
