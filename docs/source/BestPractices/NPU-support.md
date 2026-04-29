# NPU支持

我们在 ms-swift 上增加了对昇腾 NPU 的支持，用户可以在昇腾 NPU 上进行模型的微调和推理。

本文档介绍如何在昇腾 NPU 上完成环境准备、模型训练、保存合并、推理、部署和常见问题排查。

如果你是第一次在 NPU 上使用 ms-swift，推荐按以下顺序阅读：

1. 先查看“支持范围速览”，确认模型、算法和后端是否已验证。
2. 根据“选择你的使用路线”决定只装基础环境，还是额外安装 MindSpeed/Megatron-SWIFT。
3. 根据自己的环境管理习惯选择“本地环境安装”或“镜像/容器环境安装”，然后执行“NPU 可用性检查”。
4. 使用“快速跑通”完成一次 ModelScope 模型 LoRA 训练、合并、推理和部署。
5. 需要更大规模训练时，再阅读 DDP、DeepSpeed 和 MindSpeed/Megatron-SWIFT 相关章节。

## 支持范围速览

推荐基础环境版本：

| software  | version         |
| --------- | --------------- |
| Python    | >= 3.10, < 3.12 |
| CANN      | == 8.5.1        |
| torch     | == 2.7.1        |
| torch_npu | == 2.7.1.post2  |

基础环境准备请参照 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)。本文示例实验环境为 8 * 昇腾910B3 64G。

| 一级特性 | 特性                | 进展     |
| -------- | ------------------- | -------- |
| 训练范式 | CPT                 | 已支持   |
|          | SFT                 | 已支持   |
|          | DPO                 | 已支持   |
|          | RM                  | 已支持   |
| 分布式   | DDP                 | 已支持   |
|          | FSDP                | 已支持   |
|          | FSDP2               | 已支持   |
|          | DeepSpeed           | 已支持   |
|          | MindSpeed(Megatron) | 已支持   |
| 低参微调 | FULL                | 已支持   |
|          | LoRA                | 已支持   |
|          | QLoRA               | 暂不支持 |
| RLHF     | GRPO                | 已支持   |
|          | PPO                 | 已支持   |
| 性能优化 | FA 等融合算子       | 已支持   |
|          | Liger-Kernel        | 暂不支持 |
| 部署     | PT                  | 已支持   |
|          | vLLM                | 已支持   |
|          | SGLang              | 暂不支持 |

### 已验证 SFT 组合

| algorithm | model families              | strategy              | hardware          |
| --------- | --------------------------- | --------------------- | ----------------- |
| SFT       | Qwen2.5-0.5B-Instruct       | FSDP1/FSDP2/deepspeed | Atlas 900 A2 PODc |
| SFT       | Qwen2.5-1.5B-Instruct       | FSDP1/FSDP2/deepspeed | Atlas 900 A2 PODc |
| SFT       | Qwen2.5-7B-Instruct         | FSDP1/FSDP2/deepspeed | Atlas 900 A2 PODc |
| SFT       | Qwen2.5-VL-3B-Instruct      | FSDP1/FSDP2/deepspeed | Atlas 900 A2 PODc |
| SFT       | Qwen2.5-VL-7B-Instruct      | FSDP1/FSDP2/deepspeed | Atlas 900 A2 PODc |
| SFT       | Qwen2.5-Omni-3B             | FSDP1/FSDP2/deepspeed | Atlas 900 A2 PODc |
| SFT       | Qwen3-8B                    | FSDP1/FSDP2/deepspeed | Atlas 900 A2 PODc |
| SFT       | Qwen3-30B-A3B               | FSDP1/FSDP2/deepspeed | Atlas 900 A2 PODc |
| SFT       | Qwen3-32B                   | FSDP1/FSDP2/deepspeed | Atlas 900 A2 PODc |
| SFT       | Qwen3-VL-30B-A3B-Instruct   | FSDP1/FSDP2/deepspeed | Atlas 900 A2 PODc |
| SFT       | Qwen3-Omni-30B-A3B-Instruct | FSDP1/FSDP2/deepspeed | Atlas 900 A2 PODc |
| SFT       | InternVL3-8B                | FSDP1/FSDP2/deepspeed | Atlas 900 A2 PODc |
| SFT       | Ovis2.5-2B                  | FSDP1/FSDP2/deepspeed | Atlas 900 A2 PODc |
| SFT       | Qwen3.5-27B                 | FSDP1/FSDP2/deepspeed | Atlas 900 A2 PODc |
| SFT       | Qwen3.5-35B-A3B             | FSDP1/FSDP2/deepspeed | Atlas 900 A2 PODc |

### 已验证 RL 组合

| algorithm | model families      | strategy  | rollout engine | hardware          |
| --------- | ------------------- | --------- | -------------- | ----------------- |
| **GRPO**  | Qwen2.5-7B-Instruct | deepspeed | vllm-ascend    | Atlas 900 A2 PODc |
| **GRPO**  | Qwen3-8B            | deepspeed | vllm-ascend    | Atlas 900 A2 PODc |
| **DPO**   | Qwen2.5-7B-Instruct | deepspeed | vllm-ascend    | Atlas 900 A2 PODc |
| **DPO**   | Qwen3-8B            | deepspeed | vllm-ascend    | Atlas 900 A2 PODc |
| **PPO**   | Qwen2.5-7B-Instruct | deepspeed | vllm-ascend    | Atlas 900 A2 PODc |
| **PPO**   | Qwen3-8B            | deepspeed | vllm-ascend    | Atlas 900 A2 PODc |

### 暂不支持或未完全验证

| item                              |
| --------------------------------- |
| Liger-kernel                      |
| 量化/QLoRA相关                    |
| 使用sglang作为推理引擎            |
| 使用megatron时开启ETP进行lora训练 |

## 选择你的使用路线

| 场景                         | 推荐路线                                      | 是否需要 MindSpeed |
| ---------------------------- | --------------------------------------------- | ------------------ |
| 只做普通 SFT/LoRA/推理       | 本地环境安装或镜像/容器环境安装              | 不需要             |
| 需要 Megatron-SWIFT 大模型训练 | 先装基础环境，再装 MindSpeed/Megatron/mcore-bridge | 需要               |
| 需要 GRPO/PPO/DPO 等 RLHF    | 基础训练环境 + vLLM-Ascend rollout/deploy     | 通常不需要         |
| 只是验证 NPU 是否可用        | 跑 NPU 可用性检查脚本                         | 不需要             |

## 环境准备

### 镜像/容器环境安装
官方 NPU 镜像仍在发布流程中。在镜像正式发布前，推荐使用项目提供的 Dockerfile 自行构建一个包含 CANN、PyTorch、torch_npu 与 ms-swift 依赖的容器环境。容器方式的优势是依赖版本更容易固化，也便于在多台昇腾机器之间复现实验环境。

先 clone modelscope 仓库，然后使用仓库中的 [Dockerfile.ascend](https://github.com/modelscope/modelscope/blob/master/docker/Dockerfile.ascend) 和 [build_image.py](https://github.com/modelscope/modelscope/blob/master/docker/build_image.py) 构建镜像：

```shell
git clone https://github.com/modelscope/modelscope.git
cd modelscope
DOCKER_REGISTRY=ms-swift python docker/build_image.py \
  --image_type ascend \
  --python_version 3.11.11 \
  --soc_version ascend910b1 \
  --arch arm
```

当前 `build_image.py` 生成的 Ascend 镜像名格式为 `{DOCKER_REGISTRY}:{swift_branch}-{atlas_hardware}-{python_tag}-{arch}`。以上命令以 ARM 架构的 Atlas 900 A2 PODc 为例，通常会生成 `ms-swift:main-A2-py311-arm`。下面用变量保存镜像名和工作目录，实际使用时请按构建日志中的镜像名替换：

```shell
export IMAGE_NAME=ms-swift:main-A2-py311-arm
export WORKSPACE=/path/to/workspace
```

启动容器前建议先确认宿主机暴露的 NPU 设备：

```shell
ls /dev/davinci*
```

启动容器时需要把 NPU 设备、驱动、固件、`npu-smi` 和必要日志目录挂载进去。下面示例按常见 8 卡设备 `davinci0` 到 `davinci7` 编写；部分机器会额外暴露到 `davinci15`，这时请按 `ls /dev/davinci*` 的结果把对应设备都加到 `docker run` 中：

```shell
docker run -it \
  --name swift-ascend \
  --network=host --ipc=host --shm-size=128g \
  --device=/dev/davinci0 --device=/dev/davinci1 \
  --device=/dev/davinci2 --device=/dev/davinci3 \
  --device=/dev/davinci4 --device=/dev/davinci5 \
  --device=/dev/davinci6 --device=/dev/davinci7 \
  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /var/log/npu:/var/log/npu \
  -v ${WORKSPACE}:/workspace \
  ${IMAGE_NAME} \
  /bin/bash
```

进入容器后，建议先执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`，再运行后文的 NPU 可用性检查脚本，确认容器内可以正确访问昇腾设备。如果容器内无法识别 NPU，请优先检查 `/dev/davinci*`、`/dev/davinci_manager`、驱动目录和 `npu-smi` 是否正确挂载。

### 本地环境安装
```shell
# 创建新的 conda 虚拟环境（可选）
conda create -n swift-npu python=3.11 -y
conda activate swift-npu

# 注意进行后续操作前要先 source 激活 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置 pip 全局镜像（可选，加速下载）
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip install ms-swift -U

# 使用源码安装
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e .

# 安装 torch-npu
pip install torch_npu decorator
# 如果你想要使用 deepspeed（控制显存占用，训练速度会有一定下降）
pip install deepspeed

# 如果需要使用 evaluation 功能，请安装以下包
pip install evalscope[opencompass]

# 如果需要使用 vllm-ascend 进行推理，请安装以下包（更多版本请参考 [vLLM-Ascend 官网](https://docs.vllm.ai/projects/ascend/en/latest/installation.html)）
pip install vllm==0.14.0
pip install vllm-ascend==0.14.0rc1
```

### NPU 可用性检查

测试环境是否安装正确，NPU能否被正常加载：

```python
from transformers.utils import is_torch_npu_available
import torch

print(is_torch_npu_available())  # True
print(torch.npu.device_count())  # 8
print(torch.randn(10, device='npu:0'))
```

### MindSpeed/Megatron-SWIFT 可选安装

如果需要使用 MindSpeed(Megatron-LM)，请按照下面引导安装必要依赖。

```shell
# 1. 获取并切换 Megatron-LM 至 v0.15.3 版本
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout v0.15.3
cd ..

# 2. 获取并安装 MindSpeed
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
git checkout core_r0.15.3
pip install -e .
cd ..

# 3. 获取并安装 mcore-bridge
git clone https://github.com/modelscope/mcore-bridge.git
cd mcore-bridge
pip install -e .
cd ..

# 4. 设置环境变量
export PYTHONPATH=$PYTHONPATH:<your_local_megatron_lm_path>
export MEGATRON_LM_PATH=<your_local_megatron_lm_path>
```

执行如下命令验证 MindSpeed(Megatron-LM) 是否配置成功：
```shell
python -c "import mindspeed.megatron_adaptor; from swift.megatron.init import init_megatron_env; init_megatron_env(); print('✓ NPU环境下的Megatron-SWIFT配置验证成功！')"
```

### 环境查看

查看NPU的P2P连接，这里看到每个NPU都通过7条HCCS与其他NPU互联

```shell
(valle) root@valle:~/src# npu-smi info -t topo
	   NPU0       NPU1       NPU2       NPU3       NPU4       NPU5       NPU6       NPU7       CPU Affinity
NPU0       X          HCCS       HCCS       HCCS       HCCS       HCCS       HCCS       HCCS       144-167
NPU1       HCCS       X          HCCS       HCCS       HCCS       HCCS       HCCS       HCCS       144-167
NPU2       HCCS       HCCS       X          HCCS       HCCS       HCCS       HCCS       HCCS       96-119
NPU3       HCCS       HCCS       HCCS       X          HCCS       HCCS       HCCS       HCCS       96-119
NPU4       HCCS       HCCS       HCCS       HCCS       X          HCCS       HCCS       HCCS       0-23
NPU5       HCCS       HCCS       HCCS       HCCS       HCCS       X          HCCS       HCCS       0-23
NPU6       HCCS       HCCS       HCCS       HCCS       HCCS       HCCS       X          HCCS       48-71
NPU7       HCCS       HCCS       HCCS       HCCS       HCCS       HCCS       HCCS       X          48-71

Legend:

  X    = Self
  SYS  = Path traversing PCIe and NUMA nodes. Nodes are connected through SMP, such as QPI, UPI.
  PHB  = Path traversing PCIe and the PCIe host bridge of a CPU.
  PIX  = Path traversing a single PCIe switch
  PXB  = Path traversing multiple PCIe switches
  HCCS = Connection traversing HCCS.
  NA   = Unknown relationship.
```

查看NPU状态, npu-smi命令详解可以查看[官方文档](https://support.huawei.com/enterprise/zh/doc/EDOC1100079287/10dcd668)

```shell
(valle) root@valle:~/src# npu-smi info
+------------------------------------------------------------------------------------------------+
| npu-smi 24.1.rc1.b030            Version: 24.1.rc1.b030                                        |
+---------------------------+---------------+----------------------------------------------------+
| NPU   Name                | Health        | Power(W)    Temp(C)           Hugepages-Usage(page)|
| Chip                      | Bus-Id        | AICore(%)   Memory-Usage(MB)  HBM-Usage(MB)        |
+===========================+===============+====================================================+
| 0     910B3               | OK            | 101.8       43                0    / 0             |
| 0                         | 0000:C1:00.0  | 0           0    / 0          3318 / 65536         |
+===========================+===============+====================================================+
| 1     910B3               | OK            | 92.0        39                0    / 0             |
| 0                         | 0000:C2:00.0  | 0           0    / 0          3314 / 65536         |
+===========================+===============+====================================================+
| 2     910B3               | OK            | 102.0       40                0    / 0             |
| 0                         | 0000:81:00.0  | 0           0    / 0          3314 / 65536         |
+===========================+===============+====================================================+
| 3     910B3               | OK            | 99.8        40                0    / 0             |
| 0                         | 0000:82:00.0  | 0           0    / 0          3314 / 65536         |
+===========================+===============+====================================================+
| 4     910B3               | OK            | 98.6        45                0    / 0             |
| 0                         | 0000:01:00.0  | 0           0    / 0          3314 / 65536         |
+===========================+===============+====================================================+
| 5     910B3               | OK            | 99.7        44                0    / 0             |
| 0                         | 0000:02:00.0  | 0           0    / 0          3314 / 65536         |
+===========================+===============+====================================================+
| 6     910B3               | OK            | 103.8       45                0    / 0             |
| 0                         | 0000:41:00.0  | 0           0    / 0          3314 / 65536         |
+===========================+===============+====================================================+
| 7     910B3               | OK            | 98.2        44                0    / 0             |
| 0                         | 0000:42:00.0  | 0           0    / 0          3315 / 65536         |
+===========================+===============+====================================================+
```

## 快速跑通：ModelScope 模型 + 数据集

如果你想先用 ModelScope 上的模型和数据集快速验证环境，可以直接执行本节完成一次完整闭环：训练 LoRA、找到最新 checkpoint、Merge LoRA、命令行推理、启动服务、curl 验证。示例使用小模型和小规模采样，便于快速跑通；换成自己的模型或数据集时，只需要修改前面的 ID 变量。

```shell
export MODEL_ID=Qwen/Qwen3-0.6B
export DATASET_ID=AI-ModelScope/alpaca-gpt4-data-zh
export WORK_DIR=output/npu-modelscope-qwen3-0_6b-lora
```

训练并保存 LoRA checkpoint：

```shell
ASCEND_RT_VISIBLE_DEVICES=0 \
swift sft \
    --model $MODEL_ID \
    --dataset $DATASET_ID#1000 \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --tuner_type lora \
    --target_modules all-linear \
    --lora_rank 8 \
    --lora_alpha 32 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --max_length 2048 \
    --save_steps 100 \
    --eval_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --output_dir $WORK_DIR
```

训练结束后，checkpoint 会保存在 `$WORK_DIR/*/checkpoint-*` 目录下。可以用下面的命令取最新 checkpoint，并将 LoRA 合并保存为完整模型权重：

```shell
export CKPT_DIR=$(ls -dt $WORK_DIR/*/checkpoint-* | head -n 1)

ASCEND_RT_VISIBLE_DEVICES=0 \
swift export \
    --adapters $CKPT_DIR \
    --merge_lora true

export MERGED_DIR=${CKPT_DIR}-merged
```

推理验证可以直接加载 LoRA checkpoint，也可以加载合并后的完整权重：

```shell
# 直接加载 LoRA checkpoint
ASCEND_RT_VISIBLE_DEVICES=0 \
swift infer \
    --adapters $CKPT_DIR \
    --stream true \
    --temperature 0 \
    --max_new_tokens 512

# 加载 Merge 后的完整权重
ASCEND_RT_VISIBLE_DEVICES=0 \
swift infer \
    --model $MERGED_DIR \
    --stream true \
    --temperature 0 \
    --max_new_tokens 512
```

如果需要启动 OpenAI 兼容的部署服务，可以使用合并后的完整权重：

```shell
ASCEND_RT_VISIBLE_DEVICES=0 \
swift deploy \
    --model $MERGED_DIR \
    --host 0.0.0.0 \
    --port 8000 \
    --max_new_tokens 512 \
    --served_model_name npu-modelscope-qwen3-0_6b
```

服务启动后，用 curl 验证接口：

```shell
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "npu-modelscope-qwen3-0_6b",
"messages": [{"role": "user", "content": "用一句话介绍昇腾NPU。"}],
"max_tokens": 128,
"temperature": 0
}'
```

## 训练

以下介绍LoRA的微调, 全参数微调设置参数`--tuner_type full`即可. **更多训练脚本**参考[这里](https://github.com/modelscope/ms-swift/tree/main/examples/ascend/train)。如果需要了解预训练、SFT、LoRA、全参数训练、自定义数据集等通用能力，可以继续阅读[预训练与微调文档](../Instruction/Pre-training-and-Fine-tuning.md)。


| 模型大小 | NPU数量 | deepspeed类型 | 最大显存占用量 |
| -------- | ------- | ------------- | -------------- |
| 7B       | 1       | None          | 1 * 28 GB      |
| 7B       | 4       | None          | 4 * 22 GB      |
| 7B       | 4       | zero2         | 4 * 28 GB      |
| 7B       | 4       | zero3         | 4 * 22 GB      |
| 7B       | 8       | None          | 8 * 22 GB      |
| 14B      | 1       | None          | 1 * 45 GB      |
| 14B      | 8       | None          | 8 * 51 GB      |
| 14B      | 8       | zero2         | 8 * 49 GB      |
| 14B      | 8       | zero3         | 8 * 31 GB      |

### 单卡训练

通过如下命令启动单卡微调:

```shell
# 实验环境: 昇腾910B3
# 显存需求: 28 GB
# 运行时长: 8小时
ASCEND_RT_VISIBLE_DEVICES=0 \
swift sft \
    --model Qwen/Qwen2-7B-Instruct \
    --dataset AI-ModelScope/blossom-math-v2 \
    --split_dataset_ratio 0.01 \
    --num_train_epochs 5 \
    --tuner_type lora \
    --output_dir output \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 16 \
    --save_steps 100 \
    --eval_steps 100

```


### 数据并行训练

我们使用其中的4卡进行ddp训练

```shell
# 实验环境: 4 * 昇腾910B3
# 显存需求: 4 * 22 GB
# 运行时长: 2小时
NPROC_PER_NODE=4 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model Qwen/Qwen2-7B-Instruct \
    --dataset AI-ModelScope/blossom-math-v2 \
    --split_dataset_ratio 0.01 \
    --num_train_epochs 5 \
    --tuner_type lora \
    --output_dir output \
    ...
```


### Deepspeed训练

ZeRO2:

```shell
# 实验环境: 4 * 昇腾910B3
# 显存需求: 4 * 28GB
# 运行时长: 3.5小时
NPROC_PER_NODE=4 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model Qwen/Qwen2-7B-Instruct \
    --dataset AI-ModelScope/blossom-math-v2 \
    --split_dataset_ratio 0.01 \
    --num_train_epochs 5 \
    --tuner_type lora \
    --output_dir output \
    --deepspeed zero2 \
    ...
```

ZeRO3:

```shell
# 实验环境: 4 * 昇腾910B3
# 显存需求: 4 * 22 GB
# 运行时长: 8.5小时
NPROC_PER_NODE=4 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model Qwen/Qwen2-7B-Instruct \
    --dataset AI-ModelScope/blossom-math-v2 \
    --split_dataset_ratio 0.01 \
    --num_train_epochs 5 \
    --tuner_type lora \
    --output_dir output \
    --deepspeed zero3 \
    ...
```

### Qwen3.5 单机多卡 LoRA 示例

下面给出一个更新模型的 NPU LoRA 示例。这里使用 Qwen3.5-4B 做演示，4 卡数据并行通常比单卡更快；如果本地已经下载好模型和数据集，可以把 `--model`、`--dataset` 替换成本地路径。

```shell
# 实验环境: 4 * 昇腾910B3
NPROC_PER_NODE=4 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model Qwen/Qwen3.5-4B \
    --dataset AI-ModelScope/alpaca-gpt4-data-zh#2000 \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --tuner_type lora \
    --target_modules all-linear \
    --lora_rank 8 \
    --lora_alpha 32 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --max_length 2048 \
    --group_by_length true \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --save_steps 100 \
    --eval_steps 100 \
    --save_total_limit 2 \
    --output_dir output/Qwen3.5-4B-NPU
```

调参时可以先抓住三个目标：显存、吞吐和稳定性。

- 降低显存：优先减小 `--max_length`、`--per_device_train_batch_size` 和 `--lora_rank`；仍然 OOM 时再启用 `--deepspeed zero2/zero3`。ZeRO 可以明显降低显存压力，但会增加通信和调度开销。
- 提高吞吐：在显存允许的情况下增大 `--per_device_train_batch_size`，再用 `--gradient_accumulation_steps` 保持全局 batch size；数据预处理较慢时增加 `--dataset_num_proc`，数据读取跟不上时增加 `--dataloader_num_workers`。
- 控制保存成本：`--save_steps` 不宜过小，否则频繁保存会拖慢训练；`--save_total_limit 2` 通常足够保留 best checkpoint 和 last checkpoint。
- 提高稳定性：NPU 上建议优先使用 `bfloat16`；如果遇到 loss 异常或 NaN，可以先缩小学习率、降低 batch，必要时再临时切到 `float32` 做对照定位。

更多参数含义可以在[命令行参数文档](../Instruction/Command-line-parameters.md)中查询。

### NPU模型Patch开关

ms-swift 在 NPU 环境下默认会启用模型层 patch，以适配部分 Transformers 模型在昇腾 NPU 上的算子和兼容性需求。通常不需要关闭；如果怀疑某个模型的 loss 异常、forward 报错与 NPU 模型 patch 有关，需要临时切回 Transformers 原生实现做对比，可以设置：

```shell
swift sft ... --enable_npu_model_patch false
```

## 模型保存、Merge LoRA 和断点续训

训练时通过 `--output_dir` 指定输出目录，通过 `--save_steps` 控制 checkpoint 保存间隔，通过 `--save_total_limit` 控制最多保留多少个 checkpoint。LoRA 训练结束后，checkpoint 目录中会保存 adapter 权重、训练参数和 trainer 状态；常见目录形态如下：

```text
output/Qwen3.5-4B-NPU/vx-xxx/
├── checkpoint-100/
├── checkpoint-200/
└── ...
```

如果只做推理或继续 LoRA 训练，可以直接使用 checkpoint 目录。若希望得到一个独立的完整模型目录，便于 vLLM-Ascend 部署、离线分发或后续量化，可以执行 Merge LoRA：

```shell
ASCEND_RT_VISIBLE_DEVICES=0 \
swift export \
    --adapters output/Qwen3.5-4B-NPU/vx-xxx/checkpoint-xxx \
    --merge_lora true
```

合并后的模型默认保存在 `checkpoint-xxx-merged` 目录。之后可以像加载普通模型一样使用 `--model checkpoint-xxx-merged`。

如果训练中断，需要从 checkpoint 恢复训练，请保持原训练参数不变，只额外增加 `--resume_from_checkpoint`：

```shell
NPROC_PER_NODE=4 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model Qwen/Qwen3.5-4B \
    --dataset AI-ModelScope/alpaca-gpt4-data-zh#2000 \
    --tuner_type lora \
    --output_dir output/Qwen3.5-4B-NPU \
    --resume_from_checkpoint output/Qwen3.5-4B-NPU/vx-xxx/checkpoint-xxx \
    ...
```

`--resume_from_checkpoint` 会恢复模型权重、优化器状态、随机种子和训练进度。如果只想加载模型权重而不恢复优化器和数据跳过状态，可以额外设置 `--resume_only_model true`。相关参数可参考[命令行参数文档](../Instruction/Command-line-parameters.md)中的 `resume_from_checkpoint`、`resume_only_model`、`save_steps` 和 `save_total_limit`。

## 推理

原始模型:

```shell
ASCEND_RT_VISIBLE_DEVICES=0 swift infer \
    --model Qwen/Qwen2-7B-Instruct \
    --stream true --max_new_tokens 2048
```

LoRA微调后:

```shell
ASCEND_RT_VISIBLE_DEVICES=0 swift infer \
    --adapters xxx/checkpoint-xxx --load_data_args true \
    --stream true --max_new_tokens 2048
```

全参数训练或 Merge LoRA 后的模型，可以通过 `--model` 指向对应的完整权重目录：

```shell
ASCEND_RT_VISIBLE_DEVICES=0 swift infer \
    --model xxx/checkpoint-xxx-merged \
    --stream true --max_new_tokens 2048
```


## 部署

### 使用原生transformers进行部署

原始模型:

```shell
ASCEND_RT_VISIBLE_DEVICES=0 swift deploy --model Qwen/Qwen2-7B-Instruct --max_new_tokens 2048
```

LoRA微调后:

```shell
ASCEND_RT_VISIBLE_DEVICES=0 swift deploy --adapters xxx/checkpoint-xxx --max_new_tokens 2048

# Merge LoRA 后部署完整权重
ASCEND_RT_VISIBLE_DEVICES=0 swift export --adapters xx/checkpoint-xxx --merge_lora true
ASCEND_RT_VISIBLE_DEVICES=0 swift deploy --model xxx/checkpoint-xxx-merged --max_new_tokens 2048
```

### 使用vLLM-ascend进行部署
使用pypi进行安装：
```shell
# 请以 vLLM-Ascend 官方兼容矩阵为准；以下为本文验证版本。
pip install vllm==0.14.0
pip install vllm-ascend==0.14.0rc1
```
原始模型：
```shell
ASCEND_RT_VISIBLE_DEVICES=0 swift deploy \
    --model Qwen/Qwen2.5-7B-Instruct \
    --infer_backend vllm \
    --max_new_tokens 2048
```

LoRA微调后:

```shell
ASCEND_RT_VISIBLE_DEVICES=0 swift deploy \
    --adapters xxx/checkpoint-xxx \
    --infer_backend vllm \
    --max_new_tokens 2048

# Merge LoRA 后部署完整权重
ASCEND_RT_VISIBLE_DEVICES=0 swift export \
    --adapters xx/checkpoint-xxx \
    --merge_lora true

ASCEND_RT_VISIBLE_DEVICES=0 swift deploy \
    --model xxx/checkpoint-xxx-merged \
    --infer_backend vllm \
    --max_new_tokens 2048
```

## 评测
完成训练、推理或部署后，可以使用SWIFT内置的EvalScope能力对原始模型或微调后的checkpoint进行评测，完整参数说明与示例请参考[评测文档](../Instruction/Evaluation.md)。

## 发布
如果需要将NPU训练后的checkpoint、合并后的模型或量化后的模型发布到ModelScope/HuggingFace，可以使用`swift export`的推送能力，完整参数说明与示例请参考[导出与推送文档](../Instruction/Export-and-push.md#推送模型)。

## FAQ

更多通用问题请先查看[常见问题整理](../Instruction/Frequently-asked-questions.md)。下面记录 NPU 场景下更常遇到的问题和处理方式。

### Q1: 如何确认当前环境已经正确识别 NPU？

先确认已经 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`，再执行本文安装章节中的环境检查脚本。正常情况下，`is_torch_npu_available()` 应返回 `True`，`torch.npu.device_count()` 应能看到可用 NPU 数量，且可以在 `npu:0` 上创建 tensor。如果这里失败，优先检查 CANN、`torch`、`torch_npu` 版本是否和本文推荐版本一致。

### Q2: 训练时应该选择 FSDP、DeepSpeed 还是 Megatron-SWIFT？

普通 SFT 优先参考本文兼容性表中的 `FSDP1/FSDP2/deepspeed` 组合；如果模型规模较大、需要更高并行能力，再使用 Megatron-SWIFT，并按安装章节额外安装 MindSpeed、Megatron-LM 和 mcore-bridge。DeepSpeed 可以降低显存压力，但速度可能下降，遇到性能问题时可以对比 FSDP 方案。

### Q3: NPU 模型 Patch 需要手动关闭吗？

通常不需要。ms-swift 会在 NPU 环境下默认启用模型层 patch，以适配部分 Transformers 模型在昇腾 NPU 上的算子和兼容性需求。只有在排查 loss 异常、forward 报错，且怀疑问题来自 NPU patch 时，才建议临时加上 `--enable_npu_model_patch false` 和原生 Transformers 行为做对比。

### Q4: 使用 vLLM-Ascend 部署或 RL rollout 时需要注意什么？

请安装本文推荐的 `vllm` 与 `vllm-ascend` 版本，并优先使用兼容性表中已经验证过的模型和算法组合。当前 `sglang` 推理引擎未在 NPU 场景下完成支持验证，如果需要 NPU 上的高性能推理或 RL rollout，建议优先使用 `vllm-ascend`。

### Q5: 忘记执行 `source set_env.sh` 会有什么表现？

常见表现是 `is_torch_npu_available()` 返回 `False`、`torch.npu.device_count()` 为 0，或者运行时找不到 CANN/HCCL 相关动态库。进入新 shell 或新容器后，先执行：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

如果系统安装了 NNAL/ATB 等组件，也需要按实际环境 source 对应的 `set_env.sh`。

### Q6: `torch` 和 `torch_npu` 版本不匹配怎么判断？

优先对照本文推荐版本安装。版本不匹配时，常见现象包括 `import torch_npu` 失败、NPU 设备不可见、算子注册失败、运行时报 C++/符号找不到等。可以先用下面的命令确认版本：

```shell
python -c "import torch, torch_npu; print(torch.__version__); print(torch_npu.__version__)"
```

如果版本不一致，先卸载后按同一套 CANN/PyTorch/torch_npu 版本重新安装，不建议只升级其中一个包。

### Q7: `ASCEND_RT_VISIBLE_DEVICES` 和 `NPROC_PER_NODE` 不一致会怎样？

分布式训练时二者应该匹配。例如 `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3` 通常对应 `NPROC_PER_NODE=4`。如果进程数大于可见设备数，可能出现 rank 绑卡失败、多个进程抢同一张卡、初始化卡住或 HCCL 报错；如果进程数小于可见设备数，则只有部分 NPU 会被使用。

### Q8: 多卡训练卡住时先看什么？

先确认每个 rank 是否都已经启动、`ASCEND_RT_VISIBLE_DEVICES` 和 `NPROC_PER_NODE` 是否匹配，再看日志停在数据预处理、模型构建、权重加载还是 HCCL 初始化阶段。NPU/HCCL 相关底层日志可以重点查看：

```shell
ls ~/ascend/log/debug/plog
```

如果 Python 进程没有退出但长时间无输出，可以用 `pystack` 查看各 rank 当前栈，先判断是卡在数据、通信还是模型 forward/backward。

### Q9: HCCL 连接或超时问题如何初步排查？

先用 `npu-smi info` 和 `npu-smi info -t topo` 确认设备健康和拓扑，再检查是否有其他任务占用同一组 NPU。单机训练优先确认卡号、进程数和可见设备一致；多机训练还需要确认网络、rank 配置、通信端口和各节点环境变量一致。若同一机器上残留旧训练进程，先清理对应用户的训练进程后再重试。

### Q10: 容器里 `npu-smi` 不可用通常是什么原因？

通常是设备或驱动文件没有挂载完整。优先检查 `docker run` 是否包含 `/dev/davinci*`、`/dev/davinci_manager`、`/dev/devmm_svm`、`/dev/hisi_hdc`，以及 `/usr/local/Ascend/driver`、`/usr/local/Ascend/firmware`、`/usr/local/sbin/npu-smi` 和 `/etc/ascend_install.info`。如果宿主机本身 `npu-smi info` 失败，先修宿主机驱动环境。

### Q11: 原生 transformers 部署和 vLLM-Ascend 部署怎么选？

原生 transformers 部署兼容性更好，适合先验证模型、adapter、模板和输出是否正确；vLLM-Ascend 更适合高吞吐服务、RL rollout 或需要 OpenAI 兼容接口的性能场景。遇到 vLLM-Ascend 版本或算子问题时，建议先用 transformers 后端确认模型本身可用，再切换到 vLLM-Ascend 排查性能后端问题。

### Q12: vLLM-Ascend 报 device type 不匹配或 undefined symbol 怎么办？

这类问题通常不是训练脚本参数导致的，而是 `vllm-ascend` 轮子与当前硬件、PyTorch 或 C++ ABI 不匹配。可以先检查包内构建信息和当前版本：

```shell
python -c "import torch, vllm_ascend; print(torch.__version__); print(vllm_ascend.__file__)"
```

如果报错信息包含 `Current device type ... does not match the installed version's device type ...`、`undefined symbol` 等，建议按设备类型（A2/A3/其他）和官方兼容矩阵重装 `torch`、`torch_npu`、`vllm`、`vllm-ascend`，不要只单独替换一个包。

### Q13: FP8 或量化模型可以直接在 NPU 上训练吗？

不要默认可以。下载或加载大模型前，先检查 `config.json` 是否包含 `quantization_config`，再检查 safetensors 的真实 dtype。当前 NPU 支持范围中量化/QLoRA 仍属于暂不支持或未完全验证能力；如果模型权重是 FP8 block quantized，而当前 NPU 软件栈不支持对应 FP8 路径，应先换用 BF16 权重，或离线转换为 BF16 后再训练/加载。

### Q14: Megatron-SWIFT 导入到错误的 Megatron/MindSpeed 怎么排查？

跑 Megatron-SWIFT 前，`PYTHONPATH` 和 `MEGATRON_LM_PATH` 必须指向同一份 Megatron-LM 源码树。否则 Python 可能能启动，但实际导入到的是另一套 Megatron/MindSpeed 组合，后续报错会很像模型或参数问题。

```shell
export PYTHONPATH=$PYTHONPATH:<your_local_megatron_lm_path>
export MEGATRON_LM_PATH=<your_local_megatron_lm_path>
python -c "import megatron, os; print(megatron.__file__); print(os.environ.get('MEGATRON_LM_PATH'))"
```

如果二者不一致，先修环境变量，再继续排查模型构建、权重加载或并行配置。


## NPU微信群

<img src="https://raw.githubusercontent.com/modelscope/ms-swift/main/docs/resources/wechat/npu.png" width="250">
