# NPU支持

我们在 ms-swift 上增加了对昇腾 NPU 的支持，用户可以在昇腾 NPU 上进行模型的微调和推理。

本文档介绍了如何在昇腾 NPU 上进行环境准备、模型微调、推理和部署。

## 安装

基础环境准备：

| software  | version         |
| --------- | --------------- |
| Python    | >= 3.10, < 3.12 |
| CANN      | == 8.3.RC1      |
| torch     | == 2.7.1        |
| torch_npu | == 2.7.1        |


基础环境准备请参照这份 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)。


## 环境准备

实验环境：8 * 昇腾910B3 64G
### 环境安装
```shell
# 创建新的 conda 虚拟环境（可选）
conda create -n swift-npu python=3.10 -y
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
pip install torch-npu decorator
# 如果你想要使用 deepspeed（控制显存占用，训练速度会有一定下降）
pip install deepspeed

# 如果需要使用 evaluation 功能，请安装以下包
pip install evalscope[opencompass]

# 如果需要使用 vllm-ascend 进行推理，请安装以下包
pip install vllm==0.11.0
pip install vllm-ascend==0.11.0rc3
```

测试环境是否安装正确，NPU能否被正常加载：
```python
from transformers.utils import is_torch_npu_available
import torch

print(is_torch_npu_available())  # True
print(torch.npu.device_count())  # 8
print(torch.randn(10, device='npu:0'))
```

**如果需要使用 MindSpeed(Megatron-LM)，请按照下面引导安装必要依赖**
```shell
# 1. 获取并切换 Megatron-LM 至 core_v0.12.1 版本
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
cd ..

# 2. 获取并安装 MindSpeed
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
git checkout 2.3.0_core_r0.12.1
pip install -e .
cd ..

# 3. 设置环境变量
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

## 微调

以下介绍LoRA的微调, 全参数微调设置参数`--tuner_type full`即可. **更多训练脚本**参考[这里](https://github.com/modelscope/ms-swift/tree/main/examples/ascend/train).

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

通过如下命令启动单卡微调: （注意: 如果微调期间出现nan的情况, 请设置`--torch_dtype float32`.）

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

# merge-lora并推理
ASCEND_RT_VISIBLE_DEVICES=0 swift export --adapters xx/checkpoint-xxx --merge_lora true

ASCEND_RT_VISIBLE_DEVICES=0 swift infer \
    --model xxx/checkpoint-xxx-merged --load_data_args true \
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

# merge-lora并推理
ASCEND_RT_VISIBLE_DEVICES=0 swift export --adapters xx/checkpoint-xxx --merge_lora true
ASCEND_RT_VISIBLE_DEVICES=0 swift deploy --model xxx/checkpoint-xxx-merged --max_new_tokens 2048
```

### 使用vLLM-ascend进行部署
使用pypi进行安装：
```shell
# Install vllm-project/vllm. The newest supported version is v0.11.0.
pip install vllm==0.11.0

# Install vllm-project/vllm-ascend from pypi.
pip install vllm-ascend==0.11.0rc3
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

# merge-lora并推理
ASCEND_RT_VISIBLE_DEVICES=0 swift export \
    --adapters xx/checkpoint-xxx \
    --merge_lora true

ASCEND_RT_VISIBLE_DEVICES=0 swift deploy \
    --model xxx/checkpoint-xxx-merged \
    --infer_backend vllm \
    --max_new_tokens 2048
```

## 支持现状
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

------


### 表 1：SFT 类算法

| algorithm | model families              | strategy              | hardware          |
| --------- | --------------------------- | --------------------- | ----------------- |
| SFT       | Qwen2.5-0.5B-Instruct       | FSDP1/FSDP2/deepspeed | Atlas 900 A2 PODc |
| SFT       | Qwen2.5-1.5B-Instruct       | FSDP1/FSDP2/deepspeed | Atlas 900 A2 PODc |
| SFT       | Qwen2.5-7B-Instruct         | FSDP1/FSDP2/deepspeed | Atlas 900 A2 PODc |
| SFT       | Qwen2.5-VL-3B-Instruct      | FSDP1/FSDP2/deepspeed | Atlas 900 A2 PODc |
| SFT       | Qwen2.5-VL-7B-Instruct      | FSDP1/FSDP2/deepspeed | Atlas 900 A2 PODc |
| SFT       | Qwen2.5-Omni-3B             | FSDP1/FSDP2/deepspeed | Atlas 900 A2 PODc |
| SFT       | Qwen3-8B                    | FSDP1/FSDP2/deepspeed | Atlas 900 A2 PODc |
| SFT       | Qwen3-32B                   | FSDP1/FSDP2/deepspeed | Atlas 900 A2 PODc |
| SFT       | Qwen3-VL-30B-A3B-Instruct   | FSDP1/FSDP2/deepspeed | Atlas 900 A2 PODc |
| SFT       | Qwen3-Omni-30B-A3B-Instruct | FSDP1/FSDP2/deepspeed | Atlas 900 A2 PODc |
| SFT       | InternVL3-8B                | FSDP1/FSDP2/deepspeed | Atlas 900 A2 PODc |
| SFT       | Ovis2.5-2B                  | FSDP1/FSDP2/deepspeed | Atlas 900 A2 PODc |

------

### 表 2：RL 类算法

| algorithm | model families      | strategy  | rollout engine | hardware          |
| --------- | ------------------- | --------- | -------------- | ----------------- |
| **GRPO**  | Qwen2.5-7B-Instruct | deepspeed | vllm-ascend    | Atlas 900 A2 PODc |
| **GRPO**  | Qwen3-8B            | deepspeed | vllm-ascend    | Atlas 900 A2 PODc |
| **DPO**   | Qwen2.5-7B-Instruct | deepspeed | vllm-ascend    | Atlas 900 A2 PODc |
| **DPO**   | Qwen3-8B            | deepspeed | vllm-ascend    | Atlas 900 A2 PODc |
| **PPO**   | Qwen2.5-7B-Instruct | deepspeed | vllm-ascend    | Atlas 900 A2 PODc |
| **PPO**   | Qwen3-8B            | deepspeed | vllm-ascend    | Atlas 900 A2 PODc |

---

### 表 3：当前 NPU 暂不支持 / 未完全验证的模块

| item                              |
| --------------------------------- |
| Liger-kernel                      |
| 量化/QLoRA相关                    |
| 使用sglang作为推理引擎            |
| 使用megatron时开启ETP进行lora训练 |


## NPU微信群

<img src="https://raw.githubusercontent.com/modelscope/ms-swift/main/docs/resources/wechat/npu.png" width="250">
