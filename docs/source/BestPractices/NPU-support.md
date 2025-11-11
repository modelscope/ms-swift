# NPU支持

## 环境准备

实验环境：8 * 昇腾910B3 64G (设备由[@chuanzhubin](https://github.com/chuanzhubin)提供, 感谢对modelscope和swift的支持～)

```shell
# 创建新的conda虚拟环境(可选)
conda create -n swift-npu python=3.10 -y
conda activate swift-npu

# 设置pip全局镜像 (可选,加速下载)
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip install ms-swift -U

# 安装torch-npu
pip install torch-npu decorator
# 如果你想要使用deepspeed (控制显存占用,训练速度会有一定下降)
pip install deepspeed
```

测试环境是否安装正确，NPU能否被正常加载：
```python
from transformers.utils import is_torch_npu_available
import torch

print(is_torch_npu_available())  # True
print(torch.npu.device_count())  # 8
print(torch.randn(10, device='npu:0'))
```

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
以下介绍LoRA的微调, 全参数微调设置参数`--train_type full`即可.

| 模型大小 | NPU数量 | deepspeed类型 | 最大显存占用量   |
|------|-------|-------------|-----------|
| 7B   | 1     | None        | 1 * 28 GB |
| 7B   | 4     | None        | 4 * 22 GB |
| 7B   | 4     | zero2       | 4 * 28 GB |
| 7B   | 4     | zero3       | 4 * 22 GB |
| 7B   | 8     | None        | 8 * 22 GB |
| 14B  | 1     | None        | 1 * 45 GB |
| 14B  | 8     | None        | 8 * 51 GB |
| 14B  | 8     | zero2       | 8 * 49 GB |
| 14B  | 8     | zero3       | 8 * 31 GB |

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
    --train_type lora \
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
    --train_type lora \
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
    --train_type lora \
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
    --train_type lora \
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
NPU不支持使用vllm进行推理/部署加速, 但是可以使用原生pytorch进行部署.

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
