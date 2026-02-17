# NPU Support

We add Ascend NPU support in ms-swift, so you can fine-tune and run inference on Ascend NPUs.

This document describes how to prepare the environment, fine-tune, run inference and deploy on NPUs.

## Installation

Base environment requirements:

| Software  | Version         |
| --------- | --------------- |
| Python    | >= 3.10, < 3.12 |
| CANN      | == 8.3.RC1      |
| torch     | == 2.7.1        |
| torch_npu | == 2.7.1        |

For detailed environment setup, please refer to the [Ascend PyTorch installation guide](https://gitcode.com/Ascend/pytorch).

## Environment Preparation

Experiment Environment: 8 * Ascend 910B3 64G
### Environment Installation
```shell
# Create a new conda virtual environment (optional)
conda create -n swift-npu python=3.10 -y
conda activate swift-npu

# Note: Before proceeding with subsequent operations, you need to source and activate CANN environment first
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Set pip global mirror (optional, to speed up downloads)
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip install ms-swift -U

# Install from source
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e .

# Install torch-npu
pip install torch-npu decorator
# If you want to use deepspeed (to control memory usage, training speed might decrease)
pip install deepspeed

# If you need the evaluation functionality, please install the following package
pip install evalscope[opencompass]

# If you need to use vllm-ascend for inference, please install the following packages
pip install vllm==0.11.0
pip install vllm-ascend==0.11.0rc3
```

Check if the test environment is installed correctly and whether the NPU can be loaded properly.
```python
from transformers.utils import is_torch_npu_available
import torch

print(is_torch_npu_available())  # True
print(torch.npu.device_count())  # 8
print(torch.randn(10, device='npu:0'))
```

**If you need to use MindSpeed (Megatron-LM), please follow the guide below to install the necessary dependencies**
```shell
# 1. Obtain and switch Megatron-LM to core_v0.12.1
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_v0.12.1
cd ..

# 2. Install MindSpeed
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
git checkout 2.3.0_core_r0.12.1
pip install -e .
cd ..

# 3. Set environment variables
export PYTHONPATH=$PYTHONPATH:<your_local_megatron_lm_path>
export MEGATRON_LM_PATH=<your_local_megatron_lm_path>
```
Run the following command to verify if MindSpeed (Megatron-LM) is configured successfully:
```shell
python -c "import mindspeed.megatron_adaptor; from swift.megatron.init import init_megatron_env; init_megatron_env(); print('✓ NPU environment Megatron-SWIFT configuration verified successfully!')"
```
### Environment Viewing
Check the P2P connections of the NPU, where we can see that each NPU is interconnected through 7 HCCS links with other NPUs.
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

Check the status of the NPU. For detailed information about the `npu-smi` command, please refer to the [official documentation](https://support.huawei.com/enterprise/en/doc/EDOC1100079287/10dcd668).
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

## Fine-tuning
The following introduces the fine-tuning of LoRA. To perform full-parameter fine-tuning, simply set the parameter `--tuner_type full`. For **more training scripts**, refer to [here](https://github.com/modelscope/ms-swift/tree/main/examples/ascend/train).

| Model Size | Number of NPUs | Deepspeed Type | Max Memory Usage   |
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

### Single Card Training

Start single card fine-tuning with the following command: (Note: If NaN occurs during fine-tuning, please set `--torch_dtype float32`.)

```shell
# Experiment environment: Ascend 910B3
# Memory requirement: 28 GB
# Runtime: 8 hours
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

### Data Parallel Training
We use 4 cards for DDP training.

```shell
# Experiment environment: 4 * Ascend 910B3
# Memory requirement: 4 * 22 GB
# Runtime: 2 hours
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

### Deepspeed Training

ZeRO2:
```shell
# Experiment environment: 4 * Ascend 910B3
# Memory requirement: 4 * 28GB
# Runtime: 3.5 hours
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
# Experiment environment: 4 * Ascend 910B3
# Memory requirement: 4 * 22 GB
# Runtime: 8.5 hours
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

## Inference

Original Model:
```shell
ASCEND_RT_VISIBLE_DEVICES=0 swift infer \
    --model Qwen/Qwen2-7B-Instruct \
    --stream true --max_new_tokens 2048
```

After LoRA Fine-tuning:
```shell
ASCEND_RT_VISIBLE_DEVICES=0 swift infer \
    --adapters xxx/checkpoint-xxx --load_data_args true \
    --stream true --max_new_tokens 2048

# Merge LoRA and infer
ASCEND_RT_VISIBLE_DEVICES=0 swift export --adapters xx/checkpoint-xxx --merge_lora true

ASCEND_RT_VISIBLE_DEVICES=0 swift infer \
    --model xxx/checkpoint-xxx-merged --load_data_args true \
    --stream true --max_new_tokens 2048
```

## Deployment

### Deployment with native Transformers

Original model:
```shell
ASCEND_RT_VISIBLE_DEVICES=0 swift deploy --model Qwen/Qwen2-7B-Instruct --max_new_tokens 2048
```

After LoRA fine-tuning:
```shell
ASCEND_RT_VISIBLE_DEVICES=0 swift deploy --adapters xxx/checkpoint-xxx --max_new_tokens 2048

# Merge LoRA and deploy
ASCEND_RT_VISIBLE_DEVICES=0 swift export --adapters xx/checkpoint-xxx --merge_lora true
ASCEND_RT_VISIBLE_DEVICES=0 swift deploy --model xxx/checkpoint-xxx-merged --max_new_tokens 2048
```

### Deployment with vLLM-ascend

Install via PyPI:
```shell
# Install vllm-project/vllm. The newest supported version is v0.11.0.
pip install vllm==0.11.0

# Install vllm-project/vllm-ascend from PyPI.
pip install vllm-ascend==0.11.0rc3
```

Original model:
```shell
ASCEND_RT_VISIBLE_DEVICES=0 swift deploy \
    --model Qwen/Qwen2.5-7B-Instruct \
    --infer_backend vllm \
    --max_new_tokens 2048
```

After LoRA fine-tuning:
```shell
ASCEND_RT_VISIBLE_DEVICES=0 swift deploy \
    --adapters xxx/checkpoint-xxx \
    --infer_backend vllm \
    --max_new_tokens 2048

# Merge LoRA and deploy
ASCEND_RT_VISIBLE_DEVICES=0 swift export \
    --adapters xx/checkpoint-xxx \
    --merge_lora true

ASCEND_RT_VISIBLE_DEVICES=0 swift deploy \
    --model xxx/checkpoint-xxx-merged \
    --infer_backend vllm \
    --max_new_tokens 2048
```

## Current Support Status

| Primary Feature | Feature                | Status        |
| --------------- | ---------------------- | ------------- |
| Training Paradigm | CPT                   | Supported     |
|                 | SFT                    | Supported     |
|                 | DPO                    | Supported     |
|                 | RM                     | Supported     |
| Distributed     | DDP                    | Supported     |
|                 | FSDP                   | Supported     |
|                 | FSDP2                  | Supported     |
|                 | DeepSpeed              | Supported     |
|                 | MindSpeed (Megatron)   | Supported     |
| PEFT            | FULL                   | Supported     |
|                 | LoRA                   | Supported     |
|                 | QLoRA                  | Not Supported |
| RLHF            | GRPO                   | Supported     |
|                 | PPO                    | Supported     |
| Performance Optimization | Fused ops such as FA | Supported |
|                 | Liger-Kernel           | Not Supported |
| Deployment      | PT                     | Supported     |
|                 | vLLM                   | Supported     |
|                 | SGLang                 | Not Supported |

---

### Table 1: SFT Algorithms

| Algorithm | Model Families              | Strategy              | Hardware          |
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

---

### Table 2: RL Algorithms

| Algorithm | Model Families      | Strategy  | Rollout Engine | Hardware          |
| --------- | ------------------- | --------- | -------------- | ----------------- |
| **GRPO**  | Qwen2.5-7B-Instruct | deepspeed | vllm-ascend    | Atlas 900 A2 PODc |
| **GRPO**  | Qwen3-8B            | deepspeed | vllm-ascend    | Atlas 900 A2 PODc |
| **DPO**   | Qwen2.5-7B-Instruct | deepspeed | vllm-ascend    | Atlas 900 A2 PODc |
| **DPO**   | Qwen3-8B            | deepspeed | vllm-ascend    | Atlas 900 A2 PODc |
| **PPO**   | Qwen2.5-7B-Instruct | deepspeed | vllm-ascend    | Atlas 900 A2 PODc |
| **PPO**   | Qwen3-8B            | deepspeed | vllm-ascend    | Atlas 900 A2 PODc |

---

### Table 3: Modules Not Yet Supported / Fully Verified on NPUs

| Item                     |
| ------------------------ |
| Liger-kernel             |
| Quantization/QLoRA       |
| Using SGLang as inference engine |
| Enable ETP for LoRA training when using Megatron |


## NPU WeChat Group

<img src="https://raw.githubusercontent.com/modelscope/ms-swift/main/docs/resources/wechat/npu.png" width="250">
