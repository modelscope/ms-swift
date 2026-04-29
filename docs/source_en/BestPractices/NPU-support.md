# NPU Support

ms-swift supports Ascend NPUs. You can fine-tune models and run inference on Ascend NPUs.

This document describes how to prepare the environment, train models, save and merge checkpoints, run inference, deploy services, and troubleshoot common issues on Ascend NPUs.

If this is your first time using ms-swift on NPUs, we recommend reading this document in the following order:

1. Check "Support Scope at a Glance" first to confirm whether your model, algorithm, and backend have been verified.
2. Use "Choose Your Usage Path" to decide whether you only need the base environment or also need MindSpeed/Megatron-SWIFT.
3. Choose "Local Environment Installation" or "Image/Container Environment Installation" according to your own environment management preference, then run "NPU Availability Check".
4. Use "Quick Start" to complete one ModelScope model LoRA training, merge, inference, and deployment flow.
5. For larger-scale training, continue reading the DDP, DeepSpeed, and MindSpeed/Megatron-SWIFT sections.

## Support Scope at a Glance

Recommended base environment versions:

| software  | version         |
| --------- | --------------- |
| Python    | >= 3.10, < 3.12 |
| CANN      | == 8.5.1        |
| torch     | == 2.7.1        |
| torch_npu | == 2.7.1.post2  |

For base environment setup, see the [Ascend PyTorch installation guide](https://gitcode.com/Ascend/pytorch). The examples in this document were verified on 8 * Ascend 910B3 64G.

| Primary feature | Feature               | Status        |
| --------------- | --------------------- | ------------- |
| Training        | CPT                   | Supported     |
|                 | SFT                   | Supported     |
|                 | DPO                   | Supported     |
|                 | RM                    | Supported     |
| Distributed     | DDP                   | Supported     |
|                 | FSDP                  | Supported     |
|                 | FSDP2                 | Supported     |
|                 | DeepSpeed             | Supported     |
|                 | MindSpeed(Megatron)   | Supported     |
| PEFT            | FULL                  | Supported     |
|                 | LoRA                  | Supported     |
|                 | QLoRA                 | Not supported |
| RLHF            | GRPO                  | Supported     |
|                 | PPO                   | Supported     |
| Optimization    | FA and other fused ops | Supported    |
|                 | Liger-Kernel          | Not supported |
| Deployment      | PT                    | Supported     |
|                 | vLLM                  | Supported     |
|                 | SGLang                | Not supported |

### Verified SFT Combinations

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

### Verified RL Combinations

| algorithm | model families      | strategy  | rollout engine | hardware          |
| --------- | ------------------- | --------- | -------------- | ----------------- |
| **GRPO**  | Qwen2.5-7B-Instruct | deepspeed | vllm-ascend    | Atlas 900 A2 PODc |
| **GRPO**  | Qwen3-8B            | deepspeed | vllm-ascend    | Atlas 900 A2 PODc |
| **DPO**   | Qwen2.5-7B-Instruct | deepspeed | vllm-ascend    | Atlas 900 A2 PODc |
| **DPO**   | Qwen3-8B            | deepspeed | vllm-ascend    | Atlas 900 A2 PODc |
| **PPO**   | Qwen2.5-7B-Instruct | deepspeed | vllm-ascend    | Atlas 900 A2 PODc |
| **PPO**   | Qwen3-8B            | deepspeed | vllm-ascend    | Atlas 900 A2 PODc |

### Not Yet Supported or Not Fully Verified

| item                                      |
| ----------------------------------------- |
| Liger-kernel                              |
| Quantization/QLoRA related features       |
| Using SGLang as the inference engine      |
| Enabling ETP for LoRA training in Megatron |

## Choose Your Usage Path

| Scenario                               | Recommended path                                      | Need MindSpeed |
| -------------------------------------- | ----------------------------------------------------- | -------------- |
| Ordinary SFT/LoRA/inference            | Local environment installation or image/container installation | No             |
| Megatron-SWIFT large-model training     | Install the base environment, then install MindSpeed/Megatron/mcore-bridge | Yes            |
| GRPO/PPO/DPO and other RLHF workflows   | Base training environment + vLLM-Ascend rollout/deploy | Usually no     |
| Only verifying whether NPUs are usable  | Run the NPU availability check script                 | No             |

## Environment Preparation

### Image/Container Environment Installation

The official NPU image is still being prepared for release. Before the official image is released, you can build a container environment with CANN, PyTorch, torch_npu, and ms-swift dependencies from the Dockerfile provided by the project. The container approach makes dependency versions easier to freeze and helps reproduce the same environment across multiple Ascend machines.

Clone the modelscope repository first, then build the image with [Dockerfile.ascend](https://github.com/modelscope/modelscope/blob/master/docker/Dockerfile.ascend) and [build_image.py](https://github.com/modelscope/modelscope/blob/master/docker/build_image.py):

```shell
git clone https://github.com/modelscope/modelscope.git
cd modelscope
DOCKER_REGISTRY=ms-swift python docker/build_image.py \
  --image_type ascend \
  --python_version 3.11.11 \
  --soc_version ascend910b1 \
  --arch arm
```

The current `build_image.py` generates Ascend image names in the format `{DOCKER_REGISTRY}:{swift_branch}-{atlas_hardware}-{python_tag}-{arch}`. The command above uses the ARM-based Atlas 900 A2 PODc as an example and usually generates `ms-swift:main-A2-py311-arm`. Save the image name and workspace path in variables as shown below. Replace the image name with the actual one from your build log.

```shell
export IMAGE_NAME=ms-swift:main-A2-py311-arm
export WORKSPACE=/path/to/workspace
```

Before starting the container, check which NPU devices are exposed on the host:

```shell
ls /dev/davinci*
```

When starting the container, mount the NPU devices, driver, firmware, `npu-smi`, and required log directories. The following example uses the common 8-card device range `davinci0` to `davinci7`. Some machines may also expose devices up to `davinci15`; in that case, add the corresponding devices to `docker run` according to the output of `ls /dev/davinci*`.

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

After entering the container, run `source /usr/local/Ascend/ascend-toolkit/set_env.sh` first, then run the NPU availability check below to confirm that the container can access the Ascend devices. If the container cannot detect NPUs, check `/dev/davinci*`, `/dev/davinci_manager`, the driver directory, and `npu-smi` mounts first.

### Local Environment Installation

```shell
# Create a new conda virtual environment (optional)
conda create -n swift-npu python=3.11 -y
conda activate swift-npu

# Source the CANN environment before the following steps
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Set the global pip mirror (optional, speeds up downloads)
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip install ms-swift -U

# Install from source
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e .

# Install torch-npu
pip install torch_npu decorator
# If you want to use deepspeed (to reduce memory usage, with some speed overhead)
pip install deepspeed

# If you need evaluation features, install the following package
pip install evalscope[opencompass]

# If you need vllm-ascend for inference, install the following packages (for more versions, see the [vLLM-Ascend official website](https://docs.vllm.ai/projects/ascend/en/latest/installation.html))
pip install vllm==0.14.0
pip install vllm-ascend==0.14.0rc1
```

### NPU Availability Check

Check whether the environment is installed correctly and whether NPUs can be loaded:

```python
from transformers.utils import is_torch_npu_available
import torch

print(is_torch_npu_available())  # True
print(torch.npu.device_count())  # 8
print(torch.randn(10, device='npu:0'))
```

### Optional MindSpeed/Megatron-SWIFT Installation

If you need MindSpeed(Megatron-LM), install the required dependencies as follows.

```shell
# 1. Clone Megatron-LM and switch to v0.15.3
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout v0.15.3
cd ..

# 2. Clone and install MindSpeed
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed
git checkout core_r0.15.3
pip install -e .
cd ..

# 3. Clone and install mcore-bridge
git clone https://github.com/modelscope/mcore-bridge.git
cd mcore-bridge
pip install -e .
cd ..

# 4. Set environment variables
export PYTHONPATH=$PYTHONPATH:<your_local_megatron_lm_path>
export MEGATRON_LM_PATH=<your_local_megatron_lm_path>
```

Run the following command to verify that MindSpeed(Megatron-LM) is configured correctly:

```shell
python -c "import mindspeed.megatron_adaptor; from swift.megatron.init import init_megatron_env; init_megatron_env(); print('✓ Megatron-SWIFT configuration verified successfully in the NPU environment!')"
```

### Environment Inspection

Check NPU P2P connectivity. In the example below, each NPU is connected to the other NPUs through 7 HCCS links.

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

Check NPU status. For details about the `npu-smi` command, see the [official documentation](https://support.huawei.com/enterprise/en/doc/EDOC1100079287/10dcd668).

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

## Quick Start: ModelScope Model + Dataset

If you want to quickly verify the environment with a ModelScope model and dataset, you can use this section to complete the full flow: train LoRA, find the latest checkpoint, Merge LoRA, run CLI inference, start a service, and validate it with curl. The example uses a small model and a small data sample so that it can run quickly. To use your own model or dataset, modify the ID variables at the beginning.

```shell
export MODEL_ID=Qwen/Qwen3-0.6B
export DATASET_ID=AI-ModelScope/alpaca-gpt4-data-zh
export WORK_DIR=output/npu-modelscope-qwen3-0_6b-lora
```

Train and save a LoRA checkpoint:

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

After training finishes, checkpoints are saved under `$WORK_DIR/*/checkpoint-*`. Use the following commands to select the latest checkpoint and merge LoRA into a full model:

```shell
export CKPT_DIR=$(ls -dt $WORK_DIR/*/checkpoint-* | head -n 1)

ASCEND_RT_VISIBLE_DEVICES=0 \
swift export \
    --adapters $CKPT_DIR \
    --merge_lora true

export MERGED_DIR=${CKPT_DIR}-merged
```

You can verify inference either with the LoRA checkpoint directly or with the merged full model:

```shell
# Load the LoRA checkpoint directly
ASCEND_RT_VISIBLE_DEVICES=0 \
swift infer \
    --adapters $CKPT_DIR \
    --stream true \
    --temperature 0 \
    --max_new_tokens 512

# Load the merged full model
ASCEND_RT_VISIBLE_DEVICES=0 \
swift infer \
    --model $MERGED_DIR \
    --stream true \
    --temperature 0 \
    --max_new_tokens 512
```

To start an OpenAI-compatible deployment service, use the merged full model:

```shell
ASCEND_RT_VISIBLE_DEVICES=0 \
swift deploy \
    --model $MERGED_DIR \
    --host 0.0.0.0 \
    --port 8000 \
    --max_new_tokens 512 \
    --served_model_name npu-modelscope-qwen3-0_6b
```

After the service starts, validate the API with curl:

```shell
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "npu-modelscope-qwen3-0_6b",
"messages": [{"role": "user", "content": "Briefly introduce Ascend NPU in one sentence."}],
"max_tokens": 128,
"temperature": 0
}'
```

## Training

The following examples introduce LoRA fine-tuning. For full-parameter fine-tuning, set `--tuner_type full`. For **more training scripts**, see [examples/ascend/train](https://github.com/modelscope/ms-swift/tree/main/examples/ascend/train). For general pre-training, SFT, LoRA, full-parameter training, and custom dataset usage, see [Pre-training and Fine-tuning](../Instruction/Pre-training-and-Fine-tuning.md).

| Model size | NPU count | DeepSpeed type | Max memory usage |
| ---------- | --------- | -------------- | ---------------- |
| 7B         | 1         | None           | 1 * 28 GB        |
| 7B         | 4         | None           | 4 * 22 GB        |
| 7B         | 4         | zero2          | 4 * 28 GB        |
| 7B         | 4         | zero3          | 4 * 22 GB        |
| 7B         | 8         | None           | 8 * 22 GB        |
| 14B        | 1         | None           | 1 * 45 GB        |
| 14B        | 8         | None           | 8 * 51 GB        |
| 14B        | 8         | zero2          | 8 * 49 GB        |
| 14B        | 8         | zero3          | 8 * 31 GB        |

### Single-Card Training

Start single-card fine-tuning with the following command:

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

The following example uses 4 NPUs for DDP training.

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

### DeepSpeed Training

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

### Qwen3.5 Single-Node Multi-Card LoRA Example

The following is an NPU LoRA example for a newer model. It uses Qwen3.5-4B for demonstration. Four-card data parallelism is usually faster than single-card training. If you already have local model and dataset files, replace `--model` and `--dataset` with local paths.

```shell
# Experiment environment: 4 * Ascend 910B3
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

When tuning parameters, focus on memory, throughput, and stability:

- Reduce memory usage: first reduce `--max_length`, `--per_device_train_batch_size`, and `--lora_rank`; if OOM still occurs, enable `--deepspeed zero2/zero3`. ZeRO can significantly reduce memory pressure but introduces communication and scheduling overhead.
- Improve throughput: increase `--per_device_train_batch_size` when memory allows, and use `--gradient_accumulation_steps` to keep the global batch size. Increase `--dataset_num_proc` if preprocessing is slow, and increase `--dataloader_num_workers` if data loading is the bottleneck.
- Control save overhead: do not set `--save_steps` too small, because frequent checkpoint saving slows down training. `--save_total_limit 2` is usually enough to keep the best checkpoint and the last checkpoint.
- Improve stability: on NPUs, prefer `bfloat16`. If you see abnormal loss or NaN, first lower the learning rate and batch size; if necessary, temporarily switch to `float32` for comparison.

For more parameter details, see [Command-line Parameters](../Instruction/Command-line-parameters.md).

### NPU Model Patch Switch

ms-swift enables model-level patches by default in NPU environments to adapt some Transformers models to Ascend NPU operators and compatibility requirements. You usually do not need to disable them. If you suspect abnormal loss or forward errors are related to the NPU model patch and want to compare against native Transformers behavior, set:

```shell
swift sft ... --enable_npu_model_patch false
```

## Model Saving, Merge LoRA, and Resume Training

Use `--output_dir` to set the output directory, `--save_steps` to control checkpoint save intervals, and `--save_total_limit` to control how many checkpoints to keep. After LoRA training, the checkpoint directory contains adapter weights, training arguments, and trainer state. A typical directory layout is:

```text
output/Qwen3.5-4B-NPU/vx-xxx/
├── checkpoint-100/
├── checkpoint-200/
└── ...
```

If you only need inference or want to continue LoRA training, use the checkpoint directory directly. If you want an independent full-model directory for vLLM-Ascend deployment, offline distribution, or later quantization, run Merge LoRA:

```shell
ASCEND_RT_VISIBLE_DEVICES=0 \
swift export \
    --adapters output/Qwen3.5-4B-NPU/vx-xxx/checkpoint-xxx \
    --merge_lora true
```

The merged model is saved under `checkpoint-xxx-merged` by default. You can then load it like any regular model with `--model checkpoint-xxx-merged`.

If training is interrupted and you need to resume from a checkpoint, keep the original training arguments unchanged and add `--resume_from_checkpoint`:

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

`--resume_from_checkpoint` restores model weights, optimizer state, random seeds, and training progress. If you only want to load model weights without restoring the optimizer state or data skipping state, also set `--resume_only_model true`. For details, see `resume_from_checkpoint`, `resume_only_model`, `save_steps`, and `save_total_limit` in [Command-line Parameters](../Instruction/Command-line-parameters.md).

## Inference

Original model:

```shell
ASCEND_RT_VISIBLE_DEVICES=0 swift infer \
    --model Qwen/Qwen2-7B-Instruct \
    --stream true --max_new_tokens 2048
```

After LoRA fine-tuning:

```shell
ASCEND_RT_VISIBLE_DEVICES=0 swift infer \
    --adapters xxx/checkpoint-xxx --load_data_args true \
    --stream true --max_new_tokens 2048
```

For full-parameter training or a merged LoRA model, point `--model` to the full-weight directory:

```shell
ASCEND_RT_VISIBLE_DEVICES=0 swift infer \
    --model xxx/checkpoint-xxx-merged \
    --stream true --max_new_tokens 2048
```

## Deployment

### Deployment with Native Transformers

Original model:

```shell
ASCEND_RT_VISIBLE_DEVICES=0 swift deploy --model Qwen/Qwen2-7B-Instruct --max_new_tokens 2048
```

After LoRA fine-tuning:

```shell
ASCEND_RT_VISIBLE_DEVICES=0 swift deploy --adapters xxx/checkpoint-xxx --max_new_tokens 2048

# Deploy the full weights after Merge LoRA
ASCEND_RT_VISIBLE_DEVICES=0 swift export --adapters xx/checkpoint-xxx --merge_lora true
ASCEND_RT_VISIBLE_DEVICES=0 swift deploy --model xxx/checkpoint-xxx-merged --max_new_tokens 2048
```

### Deployment with vLLM-Ascend

Install from PyPI:

```shell
# Refer to the official vLLM-Ascend compatibility matrix; the following versions are verified in this document.
pip install vllm==0.14.0
pip install vllm-ascend==0.14.0rc1
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

# Deploy the full weights after Merge LoRA
ASCEND_RT_VISIBLE_DEVICES=0 swift export \
    --adapters xx/checkpoint-xxx \
    --merge_lora true

ASCEND_RT_VISIBLE_DEVICES=0 swift deploy \
    --model xxx/checkpoint-xxx-merged \
    --infer_backend vllm \
    --max_new_tokens 2048
```

## Evaluation

After training, inference, or deployment, you can evaluate the original model or fine-tuned checkpoint with SWIFT's built-in EvalScope integration. For complete arguments and examples, see [Evaluation](../Instruction/Evaluation.md).

## Release

If you need to publish NPU-trained checkpoints, merged models, or quantized models to ModelScope/HuggingFace, use the push capability in `swift export`. For complete arguments and examples, see [Export and Push](../Instruction/Export-and-push.md#push-models).

## FAQ

For general questions, see [Frequently Asked Questions](../Instruction/Frequently-asked-questions.md). This section records common NPU-specific issues and troubleshooting steps.

### Q1: How do I confirm that the current environment detects NPUs correctly?

First confirm that you have run `source /usr/local/Ascend/ascend-toolkit/set_env.sh`, then run the environment check script in this document. Normally, `is_torch_npu_available()` should return `True`, `torch.npu.device_count()` should show the number of available NPUs, and you should be able to create a tensor on `npu:0`. If this fails, first check whether CANN, `torch`, and `torch_npu` match the recommended versions in this document.

### Q2: How should I choose between FSDP, DeepSpeed, and Megatron-SWIFT?

For ordinary SFT, first refer to the verified `FSDP1/FSDP2/deepspeed` combinations in this document. For larger models or higher parallelism requirements, use Megatron-SWIFT and install MindSpeed, Megatron-LM, and mcore-bridge as described in the installation section. DeepSpeed can reduce memory pressure but may reduce speed, so compare it with FSDP when tuning performance.

### Q3: Do I need to manually disable the NPU model patch?

Usually no. ms-swift enables model-level patches by default in NPU environments to adapt some Transformers models to Ascend NPU operators and compatibility requirements. Only when troubleshooting abnormal loss or forward errors, and when you suspect the issue is related to the NPU patch, should you temporarily set `--enable_npu_model_patch false` and compare against native Transformers behavior.

### Q4: What should I know when using vLLM-Ascend for deployment or RL rollout?

Install the `vllm` and `vllm-ascend` versions recommended in this document, and prioritize model and algorithm combinations that have been verified in the compatibility tables. The `sglang` inference engine is not yet fully verified for NPU scenarios. For high-performance inference or RL rollout on NPUs, prefer `vllm-ascend`.

### Q5: What happens if I forget to run `source set_env.sh`?

Common symptoms include `is_torch_npu_available()` returning `False`, `torch.npu.device_count()` returning 0, or runtime errors about missing CANN/HCCL shared libraries. After entering a new shell or container, run:

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

If NNAL/ATB or other components are installed, source their corresponding `set_env.sh` files according to your environment.

### Q6: How do I diagnose a `torch` and `torch_npu` version mismatch?

Install according to the recommended versions in this document. When versions mismatch, common symptoms include `import torch_npu` failures, invisible NPU devices, operator registration failures, and C++ symbol/runtime errors. Check versions with:

```shell
python -c "import torch, torch_npu; print(torch.__version__); print(torch_npu.__version__)"
```

If versions do not match, uninstall and reinstall the full CANN/PyTorch/torch_npu stack consistently. Do not upgrade only one package.

### Q7: What happens if `ASCEND_RT_VISIBLE_DEVICES` and `NPROC_PER_NODE` do not match?

For distributed training, they should match. For example, `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3` usually corresponds to `NPROC_PER_NODE=4`. If the process count is larger than the number of visible devices, ranks may fail to bind devices, multiple processes may compete for the same device, initialization may hang, or HCCL may report errors. If the process count is smaller, only part of the visible NPUs will be used.

### Q8: What should I check first when multi-card training hangs?

First confirm that every rank has started and that `ASCEND_RT_VISIBLE_DEVICES` matches `NPROC_PER_NODE`. Then identify whether logs stop during data preprocessing, model construction, weight loading, or HCCL initialization. For NPU/HCCL low-level logs, check:

```shell
ls ~/ascend/log/debug/plog
```

If the Python process is still running but produces no output for a long time, use `pystack` to inspect the stack of each rank and determine whether it is stuck in data loading, communication, or model forward/backward.

### Q9: How do I initially troubleshoot HCCL connection or timeout issues?

Use `npu-smi info` and `npu-smi info -t topo` to check device health and topology, then check whether other jobs are occupying the same NPU group. For single-node training, first verify card IDs, process count, and visible devices. For multi-node training, also verify networking, rank configuration, communication ports, and environment variables on all nodes. If old training processes remain on the same machine, clean up the corresponding training processes and retry.

### Q10: Why is `npu-smi` unavailable inside the container?

Usually some device or driver files were not mounted. Check whether `docker run` includes `/dev/davinci*`, `/dev/davinci_manager`, `/dev/devmm_svm`, `/dev/hisi_hdc`, `/usr/local/Ascend/driver`, `/usr/local/Ascend/firmware`, `/usr/local/sbin/npu-smi`, and `/etc/ascend_install.info`. If `npu-smi info` fails on the host itself, fix the host driver environment first.

### Q11: How should I choose between native transformers deployment and vLLM-Ascend deployment?

Native transformers deployment has better compatibility and is suitable for validating whether the model, adapter, template, and output are correct. vLLM-Ascend is better for high-throughput services, RL rollout, or OpenAI-compatible serving. If you encounter vLLM-Ascend version or operator issues, first confirm that the model itself works with the transformers backend, then switch to vLLM-Ascend to troubleshoot the performance backend.

### Q12: What should I do if vLLM-Ascend reports device type mismatch or undefined symbol?

This is usually not caused by training script arguments. It often means the `vllm-ascend` wheel does not match the current hardware, PyTorch version, or C++ ABI. First check package build information and current versions:

```shell
python -c "import torch, vllm_ascend; print(torch.__version__); print(vllm_ascend.__file__)"
```

If the error message contains `Current device type ... does not match the installed version's device type ...`, `undefined symbol`, or similar text, reinstall `torch`, `torch_npu`, `vllm`, and `vllm-ascend` according to the device type (A2/A3/other) and the official compatibility matrix. Do not replace only one package.

### Q13: Can FP8 or quantized models be trained directly on NPUs?

Do not assume they can. Before downloading or loading a large model, check whether `config.json` contains `quantization_config`, and check the actual dtype in safetensors. Quantization/QLoRA is still listed as not supported or not fully verified in the NPU support scope. If model weights are FP8 block quantized and your NPU software stack does not support that FP8 path, use BF16 weights first, or convert the model offline to BF16 before training/loading.

### Q14: How do I troubleshoot Megatron-SWIFT importing the wrong Megatron/MindSpeed?

Before running Megatron-SWIFT, `PYTHONPATH` and `MEGATRON_LM_PATH` must point to the same Megatron-LM source tree. Otherwise Python may start successfully while importing a different Megatron/MindSpeed combination, which can make later errors look like model or argument issues.

```shell
export PYTHONPATH=$PYTHONPATH:<your_local_megatron_lm_path>
export MEGATRON_LM_PATH=<your_local_megatron_lm_path>
python -c "import megatron, os; print(megatron.__file__); print(os.environ.get('MEGATRON_LM_PATH'))"
```

If they do not match, fix the environment variables before continuing with model construction, weight loading, or parallel configuration troubleshooting.

## NPU WeChat Group

<img src="https://raw.githubusercontent.com/modelscope/ms-swift/main/docs/resources/wechat/npu.png" width="250">
