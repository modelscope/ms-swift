# NPU支持

我们在 ms-swift 上增加了对昇腾 NPU 的支持，用户可以在昇腾 NPU 上进行模型的微调和推理。

本文档介绍了如何在昇腾 NPU 上进行环境准备、模型微调、推理和部署。

## 安装

基础环境准备：

| software  | version         |
| --------- | --------------- |
| Python    | >= 3.10, < 3.12 |
| CANN      | == 8.5.1        |
| torch     | == 2.7.1        |
| torch_npu | == 2.7.1.post2  |


基础环境准备请参照这份 [Ascend PyTorch 安装文档](https://gitcode.com/Ascend/pytorch)。

实验环境：8 * 昇腾910B3 64G

### 镜像/容器环境安装
官方 NPU 镜像仍在发布流程中。在镜像正式发布前，推荐使用项目提供的 Dockerfile 自行构建一个包含 CANN、PyTorch、torch_npu 与 ms-swift 依赖的容器环境。容器方式的优势是依赖版本更容易固化，也便于在多台昇腾机器之间复现实验环境。

先 clone modelscope 仓库，然后使用仓库中的 [Dockerfile.ascend](https://github.com/modelscope/modelscope/blob/master/docker/Dockerfile.ascend) 和 [build_image.py](https://github.com/modelscope/modelscope/blob/master/docker/build_image.py) 构建镜像：

```shell
git clone https://github.com/modelscope/modelscope.git
cd modelscope
DOCKER_REGISTRY=ms-swift python docker/build_image.py --image_type ascend 
```


启动容器时需要把 NPU 设备、驱动、固件、`npu-smi` 和必要日志目录挂载进去。下面示例按 8 卡机器配置，如果只使用部分卡，可以按实际情况减少 `--device=/dev/davinci*` 的数量：

```shell
docker run -it \
  --name swift-ascend \
  --network=host --ipc=host --shm-size=128g \
  --device=/dev/davinci0 --device=/dev/davinci1 \
  --device=/dev/davinci2 --device=/dev/davinci3 \
  --device=/dev/davinci4 --device=/dev/davinci5 \
  --device=/dev/davinci6 --device=/dev/davinci7 \
  --device=/dev/davinci8 --device=/dev/davinci9 \
  --device=/dev/davinci10 --device=/dev/davinci11 \
  --device=/dev/davinci12 --device=/dev/davinci13 \
  --device=/dev/davinci14 --device=/dev/davinci15 \
  --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /var/log/npu:/var/log/npu \
  -v /home/zyh:/workspace/zyh \
  dsa:8.5.1-a3-ubuntu22.04-py3.11-ascend910_9391-torch2.7.1-9.99.0-ascend-test \
  /bin/bash
```

进入容器后，建议先执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`，再运行后文的 NPU 可用性检查脚本，确认容器内可以正确访问昇腾设备。

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

# 如果需要使用 vllm-ascend 进行推理，请安装以下包
pip install vllm==0.14.0
pip install vllm-ascend==0.14.0rc1
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

# merge-lora并推理
ASCEND_RT_VISIBLE_DEVICES=0 swift export --adapters xx/checkpoint-xxx --merge_lora true
ASCEND_RT_VISIBLE_DEVICES=0 swift deploy --model xxx/checkpoint-xxx-merged --max_new_tokens 2048
```

### 使用vLLM-ascend进行部署
使用pypi进行安装：
```shell
# Install vllm-project/vllm. The newest supported version is v0.11.0.
pip install vllm==0.14.0

# Install vllm-project/vllm-ascend from pypi.
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

# merge-lora并推理
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

## 本地模型和数据集端到端示例

下面给出一个从训练、保存、推理到部署的完整流程。示例使用本机已有的小模型和数据集，便于快速跑通；如果需要换成其他本地模型或数据集，只需要替换前面的路径变量。

```shell
export MODEL_DIR=/home/model/Qwen3-0.6B
export DATASET_DIR=/home/zyh/dataset/alpaca-gpt4-data-zh
export WORK_DIR=output/npu-local-qwen3-0_6b-lora
```

训练并保存LoRA checkpoint：

```shell
ASCEND_RT_VISIBLE_DEVICES=0 \
swift sft \
    --model $MODEL_DIR \
    --dataset $DATASET_DIR#1000 \
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
# 直接加载LoRA checkpoint
ASCEND_RT_VISIBLE_DEVICES=0 \
swift infer \
    --adapters $CKPT_DIR \
    --stream true \
    --temperature 0 \
    --max_new_tokens 512

# 加载merge后的完整权重
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
    --served_model_name npu-local-qwen3-0_6b
```

服务启动后，用 curl 验证接口：

```shell
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "npu-local-qwen3-0_6b",
"messages": [{"role": "user", "content": "用一句话介绍昇腾NPU。"}],
"max_tokens": 128,
"temperature": 0
}'
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


## NPU微信群

<img src="https://raw.githubusercontent.com/modelscope/ms-swift/main/docs/resources/wechat/npu.png" width="250">
