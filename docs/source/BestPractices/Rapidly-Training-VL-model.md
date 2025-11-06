# 快速训练VL模型

本文档提供从零开始快速训练视觉语言(Vision-Language, VL)模型的最佳实践。

涉及的模型链接：
- [Qwen2.5-VL-7B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct)
- [Qwen3-8B](https://www.modelscope.cn/models/Qwen/Qwen3-8B)

训练的模型链接：
- [Simple-VL-8B](https://www.modelscope.cn/models/swift/Simple-VL-8B/summary)


本训练流程基于 Qwen2.5-VL-7B-Instruct 模型架构，将其内部的语言模型（LLM）部分替换为 Qwen3-8B 的权重，训练模型的视觉理解能力。具体步骤如下：

1. 修改原始模型的配置文件 config.json，使其适配 Qwen3-8B 的模型结构。
2. 初始化并加载新的模型权重，保存为新模型。
3. 对新模型进行两阶段微调：
    1. 第一阶段：仅训练视觉到语言的对齐模块（aligner），冻结 ViT 和 LLM 部分。
    2. 第二阶段：解冻所有模块，联合训练提升整体性能。


## 模型修改

### 修改配置文件 config.json
因为 Qwen2.5-VL-7B-Instruct 模型的底模 Qwen2.5-7B-Instruct 与 Qwen3-8B 在模型结构上存在部分差异（比如层数，hidden_state_dims），我们首先需要基于Qwen2.5-VL-7B-Instruct的config.json文件，创建一个新的config.json文件，并修改以下参数对齐Qwen3-8B

```
修改
1. hidden_size 3584->4096
2. intermediate_size: 18944->12288
3. num_attention_heads: 28->32
4. num_key_value_heads: 4->8
5. num_hidden_layers: 28->36
6. vocab_size:152064->151936
7. max_window_layers:28->36
8. out_hidden_size: 3584->4096

新增
1. head_dim： 128
```

### 模型权重初始化与替换
使用以下 Python 脚本完成模型权重的初始化、替换与保存：

```python
import torch
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoModelForCausalLM, AutoConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLPatchMerger, Qwen2_5_VLModel
from accelerate import Accelerator

# 加载原始 VL 模型和 Qwen3-8B 模型
qwen2_5_vl_7b_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    device_map="cuda",
    torch_dtype=torch.bfloat16
)
device = qwen2_5_vl_7b_model.device
qwen3_8b_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    device_map=device,
    torch_dtype=torch.bfloat16
)

# 加载配置
old_config = AutoConfig.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
new_config = AutoConfig.from_pretrained("/path/to/new_config_dir") # 新 config 的文件夹路径
new_visual_config = new_config.vision_config

# 1. 替换 ViT 到 LLM 的 merger(aligner) 层
new_merger = Qwen2_5_VLPatchMerger(
            dim=new_visual_config.out_hidden_size,
            context_dim=new_visual_config.hidden_size,
            spatial_merge_size=new_visual_config.spatial_merge_size,
        ).to(device).to(torch.bfloat16)
qwen2_5_vl_7b_model.visual.merger = new_merger

# 2. 替换 VL 模型的 LLM 部分
new_llm_model = Qwen2_5_VLModel(new_config).to(device).to(torch.bfloat16)

for name, param in qwen3_8b_model.model.named_parameters():
    if name in new_llm_model.state_dict():
        new_llm_model.state_dict()[name].copy_(param)

qwen2_5_vl_7b_model.model = new_llm_model
qwen2_5_vl_7b_model.lm_head = qwen3_8b_model.lm_head

# 3. 保存修改后的模型
accelerator = Accelerator()
accelerator.save_model(
    model=qwen2_5_vl_7b_model,
    save_directory="/path/to/save/Qwen3-VL-Model",
    max_shard_size="4GB",
    safe_serialization=True
)
```

保存完权重后，将原 Qwen2.5-VL-7B-Instruct 模型文件夹中除模型权重的文件(包括`model.safetensors.index.json`) 复制到新的模型权重文件夹中，并替换 config.json 为新修改的 config.json文件。

## 训练

为简化流程，我们跳过预训练（pretrain），直接进入监督微调（SFT）。训练分为两个阶段：

### stage1 训练 Aligner 层
仅训练视觉到语言的对齐层（Aligner），冻结 ViT 和 LLM 部分：

```bash
NNODES=$WORLD_SIZE \
NODE_RANK=$RANK \
NPROC_PER_NODE=8 \
MAX_PIXELS=1003520 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model /path/to/new_vl_model \
    --model_type qwen2_5_vl \
    --train_type full \
    --dataset xxx  \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --freeze_vit true \
    --freeze_llm true \
    --freeze_aligner false \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-6 \
    --gradient_accumulation_steps 8 \
    --eval_steps -1 \
    --save_steps 1000 \
    --save_total_limit 10 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --deepspeed zero2
```

### stage2 训练整个模型
解冻所有模块，联合训练以增强模型的整体视觉理解能力：

```bash
NNODES=$WORLD_SIZE \
NODE_RANK=$RANK \
NPROC_PER_NODE=8 \
MAX_PIXELS=1003520 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift sft \
    --model /path/to/stage1_checkpoint \
    --model_type qwen2_5_vl \
    --train_type full \
    --dataset xxx \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --freeze_vit false \
    --freeze_llm false \
    --freeze_aligner false \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --learning_rate 5e-6 \
    --gradient_accumulation_steps 8 \
    --eval_steps -1 \
    --save_steps 1000 \
    --save_total_limit 10 \
    --logging_steps 5 \
    --max_length 8192 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --deepspeed zero2
```

## 推理/部署/评测

### 推理
通过`swift infer`来推理训练得到的模型
```bash
swift infer \
    --model /path/to/stage2_checkpoint

```

### 部署
使用 vLLM 加速模型服务部署：

```
CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
swift deploy \
    --model /path/to/stage2_checkpoint \
    --infer_backend vllm \
    --vllm_gpu_memory_utilization 0.9 \
    --vllm_max_model_len 8192 \
    --max_new_tokens 2048 \
    --vllm_limit_mm_per_prompt '{"image": 5, "video": 2}' \
    --served_model_name Qwen3-VL
```

### 评测
通过 [EvalScope](https://github.com/modelscope/evalscope/) 对训练得到的 VL 模型进行评测

以下是以 MMMU benchmark 为例的评测代码：
```python
from evalscope import TaskConfig, run_task

task_cfg_dict = TaskConfig(
    work_dir='outputs',
    eval_backend='VLMEvalKit',
    eval_config={
        'data': ['MMMU_DEV_VAL'],
        'mode': 'all',
        'model': [
            {'api_base': 'http://localhost:8000/v1/chat/completions',
            'key': 'EMPTY',
            'name': 'CustomAPIModel',
            'temperature': 0.6,
            'type': 'Qwen3-VL',
            'img_size': -1,
            'video_llm': False,
            'max_tokens': 512,}
            ],
        'reuse': False,
        'nproc': 64,
        'judge': 'exact_matching'},
)

run_task(task_cfg=task_cfg_dict)
```
