# Mcore Bridge

Megatron 以其卓越的训练速度和丰富的并行技术而著称，但也因此带来了较高的使用门槛。因此mcore-bridge 应运而生，旨在让 Megatron 训练像 transformers 一样简单易用。通过 Mcore-Bridge，用户可以：
1. 直接加载 safetensors 格式的模型权重，无缝使用 Megatron 进行高效训练。直接保存 训练权重为 safetensors 格式，无需额外转换。
2. 兼容 LoRA 增量权重的双向转换。
3. 兼容GRPO/GKD等算法的`Megatron->vLLM`权重同步。
4. 支持多机转换超大规模模型。

Mcore-Bridge 兼容 Dense/MoE/多模态等多种模型架构。训练完成后，转换后的模型可直接使用 transformers、vLLM、SGLang 等主流推理框架部署。

## 无缝训练
目前Mcore-Bridge已支持TP/PP/EP/ETP/VPP等并行技术，支持所有Megatron-SWIFT支持的模型架构，参考[支持的模型文档](../Instruction/Supported-models-and-datasets.md)。以下介绍Mcore-Bridge的无缝训练能力，分别介绍Dense模型和Moe模型。

### Dense模型
以下为多模态模型Qwen3-VL模型训练的例子:
```shell
# 2 * 76GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --model Qwen/Qwen3-VL-8B-Instruct \
    --load_safetensors true \
    --save_safetensors true \
    --dataset 'AI-ModelScope/LaTeX_OCR:human_handwrite#5000' \
    --load_from_cache_file true \
    --tensor_model_parallel_size 2 \
    --sequence_parallel true \
    --packing true \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --split_dataset_ratio 0.01 \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --max_epochs 1 \
    --save megatron_output/Qwen3-VL-8B-Instruct \
    --save_interval 200 \
    --vit_gradient_checkpointing false \
    --max_length 2048 \
    --num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 8
```

然后我们对验证集部分进行推理：
```shell
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model megatron_output/Qwen3-VL-8B-Instruct/vx-xxx/checkpoint-xxx \
    --load_data_args true \
    --stream true
```

### Moe模型
以下为纯文本模型Qwen3-Moe模型CoT训练的例子:

```shell
# 8 * 76GiB, 3s/it
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
megatron sft \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --load_safetensors true \
    --save_safetensors true \
    --dataset 'swift/Chinese-Qwen3-235B-Thinking-2507-Distill-data-110k-SFT#20000' \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --moe_permute_fusion true \
    --pipeline_model_parallel_size 2 \
    --decoder_first_pipeline_num_layers 25 \
    --tensor_model_parallel_size 4 \
    --expert_model_parallel_size 4 \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-6 \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --save megatron_output/Qwen3-30B-A3B-Instruct-2507 \
    --eval_interval 500 \
    --save_interval 500 \
    --max_length 8192 \
    --packing true \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --moe_expert_capacity_factor 2 \
    --attention_backend flash
```

对训练后的权重进行推理：
```shell
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model megatron_output/Qwen3-30B-A3B-Instruct-2507/vx-xxx/checkpoint-xxx \
    --stream true \
    --max_new_tokens 1024
```

## LoRA导出

Mcore-Bridge除了支持全参数的导入导出，还支持单独对LoRA增量模型进行导入导出。

以下为纯文本模型Qwen3-Moe模型使用LoRA自我认知训练的例子：
- 若你希望导出merge后的权重，而不是LoRA增量权重，请设置`--merge_lora true`。设置`--merge_lora true`的兼容性更好，支持所有系列模型。
- 注意：由于transformers和Megatron模型结构并不一定一致（例如transformers的Qwen3-VL-Moe的专家部分并不是Linear实现，而是Parameters），因此部分模型无法转换LoRA增量权重（若Qwen3-VL-Moe只设置linear_proj和linear_qkv训练LoRA也支持转换）。但大多数的模型支持LoRA转换，例如：Qwen3-Moe，Qwen3-Omni-Moe，GLM4.5-V等。
```shell
# 50GiB
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --load_safetensors true \
    --save_safetensors true \
    --merge_lora false \
    --dataset 'swift/Chinese-Qwen3-235B-2507-Distill-data-110k-SFT#2000' \
              'swift/self-cognition#1000' \
    --load_from_cache_file true \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --split_dataset_ratio 0.01 \
    --moe_permute_fusion true \
    --expert_model_parallel_size 2 \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 8 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --save megatron_output/Qwen3-30B-A3B-Instruct-2507 \
    --eval_interval 200 \
    --save_interval 200 \
    --max_length 2048 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --moe_expert_capacity_factor 2 \
    --attention_backend flash \
    --model_author swift \
    --model_name swift-robot
```

对导出的LoRA权重进行推理：
```shell
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --adapters megatron_output/Qwen3-30B-A3B-Instruct-2507/vx-xxx/checkpoint-xxx \
    --stream true
```

## 导出与转换精度测试

Mcore-Bridge除了支持在训练中进行safetensors的转换和保存，也支持了`megatron export`命令用于单独的权重导出。`megatron export`支持在权重转换时，对转换精度进行测试，这在接入新模型时验证接入准确性很有帮助。通常，Megatron-SWIFT已经接入的模型不会出现精度不对齐的情况，你可以放心设置`--test_convert_precision false`。
- 提示：多模态模型请关注`mean_diff (with loss)`字段，`mean_diff`因包含图像tokens且该部分不计算损失，有较大的diff。

全参数权重：
```shell
# safetensors -> torch_dist
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
megatron export \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --save Qwen3-30B-A3B-Instruct-2507-mcore \
    --to_mcore true \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 2 \
    --pipeline_model_parallel_size 2 \
    --test_convert_precision true
```

```shell
# torch_dist -> safetensors
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
megatron export \
    --load Qwen3-30B-A3B-Instruct-2507-mcore \
    --save Qwen3-30B-A3B-Instruct-2507-hf \
    --to_hf true \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 2 \
    --pipeline_model_parallel_size 2 \
    --test_convert_precision true
```

LoRA权重：
```shell
# torch_dist -> safetensors
# 若你需要进行merge-lora，并测试merge-lora后的精度对齐，你只需要设置`--merge_lora true`即可
# 你也可以将`--model safetensors-path`修改为`--load torch-dist-path`。这两种方式是等价的，mcore-bridge会自动处理。
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
megatron export \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --adapter_load megatron_output/Qwen3-30B-A3B-Instruct-2507/vx-xxx \
    --save megatron_output/Qwen3-30B-A3B-Instruct-2507/vx-xxx-lora \
    --merge_lora false \
    --to_hf true \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 2 \
    --pipeline_model_parallel_size 2 \
    --test_convert_precision true
```

```shell
# safetensors -> torch_dist
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
megatron export \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --adapters megatron_output/Qwen3-30B-A3B-Instruct-2507/vx-xxx-lora \
    --save megatron_output/Qwen3-30B-A3B-Instruct-2507/vx-xxx-mcore \
    --merge_lora false \
    --to_mcore true \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 2 \
    --pipeline_model_parallel_size 2 \
    --test_convert_precision true
```

Merge-LoRA:
```shell
# torch_dist -> torch_dist
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
megatron export \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --adapter_load megatron_output/Qwen3-30B-A3B-Instruct-2507/vx-xxx \
    --save megatron_output/Qwen3-30B-A3B-Instruct-2507/vx-xxx-merged \
    --merge_lora true \
    --to_mcore true \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 2 \
    --pipeline_model_parallel_size 2
```


## 使用代码

你需要创建以下文件（test.py），然后运行`CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 test.py`。以下为使用Mcore-Bridge进行权重加载、导出、保存的示例代码。

```python
import torch

from swift.megatron import MegatronArguments, convert_hf_config, get_megatron_model_meta
from swift.llm import get_model_tokenizer
from megatron.training.initialize import initialize_megatron

model_id = 'Qwen/Qwen3-4B-Instruct-2507'
_, processor = get_model_tokenizer(model_id, load_model=False, download_model=True)
model_info = processor.model_info
megatron_model_meta = get_megatron_model_meta(model_info.model_type)
config_kwargs = convert_hf_config(model_info.config)
megatron_args = MegatronArguments(
    model=model_id,
    tensor_model_parallel_size=2,
    torch_dtype=torch.bfloat16,
    **config_kwargs,
)
extra_args = megatron_args.parse_to_megatron()
initialize_megatron(args_defaults=extra_args)
mg_model = megatron_model_meta.model_provider()
bridge = megatron_model_meta.bridge_cls()
# 加载权重
bridge.load_weights(mg_model, model_info.model_dir)
# 导出权重
for name, parameters in bridge.export_weights([mg_model]):
    pass
# 保存权重
bridge.save_weights([mg_model], 'output/Qwen3-4B-Instruct-2507-new')
```

推理新产生的权重：
```shell
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model output/Qwen3-4B-Instruct-2507-new \
    --model_type qwen3_nothinking \
    --stream true
```

LoRA权重的加载、导出和存储同理，运行`CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 test.py`
```python
import torch

from swift.megatron import (
    MegatronArguments, convert_hf_config, get_megatron_model_meta, prepare_mcore_model
)
from swift.llm import get_model_tokenizer
from megatron.training.initialize import initialize_megatron

model_id = 'Qwen/Qwen3-30B-A3B-Instruct-2507'
_, processor = get_model_tokenizer(model_id, load_model=False, download_model=True)
model_info = processor.model_info
megatron_model_meta = get_megatron_model_meta(model_info.model_type)
config_kwargs = convert_hf_config(model_info.config)
megatron_args = MegatronArguments(
    model=model_id,
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=2,
    expert_model_parallel_size=2,
    sequence_parallel=True,
    torch_dtype=torch.bfloat16,
    train_type='lora',
    **config_kwargs,
)
extra_args = megatron_args.parse_to_megatron()
initialize_megatron(args_defaults=extra_args)
mg_model = megatron_model_meta.model_provider()
# 加载权重
bridge = megatron_model_meta.bridge_cls()
bridge.load_weights(mg_model, model_info.model_dir)
# 准备LoRA并加载
peft_model = prepare_mcore_model(mg_model)
print(f'peft_model: {peft_model}')
# bridge.load_weights(mg_model, 'adapter-path', is_peft_format=True)
# 导出权重
for name, parameters in bridge.export_weights([mg_model], is_peft_format=True):
    pass
bridge.save_weights([mg_model], 'output/Qwen3-30B-A3B-Instruct-2507-lora', is_peft_format=True)
```

推理新产生的权重：
```shell
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --adapters output/Qwen3-30B-A3B-Instruct-2507-lora \
    --stream true
```
