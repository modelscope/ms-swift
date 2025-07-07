
# Megatron-SWIFT训练

SWIFT引入了Megatron的并行技术来加速大模型的训练，包括数据并行、张量并行、流水线并行、序列并行，上下文并行，专家并行。支持Qwen3、[Qwen3-MoE](https://github.com/modelscope/ms-swift/blob/main/examples/train/megatron/qwen3_moe.sh)、Qwen2.5、Llama3、Deepseek-R1蒸馏系等模型的预训练和微调。完整支持的模型可以参考[支持的模型与数据集文档](./支持的模型和数据集.md)。

## 环境准备
使用Megatron-SWIFT，除了安装swift依赖外，还需要安装以下内容：

```shell
# 推荐torch版本：2.5 / 2.6
pip install pybind11
# transformer_engine
# 若出现安装错误，可以参考该issue解决: https://github.com/modelscope/ms-swift/issues/3793
pip install git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.3

# apex
git clone https://github.com/NVIDIA/apex
cd apex
# https://github.com/modelscope/ms-swift/issues/4176
git checkout e13873debc4699d39c6861074b9a3b2a02327f92
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# megatron-core
pip install git+https://github.com/NVIDIA/Megatron-LM.git@core_r0.12.0
```

或者你也可以使用镜像：
```
modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.4.0-py310-torch2.6.0-vllm0.8.5.post1-modelscope1.27.1-swift3.5.3
modelscope-registry.cn-beijing.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.4.0-py310-torch2.6.0-vllm0.8.5.post1-modelscope1.27.1-swift3.5.3
modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.4.0-py310-torch2.6.0-vllm0.8.5.post1-modelscope1.27.1-swift3.5.3
```

依赖库Megatron-LM中的训练模块将由swift进行git clone并安装。你也可以通过环境变量`MEGATRON_LM_PATH`指向已经下载好的repo路径（断网环境，[core_r0.12.0分支](https://github.com/NVIDIA/Megatron-LM/tree/core_r0.12.0)）。


## 快速入门案例

这里介绍使用2卡80GiB A100对Qwen2.5-7B-Instruct模型进行自我认知微调的快速入门案例，以下最佳实践可以在10分钟内完成。

首先，我们需要将HF格式的权重转为Megatron格式：
- 若出现OOM，将`CUDA_VISIBLE_DEVICES=0`删除即可。
- "ms-swift>=3.6"推荐增加`--test_convert_precision true`参数测试转换精度。
```shell
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model Qwen/Qwen2.5-7B-Instruct \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir Qwen2.5-7B-Instruct-mcore
```

然后，使用以下脚本进行训练，训练所需显存资源为2*80GiB：
- 若使用多机训练，建议共享磁盘，并将`--save`指定为相同的路径。
```shell
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --load Qwen2.5-7B-Instruct-mcore \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --tensor_model_parallel_size 2 \
    --sequence_parallel true \
    --micro_batch_size 16 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --max_epochs 1 \
    --save megatron_output/Qwen2.5-7B-Instruct \
    --save_interval 100 \
    --max_length 2048 \
    --system 'You are a helpful assistant.' \
    --num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 4 \
    --model_author swift \
    --model_name swift-robot
```

最后，将Megatron格式权重转为HF格式：
- 注意：`--mcore_model`请指向`iter_xxx`的上级目录。默认会使用`latest_checkpointed_iteration.txt`中对应的checkpoint。
- 若出现OOM，将`CUDA_VISIBLE_DEVICES=0`删除即可。
- "ms-swift>=3.6"推荐增加`--test_convert_precision true`参数测试转换精度。
```shell
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --mcore_model megatron_output/Qwen2.5-7B-Instruct/vx-xxx \
    --to_hf true \
    --torch_dtype bfloat16 \
    --output_dir megatron_output/Qwen2.5-7B-Instruct/vx-xxx-hf
```

我们对生成的HF格式权重进行推理：
```shell
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model megatron_output/Qwen2.5-7B-Instruct/vx-xxx-hf \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048
```

推理结果如下：
```
<<< who are you?
I am a language model developed by swift, you can call me swift-robot. How can I assist you?
```

- 若要进行预训练，你可以使用`megatron pt`替代`megatron sft`，这将会使用生成式的template进行训练。
- **更多案例**：包括packing、多机、32K上下文、DPO、MoE模型、预训练，可以查看[这里](https://github.com/modelscope/ms-swift/tree/main/examples/train/megatron)。
- 自定义数据集格式和ms-swift相同，参考[自定义数据集文档](../Customization/自定义数据集.md)。

## Benchmark

使用`megatron sft`和`swift sft`在单机八卡A800环境下进行Dense/MoE模型全参数训练的速度对比如下，对应脚本参考[这里](https://github.com/modelscope/ms-swift/tree/main/examples/train/megatron/benchmark)。

**Dense** Qwen2.5-14B:

|          | Megatron-LM | Deepspeed-ZeRO2 | Deepspeed-ZeRO3 |
| -------- | ----------- | ---------- | ---------- |
| 训练速度 |      9.04s/it       |  10.32s/it   | 10.56s/it |
| 显存占用 | 8\*64GB     |  8\*80GB   | 8\*58GB |

**MoE** Qwen1.5-MoE-A2.7B:

|          | Megatron-LM | Deepspeed-ZeRO2 | Deepspeed-ZeRO3 |
| -------- | ----------- | ---------- | ---------- |
| 训练速度 |      2.93s/it       |  6.02s/it   | 24.30s/it |
| 显存占用 | 8\*66GB     |  8\*72GB   | 8\*50GB |


## 命令行参数

### Megatron参数


**训练参数**:
- 🔥micro_batch_size: 每个device的批次大小，默认为1。
- 🔥global_batch_size: 总批次大小，等价于`micro_batch_size*数据并行大小*梯度累加步数`。默认为16。
- 🔥recompute_granularity: 重新计算激活的粒度，可选项为'full', 'selective'。其中full代表重新计算整个transformer layer，selective代表只计算transformer layer中的核心注意力部分。通常'selective'是推荐的。默认为'selective'。
- 🔥recompute_method: 该参数需将recompute_granularity设置为'full'才生效，可选项为'uniform', 'block'。默认为None。
- 🔥recompute_num_layers: 该参数需将recompute_granularity设置为'full'才生效，默认为None。若`recompute_method`设置为uniform，该参数含义为每个均匀划分的重新计算单元的transformer layers数量。例如你可以指定为`--recompute_granularity full --recompute_method uniform --recompute_num_layers 4`。recompute_num_layers越大，显存占用越小，计算成本越大。默认为None。
- recompute_modules: 选项包括"core_attn", "moe_act", "layernorm", "mla_up_proj", "mlp", "moe" ，默认值为，["core_attn"]。例如在MoE训练时，你可以通过指定`--recompute_granularity selective --recompute_modules core_attn moe`降低显存。其中"core_attn"、"mlp" 和 "moe" 使用常规检查点，"moe_act"、"layernorm" 和 "mla_up_proj" 使用输出丢弃检查点。
  - "core_attn"：重新计算 Transformer 层中的核心注意力部分。
  - "mlp"：重新计算密集的 MLP 层。
  - "moe"：重新计算 MoE 层。
  - "moe_act"：重新计算 MoE 中的 MLP 激活函数部分。
  - "layernorm"：重新计算 input_layernorm 和 pre_mlp_layernorm。
  - "mla_up_proj"：重新计算 MLA 上投影和 RoPE 应用部分。
- deterministic_mode: 确定性模式，这会导致训练速度下降，默认为False。
- 🔥train_iters: 训练的总迭代次数，默认为None。
- 🔥log_interval: log的时间间隔（单位：iters），默认为5。
- tensorboard_dir: tensorboard日志写入的目录。默认None，即存储在`f'{save}/runs'`目录下。
- no_masked_softmax_fusion: 默认为False。用于禁用query_key_value的scaling, masking, and softmax融合。
- no_bias_dropout_fusion: 默认为False。用于禁用bias和dropout的融合。
- no_bias_swiglu_fusion: 默认为False。指定`--no_bias_dropout_fusion true`，用于禁止bias和swiglu融合。
- no_rope_fusion: 默认为False。指定`--no_rope_fusion true`用于禁止rope融合。
- no_gradient_accumulation_fusion: 默认为False。指定`--no_gradient_accumulation_fusion true`用于禁用梯度累加融合。
- 🔥cross_entropy_loss_fusion: 启动交叉熵损失计算融合。默认为False。
- cross_entropy_fusion_impl: 交叉熵损失融合的实现。可选为'native'和'te'。默认为'native'。
- calculate_per_token_loss: 根据全局批次中的非填充token数量来对交叉熵损失进行缩放。默认为True。
- 🔥attention_backend: 使用的注意力后端 (flash、fused、unfused、local、auto)。默认为 auto。
- optimizer: 优化器类型，可选为'adam'、'sgd'。默认为adam。
- 🔥optimizer_cpu_offload: 将优化器状态卸载到 CPU。默认为False。
- optimizer_offload_fraction: 卸载到 CPU 的优化器状态所占比例。默认为1.。
- use_precision_aware_optimizer: 使用 TransformerEngine 中的精度感知优化器，该优化器允许将主参数和优化器状态设置为较低精度，例如 fp16 和 fp8。
- main_grads_dtype: 启用 use_precision_aware_optimizer 时主梯度的 dtype。可选为'fp32', 'bf16'。默认为'fp32'。
- main_params_dtype: 启用 use_precision_aware_optimizer 时主参数的 dtype。可选为'fp32', 'fp16'。默认为'fp32'。
- exp_avg_dtype: 启用 use_precision_aware_optimizer 时，adam 优化器中 exp_avg（即一阶矩）的 dtype。该 dtype 用于在训练过程中将优化器状态存储在内存中，但不会影响内核计算时的精度。可选为'fp32', 'fp16', 'bf16', 'fp8'。默认为'fp32'。
- exp_avg_sq_dtype: 启用 use_precision_aware_optimizer 时，adam 优化器中 exp_avg_sq（即二阶矩）的 dtype。该 dtype 用于在训练过程中将优化器状态存储在内存中，但不会影响内核计算的精度。可选为'fp32', 'fp16', 'bf16', 'fp8'。默认为'fp32'。
- dataloader_type: 默认为'cyclic'，可选为'single', 'cyclic', 'external'。若开启`--streaming`，则设置为`external`。
- manual_gc: 禁用默认垃圾回收器，手动触发垃圾回收。默认为False。
- manual_gc_interval: 触发垃圾回收的间隔。默认为0。
- seed: python、numpy、pytorch和cuda的随机种子，默认为42。
- 🔥num_workers: dataloder的workers数量，默认为4。
  - 注意：若设置`--streaming true`，则设置为1。
- seq_length: 默认为None，即设置为`max_length`。对数据集长度进行限制请使用基本参数中的`--max_length`控制，无需设置此参数。
- use_cpu_initialization: 在cpu上初始化权重，默认为False。在进行HF和MCore权重转换时会被使用。
- no_create_attention_mask_in_dataloader: 在dataloader中不创建attention mask，默认为True。
- extra_megatron_kwargs: 传入megatron的其他参数，使用json传递。默认为None。

**学习率参数**:
- 🔥lr: 初始学习率，最终会根据学习率预热策略和衰减策略决定每个迭代的学习率，默认为1e-5。
- lr_decay_style: 学习率衰减策略，默认为'cosine'。通常设置为'cosine', 'linear', 'constant'。
- 🔥lr_decay_iters: 学习率衰减的迭代次数。默认为None，则设置为`--train_iters`。
- lr_warmup_iters: 线性学习率预热的迭代次数，默认为0。
- 🔥lr_warmup_fraction: 线性学习率预热阶段所占比例，默认为None。
- 🔥min_lr: 学习率的最小值，将低于改阈值的学习率裁剪为该值，默认为0。

**正则化参数**:
- 🔥weight_decay: 默认为0.1。
- 🔥clip_grad: l2梯度裁剪，默认为1.0。
- adam_beta1: 默认0.9。
- adam_beta2: 默认0.95。
- adam_eps: 默认1e-8。
- sgd_momentum: 默认为0.9。

**checkpoint参数**:
- 🔥save: checkpoint的输出目录，默认None。在训练中，若未设置该参数，则默认为`f'megatron_output/{model_suffix}'`，例如`'megatron_output/Qwen2.5-7B-Instruct'`。
  - 注意：若在多机训练时，请确保每个节点的保存路径指向相同位置。否则你需要在训练后手动集中这些权重。
- 🔥save_interval: checkpoint保存的间隔（steps），默认为500。
  - 注意：训练结束时一定会保存权重。
- 🔥no_save_optim: 不保存optimizer，默认为False。
- 🔥no_save_rng: 不保存rng，默认为False。
- 🔥load: 加载的checkpoint目录，默认None。
- 🔥no_load_optim: 不载入optimizer，默认为False。
- 🔥no_load_rng: 不载入rng，默认为False。
- 🔥finetune: 将模型加载并微调。不加载检查点的优化器和随机种子状态，并将迭代数设置为0。默认为False。
  - 注意：断点续训`--load`，若设置`--finetune true`，将不会跳过数据集；若不设置，将跳过之前训练的数据集数量。
  - 流式数据集`--streaming`，暂不支持跳过数据集。
- ckpt_format: checkpoint的格式。可选为'torch', 'torch_dist', 'zarr'。默认为'torch_dist'。
- no_initialization: 不对权重进行初始化，默认为True。
- auto_detect_ckpt_format: 自动检测ckpt format为legacy还是distributed格式。默认为True。
- exit_on_missing_checkpoint: 如果设置了`–-load`，但找不到检查点，则直接退出，而不是初始化。默认为True。

**分布式参数**:
- distributed_backend: 分布式后端，可选为'nccl', 'gloo'。默认为nccl。
- 🔥use_distributed_optimizer: 使用分布式优化器。默认为True。
- 🔥tensor_model_parallel_size: tp数，默认为1。
- 🔥pipeline_model_parallel_size: pp数，默认为1。
- 🔥decoder_first_pipeline_num_layers: decoder第一个流水线阶段所包含的Transformer层数。默认为 None，表示将Transformer层数平均分配到所有流水线阶段。
- 🔥decoder_last_pipeline_num_layers: decoder最后一个流水线阶段所包含的Transformer层数。默认为 None，表示将Transformer层数平均分配到所有流水线阶段。
- 🔥sequence_parallel: 启动序列并行的优化器。默认为False。
- 🔥context_parallel_size: cp数，默认为1。
- tp_comm_overlap: 启用张量并行通信与GEMM（通用矩阵乘法）内核的重叠（降低通信耗时）。默认为False。
- overlap_grad_reduce: 启用DDP中grad reduce操作的重叠（降低DP通信耗时）。默认为False。
- overlap_param_gather: 启用分布式优化器中参数all-gather的重叠（降低DP通信耗时）。默认为False。
- distributed_timeout_minutes: torch.distributed的timeout时间（单位为分钟），该参数失效，使用[基础参数](./命令行参数.md#基本参数)中的ddp_timeout控制，默认为300000分钟。

**日志参数**:
- log_params_norm: 记录参数的norm。默认为False。
- log_throughput: 记录每个GPU的吞吐量。默认为False。
  - 注意：在非packing情况下，log_throughput并不准确，因为`seq_length`并不等于真实序列长度。
- tensorboard_log_interval: 记录到tensorboard的间隔（steps），默认为1。
- tensorboard_queue_size: 队列长度（与磁盘IO相关），类似于写入的间隔。默认为50。
- log_timers_to_tensorboard: 记录timers到tensorboard。默认为True。
- no_log_learning_rate_to_tensorboard: 不记录学习率到tensorboard。默认为False。
- log_validation_ppl_to_tensorboard: 将验证困惑度写入tensorboard。默认为True。
- log_memory_to_tensorboard: 将内存日志写入tensorboard。默认为True。
- logging_level: 日志级别。默认为None。
- wandb_project: wandb 项目名称。默认为''，即忽略wandb。
- wandb_exp_name: wandb 实验名称。默认为''。
- wandb_save_dir: 本地保存 wandb 结果的路径。默认为''。

**评估参数**:
- 🔥eval_iters: 评估的迭代次数，默认为-1，根据验证数据集的数量设置合适的值。
  - 注意：若使用流式数据集，该值需要手动设置。
- 🔥eval_interval: 评估的间隔（steps），即每训练多少steps进行评估，默认为None，即设置为save_interval。

**fp8参数**:
- fp8_format: 用于前向和反向传播中FP8张量的FP8格式方案。可选为'e4m3'，'hybrid'。默认为None。
- fp8_recipe: 用于前向和反向传播中 FP8 张量的 FP8 算法方案。可选为'tensorwise', 'delayed', 'mxfp8', 'blockwise'。默认为'delayed'。
- fp8_amax_history_len: 每个张量记录 amax 历史的步数。默认为1024。
- fp8_amax_compute_algo: 用于根据历史记录计算 amax 的算法。可选为'most_recent','max'。默认为'max'。
- fp8_param_gather: 保持计算参数为 fp8（不使用任何其他中间数据类型），并在 fp8 格式下执行参数的 all-gather 操作。默认为False。

**混合精度参数**:
- fp16: fp16模式。默认为None，会根据模型的torch_dtype进行设置。torch_dtype默认读取config.json。
- bf16: bf16模式。默认为None，会根据模型的torch_dtype进行设置。
- apply_query_key_layer_scaling: 将`Q * K^T` 缩放为 `1 / 层数`（例如：第layer_num层则除以layer_num）。这对fp16训练很有帮助。默认为None，即若使用`--fp16`，则设置为True。
- attention_softmax_in_fp32: 在attention_mask和softmax中使用fp32进行计算。默认为True。

**模型参数**: （以下参数通常不需要进行设置，会根据HF模型的config.json进行配置，用户无需关心）
- num_layers: transformer layers的层数，默认为None。
- hidden_size: transformer hidden size，默认为None。
- ffn_hidden_size: transformer FFN层的hidden size。默认为None，设置为`4*hidden_size`。
- num_attention_heads: transformer attention heads的个数，默认为None。
- group_query_attention: 默认为None。若`num_query_groups>1`，group_query_attention设置为True，否则为False。
- num_query_groups: 默认为1。
- max_position_embeddings: 位置编码的最大长度，默认为None。
- position_embedding_type: 位置编码的类型，可选为'learned_absolute'、'rope'、'mrope'、'relative'和'none'，默认为'rope'。
- rotary_base: 默认为10000。
- rotary_percent: 默认为1.。
- normalization: 可选为'LayerNorm', 'RMSNorm'，默认为RMSNorm。
- norm_epsilon: 默认为1e-5。
- swiglu: 使用swiglu替代默认的gelu。默认为True。
- untie_embeddings_and_output_weights: 解开embedding和输出权重的绑定，默认为True。
- disable_bias_linear: 禁用linear层的bias。默认为True。
- add_qkv_bias: 仅在QKV的linear中增加bias，默认为True。
- attention_dropout: 默认为0.。
- hidden_dropout: 默认为0.。
- kv_channels: 默认为None，设置为`args.hidden_size // args.num_attention_heads`。
- qk_layernorm: 是否对Q和K进行层归一化。
- transformer_impl: 使用哪种transformer实现，可选项为'local'和'transformer_engine'。默认为transformer_engine。
- padded_vocab_size: 完整词表大小，默认为None。
- rope_scaling: rope_scaling相关参数，默认为None。格式参考[llama3.1 config.json](https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B-Instruct/file/view/master?fileName=config.json&status=1)，传入json字符串。


**MoE参数**:
- num_experts: MoE的专家数，默认为None。自动从config.json读取。
- moe_layer_freq: MoE 层与 Dense 层之间的分布频率。默认为None。从config.json中读取。
- moe_ffn_hidden_siz: 每个专家的前馈网络（ffn）的隐藏层大小。默认为None，自动从config.json读取。若未读取到且`num_experts`不为None，则设置为ffn_hidden_size。
- moe_shared_expert_intermediate_size: 共享专家的总FFN隐藏层大小。如果有多个共享专家，它应等于 `num_shared_experts * ffn_size_of_each_shared_expert`。 默认为None。自动从config.json读取。
- moe_router_topk: 每个token路由到的专家数量。默认为None。自动从config.json读取。
- moe_router_pre_softmax: 为MoE启用预softmax路由，这意味着softmax会在top-k选择之前进行。默认为None。自动从config.json读取。
- 🔥moe_router_dtype: 用于路由计算和专家输出加权平均的数据类型。可选为'none', 'fp32'、'fp64'，这增强了数值稳定性，尤其是在专家数量较多时。与`moe_permute_fusion`一起使用时，性能影响可以忽略不计。默认为'fp32'。'none'代表不改变数据类型。
- moe_router_score_function: MoE TopK 路由的评分函数。可以为 "softmax" 或 "sigmoid"。默认为None，从config.json中读取。
- moe_router_bias_update_rate: 在无辅助损失负载均衡策略中，专家偏置的更新速率。专家偏置根据每个专家在全局批次中被分配的 token 数量进行更新，对于分配到的 token 较少的专家，偏置会增加；对于分配到的 token 较多的专家，偏置会减少。默认值 1e-3，与 DeepSeekV3 中使用的值相同。
- moe_router_enable_expert_bias: 在无辅助损失负载均衡策略中，带有动态专家偏置的 TopK 路由。路由决策基于路由分数与专家偏置之和。详情请参见：https://arxiv.org/abs/2408.15664。默认为None，自动从config.json读取。
- moe_router_topk_scaling_factor: 默认为None。从config.json中读取。
- moe_router_load_balancing_type: 确定路由器的负载均衡策略。可选项为"aux_loss"、"seq_aux_loss"、"sinkhorn"、"none"。默认值为 None。从config.json中读取。
- 🔥expert_model_parallel_size: 专家并行数，默认为1。
- moe_token_dispatcher_type: 要使用的token分发器类型。可选选项包括 'allgather'、'alltoall'、'flex'和'alltoall_seq'。默认值为'alltoall'。
- moe_enable_deepep: 实验性功能，启用DeepSeek/DeepEP以实现 MoE 模型中的高效令牌分发与组合。仅在设置`--moe_token_dispatcher_type flex`使用灵活令牌分发器时生效。
- 🔥moe_grouped_gemm: 当每个rank包含多个专家时，通过在多个流中启动多个本地 GEMM 内核，利用 TransformerEngine中的GroupedLinear提高利用率和性能。默认为False。
- 🔥moe_permute_fusion: 在令牌分发过程中融合令牌重排操作。默认为False。
- 🔥moe_aux_loss_coeff: 辅助损失的缩放系数：建议的初始值为 1e-2。默认为None。自动从config.json读取。
- moe_z_loss_coeff: z-loss 的缩放系数。默认为None。
- moe_expert_capacity_factor: 每个专家的容量因子，None表示不会丢弃任何token。默认为None。自动从config.json读取。
- 🔥moe_shared_expert_overlap: 启用共享专家计算与调度器通信之间的重叠。如果不启用此选项，共享专家将在路由专家之后执行。仅在设置了`moe_shared_expert_intermediate_size`时有效。默认为False。
- moe_token_drop_policy: 可选为'probs', 'position'。默认为'probs'。

**mla参数**
- multi_latent_attention: 是否使用MLA。默认为False。
- q_lora_rank: Query 张量低秩表示的rank值。默认为None，自动从config.json读取。
- kv_lora_rank: Key 和 Value 张量低秩表示的秩（rank）值。默认为None，自动从config.json读取。
- qk_head_dim: QK 投影中 head 的维度。 `q_head_dim = qk_head_dim + qk_pos_emb_head_dim`。默认为None，自动从config.json读取。
- qk_pos_emb_head_dim: QK 投影中位置嵌入的维度。默认为None，自动从config.json读取。

**DPO参数**:
- ref_load: ref_model的加载路径。默认为None，即设置为`load`。
- beta: 含义与[TRL](https://huggingface.co/docs/trl/main/en/dpo_trainer#trl.DPOConfig)相同。控制与参考模型偏差程度的参数。beta值越高，表示与参考模型的偏差越小。对于 IPO 损失函数 (loss_type="ipo")，beta是[论文](https://huggingface.co/papers/2310.12036)中所指的正则化参数。默认为0.1。
- rpo_alpha: 来自[RPO 论文](https://huggingface.co/papers/2404.19733)中的参数，用于控制损失函数中NLL项的权重（即SFT损失）。`loss = dpo_loss + rpo_alpha * nll_loss`。默认为1。
- reference_free: 是否忽略提供的参考模型，并隐式地使用一个对所有响应赋予相等概率的参考模型。默认为False。
- label_smoothing: 默认为0.。
- f_divergence_type: 默认为`reverse_kl`。可选值参考[TRL文档](https://huggingface.co/docs/trl/main/en/dpo_trainer)。
- loss_type: 默认为'sigmoid'。可选值参考[TRL文档](https://huggingface.co/docs/trl/main/en/dpo_trainer)。


### 训练参数

Megatron训练参数继承自Megatron参数和基本参数。基本参数的内容可以参考[这里](./命令行参数.md#基本参数)。此外还包括以下参数：

- add_version: 在`save`上额外增加目录`'<版本号>-<时间戳>'`防止权重覆盖，默认为True。
- 🔥packing: 是否使用序列packing，默认为False。当前支持`megatron pt/sft`。
- 🔥packing_cache: 指定 packing 缓存目录。默认值为`None`，表示缓存将存储在环境变量 `$MODELSCOPE_CACHE`所指定的路径下。在跨节点使用 packing 功能时，需确保所有节点的 packing 缓存路径共享且一致。你可以通过设置`MODELSCOPE_CACHE`环境变量，或在命令行中添加 `--packing_cache <shared_path>`参数来实现这一要求。
- 🔥streaming: 流式读取并处理数据集，默认False。通常在处理大型数据集时，设置为True。更多流式的参数查看命令行参数文档。
- lazy_tokenize: 默认为False。若该参数设置为False，则在训练之前对所有的数据集样本进行tokenize（这可以避免在训练中出现报错）；设置为True，则在训练中对数据集进行tokenize（这可以节约内存）。
- max_epochs: 训练到`max_epochs`时强制退出训练，并对权重进行验证和保存。该参数在使用流式数据集时很有用。默认为None。
  - 注意：如果你使用非流式数据集，该参数会为你自动计算train_iters，你不需要手动传入`train_iters`。


### RLHF参数
除了继承训练参数外，还支持以下参数：
- rlhf_type: 默认为'dpo'。目前可选择为'dpo'。
- loss_scale: 覆盖[基本参数](./命令行参数.md)中的loss_scale。默认为'last_round'。
- calculate_per_token_loss: 覆盖Megatron参数，默认为False。
