# 命令行参数

## Megatron参数

**训练参数**:
- 🔥micro_batch_size: 每个device的批次大小，默认为1。
- 🔥global_batch_size: 总批次大小，等价于`micro_batch_size*数据并行大小*梯度累加步数`。默认为16。
  - 其中，`数据并行大小 (DP) = 总GPU数 / (TP × PP × CP)`。
- 🔥recompute_granularity: 重新计算激活的粒度，可选项为'full', 'selective' and 'none'（其中'none'为 ms-swift>=3.12.3支持）。其中full代表重新计算整个transformer layer，selective代表只计算transformer layer中的核心注意力部分。通常'selective'是推荐的。默认为'selective'。
  - 当你设置为'selective'时，你可以通过指定`--recompute_modules`来选择对哪些部分进行重新计算。
- 🔥recompute_method: 该参数需将recompute_granularity设置为'full'才生效，可选项为'uniform', 'block'。默认为None。
- 🔥recompute_num_layers: 该参数需将recompute_granularity设置为'full'才生效，默认为None。若`recompute_method`设置为uniform，该参数含义为每个均匀划分的重新计算单元的transformer layers数量。例如你可以指定为`--recompute_granularity full --recompute_method uniform --recompute_num_layers 4`。recompute_num_layers越大，显存占用越小，计算成本越大。注意：当前进程中的模型层数需能被`recompute_num_layers`整除。默认为None。
- 🔥recompute_modules: 选项包括"core_attn", "moe_act", "layernorm", "mla_up_proj", "mlp", "moe"，默认值为`["core_attn"]`。该参数在`--recompute_granularity selective`时生效。例如在MoE训练时，你可以通过指定`--recompute_granularity selective --recompute_modules core_attn moe`降低显存。其中"core_attn"、"mlp" 和 "moe" 使用常规检查点，"moe_act"、"layernorm" 和 "mla_up_proj" 使用输出丢弃检查点。
  - "core_attn"：重新计算 Transformer 层中的核心注意力部分。
  - "mlp"：重新计算密集的 MLP 层。
  - "moe"：重新计算 MoE 层。
  - "moe_act"：重新计算 MoE 中的 MLP 激活函数部分。
  - "layernorm"：重新计算 input_layernorm 和 pre_mlp_layernorm。
  - "mla_up_proj"：重新计算 MLA 上投影和 RoPE 应用部分。
- deterministic_mode: 确定性模式，这会导致训练速度下降，默认为False。
- 🔥train_iters: 训练的总迭代次数，默认为None。
  - 提示：你可以通过设置`--max_epochs`来设置训练的epochs数。在使用非流式数据集时，会自动根据数据集数量计算`train_iters`（兼容packing）。
- 🔥max_epochs: 指定训练的epochs数。当使用非流式数据集时，该参数会为你自动计算train_iters而不需要手动传入`train_iters`。当使用流式数据集时，该参数会在训练到`max_epochs`时强制退出训练，并对权重进行验证和保存。默认为None。
- 🔥log_interval: log的时间间隔（单位：iters），默认为5。
- tensorboard_dir: tensorboard日志写入的目录。默认None，即存储在`f'{save}/runs'`目录下。
- no_masked_softmax_fusion: 默认为False。用于禁用query_key_value的scaling, masking, and softmax融合。
- no_bias_dropout_fusion: 默认为False。用于禁用bias和dropout的融合。
- no_bias_swiglu_fusion: 默认为False。指定`--no_bias_dropout_fusion true`，用于禁止bias和swiglu融合。
- no_rope_fusion: 默认为False。指定`--no_rope_fusion true`用于禁止rope融合。
  - **当使用mrope等不支持rope_fusion的位置编码时，该参数会自动设置为True**。
- no_gradient_accumulation_fusion: 默认为False。指定`--no_gradient_accumulation_fusion true`用于禁用梯度累加融合。
- 🔥cross_entropy_loss_fusion: 启动交叉熵损失计算融合。默认为False。
- cross_entropy_fusion_impl: 交叉熵损失融合的实现。可选为'native'和'te'。默认为'native'。
- calculate_per_token_loss: 根据全局批次中的非填充token数量来对交叉熵损失进行缩放。默认为True。
  - 注意：该参数在rlhf训练或者`task_type`不等于'causal_lm'时，默认为False。
- 🔥attention_backend: 使用的注意力后端 (flash、fused、unfused、local、auto)。默认为 flash。
  - **注意：推荐flash_attn版本：2.8.3**。在"ms-swift<3.7"的版本中，该参数的默认为'auto'。
  - 如果安装'flash_attention_3'，`--attention_backend flash`则优先使用fa3。训练脚本参考[这里](https://github.com/modelscope/ms-swift/tree/main/examples/train/flash_attention_3)。多模态模型的vit部分要使用flash_attention_3，请设置`--attn_impl flash_attention_3`。
  - 有些模型可能不支持flash，你需要手动设置`--attention_backend unfused/fused --padding_free false`，例如：Llama4, GPT-OSS。
- optimizer: 优化器类型，可选为'adam'、'sgd'。默认为adam。
  - 注意：此'adam'为'adamw'，参考[这里](https://github.com/NVIDIA/TransformerEngine/blob/d8f1e68f7c414f3e7985a8b41de4443b2f819af3/transformer_engine/pytorch/optimizers/fused_adam.py#L69-L70)。
- 🔥optimizer_cpu_offload: 将优化器状态卸载到 CPU，例如设置：`--use_precision_aware_optimizer true --optimizer_cpu_offload true --optimizer_offload_fraction 0.7`。默认为False。
  - 该参数可以显著降低显存占用（但增加内存占用）。若global_batch_size较大，则对训练速度的影响不大。
- 🔥optimizer_offload_fraction: 卸载到 CPU 的优化器状态所占比例。默认为1.。
- use_precision_aware_optimizer: 使用 TransformerEngine 中的精度感知优化器，该优化器允许将主参数和优化器状态设置为较低精度，例如 fp16 和 fp8。
- main_grads_dtype: 启用 use_precision_aware_optimizer 时主梯度的 dtype。可选为'fp32', 'bf16'。默认为'fp32'。
- main_params_dtype: 启用 use_precision_aware_optimizer 时主参数的 dtype。可选为'fp32', 'fp16'。默认为'fp32'。
- exp_avg_dtype: 启用 use_precision_aware_optimizer 时，adam 优化器中 exp_avg（即一阶矩）的 dtype。该 dtype 用于在训练过程中将优化器状态存储在内存中，但不会影响内核计算时的精度。可选为'fp32', 'fp16', 'bf16', 'fp8'。默认为'fp32'。
- exp_avg_sq_dtype: 启用 use_precision_aware_optimizer 时，adam 优化器中 exp_avg_sq（即二阶矩）的 dtype。该 dtype 用于在训练过程中将优化器状态存储在内存中，但不会影响内核计算的精度。可选为'fp32', 'fp16', 'bf16', 'fp8'。默认为'fp32'。
- dataloader_type: 默认为'cyclic'，可选为'single', 'cyclic', 'external'。若开启`--streaming`，则设置为`external`。
- manual_gc: 禁用默认垃圾回收器，手动触发垃圾回收。默认为False。
- manual_gc_interval: 手动触发垃圾回收的间隔。默认为0。
- seed: python、numpy、pytorch和cuda的随机种子，默认为42。
- 🔥num_workers: dataloader的workers数量，默认为4。
  - 注意：若设置`--streaming true`，则设置为1。
- no_data_sharding: 当`--train_dataloader_shuffle true`时对 train_dataloader 生效，默认为False。该参数控制数据集随机的范围。若设置为False，则先对数据集进行分片，然后对每个分片进行随机处理（略节约内存）；若设置为True，则先对数据集进行随机，再进行分片（更好的随机效果）。使用该参数需"ms-swift>=3.12"。
- seq_length: 默认为None，即设置为`max_length`。对数据集长度进行限制建议使用“基本参数”中的`--max_length`控制，无需设置此参数。
- use_cpu_initialization: 在cpu上初始化权重，默认为False。在进行HF和MCore权重转换时会被使用。通常不需要修改该值。
- 🔥megatron_extra_kwargs: 额外需要透传入megatron的其他参数，使用json传递。默认为None。
  - 在"ms-swift<3.10"，该参数名为`--extra_megatron_kwargs`。

**学习率参数**:
- 🔥lr: 初始学习率，最终会根据学习率预热策略和衰减策略决定每个迭代的学习率。默认为None，**全参数训练默认为1e-5，LoRA训练默认为1e-4**。
- lr_decay_style: 学习率衰减策略，默认为'cosine'。通常设置为'cosine', 'linear', 'constant'。
- 🔥lr_decay_iters: 学习率衰减的迭代次数。默认为None，则设置为`--train_iters`。
- lr_warmup_iters: 线性学习率预热的迭代次数，默认为0。
- 🔥lr_warmup_fraction: 线性学习率预热阶段所占比例，默认为None。
- 🔥min_lr: 学习率的最小值，将低于该阈值的学习率裁剪为该值，默认为0。

**正则化参数**:
- 🔥weight_decay: 默认为0.1。
- 🔥clip_grad: l2梯度裁剪，默认为1.0。
  - 日志中打印的grad_norm为未裁剪前的值。
- adam_beta1: 默认0.9。
- adam_beta2: 默认0.95。
- adam_eps: 默认1e-8。
- sgd_momentum: 设置`--optimizer sgd`时生效，默认为0.9。

**checkpoint参数**:
- 🔥save: checkpoint的输出目录，默认None。在训练中，若未设置该参数，则默认为`f'megatron_output/{model_suffix}'`，例如`'megatron_output/Qwen2.5-7B-Instruct'`。
  - 注意：**若在多机训练时，请确保每个节点的保存路径指向相同位置**，否则你需要在训练后手动集中这些权重。
- 🔥save_interval: checkpoint保存的间隔（steps），默认为500。
  - 注意：训练结束时一定会保存权重。
- 🔥save_retain_interval: 保留检查点的迭代间隔。只有迭代步数是该间隔倍数的检查点才会被保留（最后一个检查点除外）。
  - 提示：你可以设置为一个很大的值来只保存最后一个检查点。
- 🔥no_save_optim: 不保存optimizer，默认为False。在全参数训练时，可以显著降低存储时间。
- 🔥no_save_rng: 不保存rng，默认为False。
- 🔥load: 加载的checkpoint目录，默认None。对于断点续训的介绍，请查看`--finetune`参数的介绍。
  - 注意：若未使用ms-swift提供的`swift export`进行权重转换，你需要额外设置`--model <hf-repo>`用于加载`config.json`配置文件。
  - 注意：在"ms-swift>3.10"，支持直接加载和存储safetensors权重，参考[mcore-bridge文档](./Mcore-Bridge.md)。
  - `--model`与`--load`的区别：`--model/--adapters/--ref_model/--ref_adapters`后加safetensors权重目录，`--load/--adapter_load/--ref_load/--ref_adapter_load`后加mcore权重目录。`--model/--adapters`不支持加载断点续训状态，因此在"ms-swift>=3.12"，若设置`--no_save_optim false`，将额外存储mcore权重格式用于断点续训，你需要使用`--load/--adapter_load`来加载断点续训的状态。
- 🔥no_load_optim: 不载入optimizer，默认为False。
  - 注意：断点续训时，设置`--no_load_optim false`读取优化器状态通常比`--no_load_optim true`不读取优化器状态消耗更大的显存资源。
- 🔥no_load_rng: 不载入rng，默认为False。
- 🔥finetune: 将模型加载并微调。**不加载检查点的优化器和随机种子状态，并将迭代数设置为0**。默认为False。
  - 注意：**断点续训**你需要设置`--load`（lora训练需要额外设置`--adapter_load`），若设置`--finetune true`，将不加载优化器状态和随机种子状态并将迭代数设置为0，不会进行数据集跳过；若设置`--finetune false`，将读取迭代数并跳过之前训练的数据集数量，优化器状态和随机种子状态的读取通过`--no_load_optim`和`--no_load_rng`控制。
  - 流式数据集`--streaming`，暂不支持跳过数据集。
- ckpt_format: checkpoint的格式。可选为'torch', 'torch_dist', 'zarr'。默认为'torch_dist'。（暂时权重转换只支持'torch_dist'格式）
- no_initialization: 不对权重进行初始化，默认为True。
- auto_detect_ckpt_format: 自动检测ckpt format为legacy还是distributed格式。默认为True。
- exit_on_missing_checkpoint: 如果设置了`–-load`，但**找不到检查点，则直接退出**，而不是初始化。默认为True。
- 🔥async_save: 使用异步检查点保存。目前仅适用于`torch_dist`分布式检查点格式。默认为False。
- use_persistent_ckpt_worker: 使用持久化检查点工作进程用于异步保存，即创建专门后台进程来处理异步保存。默认为False。
- ckpt_fully_parallel_load: 跨 DP 对分布式检查点使用完全加载并行化，加速权重加载速度。默认为False。
- ckpt_assume_constant_structure: 如果在单个训练中，模型和优化器状态字典结构保持不变，允许Megatron进行额外检查点性能优化。默认为False。

**分布式参数**:
并行技术的选择请参考[训练技巧文档](Quick-start.md#训练技巧)。

- distributed_backend: 分布式后端，可选为'nccl', 'gloo'。默认为nccl。
- 🔥use_distributed_optimizer: 使用分布式优化器（即zero1）。默认为True。
- 🔥tensor_model_parallel_size: tp数，默认为1。
- 🔥pipeline_model_parallel_size: pp数，默认为1。
- 🔥decoder_first_pipeline_num_layers: decoder第一个流水线阶段所包含的Transformer层数。默认为 None，表示将Transformer层数平均分配到所有流水线阶段。
  - 该参数通常用于**Transformer层数无法被PP整除**，或者多模态模型第0个pp阶段显存占用过高的情况。
- 🔥decoder_last_pipeline_num_layers: decoder最后一个流水线阶段所包含的Transformer层数。默认为 None，表示将Transformer层数平均分配到所有流水线阶段。
- account_for_embedding_in_pipeline_split: 如果设置为 True，在流水线并行的划分和放置策略中，输入 embedding 层会被视为一个标准的 Transformer 层来处理。默认为False。
- account_for_loss_in_pipeline_split: 如果设置为 True，在流水线并行的划分和放置策略中，loss 层会被视为一个标准的 Transformer 层来处理。默认为False。
- 🔥sequence_parallel: 启动序列并行优化，该参数需要设置`tensor_model_parallel_size`才生效。默认为False。
- 🔥context_parallel_size: cp数，默认为1。
- tp_comm_overlap: 启用张量并行通信与GEMM（通用矩阵乘法）内核的重叠（降低通信耗时）。默认为False。
- 🔥overlap_grad_reduce: 启用DDP中grad reduce操作的重叠（降低DP通信耗时）。默认为False。
- 🔥overlap_param_gather: 启用分布式优化器中参数all-gather的重叠（降低DP通信耗时）。默认为False。
- distributed_timeout_minutes: torch.distributed的timeout时间（单位为分钟），该参数失效，使用[基础参数](../Instruction/Command-line-parameters.md#基本参数)中的ddp_timeout控制，默认为300000分钟。
- num_layers_per_virtual_pipeline_stage: 每个虚拟流水线阶段的层数。默认为None。该参数和`--num_virtual_stages_per_pipeline_rank`参数都可以用来设置vpp并行。
- num_virtual_stages_per_pipeline_rank: 每个流水线并行 rank 的虚拟流水线阶段数量。默认为None。vpp并行，用于减少pp并行的计算空泡，提高GPU利用率，但会略微提高通信量。
- microbatch_group_size_per_virtual_pipeline_stage: 每个虚拟流水线阶段处理的连续微批次数量。默认为None，等于pipeline_model_parallel_size。
- 🔥pipeline_model_parallel_layout: 一个描述自定义流水线（pp/vpp）模型并行布局的字符串。例如："E|(t|)*3,m|m||L"。其中 E、L、t、m 分别表示嵌入层（embedding）、损失层（loss）、Transformer 解码器层和 MTP 层。阶段之间用 "|" 分隔。重复的阶段或层可以通过乘法表示。逗号仅用于提升可读性（无实际语法作用）。默认值为 None，表示不使用此参数设置布局。
  - 该参数通常在异构GPU集群上使用。

**日志参数**:
- log_params_norm: 记录参数的norm。默认为False。
- log_throughput: 记录每个GPU的吞吐量（理论值）。默认为False。
  - 注意：在非packing情况下，log_throughput并不准确，因为`seq_length`并不等于真实序列长度。
- tensorboard_log_interval: 记录到tensorboard的间隔（steps），默认为1。
- tensorboard_queue_size: 用于暂存事件和摘要的 TensorBoard 队列大小；当队列中待处理的事件和摘要数量达到该大小时，下一次调用 "add" 相关方法会触发将数据刷新写入磁盘。默认为50。
- log_timers_to_tensorboard: 记录timers到tensorboard。默认为True。
- no_log_learning_rate_to_tensorboard: 不记录学习率到tensorboard。默认为False。
- log_validation_ppl_to_tensorboard: 将验证困惑度写入tensorboard。默认为True。
- log_memory_to_tensorboard: 将内存日志写入tensorboard。默认为True。
- logging_level: 日志级别。默认为None。
- report_to: (ms-swift>=3.12) 启用的日志后端。默认为None。可选项为'wandb'和'swanlab'。（tensorboard会一直启动）。登陆可以使用`WANDB_API_KEY`、`SWANLAB_API_KEY`环境变量。
- wandb_project: wandb/swanlab 项目名称，取决于`report_to`。默认为'megatron-swift'。
- wandb_exp_name: wandb/swanlab 实验名称。默认为`--save`的值。
- wandb_save_dir: 本地保存 wandb/swanlab 结果的路径。默认为None，即存储在`f'{args.save}/wandb'`或`f'{args.save}/swanlab'`。

**评估参数**:
- 🔥eval_iters: 评估的迭代次数，默认为`-1`，根据验证数据集的数量设置合适的值。**若验证集数量少于global_batch_size，则不进行评估**。若使用流式数据集，该值需要手动设置。
- 🔥eval_interval: 评估的间隔（steps），即每训练多少steps进行评估，默认为None，即设置为save_interval。

**fp8参数**:
- fp8_format: 用于前向和反向传播中FP8张量的FP8格式方案。可选为'e4m3'，'hybrid'。默认为None。
- fp8_recipe: 用于前向和反向传播中 FP8 张量的 FP8 算法方案。可选为'tensorwise', 'delayed', 'mxfp8', 'blockwise'。默认为'delayed'。其中blockwise fp8需要 cuda129 以上版本。
- fp8_amax_history_len: 每个张量记录 amax 历史的步数。默认为1024。
- fp8_amax_compute_algo: 用于根据历史记录计算 amax 的算法。可选为'most_recent', 'max'。默认为'max'。
- fp8_param_gather: 保持计算参数为 fp8（不使用任何其他中间数据类型），并在 fp8 格式下执行参数的 all-gather 操作。默认为False。
  - 提示：若想导出FP8权重格式，设置为True；否则设置为False。

**混合精度参数**:
- fp16: fp16模式。默认为None，会根据模型的torch_dtype进行设置，即torch_dtype为float16或者float32则fp16设置为True。torch_dtype默认读取config.json。
- bf16: bf16模式。默认为None，会根据模型的torch_dtype进行设置，即torch_dtype为bfloat16则bf16设置为True。
- apply_query_key_layer_scaling: 将`Q * K^T` 缩放为 `1 / 层数`（例如：第layer_num层则除以layer_num）。这对fp16训练很有帮助。默认为None，即若使用`--fp16`，则设置为True。
- 🔥attention_softmax_in_fp32: 在attention_mask和softmax中使用fp32进行计算。默认为True。

**模型参数**: （**以下参数通常不需要进行设置，会根据HF模型的config.json进行配置**，用户无需关心）
- num_layers: transformer layers的层数，默认为None。
- hidden_size: transformer hidden size，默认为None。
- ffn_hidden_size: transformer FFN层的hidden size。默认为None，设置为`4*hidden_size`。
- num_attention_heads: transformer attention heads的个数，默认为None。
- group_query_attention: 默认为None。若`num_query_groups>1`，group_query_attention设置为True，否则为False。
- num_query_groups: 默认为1。
- softmax_type: 用于注意力机制的 softmax 类型。支持固定偏移和可学习偏移两种方式。可选项为'vanilla'、'off-by-one'和'learnable'，默认为'vanilla'。
- window_size: 窗口注意力（window attention）的窗口大小，例如`'128,0'`。如果未提供，则禁用窗口注意力。默认为None。
- window_attn_skip_freq: 跳过窗口注意力层的频率。默认为None。
- max_position_embeddings: 位置编码的最大长度，默认为None。
- position_embedding_type: 位置编码的类型，可选为'learned_absolute'、'rope'、'mrope'、'relative'和'none'，默认为'rope'。
- rotary_base: 默认为10000。
- rotary_percent: 默认为1.。
- normalization: 可选为'LayerNorm', 'RMSNorm'，默认为RMSNorm。
- norm_epsilon: 默认为1e-5。
- swiglu: 使用swiglu替代默认的gelu。默认为True。
- quick_geglu: 使用快速 geglu 激活函数而不是默认的 gelu。默认为False。
- activation_func_clamp_value: 限制激活函数中 linear_fc1 的输出值范围。仅在 `activation_func` 为 `quick_gelu` 时使用。默认为None。
- glu_linear_offset: GLU 激活函数中的偏移项：`activation_func(x[0]) * (x[1] + offset)`。仅在 gated_linear_unit 为 True 时使用。默认为0.。
- untie_embeddings_and_output_weights: 解开embedding和输出权重的绑定，默认为True。
- disable_bias_linear: 禁用linear层的bias。默认为True。
- add_qkv_bias: 仅在QKV的linear中增加bias，默认为True。
- attention_dropout: 默认为0.。
- hidden_dropout: 默认为0.。
- kv_channels: 默认为None，设置为`args.hidden_size // args.num_attention_heads`。
- qk_layernorm: 是否对Q和K进行层归一化。
- qk_l2_norm: 使用 Llama 4 的 QK L2 范数。
- no_rope_freq: 控制在哪些层上跳过应用旋转位置编码（RoPE）。默认该参数为None，表示在每一层都执行 RoPE。
- moe_apply_probs_on_input: 在 MoE 路由中，在 MLP 激活函数之前应用概率（probs）。
- transformer_impl: 使用哪种transformer实现，可选项为'local'和'transformer_engine'。默认为transformer_engine。
- padded_vocab_size: 完整词表大小，默认为None。
- rope_scaling: rope_scaling相关参数，默认为None。格式参考[llama3.1 config.json](https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B-Instruct/file/view/master?fileName=config.json&status=1)，传入json字符串。
  - **目前rope_scaling模块使用transformers实现，支持transformers支持的所有rope_scaling。**


**MoE参数**:
- num_experts: MoE的专家数，默认为None。自动从config.json读取。
- moe_layer_freq: MoE 层与 Dense 层之间的分布频率。默认为None。从config.json中读取。
- moe_ffn_hidden_size: 每个专家的前馈网络（ffn）的隐藏层大小。默认为None，自动从config.json读取。若未读取到且`num_experts`不为None，则设置为ffn_hidden_size。
- moe_shared_expert_intermediate_size: 共享专家的总FFN隐藏层大小。如果有多个共享专家，它应等于 `num_shared_experts * ffn_size_of_each_shared_expert`。 默认为None。自动从config.json读取。
- moe_router_topk: 每个token路由到的专家数量。默认为None。自动从config.json读取。
- moe_router_num_groups: 将专家分成的组数，用于组限制路由。参考DeepSeek-V2和DeepSeek-V3。默认为None。自动从config.json读取。
- moe_router_group_topk: 组限制路由中选择的组数。默认为None。自动从config.json读取。
- moe_router_pre_softmax: 为MoE启用预softmax路由，这意味着softmax会在top-k选择之前进行。默认为None。自动从config.json读取。
- 🔥moe_router_dtype: 用于路由计算和专家输出加权平均的数据类型。可选为'none', 'fp32'、'fp64'，这增强了数值稳定性，尤其是在专家数量较多时。与`moe_permute_fusion`一起使用时，性能影响可以忽略不计。默认为'fp32'。'none'代表不改变数据类型。
- moe_router_score_function: MoE TopK 路由的评分函数。可以为 "softmax" 或 "sigmoid"。默认为None，从config.json中读取。
- moe_router_bias_update_rate: 在无辅助损失负载均衡策略中，专家偏置的更新速率。专家偏置根据每个专家在全局批次中被分配的 token 数量进行更新，对于分配到的 token 较少的专家，偏置会增加；对于分配到的 token 较多的专家，偏置会减少。默认为None，从config.json中读取。
- moe_router_enable_expert_bias: 在无辅助损失负载均衡策略中，带有动态专家偏置的 TopK 路由。路由决策基于路由分数与专家偏置之和。详情请参见：https://arxiv.org/abs/2408.15664。默认为None，自动从config.json读取。
- moe_router_topk_scaling_factor: 默认为None。从config.json中读取。
- moe_router_load_balancing_type: 确定路由器的负载均衡策略。可选项为"aux_loss"、"seq_aux_loss"、"global_aux_loss"、"sinkhorn"、"none"。其中, "global_aux_loss"需要"megatron-core>=0.15"。默认值为 None。从config.json中读取。
- 🔥expert_model_parallel_size: 专家并行数，默认为1。
- 🔥expert_tensor_parallel_size: 专家TP并行度。默认值为1。
  - 在"ms-swift<3.9"，其默认值为None，即等于`--tensor_model_parallel_size` 的数值，该默认值将在"ms-swift>=3.9"被修改。
- moe_token_dispatcher_type: 要使用的token分发器类型。可选选项包括 'allgather'、'alltoall'、'flex'和'alltoall_seq'。默认值为'alltoall'。
- moe_enable_deepep: 启用 DeepEP 以实现 MoE 模型中的高效 token 调度和合并。仅在通过设置 `--moe_token_dispatcher_type flex` 使用弹性 token 调度器时有效。
- 🔥moe_grouped_gemm: 当每个rank包含多个专家时，通过在多个流中启动多个本地 GEMM 内核，利用 TransformerEngine中的GroupedLinear提高利用率和性能。默认为True。
  - 在"ms-swift>=3.10"，该参数默认值从False修改为True。
- 🔥moe_permute_fusion: 在令牌分发过程中融合令牌重排操作。默认为False。
- 🔥moe_aux_loss_coeff: 默认为0，不使用aux_loss。**通常情况下，该值设置的越大，训练效果越差，但MoE负载越均衡**，请根据实验效果，选择合适的值。
  - 注意：在"ms-swift<3.7.1"，其默认为None，自动从config.json读取。
- moe_z_loss_coeff: z-loss 的缩放系数。默认为None。
- 🔥moe_shared_expert_overlap: 启用共享专家计算与调度器通信之间的重叠。如果不启用此选项，共享专家将在路由专家之后执行。仅在设置了`moe_shared_expert_intermediate_size`时有效。默认为False。
- 🔥moe_expert_capacity_factor: 每个专家的容量因子，None表示不会丢弃任何token。默认为None。通过设置 `--moe_expert_capacity_factor`，超出专家容量的 token 会基于其被选中的概率被丢弃。可以**令训练负载均匀，提升训练速度**（例如设置为1或2）。
- moe_pad_expert_input_to_capacity: 对每个专家（expert）的输入进行填充，使其长度与专家容量（expert capacity length）对齐，默认为False。该操作仅在设置了 `--moe_expert_capacity_factor` 参数后才生效。
- moe_token_drop_policy: 可选为'probs', 'position'。默认为'probs'。

**mla参数**
- multi_latent_attention: 是否使用MLA。默认为False。
- q_lora_rank: Query 张量低秩表示的rank值。默认为None，自动从config.json读取。
- kv_lora_rank: Key 和 Value 张量低秩表示的秩（rank）值。默认为None，自动从config.json读取。
- qk_head_dim: QK 投影中 head 的维度。 `q_head_dim = qk_head_dim + qk_pos_emb_head_dim`。默认为None，自动从config.json读取。
- qk_pos_emb_head_dim: QK 投影中位置嵌入的维度。默认为None，自动从config.json读取。
- v_head_dim: V 投影中的 head 维度。默认为None，自动从config.json读取。

**MTP参数**
- mtp_num_layers: 多token预测（MTP）层的数量。MTP将每个位置的预测范围扩展到多个未来token。此MTP实现使用D个顺序模块依次预测D个额外的token。默认为None。（需要"megatron-core>=0.14"）
  - 注意：mtp_num_layers的值，将不自动从config.json获取，需手动设置。你可以参考config.json中的`num_nextn_predict_layers`字段填写该值。使用mcore-bridge时，将优先从safetensors文件中加载MTP权重，若无法找到，则进行随机初始化。（若要使用blockwise fp8 + mtp，请使用mcore>=0.15）
- mtp_loss_scaling_factor: 多token预测（MTP）损失的缩放因子。我们计算所有深度上MTP损失的平均值，然后乘以该缩放因子得到总体MTP损失，它将作为一个额外的训练目标。默认为0.1。

**Tuner参数**:
- tuner_type: 可选为'lora'和'full'。默认为'full'。（**在ms-swift3.x中参数名为`train_type`**）
- 🔥freeze_llm: 该参数只对多模态模型生效，可用于全参数训练和LoRA训练，但会产生不同的效果。若是全参数训练，将freeze_llm设置为True会将LLM部分权重进行冻结；若是LoRA训练且`target_modules`设置为'all-linear'，将freeze_llm设置为True将会取消在LLM部分添加LoRA模块。该参数默认为False。
- 🔥freeze_vit: 该参数只对多模态模型生效，可用于全参数训练和LoRA训练，但会产生不同的效果。若是全参数训练，将freeze_vit设置为True会将vit部分权重进行冻结；若是LoRA训练且`target_modules`设置为'all-linear'，将freeze_vit设置为True将会取消在vit部分添加LoRA模块。该参数默认为True。
  - 注意：**这里的vit不仅限于vision_tower, 也包括audio_tower**。若是Omni模型，若你只希望对vision_tower加LoRA，而不希望对audio_tower加LoRA，你可以修改[这里的代码](https://github.com/modelscope/ms-swift/blob/a5d4c0a2ce0658cef8332d6c0fa619a52afa26ff/swift/llm/model/model_arch.py#L544-L554)。
- 🔥freeze_aligner: 该参数只对多模态模型生效，可用于全参数训练和LoRA训练，但会产生不同的效果。若是全参数训练，将freeze_aligner设置为True会将aligner（也称为projector）部分权重进行冻结；若是LoRA训练且`target_modules`设置为'all-linear'，将freeze_aligner设置为True将会取消在aligner部分添加LoRA模块。该参数默认为True。

全参数训练：
- freeze_parameters: 需要被冻结参数的前缀，默认为`[]`。
- freeze_parameters_regex: 需要被冻结参数的正则表达式，默认为None。
- freeze_parameters_ratio: 从下往上冻结的参数比例，默认为0。可设置为1将所有参数冻结，结合`trainable_parameters`设置可训练参数。除了设置为0/1，该参数不兼容pp并行。
- trainable_parameters: 额外可训练参数的前缀，默认为`[]`。
- trainable_parameters_regex: 匹配额外可训练参数的正则表达式，默认为None。

lora训练：
- adapter_load: 加载adapter的权重路径，用于lora断点续训，默认为None。lora断点续训方式与全参数一致，请关注`--finetune`参数的含义。
- 🔥target_modules: 指定lora模块的后缀，例如：你可以设置为`--target_modules linear_qkv linear_proj`。默认为`['all-linear']`，代表将所有的linear设置为target_modules。
  - 注意：在LLM和多模态LLM中，'all-linear'的行为有所不同。若是LLM则自动寻找除lm_head外的linear并附加tuner；**若是多模态LLM，则默认只在LLM上附加tuner，该行为可以被`freeze_llm`、`freeze_vit`、`freeze_aligner`控制**。
  - 注意：若需要将所有的router设置为target_modules, 你可以额外设置`--target_modules all-router ...`，例如：`--target_modules all-router all-linear`。
  - transformers和Megatron的Linear层后缀名称不同，在Megatron中，`linear_proj`代表`o_proj`，`linear_qkv`代表`q_proj, k_proj, v_proj`的拼接，`linear_fc1`代表`gate_proj`, `up_proj`的拼接，`linear_fc2`代表`down_proj`。
- 🔥target_regex: 指定lora模块的regex表达式，默认为`None`。如果该值传入，则target_modules参数失效。
- 🔥modules_to_save: 在已附加tuner后，额外指定一部分原模型模块参与训练和存储。默认为`[]`。例如设置为`--modules_to_save word_embeddings output_layer`，在LoRA训练中解开`word_embeddings`和`output_layer`层进行训练，这两部分的权重信息最终会进行保存。
- 🔥lora_rank: 默认为`8`。
- 🔥lora_alpha: 默认为`32`。
- lora_dropout: 默认为`0.05`。
- lora_bias: 默认为`'none'`，可以选择的值: 'none'、'all'。如果你要将bias全都设置为可训练，你可以设置为`'all'`。
- use_rslora: 默认为`False`，是否使用`RS-LoRA`。

**Mcore-Bridge参数**
- 🔥load_safetensors: 该参数在"ms-swift>=3.12"将失效（之前版本默认为False），将根据优先级加载权重：若`--load`不存在，则加载safetensors权重`--model`；`--adapters`和`--adapter_load`等同理。
  - 注意：在"ms-swift>=3.12"，为保持shell脚本兼容性，该参数被保留，但不再发挥任何作用。
- 🔥save_safetensors: 默认为True，是否直接保存成safetensors权重。该参数在"ms-swift>=3.12"支持了对优化器权重、随机数状态等断点续训内容进行保存（额外存储mcore格式权重），使用`--no_save_optim`和`--no_save_rng`控制。断点续训时使用`--load/--adapter_load`参数加载mcore格式权重。
- model: safetensors权重的model_id或者model_path。默认为None。
- model_type: 模型类型。介绍参考[ms-swift命令行参数文档](../Instruction/Command-line-parameters.md)。
- adapters: safetensors格式的LoRA增量权重的adapter_id或者adapter_path。默认为`[]`。
- ref_model: ref_model safetensors权重的model_id或者model_path。采用grpo、dpo、kto算法且使用全参数训练时需要传入。默认为None，设置为`--model`。
- ref_adapters: ref_adapters safetensors权重的adapter_id或者adapter_path的列表（目前只支持长度为1），默认为`[]`。
- use_hf: 控制模型下载、数据集下载、模型推送使用ModelScope还是HuggingFace。默认为False，使用ModelScope。
- hub_token: hub token. modelscope的hub token可以查看[这里](https://modelscope.cn/my/myaccesstoken)。默认为None。
- merge_lora: 是否存储合并后的权重。默认为None，若`save_safetensors`设置为True，该参数默认值为`True`，否则为False。即默认情况下，存储为safetensors格式时会合并LoRA；存储为torch_dist格式时，不会合并LoRA。
- max_shard_size: safetensors格式存储文件最大大小，默认'5GB'。
- 🔥offload_bridge: Megatron导出的用于vLLM更新HF格式权重使用CPU主存存放，以降低 GPU 显存占用。默认为 False。

## 训练参数

Megatron训练参数继承自Megatron参数和基本参数（**与ms-swift共用dataset、template等参数，也支持ms-swift中的特定模型参数**）。基本参数的内容可以参考[这里](../Instruction/Command-line-parameters.md#基本参数)。此外还包括以下参数：

- add_version: 在`save`上额外增加目录`'<版本号>-<时间戳>'`防止权重覆盖，默认为True。
- check_model: 检查本地模型文件有损坏或修改并给出提示，默认为True。**如果是断网环境，请设置为False**。
- padding_free: 将一个batch中的数据进行展平而避免数据padding，从而降低显存占用并加快训练。默认为True。
  - 若要自定义attention_mask，你可以设置`--padding_free false`。
  - 注意：**Megatron-SWIFT训练特性优先支持padding_free格式**，若非特殊情况，请勿修改该值。
- mlp_padding_free: 默认为False。用于padding_free设置为false时，对mlp进行padding_free优化。这可以在自定义attention_mask的同时，提升训练速度和减少显存占用。
- vit_gradient_checkpointing: 多模态模型训练时，是否对vit部分开启gradient_checkpointing。默认为True。（**Megatron-SWIFT的vit实现使用transformers实现**）
- attn_impl: 多模态模型训练时，设置vit部分的attn_impl实现。默认为'flash_attn'。
- vit_lr: 当训练多模态大模型时，该参数指定vit的学习率，默认为None，等于learning_rate。通常与`--freeze_vit`、`--freeze_aligner`参数结合使用。
  - 提示：在日志中打印的"learning rate"为llm的学习率。
- aligner_lr: 当训练多模态大模型时，该参数指定aligner的学习率，默认为None，等于learning_rate。
- gradient_checkpointing_kwargs: 传入`torch.utils.checkpoint`中的参数。例如设置为`--gradient_checkpointing_kwargs '{"use_reentrant": false}'`。默认为None。该参数只对`vit_gradient_checkpointing`生效。
- 🔥packing: 使用`padding_free`的方式将不同长度的数据样本打包成**近似**统一长度的样本（packing能保证不对完整的序列进行切分），实现训练时各节点与进程的负载均衡（避免长文本拖慢短文本的训练速度），从而提高GPU利用率，保持显存占用稳定。当使用 `--attention_backend flash` 时，可确保packed样本内的不同序列之间相互独立，互不可见（除Qwen3-Next，因为含有linear-attention）。该参数默认为`False`。Megatron-SWIFT的所有训练任务都支持该参数。注意：**packing会导致数据集样本数减少，请自行调节梯度累加数和学习率**。
- packing_length: packing的长度。默认为None，设置为max_length。
- packing_num_proc: packing的进程数，默认为1。需要注意的是，不同的`packing_num_proc`，最终形成的packed数据集是不同的。（该参数在流式packing时不生效）。通常不需要修改该值，packing速度远快于tokenize速度。
- streaming: 流式读取并处理数据集，默认False。（流式数据集的随机并不彻底，可能导致loss波动剧烈。）
  - 注意：因为流式数据集无法获得其长度，因此需要设置`--train_iters`参数。设置`max_epochs`参数确保训练到对应epochs时退出训练，并对权重进行验证和保存。
  - 注意：流式数据集可以跳过预处理等待，将预处理时间与训练时间重叠。流式数据集的预处理只在rank0上进行，并通过数据分发的方式同步到其他进程，**其通常效率不如非流式数据集采用的数据分片读取方式**。当训练的world_size较大时，预处理和数据分发将成为训练瓶颈。
- lazy_tokenize: 是否使用lazy_tokenize。若该参数设置为False，则在训练之前对所有的数据集样本进行tokenize（多模态模型则包括从磁盘中读取图片）。该参数默认为None，在LLM训练中默认为False，而MLLM训练默认为True，节约内存。
- enable_dft_loss: 是否在SFT训练中使用[DFT](https://arxiv.org/abs/2508.05629) (Dynamic Fine-Tuning) loss，默认为False。
- enable_channel_loss: 启用channel loss，默认为`False`。你需要在数据集中准备"channel"字段，ms-swift会根据该字段分组统计loss（若未准备"channel"字段，则归为默认`None` channel）。数据集格式参考[channel loss](../Customization/Custom-dataset.md#channel-loss)。channel loss兼容packing/padding_free/loss_scale等技术。
- new_special_tokens: 需要新增的特殊tokens。默认为`[]`。例子参考[这里](https://github.com/modelscope/ms-swift/blob/main/examples/megatron/lora/new_special_tokens.sh)。
  - 注意：你也可以传入以`.txt`结尾的文件路径，每行为一个special token。
- 🔥task_type: 默认为'causal_lm'。可选为'causal_lm'、'seq_cls'、'embedding'和'generative_reranker'。
- num_labels: 分类模型（即`--task_type seq_cls`）需要指定该参数。代表标签数量，默认为None。
- problem_type: 分类模型（即`--task_type seq_cls`）需要指定该参数。可选为'regression', 'single_label_classification', 'multi_label_classification'。默认为None，若模型为 reward_model 或 num_labels 为1，该参数为'regression'，其他情况，该参数为'single_label_classification'。
- 🔥save_strategy: 保存策略，可选项为'steps'和'epoch'。默认为'steps'。当设置为'epoch'时，'save_interval'和'eval_interval'都会强制设置为1，代表每个epoch存储权重，'save_retain_interval'可设置为整数，代表多少个epoch存储保留检查点。
- dataset_shuffle: 是否对dataset进行随机操作。默认为True。
  - 注意：**Megatron-SWIFT的随机包括两个部分**：数据集的随机，由`dataset_shuffle`控制；train_dataloader中的随机，由`train_dataloader_shuffle`控制。
- train_dataloader_shuffle: 是否对train_dataloader使用随机，默认为True。该参数需"ms-swift>=3.12"。
  - 在"ms-swift>3.12"，将不再对val_dataset进行随机操作。
- dataloader_pin_memory: 默认为True。使用该参数需"ms-swift>=3.12"。
- dataloader_persistent_workers: 默认为True。使用该参数需"ms-swift>=3.12"。
- dataloader_prefetch_factor: 默认为2。使用该参数需"ms-swift>=3.12"。
- 🔥group_by_length: (ms-swift>=3.12) 是否在训练数据集中将长度大致相同的样本分组在一起（有随机因素），以最小化填充并确保各节点与进程的负载均衡以提高效率。默认为False。具体算法参考`transformers.trainer_pt_utils.get_length_grouped_indices`。


## RLHF参数
除了继承训练参数外，还支持以下参数：
- 🔥rlhf_type: 默认为'dpo'。目前可选择为'dpo'、'grpo'、'kto'、'rm'和'gkd'。
- loss_scale: 覆盖[基本参数](../Instruction/Command-line-parameters.md)中的loss_scale。默认为'last_round'。
- calculate_per_token_loss: 覆盖Megatron参数，默认为False。


### DPO参数
- ref_load: ref_model的加载路径。采用DPO/GRPO/KTO算法且使用全参数训练时需要传入。默认为None，即设置为`load`。
- ref_adapter_load: 加载ref_adapter的权重路径，默认为None。若你要使用SFT产生的LoRA权重进行DPO，请使用"ms-swift>=3.8"，并在训练时设置`--adapter_load sft_ckpt --ref_adapter_load sft_ckpt --finetune true`。若是此场景的断点续训，则设置`--adapter_load rlhf_ckpt --ref_adapter_load sft_ckpt --finetune false`。
- beta: 含义与[TRL](https://huggingface.co/docs/trl/main/en/dpo_trainer#trl.DPOConfig)相同。控制与参考模型偏差程度的参数。beta值越高，表示与参考模型的偏差越小。对于 IPO 损失函数 (loss_type="ipo")，beta是[论文](https://huggingface.co/papers/2310.12036)中所指的正则化参数。默认为0.1。
- 🔥rpo_alpha: 来自[RPO 论文](https://huggingface.co/papers/2404.19733)中的参数，用于控制损失函数中NLL项的权重（即SFT损失），`loss = dpo_loss + rpo_alpha * sft_loss`，论文中推荐设置为`1.`。默认为`None`，即默认不引入sft_loss。
  - **注意**：在"ms-swift<3.8"，其默认值为`1.`。在"ms-swift>=3.8"该默认值修改为`None`。
- reference_free: 是否忽略提供的参考模型，并隐式地使用一个对所有响应赋予相等概率的参考模型。默认为False。
- label_smoothing: 默认为0.。
- f_divergence_type: 默认为`reverse_kl`。可选值参考[TRL文档](https://huggingface.co/docs/trl/main/en/dpo_trainer)。
- loss_type: 默认为'sigmoid'。可选值参考[TRL文档](https://huggingface.co/docs/trl/main/en/dpo_trainer#loss-functions)。

### KTO参数
- ref_load: 含义同DPO。
- ref_adapter_load: 含义同DPO。
- beta: 控制与 ref_model 偏离程度的参数。较高的 beta 表示与 ref_model 偏离更小。默认为`0.1`。
- loss_type: 默认为'kto'。可选值参考[TRL文档](https://huggingface.co/docs/trl/main/en/kto_trainer#trl.KTOConfig.loss_type)。
- desirable_weight: 抵消 desirable 和 undesirable 数量不均衡的影响，对 desirable 损失按该系数进行加权，默认为`1.`。
- undesirable_weight: 抵消 desirable 和 undesirable 数量不均衡的影响，对 undesirable 损失按该系数进行加权，默认为`1.`。

### RM参数
- center_rewards_coefficient: 用于激励奖励模型输出均值为零的奖励的系数，具体查看这篇[论文](https://huggingface.co/papers/2312.09244)。推荐值：0.01。

### GRPO参数
- ref_load: 含义同DPO。
- ref_adapter_load: 含义同DPO。
- beta: KL正则系数，默认为0.04，设置为0时不加载ref model。
- micro_batch_size: 每个device的批次大小，默认为1。
- global_batch_size: 总批次大小，等价于`micro_batch_size*数据并行大小*梯度累加步数`。默认为16。
- steps_per_generation：每轮生成的优化步数，即采样批量大小相对global_batch_size的倍数，默认为1。
- generation_batch_size: 采样批量大小，需要是global_batch_size的倍数，默认等于global_batch_size*steps_per_generation。
- num_generations: 每个prompt采样的数量，论文中的G值，默认为8。
- num_generations_eval: 评估阶段每个prompt采样的数量。允许在评估时使用较少的生成数量以节省计算资源。如果为 None，则使用 num_generations 的值。默认为 None。
- reward_funcs: GRPO算法奖励函数，可选项为`accuracy`、`format`、`cosine`、`repetition`和`soft_overlong`，见swift/rewards/orm.py。你也可以在plugin中自定义自己的奖励函数。默认为`[]`。
- reward_weights: 每个奖励函数的权重。必须与奖励函数和奖励模型的总数量匹配。默认为 None，即所有奖励的权重都相等，为`1.0`。
  - 提示：如果GRPO训练中包含`--reward_model`，则其加在奖励函数的最后位置。
- truncation_strategy: 对输入长度超过 `max_length`的处理方式，支持`delete`和`left`，代表删除、左侧裁剪，默认为`left`。注意对于多模态模型，
左裁剪可能会裁剪掉多模态token导致模型前向报错shape mismatch。使用`delete`方式，对于超长数据和编码失败的样例会在原数据集中重采样其他数据作为补充。
- loss_type: loss 归一化的类型，可选项为['grpo', 'bnpo', 'dr_grpo'], 默认为'grpo', 具体查看该[pr](https://github.com/huggingface/trl/pull/3256#discussion_r2033213348)。
- log_completions: 是否记录训练中的模型生成内容，默认为False。
- vllm_mode: vLLM 集成模式，可选项为 `server` 和 `colocate`。server 模式使用 `swift rollout` 拉起的 vLLM 服务器进行采样，colocate 模式在程序内部署 vLLM。使用server端时，
- vllm_mode server 参数
  - vllm_server_host: vLLM server host地址，默认为None。
  - vllm_server_port: vLLM server 服务端口，默认为8000。
  - vllm_server_base_url: vLLM server的Base URL(比如 http://local_host:8000), 默认为None。设置后，忽略host和port设置。
  - vllm_server_timeout: 连接vLLM server的超时时间，默认为 240s。
  - vllm_server_pass_dataset: 透传额外的数据集信息到vLLM server，用于多轮训练。
  - async_generate: 异步rollout以提高训练速度，注意开启时采样会使用上一轮更新的模型进行采样，不支持多轮场景。默认`false`.
  - SWIFT_UPDATE_WEIGHTS_BUCKET_SIZE: 环境变量，用于控制权重同步时的传输桶大小（bucket size），适用于 Server Mode 下的全参数训练，单位为 MB，默认值为 512 MB。
- vllm_mode colocate 参数（更多参数支持参考[vLLM参数](#vLLM参数)。）
  - vllm_gpu_memory_utilization: vllm透传参数，默认为0.9。
  - vllm_max_model_len: vllm透传参数，默认为None。
  - vllm_enforce_eager: vllm透传参数，默认为False。
  - vllm_limit_mm_per_prompt: vllm透传参数，默认为None。
  - vllm_enable_prefix_caching: vllm透传参数，默认为True。
  - vllm_tensor_parallel_size: tp并行数，默认为`1`。
  - vllm_enable_lora: 支持vLLM Engine 加载 LoRA adapter，默认为False。用于加速LoRA训练的权重同步，具体参考[文档](../Instruction/GRPO/GetStarted/GRPO.md#权重同步加速)。
  - sleep_level: 训练时释放 vLLM 显存，可选项为[0, 1, 2], 默认为0，不释放。
  - offload_optimizer: 是否在vLLM推理时offload optimizer参数，默认为False。
  - offload_model: 是否在vLLM推理时 offload 模型，默认为False。
- num_iterations: 每条数据的更新次数，[GRPO论文](https://arxiv.org/abs/2402.03300)中的 $\mu$ 值，默认为1。
- epsilon: clip 系数，默认为0.2。
- epsilon_high: upper clip 系数，默认为None，设置后与epsilon共同构成[epsilon, epsilon_high]裁剪范围。
- dynamic_sample：筛除group内奖励标准差为0的数据，额外采样新数据，默认为False。
- max_resample_times：dynamic_sample设置下限制重采样次数，默认3次。
- overlong_filter：跳过超长截断的样本，不参与loss计算，默认为False。
- delta: [INTELLECT-2 tech report](https://huggingface.co/papers/2505.07291)中双侧 GRPO 上界裁剪值。若设置，建议大于 1 + epsilon。默认为None。
- importance_sampling_level: 控制重要性采样比计算，可选项为 `token` 和 `sequence`，`token` 模式下保留原始的每个 token 的对数概率比，`sequence` 模式下则会对序列中所有有效 token 的对数概率比进行平均。[GSPO论文](https://arxiv.org/abs/2507.18071)中使用sequence级别计算来稳定训练，默认为`token`。
- scale_rewards：指定奖励的缩放策略。可选值包括 `group`（按组内标准差缩放）、`batch`（按整个批次的标准差缩放）、`none`（不进行缩放）、`gdpo`（对每个奖励函数分别进行组内归一化后加权聚合，参考 [GDPO 论文](https://arxiv.org/abs/2601.05242)）。在 ms-swift < 3.10 版本中，该参数为布尔类型，`true` 对应 `group`，`false` 对应 `none`。默认值与 `advantage_estimator` 绑定：`grpo` 对应 `group`，`rloo` 对应 `none`，`reinforce_plus_plus` 对应 `batch`。
  - 注意：`gdpo` 模式不支持 `kl_in_reward=True`，若同时设置会自动将 `kl_in_reward` 设为 `False`。
  - GDPO 适用于多奖励优化场景：当使用多个奖励函数时，GDPO 会对每个奖励函数分别在组内进行标准化（减均值、除标准差），然后使用 `reward_weights` 进行加权求和，最后再进行批次级别的标准化。这种方式可以更好地保留各个奖励的相对差异，避免不同奖励组合坍塌成相同的 advantage 值。
- rollout_importance_sampling_mode: 训推不一致校正模式，可选项为 `token_truncate`、`token_mask`、`sequence_truncate`、`sequence_mask`。默认为None，不启用校正。具体参考[文档](../Instruction/GRPO/AdvancedResearch/training_inference_mismatch.md)。
- rollout_importance_sampling_threshold: 重要性采样权重的阈值，用于截断或屏蔽极端权重。默认为2.0。
- log_rollout_offpolicy_metrics: 当 `rollout_importance_sampling_mode` 未设置时，是否记录训推不一致诊断指标（KL、PPL、χ²等）。当设置了 `rollout_importance_sampling_mode` 时，指标会自动记录。默认为False。
- off_policy_sequence_mask_delta: Off-Policy Sequence Masking 阈值，来自 DeepSeek-V3.2 论文。当设置此值时，会计算每个序列的 `mean(old_policy_logps - policy_logps)`，若该值大于阈值且该序列的优势为负，则 mask 掉该序列不参与损失计算。默认为None，不启用。具体参考[文档](../Instruction/GRPO/AdvancedResearch/training_inference_mismatch.md#off-policy-sequence-masking)。

内置奖励函数参数参考[文档](../Instruction/Command-line-parameters.md#奖励函数参数)

### GKD参数
- teacher_model: 教师模型的路径或模型 ID，必需参数。
- teacher_model_type: 教师模型类型，默认为None，自动检测。
- teacher_model_revision: 教师模型版本，默认为None。
- beta: JSD 散度插值系数。0.0 代表 Forward KL，0.5 代表对称 JSD，1.0 代表 Reverse KL。默认为0.5。
- lmbda: On-Policy 学习触发概率。0.0 代表纯 Off-Policy，1.0 代表纯 On-Policy。默认为0.5。
- seq_kd: 是否使用教师生成的响应（Sequential KD），当前暂不支持。默认为False。
- temperature: 用于采样和损失计算的温度参数。默认为0.9。
- offload_teacher_model: 是否将教师模型卸载到 CPU 以节省 GPU 显存。默认为False。
- sft_alpha: SFT 损失的混合系数，`loss = jsd_loss + sft_alpha * sft_loss`。当使用数据集响应（Off-Policy）时生效。默认为0。
- max_completion_length: 生成时的最大 token 数。默认为512。
- vllm_mode: 同 GRPO 参数，用于 On-Policy 生成。colocate 模式下在程序内部署 vLLM。
  - 注意：On-Policy 生成需要启用 vLLM（`--use_vllm true --vllm_mode colocate/server`）。
  - 当 `lmbda > 0` 但未启用 vLLM 时，将自动回退到 Off-Policy 模式。

## 导出参数
这里介绍`megatron export`的参数（需"ms-swift>=3.10"），若要使用`swift export`导出命令，请参考[ms-swift命令行参数文档](../Instruction/Command-line-parameters.md#导出参数)。`megatron export`相比`swift export`，支持分布式和多机导出。Megatron导出参数继承自Megatron参数和基本参数。
- 🔥to_mcore: HF格式权重转成Megatron格式。默认为False。
- 🔥to_hf: Megatron格式权重转成HF格式。默认为False。
- 🔥merge_lora: 默认为None，若`to_hf`设置为True，该参数默认值为`True`，否则为False。即默认情况下，存储为safetensors格式时会合并LoRA；存储为torch_dist格式时，不会合并LoRA。合并后的权重存储在`--save`目录下。
  - 注意：由于transformers和Megatron模型结构并不一定一致（例如transformers的Qwen3-VL-Moe的专家部分并不是Linear实现，而是Parameters），因此部分模型无法转换（若Qwen3-VL-Moe只设置linear_proj和linear_qkv训练LoRA也支持转换）。但大多数的模型支持LoRA转换，例如：Qwen3-Moe，Qwen3-Omni-Moe，GLM4.5-V等。
- 🔥test_convert_precision: 测试HF和Megatron格式权重转换的精度误差。默认为False。
- test_convert_dtype: 转换精度测试使用的dtype，默认为'float32'。
- exist_ok: 如果`args.save`存在，不抛出异常，进行覆盖。默认为False。
- device_map: 设置`--test_convert_precision true`时生效，控制HF模型的加载位置，默认为'auto'。你可以设置为'cpu'节约显存资源。
