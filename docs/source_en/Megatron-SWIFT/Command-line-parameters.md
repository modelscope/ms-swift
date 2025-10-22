# Command Line Arguments

## Megatron Parameters

**Training Parameters**:

- 🔥micro_batch_size: Batch size per device, default is 1.
- 🔥global_batch_size: Total batch size, equivalent to `micro_batch_size * data parallel size * gradient accumulation steps`. Default is 16.
- 🔥recompute_granularity: Granularity of activation recomputation, options are 'full', 'selective'. 'full' means recomputing the entire transformer layer, while 'selective' means only recomputing the core attention part of the transformer layer. 'selective' is generally recommended. Default is 'selective'.
  - When you set it to 'selective', you can specify `--recompute_modules` to choose which parts to recompute.
- 🔥recompute_method: This parameter takes effect only when recompute_granularity is set to 'full', options are 'uniform', 'block'. Default is None.
- 🔥recompute_num_layers: This parameter takes effect only when recompute_granularity is set to 'full'. Default is None. If `recompute_method` is set to uniform, this parameter specifies the number of transformer layers in each uniformly divided recomputation unit. For example, you can specify `--recompute_granularity full --recompute_method uniform --recompute_num_layers 4`. The larger the recompute_num_layers, the smaller the memory usage but higher computation cost. Note: The number of model layers in the current process must be divisible by `recompute_num_layers`. Default is None.
- 🔥recompute_modules: Options include "core_attn", "moe_act", "layernorm", "mla_up_proj", "mlp", and "moe". The default value is `["core_attn"]`. This parameter takes effect when `--recompute_granularity selective` is set. For example, during MoE training, you can reduce memory usage by specifying `--recompute_granularity selective --recompute_modules core_attn moe`. Among these, "core_attn", "mlp", and "moe" use normal checkpointing, while "moe_act", "layernorm", and "mla_up_proj" use output-discarding checkpointing.
  - "core_attn": Recomputes the core attention part of the Transformer layer.
  - "mlp": Recomputes the dense MLP layer.
  - "moe": Recomputes the MoE layer.
  - "moe_act": Recomputes the MLP activation function part in the MoE module.
  - "layernorm": Recomputes the input_layernorm and pre_mlp_layernorm.
  - "mla_up_proj": Recomputes the MLA up-projection and RoPE application parts.
- deterministic_mode: Deterministic mode, which may lead to slower training speed, default is False.
- 🔥train_iters: Total number of training iterations, default is None.
  - Tip: You can set `--max_epochs` to specify the number of training epochs. When using a non-streaming dataset, `train_iters` will be automatically calculated based on the dataset size (compatible with packing).
- 🔥max_epochs: Specifies the number of training epochs. When using a non-streaming dataset, this parameter automatically calculates `train_iters`, eliminating the need to manually provide `train_iters`. When using a streaming dataset, training will be forcibly terminated upon reaching `max_epochs`, and the model weights will be validated and saved. Default is None.
- 🔥log_interval: Log interval (unit: iters), default is 5.
- tensorboard_dir: Directory where TensorBoard logs are written. Default is None, meaning logs will be stored in the `f'{save}/runs'` directory.
- no_masked_softmax_fusion: Default is False. Disables scaling, masking, and softmax fusion for query_key_value.
- no_bias_dropout_fusion: Default is False. Disables bias and dropout fusion.
- no_bias_swiglu_fusion: Default is False. Specify `--no_bias_dropout_fusion true` to disable bias and swiglu fusion.
- no_rope_fusion: Default is False. Specify `--no_rope_fusion true` to disable rope fusion.
  - **When using position embedding such as mrope that do not support RoPE fusion, this parameter will be automatically set to True**.
- no_gradient_accumulation_fusion: Default is False. Specify `--no_gradient_accumulation_fusion true` to disable gradient accumulation fusion.
- 🔥cross_entropy_loss_fusion: Enables cross-entropy loss calculation fusion. Default is False.
- cross_entropy_fusion_impl: Implementation of cross-entropy loss fusion. Options include 'native' and 'te'. Defaults to 'native'.
- calculate_per_token_loss: Scales the cross-entropy loss according to the number of non-padded tokens in the global batch. Default is True.
  - Note: This parameter defaults to False during RLHF training or when `task_type` is not equal to 'causal_lm'.
- 🔥attention_backend: The attention backend to use (flash, fused, unfused, local, auto). Default is flash.
  - **Note: The recommended `flash_attn` version is 2.7.4.post1/2.8.1**. In versions of `ms-swift` prior to 3.7, the default value for this parameter is `'auto'`.
  - If `flash_attention_3` is installed, specifying `--attention_backend flash` will prioritize using FA3. Refer to the training script [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/flash_attention_3).
- optimizer: Optimizer type, options are 'adam', 'sgd'. Default is adam.
- 🔥optimizer_cpu_offload: Offloads optimizer states to the CPU. For example, set: `--use_precision_aware_optimizer true --optimizer_cpu_offload true --optimizer_offload_fraction 0.7`. Defaults to `False`.
  - This parameter can significantly reduce GPU memory usage (at the cost of increased CPU memory consumption). When the `global_batch_size` is large, its impact on training speed is minimal.
- 🔥optimizer_offload_fraction: The fraction of the optimizer state to offload to CPU. Default is `1.0`.
- use_precision_aware_optimizer: Use the precision-aware optimizer in TransformerEngine, which allows setting the main parameters and optimizer states to lower precision, such as fp16 and fp8.
- main_grads_dtype: The dtype of main gradients when use_precision_aware_optimizer is enabled. Options are 'fp32' and 'bf16'. Default is 'fp32'.
- main_params_dtype: The dtype of main parameters when use_precision_aware_optimizer is enabled. Options are 'fp32' and 'fp16'. Default is 'fp32'.
- exp_avg_dtype: The dtype of exp_avg (i.e., the first moment in the Adam optimizer) when use_precision_aware_optimizer is enabled. This dtype is used for storing the optimizer state in memory during training, but does not affect the precision in kernel computation. Options are 'fp32', 'fp16', 'bf16', and 'fp8'. Default is 'fp32'.
- exp_avg_sq_dtype: The dtype of exp_avg_sq (i.e., the second moment in the Adam optimizer) when use_precision_aware_optimizer is enabled. This dtype is used for storing the optimizer state in memory during training, but does not affect the precision in kernel computation. Options are 'fp32', 'fp16', 'bf16', and 'fp8'. Default is 'fp32'.
- dataloader_type: Default is 'cyclic', options are 'single', 'cyclic', 'external'. If `--streaming` is enabled, set it to external.
- manual_gc: Disables the default garbage collector and manually triggers garbage collection. Default is False.
- manual_gc_interval: The interval for manually triggering garbage collection. Defaults to 0.
- seed: Random seed for python, numpy, pytorch, and cuda, default is 42.
- 🔥num_workers: Number of workers for the dataloader, default is 4.
  - Note: If `--streaming true` is set, it will be set to 1.
- seq_length: Defaults to `None`, which means it will be set to `max_length`. To limit the sequence length of the dataset, it is recommended to use the `--max_length` argument under "Basic Parameters" instead; this parameter does not need to be set explicitly.
- use_cpu_initialization: Initialize weights on the CPU. Defaults to `False`. This option is used during weight conversion between Hugging Face (HF) and MCore formats. The value typically does not need to be modified.
- 🔥megatron_extra_kwargs: Additional arguments to be passed through directly to Megatron, provided as a JSON string. Defaults to `None`.
  - In "ms-swift<3.10", this parameter was `--extra_megatron_kwargs`.

**Learning Rate Parameters**:

- 🔥lr: The initial learning rate. The actual learning rate for each iteration will be determined based on the learning rate warmup and decay strategies. The default value is None; **for full-parameter training, the default is 1e-5, while for LoRA training, the default is 1e-4**.
- lr_decay_style: Learning rate decay strategy, default is 'cosine'. Commonly set to 'cosine', 'linear', or 'constant'.
- 🔥lr_decay_iters: Number of iterations for learning rate decay. Default is None, meaning it will be set to `--train_iters`.
- lr_warmup_iters: Number of iterations for linear learning rate warm-up, default is 0.
- 🔥lr_warmup_fraction: The fraction of the linear learning rate warmup phase, defaults to None.
- 🔥min_lr: Minimum value of the learning rate, clipping any learning rate below this threshold to this value, default is 0.

**Regularization Parameters**:

- 🔥weight_decay: Default is 0.1.
- 🔥clip_grad: L2 gradient clipping, default is 1.0.
  - The `grad_norm` printed in logs is the value before clipping.
- adam_beta1: Default is 0.9.
- adam_beta2: Default is 0.95.
- adam_eps: Default is 1e-8.
- sgd_momentum: Takes effect when `--optimizer sgd` is set. Defaults to 0.9.

**Checkpoint Parameters**:

- 🔥save: Output directory for checkpoints, default is None. During training, if this parameter is not set, it defaults to `f'megatron_output/{model_suffix}'`, e.g., `'megatron_output/Qwen2.5-7B-Instruct'`.
  - Note: **When training on multiple machines, ensure that the save paths on each node point to the same location**. Otherwise, you will need to manually consolidate these weights after training.
- 🔥save_interval: Checkpoint saving interval (steps), default is 500.
  - Note: Weights will always be saved at the end of training.
- 🔥save_retain_interval: Interval (in iterations) for retaining checkpoints. Checkpoints not at a multiple of this interval are deleted, except for the final one.
- 🔥no_save_optim: Do not save optimizer, default is False. When performing full-parameter training, this can significantly reduce storage time.
- 🔥no_save_rng: Do not save RNG, default is False.
- 🔥load: Directory of the checkpoint to load, default is None.
  - Note: If you did not convert the weights with ms-swift’s `swift export`, you must also specify `--model <hf-repo>` so that the `config.json` configuration file can be loaded.
  - For details on resuming training from a checkpoint, please refer to the description of the `--finetune` argument.
- 🔥no_load_optim: Do not load optimizer, default is False.
  - Note: When resuming training from a checkpoint, setting `--no_load_optim false` (i.e., loading the optimizer state) typically consumes significantly more GPU memory than setting `--no_load_optim true` (i.e., skipping the optimizer state).
- 🔥no_load_rng: Do not load RNG, default is False.
- 🔥finetune: Load and fine-tune the model. **Optimizer and random seed states from the checkpoint will not be loaded, and the number of iterations will be set to 0**. The default is False.
  - Note: For resuming training from a checkpoint, you should set `--load` (and additionally `--adapter_load` for LoRA training). If `--finetune true` is set, the optimizer and RNG states will not be loaded, the iteration count will be reset to 0, and no dataset skipping will occur. If `--finetune false` is set, the iteration count will be restored, and the corresponding number of previously trained samples will be skipped in the dataset. Loading of the optimizer and RNG states is controlled by `--no_load_optim` and `--no_load_rng`, respectively.
  - Streaming datasets (`--streaming`) are currently not supported for skipping datasets.
- ckpt_format: Format of the checkpoint. Options are 'torch', 'torch_dist', 'zarr'. Default is 'torch_dist'. (Currently, weight conversion only supports the 'torch_dist' format.)
- no_initialization: Do not initialize weights, default is True.
- auto_detect_ckpt_format: Automatically detect whether the checkpoint format is legacy or distributed. Default is True.
- exit_on_missing_checkpoint: If `--load` is set but **no checkpoint is found, exit directly** instead of initializing. Default is True.
- 🔥async_save: Use asynchronous checkpoint saving. Currently only applicable to the `torch_dist` distributed checkpoint format. Defaults to False.
- use_persistent_ckpt_worker: Use a persistent checkpoint worker process for async saving, i.e., create a dedicated background process to handle asynchronous saving. Defaults to False.
- ckpt_fully_parallel_load: Apply full load parallelization across DP for distributed checkpoints to accelerate weight loading speed. Defaults to False.
- ckpt_assume_constant_structure: If the model and optimizer state dict structure remains constant throughout a single training job, allows Megatron to perform additional checkpoint performance optimizations. Defaults to False.


**Distributed Parameters**:
For guidance on selecting parallelization strategies, please refer to the [Training Tips documentation](./Quick-start.md#training-tips).

- distributed_backend: Distributed backend, options are 'nccl', 'gloo'. Default is nccl.
- 🔥use_distributed_optimizer: Use a distributed optimizer (i.e., ZeRO-1). Default is True.
- 🔥tensor_model_parallel_size: TP (Tensor Parallelism) size, default is 1.
- 🔥pipeline_model_parallel_size: PP (Pipeline Parallelism) size, default is 1.
- 🔥decoder_first_pipeline_num_layers: The number of Transformer layers in the first pipeline stage of the decoder. Default is None, which means the Transformer layers are evenly distributed across all pipeline stages.
  - This parameter is typically used when **the total number of Transformer layers is not divisible by the pipeline parallelism (PP) size**, or when the first pipeline stage (PP stage 0) of a multimodal model consumes excessive GPU memory.
- 🔥decoder_last_pipeline_num_layers: The number of Transformer layers in the last pipeline stage of the decoder. Default is None, which means the Transformer layers are evenly distributed across all pipeline stages.
- 🔥sequence_parallel: Enables sequence parallel optimization; this option takes effect only when `tensor_model_parallel_size` is set. Default is False.
- 🔥context_parallel_size: CP (Context Parallelism) size, default is 1.
- tp_comm_overlap: Overlap tensor parallel communication with GEMM (General Matrix Multiplication) kernels (to reduce communication time). Default is False.
- 🔥overlap_grad_reduce: Overlap grad reduction operations in DDP (to reduce DP communication time). Default is False.
- 🔥overlap_param_gather: Overlap all-gather of parameters in the distributed optimizer (to reduce DP communication time). Default is False.
- distributed_timeout_minutes: The timeout duration for torch.distributed (in minutes). This parameter is deprecated and is now controlled by the `ddp_timeout` in the [Base Arguments](../Instruction/Command-line-parameters.md#base-arguments), with a default value of 300000 minutes.
- num_layers_per_virtual_pipeline_stage: Number of layers in each virtual pipeline stage. Default is `None`. This parameter and `--num_virtual_stages_per_pipeline_rank` can both be used to configure VPP (Virtual Pipeline Parallelism).
- num_virtual_stages_per_pipeline_rank: Number of virtual pipeline stages per pipeline-parallel rank. Default is `None`. VPP is used to reduce computational bubbles in pipeline parallelism (PP) and improve GPU utilization, at the cost of slightly increased communication overhead.
- microbatch_group_size_per_virtual_pipeline_stage: Number of consecutive microbatches processed by each virtual pipeline stage. Default is `None`, which equals `pipeline_model_parallel_size`.
- 🔥pipeline_model_parallel_layout: A string describing a custom pipeline (pp/vpp) model parallel layout. For example: "E|(t|)*3,m|m||L". Here, E, L, t, and m denote the embedding layer, loss layer, Transformer decoder layer, and MTP layer, respectively. Stages are separated by "|". Repeated stages or layers can be expressed using multiplication. Commas are only for cosmetic readability and have no syntactic meaning. The default value is None, indicating that this argument is not used to set the layout.
  - This parameter is typically used on heterogeneous GPU clusters.

**Logging Parameters**:

- log_params_norm: Logs the norm of parameters. Default is False.
- log_throughput: Log the theoretical throughput per GPU. Defaults to False.
  - Note: In non-packing scenarios, log_throughput is not accurate because `seq_length` does not equal the actual sequence length.
- tensorboard_log_interval: Interval (steps) for logging to TensorBoard, default is 1.
- tensorboard_queue_size: Queue length (related to disk I/O), similar to write intervals. Default is 50.
- log_timers_to_tensorboard: Logs timers to TensorBoard. Default is True.
- no_log_learning_rate_to_tensorboard: Do not log learning rate to TensorBoard. Default is False.
- log_validation_ppl_to_tensorboard: Writes validation perplexity to TensorBoard. Default is True.
- log_memory_to_tensorboard: Writes memory logs to TensorBoard. Default is True.
- logging_level: Logging level. Default is None.
- wandb_project: The name of the wandb project. Defaults to '', which means ignoring wandb.
- wandb_exp_name: The name of the wandb experiment. Defaults to ''.
- wandb_save_dir: The local path to save wandb results. Defaults to ''.

**Evaluation Parameters**:

- 🔥eval_iters: Number of iterations for evaluation. Defaults to `-1`, in which case an appropriate value is automatically determined based on the size of the validation dataset. **If the validation dataset size is smaller than the global batch size, evaluation will not be performed.** When using streaming datasets, this value must be set manually.
- 🔥eval_interval: The evaluation interval (steps), i.e., how many steps between each evaluation. The default is None, which means it will be set to save_interval.


**FP8 Parameters**:
- fp8_format: The FP8 format scheme used for FP8 tensors in the forward and backward pass. Options are 'e4m3' and 'hybrid'. Default is None.
- fp8_recipe: The FP8 recipe (algorithm scheme) used for FP8 tensors in the forward and backward pass. Options are 'tensorwise', 'delayed', 'mxfp8', and 'blockwise'. Default is 'delayed'.
- fp8_amax_history_len: Number of steps for which amax history is recorded per tensor. Default is 1024.
- fp8_amax_compute_algo: Algorithm for computing amax from history. Options are 'most_recent' and 'max'. Default is 'max'.
- fp8_param_gather: Keep the compute parameter in FP8 (do not use any other intermediate dtype) and perform the parameter all-gather in FP8 format. Default is False.


**Mixed Precision Parameters**:

- fp16: FP16 mode. Defaults to `None`, and will be automatically set based on the model's `torch_dtype`—specifically, `fp16` is set to `True` if `torch_dtype` is `float16` or `float32`. The `torch_dtype` is by default read from `config.json`.
- bf16: BF16 mode. Defaults to `None`, and will be automatically set based on the model's `torch_dtype`—specifically, `bf16` is set to `True` if `torch_dtype` is `bfloat16`
- apply_query_key_layer_scaling: Scales `Q * K^T` by `1 / layer number` (e.g., divide by layer_num for layer_num-th layer). This is helpful for FP16 training. Default is None, meaning that if `--fp16` is used, it will be set to True.
- 🔥attention_softmax_in_fp32: Uses FP32 for computations in attention_mask and softmax. Default is True.

**Model Parameters**: (**The following parameters typically do not need to be set as they will be configured based on the HF model’s config.json**; users don’t need to worry about them)

- num_layers: Number of transformer layers, default is None.
- hidden_size: Transformer hidden size, default is None.
- ffn_hidden_size: Hidden size of the FFN layer in the transformer. Default is None, set to `4*hidden_size`.
- num_attention_heads: Number of transformer attention heads, default is None.
- group_query_attention: Default is None. If `num_query_groups > 1`, group_query_attention is set to True, otherwise False.
- num_query_groups: Default is 1.
- max_position_embeddings: Maximum length of positional embeddings, default is None.
- position_embedding_type: Type of positional embedding, options are 'learned_absolute', 'rope', 'mrope', 'relative', and 'none'. Default is 'rope'.
- rotary_base: Default is 10000.
- rotary_percent: Default is 1.
- normalization: Options are 'LayerNorm', 'RMSNorm'. Default is RMSNorm.
- norm_epsilon: Default is 1e-5.
- swiglu: Uses swiglu instead of the default gelu. Default is True.
- untie_embeddings_and_output_weights: Unties embedding and output weights. Default is True.
- disable_bias_linear: Disables bias in linear layers. Default is True.
- add_qkv_bias: Adds bias only to QKV linear layers. Default is True.
- attention_dropout: Default is 0.
- hidden_dropout: Default is 0.
- kv_channels: Defaults to None, set to `args.hidden_size // args.num_attention_heads`.
- qk_layernorm: Whether to apply layer normalization to Q and K.
- transformer_impl: Which transformer implementation to use, options are 'local' and 'transformer_engine'. Default is transformer_engine.
- padded_vocab_size: Full vocabulary size, default is None.
- rope_scaling: Related parameters for rope_scaling, default is None. Refer to the format in [llama3.1 config.json](https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B-Instruct/file/view/master?fileName=config.json&status=1). Pass the value as a JSON string.
  - **Currently the rope_scaling module is implemented using Transformers and supports all rope_scaling options that Transformers supports.**


**MoE Parameters**:

- num_experts: The number of experts in MoE, default is None. Automatically read from config.json.
- moe_layer_freq: Frequency distribution between MoE layers and Dense layers. Default is None. This parameter is read from config.json.
- moe_ffn_hidden_size: Hidden layer size of the feedforward network (ffn) for each expert. Default is None and will be automatically read from config.json. If not found and `num_experts` is not None, it will be set to ffn_hidden_size.
- moe_shared_expert_intermediate_size: The total FFN hidden layer size for shared experts. If there are multiple shared experts, it should equal `num_shared_experts * ffn_size_of_each_shared_expert`. Default is None. Automatically read from config.json.
- moe_router_topk: The number of experts each token is routed to. Default is None. Automatically read from config.json.
- moe_router_pre_softmax: Enable pre-softmax routing for MoE, meaning that softmax will be applied before top-k selection. Default is None. Automatically read from config.json.
- 🔥moe_router_dtype: Data type used for routing computation and expert output weighted averaging. Options are 'none', 'fp32', and 'fp64', which enhances numerical stability, especially when the number of experts is large. When used together with `moe_permute_fusion`, the performance impact is negligible. Default is 'fp32'. 'none' means no change to data type.
- moe_router_score_function: Scoring function for MoE TopK routing. Can be "softmax" or "sigmoid". Default is None and is read from config.json.
- moe_router_bias_update_rate: Update rate of expert bias in the auxiliary-loss-free load balancing strategy. Expert bias is updated based on the number of tokens each expert is assigned in the global batch: bias increases for experts assigned fewer tokens, and decreases for those assigned more tokens. Default is 1e-3, same as used in DeepSeekV3.
- moe_router_enable_expert_bias: TopK routing with dynamic expert bias in the auxiliary-loss-free load balancing strategy. Routing decisions are based on the sum of routing scores and expert bias. See details at: https://arxiv.org/abs/2408.15664. Default is None and is automatically read from config.json.
- moe_router_topk_scaling_factor: Default is None. This parameter is read from config.json.
- moe_router_load_balancing_type: Determines the router’s load balancing strategy. Options are "aux_loss", "seq_aux_loss", "sinkhorn", and "none". Default is None and is read from config.json.
- 🔥expert_model_parallel_size: The degree of expert parallelism, default is 1.
- 🔥expert_tensor_parallel_size: expert tensor-parallel size. Default is 1.
  - In "ms-swift<3.9", its default is `None`, which means it equals the value of `--tensor_model_parallel_size`. This default will be changed in "ms-swift>=3.9".
- moe_token_dispatcher_type: The type of token dispatcher to use. Options include 'allgather', 'alltoall', 'flex', and 'alltoall_seq'. Default is 'alltoall'.
- 🔥moe_grouped_gemm: When each rank contains multiple experts, multiple local GEMM kernels can be launched in parallel streams to improve utilization and performance by using GroupedLinear from TransformerEngine. Default is False.
- 🔥moe_permute_fusion: Fuses token permutation operations during token dispatch. Default is False.
- 🔥moe_aux_loss_coeff: Defaults to 0, meaning the auxiliary loss is not used. **Generally, a higher value leads to worse training performance but more balanced MoE expert utilization.** Please choose an appropriate value based on experimental results.
  - Note: In ms-swift versions earlier than 3.7.1, the default is None and the value is automatically loaded from config.json.
- moe_z_loss_coeff: Scaling coefficient for z-loss. Default is None.
- 🔥moe_shared_expert_overlap: Enables overlap between shared expert computation and the dispatcher. If not enabled, shared expert computation will be performed after routing experts. Only effective when `moe_shared_expert_intermediate_size` is set. Default is False.
- 🔥moe_expert_capacity_factor: Capacity factor for each expert. `None` means no tokens will be dropped. Default is `None`. When `--moe_expert_capacity_factor` is set, tokens exceeding an expert’s capacity will be dropped based on their selection probability. This can **balance the training load and improve training speed** (for example, set it to 1. or 2.).
- moe_pad_expert_input_to_capacity: Pad the input of each expert so that its length aligns with the expert capacity length. Default is `False`. This option only takes effect if `--moe_expert_capacity_factor` is set.
- moe_token_drop_policy: Options are 'probs' and 'position'. Default is 'probs'.

**MLA Parameters**

- multi_latent_attention: Whether to use MLA. Default is False.
- q_lora_rank: Low-rank representation rank value of the Query tensor. Default is None and will be automatically read from config.json.
- kv_lora_rank: Low-rank representation rank value of the Key and Value tensors. Default is None and will be automatically read from config.json.
- qk_head_dim: Dimension of the head in the QK projection. `q_head_dim = qk_head_dim + qk_pos_emb_head_dim`. Default is None and will be automatically read from config.json.
- qk_pos_emb_head_dim: Dimension of the position embedding in the QK projection. Default is None and will be automatically read from config.json.

**Tuner Parameters**:

- train_type: Options are `'lora'` and `'full'`. Default is `'full'`.
- 🔥freeze_llm: This argument only takes effect for multimodal models and can be used in both full-parameter and LoRA training, but with different behaviors. In full-parameter training, setting `freeze_llm=True` freezes the LLM component's weights. In LoRA training with `target_modules='all-linear'`, setting `freeze_llm=True` prevents LoRA modules from being added to the LLM part. Default is `False`.
- 🔥freeze_vit: This argument only applies to multimodal models and behaves differently depending on the training mode. In full-parameter training, setting `freeze_vit=True` freezes the ViT (vision transformer) component's weights. In LoRA training with `target_modules='all-linear'`, setting `freeze_vit=True` prevents LoRA modules from being added to the ViT part. Default is `True`.
  - Note: **Here, "vit" refers not only to `vision_tower`, but also to `audio_tower`**. For Omni models, if you want to apply LoRA only to `vision_tower` and not `audio_tower`, you can modify [this code](https://github.com/modelscope/ms-swift/blob/a5d4c0a2ce0658cef8332d6c0fa619a52afa26ff/swift/llm/model/model_arch.py#L544-L554).
- 🔥freeze_aligner: This argument only affects multimodal models. In full-parameter training, setting `freeze_aligner=True` freezes the aligner (also known as projector) weights. In LoRA training with `target_modules='all-linear'`, setting `freeze_aligner=True` prevents LoRA modules from being added to the aligner component. Default is `True`.

Full-parameter Training:

- freeze_parameters: Prefixes of parameters to be frozen. Default is `[]`.
- freeze_parameters_regex: Regex expression for parameters to be frozen. Default is `None`.
- freeze_parameters_ratio: The proportion of parameters to freeze from bottom to top. Default is `0`. Setting this to `1` will freeze all parameters; you can set trainable parameters separately using `trainable_parameters`. Except for values 0 or 1, this parameter is incompatible with pipeline parallelism (PP).
- trainable_parameters: Prefixes of additional trainable parameters. Default is `[]`.
- trainable_parameters_regex: Regex expression to match additional trainable parameters. Default is `None`.

LoRA Training:

- adapter_load: The path to the adapter weights for loading, used for resuming LoRA training from a checkpoint. The default is None. The method for resuming LoRA training from a checkpoint is the same as for full-parameter training. Please pay attention to the meaning of the `--finetune` parameter.
- 🔥target_modules: Specifies the suffixes of modules to apply LoRA to. For example, you can set it as `--target_modules linear_qkv linear_proj`. The default is `['all-linear']`, which means all linear layers will be set as target modules.
  - Note: The behavior of `'all-linear'` differs between LLMs and multimodal LLMs. For standard LLMs, it automatically finds all linear layers except `lm_head` and attaches tuners. **For multimodal LLMs, tuners are by default only attached to the LLM component; this behavior can be controlled via `freeze_llm`, `freeze_vit`, and `freeze_aligner`**.
  - Note: If you want to set all router layers as target modules, you can specify `--target_modules all-router ...`. For example: `--target_modules all-router all-linear`.
- 🔥target_regex: Regex expression to specify LoRA modules. Default is `None`. If this value is provided, the `target_modules` parameter will be ignored.
- 🔥modules_to_save: After attaching a tuner, explicitly specifies additional original model modules to participate in training and storage. The default is `[]`. For example, setting `--modules_to_save word_embeddings output_layer` will unfreeze the `word_embeddings` and `output_layer` layers during LoRA training, and the weights of these modules will be saved in the final checkpoint.
- 🔥lora_rank: Default is `8`.
- 🔥lora_alpha: Default is `32`.
- lora_dropout: Default is `0.05`.
- lora_bias: Default is `'none'`. Available options: `'none'`, `'all'`. If you want all biases to be set as trainable, set this to `'all'`.
- use_rslora: Default is `False`. Whether to use `RS-LoRA`.

**DPO Parameters**
- ref_load: The loading path for the reference model. This must be provided when using DPO/KTO algorithms with full-parameter training. Defaults to `None`, which means it will be set to the same value as `load`.
- ref_adapter_load: The path to load the ref_adapter weights, default is `None`. If you want to use LoRA weights generated from SFT for DPO, please use "ms-swift>=3.8" and set `--adapter_load sft_ckpt --ref_adapter_load sft_ckpt --finetune true` during training. For resuming training from a checkpoint in this scenario, set `--adapter_load rlhf_ckpt --ref_adapter_load sft_ckpt --finetune false`.
- beta: Has the same meaning as in [TRL](https://huggingface.co/docs/trl/main/en/dpo_trainer#trl.DPOConfig). It controls the degree of deviation from the reference model. A higher beta value indicates less deviation from the reference model. For the IPO loss function (`loss_type="ipo"`), beta is the regularization parameter as mentioned in the [paper](https://huggingface.co/papers/2310.12036). Default is 0.1.
- 🔥rpo_alpha: A parameter from the [RPO paper](https://huggingface.co/papers/2404.19733) that controls the weight of the NLL term (i.e., the SFT loss) in the loss function, where `loss = dpo_loss + rpo_alpha * sft_loss`. The paper recommends setting it to `1.`. The default value is `None`, meaning the SFT loss is not included by default.
  - **Note**: In "ms-swift<3.8", the default value was `1.`. Starting from "ms-swift>=3.8", the default has been changed to `None`.
- reference_free: Whether to ignore the provided reference model and implicitly use a reference model that assigns equal probability to all responses. Default is `False`.
- label_smoothing: Default is 0.
- f_divergence_type: Default is `reverse_kl`. See the [TRL documentation](https://huggingface.co/docs/trl/main/en/dpo_trainer) for possible values.
- loss_type: Default is `'sigmoid'`. See the [TRL documentation](https://huggingface.co/docs/trl/main/en/dpo_trainer#loss-functions) for possible values.

**KTO Parameters**:
- ref_load: same meaning as in DPO.
- ref_adapter_load: same meaning as in DPO.
- beta: parameter controlling the deviation from the ref_model. Higher `beta` means less deviation from the ref_model. Default is `0.1`.
- loss_type: default is `'kto'`. See possible values in the TRL docs: https://huggingface.co/docs/trl/main/en/kto_trainer#trl.KTOConfig.loss_type.
- desirable_weight: factor to weight desirable losses to counter imbalance between desirable and undesirable pairs. Default is `1.`.
- undesirable_weight: factor to weight undesirable losses to counter imbalance between desirable and undesirable pairs. Default is `1.`.

**RM Parameters**:
- center_rewards_coefficient: A coefficient used in reward model (RM) training to incentivize the model to output rewards with zero mean. See this [paper](https://huggingface.co/papers/2312.09244) for details. Recommended value: 0.01.


## Training Parameters

Megatron training parameters are inherited from Megatron parameters and basic parameters (**sharing dataset, template, etc. with ms-swift, and also supporting model-specific parameters from ms-swift**). For details on basic parameters, please refer to [here](../Instruction/Command-line-parameters.md#base-arguments). Additionally, the following parameters are included:

- add_version: Adds a directory `<version>-<timestamp>` to `save` to prevent overwriting weights, default is True.
- padding_free: Flattens the data in a batch to avoid padding, thereby reducing memory usage and accelerating training. Default is True.
  - If you wish to customize the attention_mask, you can set `--padding_free false`.
  - Note: **The Megatron-SWIFT training feature prioritizes support for the padding-free format**. Unless under special circumstances, please do not modify this value.
- mlp_padding_free: The default is False. This is used for applying padding-free optimization to the MLP when padding_free is set to false. It allows for improved training speed and reduced memory usage while customizing the attention_mask.
- vit_gradient_checkpointing: Whether to enable gradient checkpointing for the ViT (Vision Transformer) component during multimodal model training. Defaults to `True`. (**The ViT implementation in Megatron-SWIFT uses the Hugging Face `transformers` library.**)
- gradient_checkpointing_kwargs: Arguments passed to `torch.utils.checkpoint`. For example: set `--gradient_checkpointing_kwargs '{"use_reentrant": false}'`. Defaults to `None`. This parameter only takes effect when `vit_gradient_checkpointing` is enabled.
- 🔥packing: Whether to use sequence packing to improve computational efficiency (achieving better load balancing across nodes and processes, and higher GPU utilization), at the cost of additional preprocessing time, while also stabilizing GPU memory usage. Defaults to `False`. Currently supported for CPT, SFT, DPO, KTO and RM.
  - Note: **Sequences within the same batch remain mutually invisible**, except for Qwen3-Next.
  - Note: **Packing reduces the number of samples in the dataset; please adjust the gradient accumulation steps and learning rate accordingly**.
- streaming: Stream data loading and processing, default is False.
  - Note: Since the length of a streaming dataset cannot be determined, the `--train_iters` parameter must be set. Also set the `max_epochs` parameter to ensure training exits after the specified number of epochs, and to validate and save the model weights accordingly.
  - Note: Streaming datasets can skip preprocessing wait time by overlapping preprocessing with training. Preprocessing for streaming datasets is performed only on rank 0 and then synchronized to other processes via data distribution. **This is generally less efficient than the data sharding approach used in non-streaming datasets.** When the training world_size is large, preprocessing and data distribution can become a training bottleneck.
- lazy_tokenize: Whether to use lazy tokenization. If set to `False`, all dataset samples will be tokenized (and for multimodal models, images will be loaded from disk) before training begins. Default is `None`: in LLM training, it defaults to `False`; in MLLM training, it defaults to `True` to save memory.
- cached_dataset: Use a cached dataset (generated with `swift export --to_cached_dataset true ...`) during training to avoid GPU time spent on tokenizing large datasets. Default is `[]`. Example: [here](https://github.com/modelscope/ms-swift/tree/main/examples/export/cached_dataset).
  - Note: cached_dataset supports `--packing` but does not support `--lazy_tokenize` or `--streaming`. Cached dataset is currently not supported for CP.
- enable_dft_loss: Whether to use [DFT](https://arxiv.org/abs/2508.05629) (Dynamic Fine-Tuning) loss in SFT training, default is False.
- enable_channel_loss: Enable channel-based loss. Default is `False`. Requires a `"channel"` field in the dataset. ms-swift groups and computes loss by this field (samples without `"channel"` are grouped into the default `None` channel). Dataset format reference: [channel loss](../Customization/Custom-dataset.md#channel-loss).  Channel loss is compatible with packing, padding_free, and loss_scale techniques.
- new_special_tokens: List of additional special tokens to be added. Default is `[]`. Example usage can be found [here](https://github.com/modelscope/ms-swift/blob/main/examples/megatron/lora/new_special_tokens.sh).
  - Note: You can also pass a `.txt` file path where each line contains one special token.
- 🔥task_type: Defaults to "causal_lm". Options: "causal_lm", "seq_cls".
- num_labels: Required for classification models (i.e., `--task_type seq_cls`). Represents the number of labels; default is None.
- problem_type: Required for classification models (i.e., `--task_type seq_cls`). Options: "regression", "single_label_classification", "multi_label_classification". Defaults to None. If the model is a reward_model or num_labels equals 1, this parameter is 'regression'; otherwise it is 'single_label_classification'.


## RLHF Parameters

In addition to inheriting the training parameters, the following parameters are also supported:

- 🔥rlhf_type: Default is 'dpo'. Currently, 'dpo', 'kto', and 'rm' are available.
- loss_scale: Overrides the `loss_scale` in [basic parameters](../Instruction/Command-line-parameters.md). Default is 'last_round'.
- calculate_per_token_loss: Overrides the Megatron parameter. Default is False.
