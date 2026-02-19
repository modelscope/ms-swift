# Command Line Arguments

## Megatron Parameters

**Training Parameters**:

- 🔥micro_batch_size: Batch size per device, default is 1.
- 🔥global_batch_size: Total batch size, equivalent to `micro_batch_size * data parallel size * gradient accumulation steps`. Default is 16.
  - Here, `Data Parallelism size (DP) = Total number of GPUs / (TP × PP × CP)`.
- 🔥recompute_granularity: Granularity of activation recomputation, options are 'full', 'selective' and 'none'. 'full' means recomputing the entire transformer layer, while 'selective' means only recomputing the core attention part of the transformer layer. 'selective' is generally recommended. Default is 'selective'.
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
  - Tip: You can set `--num_train_epochs` to specify the number of training epochs. When using a non-streaming dataset, `train_iters` will be automatically calculated based on the dataset size (compatible with packing).
- 🔥num_train_epochs: Specifies the number of training epochs. When using non-streaming datasets, this parameter will automatically calculate train_iters for you without the need to manually pass `train_iters`. When using streaming datasets, this parameter will force exit the training when `num_train_epochs` is reached, and perform validation and saving of weights. Defaults to None.
- masked_softmax_fusion: Defaults to True. Used to enable the fusion of scaling, masking, and softmax for query_key_value.
- bias_dropout_fusion: Defaults to True. Used to enable the fusion of bias and dropout.
- bias_activation_fusion: If True, fuses bias addition and activation function when possible. Defaults to True.
- apply_rope_fusion: Defaults to True. Used to enable rope fusion.
  - **When using position encodings that don't support rope_fusion such as mrope, this parameter will be automatically set to False**.
- gradient_accumulation_fusion: Defaults to True. Used to enable gradient accumulation fusion.
- 🔥cross_entropy_loss_fusion: Enables cross-entropy loss computation fusion. Defaults to True.
- cross_entropy_fusion_impl: Implementation of cross-entropy loss fusion. Options include 'native' and 'te'. Defaults to 'native'.
- calculate_per_token_loss: Scales the cross-entropy loss according to the number of non-padding tokens in the global batch. Defaults to None. When `task_type` is 'causal_lm' and during pretraining/fine-tuning, it defaults to True; otherwise, it defaults to False.
- 🔥attention_backend: The attention backend to use (flash, fused, unfused, local, auto). Default is flash.
  - Some models may not support flash attention; you need to manually set `--attention_backend unfused/fused --padding_free false`, for example: Llama4, GPT-OSS.
  - If `flash_attention_3` is installed, specifying `--attention_backend flash` will prioritize using FA3. Refer to the training script [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/flash_attention_3).
- optimizer: Optimizer type, options are 'adam', 'sgd'. Default is adam.
  - Note: This 'adam' is actually 'adamw'. See [here](https://github.com/NVIDIA/TransformerEngine/blob/d8f1e68f7c414f3e7985a8b41de4443b2f819af3/transformer_engine/pytorch/optimizers/fused_adam.py#L69-L70) for reference.
- 🔥optimizer_cpu_offload: Offloads optimizer states to the CPU. For example, set: `--use_precision_aware_optimizer true --optimizer_cpu_offload true --optimizer_offload_fraction 0.7`. Defaults to `False`.
  - This parameter can significantly reduce GPU memory usage (at the cost of increased CPU memory consumption). When the `global_batch_size` is large, its impact on training speed is minimal.
- 🔥optimizer_offload_fraction: The fraction of the optimizer state to offload to CPU. Default is `1.0`.
- use_precision_aware_optimizer: Use the precision-aware optimizer in TransformerEngine, which allows setting the main parameters and optimizer states to lower precision, such as fp16 and fp8.
- main_grads_dtype: The dtype of main gradients when use_precision_aware_optimizer is enabled. Options are 'fp32' and 'bf16'. Default is 'fp32'.
- main_params_dtype: The dtype of main parameters when use_precision_aware_optimizer is enabled. Options are 'fp32' and 'fp16'. Default is 'fp32'.
- exp_avg_dtype: The dtype of exp_avg (i.e., the first moment in the Adam optimizer) when use_precision_aware_optimizer is enabled. This dtype is used for storing the optimizer state in memory during training, but does not affect the precision in kernel computation. Options are 'fp32', 'fp16', 'bf16', and 'fp8'. Default is 'fp32'.
- exp_avg_sq_dtype: The dtype of exp_avg_sq (i.e., the second moment in the Adam optimizer) when use_precision_aware_optimizer is enabled. This dtype is used for storing the optimizer state in memory during training, but does not affect the precision in kernel computation. Options are 'fp32', 'fp16', 'bf16', and 'fp8'. Default is 'fp32'.
- manual_gc: Disables the default garbage collector and manually triggers garbage collection. Default is False.
- manual_gc_interval: The interval for manually triggering garbage collection. Defaults to 0.
- manual_gc_eval: When using manual garbage collection (`--manual_gc true`), disables garbage collection at the beginning and end of each evaluation run. Defaults to True.

**Data Parameters**:

- seed: Random seed for python, numpy, pytorch, and cuda, default is 42.
- dataset_shuffle: Whether to shuffle the dataset. Defaults to True.
  - Note: **Megatron-SWIFT's shuffling includes two parts**: dataset shuffling, controlled by `dataset_shuffle`; and shuffling in train_dataloader, controlled by `train_dataloader_shuffle`.
- train_dataloader_shuffle: Whether to use shuffling for train_dataloader. Defaults to True. val_dataset is not shuffled.
- 🔥dataloader_num_workers: Number of workers for the dataloader. Defaults to 4.
  - Note: If `--streaming true` is set, it will be set to 1.
- dataloader_pin_memory: Defaults to True.
- dataloader_persistent_workers: Defaults to True.
- dataloader_prefetch_factor: Defaults to 2.
- data_sharding: Takes effect on train_dataloader when `--train_dataloader_shuffle true`. Defaults to False. This parameter controls the scope of dataset shuffling. If set to True, the dataset is first sharded, then each shard is shuffled (slightly saves memory); if set to False, the dataset is shuffled first, then sharded (better shuffling effect).
- 🔥group_by_length: Whether to group samples with roughly similar lengths together in the training dataset (with randomness), to minimize padding and ensure load balancing across nodes and processes for improved efficiency. Defaults to False. For the specific algorithm, refer to `transformers.trainer_pt_utils.get_length_grouped_indices`.
- te_rng_tracker: Use the Transformer Engine version of the random number generator. Defaults to False.
- data_parallel_random_init: Enable different random initializations across data parallel ranks. Defaults to False.
- padding_free: Flatten the data in a batch to avoid data padding, thereby reducing memory usage and accelerating training. Defaults to True.
  - If you want to customize attention_mask, you can set `--padding_free false`.
  - Note: **Megatron-SWIFT training features prioritize support for padding_free format**. Unless there are special circumstances, please do not modify this value.
- mlp_padding_free: Defaults to False. Used for padding_free optimization of mlp when padding_free is set to false. This can improve training speed and reduce memory usage while customizing attention_mask.


**Learning Rate Parameters**:
- lr_warmup_init: The initial value for learning rate warmup. The learning rate scheduler starts warming up from this value. Defaults to 0.
- 🔥lr: The initial learning rate. The actual learning rate for each iteration will be determined based on the learning rate warmup and decay strategies. The default value is None; **for full-parameter training, the default is 1e-5, while for LoRA training, the default is 1e-4**.
- lr_decay_style: Learning rate decay strategy. Defaults to 'cosine'. Typically set to 'constant', 'linear', 'cosine', 'inverse-square-root', or 'WSD'.
- 🔥lr_decay_iters: Number of iterations for learning rate decay. Default is None, meaning it will be set to `--train_iters`.
- lr_warmup_iters: Number of iterations for linear learning rate warm-up, default is 0.
- 🔥lr_warmup_fraction: The fraction of the linear learning rate warmup phase, defaults to None.
- 🔥min_lr: Minimum value of the learning rate, clipping any learning rate below this threshold to this value, default is 0.
- lr_wsd_decay_style: The decay method for the WSD annealing phase. Defaults to 'exponential'.
- lr_wsd_decay_iters: The number of iterations for learning rate decay. Defaults to None.


**Regularization Parameters**:

- 🔥weight_decay: Default is 0.1.
- weight_decay_incr_style: The increment function for weight decay. Options are 'constant', 'linear', 'cosine'. Defaults to 'constant'.
- start_weight_decay: The initial weight decay coefficient for L2 regularization.
- end_weight_decay: The weight decay coefficient for L2 regularization at the end of training.
- 🔥clip_grad: L2 gradient clipping, default is 1.0.
  - The `grad_norm` printed in logs is the value before clipping.
- adam_beta1: Default is 0.9.
- adam_beta2: Default is 0.95.
- adam_eps: Default is 1e-8.
- sgd_momentum: Takes effect when `--optimizer sgd` is set. Defaults to 0.9.

**Checkpoint Parameters**:

- 🔥output_dir: Output directory for checkpoints, default is None. During training, if this parameter is not set, it defaults to `f'megatron_output/{model_suffix}'`, e.g., `'megatron_output/Qwen2.5-7B-Instruct'`.
  - Note: **When training on multiple machines, ensure that the save paths on each node point to the same location**. Otherwise, you will need to manually consolidate these weights after training.
- 🔥save_interval: Checkpoint saving interval (steps), default is 500.
  - Note: Weights will always be saved at the end of training.
- 🔥no_save_optim: Do not save optimizer, default is False. When performing full-parameter training, this can significantly reduce storage time.
- 🔥no_save_rng: Do not save RNG, default is False.
- 🔥mcore_model: The checkpoint directory to load (mcore storage format). Defaults to None. For information about resuming training from checkpoints, please refer to the description of the `--finetune` parameter.
  - megatron-swift recommends directly loading and storing safetensors weights, refer to [mcore-bridge documentation](./Mcore-Bridge.md).
  - Difference between `--model` and `--mcore_model`: `--model/--adapters/--ref_model/--ref_adapters` are followed by safetensors weight directories, while `--mcore_model/--mcore_adapter/--mcore_ref_model/--mcore_ref_adapter` are followed by mcore weight directories. `--model/--adapters` do not support loading checkpoint resumption states. Therefore, if you set `--no_save_optim false`, mcore weight format will be additionally stored for checkpoint resumption, and you need to use `--mcore_model/--mcore_adapter` to load the checkpoint resumption state.
- 🔥no_load_optim: Do not load optimizer, default is False.
  - Note: When resuming training from a checkpoint, setting `--no_load_optim false` (i.e., loading the optimizer state) typically consumes significantly more GPU memory than setting `--no_load_optim true` (i.e., skipping the optimizer state).
- 🔥no_load_rng: Do not load RNG, default is False.
- 🔥finetune: Load and fine-tune the model. Does not load the optimizer and random seed state from the checkpoint, and sets the iteration count to 0. Defaults to True.
  - Note: For **checkpoint resumption**, you need to set `--mcore_model` (LoRA training requires additionally setting `--mcore_adapter`). If you set `--finetune true`, it will not load the optimizer state and random seed state, will set the iteration count to 0, and will not skip the dataset; if you set `--finetune false`, it will read the iteration count and skip the previously trained dataset amount, and the loading of optimizer state and random seed state is controlled by `--no_load_optim` and `--no_load_rng`.
  - Streaming datasets (`--streaming`) are currently not supported for skipping datasets.
- perform_initialization: Initialize the weights. Defaults to False.
- use_cpu_initialization: Initialize weights on the CPU. Defaults to `False`. This option is used during weight conversion between Hugging Face (HF) and MCore formats. The value typically does not need to be modified.
- 🔥async_save: Use asynchronous checkpoint saving. Currently only applicable to the `torch_dist` distributed checkpoint format. Defaults to False.
- use_persistent_ckpt_worker: Enable a persistent checkpoint worker process for async save. Defaults to False.
- dist_ckpt_save_pre_mcore_014: Save in the format prior to Megatron-Core 0.14. Defaults to False.
- dist_ckpt_optim_fully_reshardable: Make optimizer distributed checkpoint fully reshardable (TP/PP/EP/DP) as opposed to plain DP reshardability. Defaults to False.
- distrib_optim_fully_reshardable_mem_efficient: During distributed optimizer checkpoint save and load, tries to use as little memory as possible by using Gloo (instead of NCCL) and only one rank for saving. Turn on only if experiencing host or device memory issues. Has effect only when `--dist-ckpt-optim-fully-reshardable` flag is set. Defaults to False.


**Distributed Parameters**:
For guidance on selecting parallelization strategies, please refer to the [Training Tips documentation](./Quick-start.md#training-tips).

- ddp_backend: Distributed backend. Options are 'nccl' or 'gloo'. Defaults to nccl.
- ddp_timeout: Defaults to 18000000, in seconds.
- 🔥use_distributed_optimizer: Use a distributed optimizer (i.e., ZeRO-1). Default is True.
- 🔥tensor_model_parallel_size: TP (Tensor Parallelism) size, default is 1.
- 🔥pipeline_model_parallel_size: PP (Pipeline Parallelism) size, default is 1.
- 🔥decoder_first_pipeline_num_layers: The number of Transformer layers in the first pipeline stage of the decoder. Default is None, which means the Transformer layers are evenly distributed across all pipeline stages.
  - This parameter is typically used when **the total number of Transformer layers is not divisible by the pipeline parallelism (PP) size**, or when the first pipeline stage (PP stage 0) of a multimodal model consumes excessive GPU memory.
- 🔥decoder_last_pipeline_num_layers: The number of Transformer layers in the last pipeline stage of the decoder. Default is None, which means the Transformer layers are evenly distributed across all pipeline stages.
- account_for_embedding_in_pipeline_split: If set to `True`, the input embedding layer will be treated as a standard Transformer layer in the context of partitioning and placement for pipeline parallelism. The default is `False`.
- account_for_loss_in_pipeline_split: If set to `True`, the loss layer will be treated as a standard Transformer layer in the context of partitioning and placement for pipeline parallelism. The default is `False`.
- overlap_p2p_comm: Overlap pipeline parallel communication with forward and backward blocks in 1F1B. Defaults to True.
- align_param_gather: When set to True, all PP stages will launch parameter all-gather operations simultaneously. Otherwise, each PP stage will launch independently as needed. Defaults to True.
- 🔥sequence_parallel: Enables sequence parallel optimization; this option takes effect only when `tensor_model_parallel_size` is set. Default is False.
- 🔥context_parallel_size: CP (Context Parallelism) size, default is 1.
- tp_comm_overlap: Overlap tensor parallel communication with GEMM (General Matrix Multiplication) kernels (to reduce communication time). Default is False.
- 🔥overlap_grad_reduce: Overlap grad reduction operations in DDP (to reduce DP communication time). Default is False.
- 🔥overlap_param_gather: Overlap all-gather of parameters in the distributed optimizer (to reduce DP communication time). Default is False.
- virtual_pipeline_model_parallel_size: The number of virtual pipeline stages per pipeline parallel rank. Defaults to None. VPP parallelism is used to reduce computation bubbles in PP parallelism and improve GPU utilization, but will slightly increase communication overhead.
- microbatch_group_size_per_vp_stage: The number of consecutive microbatches processed by each virtual pipeline stage. Defaults to None, which equals `pipeline_model_parallel_size`.
- 🔥pipeline_model_parallel_layout: A string describing a custom pipeline (pp/vpp) model parallel layout. For example: "E|(t|)*3,m|m||L". Here, E, L, t, and m denote the embedding layer, loss layer, Transformer decoder layer, and MTP layer, respectively. Stages are separated by "|". Repeated stages or layers can be expressed using multiplication. Commas are only for cosmetic readability and have no syntactic meaning. The default value is None, indicating that this argument is not used to set the layout.
  - This parameter is typically used on heterogeneous GPU clusters.
- 🔥expert_model_parallel_size: The degree of expert parallelism, default is 1.
- 🔥expert_tensor_parallel_size: expert tensor-parallel size. Default is 1.


**Logging Parameters**:
- report_to: Enabled logging backends. Defaults to `['tensorboard']`. Options are 'tensorboard', 'wandb', and 'swanlab'. Login for 'wandb' and 'swanlab' can use `WANDB_API_KEY` and `SWANLAB_API_KEY` environment variables.
- 🔥log_interval: Time interval for logging (unit: iters). Defaults to 5.
- tensorboard_dir: Directory where tensorboard logs are written. Defaults to None, which means logs are stored in the `f'{save}/runs'` directory.
- tensorboard_queue_size: Size of the TensorBoard queue for buffering pending events and summaries. When the number of pending items reaches this value, the next call to an "add" method will trigger a flush to disk. The default is 50.
- wandb_project: Wandb project name. Defaults to 'megatron-swift'.
- wandb_exp_name: Wandb experiment name. Defaults to the value of `--output_dir`.
- swanlab_project: Swanlab project name. Defaults to 'megatron-swift'.
- swanlab_exp_name: Swanlab experiment name. Defaults to the value of `--output_dir`.


**Evaluation Parameters**:

- 🔥eval_iters: Number of iterations for evaluation. Defaults to `-1`, in which case an appropriate value is automatically determined based on the size of the validation dataset. **If the validation dataset size is smaller than the global batch size, evaluation will not be performed.** When using streaming datasets, this value must be set manually.
- 🔥eval_interval: The evaluation interval (steps), i.e., how many steps between each evaluation. The default is None, which means it will be set to save_interval.


**FP8 Parameters**:
- fp8_format: The FP8 format scheme used for FP8 tensors in the forward and backward pass. Options are 'e4m3' and 'hybrid'. Default is None.
- fp8_recipe: The FP8 recipe (algorithm scheme) used for FP8 tensors in the forward and backward pass. Options are 'tensorwise', 'delayed', 'mxfp8', and 'blockwise'. Default is 'delayed'. Note that blockwise fp8 requires CUDA version 12.9 or higher.
- fp8_amax_history_len: Number of steps for which amax history is recorded per tensor. Default is 1024.
- fp8_amax_compute_algo: Algorithm for computing amax from history. Options are 'most_recent' and 'max'. Default is 'max'.
- fp8_param_gather: Keep the compute parameter in FP8 (do not use any other intermediate dtype) and perform the parameter all-gather in FP8 format. Default is False.
  - Tips: Set this to True if you want to export weights in FP8 format; otherwise, set it to False.


**Mixed Precision Parameters**:

- fp16: FP16 mode. Defaults to `None`, and will be automatically set based on the model's `torch_dtype`—specifically, `fp16` is set to `True` if `torch_dtype` is `float16` or `float32`. The `torch_dtype` is by default read from `config.json`.
- bf16: BF16 mode. Defaults to `None`, and will be automatically set based on the model's `torch_dtype`—specifically, `bf16` is set to `True` if `torch_dtype` is `bfloat16`
- apply_query_key_layer_scaling: Scales `Q * K^T` by `1 / layer number` (e.g., divide by layer_num for layer_num-th layer). This is helpful for FP16 training. Default is None, meaning that if `--fp16` is used, it will be set to True.
- 🔥attention_softmax_in_fp32: Uses FP32 for computations in attention_mask and softmax. Default is True.
- accumulate_allreduce_grads_in_fp32: Perform gradient accumulation and allreduce operations in fp32 precision. If `--bf16` is enabled and `main_params_dtype` is 'fp32', this is set to True. Otherwise, it defaults to False.

**MoE Parameters**:
- moe_router_load_balancing_type: Determines the load balancing strategy for the router. Options include "aux_loss", "seq_aux_loss", "global_aux_loss", "sinkhorn", and "none". Note that "global_aux_loss" requires "megatron-core>=0.15". Default value is None. Read from config.json.
- 🔥moe_router_dtype: Data type used for routing computation and expert output weighted averaging. Options are 'none', 'fp32', and 'fp64', which enhances numerical stability, especially when the number of experts is large. When used together with `moe_permute_fusion`, the performance impact is negligible. Default is 'fp32'. 'none' means no change to data type.
- moe_token_dispatcher_type: The type of token dispatcher to use. Options include 'allgather', 'alltoall', 'flex', and 'alltoall_seq'. Default is 'alltoall'.
- moe_enable_deepep: Enable DeepEP for efficient token dispatching and combine in MoE models. Only works with flex token dispatcher by setting `--moe_token_dispatcher_type flex`.
- 🔥moe_grouped_gemm: When each rank contains multiple experts, multiple local GEMM kernels can be launched in parallel streams to improve utilization and performance by using GroupedLinear from TransformerEngine. Default is True.
- 🔥moe_permute_fusion: Fuses token permutation operations during token dispatch. Default is False.
- 🔥moe_aux_loss_coeff: Defaults to 0, meaning the auxiliary loss is not used. **Generally, a higher value leads to worse training performance but more balanced MoE expert utilization.** Please choose an appropriate value based on experimental results.
- moe_z_loss_coeff: Scaling coefficient for z-loss. Default is None.
- 🔥moe_shared_expert_overlap: Enables overlap between shared expert computation and the dispatcher. If not enabled, shared expert computation will be performed after routing experts. Only effective when `moe_shared_expert_intermediate_size` is set. Default is False.
- 🔥moe_expert_capacity_factor: Capacity factor for each expert. `None` means no tokens will be dropped. Default is `None`. When `--moe_expert_capacity_factor` is set, tokens exceeding an expert’s capacity will be dropped based on their selection probability. This can **balance the training load and improve training speed** (for example, set it to 1. or 2.).
- moe_pad_expert_input_to_capacity: Pad the input of each expert so that its length aligns with the expert capacity length. Default is `False`. This option only takes effect if `--moe_expert_capacity_factor` is set.
- moe_token_drop_policy: Options are 'probs' and 'position'. Default is 'probs'.


**MTP Parameters**
- mtp_num_layers: Number of Multi-Token Prediction (MTP) layers. MTP extends the prediction scope at each position to multiple future tokens. This MTP implementation uses D sequential modules to sequentially predict D additional tokens. Default is None. (requires "megatron-core>=0.14")
  - Note: The value of mtp_num_layers will not be automatically retrieved from config.json and must be set manually. You can refer to the `num_nextn_predict_layers` field in config.json to fill in this value. When using mcore-bridge, MTP weights will be loaded from safetensors files first. If not found, random initialization will be performed. (To use blockwise fp8 + mtp, please use mcore>=0.15)
- mtp_loss_scaling_factor: Scaling factor of Multi-Token Prediction (MTP) loss. We compute the average of MTP losses across all depths, then multiply it by this scaling factor to obtain the overall MTP loss, which serves as an additional training objective. Default is 0.1.

**Tuner Parameters**:

- tuner_type: Options are `'lora'` and `'full'`. Default is `'full'`. (**In ms-swift 3.x, the parameter name is `train_type`**)
- 🔥freeze_llm: This argument only takes effect for multimodal models and can be used in both full-parameter and LoRA training, but with different behaviors. In full-parameter training, setting `freeze_llm=True` freezes the LLM component's weights. In LoRA training with `target_modules=['all-linear']`, setting `freeze_llm=True` prevents LoRA modules from being added to the LLM part. Default is `False`.
- 🔥freeze_vit: This argument only applies to multimodal models and behaves differently depending on the training mode. In full-parameter training, setting `freeze_vit=True` freezes the ViT (vision transformer) component's weights. In LoRA training with `target_modules=['all-linear']`, setting `freeze_vit=True` prevents LoRA modules from being added to the ViT part. Default is `True`.
  - Note: **Here, "vit" refers not only to `vision_tower`, but also to `audio_tower`**. For Omni models, if you want to apply LoRA only to `vision_tower` and not `audio_tower`, you can modify [this code](https://github.com/modelscope/ms-swift/blob/a5d4c0a2ce0658cef8332d6c0fa619a52afa26ff/swift/llm/model/model_arch.py#L544-L554).
- 🔥freeze_aligner: This argument only affects multimodal models. In full-parameter training, setting `freeze_aligner=True` freezes the aligner (also known as projector) weights. In LoRA training with `target_modules=['all-linear']`, setting `freeze_aligner=True` prevents LoRA modules from being added to the aligner component. Default is `True`.

Full-parameter Training:

- freeze_parameters: Prefixes of parameters to be frozen. Default is `[]`.
- freeze_parameters_regex: Regex expression for parameters to be frozen. Default is `None`.
- freeze_parameters_ratio: The proportion of parameters to freeze from bottom to top. Default is `0`. Setting this to `1` will freeze all parameters; you can set trainable parameters separately using `trainable_parameters`. Except for values 0 or 1, this parameter is incompatible with pipeline parallelism (PP).
- trainable_parameters: Prefixes of additional trainable parameters. Default is `[]`.
- trainable_parameters_regex: Regex expression to match additional trainable parameters. Default is `None`.

LoRA Training:

- mcore_adapter: The path to the adapter weights for loading, used for resuming LoRA training from a checkpoint. The default is None. The method for resuming LoRA training from a checkpoint is the same as for full-parameter training. Please pay attention to the meaning of the `--finetune` parameter.
- 🔥target_modules: Specifies the suffixes of modules to apply LoRA to. For example, you can set it as `--target_modules linear_qkv linear_proj`. The default is `['all-linear']`, which means all linear layers will be set as target modules.
  - Note: The behavior of `'all-linear'` differs between LLMs and multimodal LLMs. For standard LLMs, it automatically finds all linear layers except `lm_head` and attaches tuners. **For multimodal LLMs, tuners are by default only attached to the LLM component; this behavior can be controlled via `freeze_llm`, `freeze_vit`, and `freeze_aligner`**.
  - Note: If you want to set all router layers as target modules, you can specify `--target_modules all-router ...`. For example: `--target_modules all-router all-linear`.
  - The suffix names of Linear layers differ between transformers and Megatron. In Megatron, `linear_proj` represents `o_proj`, `linear_qkv` represents the concatenation of `q_proj, k_proj, v_proj`, `linear_fc1` represents the concatenation of `gate_proj` and `up_proj`, and `linear_fc2` represents `down_proj`.
- 🔥target_regex: Regex expression to specify LoRA modules. Default is `None`. If this value is provided, the `target_modules` parameter will be ignored.
- 🔥modules_to_save: After attaching a tuner, explicitly specifies additional original model modules to participate in training and storage. The default is `[]`. For example, setting `--modules_to_save word_embeddings output_layer` will unfreeze the `word_embeddings` and `output_layer` layers during LoRA training, and the weights of these modules will be saved in the final checkpoint.
- 🔥lora_rank: Default is `8`.
- 🔥lora_alpha: Default is `32`.
- lora_dropout: Default is `0.05`.
- lora_bias: Default is `'none'`. Available options: `'none'`, `'all'`. If you want all biases to be set as trainable, set this to `'all'`.
- use_rslora: Default is `False`. Whether to use `RS-LoRA`.

**Mcore-Bridge Parameters**:

- model: The model_id or model_path of safetensors weights. Default is None. Supports resume training from checkpoint using `--no_load_optim false --no_load_rng false`.
- model_type: Model type. For details, refer to [ms-swift command-line parameters documentation](../Instruction/Command-line-parameters.md).
- 🔥save_safetensors: Defaults to True, whether to directly save as safetensors weights. If `--no_save_optim false` is set, additional mcore format weights and optimizer weights will be saved (also saved in `output_dir`). When resuming from checkpoint, use `--mcore_model/--mcore_adapter/--no_load_optim/--no_load_rng` parameters to load mcore format weights.
- adapters: adapter_id or adapter_path of LoRA incremental weights in safetensors format. Default is `[]`.
- ref_model: model_id or model_path of ref_model safetensors weights. Required when using DPO/GRPO/KTO algorithms with full-parameter training. Default is None, set to `--model`.
- ref_adapters: List of adapter_id or adapter_path of ref_adapters safetensors weights (currently only supports length of 1). Default is `[]`.
- use_hf: Controls whether to use ModelScope or HuggingFace for model download, dataset download, and model push. Default is False, using ModelScope.
- hub_token: Hub token. ModelScope hub token can be found [here](https://modelscope.cn/my/myaccesstoken). Default is None.
- merge_lora: Whether to store merged weights. Defaults to None. If `save_safetensors` is set to True, this parameter defaults to `True`; otherwise, it defaults to False. That is, by default, LoRA will be merged when storing in safetensors format; LoRA will not be merged when storing in torch_dist format.
- max_shard_size: Maximum file size for safetensors format storage, defaults to '5GB'.
- 🔥offload_bridge: Use CPU main memory to store HF format weights exported by Megatron for vLLM updates, to reduce GPU memory usage. Defaults to False. (Takes effect in GRPO/GKD algorithms)

**Multimodal Parameters**:
- vit_gradient_checkpointing: Whether to enable gradient checkpointing for the ViT (Vision Transformer) component during multimodal model training. Defaults to `True`. (**The ViT implementation in Megatron-SWIFT uses the Hugging Face `transformers` library.**)
- attn_impl: When training a multimodal model, sets the `attn_impl` implementation used for the ViT part. Defaults to `'flash_attn'`.
- vit_lr: Specifies the learning rate for the ViT module when training multimodal models. Default is `None`, same as `learning_rate`. Typically used together with `--freeze_vit` and `--freeze_aligner`.
  - Note: The "learning rate" printed in the logs is the learning rate of the LLM.
- aligner_lr: Specifies the learning rate for the aligner module in multimodal models. Default is `None`, same as `learning_rate`.
- gradient_checkpointing_kwargs: Arguments passed to `torch.utils.checkpoint`. For example: set `--gradient_checkpointing_kwargs '{"use_reentrant": false}'`. Defaults to `None`. This parameter only takes effect when `vit_gradient_checkpointing` is enabled.


**Other Parameters**:

- check_model: Check local model files for corruption or modifications and provide prompts. Defaults to True. **If in an offline environment, please set to False**.
- rope_scaling: Parameters related to rope_scaling. Defaults to None. For format reference, see [llama3.1 config.json](https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B-Instruct/file/view/master?fileName=config.json&status=1), pass as a JSON string.
  - **Currently the rope_scaling module uses transformers implementation and supports all rope_scaling supported by transformers.**
- apply_wd_to_qk_layernorm: Used for Qwen3-Next/Qwen3.5 full-parameter training to apply weight decay to qk layernorm. Defaults to False.
- enable_dft_loss: Whether to use [DFT](https://arxiv.org/abs/2508.05629) (Dynamic Fine-Tuning) loss in SFT training. Defaults to False.
- enable_channel_loss: Enable channel loss. Defaults to `False`. You need to prepare a "channel" field in the dataset, and ms-swift will group and calculate loss based on this field (if the "channel" field is not prepared, it will be classified under the default `None` channel). For dataset format, refer to [channel loss](../Customization/Custom-dataset.md#channel-loss). Channel loss is compatible with techniques such as packing/padding_free/loss_scale.
- 🔥task_type: Defaults to 'causal_lm'. Options are 'causal_lm', 'seq_cls', 'embedding', and 'generative_reranker'.
- num_labels: This parameter needs to be specified for classification models (i.e., `--task_type seq_cls`). Represents the number of labels. Defaults to None.
- problem_type: This parameter needs to be specified for classification models (i.e., `--task_type seq_cls`). Options are 'regression', 'single_label_classification', 'multi_label_classification'. Defaults to None. If the model is reward_model or num_labels is 1, this parameter is 'regression'; otherwise, it is 'single_label_classification'.
- 🔥save_strategy: Save strategy. Options are 'steps' and 'epoch'. Defaults to 'steps'. When set to 'epoch', both 'save_interval' and 'eval_interval' are forced to be set to 1, meaning weights are saved every epoch.
- callbacks: Custom trainer callbacks. Defaults to `[]`.


## Training Parameters

Megatron training parameters are inherited from Megatron parameters and basic parameters (**sharing dataset, template, etc. with ms-swift, and also supporting model-specific parameters from ms-swift**). For details on basic parameters, please refer to [here](../Instruction/Command-line-parameters.md#base-arguments). Additionally, the following parameters are included:

- add_version: Adds a directory `<version>-<timestamp>` to `save` to prevent overwriting weights, default is True.
- 🔥packing: Use the `padding_free` method to pack data samples of different lengths into samples of **approximately** uniform length (packing ensures that complete sequences are not split), achieving load balancing across nodes and processes during training (preventing long texts from slowing down short text training), thereby improving GPU utilization and maintaining stable memory usage. When using `--attention_backend flash`, it ensures that different sequences within packed samples remain independent and invisible to each other (except for Qwen3-Next, which contains linear-attention). This parameter defaults to `False`. All training tasks in Megatron-SWIFT support this parameter. Note: **packing will reduce the number of dataset samples, please adjust gradient accumulation steps and learning rate accordingly**.
- packing_length: the length to use for packing. Defaults to None, in which case it is set to max_length.
- packing_num_proc: Number of processes for packing, default is 1. Note that different values of `packing_num_proc` will result in different packed datasets. (This parameter does not take effect during streaming packing). Usually there is no need to modify this value, as packing speed is much faster than tokenization speed.
- streaming: Stream data loading and processing, default is False. (The shuffling of streaming datasets is not thorough, which may lead to severe loss fluctuations.)
  - Note: Since the length of a streaming dataset cannot be determined, the `--train_iters` parameter must be set. Also set the `num_train_epochs` parameter to ensure training exits after the specified number of epochs, and to validate and save the model weights accordingly.
  - Note: Streaming datasets can skip preprocessing wait time by overlapping preprocessing with training. Preprocessing for streaming datasets is performed only on rank 0 and then synchronized to other processes via data distribution. **This is generally less efficient than the data sharding approach used in non-streaming datasets.** When the training world_size is large, preprocessing and data distribution can become a training bottleneck.
- lazy_tokenize: Whether to use lazy tokenization. If set to `False`, all dataset samples will be tokenized (and for multimodal models, images will be loaded from disk) before training begins. Default is `None`: in LLM training, it defaults to `False`; in MLLM training, it defaults to `True` to save memory.
- new_special_tokens: List of additional special tokens to be added. Default is `[]`. Example usage can be found [here](https://github.com/modelscope/ms-swift/blob/main/examples/megatron/lora/new_special_tokens.sh).
  - Note: You can also pass a `.txt` file path where each line contains one special token.


## RLHF Parameters

In addition to inheriting the training parameters, the following parameters are also supported:

- 🔥rlhf_type: Default is 'dpo'. Currently, 'dpo', 'grpo', 'kto', 'rm', and 'gkd' are available.
- loss_scale: Overrides the `loss_scale` in [basic parameters](../Instruction/Command-line-parameters.md). Default is 'last_round'.
- calculate_per_token_loss: Overrides the Megatron parameter. Default is False.


### DPO Parameters

- mcore_ref_model: Loading path for ref_model. Required when using DPO/GRPO/KTO algorithms with full parameter training. Defaults to None, which sets it to `mcore_model`.
- mcore_ref_adapter: Weight loading path for ref_adapter. Defaults to None. If you want to use LoRA weights generated from SFT for DPO, set `--mcore_adapter sft_ckpt --mcore_ref_adapter sft_ckpt --finetune true` during training. For checkpoint resumption in this scenario, set `--mcore_adapter rlhf_ckpt --mcore_ref_adapter sft_ckpt --finetune false`.
- beta: Has the same meaning as in [TRL](https://huggingface.co/docs/trl/main/en/dpo_trainer#trl.DPOConfig). It controls the degree of deviation from the reference model. A higher beta value indicates less deviation from the reference model. For the IPO loss function (`loss_type="ipo"`), beta is the regularization parameter as mentioned in the [paper](https://huggingface.co/papers/2310.12036). Default is 0.1.
- 🔥rpo_alpha: A parameter from the [RPO paper](https://huggingface.co/papers/2404.19733) that controls the weight of the NLL term (i.e., the SFT loss) in the loss function, where `loss = dpo_loss + rpo_alpha * sft_loss`. The paper recommends setting it to `1.`. The default value is `None`, meaning the SFT loss is not included by default.
- reference_free: Whether to ignore the provided reference model and implicitly use a reference model that assigns equal probability to all responses. Default is `False`.
- label_smoothing: Default is 0.
- f_divergence_type: Default is `reverse_kl`. See the [TRL documentation](https://huggingface.co/docs/trl/main/en/dpo_trainer) for possible values.
- loss_type: Default is `'sigmoid'`. See the [TRL documentation](https://huggingface.co/docs/trl/main/en/dpo_trainer#loss-functions) for possible values.

### KTO Parameters

- mcore_ref_model: same meaning as in DPO.
- mcore_ref_adapter: same meaning as in DPO.
- beta: parameter controlling the deviation from the ref_model. Higher `beta` means less deviation from the ref_model. Default is `0.1`.
- loss_type: default is `'kto'`. See possible values in the TRL docs: https://huggingface.co/docs/trl/main/en/kto_trainer#trl.KTOConfig.loss_type.
- desirable_weight: factor to weight desirable losses to counter imbalance between desirable and undesirable pairs. Default is `1.`.
- undesirable_weight: factor to weight undesirable losses to counter imbalance between desirable and undesirable pairs. Default is `1.`.

### RM Parameters

- center_rewards_coefficient: A coefficient used in reward model (RM) training to incentivize the model to output rewards with zero mean. See this [paper](https://huggingface.co/papers/2312.09244) for details. Recommended value: 0.01.

### GRPO Parameters

- mcore_ref_model: Same meaning as in DPO.
- mcore_ref_adapter: Same meaning as in DPO.
- beta: KL regularization coefficient, default is 0.04. When set to 0, the ref model is not loaded.
- micro_batch_size: Batch size per device, default is 1.
- global_batch_size: Total batch size, equivalent to `micro_batch_size * data parallel size * gradient accumulation steps`. Default is 16.
- steps_per_generation: Number of optimization steps per generation round, i.e., the ratio of sampling batch size to global_batch_size. Default is 1.
- generation_batch_size: Sampling batch size, must be a multiple of global_batch_size. Default equals global_batch_size * steps_per_generation.
- num_generations: Number of samples per prompt, the G value in the paper, default is 8.
- num_generations_eval: Number of generations to sample during evaluation. This allows using fewer generations during evaluation to save computation. If `None`, uses the value of `num_generations`. Default is None.
- reward_funcs: GRPO algorithm reward functions. Options include `accuracy`, `format`, `cosine`, `repetition`, and `soft_overlong`. See swift/rewards/orm.py. You can also customize your own reward functions in the plugin. Default is `[]`.
- reward_weights: Weights for each reward function. Must match the total number of reward functions and reward models. Default is None, meaning all rewards have equal weights of `1.0`.
  - Tip: If GRPO training includes `--reward_model`, it is added at the end of the reward functions.
- truncation_strategy: The method to handle inputs exceeding `max_length`. Supported values are `delete` and `left`, representing deletion and left-side truncation respectively. The default is `left`. Note that for multi-modal models, left-side truncation may remove multi-modal tokens and cause a shape mismatch error during model forward. With the delete strategy, over-long or encoding-failed samples are discarded, and new samples are resampled from the original dataset to maintain the intended batch size.
- loss_type: Loss normalization type. Options are `['grpo', 'bnpo', 'dr_grpo']`. Default is `'grpo'`. See this [PR](https://github.com/huggingface/trl/pull/3256#discussion_r2033213348) for details.
- log_completions: Whether to log model-generated content during training. Default is False.
- vllm_mode: vLLM integration mode. Options are `server` and `colocate`. Server mode uses the vLLM server launched by `swift rollout` for sampling, while colocate mode deploys vLLM within the program. When using server mode:
- vllm_mode server parameters:
  - vllm_server_host: vLLM server host address. Default is None.
  - vllm_server_port: vLLM server port. Default is 8000.
  - vllm_server_base_url: Base URL of the vLLM server (e.g., http://local_host:8000). Default is None. When set, host and port settings are ignored.
  - vllm_server_timeout: Timeout for connecting to the vLLM server. Default is 240s.
  - vllm_server_pass_dataset: Pass additional dataset information to the vLLM server for multi-round training.
  - async_generate: Asynchronous rollout to improve training speed. Note: When enabled, sampling uses the model from the previous round update, and multi-round scenarios are not supported. Default is `false`.
  - SWIFT_UPDATE_WEIGHTS_BUCKET_SIZE: Environment variable for controlling the bucket size during weight synchronization. Applicable to full-parameter training in Server Mode. Unit is MB, default value is 512 MB.
- vllm_mode colocate parameters (for more parameter support, refer to [vLLM parameters](#vllm-parameters)):
  - vllm_gpu_memory_utilization: vLLM passthrough parameter. Default is 0.9.
  - vllm_max_model_len: vLLM passthrough parameter. Default is None.
  - vllm_enforce_eager: vLLM passthrough parameter. Default is False.
  - vllm_limit_mm_per_prompt: vLLM passthrough parameter. Default is None.
  - vllm_enable_prefix_caching: vLLM passthrough parameter. Default is True.
  - vllm_tensor_parallel_size: Tensor parallel size. Default is `1`.
  - vllm_enable_lora: Support loading LoRA adapters in the vLLM Engine. Default is False. Used to accelerate weight synchronization in LoRA training. See [documentation](../Instruction/GRPO/GetStarted/GRPO.md#weight-synchronization-acceleration) for details.
  - sleep_level: Release vLLM GPU memory during training. Options are `[0, 1, 2]`. Default is 0, meaning no release.
  - offload_optimizer: Whether to offload optimizer parameters during vLLM inference. Default is False.
  - offload_model: Whether to offload the model during vLLM inference. Default is False.
- num_iterations: Number of updates per data sample, the $\mu$ value in the [GRPO paper](https://arxiv.org/abs/2402.03300). Default is 1.
- epsilon: Clip coefficient. Default is 0.2.
- epsilon_high: Upper clip coefficient. Default is None. When set, together with epsilon, forms the clipping range `[epsilon, epsilon_high]`.
- dynamic_sample: Filter out data with zero reward standard deviation within groups and sample additional new data. Default is False.
- max_resample_times: Limit the number of resampling times under dynamic_sample setting. Default is 3.
- overlong_filter: Skip overlong truncated samples, which do not participate in loss calculation. Default is False.
- delta: Bilateral GRPO upper bound clipping value from the [INTELLECT-2 tech report](https://huggingface.co/papers/2505.07291). If set, it is recommended to be greater than 1 + epsilon. Default is None.
- importance_sampling_level: Controls importance sampling ratio calculation. Options are `token` and `sequence`. In `token` mode, the original log probability ratio for each token is preserved. In `sequence` mode, the log probability ratios of all valid tokens in the sequence are averaged. The [GSPO paper](https://arxiv.org/abs/2507.18071) uses sequence-level calculation to stabilize training. Default is `token`.
- scale_rewards: Specifies the reward scaling strategy. Options include `group` (scale by within-group standard deviation), `batch` (scale by batch-wide standard deviation), `none` (no scaling), and `gdpo` (normalize each reward function separately within groups before weighted aggregation, see [GDPO paper](https://arxiv.org/abs/2601.05242)). In ms-swift < 3.10, this parameter is boolean, where `true` corresponds to `group` and `false` corresponds to `none`. The default value is bound to `advantage_estimator`: `grpo` corresponds to `group`, `rloo` corresponds to `none`, and `reinforce_plus_plus` corresponds to `batch`.
  - Note: `gdpo` mode does not support `kl_in_reward=True`. If both are set, `kl_in_reward` will be automatically set to `False`.
  - GDPO is designed for multi-reward optimization: When using multiple reward functions, GDPO normalizes each reward function separately within groups (subtract mean, divide by std), then performs weighted aggregation using `reward_weights`, and finally applies batch-level normalization. This approach better preserves the relative differences between rewards and prevents different reward combinations from collapsing into identical advantage values.
- rollout_importance_sampling_mode: Training-inference mismatch correction mode. Options are `token_truncate`, `token_mask`, `sequence_truncate`, `sequence_mask`. Default is None (disabled). For details, refer to the [documentation](../Instruction/GRPO/AdvancedResearch/training_inference_mismatch.md).
- rollout_importance_sampling_threshold: Threshold for importance sampling weights, used for truncating or masking extreme weights. Default is 2.0.
- log_rollout_offpolicy_metrics: Whether to log training-inference mismatch diagnostic metrics (KL, PPL, χ², etc.) when `rollout_importance_sampling_mode` is not set. When `rollout_importance_sampling_mode` is set, metrics are always logged. Default is False.
- off_policy_sequence_mask_delta: Off-Policy Sequence Masking threshold from [DeepSeek-V3.2 paper](https://arxiv.org/abs/2512.02556). When set, computes `mean(old_policy_logps - policy_logps)` for each sequence. If this value exceeds the threshold AND the sequence has negative advantage, the sequence is masked out from loss computation. For details, refer to the [documentation](../Instruction/GRPO/AdvancedResearch/training_inference_mismatch.md#off-policy-sequence-masking).

Built-in reward function parameters refer to the [documentation](../Instruction/Command-line-parameters.md#reward-function-parameters).

### GKD Parameters

- teacher_model: Path or model ID of the teacher model. Required.
- teacher_model_type: Teacher model type. Default is None, auto-detected.
- teacher_model_revision: Teacher model version. Default is None.
- beta: JSD divergence interpolation coefficient. 0.0 means Forward KL, 0.5 means symmetric JSD, 1.0 means Reverse KL. Default is 0.5.
- lmbda: On-Policy learning probability. 0.0 means pure Off-Policy, 1.0 means pure On-Policy. Default is 0.5.
- seq_kd: Whether to use teacher-generated responses (Sequential KD), not yet supported. Default is False.
- temperature: Temperature for sampling and loss computation. Default is 0.9.
- offload_teacher_model: Whether to offload teacher model to CPU to save GPU memory. Default is False.
- sft_alpha: Mixing coefficient for SFT loss, `loss = jsd_loss + sft_alpha * sft_loss`. Takes effect when using dataset responses (Off-Policy). Default is 0.
- max_completion_length: Maximum tokens for generation. Default is 512.
- vllm_mode: Same as GRPO parameter, used for On-Policy generation. Colocate mode deploys vLLM within the program.
  - Note: On-Policy generation requires vLLM (`--use_vllm true --vllm_mode colocate/server`).
  - When `lmbda > 0` but vLLM is not enabled, it will automatically fall back to Off-Policy mode.

## Export Parameters

This section introduces the parameters for `megatron export`. To use the `swift export` command for exporting, please refer to the [ms-swift Command Line Parameters Documentation](../Instruction/Command-line-parameters.md#export-arguments). Compared to `swift export`, `megatron export` supports distributed and multi-node exporting. Megatron export parameters inherit from Megatron parameters and basic parameters.
- 🔥to_mcore: Convert HF format weights to Megatron format. Defaults to False.
- 🔥to_hf: Convert Megatron format weights to HF format. Defaults to False.
- 🔥merge_lora: Defaults to None. If `to_hf` is set to True, this parameter defaults to `True`, otherwise False. In other words, by default, LoRA will be merged when saving in safetensors format; when saving in torch_dist format, LoRA will not be merged. The merged weights are stored in the `--save` directory.
  - Note: Transformers 5.0 has refactored the model architecture for MoE models. This new structure does not support MoE LoRA inference and may cause inference errors. **It is recommended to merge LoRA weights for MoE models** (vLLM is not affected).
  - Note: The expert structure differs between Transformers and Megatron models. For example, the expert layers in Transformers' Qwen3-VL-MoE are implemented as Parameters rather than Linear layers. As a result, some models cannot convert LoRA delta weights (though Qwen3-VL-MoE supports conversion if LoRA is trained only on linear_proj and linear_qkv). However, most models support LoRA conversion, such as Qwen3-MoE, Qwen3-Omni-MoE, and GLM4.5-V.
- 🔥test_convert_precision: Test the precision error of HF and Megatron format weight conversion. Defaults to False.
- test_convert_dtype: The dtype used for conversion precision testing, defaults to 'float32'.
- exist_ok: If `args.save` exists, do not throw an exception and perform overwriting. Defaults to False.
- device_map: Takes effect when `--test_convert_precision true` is set and controls where the HF model is loaded. The default is `'auto'`. You can set it to `'cpu'` to save GPU memory.
