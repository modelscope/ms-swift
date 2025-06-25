
# Megatron-SWIFT Training

SWIFT incorporates Megatron's parallelization techniques to accelerate the training of large models, including data parallelism, tensor parallelism, pipeline parallelism, sequence parallelism, context parallelism, and expert parallelism. It supports the pre-training and fine-tuning of models such as Qwen3, [Qwen3-MoE](https://github.com/modelscope/ms-swift/blob/main/examples/train/megatron/qwen3_moe.sh), Qwen2.5, Llama3, and the Deepseek-R1 distillation series. For a complete list of supported models, please refer to the [Supported Models and Datasets documentation](./Supported-models-and-datasets.md).

## Environment Setup

To use Megatron-SWIFT, in addition to installing the `swift` dependencies, you also need to install the following:

```shell
# Recommended PyTorch version: 2.5 / 2.6
pip install pybind11
# transformer_engine
# If an installation error occurs, you can refer to this issue for resolution: https://github.com/modelscope/ms-swift/issues/3793
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

Alternatively, you can also use the image:
```
modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.4.0-py311-torch2.6.0-vllm0.8.5.post1-modelscope1.27.0-swift3.5.1
modelscope-registry.cn-beijing.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.4.0-py311-torch2.6.0-vllm0.8.5.post1-modelscope1.27.0-swift3.5.1
modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.4.0-py311-torch2.6.0-vllm0.8.5.post1-modelscope1.27.0-swift3.5.1
```

The training module in the dependent library Megatron-LM will be cloned and installed by swift via `git clone`. Alternatively, you can use the environment variable `MEGATRON_LM_PATH` to point to the path of an already downloaded repository (in offline environments, use the [core_r0.12.0 branch](https://github.com/NVIDIA/Megatron-LM/tree/core_r0.12.0)).


## Quick Start Example

This section introduces a quick start example for fine-tuning the self-awareness of the Qwen2.5-7B-Instruct model using two 80GiB A100 GPUs. The following best practices can be completed within 10 minutes.

First, we need to convert the weights from HF (Hugging Face) format to Megatron format:
- If OOM occurs, simply remove `CUDA_VISIBLE_DEVICES=0`.
```shell
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model Qwen/Qwen2.5-7B-Instruct \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir Qwen2.5-7B-Instruct-mcore
```

Next, use the following script to start training. The required GPU memory resources are 2*80GiB:
- If using multi-machine training, it is recommended to share a disk and specify the same path for `--save`.
```shell
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
    --lr_warmup_iters 10 \
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

Finally, convert the Megatron format weights back to HF format:
- Note: Please point `--mcore_model` to the parent directory of `iter_xxx`. By default, the corresponding checkpoint from `latest_checkpointed_iteration.txt` will be used.

```shell
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --mcore_model megatron_output/Qwen2.5-7B-Instruct/vx-xxx \
    --to_hf true \
    --torch_dtype bfloat16 \
    --output_dir megatron_output/Qwen2.5-7B-Instruct/vx-xxx-hf
```

We then perform inference on the generated HF format weights:

```shell
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model megatron_output/Qwen2.5-7B-Instruct/vx-xxx-hf \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048
```

The inference results are as follows:

```
<<< who are you?
I am a language model developed by swift, you can call me swift-robot. How can I assist you?
```

- For pretraining, you can use `megatron pt` instead of `megatron sft`, which will use a generative template for training.
- **More examples**: Including packing, multi-node training, 32K context, DPO, MoE models, and pre-training, can be found [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/megatron).
- The custom dataset format is the same as `ms-swift`. Refer to the [custom dataset documentation](../Customization/Custom-dataset.md).

## Benchmark
The speed comparison of full-parameter training for Dense/MoE models using `megatron sft` and `swift sft` on a single machine with eight A800 GPUs is shown below. The corresponding scripts can be found [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/megatron/benchmark).

**Dense** Qwen2.5-14B:


|                  | Megatron-LM | Deepspeed-ZeRO2 | Deepspeed-ZeRO3 |
| ---------------- | ----------- | --------------- | --------------- |
| Training Speed   | 9.04s/it    | 10.32s/it       | 10.56s/it       |
| GPU Memory Usage | 8\*64GB      | 8\*80GB          | 8\*58GB          |

**MoE** Qwen1.5-MoE-A2.7B:

|                  | Megatron-LM | Deepspeed-ZeRO2 | Deepspeed-ZeRO3 |
| ---------------- | ----------- | --------------- | --------------- |
| Training Speed   | 2.93s/it    | 6.02s/it        | 24.30s/it       |
| GPU Memory Usage | 8\*66GB      | 8\*72GB          | 8\*50GB          |

## Command Line Arguments

### Megatron Parameters

**Training Parameters**:

- 🔥micro_batch_size: Batch size per device, default is 1.
- 🔥global_batch_size: Total batch size, equivalent to `micro_batch_size * data parallel size * gradient accumulation steps`. Default is 16.
- 🔥recompute_granularity: Granularity of activation recomputation, options are 'full', 'selective'. 'full' means recomputing the entire transformer layer, while 'selective' means only recomputing the core attention part of the transformer layer. 'selective' is generally recommended. Default is 'selective'.
- 🔥recompute_method: This parameter takes effect only when recompute_granularity is set to 'full', options are 'uniform', 'block'. Default is None.
- 🔥recompute_num_layers: This parameter takes effect only when recompute_granularity is set to 'full'. Default is None. If `recompute_method` is set to uniform, this parameter specifies the number of transformer layers in each uniformly divided recomputation unit. For example, you can specify `--recompute_granularity full --recompute_method uniform --recompute_num_layers 4`. The larger the recompute_num_layers, the smaller the memory usage but higher computation cost. Default is None.
- recompute_modules: Options include "core_attn", "moe_act", "layernorm", "mla_up_proj", "mlp", and "moe". The default value is `["core_attn"]`. For example, during MoE training, you can reduce memory usage by specifying `--recompute_granularity selective --recompute_modules core_attn moe`. Among these, "core_attn", "mlp", and "moe" use normal checkpointing, while "moe_act", "layernorm", and "mla_up_proj" use output-discarding checkpointing.
  - "core_attn": Recomputes the core attention part of the Transformer layer.
  - "mlp": Recomputes the dense MLP layer.
  - "moe": Recomputes the MoE layer.
  - "moe_act": Recomputes the MLP activation function part in the MoE module.
  - "layernorm": Recomputes the input_layernorm and pre_mlp_layernorm.
  - "mla_up_proj": Recomputes the MLA up-projection and RoPE application parts.
- deterministic_mode: Deterministic mode, which may lead to slower training speed, default is False.
- 🔥train_iters: Total number of training iterations, default is None.
- 🔥log_interval: Log interval (unit: iters), default is 5.
- tensorboard_dir: Directory where TensorBoard logs are written. Default is None, meaning logs will be stored in the `f'{save}/runs'` directory.
- no_masked_softmax_fusion: Default is False. Disables scaling, masking, and softmax fusion for query_key_value.
- no_bias_dropout_fusion: Default is False. Disables bias and dropout fusion.
- no_bias_swiglu_fusion: Default is False. Specify `--no_bias_dropout_fusion true` to disable bias and swiglu fusion.
- no_rope_fusion: Default is False. Specify `--no_rope_fusion true` to disable rope fusion.
- no_gradient_accumulation_fusion: Default is False. Specify `--no_gradient_accumulation_fusion true` to disable gradient accumulation fusion.
- 🔥cross_entropy_loss_fusion: Enables cross-entropy loss calculation fusion. Default is False.
- cross_entropy_fusion_impl: Implementation of cross-entropy loss fusion. Options include 'native' and 'te'. Defaults to 'native'.
- calculate_per_token_loss: Scales the cross-entropy loss according to the number of non-padded tokens in the global batch. Default is True.
- 🔥attention_backend: The attention backend to use (flash, fused, unfused, local, auto). Defaults to auto.
- optimizer: Optimizer type, options are 'adam', 'sgd'. Default is adam.
- optimizer_cpu_offload: Offloads the optimizer state to CPU. Default is `False`.
- optimizer_offload_fraction: The fraction of the optimizer state to offload to CPU. Default is `1.0`.
- use_precision_aware_optimizer: Use the precision-aware optimizer in TransformerEngine, which allows setting the main parameters and optimizer states to lower precision, such as fp16 and fp8.
- dataloader_type: Default is 'cyclic', options are 'single', 'cyclic', 'external'. If `--streaming` is enabled, set it to external.
- manual_gc: Disables the default garbage collector and manually triggers garbage collection. Default is False.
- manual_gc_interval: Interval at which garbage collection is triggered. Default is 0.
- seed: Random seed for python, numpy, pytorch, and cuda, default is 42.
- 🔥num_workers: Number of workers for the dataloader, default is 4.
  - Note: If `--streaming true` is set, it will be set to 1.
seq_length: Defaults to None, meaning it is set to `max_length`. To restrict the dataset length, please use the `--max_length` parameter in the basic arguments; there is no need to set this parameter.
- use_cpu_initialization: Initializes weights on the CPU, default is False. Used during HF and MCore weight conversion.
- no_create_attention_mask_in_dataloader: Does not create an attention mask in the dataloader, default is True.
- extra_megatron_kwargs: Additional parameters passed to Megatron, provided as a JSON object. Defaults to None.

**Learning Rate Parameters**:

- 🔥lr: Initial learning rate, which will ultimately determine the learning rate for each iteration based on the warm-up and decay strategy, default is 1e-5.
- lr_decay_style: Learning rate decay strategy, default is 'cosine'. Commonly set to 'cosine', 'linear', or 'constant'.
- 🔥lr_decay_iters: Number of iterations for learning rate decay. Default is None, meaning it will be set to `--train_iters`.
- lr_warmup_iters: Number of iterations for linear learning rate warm-up, default is 0.
- 🔥lr_warmup_fraction: The fraction of the linear learning rate warmup phase, defaults to None.
- 🔥min_lr: Minimum value of the learning rate, clipping any learning rate below this threshold to this value, default is 0.

**Regularization Parameters**:

- 🔥weight_decay: Default is 0.1.
- 🔥clip_grad: L2 gradient clipping, default is 1.0.
- adam_beta1: Default is 0.9.
- adam_beta2: Default is 0.95.
- adam_eps: Default is 1e-8.
- sgd_momentum: Default is 0.9.

**Checkpoint Parameters**:

- 🔥save: Output directory for checkpoints, default is None. During training, if this parameter is not set, it defaults to `f'megatron_output/{model_suffix}'`, e.g., `'megatron_output/Qwen2.5-7B-Instruct'`.
  - Note: When training on multiple machines, ensure that the save paths on each node point to the same location. Otherwise, you will need to manually consolidate these weights after training.
- 🔥save_interval: Checkpoint saving interval (steps), default is 500.
  - Note: Weights will always be saved at the end of training.
- 🔥no_save_optim: Do not save optimizer, default is False.
- 🔥no_save_rng: Do not save RNG, default is False.
- 🔥load: Directory of the checkpoint to load, default is None.
- 🔥no_load_optim: Do not load optimizer, default is False.
- 🔥no_load_rng: Do not load RNG, default is False.
- 🔥finetune: Load the model and fine-tune. Does not load the optimizer and random seed states from the checkpoint and resets the iteration count to 0. Default is False.
- ckpt_format: Format of the checkpoint. Options are 'torch', 'torch_dist', 'zarr'. Default is 'torch_dist'.
- no_initialization: Do not initialize weights, default is True.
- auto_detect_ckpt_format: Automatically detect whether the checkpoint format is legacy or distributed. Default is True.
- exit_on_missing_checkpoint: If `--load` is set but no checkpoint is found, exit directly instead of initializing. Default is True.

**Distributed Parameters**:

- distributed_backend: Distributed backend, options are 'nccl', 'gloo'. Default is nccl.
- 🔥use_distributed_optimizer: Use a distributed optimizer. Default is True.
- 🔥tensor_model_parallel_size: TP (Tensor Parallelism) size, default is 1.
- 🔥pipeline_model_parallel_size: PP (Pipeline Parallelism) size, default is 1.
- 🔥decoder_first_pipeline_num_layers: The number of Transformer layers in the first pipeline stage of the decoder. Default is None, which means the Transformer layers are evenly distributed across all pipeline stages.
- 🔥decoder_last_pipeline_num_layers: The number of Transformer layers in the last pipeline stage of the decoder. Default is None, which means the Transformer layers are evenly distributed across all pipeline stages.
- 🔥sequence_parallel: Enable sequence parallel optimization. Default is False.
- 🔥context_parallel_size: CP (Context Parallelism) size, default is 1.
- tp_comm_overlap: Overlap tensor parallel communication with GEMM (General Matrix Multiplication) kernels (to reduce communication time). Default is False.
- overlap_grad_reduce: Overlap grad reduction operations in DDP (to reduce DP communication time). Default is False.
- overlap_param_gather: Overlap all-gather of parameters in the distributed optimizer (to reduce DP communication time). Default is False.
- distributed_timeout_minutes: The timeout duration for torch.distributed (in minutes). This parameter is deprecated and is now controlled by the `ddp_timeout` in the [Base Arguments](./Command-line-parameters.md#base-arguments), with a default value of 300000 minutes.

**Logging Parameters**:

- log_params_norm: Logs the norm of parameters. Default is False.
- log_throughput: Logs throughput per GPU. Default is False.
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

- 🔥eval_iters: The number of iterations for evaluation. Defaults to -1, and a suitable value will be set based on the size of the validation dataset.
  - Note: If using a streaming dataset, this value needs to be set manually.
- 🔥eval_interval: Evaluation interval (steps), default is None, meaning it will be set to save_interval.

**Mixed Precision Parameters**:

- fp16: FP16 mode. The default is None, and it will be set according to the model's torch_dtype. The torch_dtype is read from the config.json by default.
- bf16: BF16 mode. The default is None, and it will be set according to the model's torch_dtype.
- apply_query_key_layer_scaling: Scales `Q * K^T` by `1 / layer number` (e.g., divide by layer_num for layer_num-th layer). This is helpful for FP16 training. Default is None, meaning that if `--fp16` is used, it will be set to True.
- attention_softmax_in_fp32: Uses FP32 for computations in attention_mask and softmax. Default is True.

**Model Parameters**: (The following parameters typically do not need to be set as they will be configured based on the HF model’s config.json; users don’t need to worry about them)

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


**MoE Parameters**:

- num_experts: The number of experts in MoE, default is None. Automatically read from config.json.
- moe_ffn_hidden_size: The hidden layer size of the feed-forward network (ffn) for each expert. Default is None, set to ffn_hidden_size. Automatically read from config.json.
- moe_shared_expert_intermediate_size: The total FFN hidden layer size for shared experts. If there are multiple shared experts, it should equal `num_shared_experts * ffn_size_of_each_shared_expert`. Default is None. Automatically read from config.json.
- moe_router_topk: The number of experts each token is routed to. Default is None. Automatically read from config.json.
- moe_router_pre_softmax: Enable pre-softmax routing for MoE, meaning that softmax will be applied before top-k selection. Default is None. Automatically read from config.json.
- 🔥moe_aux_loss_coeff: Scaling coefficient for the auxiliary loss: the recommended initial value is 1e-2. Default is None. Automatically read from config.json.
- moe_router_dtype: Data type for routing computation and expert output weighted averaging. Options include 'none', 'fp32' and 'fp64', which enhance numerical stability, particularly with a large number of experts. When used with `moe_permute_fusion`, the performance impact is negligible. The default is 'fp32'. 'none' means no change to the data type.
- moe_permute_fusion: Fuses token rearrangement operations during token dispatching. Defaults to False.
- 🔥expert_model_parallel_size: The degree of expert parallelism, default is 1.
- moe_token_dispatcher_type: The type of token dispatcher to use. Options include 'allgather', 'alltoall', 'flex', and 'alltoall_seq'. Default is 'alltoall'.
- moe_enable_deepep: Experimental feature, Enables DeepSeek/DeepEP for efficient token dispatching and combination in MoE models. Only works when using the flexible token dispatcher by setting `--moe_token_dispatcher_type flex`.
- moe_grouped_gemm: When each rank contains multiple experts, improve utilization and performance by launching multiple local GEMM kernels across multiple streams using GroupedLinear in TransformerEngine. Default is False.
- moe_router_load_balancing_type: Determines the load balancing strategy for the router. Options are "aux_loss", "seq_aux_loss", "sinkhorn", "none". Default is "aux_loss".
- moe_z_loss_coeff: Scaling coefficient for z-loss. Default is None.
- moe_expert_capacity_factor: Capacity factor for each expert, None means no tokens will be dropped. Default is None.
- moe_shared_expert_overlap: Enable overlapping of shared expert computation with scheduler communication. If this option is not enabled, shared experts will execute after the routing experts. Only effective when `moe_shared_expert_intermediate_size` is set. Default is False.

**DPO Parameters**
- ref_load: The path to load the reference model. Defaults to `None`, which means it will be set to `load`.
- beta: Has the same meaning as in [TRL](https://huggingface.co/docs/trl/main/en/dpo_trainer#trl.DPOConfig). It controls the degree of deviation from the reference model. A higher beta value indicates less deviation from the reference model. For the IPO loss function (`loss_type="ipo"`), beta is the regularization parameter as mentioned in the [paper](https://huggingface.co/papers/2310.12036). Default is 0.1.
- rpo_alpha: A parameter from the [RPO paper](https://huggingface.co/papers/2404.19733) used to control the weight of the NLL term (i.e., SFT loss) in the loss function. The total loss is calculated as `loss = dpo_loss + rpo_alpha * nll_loss`. Default is 1.
- reference_free: Whether to ignore the provided reference model and implicitly use a reference model that assigns equal probability to all responses. Default is `False`.
- label_smoothing: Default is 0.
- f_divergence_type: Default is `reverse_kl`. See the [TRL documentation](https://huggingface.co/docs/trl/main/en/dpo_trainer) for possible values.
- loss_type: Default is `'sigmoid'`. See the [TRL documentation](https://huggingface.co/docs/trl/main/en/dpo_trainer) for possible values.


### Training Parameters

Megatron training parameters inherit from Megatron parameters and basic parameters. For information on basic parameters, see [here](./Command-line-parameters.md#base-arguments). Additionally, the following parameters are included:

- add_version: Adds a directory `<version>-<timestamp>` to `save` to prevent overwriting weights, default is True.
- 🔥packing: Whether to use sequence packing, defaults to False. Currently supports `megatron pt/sft`.
- 🔥packing_cache: Specifies the directory for packing cache. The default value is `None`, which means the cache will be stored in the path defined by the environment variable `$MODELSCOPE_CACHE`. When using the packing feature across multiple nodes, ensure that all nodes share the same packing cache directory. You can achieve this by setting the `MODELSCOPE_CACHE` environment variable or by adding the `--packing_cache <shared_path>` argument in the command line.
- 🔥streaming: Stream reading and processing of the dataset, default is False. It is typically set to True when handling large datasets. For more information on streaming parameters, refer to the command-line parameters documentation.
- lazy_tokenize: Default is False. If this parameter is set to False, all dataset samples are tokenized before training (this avoids errors during training); if set to True, tokenization occurs during training (this saves memory).
- max_epochs: Forces the training to exit after reaching `max_epochs`, and performs validation and saving of the model weights. This parameter is especially useful when using a streaming dataset. Default is None.
  - Note: If you use a non-streaming dataset, this parameter will automatically calculate train_iters for you, so there is no need to pass `train_iters` manually.


### RLHF Parameters

In addition to inheriting the training parameters, the following parameters are also supported:

- rlhf_type: Default is 'dpo'. Currently, only 'dpo' is available.
- loss_scale: Overrides the `loss_scale` in [basic parameters](./Command-line-parameters.md). Default is 'last_round'.
- calculate_per_token_loss: Overrides the Megatron parameter. Default is False.
