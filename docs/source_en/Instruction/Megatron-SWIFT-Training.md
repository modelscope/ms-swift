
# Megatron-SWIFT Training

SWIFT incorporates Megatron's parallelization techniques to accelerate the training of large models, including data parallelism, tensor parallelism, pipeline parallelism, sequence parallelism, and context parallelism. For models that support Megatron training, please refer to the [Supported Models and Datasets documentation](./Supported-models-and-datasets.md).

## Environment Setup

To use Megatron-SWIFT, in addition to installing the `swift` dependencies, you also need to install the following:

```shell
pip install pybind11
# transformer_engine
pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable

# apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```

The dependency library Megatron-LM will be git cloned and installed by swift, no manual installation by the user is required. You can also use the environment variable `MEGATRON_LM_PATH` to point to the already downloaded repo path (for offline environments, use the [core_r0.11.0 branch](https://github.com/NVIDIA/Megatron-LM/tree/core_r0.11.0)).


## Quick Start Example

This section introduces a quick start example for fine-tuning the self-awareness of the Qwen2.5-7B-Instruct model using two 80GiB A100 GPUs. The following best practices can be completed within 10 minutes.

First, we need to convert the weights from HF (Hugging Face) format to Megatron format:

```shell
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model Qwen/Qwen2.5-7B-Instruct \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --test_convert_precision true \
    --output_dir Qwen2.5-7B-Instruct-mcore
```

Next, use the following script to start training. The required GPU memory resources are 2*80GiB:

```shell
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
megatron sft \
    --load Qwen2.5-7B-Instruct-mcore \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --tensor_model_parallel_size 2 \
    --micro_batch_size 4 \
    --global_batch_size 16 \
    --recompute_granularity selective \
    --train_iters 100 \
    --eval_iters 5 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_iters 10 \
    --min_lr 1e-6 \
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

```shell
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --mcore_model megatron_output/Qwen2.5-7B-Instruct/vx-xxx \
    --to_hf true \
    --torch_dtype bfloat16 \
    --test_convert_precision true \
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

- More cases can be viewed [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/megatron).
- For pretraining, you can use `megatron pt` instead of `megatron sft`, which will use a generative template for training.

## Command Line Arguments

### Megatron Parameters

**Training Parameters**:

- 🔥micro_batch_size: Batch size per device, default is 1.
- 🔥global_batch_size: Total batch size, equivalent to `micro_batch_size * data parallel size * gradient accumulation steps`. Default is 16.
- 🔥recompute_granularity: Granularity of activation recomputation, options are 'full', 'selective'. 'full' means recomputing the entire transformer layer, while 'selective' means only recomputing the core attention part of the transformer layer. 'selective' is generally recommended. Default is 'selective'.
- recompute_method: This parameter takes effect only when recompute_granularity is set to 'full', options are 'uniform', 'block'. Default is None.
- recompute_num_layers: This parameter takes effect only when recompute_granularity is set to 'full'. Default is None. If `recompute_method` is set to uniform, this parameter specifies the number of transformer layers in each uniformly divided recomputation unit. For example, you can specify `--recompute_granularity full --recompute_method uniform --recompute_num_layers 4`. The larger the recompute_num_layers, the smaller the memory usage but higher computation cost. Default is None.
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
- 🔥use_flash_attn: Uses FlashAttention mechanism implementation, default is False.
- optimizer: Optimizer type, options are 'adam', 'sgd'. Default is adam.
- dataloader_type: Default is 'cyclic', options are 'single', 'cyclic', 'external'.
- manual_gc: Disables the default garbage collector and manually triggers garbage collection. Default is False.
- manual_gc_interval: Interval at which garbage collection is triggered. Default is 0.
- seed: Random seed for python, numpy, pytorch, and cuda, default is 42.
- 🔥num_workers: Number of workers for the dataloader, default is 4.
- seq_length: Maximum sequence length to process. Default is None, meaning it will be set to `max_position_embeddings`. Megatron-SWIFT uses dynamic padding during training, so usually there is no need to modify this parameter. To limit dataset length, use the `--max_length` control in basic parameters.
- use_cpu_initialization: Initializes weights on the CPU, default is False. Used during HF and MCore weight conversion.
- no_create_attention_mask_in_dataloader: Does not create an attention mask in the dataloader, default is True.

**Learning Rate Parameters**:

- 🔥lr: Initial learning rate, which will ultimately determine the learning rate for each iteration based on the warm-up and decay strategy, default is 1e-5.
- lr_decay_style: Learning rate decay strategy, default is 'cosine'. Commonly set to 'cosine', 'linear', or 'constant'.
- 🔥lr_decay_iters: Number of iterations for learning rate decay. Default is None, meaning it will be set to `--train_iters`.
- 🔥lr_warmup_iters: Number of iterations for linear learning rate warm-up, default is 0.
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
- 🔥sequence_parallel: Enable sequence parallel optimization. Default is False.
- 🔥context_parallel_size: CP (Context Parallelism) size, default is 1.
- tp_comm_overlap: Overlap tensor parallel communication with GEMM (General Matrix Multiplication) kernels (to reduce communication time). Default is False.
- overlap_grad_reduce: Overlap grad reduction operations in DDP (to reduce DP communication time). Default is False.
- overlap_param_gather: Overlap all-gather of parameters in the distributed optimizer (to reduce DP communication time). Default is False.
- distributed_timeout_minutes: Timeout duration for torch.distributed (in minutes), default is 60 minutes.

**Logging Parameters**

- log_params_norm: Logs the norm of parameters. Default is True.
- log_throughput: Logs throughput per GPU. Default is True.
- tensorboard_log_interval: Interval (steps) for logging to TensorBoard, default is 1.
- tensorboard_queue_size: Queue length (related to disk I/O), similar to write intervals. Default is 50.
- log_timers_to_tensorboard: Logs timers to TensorBoard. Default is True.
- no_log_learning_rate_to_tensorboard: Do not log learning rate to TensorBoard. Default is False.
- log_validation_ppl_to_tensorboard: Writes validation perplexity to TensorBoard. Default is True.
- log_memory_to_tensorboard: Writes memory logs to TensorBoard. Default is True.
- logging_level: Logging level. Default is None.

**Evaluation Parameters**

- 🔥eval_iters: Number of evaluation iterations, default is 100.
- 🔥eval_interval: Evaluation interval (steps), default is None, meaning it will be set to save_interval.

**Mixed Precision Parameters**

- fp16: FP16 mode. Default is False. Set according to the model's torch_dtype.
- bf16: BF16 mode. Default is False. Set according to the model's torch_dtype.
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
- position_embedding_type: Type of positional embedding, options are 'learned_absolute', 'rope', 'relative', and 'none'. Default is 'rope'.
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
- transformer_impl: Which transformer implementation to use, options are 'local' and 'transformer_engine'. Default is transformer_engine.
- padded_vocab_size: Full vocabulary size, default is None.
- rope_scaling: Related parameters for rope_scaling, default is None. Refer to the format in [llama3.1 config.json](https://modelscope.cn/models/LLM-Research/Meta-Llama-3.1-8B-Instruct/file/view/master?fileName=config.json&status=1). Pass the value as a JSON string.

### Megatron Training Parameters

Megatron training parameters inherit from Megatron parameters and basic parameters. For information on basic parameters, see [here](./Command-line-parameters.md#base-arguments). Additionally, the following parameters are included:

- add_version: Adds a directory `<version>-<timestamp>` to `save` to prevent overwriting weights, default is True.
- 🔥lazy_tokenize: Default is False. If this parameter is set to False, all dataset samples are tokenized before training (this avoids errors during training); if set to True, tokenization occurs during training (this saves memory).
