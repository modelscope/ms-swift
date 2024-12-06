# Command-Line Parameters

The following command-line parameters are passed in the format `xxx xxx`. Alternatively, you can construct `XXXArguments(xxx=xxx)` directly in the code.

## Basic Parameters

Basic parameters are included in the various capabilities of SWIFT.

### Model Parameters

- model: Model ID or local path. If it’s a custom model, use with `model_type` and `template`.
- model_type: The model group defined by SWIFT: models with the same architecture, template, and loading process can be defined as a group.
- model_revision: Model version.
- torch_dtype: Data type of the model weights. Supports `float16`, `bfloat16`, `float32`. Defaults to reading from the config file.
- attn_impl: Type of attention. Supports `flash_attn`, `sdpa`, `eager`. Defaults to sdpa.
- rope_scaling: Type of rope. Supports `linear` and `dynamic`. Use with `max_length`.
- device_map: Configuration for the device_map used by the model.
- local_repo_path: Some models rely on GitHub repos when loading. To avoid network issues with `git clone`, use the local repo path instead. Defaults to `None`.

### Data Parameters

- dataset: Dataset ID or path. Format: `dataset_id or dataset_path:subset#sample_size`, use space to separate multiple entries. Local datasets support jsonl, csv, json, and folders.
- val_dataset: Validation dataset ID or path, used in the same way as dataset.
- split_dataset_ratio: Default for how to split the training and validation datasets if `val_dataset` is not specified. Defaults to 0.01.
- data_seed: Random seed for the dataset. Defaults to 42.
- dataset_num_proc: Number of processes for dataset preprocessing. Defaults to 1.
- streaming: Stream processing. Defaults to False.
- load_from_cache_file: Whether to use cache for dataset preprocessing. Defaults to False.
  - Note: If set to True, changes to the dataset may not take effect. If training behaves unexpectedly after changing this, consider setting it to False.
- download_mode: Dataset download mode. Options include `reuse_dataset_if_exists` and `force_redownload`. Defaults to reuse_dataset_if_exists.
- strict: If True, an error is thrown whenever there is an issue in a dataset row; otherwise, the erroneous row is discarded. Defaults to False.
- model_name: For self-recognition tasks, pass in the model's name in Chinese and English, separated by a space.
- model_author: For self-recognition tasks, pass in the model author's name in Chinese and English, separated by a space.
- custom_register_path: Custom registration of complex datasets. Refer to [Adding a Dataset](../Customization/New-dataset.md).
- custom_dataset_info: Custom registration of simple datasets. Refer to [Adding a Dataset](../Customization/New-dataset/md).

### Template Parameters

- template: Template type. Refer to [Supported Models and Datasets](./Supported-models-datasets.md). Defaults to the template type associated with the model. If the model is custom, this field must be set manually.
- system: Custom system field. Defaults to the system defined by the template.
- max_length: Maximum length of tokens for a single sample.
- truncation_strategy: How to handle overly long inputs. Supports `delete` and `left`, indicating deletion and left truncation, respectively. Defaults to left.
- max_pixels: Maximum number of pixels for image preprocessing in multimodal models. Defaults to no scaling.
- tools_prompt: List of tools for agent training formatted for the system. Refer to [Agent Training](./Agent-support.md).
- loss_scale: How to adjust the loss weight for added tokens during training. Defaults to `default`, where all responses are calculated as 1 for cross-entropy loss. Specific details can be found in [Pluginization](../Customization/Plugin.md) and [Agent Training](./Agent-support.md).
- sequence_parallel_size: Number of sequences for parallel processing. Refer to [example](https://github.com/modelscope/ms-swift/tree/main/examples/train/sequence_parallel/train.sh).
- use_chat_template: Whether to use the chat template or generation template. Defaults to `True`.
- template_backend: Use Swift or Jinja for inference. If using Jinja, the Transformers `apply_chat_template` is employed. Defaults to Swift.

### Generation Parameters

Refer to [generation_config](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig) documentation.

- max_new_tokens: Maximum number of new tokens supported during inference.
- temperature: Temperature parameter.
  - The do_sample parameter has been removed in this version. Set temperature to 0 for the same effect.
- top_k: top_k parameter.
- top_p: top_p parameter.
- repetition_penalty: Penalty for repetition.
- num_beams: The parallel retention number of beam search.
- stream: Streaming output. Defaults to `False`.
- stop_words: Additional stop words.

### Quantization Parameters

- quant_method: Quantization method. Options include `bnb`, `hqq`, `eetq`. `gptq`, `awq`, and `aqlm` quantization will read from the config file.
- quant_bits: Number of bits for quantization. Different methods may support different bits; see [quantization](https://huggingface.co/docs/transformers/main/en/main_classes/quantization) documentation for details.
- hqq_axis: Axis for hqq quantization.
- bnb_4bit_compute_dtype: Compute type for bnb quantization. Supports `float16`, `bfloat16`, `float32`. Defaults to using torch_dtype.
- bnb_4bit_quant_type: Type of bnb quantization. Supports `fp4` and `nf4`. Defaults to `nf4`.
- bnb_4bit_use_double_quant: Whether to use double quantization. Defaults to `True`.
- bnb_4bit_quant_storage: Type for storing bnb quantization. Defaults to None.

## Training Related Parameters

### LoRA Parameters

- target_modules: Specify LoRA modules. Defaults to `all-linear`, automatically finds linear layers (excluding lm_head) and attaches tuners.
  - Note: This parameter applies to multiple tuners.
- target_regex: Regular expression for specifying LoRA modules. `Optional[str]` type. Defaults to `None`. If this value is provided, target_modules will not take effect.
  - Note: This parameter applies to multiple tuners.
- modules_to_save: Defaults to `[]`. After attaching tuners, specifies the modules of the original model that will participate in training and storage.
  - Note: This parameter applies to multiple tuners.

- lora_rank: Defaults to `8`.
- lora_alpha: Defaults to `32`.
- lora_dropout: Defaults to `0.05`.
- init_lora_weights: Method for initializing LoRA weights, can be set to `true`, `false`, `gaussian`, `pissa`, `pissa_niter_[number of iters]`. Default value is `true`.
- lora_bias_trainable: Defaults to `'none'`. Possible values: 'none', 'all'. You can set this to `'all'` to make all biases trainable.
- lora_dtype: Specify the dtype for the LoRA module. Supports 'float16', 'bfloat16', 'float32'. Defaults to following the original model type.
- use_dora: Defaults to `False`. Whether to use `DoRA`.
- use_rslora: Defaults to `False`. Whether to use `RS-LoRA`.

### NEFTune Parameters

- neftune_noise_alpha: Noise coefficient added by `NEFTune`, which can enhance model performance during instruction fine-tuning. Defaults to `None`. Generally can be set to 5, 10, or 15. You can refer to [related paper](https://arxiv.org/abs/2310.05914).

### FourierFt Parameters

FourierFt uses `target_modules`, `target_regex`, and `modules_to_save` parameters.

- fourier_n_frequency: Number of frequencies for Fourier transform. `int` type, similar to `r` in LoRA. Default value is `2000`.
- fourier_scaling: Scaling value for matrix W. `float` type, similar to lora_alpha in LoRA. Default value is `300.0`.

### BOFT Parameters

BOFT uses `target_modules`, `target_regex`, and `modules_to_save` parameters.

- boft_block_size: Size of BOFT blocks. Default value is 4.
- boft_block_num: Number of BOFT blocks. Cannot be used together with `boft_block_size`.
- boft_dropout: Dropout value for BOFT. Default is 0.0.

### Vera Parameters

Vera uses `target_modules`, `target_regex`, and `modules_to_save` ，three parameters.

- vera_rank: Size of Vera Attention. Default value is 256.
- vera_projection_prng_key: Whether to store the Vera mapping matrix. Defaults to True.
- vera_dropout: Dropout value for Vera. Default is `0.0`.
- vera_d_initial: Initial value for Vera's d matrix. Default is `0.1`.

### LoRA + Fine-tuning Parameters

- lorap_lr_ratio: Defaults to `None`. Recommended value is `10~16`. Specify this parameter when using LoRA for LoRA +.

### GaLore Fine-tuning Parameters

- use_galore: Defaults to `False`. Whether to use GaLore.
- galore_target_modules: Defaults to `None`. If not provided, applies GaLore to attention and MLP.
- galore_rank: Defaults to 128. Rank value for GaLore.
- galore_update_proj_gap: Defaults to 50. Update interval for the decomposed matrix.
- galore_scale: Defaults to 1.0. Matrix weight coefficient.
- galore_proj_type: Defaults to `std`. Matrix decomposition type for GaLore.
- galore_optim_per_parameter: Defaults to `False`. Whether to set a separate optimizer for each GaLore target parameter.
- galore_with_embedding: Defaults to `False`. Whether to apply GaLore to embeddings.
- galore_quantization: Whether to use q-galore. Defaults to `False`.
- galore_proj_quant: Whether to quantize the SVD decomposed matrix. Defaults to `False`.
- galore_proj_bits: Bit depth for SVD quantization.
- galore_proj_group_size: Group size for SVD quantization.
- galore_cos_threshold: Cosine similarity threshold for updating projection matrices. Default is 0.4.
- galore_gamma_proj: Increases the update interval when projection matrices become similar. This parameter is the coefficient for the interval extension. Defaults to 2.
- galore_queue_size: Queue length for calculating projection matrix similarity. Default is 5.

### LISA Fine-tuning Parameters

Note: LISA only supports full parameter training, i.e., `train_type full`.

- lisa_activated_layers: Default value is `0`. Represents not using LISA. Change to a non-zero value to specify the number of layers to activate. Suggested to set to 2 or 8.
- lisa_step_interval: Default value is `20`. Number of iterations to switch layers that allow backpropagation.

### UNSLOTH Fine-tuning Parameters

No new parameters for unsloth. Adjust existing parameters to support:

```
--tuner_backend unsloth
--train_type full/lora
--quant_bits 4
```

### LLAMAPRO Fine-tuning Parameters

- llamapro_num_new_blocks: Default value is `4`. Total number of new layers to insert.
- llamapro_num_groups: Defaults to `None`. Number of groups for inserting new blocks. If `None`, equals `llamapro_num_new_blocks`, meaning each new layer is inserted separately to the original model.

### AdaLoRA Fine-tuning Parameters

These parameters take effect when `train_type` is set to `adalora`. Adalora's `target_modules` and other parameters inherit from the corresponding parameters in LoRA, but the `lora_dtype` parameter does not apply.

- adalora_target_r: Default value is `8`. Average rank for adalora.
- adalora_init_r: Default value is `12`. Initial rank for adalora.
- adalora_tinit: Default value is `0`. Initial warmup time for adalora.
- adalora_tfinal: Default value is `0`. Final warmup time for adalora.
- adalora_deltaT: Default value is `1`. Step interval for adalora.
- adalora_beta1: Default value is `0.85`. EMA parameter for adalora.
- adalora_beta2: Default value is `0.85`. EMA parameter for adalora.
- adalora_orth_reg_weight: Default value is `0.5`. Regularization parameter for adalora.

### ReFT Fine-tuning Parameters

These parameters take effect when `train_type` is set to `reft`.

> 1. ReFT cannot merge tuners.
> 2. ReFT is not compatible with gradient_checkpointing.
> 3. If using DeepSpeed and encounter issues, please temporarily uninstall DeepSpeed.

- reft_layers: Which layers ReFT applies to. Defaults to `None`, indicating all layers. You can provide a list of layer numbers, e.g., `reft_layers 1 2 3 4`.
- reft_rank: Rank for the ReFT matrix. Default is `4`.
- reft_intervention_type: Type of ReFT. Supports 'NoreftIntervention', 'LoreftIntervention', 'ConsreftIntervention', 'LobireftIntervention', 'DireftIntervention', 'NodireftIntervention'. Defaults to `LoreftIntervention`.
- reft_args: Other supported parameters in ReFT intervention, entered in json-string format.

### Liger Fine-tuning Parameters

- use_liger: Use liger-kernel for training.

### TorchAcc Parameters

- model_layer_cls_name: Class name for the decoder layer.
- metric_warmup_step: Warmup steps for TorchAcc. Defaults to 0.
- fsdp_num: Number of FSDP. Defaults to 1.
- acc_steps: Accumulation steps. Defaults to 1.

### Pre-training and Fine-tuning Parameters

Training parameters include all of the above parameters, plus other parameters listed below.

The following parameters come from Transformers, and SWIFT has overridden their default values. SWIFT supports all parameters of the Transformers trainer. For any not listed, please refer to [hf official parameters](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Seq2SeqTrainingArguments).

- output_dir: SWIFT's output_dir is `output/model_code`.
- gradient_checkpointing: Whether to use gradient_checkpointing. SWIFT defaults to True.
- per_device_train_batch_size: Defaults to 1.
- per_device_eval_batch_size: Defaults to 1.
- logging_steps: Interval for logging outputs. Defaults to 5.
- learning_rate: Learning rate. Defaults to 1e-5 for full parameters and 1e-4 for tuners.
- weight_decay: Weight decay coefficient. Defaults to 0.1.
- lr_scheduler_type: Type of lr_scheduler. Defaults to cosine.
- lr_scheduler_kwargs: Other parameters for lr_scheduler.
- report_to: Defaults to tensorboard, does not use wandb.
- remove_unused_columns: Defaults to False.
- logging_first_step: Whether to log the first step. Defaults to True.

The following parameters are unique to SWIFT:

- add_version: Whether to add a version number to output_dir. Defaults to True.
- resume_only_model: If resuming from checkpoint, whether to only resume model weights. Defaults to False.
- check_model: Whether to check if the local model files are corrupted or altered and provide prompts. Defaults to True.
- loss_type: Type of loss. Defaults to normal CE.
- num_labels: Requires specification for classification tasks. The number of labels.

- packing: Whether to use packing. Defaults to False.
- lazy_tokenize: Whether to use lazy_tokenize. Defaults to False for LLM training and True for MLLM training.

- acc_strategy: Strategy for training accumulations. Can be `sentence` or `token` level accumulations. Defaults to `token`.
- max_new_tokens: Maximum number of tokens when `predict_with_generate=True`. Defaults to 64.
- temperature: Temperature when `predict_with_generate=True`. Defaults to 0.
- optimizer: Custom optimizer name for plugin.
- metric: Custom metric name for plugin.

### Human Alignment Parameters

Human alignment parameters include the above [training parameters](#预训练和微调参数) and also support the following parameters:

- rlhf_type: Type of alignment algorithm. Supports `dpo`, `orpo`, `simpo`, `kto`, `cpo`.
- ref_model: Original comparison model in algorithms like DPO.
- ref_model_type: Same as model_type.
- ref_model_revision: Same as model_revision.

- beta: Coefficient for the KL regularization term. Defaults to `None`. The `simpo` algorithm defaults to `2.`, and other algorithms to `0.1`. See the [documentation](./Human-alignment) for details.
- label_smoothing: Whether to use DPO smoothing. Defaults to `0`, generally set between 0 and 0.5.

- rpo_alpha: Controls the weight of adding sft_loss in DPO. Defaults to `1`. The final loss is `KL_loss + rpo_alpha * sft_loss`.

- cpo_alpha: Coefficient for nll loss in CPO/SimPO loss. Defaults to `1.`.

- simpo_gamma: Reward margin term in the SimPO algorithm. The paper recommends setting it to between 0.5 and 1.5, defaults to `1.`.

- desirable_weight: Loss weight for desirable responses in the KTO algorithm, denoted as $\lambda_D$. Defaults to `1.`.
- undesirable_weight: Loss weight for undesirable responses in the KTO paper, denoted as $\lambda_U$. Defaults to `1.`. Let $n_d$ and $n_u$ represent counts of desirable and undesirable examples in the dataset, respectively. The paper recommends controlling $\frac{\lambda_D n_D}{\lambda_U n_U} \in [1, \frac{4}{3}]$.

## Inference and Deployment Parameters

### LMDeploy Parameters

Parameter meanings can be found in the [lmdeploy documentation](https://lmdeploy.readthedocs.io/en/latest/api/pipeline.html#turbomindengineconfig).

- tp: Tensor parallelism degree. Defaults to `1`.
- session_len: Defaults to `None`.
- cache_max_entry_count: Defaults to `0.8`.
- quant_policy: Defaults to `0`.
- vision_batch_size: Defaults to `1`.

### vLLM Parameters

Parameter meanings can be found in the [vllm documentation](https://docs.vllm.ai/en/latest/models/engine_args.html).

- gpu_memory_utilization: Defaults to `0.9`.
- tensor_parallel_size: Defaults to `1`.
- pipeline_parallel_size: Defaults to `1`.
- max_num_seqs (int): Defaults to `256`.
- max_model_len: Defaults to `None`.
- disable_custom_all_reduce: Whether to disable custom all-reduce kernel and revert to NCCL. Defaults to `True`, which differs from vLLM's default.
- enforce_eager: Whether to use PyTorch eager mode or build a CUDA graph in vllm. Defaults to `False`. Setting to True can save GPU memory but may affect efficiency.
- limit_mm_per_prompt: Controls vllm's usage of multiple graphs. Defaults to `None`. For example, input `--limit_mm_per_prompt '{"image": 10, "video": 5}'`.
- vllm_max_lora_rank: Defaults to `16`. Parameters supported by vllm for LoRA.

### Merge Parameters

- merge_lora: Whether to merge LoRA. This parameter also supports llamapro and longlora.
- safe_serialization: Whether to store safetensors.
- max_shard_size: Maximum size per storage file. Defaults to 5 GiB.

### Inference Parameters

Inference parameters include [basic parameters](#基本参数), [merge parameters](#合并参数), [vLLM parameters](#vLLM参数), [LMDeploy parameters](#LMDeploy参数), plus the following:

- ckpt_dir: Checkpoint path.
- infer_backend: Inference backend, supports pt, vLLM, and LMDeploy frameworks. Defaults to `pt`.
- result_path: Path to store inference results. Defaults to the same folder as the model.
- writer_buffer_size: Default value is 65536, the size of the buffer for writing results.
- max_batch_size: Batch size for pt backend.
- val_dataset_sample: Sample size for the inference dataset.

### Deployment Parameters

Deployment parameters inherit from [inference parameters](#推理参数).

- host: Service host. Defaults to '0.0.0.0'.
- port: Port number. Defaults to 8000.
- api_key: Key required for access.
- ssl_keyfile: SSL key file.
- ssl_certfile: SSL cert file.

- owned_by: Owner of the service.
- served_model_name: Name of the model providing the service.
- verbose: Whether to print access logs. Defaults to True.
- log_interval: Interval for printing statistics. Defaults to 20 seconds.
- max_logprobs: Maximum number of logprobs to return. Defaults to 20.


## Evaluation Parameters

Evaluation parameters inherit from [deployment parameters](#评测参数).

- eval_dataset: Evaluation dataset. Please see [Evaluation](./Evaluation.md).
- eval_limit: Sample size for each evaluation set.
- eval_output_dir: Directory to store evaluation results. Defaults to a subfolder named `eval_output` in the current folder.
- temperature: Overrides the base class parameter, using `0` as the default.
- verbose: This parameter is passed in locally during evaluation in DeployArguments. Defaults to `False`.
- max_batch_size: Maximum batch size. Defaults to 256 for text evaluation and 16 for multimodal evaluation.
- eval_url: URL for evaluation. Requires `model` (name of the accessed model) and `api_key` (access password) parameters. Defaults to None, opting for local deployment evaluation.

## Export Parameters

Inference parameters include [basic parameters](#基本参数) and [merge parameters](#合并参数), along with the following:

- ckpt_dir: Checkpoint path.
- output_dir: Path to store exported results. Different export capabilities have different default storage folders:
  - Merged LoRA will be stored in `ckpt_dir` with `-merged` appended to the path.
  - Quantization will be stored in `ckpt_dir` with `-quant_method-quant_bits` appended to the path.

- quant_n_samples: Sample size for validation set in gptq/awq. Defaults to 256.
- quant_seqlen: Sequence length of the validation set. Defaults to 2048.
- quant_batch_size: Quantization batch size. Defaults to 1.
- group_size: Group size for quantization. Defaults to 128.

- push_to_hub: Whether to push to the hub.
- hub_model_id: Model ID, formatted as group/model_code.
- hub_private_repo: Whether it is a private repo.
- commit_message: Commit message.