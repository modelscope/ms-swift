# Command Line Parameters

The introduction to command line parameters will cover basic parameters, atomic parameters, and integration parameters. The final list of parameters used in the command line is the integration parameters. The integration parameters inherit from the basic parameters and some atomic parameters.

## Basic Parameters

- 🔥tuner_backend: Optional values are 'peft' and 'unsloth', default is 'peft'
- 🔥train_type: Default is 'lora'. Optional values: 'lora', 'full', 'longlora', 'adalora', 'llamapro', 'adapter', 'vera', 'boft', 'fourierft', 'reft'
- seed: Default is 42
- model_kwargs: Extra parameters specific to the model. This parameter list will be logged during training for reference.
- load_dataset_config: When specifying resume_from_checkpoint/ckpt_dir, it will read the `args.json` in the saved file and assign values to any parameters that are None (can be overridden by manual input). If this parameter is set to True, it will read the data parameters as well. Default is False.
- use_hf: Default is False. Controls model and dataset downloading, and model pushing to the hub.
- hub_token: Hub token. You can check the modelscope hub token [here](https://modelscope.cn/my/myaccesstoken).

### Model Parameters

- 🔥model: Model ID or local path to the model. If it's a custom model, please use it with `model_type` and `template`.
- 🔥model_type: Model type. The same model architecture, template, and loading process define a model type.
- model_revision: Model version.
- 🔥torch_dtype: Data type for model weights, supports `float16`, `bfloat16`, `float32`, default is read from the config file.
- attn_impl: Attention type, supports `flash_attn`, `sdpa`, `eager`, default is sdpa.
- rope_scaling: Rope type, supports `linear` and `dynamic`, to be used with `max_length`.
- device_map: Configuration of the device map used by the model, e.g., 'auto', 'cpu', json string, json file path.
- local_repo_path: Some models require a GitHub repo when loading. To avoid network issues during `git clone`, you can directly use a local repo. This parameter needs to pass the local repo path, default is `None`.

### Data Parameters

- 🔥dataset: Dataset ID or path. The format is `dataset_id or dataset_path:sub_dataset#sample_count`, where sub_dataset and sample_count are optional. Use spaces to pass multiple datasets. Local datasets support jsonl, csv, json, and folders.
- 🔥val_dataset: Validation dataset ID or path.
- 🔥split_dataset_ratio: How to split the training and validation sets when val_dataset is not specified, default is 0.01.
- data_seed: Random seed for the dataset, default is 42.
- 🔥dataset_num_proc: Number of processes for dataset preprocessing, default is 1.
- 🔥streaming: Stream read and process the dataset, default is False.
- load_from_cache_file: Use cache for dataset preprocessing, default is False.
  - Note: If set to True, it may not take effect if the dataset changes. If modifying this parameter leads to issues during training, consider setting it to False.
- download_mode: Dataset download mode, including `reuse_dataset_if_exists` and `force_redownload`, default is reuse_dataset_if_exists.
- strict: If True, the dataset will throw an error if any row has a problem; otherwise, it will discard the erroneous row. Default is False.
- 🔥model_name: For self-awareness tasks, input the model's Chinese and English names separated by space.
- 🔥model_author: For self-awareness tasks, input the model author's Chinese and English names separated by space.
- custom_dataset_info: Custom simple dataset registration, refer to [Add New Dataset](../Customization/New-dataset.md).
- custom_register_path: Custom complex dataset registration, refer to [Add New Dataset](../Customization/New-dataset.md).

### Template Parameters

- 🔥template: Template type, default uses the corresponding template type of the model. If it is a custom model, please refer to [Supported Models and Datasets](./Supported-models-and-datasets) and manually input this field.
- 🔥system: Custom system field, default is None, uses the default system of the template.
- 🔥max_length: Maximum length of tokens for a single sample, default is None (no limit).
- truncation_strategy: How to handle overly long tokens, supports `delete` and `left`, representing deletion and left trimming, default is left.
- 🔥max_pixels: Maximum pixel count for pre-processing images in multimodal models (H*W), default is no scaling.
- tools_prompt: The list of tools for agent training converted to system format, refer to [Agent Training](./Agent-support.md), default is 'react_en'.
- loss_scale: How to add token loss weight during training. Default is `'default'`, meaning all responses (including history) are treated as 1 for cross-entropy loss. For specifics, see [Plugin](../Customization/plugin.md) and [Agent Training](./Agent-support.md).
- sequence_parallel_size: Number of sequence parallelism. Refer to [example](https://github.com/modelscope/ms-swift/tree/main/examples/train/sequence_parallel/train.sh).
- use_chat_template: Use chat template or generation template, default is `True`. `swift pt` is automatically set to the generation template.
- template_backend: Use swift or jinja for inference. If using jinja, it will utilize transformers' `apply_chat_template`. Default is swift.

### Generation Parameters

Refer to the [generation_config](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig) documentation.

- 🔥max_new_tokens: Maximum new token count supported during inference, default is None (no limit).
- temperature: Temperature parameter, default is None, read from generation_config.json.
  - Note: The do_sample parameter has been removed in this version; set temperature to 0 for the same effect.
- top_k: Top_k parameter, read from generation_config.json.
- top_p: Top_p parameter, read from generation_config.json.
- repetition_penalty: Penalty for repetition, default is None, read from generation_config.json.
- num_beams: Number of beams for beam search, default is 1.
- 🔥stream: Stream output, default is `False`.
- stop_words: Additional stop words, default is `[]`.

### Quantization Parameters

The following are quantization parameters for loading models. For specific meanings, see the [Quantization](https://huggingface.co/docs/transformers/main/en/main_classes/quantization) documentation. This does not include quantization parameters related to `swift export`, such as `gptq` and `awq`.

- 🔥quant_method: Quantization method used when loading the model, options are `bnb`, `hqq`, `eetq`.
- 🔥quant_bits: Number of bits for quantization, default is None.
- hqq_axis: HQQ quantization axis, default is None.
- bnb_4bit_compute_dtype: BNB quantization compute type, options are `float16`, `bfloat16`, `float32`, default is set to `torch_dtype`.
- bnb_4bit_quant_type: BNB quantization type, supports `fp4` and `nf4`, default is `nf4`.
- bnb_4bit_use_double_quant: Whether to use double quantization, default is `True`.
- bnb_4bit_quant_storage: BNB quantization storage type, default is None.

## Atomic Parameters

### Seq2SeqTrainer Parameters

This parameter list inherits from transformers `Seq2SeqTrainingArguments`, with default values overridden by ms-swift. For unlisted items, refer to the [HF Official Documentation](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Seq2SeqTrainingArguments).

- 🔥output_dir: Default is `output/<model_name>`.
- 🔥gradient_checkpointing: Whether to use gradient checkpointing, default is True.
- 🔥deepspeed: Default is None. Can be set to 'zero2', 'zero3', 'zero2-offload', 'zero3-offload' to use the built-in deepspeed configuration files from ms-swift.
- 🔥per_device_train_batch_size: Default is 1.
- 🔥per_device_eval_batch_size: Default is 1.
- weight_decay: Weight decay coefficient, default value is 0.1.
- 🔥learning_rate: Learning rate, default is 1e-5 for all parameters, and 1e-4 for the tuner.
- lr_scheduler_type: LR scheduler type, default is cosine.
- lr_scheduler_kwargs: Other parameters for the LR scheduler.
- report_to: Default is `tensorboard`.
- remove_unused_columns: Default is False.
- logging_first_step: Whether to log the first step print, default is True.
- logging_steps: Interval for logging prints, default is 5.
- metric_for_best_model: Default is None. When `predict_with_generate` is set to False, it is 'loss'; otherwise, it is 'rouge-l'.
- greater_is_better: Default is None. When `metric_for_best_model` contains 'loss', set to False; otherwise, set to True.

Other important parameters:
- 🔥num_train_epochs: Number of training epochs, default is 3.
- 🔥gradient_accumulation_steps: Gradient accumulation, default is 1.
- 🔥gradient_checkpointing_kwargs: Parameters passed to `torch.utils.checkpoint`. For example, set to `{"use_reentrant": false}`.
- save_strategy: Strategy for saving the model, options are 'no', 'steps', 'epoch', default is 'steps'.
- 🔥save_steps: Default is 500.
- 🔥save_total_limit: Default is None, saving all checkpoints.
- 🔥eval_strategy: Evaluation strategy, follows `save_strategy`.
- 🔥eval_steps: Default is None. If evaluation dataset exists, follows `save_steps`.
- max_steps: Default is -1, maximum number of training steps. Must be set when the dataset is streaming.
- 🔥warmup_ratio: Default is 0.
- save_on_each_node: Default is False. To be considered in multi-machine training.
- save_only_model: Default is False. Whether to save only model weights.
- 🔥resume_from_checkpoint: Checkpoint resume parameter, default is None.
- 🔥ddp_backend: Default is None.
- 🔥ddp_find_unused_parameters: Default is None.
- 🔥dataloader_num_workers: Default is 0.
- 🔥neftune_noise_alpha: Noise coefficient added by neftune, default is 0. Generally can be set to 5, 10, 15.
- push_to_hub: Push training weights to hub, default is False.
- hub_model_id: Default is None.
- hub_private_repo: Default is False.

### Tuner Parameters

- 🔥freeze_vit: Freeze ViT. Default is True. Applicable for full parameters and LoRA.
- 🔥freeze_aligner: Freeze aligner. Default is True, applicable for full parameters and LoRA.
- 🔥freeze_llm: Freeze LLM. Default is False. Applicable for full parameters and LoRA.
- 🔥target_modules: Specify the LoRA module, default is `all-linear`, automatically finds linear layers except for lm_head and attaches the tuner. This parameter is not limited to LoRA.
- 🔥target_regex: Specify a regex expression for the LoRA module. Default is `None`, if this value is provided, target_modules does not take effect. This parameter is not limited to LoRA.
- modules_to_save: After the tuner is attached, the original model's modules used during training and storage, default is `[]`. This parameter is not limited to LoRA.

#### Full Parameters

- freeze_parameters: Prefix of parameters to be frozen, default is `[]`.
- freeze_parameters_ratio: Ratio of parameters to freeze from the bottom up, default is 0. Setting it to 1 will freeze all parameters. Combine with `trainable_parameters` to set trainable parameters.
- trainable_parameters: Prefix of trainable parameters, default is `[]`.

#### LoRA

- 🔥lora_rank: Default is `8`.
- 🔥lora_alpha: Default is `32`.
- lora_dropout: Default is `0.05`.
- 🔥init_lora_weights: Method to initialize LoRA weights, can be specified as `true`, `false`, `gaussian`, `pissa`, `pissa_niter_[number of iters]`, default is `true`.
- lora_bias: Default is `'none'`, selectable values are: 'none', 'all'. If you want to set all biases as trainable, you can set it to `'all'`.
- lora_dtype: Specify the dtype of the LoRA module. Supports 'float16', 'bfloat16', 'float32', defaults to the original model type.
- 🔥use_dora: Default is `False`, whether to use `DoRA`.
- use_rslora: Default is `False`, whether to use `RS-LoRA`.
- 🔥lorap_lr_ratio: LoRA+ parameter, default value is `None`, recommended values `10~16`, specifying this parameter allows using lora+ when using LoRA.

#### FourierFt

FourierFt uses the three parameters `target_modules`, `target_regex`, and `modules_to_save`.

- fourier_n_frequency: Number of frequencies in Fourier transform, an `int`, similar to `r` in LoRA. Default value is `2000`.
- fourier_scaling: Scaling value of matrix W, a `float`, similar to `lora_alpha` in LoRA. Default value is `300.0`.

#### BOFT

BOFT uses the three parameters `target_modules`, `target_regex`, and `modules_to_save`.

- boft_block_size: Size of BOFT blocks, default value is 4.
- boft_block_num: Number of BOFT blocks, cannot be used simultaneously with `boft_block_size`.
- boft_dropout: Dropout value for BOFT, default is 0.0.

#### Vera

Vera uses the three parameters `target_modules`, `target_regex`, and `modules_to_save`.

- vera_rank: Size of Vera Attention, default value is 256.
- vera_projection_prng_key: Whether to store the Vera mapping matrix, default is True.
- vera_dropout: Dropout value for Vera, default is `0.0`.
- vera_d_initial: Initial value of Vera's d matrix, default is `0.1`.

#### GaLore

- 🔥use_galore: Default value is False, whether to use GaLore.
- galore_target_modules: Default is None, if not provided, applies GaLore to attention and MLP.
- galore_rank: Default value is 128, GaLore rank value.
- galore_update_proj_gap: Default is 50, interval for updating decomposed matrices.
- galore_scale: Default is 1.0, matrix weight coefficient.
- galore_proj_type: Default is `std`, type of GaLore matrix decomposition.
- galore_optim_per_parameter: Default value is False, whether to set a separate optimizer for each Galore target parameter.
- galore_with_embedding: Default value is False, whether to apply GaLore to embedding.
- galore_quantization: Whether to use q-galore, default is `False`.
- galore_proj_quant: Whether to quantize the SVD decomposition matrix, default is `False`.
- galore_proj_bits: Number of bits for SVD quantization.
- galore_proj_group_size: Number of groups for SVD quantization.
- galore_cos_threshold: Cosine similarity threshold for updating projection matrices. Default value is 0.4.
- galore_gamma_proj: As the projection matrix becomes more similar over time, this parameter is the coefficient for extending the update interval. Default value is 2.
- galore_queue_size: Length of the queue for calculating projection matrix similarity, default is 5.

#### LISA

Note: LISA only supports full parameters, i.e., `train_type full`.

- 🔥lisa_activated_layers: Default value is `0`, representing LISA is not used. Setting to a non-zero value activates that many layers, it is recommended to set to 2 or 8.
- lisa_step_interval: Default value is `20`, number of iter to switch to layers that can be backpropagated.

#### UNSLOTH

🔥Unsloth has no new parameters; adjusting existing ones will suffice to support it:

```
--tuner_backend unsloth
--train_type full/lora
--quant_bits 4
```

#### LLAMAPRO

- 🔥llamapro_num_new_blocks: Default value is `4`, total number of new layers to insert.
- llamapro_num_groups: Default value is `None`, number of groups to insert new blocks. If `None`, it equals `llamapro_num_new_blocks`, meaning each new layer is inserted separately into the original model.

#### AdaLoRA

When the `train_type` parameter is set to `adalora`, the following parameters take effect. The `adalora` parameters such as `target_modules` inherit from the corresponding parameters of `lora`, but the `lora_dtype` parameter does not take effect.

- adalora_target_r: Default value is `8`, average rank of AdaLoRA.
- adalora_init_r: Default value is `12`, initial rank of AdaLoRA.
- adalora_tinit: Default value is `0`, initial warmup of AdaLoRA.
- adalora_tfinal: Default value is `0`, final warmup of AdaLoRA.
- adalora_deltaT: Default value is `1`, step interval of AdaLoRA.
- adalora_beta1: Default value is `0.85`, EMA parameter of AdaLoRA.
- adalora_beta2: Default value is `0.85`, EMA parameter of AdaLoRA.
- adalora_orth_reg_weight: Default value is `0.5`, regularization parameter for AdaLoRA.

#### ReFT

The following parameters are effective when `train_type` is set to `reft`.

> 1. ReFT cannot merge tuners.
> 2. ReFT is not compatible with gradient checkpointing.
> 3. If experiencing issues while using DeepSpeed, please uninstall DeepSpeed temporarily.

- 🔥reft_layers: Which layers ReFT is applied to, default is `None`, representing all layers. You can input a list of layer numbers, e.g., `reft_layers 1 2 3 4`.
- 🔥reft_rank: Rank of ReFT matrix, default is `4`.
- reft_intervention_type: Type of ReFT, supports 'NoreftIntervention', 'LoreftIntervention', 'ConsreftIntervention', 'LobireftIntervention', 'DireftIntervention', 'NodireftIntervention', default is `LoreftIntervention`.
- reft_args: Other supported parameters for ReFT Intervention, input in json-string format.

#### Liger

- use_liger: Use liger-kernel for training.

#### TorchAcc

- model_layer_cls_name: Class name of Decoder layer.
- metric_warmup_step: Warmup steps for TorchAcc, default is 1.
- fsdp_num: Number of FSDP, default is 1.
- acc_steps: Number of steps for evaluating accuracy during training, default is 1.

### LMDeploy Parameters

Parameter meanings can be found in the [lmdeploy documentation](https://lmdeploy.readthedocs.io/en/latest/api/pipeline.html#turbomindengineconfig).

- 🔥tp: Tensor parallelism degree, default value is `1`.
- session_len: Default value is `None`.
- cache_max_entry_count: Default value is `0.8`.
- quant_policy: Default value is `0`.
- vision_batch_size: Default value is `1`.

### vLLM Parameters

Parameter meanings can be found in the [vllm documentation](https://docs.vllm.ai/en/latest/models/engine_args.html).

- 🔥gpu_memory_utilization: Default value is `0.9`.
- 🔥tensor_parallel_size: Default is `1`.
- pipeline_parallel_size: Default is `1`.
- max_num_seqs: Default is `256`.
- 🔥max_model_len: Default is `None`.
- disable_custom_all_reduce: Default is `False`.
- enforce_eager: Whether vllm uses pytorch eager mode or establishes a cuda graph. Default is `False`. Setting to True can save memory but may affect efficiency.
- 🔥limit_mm_per_prompt: Controls vllm using multiple images, default is `None`. For example, use `--limit_mm_per_prompt '{"image": 10, "video": 5}'`.
- vllm_max_lora_rank: Default value is `16`. Parameters supported by vllm for LoRA.
- lora_modules: Used to support dynamic switching between multiple LoRAs, default is `[]`.

### Merge Parameters

- 🔥merge_lora: Whether to merge LoRA. This parameter supports LoRA, llmpro, longlora, default is False.
- safe_serialization: Whether to store safetensors, default is True.
- max_shard_size: Maximum size of a single storage file, default is '5GB'.

## Integration Parameters

### Training Parameters

Training parameters include the [basic parameters](#基本参数), [Seq2SeqTrainer parameters](#Seq2SeqTrainer参数), [tuner parameters](#tuner参数), [torchacc parameters](#torchacc参数), and also include the following部分:

- add_version: Add directory to output_dir with `'<version>-<timestamp>'` to prevent weight overwrite, default is True.
- resume_only_model: If resume_from_checkpoint, only resume model weights, default is False.
- check_model: Check local model files for corruption or modification and give a prompt, default is True. If in an offline environment, please set to False.
- loss_type: Type of loss, default uses the model's built-in loss function.
- num_labels: To be specified for classification models, representing the number of labels, default is None.
-
- packing: Whether to use packing, default is False.
- 🔥lazy_tokenize: Whether to use lazy_tokenize, default is False during LLM training, default is True during MLLM training.

- acc_strategy: Strategy for training accuracy, can be `sentence` or `token` level accuracy, default is `token`.
- max_new_tokens: Maximum generated token count when `predict_with_generate=True`, default 64.
- temperature: Temperature when `predict_with_generate=True`, default 0.
- optimizer: Custom optimizer name for plugin.
- metric: Custom metric name for plugin.

### RLHF Parameters

RLHF parameters include the [training parameters](#training parameters) and also contain the following:

- 🔥rlhf_type: Alignment algorithm type, supports `dpo`, `orpo`, `simpo`, `kto`, `cpo`.
- ref_model: Original comparison model in algorithms like DPO.
- ref_model_type: Same as model_type.
- ref_model_revision: Same as model_revision.

- 🔥beta: KL regularization term coefficient, default is `None`, i.e., for `simpo` algorithm default is `2.`, for other algorithms default is `0.1`. Refer to the [documentation](./Human-alignment.md) for specifics.
- label_smoothing: Whether to use DPO smoothing, default value is `0`, generally set between 0~0.5.
-
- 🔥rpo_alpha: Weight for adding sft_loss in DPO, default is `1`. The final loss is `KL_loss + rpo_alpha * sft_loss`.
-
- cpo_alpha: The coefficient of nll loss in CPO/SimPO loss, default is `1.`.
-
- simpo_gamma: Reward margin term in SimPO algorithm, recommended to set between 0.5-1.5 in the paper, default is `1.`.
-
- desirable_weight: Loss weight for desirable response in KTO algorithm $\lambda_D$, default is `1.`.
- undesirable_weight: Loss weight for undesirable response in KTO paper $\lambda_U$, default is `1.`.

### Inference Parameters

Inference parameters include the [basic parameters](#basic parameters), [merge parameters](#merge parameters), [vLLM parameters](#vllm parameters), [LMDeploy parameters](#LMDeploy parameters), and also contain the following:

- 🔥ckpt_dir: Path to the model checkpoint folder, default is None.
- 🔥infer_backend: Inference backend, supports 'pt', 'vllm', 'lmdeploy', default is 'pt'.
- 🔥max_batch_size: Batch size for pt backend, default is 1.
- result_path: Path to store inference results, default is None.
- val_dataset_sample: Number of samples from the inference dataset, default is None.

### Deployment Parameters

Deployment parameters inherit from the [inference parameters](#inference parameters).

- host: Service host, default is '0.0.0.0'.
- port: Port number, default is 8000.
- api_key: Access key required for access.
- owned_by: Default is `swift`.
- 🔥served_model_name: Model name for serving, defaults to the model's suffix.
- verbose: Print access logs, default is True.
- log_interval: Interval for printing tokens/s statistics, default is 20 seconds.
- max_logprobs: Maximum number of logprobs to return, default is 20.

### Evaluation Parameters

Evaluation parameters inherit from the [deployment parameters](#deployment parameters).

- 🔥eval_dataset: Evaluation dataset, refer to [Evaluation](./Evaluation.md).
- eval_limit: Number of samples for each evaluation set, default is None.
- eval_output_dir: Folder for storing evaluation results, default is 'eval_output'.
- temperature: Default is 0.
- verbose: This parameter is passed to DeployArguments during local evaluation, default is `False`.
- max_batch_size: Maximum batch size, default is 256 for text evaluation, 16 for multimodal.
- 🔥eval_url: Evaluation URL. Default is None, uses local deployment for evaluation.

### Export Parameters

Export parameters include the [basic parameters](#basic parameters) and [merge parameters](#merge parameters), and also contain the following:

- 🔥ckpt_dir: Checkpoint path, default is None.
- 🔥output_dir: Path for storing export results, default is None.

- 🔥quant_method: Options are 'gptq' and 'awq', default is None.
- quant_n_samples: Sampling size for the validation set in gptq/awq, default is 256.
- max_length: Max length for the calibration set, default value is 2048.
- quant_batch_size: Quantization batch size, default is 1.
- group_size: Group size for quantization, default is 128.

- 🔥push_to_hub: Whether to push to the hub, default is False.
- hub_model_id: Model ID for pushing, default is None.
- hub_private_repo: Whether it is a private repo, default is False.
- commit_message: Commit message, default is 'update files'.
