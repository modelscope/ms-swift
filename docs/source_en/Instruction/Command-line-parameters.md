# Command Line Parameters

The introduction to command line parameters will cover base arguments, atomic arguments, and integrated arguments, and specific model arguments. The final list of arguments used in the command line is the integration arguments. Integrated arguments inherit from basic arguments and some atomic arguments. Specific model arguments are designed for specific models and can be set using `--model_kwargs'` or the environment variable.

Hints:

- For passing a list in the command line, you can separate items with spaces. For example: `--dataset <dataset_path1> <dataset_path2>`.
- For passing a dict in the command line, use JSON format. For example: `--model_kwargs '{"fps_max_frames": 12}'`.
- Parameters marked with 🔥 are important. New users familiarizing themselves with ms-swift can focus on these command line parameters first.

## Base Arguments

- 🔥tuner_backend: Options are 'peft', 'unsloth'. Default is 'peft'.
- 🔥train_type: Options are: 'lora', 'full', 'longlora', 'adalora', 'llamapro', 'adapter', 'vera', 'boft', 'fourierft', 'reft'. Default is 'lora'.
- 🔥adapters: A list used to specify the id/path of the adapter. Default is `[]`.
- seed: Default is 42
- model_kwargs: Additional parameters specific to the model that can be passed in. This list of parameters will log a message during training and inference for reference. For example, `--model_kwargs '{"fps_max_frames": 12}'`.
- load_args: When specifying `--resume_from_checkpoint`, `--model`, or `--adapters`, it will read the `args.json` file saved in the checkpoint, assigning values to the default None `basic arguments` (excluding data and generation arguments) which can be overridden by manually passing them in. The default is True for inference and export, and False for training.
- load_data_args: If this parameter is set to True, additional data parameters will be read from args.json. The default is False.
- use_hf: Controls whether ModelScope or HuggingFace is used for model and dataset downloads, and model pushing. Defaults to False, meaning ModelScope is used.
- hub_token: Hub token. The hub token for ModelScope can be viewed [here](https://modelscope.cn/my/myaccesstoken).
- custom_register_path: A list of paths to `.py` files for custom registration of models, dialogue templates, and datasets. Defaults to `[]`.

### Model Arguments
- 🔥model: Model ID or local path to the model. If it's a custom model, please use it with `model_type` and `template`. The specific details can be referred to in the [Custom Model](../Customization/Custom-model.md).
- model_type: Model type. The same model architecture, template, and model loading process are defined as a model_type. The default is None, and it will be automatically selected based on the suffix of `--model` and the architectures attribute in config.json.
- model_revision: Model revision, default is None.
- task_type: Default is 'causal_lm' (if `--num_labels` is set, this parameter will be automatically set to 'seq_cls'). Options are 'causal_lm' and 'seq_cls'. Examples of seq_cls can be found [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/seq_cls).
- 🔥torch_dtype: Data type of model weights, supports `float16`, `bfloat16`, `float32`. The default is None, and it is read from the 'config.json' file.
- attn_impl: Type of attention, options are`flash_attn`, `sdpa`, `eager`. The default is sdpa; if not supported, eager is used.
  - Note: These three implementations may not all be supported, depending on the support of the corresponding model.
- num_labels: This parameter needs to be specified for classification models. It represents the number of labels and defaults to None.
- rope_scaling: Type of rope, supports `linear` and `dynamic`, should be used in conjunction with `max_length`. Default is None.
- device_map: Device map configuration used by the model, such as 'auto', 'cpu', JSON string, or the path of a JSON file. The default is None, automatically set based on the device and distributed training conditions.
- max_memory: When device_map is set to 'auto' or 'sequential', the model weights will be allocated to devices based on max_memory, for example: `--max_memory '{0: "20GB", 1: "20GB"}'`. The default value is None.
- local_repo_path: Some models depend on a GitHub repo when loading. To avoid network issues during `git clone`, a local repo can be used directly. This parameter needs to be passed with the path to the local repo, with the default being `None`.

### Data Arguments
- 🔥dataset: A list of dataset IDs or paths. Default is `[]`. The input format for each dataset is: `dataset ID or dataset path:sub-dataset#sampling size`, where sub-dataset and sampling data are optional. Local datasets support jsonl, csv, json, folders, etc. Open-source datasets can be cloned locally via git and used offline by passing the folder. For custom dataset formats, refer to [Custom Dataset](../Customization/Custom-dataset.md).
  - Sub-dataset: This parameter is effective only when the dataset is an ID or folder. If a subset was specified during registration, and only one sub-dataset exists, the registered sub-dataset is selected by default; otherwise, it defaults to 'default'. You can use `/` to select multiple sub-datasets, e.g., `<dataset_id>:subset1/subset2`. You can also use 'all' to select all sub-datasets, e.g., `<dataset_id>:all`.
  - Sampling Size: By default, the complete dataset is used. If the sampling size is less than the total number of data samples, samples are selected randomly without repetition. If the sampling size exceeds the total number of data samples, then `sampling size%total data samples` samples are randomly sampled additionally, and data samples are repetitively sampled `sampling size//total data samples` times.
- 🔥val_dataset: A list of validation set IDs or paths. Default is `[]`.
- 🔥split_dataset_ratio: Ratio for splitting the training set and validation set when val_dataset is not specified, default is 0.01. Set to 0 if no validation set split is needed.
- data_seed: Random seed for the dataset, default is 42.
- 🔥dataset_num_proc: Number of processes for dataset preprocessing, default is 1.
- 🔥streaming: Stream reading and processing of the dataset, default is False. It is typically set to True when handling large datasets.
- enable_cache: Use cache for dataset preprocessing, default is False.
- download_mode: Dataset download mode, including `reuse_dataset_if_exists` and `force_redownload`, default is reuse_dataset_if_exists.
- columns: Used for column mapping of the dataset to ensure that the dataset conforms to the format that AutoPreprocessor can handle. For more details, see [here](../Customization/Custom-dataset.md). You can pass in a JSON string, for example: `'{"text1": "query", "text2": "response"}'`, with the default being None.
- strict: If set to True, any row with an issue in the dataset will throw an error immediately, otherwise, erroneous data samples will be discarded. Default is False.
- remove_unused_columns: Whether to remove unused columns in the dataset, defaults to True.
- 🔥model_name: Only applicable to the self-cognition task and effective only on the `swift/self-cognition` dataset. It replaces the `{{NAME}}` placeholder in the dataset. Input the model's name in both Chinese and English, separated by a space, for example: `--model_name 小黄 'Xiao Huang'`. Default is None.
- 🔥model_author: Only applicable to the self-cognition task and effective only on the `swift/self-cognition` dataset. It replaces the `{{AUTHOR}}` placeholder in the dataset. Input the model author's name in both Chinese and English, separated by a space, for example: `--model_author '魔搭' 'ModelScope'`. Default is None.
- custom_dataset_info: The path to the JSON file for custom dataset registration. Refer to [Custom Dataset](../Customization/Custom-dataset.md). Default is `[]`.


### Template Arguments
- 🔥template: Type of dialogue template. Default is None, which automatically selects the corresponding model's template type.
- 🔥system: Custom system field, can take a string or txt file path as input. Default is None, uses the default system of the template.
- 🔥max_length: The maximum length of tokens for a single sample. Defaults to None, set to the maximum length of tokens supported by the model (max_model_len).
  - Note: In the cases of PPO, GRPO, and inference, max_length represents max_prompt_length.
- truncation_strategy: Strategy for handling single sample tokens that exceed `max_length`. Options are `delete`, `left`, and `right`, representing deletion, left-side truncation, and right-side truncation, respectively. The default is 'delete'.
- 🔥max_pixels: The maximum number of pixels (H*W) for input images to a multimodal model. Images exceeding this limit will be scaled. Default is None, meaning no maximum pixel limit.
- tools_prompt: Converts the tool list during agent training to the system format. Please refer to [Agent Training](./Agent-support.md). Options are 'react_en', 'react_zh', 'glm4', 'toolbench', 'qwen', with 'react_en' as the default.
- norm_bbox: Controls how to scale bounding boxes (bbox). Options are 'norm1000' and 'none'. 'norm1000' represents scaling bbox coordinates to one-thousandths, and 'none' means no scaling. Default is None, automatically selected based on the model.
- padding_side: Padding side when `batch_size>=2` during training. Options are 'left' and 'right', with 'right' as the default. (For inference with batch_size>=2, only left padding is applied.)
- loss_scale: Setting for the loss weight of training tokens. Default is `'default'`, meaning all responses (including history) are calculated with a cross-entropy loss of 1. Options are 'default', 'last_round', 'all', and agent-specific loss scales: 'react', 'agentflan', 'alpha_umi', and 'qwen'. 'last_round' means calculating only the loss of the last round's response, and 'all' calculates the loss for all tokens. For agent parts, see [Pluginization](../Customization/Pluginization.md) and [Agent Training](./Agent-support.md).
- sequence_parallel_size: Number of sequence parallels, default is 1. Refer to [example](https://github.com/modelscope/ms-swift/tree/main/examples/train/sequence_parallel/train.sh).
- use_chat_template: Use chat template or generation template, default is `True`. `swift pt` is automatically set to the generation template.
- template_backend: Selection of the template backend. Options are 'swift' and 'jinja', with 'swift' as the default. If using jinja, it applies transformer's `apply_chat_template`.
  - Note: The jinja template backend supports only inference, not training.

### Generation Arguments

Refer to the [generation_config](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig) documentation.

- 🔥max_new_tokens: The maximum number of new tokens generated during inference. Defaults to None, meaning unlimited.
- temperature: The temperature parameter. Defaults to None and is read from generation_config.json.
  - Note: The do_sample parameter has been removed in this version. Set the temperature to 0 to achieve the same effect.
- top_k: The top_k parameter, defaults to None. It is read from generation_config.json.
- top_p: The top_p parameter, defaults to None. It is read from generation_config.json.
- repetition_penalty: The repetition penalty. Defaults to None and is read from generation_config.json.
- num_beams: The number of beams reserved for parallel beam search, default is 1.
- 🔥stream: Stream output, default is `False`.
- stop_words: Additional stop words beyond eos_token, default is`[]`.
  - Note: eos_token will be removed in the output response, whereas additional stop words will be retained in the output.
- logprobs: Whether to output logprobs, default is False.
- top_logprobs: The number of top_logprobs to output, defaults to None.


### Quantization Arguments

The following are the parameters for quantization when loading a model. For detailed meanings, you can refer to the [quantization](https://huggingface.co/docs/transformers/main/en/main_classes/quantization) documentation. Note that this does not include `gptq` and `awq` quantization parameters involved in `swift export`.

- 🔥quant_method: The quantization method used when loading the model. Options are `bnb`, `hqq`, `eetq`.
- 🔥quant_bits: Number of bits for quantization, default is None.
- hqq_axis: HQQ quantization axis, default is None.
- bnb_4bit_compute_dtype: The computation type for bnb quantization. Options are `float16`, `bfloat16`, `float32`. The default is None, which sets it to `torch_dtype`.
- bnb_4bit_quant_type: BNB quantization type, supports `fp4` and `nf4`, default is `nf4`.
- bnb_4bit_use_double_quant: Whether to use double quantization, default is `True`.
- bnb_4bit_quant_storage: BNB quantization storage type, default is None.

## Atomic Arguments

### Seq2SeqTrainer Arguments

This parameter list inherits from transformers `Seq2SeqTrainingArguments`, with default values overridden by ms-swift. For unlisted items, refer to the [HF Official Documentation](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Seq2SeqTrainingArguments).

- 🔥output_dir: Defaults to None, set as `output/<model_name>`.
- 🔥gradient_checkpointing: Whether to use gradient checkpointing, default is True.
- 🔥deepspeed: Defaults to None. It can be set to 'zero0', 'zero1', 'zero2', 'zero3', 'zero2_offload', 'zero3_offload' to use the built-in deepspeed configuration file of ms-swift.
- 🔥per_device_train_batch_size: Default is 1.
- 🔥per_device_eval_batch_size: Default is 1.
- weight_decay: Weight decay coefficient, default value is 0.1.
- 🔥learning_rate: Learning rate, defaults to 1e-5 for full parameters, and 1e-4 for LoRA and other tuners.
- lr_scheduler_type: Type of lr_scheduler, defaults to 'cosine'.
- lr_scheduler_kwargs: Other parameters for the lr_scheduler, defaults to None.
- 🔥gradient_checkpointing_kwargs: Parameters for `torch.utils.checkpoint`. For example, set as `--gradient_checkpointing_kwargs '{"use_reentrant": false}'`. Defaults to None.
- report_to: Default value is `tensorboard`. You can also specify `--report_to tensorboard wandb swanlab` or `--report_to all`.
- logging_first_step: Whether to log the first step, defaults to True.
- logging_steps: Interval for logging, defaults to 5.
- predict_with_generate: Whether to use generative method during validation, default is False.
- metric_for_best_model: Default is None, which means that when predict_with_generate is set to False, it is set to 'loss'; otherwise, it is set to 'rouge-l' (during PPO training, the default value is not set; in GRPO training, it is set to 'reward').
- greater_is_better: Defaults to None, which sets it to False when `metric_for_best_model` contains 'loss', otherwise sets to True.

Other important parameters:
- 🔥num_train_epochs: Number of training epochs, default is 3.
- 🔥gradient_accumulation_steps: Gradient accumulation, default is 1.
- 🔥save_strategy: Strategy for saving the model, options include 'no', 'steps', 'epoch'. Default is 'steps'.
- 🔥save_steps: Default is 500.
- 🔥eval_strategy: Evaluation strategy. Default is None and follows the strategy of `save_strategy`.
- 🔥eval_steps: Default is None. If there is an evaluation dataset, it follows the strategy of `save_steps`.
- 🔥save_total_limit: Maximum number of checkpoints to save. Older checkpoints will be deleted. Default is None, saving all checkpoints.
- max_steps: Maximum number of training steps. Should be set when the dataset is streamed. Default is -1.
- 🔥warmup_ratio: Default is 0.
- save_on_each_node: Default is False. Should be considered in multi-node training.
- save_only_model: Whether to save only the model weights without including optimizer state, random seed state, etc. Default is False.
- 🔥resume_from_checkpoint: Parameter for resuming training from a checkpoint, pass the checkpoint path. Default is None.
  - Note: `resume_from_checkpoint` will load the model weights, optimizer weights, and random seed, and continue training from the last trained steps. You can specify `--resume_only_model` to load only the model weights.
- 🔥ddp_backend: Default is None, options include "nccl", "gloo", "mpi", "ccl", "hccl", "cncl", "mccl".
- 🔥ddp_find_unused_parameters: Default is None.
- 🔥dataloader_num_workers: Default is 0.
- 🔥neftune_noise_alpha: Coefficient of noise added by neftune, default is 0. Usually can be set to 5, 10, 15.
- average_tokens_across_devices: Whether to average the number of tokens across devices. If set to True, `num_tokens_in_batch` will be synchronized using all_reduce for accurate loss calculation. Default is False.
- max_grad_norm: Gradient clipping. Default is 1.
- push_to_hub: Push checkpoint to hub. Default is False.
- hub_model_id: Default is None.
- hub_private_repo: Default is False.

### Tuner Arguments

- 🔥freeze_llm: This parameter is only effective for multimodal models and can be used for full parameter training and LoRA, but with different meanings. In full parameter training, setting freeze_llm to True will freeze some of the LLM weights. In LoRA training, if `target_modules` is set to 'all-linear', setting freeze_llm to True will prevent adding LoRA modules to the LLM part. The default is False.
- 🔥freeze_vit: This parameter is only effective for multimodal models and can be used for full parameter training and LoRA, with similar meanings as `freeze_llm`. The default is True.
- 🔥freeze_aligner: This parameter is only effective for multimodal models and can be used for full parameter training and LoRA, with similar meanings as `freeze_llm`. The default is True.
- 🔥target_modules: Specifies LoRA modules, with a default of `all-linear`. Its behavior differs in LLM and multimodal LLM. For LLM, it automatically finds all linear modules except lm_head and adds a tuner. For multimodal LLM, by default, it only adds a tuner to the LLM part, and this behavior can be controlled by `freeze_llm`, `freeze_vit`, and `freeze_aligner`. This parameter is not limited to LoRA and can be used for other tuners.
- 🔥target_regex: Specifies a regex expression for LoRA modules, with a default of `None`. If this value is provided, the target_modules parameter becomes ineffective. This parameter is not limited to LoRA and can be used for other tuners.
- init_weights: Specifies the method for initializing weights. LoRA can specify `true`, `false`, `gaussian`, `pissa`, `pissa_niter_[number of iters]`. Bone can specify `true`, `false`, `bat`. The default is `true`.
- 🔥modules_to_save: After attaching a tuner, explicitly specifies additional original model modules to participate in training and storage. The default is `[]`. This parameter is not limited to LoRA and can be used for other tuners.

#### Full Arguments

- freeze_parameters: Prefix of the parameters to be frozen, default is `[]`.
- freeze_parameters_ratio: Ratio of parameters to freeze from bottom to top, default is 0. It can be set to 1 to freeze all parameters, and trainable parameters can be set in conjunction with this.
- trainable_parameters: Prefix of additional trainable parameters, default is `[]`.
  - Note: `trainable_parameters` takes precedence over `freeze_parameters` and `freeze_parameters_ratio`. When full parameter training is specified, all modules are set to trainable, then some parameters are frozen according to `freeze_parameters` and `freeze_parameters_ratio`, and finally, some parameters are reopened for training according to `trainable_parameters`.

#### LoRA

- 🔥lora_rank: Default is `8`.
- 🔥lora_alpha: Default is `32`.
- lora_dropout: Default is `0.05`.
- lora_bias: Defaults to `'none'`. Possible values are 'none', 'all'. If you want to make all biases trainable, you can set it to `'all'`.
- lora_dtype: Specifies the dtype type for the LoRA modules. Supported types are 'float16', 'bfloat16', 'float32'. The default is None, which follows the original model type.
- 🔥use_dora: Defaults to `False`, indicating whether to use `DoRA`.
- use_rslora: Defaults to `False`, indicating whether to use `RS-LoRA`.
- 🔥lorap_lr_ratio: LoRA+ parameter, default value `None`, recommended values `10~16`. Specify this parameter when using LoRA to enable LoRA+.


##### LoRA-GA
- lora_ga_batch_size: The default value is `2`. The batch size used for estimating gradients during initialization in LoRA-GA.
- lora_ga_iters: The default value is `2`. The number of iterations for estimating gradients during initialization in LoRA-GA.
- lora_ga_max_length: The default value is `1024`. The maximum input length for estimating gradients during initialization in LoRA-GA.
- lora_ga_direction: The default value is `ArB2r`. The initial direction used for gradient estimation during initialization in LoRA-GA. Allowed values are: `ArBr`, `A2rBr`, `ArB2r`, and `random`.
- lora_ga_scale: The default value is `stable`. The scaling method for initialization in LoRA-GA. Allowed values are: `gd`, `unit`, `stable`, and `weightS`.
- lora_ga_stable_gamma: The default value is `16`. The gamma value when choosing `stable` scaling for initialization.

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

### LMDeploy Arguments

Parameter meanings can be found in the [lmdeploy documentation](https://lmdeploy.readthedocs.io/en/latest/api/pipeline.html#turbomindengineconfig).

- 🔥tp: tensor parallelism degree. Default is `1`.
- session_len: Default is `None`.
- cache_max_entry_count: Default is `0.8`.
- quant_policy: Default is `0`.
- vision_batch_size: Default is `1`.

### vLLM Arguments

Parameter meanings can be found in the [vllm documentation](https://docs.vllm.ai/en/latest/serving/engine_args.html).

- 🔥gpu_memory_utilization: Default value is `0.9`.
- 🔥tensor_parallel_size: Default is `1`.
- pipeline_parallel_size: Default is `1`.
- max_num_seqs: Default is `256`.
- 🔥max_model_len: Default is `None`.
- disable_custom_all_reduce: Default is `False`.
- enforce_eager: Determines whether vllm uses PyTorch eager mode or constructs a CUDA graph, default is `False`. Setting it to True can save memory but may affect efficiency.
- 🔥limit_mm_per_prompt: Controls the use of multiple media in vllm, default is `None`. For example, you can pass in `--limit_mm_per_prompt '{"image": 5, "video": 2}'`.
- vllm_max_lora_rank: Default is `16`. This is the parameter supported by vllm for lora.
- enable_prefix_caching: Enable the automatic prefix caching of vllm to save processing time for querying repeated prefixes. The default is `False`.

### Merge Arguments

- 🔥merge_lora: Indicates whether to merge lora; this parameter supports lora, llamapro, and longlora, default is `False`. Example parameters [here](https://github.com/modelscope/ms-swift/blob/main/examples/export/merge_lora.sh).
- safe_serialization: Whether to store safetensors, default is True.
- max_shard_size: Maximum size of a single storage file, default is '5GB'.

## Integration Arguments

### Training Arguments

Training arguments include the [base arguments](#base-arguments), [Seq2SeqTrainer arguments](#Seq2SeqTrainer-arguments), [tuner arguments](#tuner-arguments), and also include the following parts:

- add_version: Add directory to output_dir with `'<version>-<timestamp>'` to prevent weight overwrite, default is True.
- resume_only_model: Defaults to False. If set to True in conjunction with `resume_from_checkpoint`, only the model weights are resumed.
- check_model: Check local model files for corruption or modification and give a prompt, default is True. If in an offline environment, please set to False.
- 🔥create_checkpoint_symlink: Creates additional checkpoint symlinks to facilitate writing automated training scripts. The symlink paths for `best_model` and `last_model` are `f'{output_dir}/best'` and `f'{output_dir}/last'` respectively.
- external_plugins: A list of external plugin py files which will be registered into the plugin mappings，please check [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo/run_external_rm.sh)
- loss_type: Type of loss. Defaults to None, which uses the model's built-in loss function.
- packing: Whether to use sequence packing, defaults to False.
- 🔥lazy_tokenize: Whether to use lazy tokenization. If set to False, all dataset samples are tokenized before training (for multimodal models, this includes reading images from disk). This parameter defaults to False for LLM training, and True for MLLM training, to save memory.
- acc_strategy: Strategy for calculating accuracy during training and validation. Options are `seq`-level and `token`-level accuracy, with `token` as the default.
- max_new_tokens: Generation parameter override. The maximum number of tokens to generate when `predict_with_generate=True`, defaulting to 64.
- temperature: Generation parameter override. The temperature setting when `predict_with_generate=True`, defaulting to 0.
- optimizer: Custom optimizer name for the plugin, defaults to None.
- metric: Custom metric name for the plugin. Defaults to None, with the default set to 'acc' when `predict_with_generate=False` and 'nlg' when `predict_with_generate=True`.

### RLHF Arguments

RLHF arguments inherit from the [training arguments](#training-arguments).

- 🔥rlhf_type: Type of human alignment algorithm, supporting `dpo`, `orpo`, `simpo`, `kto`, `cpo`, `rm`, and `ppo`. Default is 'dpo'.
- ref_model: Required for full parameter training when using the dpo, kto, or ppo algorithms. Default is None.
- ref_model_type: Same as model_type. Default is None.
- ref_model_revision: Same as model_revision. Default is None.
- 🔥beta: Coefficient for the KL regularization term. Default is `None`, meaning `simpo` algorithm defaults to `2.`, `grpo` algorithm defaults to `0.04`, and other algorithms default to `0.1`. For more details, refer to the [documentation](./RLHF.md).
- label_smoothing: Whether to use DPO smoothing, default value is `0`.
- 🔥rpo_alpha: The weight of sft_loss added to DPO, default is `1`. The final loss is `KL_loss + rpo_alpha * sft_loss`.
- cpo_alpha: Coefficient for nll loss in CPO/SimPO loss, default is `1.`.
- simpo_gamma: Reward margin term in the SimPO algorithm, with a paper-suggested setting of 0.5-1.5, default is `1.`.
- desirable_weight: Loss weight $\lambda_D$ for desirable response in the KTO algorithm, default is `1.`.
- undesirable_weight: Loss weight $\lambda_U$ for undesirable response in the KTO algorithm, default is `1.`.
- loss_scale: Override template arguments, default is 'last_round'.
- temperature: Default is 0.9; this parameter will be used in PPO and GRPO.

#### Reward Model Parameters

The reward model parameters will be used in PPO and GRPO.

- reward_model: Default is None.
- reward_adapters: Default is `[]`.
- reward_model_type: Default is None.
- reward_model_revision: Default is None.

#### PPO Arguments

The meanings of the following parameters can be referenced [here](https://huggingface.co/docs/trl/main/ppo_trainer):

- num_ppo_epochs: Defaults to 4
- whiten_rewards: Defaults to False
- kl_coef: Defaults to 0.05
- cliprange: Defaults to 0.2
- vf_coef: Defaults to 0.1
- cliprange_value: Defaults to 0.2
- gamma: Defaults to 1.0
- lam: Defaults to 0.95
- num_mini_batches: Defaults to 1
- local_rollout_forward_batch_size: Defaults to 64
- num_sample_generations: Defaults to 10
- response_length: Defaults to 512
- missing_eos_penalty: Defaults to None


#### GRPO Arguments
- num_generations: The G value in the GRPO algorithm, default is 8.
- max_completion_length: The maximum generation length in the GRPO algorithm, default is 512.
- ds3_gather_for_generation: This parameter applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation, improving generation speed. However, disabling this option allows training models that exceed the VRAM capacity of a single GPU, albeit at the cost of slower generation. Disabling this option is not compatible with vLLM generation. The default is True.
- reward_funcs: Reward functions in the GRPO algorithm; options include `accuracy`,`format`,`cosine` and `repetition`, as seen in `swift/plugin/orm.py`. You can also customize your own reward functions in the plugin. Default is `[]`.
- reward_weights: Weights for each reward function. Must match the number of reward functions. If `None`, all rewards are weighted equally with weight `1.0`.
  - Note: If `--reward_model` is included in GRPO training, it is added to the end of the reward functions.
- log_completions: Whether to log the model-generated content during training, to be used in conjunction with `--report_to wandb`, default is False.
  - Note: If `--report_to wandb` is not set, a `completions.jsonl` will be created in the checkpoint to store the generated content.
- use_vllm: Whether to use vLLM as the infer_backend for GRPO generation, default is False.
- num_infer_workers: The number of inference workers per node. This setting is only effective when using vLLM or lmdeploy.
- vllm_device: Configures the devices for deploying vLLM. You can set it to auto, which will allocate the last few GPUs based on the value of num_infer_workers. Alternatively, specify a number of devices equal to num_infer_workers. For example: --vllm_device cuda:1 cuda:2.
- vllm_gpu_memory_utilization: vLLM passthrough parameter, default is 0.9.
- vllm_max_model_len: vLLM passthrough parameter, default is None.
- vllm_max_num_seqs: vLLM passthrough parameter, default is 256.
- vllm_enforce_eager: vLLM passthrough parameter, default is False.
- vllm_limit_mm_per_prompt: vLLM passthrough parameter, default is None.
- vllm_enable_prefix_caching: vLLM passthrough parameter, default is True.
- top_k: Default is 50.
- top_p: Default is 0.9.
- repetition_penalty: Repetition penalty term. Default is 1.
- num_iterations: number of iterations per batch. Default is 1.
- epsilon: epsilon value for clipping. Default is 0.2
- async_generate: Use async rollout to improve train speed，default `false`

cosine reward function arguments
- `cosine_min_len_value_wrong` (default: 0.0): Reward value corresponding to the minimum length when the answer is incorrect. Default is 0.0
- `cosine_max_len_value_wrong` (default: -0.5): Reward value corresponding to the maximum length when the answer is incorrect. Default is -0.5
- `cosine_min_len_value_correct` (default: 1.0): Reward value corresponding to the minimum length when the answer is correct. Default is 1.0
- `cosine_max_len_value_correct` (default: 0.5): Reward value corresponding to the maximum length when the answer is correct. Default is 0.5
- `cosine_max_len` (default value equal to the model's maximum generation capacity): Maximum length limit for generated text. Default value equal to max_completion_length

repetition penalty function arguments

- `repetition_n_grams` (default: 3): Size of the n-gram used to detect repetition.
- `repetition_max_penalty` (default: -1.0): Maximum penalty value, which controls the intensity of the penalty.

#### SWANLAB

- **swanlab_token**: SwanLab's API key
- **swanlab_project**: SwanLab's project, which needs to be created in advance on the page: [https://swanlab.cn/space/~](https://swanlab.cn/space/~)
- **swanlab_workspace**: Defaults to `None`, will use the username associated with the API key
- **swanlab_exp_name**: Experiment name, can be left empty. If empty, the value of `--output_dir` will be used by default
- **swanlab_mode**: Optional values are `cloud` and `local`, representing cloud mode or local mode


### Inference Arguments

Inference arguments include the [base arguments](#base-arguments), [merge arguments](#merge-arguments), [vLLM arguments](#vllm-arguments), [LMDeploy arguments](#LMDeploy-arguments), and also contain the following:

- 🔥infer_backend: Inference acceleration backend, supporting three inference engines: 'pt', 'vllm', and 'lmdeploy'. The default is 'pt'.
- 🔥max_batch_size: Effective when infer_backend is set to 'pt'; used for batch inference, with a default value of 1.
- ddp_backend: Effective when infer_backend is set to 'pt'; used to specify the distributed backend for multi-GPU inference. The default is None, which means automatic selection. For an example of multi-GPU inference, you can refer [here](https://github.com/modelscope/ms-swift/tree/main/examples/infer/pt).
- 🔥result_path: Path to store inference results (jsonl). The default is None, meaning results are saved in the checkpoint directory (with args.json file) or './result' directory. The final storage path will be printed in the command line.
- metric: Evaluate the results of the inference, currently supporting 'acc' and 'rouge'. The default is None, meaning no evaluation is performed.
- val_dataset_sample: Number of samples from the inference dataset, default is None.

### Deployment Arguments

Deployment Arguments inherit from the [inference arguments](#inference-arguments).

- host: Service host, default is '0.0.0.0'.
- port: Port number, default is 8000.
- api_key: The API key required for access; the default is None.
- owned_by: Default is `swift`.
- 🔥served_model_name: Model name for serving, defaults to the model's suffix.
- verbose: Print detailed logs, with a default value of True.
- log_interval: Interval for printing tokens/s statistics, default is 20 seconds. If set to -1, it will not be printed.
- max_logprobs: Maximum number of logprobs returned to the client, with a default value of 20.

### Web-UI Arguments
- server_name: Host for the web UI, default is '0.0.0.0'.
- server_port: Port for the web UI, default is 7860.
- share: Default is False.
- lang: Language for the web UI, options are 'zh', 'en'. Default is 'zh'.


### App Arguments
App parameters inherit from [deployment arguments](#deployment-arguments) and [Web-UI Arguments](#web-ui-arguments).

- base_url: The base URL for model deployment, for example, `http://localhost:8000/v1`. The default value is `None`, which means using local deployment.
- studio_title: Title of the studio. Default is None, set to the model name.
- is_multimodal: Whether to launch the multimodal version of the app. Defaults to None, automatically determined based on the model; if it cannot be determined, set to False.
- lang: Overrides the Web-UI Arguments, default is 'en'.

### Evaluation Arguments

Evaluation Arguments inherit from the [deployment arguments](#deployment-arguments).

- 🔥eval_backend: Evaluation backend, default is 'Native', but can also be specified as 'OpenCompass' or 'VLMEvalKit'
- 🔥eval_dataset: Evaluation dataset, refer to [Evaluation documentation](./Evaluation.md).
- eval_limit: Number of samples for each evaluation set, default is None.
- eval_output_dir: Folder for storing evaluation results, default is 'eval_output'.
- 🔥local_dataset: Some evaluation sets, such as `CMB`, cannot be directly used and require downloading additional data packages. Setting this parameter to `true` will automatically download the full data package, create a `data` folder in the current directory, and start the evaluation. The data package will only be downloaded once and will be cached for future use. This parameter defaults to `false`.
  - Note: By default, the evaluation will use datasets from `~/.cache/opencompass`. Specifying this parameter will directly use the data folder in the current directory.
- temperature: Overrides the generation arguments, with a default value of 0.
- verbose: This parameter is passed into DeployArguments when setting up local deployment and evaluation, and defaults to `False`.
- eval_num_proc: Maximum number of concurrent clients during evaluation, default is 16.
- 🔥eval_url: The evaluation URL, for example, `http://localhost:8000/v1`. Examples can be found [here](https://github.com/modelscope/ms-swift/tree/main/examples/eval/eval_url). The default value is None, which means using local deployment for evaluation.


### Export Arguments

Export Arguments include the [basic arguments](#base-arguments) and [merge arguments](#merge-arguments), and also contain the following:

- 🔥output_dir: The path for storing exported results. The default value is None, and an appropriate suffix path will be automatically set.
- exist_ok: If output_dir exists, do not raise an exception and overwrite the contents. The default value is False.
- 🔥quant_method: Options are 'gptq', 'awq', or 'bnb', with the default being None. Examples can be found [here](https://github.com/modelscope/ms-swift/tree/main/examples/export/quantize).
- quant_n_samples: The number of samples for the validation set used by gptq/awq, with a default of 256.
- max_length: Max length for the calibration set, default value is 2048.
- quant_batch_size: Quantization batch size, default is 1.
- group_size: Group size for quantization, default is 128.
- 🔥push_to_hub: Whether to push to the hub, with the default being False. Examples can be found [here](https://github.com/modelscope/ms-swift/blob/main/examples/export/push_to_hub.sh).
- hub_model_id: Model ID for pushing, default is None.
- hub_private_repo: Whether it is a private repo, default is False.
- commit_message: Commit message, default is 'update files'.

### Sampling Parameters

- prm_model: The type of process reward model. It can be a model ID (triggered using `pt`) or a `prm` key defined in a plugin (for custom inference processes).
- orm_model: The type of outcome reward model, typically a wildcard or test case, usually defined in a plugin.
- sampler_type: The type of sampling. Currently supports `sample` (using `do_sample` method). Future support will include `mcts` and `dvts`.
- sampler_engine: Supports `pt`, `lmdeploy`, `vllm`, `no`. Defaults to `pt`. Specifies the inference engine for the sampling model.
- output_dir: The output directory. Defaults to `sample_output`.
- output_file: The name of the output file. Defaults to `None`, which uses a timestamp as the filename. When provided, only the filename should be passed without the directory, and only JSONL format is supported.
- override_exist_file: Whether to overwrite if `output_file` already exists.
- num_sampling_per_gpu_batch_size: The batch size for each sampling operation.
- num_sampling_per_gpu_batches: The total number of batches to sample.
- n_best_to_keep: The number of best sequences to return.
- data_range: The partition of the dataset being processed for this sampling operation. The format should be `2 3`, meaning the dataset is divided into 3 parts, and this instance is processing the 3rd partition (this implies that typically three `swift sample` processes are running in parallel).
- temperature: Defaults to `1.0`.
- prm_threshold: The PRM threshold. Results below this value will be filtered out. The default value is `0`.
- easy_query_threshold: For each query, if the ORM evaluation is correct for more than this proportion of all samples, the query will be discarded to prevent overly simple queries from appearing in the results. Defaults to `None`, meaning no filtering is applied.
- engine_kwargs: Additional parameters for the `sampler_engine`, passed as a JSON string, for example, `{"cache_max_entry_count":0.7}`.
- num_return_sequences: The number of original sequences returned by sampling. Defaults to `64`. This parameter is effective for `sample` sampling.
- cache_files: To avoid loading both `prm` and `generator` simultaneously and causing GPU memory OOM, sampling can be done in two steps. In the first step, set `prm` and `orm` to `None`, and all results will be output to a file. In the second run, set `sampler_engine` to `no` and pass `--cache_files` with the output file from the first sampling. This will use the results from the first run for `prm` and `orm` evaluation and output the final results.
  - Note: When using `cache_files`, the `--dataset` still needs to be provided because the ID for `cache_files` is calculated using the MD5 of the original data. Both pieces of information need to be used together.

#### MCTS
- rollout_depth: The maximum depth during rollouts, default is `5`.
- rollout_start_depth: The depth at which rollouts begin; nodes below this depth will only undergo expand operations, default is `3`.
- max_iterations: The maximum number of iterations for MCTS, default is `100`.
- process_reward_rate: The proportion of process reward used in calculating value during selection, default is `0.0`, meaning PRM is not used.
- exploration_rate: A parameter in the UCT algorithm that balances exploration; a higher value gives more weight to nodes with fewer explorations, default is `0.5`.
- api_key: Required when using the client as an inference engine, default is `EMPTY`.
- base_url: Required when using the client as an inference engine, default is 'https://dashscope.aliyuncs.com/compatible-mode/v1'.

## Specific Model Arguments

Specific model arguments can be set using `--model_kwargs` or environment variables, for example: `--model_kwargs '{"fps_max_frames": 12}'` or `FPS_MAX_FRAMES=12`.

### qwen2_vl, qvq, qwen2_5_vl
For the meaning of the arguments, please refer to [here](https://github.com/QwenLM/Qwen2-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L24)

- IMAGE_FACTOR: Default is 28
- MIN_PIXELS: Default is `4 * 28 * 28`
- 🔥MAX_PIXELS: Default is `16384 * 28 * 28`, refer to [here](https://github.com/modelscope/ms-swift/blob/main/examples/train/multimodal/ocr.sh#L3)
- MAX_RATIO: Default is 200
- VIDEO_MIN_PIXELS: Default is `128 * 28 * 28`
- 🔥VIDEO_MAX_PIXELS: Default is `768 * 28 * 28`, refer to [here](https://github.com/modelscope/ms-swift/blob/main/examples/train/multimodal/video.sh#L7)
- VIDEO_TOTAL_PIXELS: Default is `24576 * 28 * 28`
- FRAME_FACTOR: Default is 2
- FPS: Default is 2.0
- FPS_MIN_FRAMES: Default is 4
- 🔥FPS_MAX_FRAMES: Default is 768, refer to [here](https://github.com/modelscope/ms-swift/blob/main/examples/train/multimodal/video.sh#L8)

### qwen2_audio
- SAMPLING_RATE: Default is 16000

### internvl, internvl_phi3
For the meaning of the arguments, please refer to [here](https://modelscope.cn/models/OpenGVLab/Mini-InternVL-Chat-2B-V1-5)
- MAX_NUM: Default is 12
- INPUT_SIZE: Default is 448

### internvl2, internvl2_phi3, internvl2_5
For the meaning of the arguments, please refer to [here](https://modelscope.cn/models/OpenGVLab/InternVL2_5-2B)
- MAX_NUM: Default is 12
- INPUT_SIZE: Default is 448
- VIDEO_MAX_NUM: Default is 1, which is the MAX_NUM for videos
- VIDEO_SEGMENTS: Default is 8

### minicpmv2_6, minicpmo2_6
- MAX_SLICE_NUMS: Default is 9, refer to [here](https://modelscope.cn/models/OpenBMB/MiniCPM-V-2_6/file/view/master?fileName=config.json&status=1)
- VIDEO_MAX_SLICE_NUMS: Default is 1, which is the MAX_SLICE_NUMS for videos, refer to [here](https://modelscope.cn/models/OpenBMB/MiniCPM-V-2_6)
- MAX_NUM_FRAMES: Default is 64, refer to [here](https://modelscope.cn/models/OpenBMB/MiniCPM-V-2_6)

### minicpmo2_6
- INIT_TTS: Default is False
- INIT_AUDIO: Default is False

### ovis1_6, ovis2
- MAX_PARTITION: Default is 9, refer to [here](https://github.com/AIDC-AI/Ovis/blob/d248e34d755a95d24315c40e2489750a869c5dbc/ovis/model/modeling_ovis.py#L312)

### mplug_owl3, mplug_owl3_241101
- MAX_NUM_FRAMES: Default is 16, refer to [here](https://modelscope.cn/models/iic/mPLUG-Owl3-7B-240728)

### xcomposer2_4khd
- HD_NUM: Default is 55, refer to [here](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-xcomposer2-4khd-7b)

### xcomposer2_5
- HD_NUM: Default is 24 when the number of images is 1. Greater than 1, the default is 6. Refer to [here](https://modelscope.cn/models/AI-ModelScope/internlm-xcomposer2d5-7b/file/view/master?fileName=modeling_internlm_xcomposer2.py&status=1#L254)

### video_cogvlm2
- NUM_FRAMES: Default is 24, refer to [here](https://github.com/THUDM/CogVLM2/blob/main/video_demo/inference.py#L22)

### phi3_vision
- NUM_CROPS: Default is 4, refer to [here](https://modelscope.cn/models/LLM-Research/Phi-3.5-vision-instruct)

### llama3_1_omni
- N_MELS: Default is 128, refer to [here](https://github.com/ictnlp/LLaMA-Omni/blob/544d0ff3de8817fdcbc5192941a11cf4a72cbf2b/omni_speech/infer/infer.py#L57)

### video_llava
- NUM_FRAMES: Default is 16


## Other Environment Variables

- CUDA_VISIBLE_DEVICES: Controls which GPU to use. By default, all GPUs are used.
- ASCEND_RT_VISIBLE_DEVICES: Controls which NPU (effective for ASCEND cards) are used. By default, all NPUs are used.
- MODELSCOPE_CACHE: Controls the cache path.
- NPROC_PER_NODE: Pass-through for the `--nproc_per_node` parameter in torchrun. The default is 1. If the `NPROC_PER_NODE` or `NNODES` environment variables are set, torchrun is used to start training or inference.
- MASTER_PORT: Pass-through for the `--master_port` parameter in torchrun. The default is 29500.
- MASTER_ADDR: Pass-through for the `--master_addr` parameter in torchrun.
- NNODES: Pass-through for the `--nnodes` parameter in torchrun.
- NODE_RANK: Pass-through for the `--node_rank` parameter in torchrun.
