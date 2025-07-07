# Command Line Parameters

The introduction to command line parameters will cover base arguments, atomic arguments, and integrated arguments, and specific model arguments. The final list of arguments used in the command line is the integration arguments. Integrated arguments inherit from basic arguments and some atomic arguments. Specific model arguments are designed for specific models and can be set using `--model_kwargs'` or the environment variable. The introduction to the Megatron-SWIFT command-line arguments can be found in the [Megatron-SWIFT Training Documentation](./Megatron-SWIFT-Training.md).

Hints:

- For passing a list in the command line, you can separate items with spaces. For example: `--dataset <dataset_path1> <dataset_path2>`.
- For passing a dict in the command line, use JSON format. For example: `--model_kwargs '{"fps_max_frames": 12}'`.
- Parameters marked with 🔥 are important. New users familiarizing themselves with ms-swift can focus on these command line parameters first.

## Base Arguments

- 🔥tuner_backend: Options are 'peft', 'unsloth'. Default is 'peft'.
- 🔥train_type: Options are: 'lora', 'full', 'longlora', 'adalora', 'llamapro', 'adapter', 'vera', 'boft', 'fourierft', 'reft'. Default is 'lora'.
- 🔥adapters: A list used to specify the id/path of the adapter. Default is `[]`.
- external_plugins: A list of external plugin py files which will be registered into the plugin mappings，please check [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo/plugin/run_external_reward_func.sh). Default is `[]`.
- seed: Default is 42.
- model_kwargs: Additional parameters specific to the model that can be passed in. This list of parameters will log a message during training and inference for reference. For example, `--model_kwargs '{"fps_max_frames": 12}'`. Default is None.
- load_args: When `--resume_from_checkpoint`, `--model`, or `--adapters` is specified, the `args.json` file from the saved checkpoint will be read. The keys to be read can be found in [base_args.py](https://github.com/modelscope/ms-swift/blob/main/swift/llm/argument/base_args/base_args.py). By default, this is set to `True` during inference and export, and `False` during training.
- load_data_args: If this parameter is set to True, additional data parameters will be read from args.json. The default is False.
- use_hf: Controls whether ModelScope or HuggingFace is used for model and dataset downloads, and model pushing. Defaults to False, meaning ModelScope is used.
- hub_token: Hub token. The hub token for ModelScope can be viewed [here](https://modelscope.cn/my/myaccesstoken). Default is None.
- custom_register_path: A list of paths to `.py` files for custom registration of models, dialogue templates, and datasets. Defaults to `[]`.
- ddp_timeout: The default value is 18000000, with the unit being seconds.
- ddp_backend: Options include "nccl", "gloo", "mpi", "ccl", "hccl", "cncl", and "mccl". Default is None, which allows for automatic selection.
- ignore_args_error: Used for compatibility with notebooks. The default value is False.

### Model Arguments
- 🔥model: Model ID or local path to the model. If it's a custom model, please use it with `model_type` and `template`. The specific details can be referred to in the [Custom Model](../Customization/Custom-model.md). Default is None.
- model_type: Model type. The same model architecture, template, and model loading process are defined as a model_type. The default is None, and it will be automatically selected based on the suffix of `--model` and the architectures attribute in config.json.
- model_revision: Model revision, default is None.
- task_type: The default value is 'causal_lm'. Optional values are 'causal_lm', 'seq_cls', and 'embedding'. Examples for seq_cls can be found [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/seq_cls), and examples for embedding can be found [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/embedding).
- 🔥torch_dtype: Data type of model weights, supports `float16`, `bfloat16`, `float32`. The default is None, and it is read from the 'config.json' file.
- attn_impl: The type of attention, with options including `flash_attn`, `sdpa`, and `eager`. The default is None, which reads from `config.json`.
  - Note: These three implementations may not all be supported, depending on the support of the corresponding model.
- num_labels: This parameter is required for classification models (i.e., `--task_type seq_cls`). It represents the number of labels, with a default value of None.
- problem_type: This parameter is required for classification models (i.e., `--task_type seq_cls`). The options are 'regression', 'single_label_classification', and 'multi_label_classification'. The default value is None, and it will be automatically set based on the number of labels and the dataset type.
- rope_scaling: Type of rope, supports `linear` and `dynamic` and `yarn`, should be used in conjunction with `max_length`. Default is None.
- device_map: Device map configuration used by the model, such as 'auto', 'cpu', JSON string, or the path of a JSON file. The default is None, automatically set based on the device and distributed training conditions.
- max_memory: When device_map is set to 'auto' or 'sequential', the model weights will be allocated to devices based on max_memory, for example: `--max_memory '{0: "20GB", 1: "20GB"}'`. The default value is None.
- local_repo_path: Some models depend on a GitHub repo when loading. To avoid network issues during `git clone`, a local repo can be used directly. This parameter needs to be passed with the path to the local repo, with the default being `None`.
- init_strategy: When loading the model, initialize all uninitialized parameters. Options values are 'zero', 'uniform', 'normal', 'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'orthogonal'. Default is None.

### Data Arguments
- 🔥dataset: A list of dataset IDs or paths. Default is `[]`. The input format for each dataset is: `dataset ID or dataset path:sub-dataset#sampling size`, where sub-dataset and sampling data are optional. Local datasets support jsonl, csv, json, folders, etc. Open-source datasets can be cloned locally via git and used offline by passing the folder. For custom dataset formats, refer to [Custom Dataset](../Customization/Custom-dataset.md). You can pass in `--dataset <dataset1> <dataset2>` to use multiple datasets.
  - Sub-dataset: This parameter is effective only when the dataset is an ID or folder. If a subset was specified during registration, and only one sub-dataset exists, the registered sub-dataset is selected by default; otherwise, it defaults to 'default'. You can use `/` to select multiple sub-datasets, e.g., `<dataset_id>:subset1/subset2`. You can also use 'all' to select all sub-datasets, e.g., `<dataset_id>:all`.
  - Sampling Size: By default, the complete dataset is used. If the sampling size is less than the total number of data samples, samples are selected randomly without repetition. If the sampling size exceeds the total number of data samples, then `sampling size%total data samples` samples are randomly sampled additionally, and data samples are repetitively sampled `sampling size//total data samples` times. Note: Streaming datasets only perform sequential sampling. If `--dataset_shuffle false` is set, non-streaming datasets will also perform sequential sampling.
- 🔥val_dataset: A list of validation set IDs or paths. Default is `[]`.
- 🔥split_dataset_ratio: The ratio used to split a validation set from the training set when val_dataset is not specified. The default is 0., meaning no validation set will be split from the training set.
  - Note: For "ms-swift<3.6", the default value of this parameter is 0.01.
- data_seed: Random seed for the dataset, default is 42.
- 🔥dataset_num_proc: Number of processes for dataset preprocessing, default is 1.
- 🔥load_from_cache_file: Whether to load the dataset from the cache, default is True.
  - Note: It is recommended to set this parameter to False during the debug phase.
- dataset_shuffle: Whether to shuffle the dataset. Defaults to True.
  - Note: The shuffling in CPT/SFT consists of two parts: dataset shuffling, controlled by `dataset_shuffle`; and shuffling in the train_dataloader, controlled by `train_dataloader_shuffle`.
- val_dataset_shuffle: Whether to perform shuffling on the val_dataset. Default is False.
- 🔥streaming: Stream reading and processing of the dataset, default is False. It is typically set to True when handling large datasets.
  - Note: You need to set `--max_steps` explicitly, as the streaming dataset does not have a defined length. You can achieve training equivalent to `--num_train_epochs` by setting `--save_strategy epoch` and specifying a sufficiently large `max_steps`. Alternatively, you can set `max_epochs` to ensure training exits after the corresponding number of epochs, at which point the model weights will be validated and saved.
- interleave_prob: Defaults to None. When combining multiple datasets, the `concatenate_datasets` function is used by default. If this parameter is set, the `interleave_datasets` function will be used instead. This parameter is typically used when combining streaming datasets and is passed to the `interleave_datasets` function.
- stopping_strategy: Can be either "first_exhausted" or "all_exhausted", with the default being "first_exhausted". This parameter is passed to the `interleave_datasets` function.
- shuffle_buffer_size: This parameter is used to specify the shuffle buffer size for streaming datasets. Defaults to 1000.
- download_mode: Dataset download mode, including `reuse_dataset_if_exists` and `force_redownload`, default is reuse_dataset_if_exists.
- columns: Used for column mapping of the dataset to ensure that the dataset conforms to the format that AutoPreprocessor can handle. For more details, see [here](../Customization/Custom-dataset.md). You can pass in a JSON string, for example: `'{"text1": "query", "text2": "response"}'`, which means mapping "text1" in the dataset to "query" and "text2" to "response". The query-response format can be processed by the AutoPreprocessor. The default value is None.
- strict: If set to True, any row with an issue in the dataset will throw an error immediately, otherwise, erroneous data samples will be discarded. Default is False.
- 🔥remove_unused_columns: Whether to remove unused columns in the dataset, defaults to True.
  - If this parameter is set to False, the extra dataset columns will be passed to the trainer's `compute_loss` function, making it easier to customize the loss function.
  - For GPRO, the default value of this parameter is False.
- 🔥model_name: Only applicable to the self-cognition task and effective only on the `swift/self-cognition` dataset. It replaces the `{{NAME}}` placeholder in the dataset. Input the model's name in both Chinese and English, separated by a space, for example: `--model_name 小黄 'Xiao Huang'`. Default is None.
- 🔥model_author: Only applicable to the self-cognition task and effective only on the `swift/self-cognition` dataset. It replaces the `{{AUTHOR}}` placeholder in the dataset. Input the model author's name in both Chinese and English, separated by a space, for example: `--model_author '魔搭' 'ModelScope'`. Default is None.
- custom_dataset_info: The path to the JSON file for custom dataset registration. Refer to [Custom Dataset](../Customization/Custom-dataset.md). Default is `[]`.


### Template Arguments
- 🔥template: Type of dialogue template. Default is None, which automatically selects the corresponding model's template type.
- 🔥system: Custom system field, can take a string or txt file path as input. Default is None, uses the default system of the template.
  - Note: The system priority in the dataset is the highest, followed by `--system`, and finally the `default_system` defined in the template.
- 🔥max_length: The maximum length of tokens for a single sample. Defaults to None, set to the maximum length of tokens supported by the model (max_model_len).
  - Note: In the cases of PPO, GRPO, and inference, max_length represents max_prompt_length.
- truncation_strategy: Strategy for handling single sample tokens that exceed `max_length`. Options are `delete`, `left`, and `right`, representing deletion, left-side truncation, and right-side truncation, respectively. The default is 'delete'.
  - It is currently not recommended to set the `truncation_strategy` to `left` or `right` for training multimodal models, as this may result in image tokens being truncated and causing errors (to be optimized).
- 🔥max_pixels: The maximum number of pixels (H*W) for input images to a multimodal model. Images exceeding this limit will be scaled. Default is None, meaning no maximum pixel limit.
- 🔥agent_template: Agent template, which determines how to convert the list of tools into a system, how to extract tool calls from the model's response, and specifies the template format for `{"role": "tool_call", "content": "xxx"}` and `{"role": "tool_response", "content": "xxx"}`. Optional values include "react_en", "hermes", "glm4", "qwen_en", "toolbench", etc. For more details, please check [here](https://github.com/modelscope/ms-swift/blob/main/swift/plugin/agent_template/__init__.py). The default value is None, meaning it will be selected based on the model type.
- norm_bbox: Controls how to scale bounding boxes (bbox). Options are 'norm1000' and 'none'. 'norm1000' represents scaling bbox coordinates to one-thousandths, and 'none' means no scaling. Default is None, automatically selected based on the model.
- use_chat_template: Use chat template or generation template, default is `True`. `swift pt` is automatically set to the generation template.
  - Note: `swift pt` is set to False by default, using the generation template.
- 🔥padding_free: Flattens the data in a batch to avoid padding, thereby reducing memory usage and accelerating training. Default is False. Currently supported in CPT/SFT/DPO/GRPO/GKD.
  - Note: When using `padding_free`, it should be combined with `--attn_impl flash_attn` and "transformers>=4.44". For details, see [this PR](https://github.com/huggingface/transformers/pull/31629). (Same as packing)
  - The supported multimodal models are the same as those supported for multimodal packing. Compared to packing, padding_free does not consume additional time or space. Note: Please use "ms-swift>=3.6" and follow [this PR](https://github.com/modelscope/ms-swift/pull/4838).
  - Megatron-SWIFT uses `padding_free` by default, i.e., `qkv_format='thd'`, and no additional configuration is required.
- padding_side: Padding side when `batch_size>=2` during training. Options are 'left' and 'right', with 'right' as the default. (For inference with batch_size>=2, only left padding is applied.)
  - Note: PPO and GKD are set to 'left' by default.
- loss_scale: Weight setting for the loss of training tokens. Default is `'default'`, which means that all responses (including history) are used with a weight of 1 in cross-entropy loss, and the loss from the corresponding `tool_response` in the agent_template is ignored. Possible values include: 'default', 'last_round', 'all', 'ignore_empty_think', and agent-specific options: 'react', 'hermes', 'qwen', 'agentflan', 'alpha_umi'. For more details about the agent part, please refer to [Pluginization](../Customization/Pluginization.md) and [Agent Training](./Agent-support.md).
  - 'last_round': Only calculate the loss for the last round of response.
  - 'all': Calculate the loss for all tokens.
  - 'ignore_empty_think': On top of 'default', ignore the loss calculation for empty `'<think>\n\n</think>\n\n'`. See [this issue](https://github.com/modelscope/ms-swift/issues/4030) for more details.
  - `'react'`, `'hermes'`, `'qwen'`: On top of `'default'`, set the loss weight of the `tool_call` part to 2.
- sequence_parallel_size: Sequence parallelism size, default is 1. Currently supported in CPT/SFT/DPO/GRPO. The training script refers to [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/long_text/ulysses/sequence_parallel.sh).
- response_prefix: The prefix character for the response, for example, setting the response_prefix to `'<think>\n'` for QwQ-32B. The default is None, and it is automatically set according to the model.
  - Note: If you are training the deepseek-r1/qwq model with a dataset that does not include `<think>...</think>`, please pass `--response_prefix ''` additionally when inferring after training.
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
- 🔥stream: Streaming output. Default is `None`, which means it is set to True when using the interactive interface and False during batch inference on datasets.
  - For "ms-swift<3.6", the default value of stream is False.
- stop_words: Additional stop words beyond eos_token, default is`[]`.
  - Note: eos_token will be removed in the output response, whereas additional stop words will be retained in the output.
- logprobs: Whether to output logprobs, default is False.
- top_logprobs: The number of top_logprobs to output, defaults to None.


### Quantization Arguments

The following are the parameters for quantization when loading a model. For detailed meanings, you can refer to the [quantization](https://huggingface.co/docs/transformers/main/en/main_classes/quantization) documentation. Note that this does not include `gptq` and `awq` quantization parameters involved in `swift export`.

- 🔥quant_method: The quantization method used when loading the model. Optional values are 'bnb', 'hqq', 'eetq', 'quanto', and 'fp8'. The default is None.
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
- 🔥vit_gradient_checkpointing: Whether to enable gradient_checkpointing for the vit part during multi-modal model training. Defaults to None, meaning it is set to `gradient_checkpointing`. For an example, please refer to [here](https://github.com/modelscope/ms-swift/blob/main/examples/train/multimodal/vit_gradient_checkpointing.sh).
  - Note: For multimodal models using LoRA training, when `--freeze_vit false` is set and the following warning appears in the command line: `UserWarning: None of the inputs have requires_grad=True. Gradients will be None`, please set `--vit_gradient_checkpointing false`, or raise a related issue. This problem does not occur during full-parameter training.
- 🔥deepspeed: Defaults to None. It can be set to 'zero0', 'zero1', 'zero2', 'zero3', 'zero2_offload', 'zero3_offload' to use the built-in deepspeed configuration file of ms-swift.
- zero_hpz_partition_size: Default is `None`. This parameter is a feature of `ZeRO++`, which implements model sharding within nodes and data sharding between nodes. If you encounter grad_norm `NaN` issues, please try using `--torch_dtype float16`
- 🔥per_device_train_batch_size: Default is 1.
- 🔥per_device_eval_batch_size: Default is 1.
- 🔥gradient_accumulation_steps: Gradient accumulation, default is None, meaning set gradient_accumulation_steps such that total_batch_size >= 16. The total_batch_size equals `per_device_train_batch_size * gradient_accumulation_steps * world_size`.
- weight_decay: Weight decay coefficient, default value is 0.1.
- adam_beta2: Default is 0.95.
- 🔥learning_rate: Learning rate, defaults to 1e-5 for full parameters, and 1e-4 for LoRA and other tuners.
- 🔥vit_lr: When training a multimodal large model, this parameter specifies the learning rate for the ViT. By default, it is set to None, which means it equals `learning_rate`.
  - Usually used in conjunction with the `--freeze_vit` and `--freeze_aligner` parameters.
- 🔥aligner_lr: When training a multimodal large model, this parameter specifies the learning rate for the aligner. By default, it is set to None, which means it equals `learning_rate`.
- lr_scheduler_type: Type of lr_scheduler, defaults to 'cosine'.
- lr_scheduler_kwargs: Other parameters for the lr_scheduler, defaults to None.
- 🔥gradient_checkpointing_kwargs: Parameters for `torch.utils.checkpoint`. For example, set as `--gradient_checkpointing_kwargs '{"use_reentrant": false}'`. Defaults to None.
  - Note: When using DDP without DeepSpeed/FSDP, and `gradient_checkpointing_kwargs` is `None`, it will default to `'{"use_reentrant": false}'`.
- full_determinism: Ensures reproducible results during training. Note: This will negatively impact performance. Defaults to False.
- 🔥report_to: Default value is `tensorboard`. You can also specify `--report_to tensorboard wandb swanlab` or `--report_to all`.
- logging_first_step: Whether to log the first step, defaults to True.
- logging_steps: Interval for logging, defaults to 5.
- logging_dir: The path for TensorBoard logs. Defaults to None, which means it is set to `f'{self.output_dir}/runs'`.
- predict_with_generate: Whether to use generative method during validation, default is False.
- metric_for_best_model: Default is None, which means that when predict_with_generate is set to False, it is set to 'loss'; otherwise, it is set to 'rouge-l' (during PPO training, the default value is not set; in GRPO training, it is set to 'reward').
- greater_is_better: Defaults to None, which sets it to False when `metric_for_best_model` contains 'loss', otherwise sets to True.
- max_epochs: Forces the training to exit after reaching `max_epochs`, and performs validation and saving of the model weights. This parameter is especially useful when using a streaming dataset. Default is None.

Other important parameters:
- 🔥num_train_epochs: Number of training epochs, default is 3.
- 🔥save_strategy: Strategy for saving the model, options include 'no', 'steps', 'epoch'. Default is 'steps'.
- 🔥save_steps: Default is 500.
- 🔥eval_strategy: Evaluation strategy. Default is None and follows the strategy of `save_strategy`.
  - If neither `val_dataset` nor `eval_dataset` is used and `split_dataset_ratio` is 0, the default is 'no'.
- 🔥eval_steps: Default is None. If there is an evaluation dataset, it follows the strategy of `save_steps`.
- 🔥save_total_limit: Maximum number of checkpoints to save. Older checkpoints will be deleted. Default is None, saving all checkpoints.
- max_steps: Maximum number of training steps. Should be set when the dataset is streamed. Default is -1.
- 🔥warmup_ratio: Default is 0.
- save_on_each_node: Default is False. Should be considered in multi-node training.
- save_only_model: Whether to save only the model weights without including optimizer state, random seed state, etc. Default is False.
- 🔥resume_from_checkpoint: Parameter for resuming training from a checkpoint, pass the checkpoint path. Default is None. For resuming training from a checkpoint, keep other parameters unchanged and add `--resume_from_checkpoint checkpoint_dir` additionally.
  - Note: `resume_from_checkpoint` will load the model weights, optimizer weights, and random seed, and continue training from the last trained steps. You can specify `--resume_only_model` to load only the model weights.
- 🔥ddp_find_unused_parameters: Default is None.
- 🔥dataloader_num_workers: Defaults to None. If the platform is Windows, it is set to 0; otherwise, it is set to 1.
- dataloader_pin_memory: Default is True.
- dataloader_persistent_workers: Default is False.
- dataloader_prefetch_factor: Defaults to None. If `dataloader_num_workers > 0`, it is set to 10.
- train_dataloader_shuffle: Specifies whether the dataloader for CPT/SFT training is shuffled, with the default set to True. This parameter is not applicable to IterableDataset, as IterableDataset reads in a sequential manner.
- 🔥neftune_noise_alpha: Coefficient of noise added by neftune, default is 0. Usually can be set to 5, 10, 15.
- 🔥use_liger_kernel: Whether to enable the [Liger](https://github.com/linkedin/Liger-Kernel) kernel to accelerate training and reduce GPU memory consumption. Defaults to False. Example shell script can be found [here](https://github.com/modelscope/ms-swift/blob/main/examples/train/liger).
  - Note: liger_kernel does not support device_map. Please use DDP/DeepSpeed for multi-GPU training.
- average_tokens_across_devices: Whether to average the number of tokens across devices. If set to True, `num_tokens_in_batch` will be synchronized using all_reduce for accurate loss calculation. Default is False.
- max_grad_norm: Gradient clipping. Default is 1.
- push_to_hub: Push checkpoint to hub. Default is False.
- hub_model_id: Default is None.
- hub_private_repo: Default is False.

### Tuner Arguments

- 🔥freeze_llm: This parameter is only effective for multimodal models and can be used for full parameter training and LoRA, but with different meanings. In full parameter training, setting freeze_llm to True will freeze some of the LLM weights. In LoRA training, if `target_modules` is set to 'all-linear', setting freeze_llm to True will prevent adding LoRA modules to the LLM part. The default is False.
- 🔥freeze_vit: This parameter is only effective for multimodal models and can be used for full parameter training and LoRA, with similar meanings as `freeze_llm`. The default is True.
  - Note: Here, "vit" refers not only to the vision_tower but also includes the audio_tower.
- 🔥freeze_aligner: This parameter is only effective for multimodal models and can be used for full parameter training and LoRA, with similar meanings as `freeze_llm`. The default is True.
- 🔥target_modules: Specifies LoRA modules, with a default of `all-linear`. Its behavior differs in LLM and multimodal LLM. For LLM, it automatically finds all linear modules except lm_head and adds a tuner. For multimodal LLM, by default, it only adds a tuner to the LLM part, and this behavior can be controlled by `freeze_llm`, `freeze_vit`, and `freeze_aligner`. This parameter is not limited to LoRA and can be used for other tuners.
- 🔥target_regex: Specifies a regex expression for LoRA modules, with a default of `None`. If this value is provided, the target_modules parameter becomes ineffective. This parameter is not limited to LoRA and can be used for other tuners.
- init_weights: Specifies the method for initializing weights. LoRA can specify `true`, `false`, `gaussian`, `pissa`, `pissa_niter_[number of iters]`. Bone can specify `true`, `false`, `bat`. The default is `true`.
- 🔥modules_to_save: After attaching a tuner, explicitly specifies additional original model modules to participate in training and storage. The default is `[]`. This parameter is not limited to LoRA and can be used for other tuners.

#### Full Arguments

- freeze_parameters: Prefix of the parameters to be frozen, default is `[]`.
- freeze_parameters_regex: Regex for matching the parameters to be frozen，default is None.
- freeze_parameters_ratio: Ratio of parameters to freeze from bottom to top, default is 0. It can be set to 1 to freeze all parameters, and trainable parameters can be set in conjunction with this.
- trainable_parameters: Prefix of additional trainable parameters, default is `[]`.
- trainable_parameters_regex: Regex for matching additional trainable parameters, default is None.
  - Note: `trainable_parameters`, `trainable_parameters_regex` takes precedence over `freeze_parameters`, `freeze_parameters_regex` and `freeze_parameters_ratio`. When full parameter training is specified, all modules are set to trainable, then some parameters are frozen according to `freeze_parameters`, `freeze_parameters_regex` and `freeze_parameters_ratio`, and finally, some parameters are reopened for training according to `trainable_parameters`,`trainable_parameters_regex`.

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

Note: LISA only supports full parameters, i.e., `--train_type full`.

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

### vLLM Arguments

Parameter meanings can be found in the [vllm documentation](https://docs.vllm.ai/en/latest/serving/engine_args.html).

- 🔥gpu_memory_utilization: GPU memory ratio, ranging from 0 to 1. Default is `0.9`.
- 🔥tensor_parallel_size: Tensor parallelism size. Default is `1`.
- pipeline_parallel_size: Pipeline parallelism size. Default is `1`.
- data_parallel_size: Data parallelism size, default is 1, effective in the infer and rollout commands.
- max_num_seqs: Maximum number of sequences to be processed in a single iteration. Default is `256`.
- 🔥max_model_len: Default is `None`, meaning it will be read from `config.json`.
- disable_custom_all_reduce: Disables the custom all-reduce kernel and falls back to NCCL. For stability, the default is `True`.
- enforce_eager: Determines whether vllm uses PyTorch eager mode or constructs a CUDA graph, default is `False`. Setting it to True can save memory but may affect efficiency.
- 🔥limit_mm_per_prompt: Controls the use of multiple media in vllm, default is `None`. For example, you can pass in `--limit_mm_per_prompt '{"image": 5, "video": 2}'`.
- vllm_max_lora_rank: Default is `16`. This is the parameter supported by vllm for lora.
- vllm_quantization: vllm is able to quantize model with this argument，supported values can be found [here](https://docs.vllm.ai/en/latest/serving/engine_args.html).
- enable_prefix_caching: Enable the automatic prefix caching of vllm to save processing time for querying repeated prefixes. The default is `False`.
- use_async_engine: Whether to use the async engine under the vLLM backend. The deployment status (swift deploy) defaults to True, and other statuses default to False.

### SGLang Arguments
Parameter meanings can be found in the [sglang documentation](https://docs.sglang.ai/backend/server_arguments.html).

- sglang_tp_size: Tensor parallelism size. Default is 1.
- sglang_pp_size: Pipeline parallelism size. Default is 1.
- sglang_dp_size: Data parallelism size. Default is 1.
- sglang_ep_size: Expert parallelism size. Default is 1.
- sglang_mem_fraction_static: The fraction of GPU memory used for static allocation (model weights and KV cache memory pool). If you encounter out-of-memory errors, try reducing this value. Default is None.
- sglang_context_length: The maximum context length of the model. Default is None, which means it will use the value from the model's `config.json`.
- sglang_disable_cuda_graph: Disables CUDA graph. Default is False.
- sglang_quantization: Quantization method. Default is None.
- sglang_kv_cache_dtype: Data type for KV cache storage. 'auto' means it will use the model's data type. 'fp8_e5m2' and 'fp8_e4m3' are supported on CUDA 11.8 and above. Default is 'auto'.
- sglang_enable_dp_attention: Enables data parallelism for attention and tensor parallelism for FFN. The data parallelism size (dp size) should be equal to the tensor parallelism size (tp size). Currently supports DeepSeek-V2/3 and Qwen2/3 MoE models. Default is False.
- sglang_disable_custom_all_reduce: Disables the custom all-reduce kernel and falls back to NCCL. For stability, the default is True.

### LMDeploy Arguments

Parameter meanings can be found in the [lmdeploy documentation](https://lmdeploy.readthedocs.io/en/latest/api/pipeline.html#turbomindengineconfig).

- 🔥tp: tensor parallelism degree. Default is `1`.
- session_len: Maximum session length. Default is `None`.
- cache_max_entry_count: The percentage of GPU memory occupied by the k/v cache. Default is `0.8`.
- quant_policy: Default is `0`. Set it to `4` or `8` when quantizing k/v to 4-bit or 8-bit, respectively.
- vision_batch_size: The `max_batch_size` parameter passed to `VisionConfig`. Default is `1`.

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
- loss_type: Type of loss. Defaults to None, which uses the model's built-in loss function.
- channels: Set of channels included in the dataset. Defaults to None. Used in conjunction with `--loss_type channel_loss`. Refer to [this example](https://github.com/modelscope/ms-swift/blob/main/examples/train/plugins/channel_loss.sh) for more details.
- 🔥packing: Whether to use sequence packing to improve computational efficiency. The default value is False. Currently supports `swift pt/sft`.
  - Note: When using packing, please combine it with `--attn_impl flash_attn` and ensure "transformers>=4.44". For details, see [this PR](https://github.com/huggingface/transformers/pull/31629).
  - Supported multimodal models reference: https://github.com/modelscope/ms-swift/blob/main/examples/train/packing/qwen2_5_vl.sh. Note: Please use "ms-swift>=3.6" and follow [this PR](https://github.com/modelscope/ms-swift/pull/4838).
- packing_cache: Specifies the directory for packing cache. The default value is `None`, which means the cache will be stored in the path defined by the environment variable `$MODELSCOPE_CACHE`. When using the packing feature across multiple nodes, ensure that all nodes share the same packing cache directory. You can achieve this by setting the `MODELSCOPE_CACHE` environment variable or by adding the `--packing_cache <shared_path>` argument in the command line.
- 🔥lazy_tokenize: Whether to use lazy tokenization. If set to False, all dataset samples are tokenized before training (for multimodal models, this includes reading images from disk). This parameter defaults to False for LLM training, and True for MLLM training, to save memory.
- use_logits_to_keep: Pass `logits_to_keep` in the `forward` method based on labels to reduce the computation and storage of unnecessary logits, thereby reducing memory usage and accelerating training. The default is `None`, which enables automatic selection.
  - Note: For stability, this value is set to False by default for multimodal models and needs to be manually enabled.
- acc_strategy: Strategy for calculating accuracy during training and validation. Options are `seq`-level and `token`-level accuracy, with `token` as the default.
- max_new_tokens: Generation parameter override. The maximum number of tokens to generate when `predict_with_generate=True`, defaulting to 64.
- temperature: Generation parameter override. The temperature setting when `predict_with_generate=True`, defaulting to 0.
- optimizer: Custom optimizer name for the plugin, defaults to None. Optional optimizer reference: [here](https://github.com/modelscope/ms-swift/blob/main/swift/plugin/optimizer.py).
- metric: Custom metric name for the plugin. Defaults to None, with the default set to 'acc' when `predict_with_generate=False` and 'nlg' when `predict_with_generate=True`.
- eval_use_evalscope: Whether to use evalscope for evaluation, this parameter needs to be set to enable evaluation, refer to [example](../Instruction/Evaluation.md#evaluation-during-training). Default is False.
- eval_dataset: Evaluation datasets, multiple datasets can be set, separated by spaces
- eval_dataset_args: Evaluation dataset parameters in JSON format, parameters for multiple datasets can be set
- eval_limit: Number of samples from the evaluation dataset
- eval_generation_config: Model inference configuration during evaluation, in JSON format, default is `{'max_tokens': 512}`

### RLHF Arguments

RLHF arguments inherit from the [training arguments](#training-arguments).

- 🔥rlhf_type: Type of human alignment algorithm, supporting 'dpo', 'orpo', 'simpo', 'kto', 'cpo', 'rm', 'ppo', 'grpo' and 'gkd'. Default is 'dpo'.
- ref_model: Required for full parameter training when using the dpo, kto, ppo or grpo algorithms. Default is None.
- ref_model_type: Same as model_type. Default is None.
- ref_model_revision: Same as model_revision. Default is None.
- 🔥beta: Coefficient for the KL regularization term. Default is `None`, meaning `simpo` algorithm defaults to `2.`, `grpo` algorithm defaults to `0.04`, `gkd` algorithm defaults to `0.5`, and other algorithms default to `0.1`. For more details, refer to the [documentation](./RLHF.md).
- label_smoothing: Whether to use DPO smoothing, default value is `0`.
- max_completion_length: The maximum generation length in the GRPO/PPO/GKD algorithms. Default is 512.
- 🔥rpo_alpha: The weight of sft_loss added to DPO, default is `1`. The final loss is `KL_loss + rpo_alpha * sft_loss`.
- cpo_alpha: Coefficient for nll loss in CPO/SimPO loss, default is `1.`.
- simpo_gamma: Reward margin term in the SimPO algorithm, with a paper-suggested setting of 0.5-1.5, default is `1.`.
- desirable_weight: Loss weight $\lambda_D$ for desirable response in the KTO algorithm, default is `1.`.
- undesirable_weight: Loss weight $\lambda_U$ for undesirable response in the KTO algorithm, default is `1.`.
- loss_scale: Override template arguments, default is 'last_round'.
- temperature: Default is 0.9; this parameter will be used in PPO, GRPO and GKD.
- lmbda: Default is 0.5. This parameter is used in GKD. It controls the lambda parameter for the proportion of student data (i.e., the proportion of student-generated outputs within the strategy). If lmbda is 0, student-generated data is not used.
- sft_alpha: The default value is 0. It controls the weight of sft_loss added in GKD. The final loss is `gkd_loss + sft_alpha * sft_loss`.
- seq_kd: Default is False. This parameter is used in GKD. It is the `seq_kd` parameter that controls whether to perform Sequence-Level KD (can be viewed as supervised fine-tuning on teacher-generated output).
  - Note: You can perform inference on the dataset using the teacher model in advance (accelerated by inference engines such as vLLM, SGLang, or lmdeploy), and set `seq_kd` to False during training. Alternatively, you can set `seq_kd` to True, which will use the teacher model to generate sequences during training (ensuring different generated data across multiple epochs, but at a slower efficiency).

#### Reward/Teacher Model Parameters

The reward model parameters will be used in PPO and GRPO.

- reward_model: Default is None.
- reward_adapters: Default is `[]`.
- reward_model_type: Default is None.
- reward_model_revision: Default is None.
- teacher_model: Default is None. This parameter must be provided when `rlhf_type` is `'gkd'`.
- teacher_adapters: Default is `[]`.
- teacher_model_type: Default is None.
- teacher_model_revision: Default is None.

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
- missing_eos_penalty: Defaults to None


#### GRPO Arguments
- per_device_train_batch_size: The training batch size per device. In GRPO, this refers to the batch size of completions during training.
- per_device_eval_batch_size: The evaluation batch size per device. In GRPO, this refers to the batch size of completions during evaluation.
- generation_batch_size: Batch size to use for generation. It defaults to the effective training batch size: per_device_train_batch_size * num_processes * gradient_accumulation_steps`
- steps_per_generation: Number of optimization steps per generation. It defaults to gradient_accumulation_steps. This parameter and generation_batch_size cannot be set simultaneously
- num_generations: The number of samples for each prompt, referred to as the G value in the paper, needs to be divisible by per_device_batch_size * - gradient_accumulation_steps * num_processes, default is 8.
- ds3_gather_for_generation: This parameter applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation, improving generation speed. However, disabling this option allows training models that exceed the VRAM capacity of a single GPU, albeit at the cost of slower generation. Disabling this option is not compatible with vLLM generation. The default is True.
- reward_funcs: Reward functions in the GRPO algorithm; options include `accuracy`,`format`,`cosine` and `repetition`, as seen in `swift/plugin/orm.py`. You can also customize your own reward functions in the plugin. Default is `[]`.
- reward_weights: Weights for each reward function. The number should be equal to the sum of the number of reward functions and reward models. If `None`, all rewards are weighted equally with weight `1.0`.
  - Note: If `--reward_model` is included in GRPO training, it is added to the end of the reward functions.
- reward_model_plugin: The logic for the reward model, which defaults to ORM logic. For more information, please refer to [Customized Reward Models](./GRPO/DeveloperGuide/reward_model.md#custom-reward-model).
- dataset_shuffle: Whether to shuffle the dataset randomly. Default is True.
- loss_type: The type of loss normalization. Options are ['grpo', 'bnpo', 'dr_grpo'], default is 'grpo'. For details, see this [pr](https://github.com/huggingface/trl/pull/3256#discussion_r2033213348)
- log_completions: Whether to log the model-generated content during training, to be used in conjunction with `--report_to wandb`, default is False.
  - Note: If `--report_to wandb` is not set, a `completions.jsonl` will be created in the checkpoint to store the generated content.
- use_vllm: Whether to use vLLM as the infer_backend for GRPO generation, default is False.
- vllm_mode: Mode to use for vLLM integration when `use_vllm` is set to `True`. Must be one of `server` or `colocate`
- vllm_mode server parameter
  - vllm_server_base_url: Base URL for the vLLM server (e.g., 'http://localhost:8000'). If provided, `vllm_server_host` " "and `vllm_server_port` are ignored. Default is None.
  - vllm_server_host: The host address of the vLLM server. Default is None. This is used when connecting to an external vLLM server.
  - vllm_server_port: The service port of the vLLM server. Default is 8000.
  - vllm_server_timeout: The connection timeout for the vLLM server. Default is 240 seconds.
  - async_generate: Use async rollout to improve train speed. Note that rollout will use the model updated in the previous round when enabled. Multi-turn scenarios are not supported. Default is `false`.
- vllm_mode colocate parameter
  - vllm_gpu_memory_utilization: vLLM passthrough parameter, default is 0.9.
  - vllm_max_model_len: vLLM passthrough parameter, the total length limit of model, default is None.
  - vllm_enforce_eager: vLLM passthrough parameter, default is False.
  - vllm_limit_mm_per_prompt: vLLM passthrough parameter, default is None.
  - vllm_tensor_parallel_size: the tensor parallel size of vLLM engine, default is 1.
  - sleep_level: make vllm sleep when model is training. Options are 0 or 1, default is 0, no sleep
  - offload_optimizer: Whether to offload optimizer parameters during inference with vLLM. The default is `False`.
  - offload_model: Whether to offload the model during inference with vLLM. The default is `False`.
  - completion_length_limit_scope: Specifies the scope of the `max_completion_length` limit in multi-turn conversations.
  When set to `total`, the total output length across all turns must not exceed `max_completion_length`.
  When set to `per_round`, each individual turn's output length is limited separately.
  Defaults to `per_round`. Currently only takes effect in colocate mode.
- top_k: Default is 50.
- top_p: Default is 0.9.
- repetition_penalty: Repetition penalty term. Default is 1.
- num_iterations: number of iterations per batch. Default is 1.
- epsilon: epsilon value for clipping. Default is 0.2.
- epsilon_high: Upper clip coefficient, default is None. When set, it forms a clipping range of [epsilon, epsilon_high] together with epsilon.
- delta: Delta value for the upper clipping bound in two-sided GRPO. Recommended to be > 1 + epsilon. This method was introduced in the [INTELLECT-2 tech report](https://huggingface.co/papers/2505.07291).
- sync_ref_model: Whether to synchronize the reference model. Default is False。
  - ref_model_mixup_alpha: The Parameter controls the mix between the current policy and the previous reference policy during updates. The reference policy is updated according to the equation: $π_{ref} = α * π_θ + (1 - α) * π_{ref_{prev}}$. Default is 0.6.
  - ref_model_sync_steps：The parameter determines how frequently the current policy is synchronized with the reference policy. Default is 512.
- move_model_batches: When moving model parameters to fast inference frameworks such as vLLM/LMDeploy, determines how many batches to divide the layers into. The default is `None`, which means the entire model is not split. Otherwise, the model is split into `move_model_batches + 1` (non-layer parameters) + `1` (multi-modal component parameters) batches. This parameter is only meaningful for LoRA (PEFT).
- multi_turn_scheduler: Multi-turn GRPO parameter; pass the corresponding plugin name, and make sure to implement it in plugin/multi_turn.py.
- max_turns: Maximum number of rounds for multi-turn GRPO. The default is None, which means there is no limit.
- dynamic_sample: Exclude data within the group where the reward standard deviation is 0, and additionally sample new data. Default is False.
- max_resample_times: Under the dynamic_sample setting, limit the number of resampling attempts to a maximum of 3. Default is 3 times.
- overlong_filter: Skip overlong truncated samples, which will not be included in loss calculation. Default is False.
The hyperparameters for the reward function can be found in the [Built-in Reward Functions section](#built-in-reward-functions).

cosine reward function arguments
- cosine_min_len_value_wrong (default: -0.5): Reward value corresponding to the minimum length when the answer is incorrect.
- cosine_max_len_value_wrong (default: 0.0): Reward value corresponding to the maximum length when the answer is incorrect.
- cosine_min_len_value_correct (default: 1.0): Reward value corresponding to the minimum length when the answer is correct.
- cosine_max_len_value_correct (default: 0.5): Reward value corresponding to the maximum length when the answer is correct.
- cosine_max_len (default value equal to the model's maximum generation capacity): Maximum length limit for generated text. Default value equal to max_completion_length

repetition penalty function arguments

- repetition_n_grams (default: 3): Size of the n-gram used to detect repetition.
- repetition_max_penalty (default: -1.0): Maximum penalty value, which controls the intensity of the penalty.

Soft overlong reward parameters:

- soft_max_length: L_max in the paper, the maximum generation length of the model, default is equal to max_completion_length.
- soft_cache_length: L_cache in the paper, controls the length penalty interval, which is defined as [soft_max_length - soft_cache_length, soft_max_length].

#### SWANLAB

- **swanlab_token**: SwanLab's API key
- **swanlab_project**: SwanLab's project, which needs to be created in advance on the page: [https://swanlab.cn/space/~](https://swanlab.cn/space/~)
- **swanlab_workspace**: Defaults to `None`, will use the username associated with the API key
- **swanlab_exp_name**: Experiment name, can be left empty. If empty, the value of `--output_dir` will be used by default
- swanlab_lark_webhook_url: Defaults to None. SwanLab's Lark webhook URL, used for pushing experiment results to Lark.
- swanlab_lark_secret: Defaults to None. SwanLab's Lark secret, used for pushing experiment results to Lark.
- **swanlab_mode**: Optional values are `cloud` and `local`, representing cloud mode or local mode


### Inference Arguments

Inference arguments include the [base arguments](#base-arguments), [merge arguments](#merge-arguments), [vLLM arguments](#vllm-arguments), [LMDeploy arguments](#LMDeploy-arguments), and also contain the following:

- 🔥infer_backend: Inference acceleration backend, supporting four inference engines: 'pt', 'vllm', 'sglang', and 'lmdeploy'. The default is 'pt'.
- 🔥max_batch_size: Effective when infer_backend is set to 'pt'; used for batch inference, with a default value of 1. If set to -1, there is no restriction.
- 🔥result_path: Path to store inference results (jsonl). The default is None, meaning results are saved in the checkpoint directory (with args.json file) or './result' directory. The final storage path will be printed in the command line.
  - Note: If the `result_path` file already exists, it will be appended to.
- write_batch_size: The batch size for writing results to result_path. Defaults to 1000. If set to -1, there is no restriction.
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
  - Note: In `swift app` or `swift eval`, the default is False.
- log_interval: Interval for printing tokens/s statistics, default is 20 seconds. If set to -1, it will not be printed.
- max_logprobs: Maximum number of logprobs returned to the client, with a default value of 20.
- Rollout Parameters
  - multi_turn_scheduler: Multi-turn GRPO parameter; pass the corresponding plugin name, and make sure to implement it in plugin/multi_turn.py.
  - max_turns: Maximum number of rounds for multi-turn GRPO. The default is None, which means there is no limit.

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

- 🔥eval_backend: Evaluation backend, defaults to 'Native'. It can also be specified as 'OpenCompass' or 'VLMEvalKit'.
- 🔥eval_dataset: Evaluation dataset, please refer to the [evaluation documentation](./Evaluation.md).
- eval_limit: Number of samples per evaluation set, defaults to None.
- eval_output_dir: Directory to store evaluation results, defaults to 'eval_output'.
- temperature: Override generation parameters, defaults to 0.
- eval_num_proc: Maximum client concurrency during evaluation, defaults to 16.
- eval_url: Evaluation URL, e.g., `http://localhost:8000/v1`. Examples can be found [here](https://github.com/modelscope/ms-swift/tree/main/examples/eval/eval_url). Defaults to None for local deployment evaluation.
- eval_generation_config: Model inference configuration during evaluation, should be passed as a JSON string, e.g., `'{"max_new_tokens": 512}'`; defaults to None.
- extra_eval_args: Additional evaluation parameters, should be passed as a JSON string, defaults to empty. Only effective for Native evaluation. For more parameter descriptions, please refer to [here](https://evalscope.readthedocs.io/en/latest/get_started/parameters.html).
- local_dataset: Some evaluation sets, such as `CMB`, require additional data packages to be downloaded for utilization. Setting this parameter to `true` will automatically download the full data package, create a `data` folder in the current directory, and start the evaluation. The data package will only be downloaded once, and future evaluations will use the cache. This parameter defaults to `false`.
  - Note: By default, evaluation uses the dataset under `~/.cache/opencompass`. After specifying this parameter, it will directly use the data folder in the current directory.


### Export Arguments

Export Arguments include the [basic arguments](#base-arguments) and [merge arguments](#merge-arguments), and also contain the following:

- 🔥output_dir: The path for storing exported results. The default value is None, and an appropriate suffix path will be automatically set.
- exist_ok: If output_dir exists, do not raise an exception and overwrite the contents. The default value is False.
- 🔥quant_method: Options are 'gptq', 'awq', 'bnb' or 'fp8', with the default being None. Examples can be found [here](https://github.com/modelscope/ms-swift/tree/main/examples/export/quantize).
- quant_n_samples: The number of samples for the validation set used by gptq/awq, with a default of 256.
- max_length: Max length for the calibration set, default value is 2048.
- quant_batch_size: Quantization batch size, default is 1.
- group_size: Group size for quantization, default is 128.
- to_ollama: Generate the Modelfile required by Ollama. Default is False.
- 🔥to_mcore: Convert weights from HF format to Megatron format. Default is False.
- to_hf: Convert weights from Megatron format to HF format. Default is False.
- mcore_model: Path to the mcore format model. Default is None.
- thread_count: The number of model slices when `--to_mcore true` is set. Defaults to None, and is automatically configured based on the model size, ensuring that the largest slice is less than 10GB.
- 🔥test_convert_precision: Test the precision error when converting weights between HF and Megatron formats. Default is False.
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
The parameter meanings are the same as in the `qwen_vl_utils` or `qwen_omni_utils` library. You can refer to [here](https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L24)

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

### qwen2_5_omni
qwen2_5_omni not only includes the model-specific parameters of qwen2_5_vl and qwen2_audio, but also contains the following parameter:
- USE_AUDIO_IN_VIDEO: Default is False.
- 🔥ENABLE_AUDIO_OUTPUT: Default is True. If training with zero3, set it to False.

### internvl, internvl_phi3
For the meaning of the arguments, please refer to [here](https://modelscope.cn/models/OpenGVLab/Mini-InternVL-Chat-2B-V1-5)
- MAX_NUM: Default is 12
- INPUT_SIZE: Default is 448

### internvl2, internvl2_phi3, internvl2_5, internvl3
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
- PYTORCH_CUDA_ALLOC_CONF: It is recommended to set it to `'expandable_segments:True'`, which reduces GPU memory fragmentation. For more details, please refer to the [PyTorch documentation](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management).
- MASTER_PORT: Pass-through for the `--master_port` parameter in torchrun. The default is 29500.
- MASTER_ADDR: Pass-through for the `--master_addr` parameter in torchrun.
- NNODES: Pass-through for the `--nnodes` parameter in torchrun.
- NODE_RANK: Pass-through for the `--node_rank` parameter in torchrun.
- LOG_LEVEL: The log level, default is 'INFO'. You can set it to 'WARNING', 'ERROR', etc.
- SWIFT_DEBUG: During `engine.infer(...)`, if set to '1', the content of input_ids and generate_ids will be printed.
