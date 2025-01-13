# Command Line Parameters

The introduction to command line parameters will cover base arguments, atomic arguments, and integrated arguments, and specific model arguments. The final list of arguments used in the command line is the integration arguments. Integrated arguments inherit from basic arguments and some atomic arguments. Specific model arguments are designed for specific models and can be set using `--model_kwargs'` or the environment variable.

## Base Arguments

- 🔥tuner_backend: Optional values are 'peft' and 'unsloth', default is 'peft'
- 🔥train_type: Default is 'lora'. Optional values: 'lora', 'full', 'longlora', 'adalora', 'llamapro', 'adapter', 'vera', 'boft', 'fourierft', 'reft'
- 🔥adapters: A list used to specify the ID/path of the adapter, default is `[]`.
- seed: Default is 42
- model_kwargs: Extra parameters specific to the model. This parameter list will be logged during training for reference, for example, `--model_kwargs '{"fps_max_frames": 12}'`.
- load_args: When `--resume_from_checkpoint`, `--model`, or `--adapters` is specified, it will read the `args.json` file from the saved checkpoint and assign values to the `BaseArguments` that are defaulted to None (excluding DataArguments and GenerationArguments). These can be overridden by manually passing in values. The default is `True`.
- load_data_args: If this parameter is set to True, it will additionally read the data parameters. The default is `False`.
- use_hf: Default is False. Controls model and dataset downloading, and model pushing to the hub.
- hub_token: Hub token. You can check the modelscope hub token [here](https://modelscope.cn/my/myaccesstoken).
- custom_register_path: The file path for the custom model, chat template, and dataset registration `.py` files.

### Model Arguments
- 🔥model: Model ID or local path to the model. If it's a custom model, please use it with `model_type` and `template`. The specific details can be referred to in the [Custom Model](../Customization/Custom-model.md).
- model_type: Model type. The same model architecture, template, and loading process define a model_type.
- model_revision: Model version.
- 🔥torch_dtype: Data type for model weights, supports `float16`, `bfloat16`, `float32`, default is read from the config file.
- task_type: Defaults to 'causal_lm'. Options include 'causal_lm' and 'seq_cls'. You can view examples of seq_cls [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/seq_cls).
- attn_impl: type of attention, options are `flash_attn`, `sdpa`, `eager`, with the default being `sdpa`. Note: Not all three implementations are guaranteed to be supported; it depends on the support available for the corresponding model.
- num_labels: To be specified for classification models, representing the number of labels, default is None.
- rope_scaling: Rope type, supports `linear` and `dynamic`, to be used with `max_length`.
- device_map: Configuration of the device map used by the model, e.g., 'auto', 'cpu', json string, json file path.
- local_repo_path: Some models require a GitHub repo when loading. To avoid network issues during `git clone`, you can directly use a local repo. This parameter needs to pass the local repo path, default is `None`.

### Data Arguments
- 🔥dataset: Dataset ID or path. The format is `dataset_id or dataset_path:sub_dataset#sample_count`, where sub_dataset and sample_count are optional. Use spaces to pass multiple datasets. Local datasets support jsonl, csv, json, and folders, etc. For custom datasets, you can refer to [Custom Dataset](../Customization/Custom-dataset.md).
- 🔥val_dataset: Validation dataset ID or path.
- 🔥split_dataset_ratio: How to split the training and validation sets when val_dataset is not specified, default is 0.01.
- data_seed: Random seed for the dataset, default is 42.
- 🔥dataset_num_proc: Number of processes for dataset preprocessing, default is 1.
- 🔥streaming: Stream read and process the dataset, default is False.
- enable_cache: Use cache for dataset preprocessing, default is False.
  - Note: If set to True, it may not take effect if the dataset changes. If modifying this parameter leads to issues during training, consider setting it to False.
- download_mode: Dataset download mode, including `reuse_dataset_if_exists` and `force_redownload`, default is reuse_dataset_if_exists.
- strict: If True, the dataset will throw an error if any row has a problem; otherwise, it will discard the erroneous row. Default is False.
- 🔥model_name: For self-awareness tasks, input the model's Chinese and English names separated by space.
- 🔥model_author: For self-awareness tasks, input the model author's Chinese and English names separated by space.
- custom_dataset_info: Custom simple dataset registration, refer to the [Custom Dataset](../Customization/Custom-dataset.md) Documentation.

### Template Arguments
- 🔥template: Type of dialogue template, which defaults to the template type corresponding to the model. `swift pt` will convert the dialogue template into a generation template for use.
- 🔥system: Custom system field, default is None, uses the default system of the template.
- 🔥max_length: The maximum length of tokens for a single sample. Defaults to None, set to the maximum length of tokens supported by the model (max_model_len).
- truncation_strategy: How to handle overly long tokens, supports `delete`, `left`, `right`, representing deletion, left trimming, and right trimming, default is 'delete'.
- 🔥max_pixels: Maximum pixel count for pre-processing images in multimodal models (H*W), default is no scaling.
- tools_prompt: The list of tools for agent training converted to system format, refer to [Agent Training](./Agent-support.md), default is 'react_en'.
- padding_side: The padding_side used when training with `batch_size >= 2`, with optional values of 'left' and 'right', defaulting to 'right'. (When the batch_size in `generate` is >= 2, only left padding is applied.)
- loss_scale: How to add token loss weight during training. Default is `'default'`, meaning all responses (including history) are treated as 1 for cross-entropy loss. The optional values are 'default', 'last_round', 'all', and the loss scale required by the agent: 'react', 'agentflan', 'alpha_umi', 'qwen'. For specifics, see [Pluginization](../Customization/Pluginization.md) and [Agent Training](./Agent-support.md).
- sequence_parallel_size: Number of sequence parallelism. Refer to [example](https://github.com/modelscope/ms-swift/tree/main/examples/train/sequence_parallel/train.sh).
- use_chat_template: Use chat template or generation template, default is `True`. `swift pt` is automatically set to the generation template.
- template_backend: Use swift or jinja for inference. If using jinja, it will utilize transformers' `apply_chat_template`. Default is swift.

### Generation Arguments

Refer to the [generation_config](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig) documentation.

- 🔥max_new_tokens: Maximum new token count supported during inference, default is None (no limit).
- temperature: Temperature parameter, default is None, read from generation_config.json.
  - Note: The do_sample parameter has been removed in this version; set temperature to 0 for the same effect.
- top_k: Top_k parameter, default is None, read from generation_config.json.
- top_p: Top_p parameter, default is None, read from generation_config.json.
- repetition_penalty: Penalty for repetition, default is None, read from generation_config.json.
- num_beams: Number of beams for beam search, default is 1.
- 🔥stream: Stream output, default is `False`.
- stop_words: Additional stop words, default is `[]`.
- logprobs: Whether to output logprobs, default is False.

### Quantization Arguments

The following are quantization parameters for loading models. For specific meanings, see the [Quantization](https://huggingface.co/docs/transformers/main/en/main_classes/quantization) documentation. This does not include quantization parameters related to `swift export`, such as `gptq` and `awq`.

- 🔥quant_method: Quantization method used when loading the model, options are `bnb`, `hqq`, `eetq`.
- 🔥quant_bits: Number of bits for quantization, default is None.
- hqq_axis: HQQ quantization axis, default is None.
- bnb_4bit_compute_dtype: BNB quantization compute type, options are `float16`, `bfloat16`, `float32`, default is set to `torch_dtype`.
- bnb_4bit_quant_type: BNB quantization type, supports `fp4` and `nf4`, default is `nf4`.
- bnb_4bit_use_double_quant: Whether to use double quantization, default is `True`.
- bnb_4bit_quant_storage: BNB quantization storage type, default is None.

## Atomic Arguments

### Seq2SeqTrainer Arguments

This parameter list inherits from transformers `Seq2SeqTrainingArguments`, with default values overridden by ms-swift. For unlisted items, refer to the [HF Official Documentation](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Seq2SeqTrainingArguments).

- 🔥output_dir: Default is `output/<model_name>`.
- 🔥gradient_checkpointing: Whether to use gradient checkpointing, default is True.
- 🔥deepspeed: Default is None. Can be set to 'zero0', 'zero1', 'zero2', 'zero3', 'zero2_offload', 'zero3_offload' to use the built-in deepspeed configuration files from ms-swift.
- 🔥per_device_train_batch_size: Default is 1.
- 🔥per_device_eval_batch_size: Default is 1.
- weight_decay: Weight decay coefficient, default value is 0.1.
- 🔥learning_rate: Learning rate, default is 1e-5 for all parameters, and 1e-4 for the tuner.
- lr_scheduler_type: LR scheduler type, default is cosine.
- lr_scheduler_kwargs: Other parameters for the LR scheduler.
- 🔥gradient_checkpointing_kwargs: Parameters passed to `torch.utils.checkpoint`. For example, set to `--gradient_checkpointing_kwargs '{"use_reentrant": false}'`.
- report_to: Default is `tensorboard`. You can also specify `--report_to tensorboard wandb`, `--report_to all`.
- remove_unused_columns: Default is False.
- logging_first_step: Whether to log the first step print, default is True.
- logging_steps: Interval for logging prints, default is 5.
- metric_for_best_model: Default is None. When `predict_with_generate` is set to False, it is 'loss'; otherwise, it is 'rouge-l'.
- greater_is_better: Default is None. When `metric_for_best_model` contains 'loss', set to False; otherwise, set to True.

Other important parameters:
- 🔥num_train_epochs: Number of training epochs, default is 3.
- 🔥gradient_accumulation_steps: Gradient accumulation, default is 1.
- 🔥save_strategy: Strategy for saving the model, options are 'no', 'steps', 'epoch', default is 'steps'.
- 🔥save_steps: Default is 500.
- 🔥eval_strategy: Default is None. Evaluation strategy, follows `save_strategy`.
- 🔥eval_steps: Default is None. If evaluation dataset exists, follows `save_steps`.
- 🔥save_total_limit: Default is None, saving all checkpoints.
- max_steps: Default is -1, maximum number of training steps. Must be set when the dataset is streaming.
- 🔥warmup_ratio: Default is 0.
- save_on_each_node: Default is False. To be considered in multi-machine training.
- save_only_model: Default is False. Whether to save only model weights.
- 🔥resume_from_checkpoint: Checkpoint resume parameter, default is None.
- 🔥ddp_backend: Default is None, optional values are "nccl", "gloo", "mpi", "ccl", "hccl", "cncl", "mccl".
- 🔥ddp_find_unused_parameters: Default is None.
- 🔥dataloader_num_workers: Default is 0.
- 🔥neftune_noise_alpha: Noise coefficient added by neftune, default is 0. Generally can be set to 5, 10, 15.
- average_tokens_across_devices: Whether to average the token count across devices. If set to True, it will use all_reduce to synchronize `num_tokens_in_batch` for accurate loss computation. The default is False.
- max_grad_norm: Gradient clipping. The default value is 1.
- push_to_hub: Push training weights to hub, default is False.
- hub_model_id: Default is None.
- hub_private_repo: Default is False.

### Tuner Arguments

- 🔥freeze_llm: Freeze LLM. Default is False. Applicable for full parameters and LoRA.
- 🔥freeze_vit: Freeze ViT. Default is True. Applicable for full parameters and LoRA.
- 🔥freeze_aligner: Freeze aligner. Default is True, applicable for full parameters and LoRA.
- 🔥target_modules: The specified LoRA module defaults to `all-linear`. This behavior differs in LLM and multimodal LLM. If it is LLM, it will automatically search for linear except lm_head and attach tuner. If it is multimodal LLM, it defaults to attach tuner only on LLM, and this behavior can be controlled by `freeze_llm`, `freeze_vit`, `freeze_aligner`. This parameter is not limited to LoRA.
- 🔥target_regex: Specify a regex expression for the LoRA module. Default is `None`, if this value is provided, target_modules does not take effect. This parameter is not limited to LoRA.
- 🔥init_weights: The method of init tuner weights, For lora the accepted values are `true`, `false`, `guassian`, `pissa`, `pissa_niter_[number of iters]`, for bone are `true`, `false`, `bat`, default is `true`
- modules_to_save: After the tuner is attached, the original model's modules used during training and storage, default is `[]`. This parameter is not limited to LoRA.

#### Full Arguments

- freeze_parameters: Prefix of parameters to be frozen, default is `[]`.
- freeze_parameters_ratio: Ratio of parameters to freeze from the bottom up, default is 0. Setting it to 1 will freeze all parameters. Combine with `trainable_parameters` to set trainable parameters.
- trainable_parameters: Prefix of trainable parameters, default is `[]`. The priority of `trainable_parameters` is higher than that of `freeze_parameters` and `freeze_parameters_ratio`.

#### LoRA

- 🔥lora_rank: Default is `8`.
- 🔥lora_alpha: Default is `32`.
- lora_dropout: Default is `0.05`.
- lora_bias: Default is `'none'`, selectable values are: 'none', 'all'. If you want to set all biases as trainable, you can set it to `'all'`.
- lora_dtype: Specify the dtype of the LoRA module. Supports 'float16', 'bfloat16', 'float32', defaults to the original model type.
- 🔥use_dora: Default is `False`, whether to use `DoRA`.
- use_rslora: Default is `False`, whether to use `RS-LoRA`.
- 🔥lorap_lr_ratio: LoRA+ parameter, default value is `None`, recommended values `10~16`, specifying this parameter allows using lora+ when using LoRA.
- init_weights: Weight initialization method, applicable to supported Tuners. The default value is `true`.

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

- 🔥tp: Tensor parallelism degree, default value is `1`.
- session_len: Default value is `None`.
- cache_max_entry_count: Default value is `0.8`.
- quant_policy: Default value is `0`.
- vision_batch_size: Default value is `1`.

### vLLM Arguments

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

### Merge Arguments

- 🔥merge_lora: Whether to merge LoRA. This parameter supports LoRA, llmpro, longlora, default is False.
- safe_serialization: Whether to store safetensors, default is True.
- max_shard_size: Maximum size of a single storage file, default is '5GB'.

## Integration Arguments

### Training Arguments

Training arguments include the [base arguments](#base-arguments), [Seq2SeqTrainer arguments](#Seq2SeqTrainer-arguments), [tuner arguments](#tuner-arguments), and also include the following parts:

- add_version: Add directory to output_dir with `'<version>-<timestamp>'` to prevent weight overwrite, default is True.
- resume_only_model: If resume_from_checkpoint, only resume model weights, default is False.
- check_model: Check local model files for corruption or modification and give a prompt, default is True. If in an offline environment, please set to False.
- loss_type: Type of loss, default uses the model's built-in loss function.
- packing: Whether to use packing, default is False.
- 🔥lazy_tokenize: Whether to use lazy_tokenize, default is False during LLM training, default is True during MLLM training.

- acc_strategy: Strategy for training accuracy, can be `seq` or `token` level accuracy, default is `token`.
- max_new_tokens: Maximum generated token count when `predict_with_generate=True`, default 64.
- temperature: Temperature when `predict_with_generate=True`, default 0.
- optimizer: Custom optimizer name for plugin.
- metric: Custom metric name for plugin.

### RLHF Arguments

RLHF arguments inherit from the [training arguments](#training-arguments).

- 🔥rlhf_type: Alignment algorithm type, supports `dpo`, `orpo`, `simpo`, `kto`, `cpo`, `rm`, `ppo`.
- ref_model: Original comparison model in algorithms like DPO.
- ref_model_type: Same as model_type.
- ref_model_revision: Same as model_revision.

- 🔥beta: KL regularization term coefficient, default is `None`, i.e., for `simpo` algorithm default is `2.`, for other algorithms default is `0.1`. Refer to the [documentation](./Human-alignment.md) for specifics.
- label_smoothing: Whether to use DPO smoothing, default value is `0`, generally set between 0~0.5.

- 🔥rpo_alpha: Weight for adding sft_loss in DPO, default is `1`. The final loss is `KL_loss + rpo_alpha * sft_loss`.

- cpo_alpha: The coefficient of nll loss in CPO/SimPO loss, default is `1.`.

- simpo_gamma: Reward margin term in SimPO algorithm, recommended to set between 0.5-1.5 in the paper, default is `1.`.

- desirable_weight: Loss weight for desirable response in KTO algorithm $\lambda_D$, default is `1.`.
- undesirable_weight: Loss weight for undesirable response in KTO paper $\lambda_U$, default is `1.`.

#### PPO Arguments

- reward_model: Defaults to None
- reward_adapters: Defaults to `[]`
- reward_model_type: Defaults to None
- reward_model_revision: Defaults to None

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
- temperature: Defaults to 0.7
- missing_eos_penalty: Defaults to None

### Inference Arguments

Inference arguments include the [base arguments](#base-arguments), [merge arguments](#merge-arguments), [vLLM arguments](#vllm-arguments), [LMDeploy arguments](#LMDeploy-arguments), and also contain the following:

- 🔥infer_backend: Inference backend, supports 'pt', 'vllm', 'lmdeploy', default is 'pt'.
- 🔥max_batch_size: Batch size for pt backend, default is 1.
- ddp_backend: The distributed backend for multi-gpu inference using the pt backend, default is None. Examples of multi-card inference can be found [here](https://github.com/modelscope/ms-swift/tree/main/examples/infer/pt).
- result_path: Path to store inference results (jsonl), default is None, saved in the checkpoint directory or './result' directory.
- val_dataset_sample: Number of samples from the inference dataset, default is None.

### Deployment Arguments

Deployment Arguments inherit from the [inference arguments](#inference-arguments).

- host: Service host, default is '0.0.0.0'.
- port: Port number, default is 8000.
- api_key: Access key required for access.
- owned_by: Default is `swift`.
- 🔥served_model_name: Model name for serving, defaults to the model's suffix.
- verbose: Print access logs, default is True.
- log_interval: Interval for printing tokens/s statistics, default is 20 seconds. If set to -1, it will not be printed.
- max_logprobs: Maximum number of logprobs to return, default is 20.

### Web-UI Arguments
- server_name: Host for the web UI, default is '0.0.0.0'.
- server_port: Port for the web UI, default is 7860.
- share: Default is False.
- lang: Language for the web UI, options are 'zh', 'en'. Default is 'zh'.


### App Arguments
App parameters inherit from [deployment arguments](#deployment-arguments) and [Web-UI Arguments](#web-ui-arguments).

- base_url: Base URL for the model deployment, for example, `http://localhost:8000/v1`. Default is None.
- studio_title: Title of the studio. Default is None, set to the model name.
- is_multimodal: Whether to launch the multimodal version of the app. Defaults to None, automatically determined based on the model; if it cannot be determined, set to False.
- lang: Overrides the Web-UI Arguments, default is 'en'.

### Evaluation Arguments

Evaluation Arguments inherit from the [deployment arguments](#deployment-arguments).

- 🔥eval_dataset: Evaluation dataset, refer to [Evaluation documentation](./Evaluation.md).
- eval_limit: Number of samples for each evaluation set, default is None.
- eval_output_dir: Folder for storing evaluation results, default is 'eval_output'.
- temperature: Default is 0.
- verbose: This parameter is passed to DeployArguments during local evaluation, default is `False`.
- eval_num_proc: Maximum concurrency for clients during evaluation. The default for text evaluation is 256, while for multimodal it is 16.
- 🔥eval_url: Evaluation URL, for example `http://localhost:8000/v1`. Default is None, uses local deployment for evaluation. You can view the examples [here](https://github.com/modelscope/ms-swift/tree/main/examples/eval/eval_url).

### Export Arguments

Export Arguments include the [basic arguments](#base-arguments) and [merge arguments](#merge-arguments), and also contain the following:

- 🔥output_dir: Path for storing export results, default is None.

- 🔥quant_method: Options are 'gptq' and 'awq', default is None.
- quant_n_samples: Sampling size for the validation set in gptq/awq, default is 128.
- max_length: Max length for the calibration set, default value is 2048.
- quant_batch_size: Quantization batch size, default is 1.
- group_size: Group size for quantization, default is 128.

- 🔥push_to_hub: Whether to push to the hub, default is False.
- hub_model_id: Model ID for pushing, default is None.
- hub_private_repo: Whether it is a private repo, default is False.
- commit_message: Commit message, default is 'update files'.

## Specific Model Arguments

Specific model arguments can be set using `--model_kwargs` or environment variables, for example: `--model_kwargs '{"fps_max_frames": 12}'` or `FPS_MAX_FRAMES=12`.

### qwen2_vl, qvq
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

### minicpmv2_6
- MAX_SLICE_NUMS: Default is 9, refer to [here](https://modelscope.cn/models/OpenBMB/MiniCPM-V-2_6/file/view/master?fileName=config.json&status=1)
- VIDEO_MAX_SLICE_NUMS: Default is 1, which is the MAX_SLICE_NUMS for videos, refer to [here](https://modelscope.cn/models/OpenBMB/MiniCPM-V-2_6)
- MAX_NUM_FRAMES: Default is 64, refer to [here](https://modelscope.cn/models/OpenBMB/MiniCPM-V-2_6)

### ovis1_6
- MAX_PARTITION: Refer to [here](https://github.com/AIDC-AI/Ovis/blob/d248e34d755a95d24315c40e2489750a869c5dbc/ovis/model/modeling_ovis.py#L312)

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
