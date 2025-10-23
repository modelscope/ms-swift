# Command Line Parameters

The command-line arguments will be introduced in four categories: basic arguments, atomic arguments, integrated arguments, and model-specific arguments. **The final list of arguments used in the command line consists of the integrated arguments, which inherit from the basic arguments and certain atomic arguments**. Model-specific arguments are tailored for particular models and can be configured via `--model_kwargs` or environment variables. For a detailed introduction to Megatron-SWIFT command-line arguments, please refer to the [Megatron-SWIFT Training Documentation](../Megatron-SWIFT/Command-line-parameters.md).

**Tips:**

- To pass a list via the command line, separate the elements with spaces. For example: `--dataset <dataset_path1> <dataset_path2>`.
- To pass a dictionary via the command line, use JSON format. For example: `--model_kwargs '{"fps_max_frames": 12}'`.
- Parameters marked with 🔥 are important; new users of ms-swift should prioritize these command-line arguments.

## Base Arguments

- 🔥tuner_backend: Optional values are `'peft'` and `'unsloth'`. Default is `'peft'`.
- 🔥train_type: Optional values are `'lora'`, `'full'`, `'longlora'`, `'adalora'`, `'llamapro'`, `'adapter'`, `'vera'`, `'boft'`, `'fourierft'`, `'reft'`. Default is `'lora'`.
- 🔥adapters: A list specifying adapter IDs or paths. Default is `[]`. This parameter is typically used in inference/deployment commands, for example: `swift infer --model '<model_id_or_path>' --adapters '<adapter_id_or_path>'`. It can occasionally be used for resuming training from a checkpoint. The difference between this parameter and `resume_from_checkpoint` is that **this parameter only loads adapter weights**, without restoring the optimizer state or random seed, and does not skip already-trained portions of the dataset.
- external_plugins: A list of external `plugin.py` files that will be registered into the plugin module (i.e., imported). See an example [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo/plugin/run_external_reward_func.sh). Default is `[]`.
- seed: Global random seed. Default is 42.
  - Note: This random seed is independent of `data_seed`, which controls randomness in the dataset.
- model_kwargs: Additional arguments specific to certain models. This list of parameters will be logged during training/inference. For example: `--model_kwargs '{"fps_max_frames": 12}'`. You can also set it via environment variables, e.g., `FPS_MAX_FRAMES=12`. Default is None.
  - Note: **If you specify model-specific parameters during training, please also set the corresponding parameters during inference**—this helps maintain consistent performance.
  - The meaning of model-specific parameters can usually be found in the official repository or inference code of the corresponding model. MS-Swift includes these parameters to ensure alignment between trained models and official inference behavior.
- load_args: When `--resume_from_checkpoint`, `--model`, or `--adapters` are specified, this flag controls whether to load `args.json` from the saved file. The loaded keys are defined in [base_args.py](https://github.com/modelscope/ms-swift/blob/main/swift/llm/argument/base_args/base_args.py). Default is `True` for inference and export, and `False` for training. Usually, this parameter does not need to be modified.
- load_data_args: If set to `True`, additional data-related arguments from `args.json` will be loaded. Default is `False`. **This is typically used during inference to run inference on validation sets split during training**, for example: `swift infer --adapters xxx --load_data_args true --stream true --max_new_tokens 512`.
- use_hf: Controls whether ModelScope or HuggingFace is used for model downloading, dataset downloading, and model uploading. Default is `False` (uses ModelScope).
- hub_token: Hub authentication token. For ModelScope, see [here](https://modelscope.cn/my/myaccesstoken). Default is `None`.
- custom_register_path: A list of `.py` file paths containing custom model, chat template, and dataset registrations. These files will be additionally loaded (i.e., imported). Default is `[]`.
- ddp_timeout: Default is 18000000, in seconds.
- ddp_backend: Optional values are `"nccl"`, `"gloo"`, `"mpi"`, `"ccl"`, `"hccl"`, `"cncl"`, `"mccl"`. Default is `None`, which enables automatic selection.
- ignore_args_error: Used for compatibility with Jupyter Notebook. Default is `False`.

### Model Arguments
- 🔥model:  The [model ID](https://modelscope.cn/models) or local model path. Default is `None`.
- model_type: The model type. In ms-swift, a `model_type` refers to a group of models that share the same architecture, model loading process, and template definition. Default is `None`, meaning it will be automatically inferred based on the suffix of `--model` and the 'architectures' field in config.json. Supported model types can be found in the [List of Supported Models and Datasets](./Supported-models-and-datasets.md)
  - Note: The concept of `model_type` in MS-Swift differs from the `model_type` in `config.json`.
  - Custom models typically require manually registering a `model_type` and `template`. See the [Custom Model Documentation](../Customization/Custom-model.md) for details.
- model_revision: Model version. Default is `None`.
- task_type: Default is `'causal_lm'`. Options include `'causal_lm'`, `'seq_cls'`, `'embedding'`, `'reranker'`, and `'generative_reranker'`. Examples for seq_cls can be found [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/seq_cls), and examples for embedding can be found [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/embedding).
- 🔥torch_dtype: Data type for model weights. Supported values: `float16`, `bfloat16`, `float32`. Default is `None`, which reads from the `config.json` file.
- attn_impl: Attention implementation. Options include `'sdpa'`, `'eager'`, `'flash_attn'`, `'flash_attention_2'`, `'flash_attention_3'`, etc. Default is `None`, reading from config.json.
  - Note: Not all attention implementations may be supported, depending on the underlying Transformers library's support for the specific model.
  - If set to `'flash_attn'` (for backward compatibility), `'flash_attention_2'` will be used.
- new_special_tokens: List of additional special tokens to be added. Default is `[]`. Example usage can be found [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/new_special_tokens).
  - Note: You can also pass a `.txt` file path where each line contains one special token.
- num_labels: Required for classification models (`--task_type seq_cls`). Indicates the number of labels. Default is `None`.
- problem_type: Required for classification models (`--task_type seq_cls`). Options: `'regression'`, `'single_label_classification'`, `'multi_label_classification'`. Default is `None`. If the model is a reward model or `num_labels=1`, it defaults to `'regression'`; otherwise, it defaults to `'single_label_classification'`.
- rope_scaling: Type of RoPE scaling. You can pass a string such as `'linear'`, `'dynamic'`, `'yarn'`, along with `max_model_len`, and MS-Swift will automatically configure the corresponding `rope_scaling`, overriding the value in `config.json`. Alternatively, pass a JSON string like `'{"factor": 2.0, "type": "yarn"}'`, which will directly replace the `rope_scaling` in `config.json`. Default is `None`.
- max_model_len: When using `rope_scaling` with a string input, this parameter helps calculate the RoPE scaling `factor`. Default is `None`. If specified, this value will **override** `max_position_embeddings` in `config.json`.
- device_map: Device placement configuration for the model, e.g., `'auto'`, `'cpu'`, a JSON string, or a JSON file path. This argument is **passed through** to the `from_pretrained` method in Transformers. Default is `None`, automatically determined based on available devices and distributed training setup.
- max_memory: When `device_map` is set to `'auto'` or `'sequential'`, model weights are allocated across devices according to `max_memory`, e.g., `--max_memory '{0: "20GB", 1: "20GB"}'`. Default is `None`. Passed through to the `from_pretrained` interface in Transformers.
- local_repo_path: Some models depend on GitHub repositories during loading, e.g., [deepseek-vl2](https://github.com/deepseek-ai/DeepSeek-VL2). To avoid network issues during `git clone`, you can use a local repository. This parameter takes the path to the local repo. Default is `None`.
- init_strategy: Strategy for initializing uninitialized parameters when loading a model (especially useful for custom architectures). Options: `'zero'`, `'uniform'`, `'normal'`, `'xavier_uniform'`, `'xavier_normal'`, `'kaiming_uniform'`, `'kaiming_normal'`, `'orthogonal'`. Default is `None`.

### Data Arguments
- 🔥dataset: A list of dataset IDs or paths. Default is `[]`. Each dataset should be specified in the format: `'dataset_id_or_path:subset#sample_count'`, where subset and sample count are optional. Local datasets support formats such as jsonl, csv, json, and folders. **Open-source datasets from the hub can be used offline by `git clone`-ing them locally and passing the local folder path**. For custom dataset formats, refer to the [Custom Dataset Documentation](../Customization/Custom-dataset.md). You can use multiple datasets by passing `--dataset <dataset1> <dataset2>`.
  - Subset: This parameter is only effective when the dataset is a dataset ID or a folder. If subsets were specified during registration and only one exists, that subset is selected by default; otherwise, the default subset `'default'` is used. You can select multiple subsets using `/`, e.g., `<dataset_id>:subset1/subset2`. You can also use `'all'` to select all registered subsets, e.g., `<dataset_id>:all`. See an example of registration [here](https://modelscope.cn/datasets/swift/garbage_competition).
  - Sampling count: By default, the full dataset is used. You can sample the dataset by specifying `#sample_count`. If the sample count is less than the total number of samples, random sampling without replacement is performed. If the sample count exceeds the total, the dataset is repeated `sample_count // total_samples` times, with an additional `sample_count % total_samples` samples randomly sampled. Note: For streaming datasets (`--streaming true`), only sequential sampling is performed. If `--dataset_shuffle false` is set, non-streaming datasets also use sequential sampling.
- 🔥val_dataset: A list of validation dataset IDs or paths. Default is `[]`.
- 🔥split_dataset_ratio: The ratio for splitting a validation set from the training set when `val_dataset` is not specified. Default is `0.`, meaning no splitting occurs.
  - Note: In "ms-swift<3.6", the default value was `0.01`.
- data_seed: Random seed for dataset operations. Default is `42`.
- 🔥dataset_num_proc: Number of processes for dataset preprocessing. Default is `1`.
- 🔥load_from_cache_file: Whether to load the dataset from cache. Default is `False`. **Recommended to set to `True` during actual training and `False` during debugging**.
  - Note: Note: In "ms-swift<3.9", the default value was `True`.
- dataset_shuffle: Whether to shuffle the training dataset. Default is `True`.
  - Note: **Shuffling in CPT/SFT involves two parts**: dataset-level shuffling (controlled by `dataset_shuffle`) and dataloader-level shuffling (controlled by `train_dataloader_shuffle`).
- val_dataset_shuffle: Whether to shuffle the validation dataset. Default is `False`.
- streaming: Whether to stream and process the dataset on-the-fly. Default is `False`.
  - Note: You must set `--max_steps` explicitly, as streaming datasets do not have a defined length. You can achieve behavior equivalent to `--num_train_epochs` by setting `--save_strategy epoch` and a large `max_steps`. Alternatively, set `max_epochs` to ensure training stops after the specified number of epochs, allowing model evaluation and checkpoint saving.
  - Note: Streaming avoids waiting for preprocessing by overlapping it with training. However, preprocessing is only performed on rank 0 and then distributed to other processes. **This is typically less efficient than non-streaming data sharding**. When the training `world_size` is large, preprocessing and data distribution can become a bottleneck.
- interleave_prob: Default is `None`. By default, multiple datasets are combined using `concatenate_datasets` from the datasets library. If this parameter is set, `interleave_datasets` is used instead. This is typically used for combining streaming datasets and is passed directly to `interleave_datasets`.
- stopping_strategy: Options are `"first_exhausted"` or `"all_exhausted"`. Default is `"first_exhausted"`. Passed to the `interleave_datasets` function.
- shuffle_buffer_size:  Specifies the shuffle buffer size for **streaming datasets**. Default is `1000`. Only effective when `dataset_shuffle` is `True`.
- download_mode: Dataset download mode. Options: `'reuse_dataset_if_exists'` or `'force_redownload'`. Default is `'reuse_dataset_if_exists'`.
  - Typically set to `--download_mode force_redownload` when encountering errors with hub datasets.
- columns: Used to map dataset column names so that the dataset conforms to the format accepted by `AutoPreprocessor`. See [Custom Dataset Documentation](../Customization/Custom-dataset.md) for supported formats. You can pass a JSON string, e.g., `'{"text1": "query", "text2": "response"}'`, meaning column `"text1"` is mapped to `"query"` and `"text2"` to `"response"`, which `AutoPreprocessor` can process. Default is `None`.
- strict: If `True`, any malformed row in the dataset will raise an error; otherwise, erroneous samples are dropped. Default is `False`. This is typically used for debugging.
- 🔥remove_unused_columns: Whether to remove unused columns from the dataset. Default is `True`.
  - If set to `False`, extra columns are passed to the trainer's `compute_loss` function, **facilitating custom loss functions that use additional dataset columns**.
  - Default value is `False` for GPRO.
- 🔥model_name: **Used only for self-cognition tasks**, and only affects the `swift/self-cognition` dataset. Replaces the `{{NAME}}` placeholder in the dataset. Provide the model's Chinese and English names, separated by space, e.g., `--model_name 小黄 'Xiao Huang'`. Default is `None`.
- 🔥model_author: Used only for self-cognition tasks, and only affects the `swift/self-cognition` dataset. Replaces the `{{AUTHOR}}` placeholder. Provide the model author's Chinese and English names, separated by space, e.g., `--model_author '魔搭' 'ModelScope'`. Default is `None`.
- custom_dataset_info: Path to a JSON file for custom dataset registration. See [Custom Dataset Guide](../Customization/Custom-dataset.md) and the [built-in dataset_info.json](https://github.com/modelscope/ms-swift/blob/main/swift/llm/dataset/data/dataset_info.json). Default is `[]`.


### Template Arguments
- 🔥template: The type of conversation template. Default is `None`, which automatically selects the corresponding template for the given model. See [List of Supported Models](./Supported-models-and-datasets.md) for mapping details.
- 🔥system: Custom system message field. Accepts either a string or a **path to a .txt file**. Default is `None`, using the default system message defined in the registered template.
  - Note: In terms of priority, the `system` field from the dataset takes precedence, followed by `--system`, and finally the `default_system` set in the registered template.
- 🔥max_length: Maximum token length after `tokenizer.encode` for a single data sample (to prevent OOM during training). Samples exceeding this limit are handled according to `truncation_strategy`. Default is `None`, meaning it's set to the model’s maximum supported sequence length (`max_model_len`).
  - In PPO, GRPO, and inference scenarios, `max_length` refers to `max_prompt_length`.
- truncation_strategy: How to handle samples exceeding `max_length`. Options: `'delete'`, `'left'`, `'right'`, representing deletion, left-truncation, and right-truncation respectively. Default is `'delete'`.
  - Note: For multimodal models, if `truncation_strategy` is set to `'left'` or `'right'` during training, **ms-swift preserves all image tokens and other modality-specific tokens**, which may lead to OOM.
- 🔥max_pixels: Maximum pixel count (H×W) for input images in multimodal models. Images exceeding this limit will be resized to avoid OOM during training. Default is `None` (no restriction).
  - Note: This parameter applies to all multimodal models. The Qwen2.5-VL specific parameter `MAX_PIXELS` (see bottom of doc) only affects Qwen2.5-VL.
- 🔥agent_template: Agent template that defines how the tool list `'tools'` is converted into the `'system'` message, how tool calls are extracted from model responses during inference/deployment, and the formatting of `{"role": "tool_call", "content": "xxx"}` and `{"role": "tool_response", "content": "xxx"}` in `messages`. Options include `'react_en'`, `'hermes'`, `'glm4'`, `'qwen_en'`, `'toolbench'`, etc. See [here](https://github.com/modelscope/ms-swift/blob/main/swift/plugin/agent_template/__init__.py) for more. Default is `None`, automatically selected based on model type. Refer to [Agent Documentation](./Agent-support.md).
- norm_bbox: Controls how bounding boxes ("bbox" in dataset, containing absolute coordinates; see [Custom Dataset Documentation](../Customization/Custom-dataset.md#grounding)) are normalized. Options: `'norm1000'` (scale coordinates to thousandths), `'none'` (no scaling). Default is `None`, automatically chosen based on model.
  - This also works correctly when **images are resized during training** (e.g., when `max_pixels` is set).
- use_chat_template: Whether to use a chat template or a generation template (the latter typically used in pretraining). Default is `True`.
  - Note: `swift pt` defaults to `False`, using the generation template. This setting provides good **compatibility with multimodal models**.
- 🔥padding_free: Flattens data within a batch to avoid padding, reducing GPU memory usage and accelerating training (**sequences in the same batch remain invisible to each other**). Default is `False`. Currently supported in CPT/SFT/DPO/GRPO/KTO/GKD.
  - Note: Use `padding_free` together with `--attn_impl flash_attn` and `transformers>=4.44`. See [this PR](https://github.com/huggingface/transformers/pull/31629) for details. (Same as packing.)
  - **Compared to packing, padding_free avoids extra preprocessing time, but packing offers faster training and more stable memory usage**.
- padding_side: Padding side when training with `batch_size >= 2`. Options: `'left'`, `'right'`. Default is `'right'`.
  - Note: PPO and GKD default to `'left'`. (During inference with `batch_size >= 2`, only left-padding is applied.)
- loss_scale: Loss weighting strategy for training tokens. Default is `'default'`, meaning all response tokens (including history) are weighted at 1 in cross-entropy loss (**tokens from system/user/multimodal inputs in messages, and `tool_response` in Agent training, are excluded from loss calculation**). Options include `'default'`, `'last_round'`, `'all'`, `'ignore_empty_think'`, `'last_round_with_ignore_empty_think'`, and agent-specific scales: `'react'`, `'hermes'`, `'qwen'`, `'agentflan'`, `'alpha_umi'`, etc. See [loss_scale.py](https://github.com/modelscope/ms-swift/blob/main/swift/plugin/loss_scale/loss_scale.py) for full list.
  - 'last_round': Only compute loss for the final round of response. (Commonly used; **RLHF defaults to this**)
  - 'all': Compute loss for all tokens. (**`swift pt` defaults to this**)
  - 'ignore_empty_think': Based on 'default', ignore loss computation for empty `'<think>\n\n</think>\n\n'` (as long as it matches the regex `'<think>\\s*</think>\\s*'`).
  - 'last_round_with_ignore_empty_think': Based on 'last_round', ignore loss computation for empty `'<think>\n\n</think>\n\n'` (as long as it matches the regex `'<think>\\s*</think>\\s*'`).
  - `'react'`, `'hermes'`, `'qwen'`: Based on `'default'`, adjust the loss weight of the `tool_call` portion to 2.
- sequence_parallel_size: Size for sequence parallelism. Default is 1. Currently supported in CPT/SFT/DPO/GRPO. Training scripts can be found [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/sequence_parallel).
- response_prefix: Prefix string for the model's response. For example, QwQ-32B sets `response_prefix` to `'\<think\>\n'`. This parameter only takes effect during inference. Default is `None`, automatically determined by the model.
- template_backend: Backend for template processing. Options are `'swift'` or `'jinja'`. Default is `'swift'`. If `'jinja'` is used, `apply_chat_template` from Transformers will be applied.
  - Note: The `'jinja'` backend only supports inference and does not support training (as it cannot determine the token ranges for loss computation).

### Generation Arguments

Refer to the [generation_config](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig) documentation.

- 🔥max_new_tokens: The maximum number of new tokens generated during inference. Defaults to None, meaning unlimited.
- temperature: Sampling temperature. Higher values increase output randomness. Default is `None`, reading from `generation_config.json`.
  - You can set `--temperature 0` or `--top_k 1` to disable randomness in generation.
- top_k: Top-k sampling parameter. Only the top `k` highest probability tokens are considered for generation. Default is `None`, reading from `generation_config.json`.
- top_p: Top-p (nucleus) sampling parameter. Only tokens whose cumulative probability reaches `top_p` are considered. Default is `None`, reading from `generation_config.json`.
- repetition_penalty: Penalty for repeated tokens. A value of 1.0 means no penalty. Default is `None`, reading from `generation_config.json`.
- num_beams: Number of beams for beam search. Default is 1.
- 🔥stream: Enable streaming output. Default is `None`, meaning `True` when using an interactive interface, and `False` during batch inference on datasets.
  - In "ms-swift<3.6", the default value was `False`.
- stop_words: Additional stop words besides the `eos_token`. Default is `[]`.
  - Note: The `eos_token` is removed from the output response, while additional stop words are preserved in the output.
- logprobs: Whether to return log probabilities. Default is `False`.
- top_logprobs: Number of top log probabilities to return. Default is `None`.

### Quantization Arguments

The following are parameters for quantizing models upon loading. See the [quantization documentation](https://huggingface.co/docs/transformers/main/en/main_classes/quantization) for details. These do not include `gptq` or `awq` quantization parameters used in `swift export`.

- 🔥quant_method: Quantization method used when loading the model. Options: `'bnb'`, `'hqq'`, `'eetq'`, `'quanto'`, `'fp8'`. Default is `None`.
  - If performing QLoRA training on already AWQ/GPTQ-quantized models, you do not need to set additional quantization parameters like `quant_method`.
- 🔥quant_bits: Number of bits for quantization. Default is `None`.
- hqq_axis: Axis for HQQ quantization. Default is `None`.
- bnb_4bit_compute_dtype: Computation data type for 4-bit BNB quantization. Options: `float16`, `bfloat16`, `float32`. Default is `None`, which uses the value of `torch_dtype`.
- bnb_4bit_quant_type: Type for 4-bit BNB quantization. Options: `'fp4'`, `'nf4'`. Default is `'nf4'`.
- bnb_4bit_use_double_quant: Whether to use double quantization. Default is `True`.
- bnb_4bit_quant_storage: Data type used to store quantized weights. Default is `None`.

## Atomic Arguments

### Seq2SeqTrainer Arguments

This list inherits from the Transformers `Seq2SeqTrainingArguments`, with ms-swift overriding certain default values. For arguments not listed here, please refer to the [official HF documentation](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Seq2SeqTrainingArguments).

- 🔥output_dir: Default is `None`, automatically set to `'output/<model_name>'`.
- 🔥gradient_checkpointing: Whether to use gradient checkpointing. Default is `True`. This significantly reduces GPU memory usage but slows down training.
- 🔥vit_gradient_checkpointing: For multimodal model training, whether to enable gradient checkpointing for the ViT (Vision Transformer) component. Default is `None`, meaning it follows the value of `gradient_checkpointing`. For an example, please refer to [here](https://github.com/modelscope/ms-swift/blob/main/examples/train/multimodal/vit_gradient_checkpointing.sh).
  - Note: When training multimodal models with LoRA and `--freeze_vit false`, if you see the warning: `UserWarning: None of the inputs have requires_grad=True. Gradients will be None`, try setting `--vit_gradient_checkpointing false` or open an issue. This issue does not occur in full-parameter training. (If this warning comes from the `ref_model` during RLHF LoRA training, it is normal.)
- 🔥deepspeed: Default is `None`. Can be set to `'zero0'`, `'zero1'`, `'zero2'`, `'zero3'`, `'zero2_offload'`, `'zero3_offload'` to use built-in DeepSpeed configurations in ms-swift. You can also pass a path to a custom DeepSpeed config file.
- zero_hpz_partition_size: Default is `None`. This enables ZeRO++ functionality—model sharding within nodes and data sharding across nodes. If encountering `grad_norm NaN`, try using `--torch_dtype float16`.
- deepspeed_autotp_size: DeepSpeed tensor parallelism size. Default is 1. To use DeepSpeed AutoTP, set `--deepspeed` to `'zero0'`, `'zero1'`, or `'zero2'`. (Note: Only supports full-parameter training)
- 🔥per_device_train_batch_size: Default is 1.
- 🔥per_device_eval_batch_size: Default is 1.
- 🔥gradient_accumulation_steps: Gradient accumulation steps. Default is `None`, meaning `gradient_accumulation_steps` is automatically calculated so that `total_batch_size >= 16`. Total batch size is computed as `per_device_train_batch_size * gradient_accumulation_steps * world_size`. In GRPO training, default is 1.
  - In CPT/SFT training, gradient accumulation has equivalent effects to using a larger batch size, but this equivalence does not hold in RLHF training.
- weight_decay:  Weight decay coefficient. Default is 0.1.
- adam_beta1: Default is 0.9.
- adam_beta2: Default is 0.95.
- 🔥learning_rate:  Learning rate. **Default is `1e-5` for full-parameter training, and `1e-4` for LoRA and other tuners**.
  - Tip: If you want to set `min_lr`, you can pass the arguments `--lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs '{"min_lr": 1e-6}'`.
- 🔥vit_lr: Specifies the learning rate for the ViT module when training multimodal models. Default is `None`, same as `learning_rate`.
  - Typically used together with `--freeze_vit` and `--freeze_aligner`.
- 🔥aligner_lr: Specifies the learning rate for the aligner module in multimodal models. Default is `None`, same as `learning_rate`.
- lr_scheduler_type: Type of learning rate scheduler. Default is `'cosine'`.
- lr_scheduler_kwargs: Additional arguments for the learning rate scheduler. Default is `None`.
- gradient_checkpointing_kwargs: Arguments passed to `torch.utils.checkpoint`. For example: `--gradient_checkpointing_kwargs '{"use_reentrant": false}'`. Default is `None`.
  - Note: When using DDP without DeepSpeed/FSDP and `gradient_checkpointing_kwargs` is `None`, it defaults to `'{"use_reentrant": false}'` to prevent errors.
- full_determinism: Ensures reproducible results during training. Note: This may negatively impact performance. Default is `False`.
- 🔥report_to: Default is `'tensorboard'`. You can specify multiple loggers, e.g., `--report_to tensorboard wandb swanlab`, or `--report_to all`.
- logging_first_step: Whether to log metrics at the first step. Default is `True`.
- logging_steps: Interval for logging. Default is 5.
- router_aux_loss_coef: Used in MoE model training to set the weight of auxiliary loss. Default is `0.`.
  - Note: In "ms-swift==3.7.0", the default was `None` (read from `config.json`), which changed in "ms-swift>=3.7.1".
- enable_dft_loss: Whether to use [DFT](https://arxiv.org/abs/2508.05629) (Dynamic Fine-Tuning) loss during SFT training. Default is `False`.
- enable_channel_loss: Enable channel-based loss. Default is `False`. Requires a `"channel"` field in the dataset. ms-swift groups and computes loss by this field (samples without `"channel"` are grouped into the default `None` channel). Dataset format reference: [channel loss](../Customization/Custom-dataset.md#channel-loss).  Channel loss is compatible with packing, padding_free, and loss_scale techniques.
  - Note: This argument is new in "ms-swift>=3.8". For "ms-swift<3.8", refer to v3.7 documentation.
- logging_dir: Directory for TensorBoard logs. Default is `None`, automatically set to `f'{self.output_dir}/runs'`.
- predict_with_generate: Use generation during evaluation. Default is `False`.
- metric_for_best_model: Default is `None`. If `predict_with_generate=False`, it's set to `'loss'`; otherwise `'rouge-l'` (in PPO training, no default; in GRPO, set to `'reward'`).
- greater_is_better: Default is `None`. Set to `False` if `metric_for_best_model` contains `'loss'`, otherwise `True`.
- max_epochs: Force training to stop after reaching `max_epochs`, then evaluate and save the model. Useful when using streaming datasets. Default is `None`.

Other important parameters:
- 🔥num_train_epochs: Number of training epochs. Default is 3.
- 🔥save_strategy: Strategy for saving checkpoints. Options: `'no'`, `'steps'`, `'epoch'`. Default is `'steps'`.
- 🔥save_steps: Default is 500.
- 🔥eval_strategy: Evaluation strategy. Default is `None`, following `save_strategy`.
  - If neither `val_dataset` nor `eval_dataset` is used and `split_dataset_ratio=0`, defaults to `'no'`.
- 🔥eval_steps: Default is `None`. If evaluation dataset exists, follows `save_steps`.
- 🔥save_total_limit: Maximum number of checkpoints to keep. Older checkpoints are deleted. Default is `None` (keep all).
- max_steps: Maximum number of training steps. Must be set when using streaming datasets. Default is -1.
- 🔥warmup_ratio: Default is 0.
- save_on_each_node: Save weights on every node. Default is `False`. Relevant in multi-node training.
  - Tip: In multi-node training, `output_dir` is typically set to a shared directory, so this parameter usually doesn't need to be set.
- save_only_model: Whether to save only model weights (excluding optimizer states, random seed states, etc.), reducing time and space overhead in full-parameter training. Default is `False`.
- 🔥resume_from_checkpoint: Path to resume training from. Default is `None`.
  - Tip: **To resume training, keep other parameters unchanged and add `--resume_from_checkpoint checkpoint_dir`**. Weights and states will be loaded by the trainer.
  - Note: `resume_from_checkpoint` loads model weights, optimizer state, random seed, and resumes training from the last step. Use `--resume_only_model` to load only model weights.
- resume_only_model: Default is `False`. If set to `True` along with `resume_from_checkpoint`, only model weights are resumed, ignoring optimizer state and random seed.
  - Note: In "ms-swift>=3.7", **`resume_only_model` skips already-trained data by default**, controlled via the `ignore_data_skip` argument. To restore "ms-swift<3.7" behavior, set `--ignore_data_skip true`.
- ignore_data_skip: When `resume_from_checkpoint` and `resume_only_model` are set, this controls whether to skip already-trained data and restore training states (epoch, step count, etc.). Default is `False`. If `True`, training starts from step 0 without loading previous states or skipping data.
- 🔥ddp_find_unused_parameters: Default is `None`.
- 🔥dataloader_num_workers: Default is `None`. On Windows, set to 0; otherwise, 1.
- dataloader_pin_memory: Default is `True`.
- dataloader_persistent_workers: Default is `False`.
- dataloader_prefetch_factor: Default is `None`. If `dataloader_num_workers > 0`, set to 10.
- train_dataloader_shuffle: Whether to shuffle the dataloader in CPT/SFT training. Default is `True`. Not effective for `IterableDataset`, which uses sequential loading.
- 🔥neftune_noise_alpha: Noise magnitude for NEFTune. Default is 0. Common values: 5, 10, 15.
- 🔥use_liger_kernel: Whether to enable the [Liger](https://github.com/linkedin/Liger-Kernel) kernel to accelerate training and reduce GPU memory consumption. Defaults to False. Example shell script can be found [here](https://github.com/modelscope/ms-swift/blob/main/examples/train/liger).
  - Note: Liger kernel does not support `device_map`. Use DDP or DeepSpeed for multi-GPU training. Currently, liger_kernel only supports `task_type='causal_lm'`.
- average_tokens_across_devices: Whether to average token counts across devices. If `True`, `num_tokens_in_batch` is synchronized via `all_reduce` for accurate loss computation. Default is `False`.
- max_grad_norm: Gradient clipping. Default is 1.
  - Note: The logged `grad_norm` reflects the value **before** clipping.
- push_to_hub: Push checkpoints to the hub. Default is `False`.
- hub_model_id: Model ID on the hub. Default is `None`.
- hub_private_repo: Whether the repo is private. Default is `False`.

### Tuner Arguments

- 🔥freeze_llm: This argument only takes effect for multimodal models and can be used in both full-parameter and LoRA training, but with different behaviors. In full-parameter training, setting `freeze_llm=True` freezes the LLM component's weights. In LoRA training with `target_modules='all-linear'`, setting `freeze_llm=True` prevents LoRA modules from being added to the LLM part. Default is `False`.
- 🔥freeze_vit: This argument only applies to multimodal models and behaves differently depending on the training mode. In full-parameter training, setting `freeze_vit=True` freezes the ViT (vision transformer) component's weights. In LoRA training with `target_modules='all-linear'`, setting `freeze_vit=True` prevents LoRA modules from being added to the ViT part. Default is `True`.
  - Note: **Here, "vit" refers not only to `vision_tower`, but also to `audio_tower`**. For Omni models, if you want to apply LoRA only to `vision_tower` and not `audio_tower`, you can modify [this code](https://github.com/modelscope/ms-swift/blob/a5d4c0a2ce0658cef8332d6c0fa619a52afa26ff/swift/llm/model/model_arch.py#L544-L554).
- 🔥freeze_aligner: This argument only affects multimodal models. In full-parameter training, setting `freeze_aligner=True` freezes the aligner (also known as projector) weights. In LoRA training with `target_modules='all-linear'`, setting `freeze_aligner=True` prevents LoRA modules from being added to the aligner component. Default is `True`.
- 🔥target_modules: Specifies which modules to apply LoRA to. Default is `['all-linear']`. You can also specify suffixes of modules, e.g., `--target_modules q_proj k_proj v_proj`. This argument is not limited to LoRA and can be used with other tuners.
  - Note: The behavior of `'all-linear'` differs between LLMs and multimodal LLMs. For standard LLMs, it automatically finds all linear layers except `lm_head` and attaches tuners. **For multimodal LLMs, tuners are by default only attached to the LLM component; this behavior can be controlled via `freeze_llm`, `freeze_vit`, and `freeze_aligner`**.
- 🔥target_regex: A regular expression to specify LoRA modules. Default is `None`. If provided, `target_modules` is ignored. For example: `--target_regex '^(language_model).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$'` applies LoRA to modules matching the pattern. This argument is not limited to LoRA and can be used with other tuners.
- target_parameters: List of parameter names (not module names) to replace with LoRA. Similar in behavior to `target_modules`, but operates at the parameter level. Requires "peft>=0.17.0". This is useful for models like Mixture-of-Experts (MoE) layers in Hugging Face Transformers, which may use `nn.Parameter` instead of `nn.Linear`.
- init_weights: Method for initializing weights. For LoRA: options are `'true'`, `'false'`, `'gaussian'`, `'pissa'`, `'pissa_niter_[number of iters]'`. For Bone: `'true'`, `'false'`, `'bat'`. Default is `'true'`.
- 🔥modules_to_save: Additional original model modules to include in training and saving, even after attaching a tuner. Default is `[]`. Applies to tuners beyond LoRA. For example: `--modules_to_save embed_tokens lm_head` enables training of `embed_tokens` and `lm_head` during LoRA training, and their weights will be saved in `adapter_model.safetensors`.

#### Full Arguments

- freeze_parameters: List of parameter name prefixes to freeze. Default is `[]`.
- freeze_parameters_regex: Regular expression to match parameters to freeze. Default is `None`.
- freeze_parameters_ratio: Proportion of parameters to freeze, from bottom to top layers. Default is `0`. Setting to `1` freezes all parameters; can be combined with `trainable_parameters` to specify trainable parts.
- trainable_parameters: Prefixes of additional parameters to keep trainable. Default is `[]`.
- trainable_parameters_regex: Regex to match additional trainable parameters. Default is `None`.
  - Note: `trainable_parameters` and `trainable_parameters_regex` have higher priority than `freeze_parameters`, `freeze_parameters_regex`, and `freeze_parameters_ratio`. For example, in full-parameter training, all modules are first set to trainable, then some are frozen based on the freeze rules, and finally some are re-enabled via `trainable_parameters` or `trainable_parameters_regex`.

#### LoRA

- 🔥lora_rank: Default is `8`.
- 🔥lora_alpha: Default is `32`.
- lora_dropout: Default is `0.05`.
- lora_bias: Defaults to `'none'`. Possible values are 'none', 'all'. If you want to make all biases trainable, you can set it to `'all'`.
- lora_dtype: Specifies the data type (dtype) for the LoRA modules. Supported values are 'float16', 'bfloat16', 'float32'. Default is None, which follows the default behavior of PEFT.
- 🔥use_dora: Defaults to `False`, indicating whether to use `DoRA`.
- use_rslora: Defaults to `False`, indicating whether to use `RS-LoRA`.
- 🔥lorap_lr_ratio: Parameter for LoRA+. Default is `None`. Recommended values: `10–16`. Setting this when using LoRA enables the LoRA+ variant.


##### LoRA-GA
- lora_ga_batch_size: The default value is `2`. The batch size used for estimating gradients during initialization in LoRA-GA.
- lora_ga_iters: The default value is `2`. The number of iterations for estimating gradients during initialization in LoRA-GA.
- lora_ga_max_length: The default value is `1024`. The maximum input length for estimating gradients during initialization in LoRA-GA.
- lora_ga_direction: The default value is `ArB2r`. The initial direction used for gradient estimation during initialization in LoRA-GA. Allowed values are: `ArBr`, `A2rBr`, `ArB2r`, and `random`.
- lora_ga_scale: The default value is `stable`. The scaling method for initialization in LoRA-GA. Allowed values are: `gd`, `unit`, `stable`, and `weightS`.
- lora_ga_stable_gamma: The default value is `16`. The gamma value when choosing `stable` scaling for initialization.

#### FourierFt

FourierFt uses three parameters: `target_modules`, `target_regex`, and `modules_to_save`, whose meanings are described in the documentation above. Additional parameters include:

- fourier_n_frequency: Number of frequencies in Fourier transform, an `int`, similar to `r` in LoRA. Default value is `2000`.
- fourier_scaling: Scaling value of matrix W, a `float`, similar to `lora_alpha` in LoRA. Default value is `300.0`.

#### BOFT

BOFT uses the three parameters `target_modules`, `target_regex`, and `modules_to_save`, whose meanings are described in the documentation above. Additional parameters include:

- boft_block_size: Size of BOFT blocks, default value is 4.
- boft_block_num: Number of BOFT blocks, cannot be used simultaneously with `boft_block_size`.
- boft_dropout: Dropout value for BOFT, default is 0.0.

#### Vera

Vera uses the three parameters `target_modules`, `target_regex`, and `modules_to_save`, whose meanings are described in the documentation above. Additional parameters include:

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

🔥Unsloth has no additional parameters; it can be supported by adjusting existing parameters, for example:

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

- 🔥vllm_gpu_memory_utilization: GPU memory ratio, ranging from 0 to 1. Default is `0.9`.
  - Note: For ms-swift versions earlier than 3.7, this parameter is named `gpu_memory_utilization`. The same applies to the following `vllm_` parameters. If you encounter parameter mismatch issues, please refer to the [ms-swift 3.6 documentation](https://swift.readthedocs.io/en/v3.6/Instruction/Command-line-parameters.html#vllm-arguments).
- 🔥vllm_tensor_parallel_size: Tensor parallelism size. Default is `1`.
- vllm_pipeline_parallel_size: Pipeline parallelism size. Default is `1`.
- vllm_data_parallel_size: Data parallelism size, default is `1`, effective in the `swift deploy/rollout` command.
  - In `swift infer`, use `NPROC_PER_NODE` to set the data parallelism (DP) degree. See the example [here](https://github.com/modelscope/ms-swift/blob/main/examples/infer/vllm/mllm_ddp.sh).
- vllm_enable_expert_parallel: Enable expert parallelism. Default is False.
- vllm_max_num_seqs: Maximum number of sequences to be processed in a single iteration. Default is `256`.
- 🔥vllm_max_model_len: The maximum sequence length supported by the model. Default is `None`, meaning it will be read from `config.json`.
- vllm_disable_custom_all_reduce: Disables the custom all-reduce kernel and falls back to NCCL. For stability, the default is `True`.
- vllm_enforce_eager: Determines whether vllm uses PyTorch eager mode or constructs a CUDA graph, default is `False`. Setting it to True can save memory but may affect efficiency.
- vllm_mm_processor_cache_gb: The size (in GiB) of the multimodal processor cache, used to store processed multimodal inputs (e.g., images, videos) to avoid redundant processing. Default is 4. Setting it to 0 disables the cache but may degrade performance (not recommended). This option takes effect only for multimodal models.
- vllm_disable_cascade_attn: Whether to forcibly disable the V1 engine’s cascade-attention implementation to avoid potential numerical issues. Defaults to False; vLLM’s internal heuristics determine whether cascade attention is actually used.
- 🔥vllm_limit_mm_per_prompt: Controls the use of multiple media in vllm, default is `None`. For example, you can pass in `--vllm_limit_mm_per_prompt '{"image": 5, "video": 2}'`.
- vllm_max_lora_rank: Default is `16`. This is the parameter supported by vllm for lora.
- vllm_quantization: vllm is able to quantize model with this argument, supported values can be found [here](https://docs.vllm.ai/en/latest/serving/engine_args.html).
- 🔥vllm_enable_prefix_caching: Enables vLLM's automatic prefix caching to save processing time for repeated prompt prefixes, improving inference efficiency. Default is `None`, following vLLM's default behavior.
  - The default value of this parameter is `False` in "ms-swift<3.9.1".
- vllm_use_async_engine: Whether to use the async engine under the vLLM backend. The deployment status (swift deploy) defaults to True, and other statuses default to False.
- vllm_reasoning_parser: Reasoning parser type, used for parsing the chain of thought content of reasoning models. Default is `None`. Only used for the `swift deploy` command. Available types can be found in the [vLLM documentation](https://docs.vllm.ai/en/latest/features/reasoning_outputs.html#streaming-chat-completions).
- vllm_engine_kwargs: Extra arguments for vllm, formatted as a JSON string. Default is `None`.

### SGLang Arguments
Parameter meanings can be found in the [sglang documentation](https://docs.sglang.ai/backend/server_arguments.html).

- 🔥sglang_tp_size: Tensor parallelism size. Default is 1.
- sglang_pp_size: Pipeline parallelism size. Default is 1.
- sglang_dp_size: Data parallelism size. Default is 1.
- sglang_ep_size: Expert parallelism size. Default is 1.
- sglang_enable_ep_moe: Whether to enable EP MoE. Default is False. This parameter has been removed in the latest version of SGLang.
- sglang_mem_fraction_static: The fraction of GPU memory used for static allocation (model weights and KV cache memory pool). If you encounter out-of-memory errors, try reducing this value. Default is None.
- sglang_context_length: The maximum context length of the model. Default is None, which means it will use the value from the model's `config.json`.
- sglang_disable_cuda_graph: Disables CUDA graph. Default is False.
- sglang_quantization: Quantization method. Default is None.
- sglang_kv_cache_dtype: Data type for KV cache storage. 'auto' means it will use the model's data type. 'fp8_e5m2' and 'fp8_e4m3' are supported on CUDA 11.8 and above. Default is 'auto'.
- sglang_enable_dp_attention: Enables data parallelism for attention and tensor parallelism for FFN. The data parallelism size (dp size) should be equal to the tensor parallelism size (tp size). Currently supports DeepSeek-V2/3 and Qwen2/3 MoE models. Default is False.
- sglang_disable_custom_all_reduce: Disables the custom all-reduce kernel and falls back to NCCL. For stability, the default is True.

### LMDeploy Arguments

Parameter meanings can be found in the [lmdeploy documentation](https://lmdeploy.readthedocs.io/en/latest/api/pipeline.html#turbomindengineconfig).

- 🔥lmdeploy_tp: tensor parallelism degree. Default is `1`.
- lmdeploy_session_len: Maximum session length. Default is `None`.
- lmdeploy_cache_max_entry_count: The percentage of GPU memory occupied by the k/v cache. Default is `0.8`.
- lmdeploy_quant_policy: Default is `0`. Set it to `4` or `8` when quantizing k/v to 4-bit or 8-bit, respectively.
- lmdeploy_vision_batch_size: The `max_batch_size` parameter passed to `VisionConfig`. Default is `1`.

### Merge Arguments

- 🔥merge_lora: Indicates whether to merge lora; this parameter supports lora, llamapro, and longlora, default is `False`. Example parameters [here](https://github.com/modelscope/ms-swift/blob/main/examples/export/merge_lora.sh).
- safe_serialization: Whether to save the model in safetensors format. Default is True.
- max_shard_size: Maximum size of a single storage file, default is '5GB'.

## Integration Arguments

### Training Arguments

Training arguments include the [base arguments](#base-arguments), [Seq2SeqTrainer arguments](#Seq2SeqTrainer-arguments), [tuner arguments](#tuner-arguments), and also include the following parts:

- add_version: Add directory to output_dir with `'<version>-<timestamp>'` to prevent weight overwrite, default is True.
- check_model: Check local model files for corruption or modification and give a prompt, default is True. **If in an offline environment, please set to False.**
- 🔥create_checkpoint_symlink: Creates additional checkpoint symlinks to facilitate writing automated training scripts. The symlink paths for `best_model` and `last_model` are `f'{output_dir}/best'` and `f'{output_dir}/last'` respectively.
- 🔥packing: Whether to use sequence packing to improve computational efficiency (better load balancing across nodes and processes, higher GPU utilization) and stabilize GPU memory usage. Default is False. Currently supported in CPT/SFT/DPO/KTO/GKD.
  - Note: When using packing, please combine it with `--attn_impl flash_attn` and ensure "transformers>=4.44". For details, see [this PR](https://github.com/huggingface/transformers/pull/31629).
  - Note: **Packing reduces the number of samples in the dataset; please adjust the gradient accumulation steps and learning rate accordingly**.
- packing_length: the length to use for packing. Defaults to None, in which case it is set to max_length.
- lazy_tokenize: Whether to use lazy tokenization. If set to `False`, all dataset samples will be tokenized (and for multimodal models, images will be loaded from disk) before training begins. Default is `None`: in LLM training, it defaults to `False`; in MLLM training, it defaults to `True` to save memory.
  - Note: If you want to perform image data augmentation, you need to set `lazy_tokenize` (or `streaming`) to True and modify the `encode` method in the Template class.
- cached_dataset: Use a cached dataset (generated with `swift export --to_cached_dataset true ...`) during training to avoid GPU time spent on tokenizing large datasets. Default is `[]`. Example: [here](https://github.com/modelscope/ms-swift/tree/main/examples/export/cached_dataset).
  - Note: cached_dataset supports `--packing` but does not support `--lazy_tokenize` or `--streaming`.
- use_logits_to_keep: Pass `logits_to_keep` in the `forward` method based on labels to reduce the computation and storage of unnecessary logits, thereby reducing memory usage and accelerating training. The default is `None`, which enables automatic selection.
- acc_strategy: Strategy for calculating accuracy during training and validation. Options are `seq`-level and `token`-level accuracy, with `token` as the default.
- max_new_tokens: Generation parameter override. The maximum number of tokens to generate when `predict_with_generate=True`, defaulting to 64.
- temperature: Generation parameter override. The temperature setting when `predict_with_generate=True`, defaulting to 0.
- optimizer: Custom optimizer name for the plugin, defaults to None. Optional optimizer reference: [here](https://github.com/modelscope/ms-swift/blob/main/swift/plugin/optimizer.py).
- loss_type: Custom loss function name defined in the plugin. Default is `None`, using the model's built-in loss function.
- metric: Custom metric name defined in the plugin. Default is `None`. When `predict_with_generate=True`, it defaults to `'nlg'`.
- eval_use_evalscope: Whether to use evalscope for evaluation, this parameter needs to be set to enable evaluation, refer to [example](../Instruction/Evaluation.md#evaluation-during-training). Default is False.
- eval_dataset: Evaluation datasets, multiple datasets can be set, separated by spaces
- eval_dataset_args: Evaluation dataset parameters in JSON format, parameters for multiple datasets can be set
- eval_limit: Number of samples from the evaluation dataset
- eval_generation_config: Model inference configuration during evaluation, in JSON format, default is `{'max_tokens': 512}`
- use_flash_ckpt: Whether to use [DLRover Flash Checkpoint](https://github.com/intelligent-machine-learning/dlrover). Default is `false`. If enabled, checkpoints are saved to memory synchronously, then persisted to storage asynchronously, the safetensors format is not supported currently. It's recommended to use this with the environment variable `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` to avoid CUDA OOM.
- early_stop_interval: The interval for early stopping. It will check if the best_metric has not improved within early_stop_interval periods (based on save_steps; it's recommended to set eval_steps and save_steps to the same value) and terminate training when this condition is met. The specific code implementation is in the callback plugin. Additionally, if you have more complex early stopping requirements, you can directly override the existing implementation in [callback.py](https://github.com/modelscope/ms-swift/blob/main/swift/plugin/callback.py).


#### SWANLAB

- swanlab_token: SwanLab's API key
- swanlab_project: SwanLab's project, which needs to be created in advance on the page: [https://swanlab.cn/space/~](https://swanlab.cn/space/~)
- swanlab_workspace: Defaults to `None`, will use the username associated with the API key
- swanlab_exp_name: Experiment name, can be left empty. If empty, the value of `--output_dir` will be used by default
- swanlab_lark_webhook_url: Defaults to None. SwanLab's Lark webhook URL, used for pushing experiment results to Lark.
- swanlab_lark_secret: Defaults to None. SwanLab's Lark secret, used for pushing experiment results to Lark.
- swanlab_mode: Optional values are `cloud` and `local`, representing cloud mode or local mode

### RLHF Arguments

RLHF arguments inherit from the [training arguments](#training-arguments).

- 🔥rlhf_type: Type of human alignment algorithm, supporting 'dpo', 'orpo', 'simpo', 'kto', 'cpo', 'rm', 'ppo', 'grpo' and 'gkd'. Default is 'dpo'.
- ref_model: Required for full parameter training when using the dpo, kto, ppo or grpo algorithms. Default is None, set to `--model`.
- ref_adapters: Default is `[]`. If you want to use the LoRA weights generated from SFT for DPO/KTO/GRPO, please use "ms-swift>=3.8" and set `--adapters sft_ckpt --ref_adapters sft_ckpt`. For resuming training from a checkpoint in this scenario, set `--resume_from_checkpoint rlhf_ckpt --ref_adapters sft_ckpt`.
- ref_model_type: Same as model_type. Default is None.
- ref_model_revision: Same as model_revision. Default is None.
- 🔥beta: A parameter controlling the degree of deviation from the reference model. A higher beta value indicates smaller deviation from the reference model. Default is `None`, with different default values depending on the RLHF algorithm: `2.0` for SimPO, `0.04` for GRPO, `0.5` for GKD, and `0.1` for other algorithms. See [documentation](./RLHF.md) for details.
- label_smoothing: Whether to use DPO smoothing, default value is `0`.
- max_completion_length: The maximum generation length in the GRPO/PPO/GKD algorithms. Default is 512.
- 🔥rpo_alpha: A parameter from the [RPO paper](https://arxiv.org/abs/2404.19733) that controls the weight of the NLL term (i.e., the SFT loss) in the loss function, where `loss = dpo_loss + rpo_alpha * sft_loss`. The paper recommends setting it to `1.`. The default value is `None`, meaning the SFT loss is not included by default.
- ld_alpha: From the [LD-DPO paper](https://arxiv.org/abs/2409.06411). Applies a weight α < 1 to the log-probabilities of tokens that lie beyond the shared prefix of the chosen and rejected responses, thereby mitigating length bias.
- discopop_tau: Temperature parameter τ from the [DiscoPOP paper](https://arxiv.org/abs/2406.08414) used to scale the log-ratio before the sigmoid modulation. Default 0.05; only active when loss_type is discopop.
  - **Note**: In "ms-swift<3.8", the default value was `1.`. Starting from "ms-swift>=3.8", the default has been changed to `None`.
- loss_type: Type of loss function. Default is None, with different defaults depending on the RLHF algorithm used.
  - DPO: Available options can be found in the [documentation](https://huggingface.co/docs/trl/main/en/dpo_trainer#loss-functions). Multiple values can be provided to enable mixed training ([MPO](https://arxiv.org/abs/2411.10442)); when multiple values are given, the loss_weights parameter must also be set. Default is `sigmoid`.
  - GRPO: See [GRPO parameters](#grpo-arguments) for reference.
- loss_weights: When setting multiple loss_type values in DPO training, this parameter specifies the weight for each loss component.
- cpo_alpha: Coefficient for nll loss in CPO/SimPO loss, default is `1.`.
- simpo_gamma: Reward margin term in the SimPO algorithm, with a paper-suggested setting of 0.5-1.5, default is `1.`.
- desirable_weight: In the KTO algorithm, this weight compensates for the imbalance between the number of desirable and undesirable samples by scaling the desirable loss. Default is `1.0`.
- undesirable_weight: In the KTO algorithm, this weight compensates for the imbalance between desirable and undesirable samples by scaling the undesirable loss. Default is `1.0`.
- center_rewards_coefficient: A coefficient used in reward model (RM) training to incentivize the model to output rewards with zero mean. See this [paper](https://huggingface.co/papers/2312.09244) for details. Recommended value: 0.01.
- loss_scale: Overrides the template parameter. During RLHF training, the default is `'last_round'`.
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
- teacher_deepspeed: Same as the deepspeed parameter, controls the DeepSpeed configuration for the teacher model. By default, uses the DeepSpeed configuration of the training model.


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
- beta: KL regularization coefficient; default 0.04. Setting it to 0 disables the reference model.
- per_device_train_batch_size: The training batch size per device. In GRPO, this refers to the batch size of completions during training.
- per_device_eval_batch_size: The evaluation batch size per device. In GRPO, this refers to the batch size of completions during evaluation.
- generation_batch_size: Batch size to use for generation. It defaults to the effective training batch size: per_device_train_batch_size * num_processes * gradient_accumulation_steps`
- steps_per_generation: Number of optimization steps per generation. It defaults to gradient_accumulation_steps. This parameter and generation_batch_size cannot be set simultaneously
- num_generations: The number of samples generated per prompt (corresponding to the G value in the paper). The sampling batch size (generation_batch_size or steps_per_generation × per_device_batch_size × num_processes) must be divisible by num_generations. The default value is 8.
- ds3_gather_for_generation: This parameter applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation, improving generation speed. However, disabling this option allows training models that exceed the VRAM capacity of a single GPU, albeit at the cost of slower generation. Disabling this option is not compatible with vLLM generation. The default is True.
- reward_funcs: Reward functions in the GRPO algorithm; options include `accuracy`,`format`,`cosine`,`repetition` and `soft_overlong`, as seen in `swift/plugin/orm.py`. You can also customize your own reward functions in the plugin. Default is `[]`.
- reward_weights: Weights for each reward function. The number should be equal to the sum of the number of reward functions and reward models. If `None`, all rewards are weighted equally with weight `1.0`.
  - Note: If `--reward_model` is included in GRPO training, it is added to the end of the reward functions.
- reward_model_plugin: The logic for the reward model, which defaults to ORM logic. For more information, please refer to [Customized Reward Models](./GRPO/DeveloperGuide/reward_model.md#custom-reward-model).
- dataset_shuffle: Whether to shuffle the dataset randomly. Default is True.
- truncation_strategy: The method to handle inputs exceeding `max_length`. Supported values are `delete` and `left`, representing deletion and left-side truncation respectively. The default is `left`. Note that for multi-modal models, left-side truncation may remove multi-modal tokens and cause a shape mismatch error during model forward. With the delete strategy, over-long or encoding-failed samples are discarded, and new samples are resampled from the original dataset to maintain the intended batch size.
- loss_type: The type of loss normalization. Options are ['grpo', 'bnpo', 'dr_grpo'], default is 'grpo'. For details, see this [pr](https://github.com/huggingface/trl/pull/3256#discussion_r2033213348)
- log_completions: Whether to log the model-generated content during training, to be used in conjunction with `--report_to wandb`, default is False.
  - Note: If `--report_to wandb` is not set, a `completions.jsonl` will be created in the checkpoint to store the generated content.
- use_vllm: Whether to use vLLM as the infer_backend for GRPO generation, default is False.
- vllm_mode: Mode to use for vLLM integration when `use_vllm` is set to `True`. Must be one of `server` or `colocate`
- vllm_mode server parameter
  - vllm_server_base_url: Base URL for the vLLM server (e.g., 'http://localhost:8000'). If provided, `vllm_server_host` " "and `vllm_server_port` are ignored. Default is None.
  - vllm_server_host: The host address of the vLLM server. Default is None.
  - vllm_server_port: The service port of the vLLM server. Default is 8000.
  - vllm_server_timeout: The connection timeout for the vLLM server. Default is 240 seconds.
  - vllm_server_pass_dataset: pass additional dataset information through to the vLLM server for multi-turn training.
  - async_generate: Use async rollout to improve train speed. Note that rollout will use the model updated in the previous round when enabled. Multi-turn scenarios are not supported. Default is `false`.
  - SWIFT_UPDATE_WEIGHTS_BUCKET_SIZE: An environment variable that controls the bucket size (in MB) for weight synchronization during full-parameter training in Server Mode. Default is 512 MB.
- vllm_mode colocate parameter (For more parameter support, refer to the [vLLM Arguments](#vLLM-Arguments).)
  - vllm_gpu_memory_utilization: vLLM passthrough parameter, default is 0.9.
  - vllm_max_model_len: vLLM passthrough parameter, the total length limit of model, default is None.
  - vllm_enforce_eager: vLLM passthrough parameter, default is False.
  - vllm_limit_mm_per_prompt: vLLM passthrough parameter, default is None.
  - vllm_enable_prefix_caching: A pass-through parameter for vLLM, default is True.
  - vllm_tensor_parallel_size: the tensor parallel size of vLLM engine, default is 1.
  - vllm_enable_lora: Enable the vLLM engine to load LoRA adapters; defaults to False. Used to accelerate weight synchronization during LoRA training. See the [documentation](./GRPO/GetStarted/GRPO.md#weight-sync-acceleration) for details.
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
- move_model_batches: When moving model parameters to fast inference frameworks such as vLLM/LMDeploy, determines how many batches to divide the layers into. The default is `None`, which means the entire model is not split. Otherwise, the model is split into `move_model_batches + 1` (non-layer parameters) + `1` (multi-modal component parameters) batches.
- multi_turn_scheduler: Multi-turn GRPO parameter; pass the corresponding plugin name, and make sure to implement it in plugin/multi_turn.py.
- max_turns: Maximum number of rounds for multi-turn GRPO. The default is None, which means there is no limit.
- dynamic_sample: Exclude data within the group where the reward standard deviation is 0, and additionally sample new data. Default is False.
- max_resample_times: Under the dynamic_sample setting, limit the number of resampling attempts to a maximum of 3. Default is 3 times.
- overlong_filter: Skip overlong truncated samples, which will not be included in loss calculation. Default is False.
The hyperparameters for the reward function can be found in the [Built-in Reward Functions section](#built-in-reward-functions).
- top_entropy_quantile: Only tokens whose entropy ranks within the specified top quantile are included in the loss calculation. The default is 1.0, which means low-entropy tokens are not filtered. For details, refer to the [documentation](./GRPO/AdvancedResearch/entropy_mask.md).
- log_entropy: Logs the entropy values during training. The default is False. For more information, refer to the [documentation](./GRPO/GetStarted/GRPO.md#logged-metrics).
- importance_sampling_level: Controls how the importance sampling ratio is computed. Options are `token` and `sequence`. In `token` mode, the raw per-token log-probability ratios are used. In `sequence` mode, the log-probability ratios of all valid tokens in the sequence are averaged to produce a single ratio per sequence. The [GSPO paper](https://www.arxiv.org/abs/2507.18071) uses sequence-level importance sampling to stabilize training. The default is `token`.


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

### Inference Arguments

Inference arguments include the [base arguments](#base-arguments), [merge arguments](#merge-arguments), [vLLM arguments](#vllm-arguments), [LMDeploy arguments](#LMDeploy-arguments), and also contain the following:

- 🔥infer_backend: Inference acceleration backend, supporting four inference engines: 'pt', 'vllm', 'sglang', and 'lmdeploy'. The default is 'pt'.
- 🔥max_batch_size: Effective when infer_backend is set to 'pt'; used for batch inference, with a default value of 1. If set to -1, there is no restriction.
- 🔥result_path: Path to store inference results (jsonl). The default is None, meaning results are saved in the checkpoint directory (with args.json file) or './result' directory. The final storage path will be printed in the command line.
  - Note: If the `result_path` file already exists, it will be appended to.
- write_batch_size: The batch size for writing results to result_path. Defaults to 1000. If set to -1, there is no restriction.
- metric: Evaluate the results of the inference, currently supporting 'acc' and 'rouge'. The default is None, meaning no evaluation is performed.
- val_dataset_sample: Number of samples from the inference dataset, default is None.
- reranker_use_activation: Whether to apply sigmoid activation after the score during reranker inference. Default is True.

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

### Rollout Arguments
The rollout parameters inherit from the [deployment parameters](#deployment-arguments).
- multi_turn_scheduler: The scheduler for multi-turn GRPO training. Pass the corresponding plugin name, and ensure the implementation is added in `plugin/multi_turn.py`. Default is `None`. See [documentation](./GRPO/DeveloperGuide/multi_turn.md) for details.
- max_turns: Maximum number of turns in multi-turn GRPO training. Default is `None`, meaning no limit.
- vllm_enable_lora: Enable the vLLM engine to load LoRA adapters; defaults to False. Used to accelerate weight synchronization during LoRA training. See the [documentation](./GRPO/GetStarted/GRPO.md#weight-sync-acceleration) for details.
- vllm_max_lora_rank: LoRA parameter for the vLLM engine. Must be greater than or equal to the training lora_rank; it is recommended to set them equal. Defaults to 16.

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
- to_cached_dataset: pre-tokenize the dataset and export it in advance, default is False. See the example [here](https://github.com/modelscope/ms-swift/tree/main/examples/export/cached_dataset).
  - Note: data packing is performed during training, not in this step.
- to_ollama: Generate the Modelfile required by Ollama. Default is False.
- 🔥to_mcore: Convert weights from HF format to Megatron format. Default is False.
- to_hf: Convert weights from Megatron format to HF format. Default is False.
- mcore_model: Path to the mcore format model. Default is None.
- mcore_adapters: List of paths to mcore format model adapters, default is empty list.
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

In addition to the parameters listed above, some models support additional model-specific arguments. The meanings of these parameters can usually be found in the corresponding model's official repository or its inference code. **MS-Swift includes these parameters to ensure that the trained model aligns with the behavior of the official inference implementation**.

- Model-specific parameters can be set via `--model_kwargs` or environment variables. For example: `--model_kwargs '{"fps_max_frames": 12}'` or `FPS_MAX_FRAMES=12`.
- Note: If you specify model-specific parameters during training, please also set the corresponding parameters during inference to achieve optimal performance.


### qwen2_vl, qvq, qwen2_5_vl, mimo_vl, keye_vl, keye_vl_1_5
These parameters have the same meaning as in `qwen_vl_utils<0.0.12` or the `qwen_omni_utils` library. See [here](https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L24) for details. MS-Swift adjusts these constant values to control image resolution and video frame rate, preventing out-of-memory (OOM) errors during training.

- IMAGE_FACTOR: Default is 28.
- MIN_PIXELS: Default is `4 * 28 * 28`. Minimum image resolution. It is recommended to set this as a multiple of 28×28.
- 🔥MAX_PIXELS: Default is `16384 * 28 * 28`. Maximum image resolution. It is recommended to set this as a multiple of 28×28.
- MAX_RATIO: Default is 200.
- VIDEO_MIN_PIXELS: Default is `128 * 28 * 28`. Minimum resolution per frame in a video. Recommended to be a multiple of 28×28.
- 🔥VIDEO_MAX_PIXELS: Default is `768 * 28 * 28`. Maximum resolution per frame in a video. Recommended to be a multiple of 28×28.
- VIDEO_TOTAL_PIXELS: Default is `24576 * 28 * 28`.
- FRAME_FACTOR: Default is 2.
- FPS: Default is 2.0.
- FPS_MIN_FRAMES: Default is 4. Minimum number of frames extracted from a video clip.
- 🔥FPS_MAX_FRAMES: Default is 768. Maximum number of frames extracted from a video clip.
- 🔥QWENVL_BBOX_FORMAT: (ms-swift>=3.9.1) Specifies whether to use `'legacy'` or `'new'` format for grounding. The `'legacy'` format is: `<|object_ref_start|>a dog<|object_ref_end|><|box_start|>(432,991),(1111,2077)<|box_end|>`. The `'new'` format refers to: [Qwen3-VL Cookbook](https://github.com/QwenLM/Qwen3-VL/blob/main/cookbooks/2d_grounding.ipynb). For dataset formatting, see the [Grounding Dataset Format Documentation](../Customization/Custom-dataset.md#grounding). Default: `'legacy'`.
  - Note: This environment variable applies to Qwen2/2.5/3-VL and Qwen2.5/3-Omni series models.

### qwen2_audio
- SAMPLING_RATE: Default is 16000

### qwen2_5_omni, qwen3_omni
qwen2_5_omni not only includes the model-specific parameters of qwen2_5_vl and qwen2_audio, but also contains the following parameter:
- USE_AUDIO_IN_VIDEO: Whether to use audio information from video. Default is `False`.
- 🔥ENABLE_AUDIO_OUTPUT: Defaults to None, which means the value from `config.json` will be used. If training with zero3, please set it to False.
  - Tip: ms-swift only fine-tunes the "thinker" component; it is recommended to set this to `False` to reduce GPU memory usage (only the thinker part of the model structure will be created).


### qwen3_vl
The parameter meanings are the same as in the `qwen_vl_utils>=0.0.14` library — see here: https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py#L24. By passing the following environment variables you can override the library's global default values:

- SPATIAL_MERGE_SIZE: default 2.
- IMAGE_MIN_TOKEN_NUM: default `4`, denotes the minimum number of image tokens per image.
- 🔥IMAGE_MAX_TOKEN_NUM: default `16384`, denotes the maximum number of image tokens per image. (used to avoid OOM)
- VIDEO_MIN_TOKEN_NUM: default `128`, denotes the minimum number of video tokens per frame.
- 🔥VIDEO_MAX_TOKEN_NUM: default `768`, denotes the maximum number of video tokens per frame. (used to avoid OOM)
- MAX_RATIO: default 200.
- FRAME_FACTOR: default 2.
- FPS: default 2.0.
- FPS_MIN_FRAMES: default 4, denotes the minimum number of sampled frames for a video segment.
- 🔥FPS_MAX_FRAMES: default 768, denotes the maximum number of sampled frames for a video segment. (used to avoid OOM)


### internvl, internvl_phi3
For the meaning of the arguments, please refer to [here](https://modelscope.cn/models/OpenGVLab/Mini-InternVL-Chat-2B-V1-5)
- MAX_NUM: Default is 12
- INPUT_SIZE: Default is 448

### internvl2, internvl2_phi3, internvl2_5, internvl3, internvl3_5
For the meaning of the arguments, please refer to [here](https://modelscope.cn/models/OpenGVLab/InternVL2_5-2B)
- MAX_NUM: Default is 12
- INPUT_SIZE: Default is 448
- VIDEO_MAX_NUM: Default is 1, which is the MAX_NUM for videos
- VIDEO_SEGMENTS: Default is 8

### minicpmv2_6, minicpmo2_6, minicpmv4
- MAX_SLICE_NUMS: Default is 9, refer to [here](https://modelscope.cn/models/OpenBMB/MiniCPM-V-2_6/file/view/master?fileName=config.json&status=1)
- VIDEO_MAX_SLICE_NUMS: Default is 1, which is the MAX_SLICE_NUMS for videos, refer to [here](https://modelscope.cn/models/OpenBMB/MiniCPM-V-2_6)
- MAX_NUM_FRAMES: Default is 64, refer to [here](https://modelscope.cn/models/OpenBMB/MiniCPM-V-2_6)

### minicpmo2_6
- INIT_TTS: Default is False
- INIT_AUDIO: Default is False

### ovis1_6, ovis2
- MAX_PARTITION: Default is 9, refer to [here](https://github.com/AIDC-AI/Ovis/blob/d248e34d755a95d24315c40e2489750a869c5dbc/ovis/model/modeling_ovis.py#L312)

### ovis2_5

The meanings of the following parameters can be found in the example code [here](https://modelscope.cn/models/AIDC-AI/Ovis2.5-2B).

- MIX_PIXELS: int type, default is `448 * 448`.
- MAX_PIXELS: int type, default is `1344 * 1792`. If OOM (out of memory) occurs, you can reduce this value.
- VIDEO_MAX_PIXELS: int type, default is `896 * 896`.
- NUM_FRAMES: default is 8. Used for video frame sampling.

### mplug_owl3, mplug_owl3_241101
- MAX_NUM_FRAMES: Default is 16, refer to [here](https://modelscope.cn/models/iic/mPLUG-Owl3-7B-240728)

### xcomposer2_4khd
- HD_NUM: Default is 55, refer to [here](https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-xcomposer2-4khd-7b)

### xcomposer2_5
- HD_NUM: Default is 24 when the number of images is 1. Greater than 1, the default is 6. Refer to [here](https://modelscope.cn/models/AI-ModelScope/internlm-xcomposer2d5-7b/file/view/master?fileName=modeling_internlm_xcomposer2.py&status=1#L254)

### video_cogvlm2
- NUM_FRAMES: Default is 24, refer to [here](https://github.com/zai-org/CogVLM2/blob/main/video_demo/inference.py#L22)

### phi3_vision
- NUM_CROPS: Default is 4, refer to [here](https://modelscope.cn/models/LLM-Research/Phi-3.5-vision-instruct)

### llama3_1_omni
- N_MELS: Default is 128, refer to [here](https://github.com/ictnlp/LLaMA-Omni/blob/544d0ff3de8817fdcbc5192941a11cf4a72cbf2b/omni_speech/infer/infer.py#L57)

### video_llava
- NUM_FRAMES: Default is 16


## Other Environment Variables

- CUDA_VISIBLE_DEVICES: Controls which GPU to use. By default, all GPUs are used.
- ASCEND_RT_VISIBLE_DEVICES: Controls which NPU (effective for ASCEND cards) are used. By default, all NPUs are used.
- MODELSCOPE_CACHE: Controls the cache path. (Recommended to set this value during multi-node training to ensure all nodes use the same dataset cache.)
- NPROC_PER_NODE: Pass-through for the `--nproc_per_node` parameter in torchrun. The default is 1. If the `NPROC_PER_NODE` or `NNODES` environment variables are set, torchrun is used to start training or inference.
- PYTORCH_CUDA_ALLOC_CONF: It is recommended to set it to `'expandable_segments:True'`, which reduces GPU memory fragmentation. For more details, please refer to the [PyTorch documentation](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management).
- MASTER_PORT: Pass-through for the `--master_port` parameter in torchrun. The default is 29500.
- MASTER_ADDR: Pass-through for the `--master_addr` parameter in torchrun.
- NNODES: Pass-through for the `--nnodes` parameter in torchrun.
- NODE_RANK: Pass-through for the `--node_rank` parameter in torchrun.
- LOG_LEVEL: The log level, default is 'INFO'. You can set it to 'WARNING', 'ERROR', etc.
- SWIFT_DEBUG: When set to `'1'` during `engine.infer(...)`, PtEngine will print the contents of `input_ids` and `generate_ids` to facilitate debugging and alignment.
- VLLM_USE_V1: Used to switch between V0 and V1 versions of vLLM.
- SWIFT_TIMEOUT: (ms-swift >= 3.10) If the multimodal dataset contains image URLs, this parameter controls the timeout for fetching images, defaulting to 20 seconds.
- ROOT_IMAGE_DIR: (ms-swift>=3.8) The root directory for image (multimodal) resources. By setting this parameter, relative paths in the dataset can be interpreted relative to `ROOT_IMAGE_DIR`. By default, paths are relative to the current working directory.
