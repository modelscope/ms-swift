# Command Line Arguments

## Table of Contents

- [sft Parameters](#sft-parameters)
- [dpo Parameters](#dpo-parameters)
- [merge-lora infer Parameters](#merge-lora-infer-parameters)
- [export Parameters](#export-parameters)
- [app-ui Parameters](#app-ui-parameters)
- [deploy Parameters](#deploy-parameters)

## sft Parameters

- `--model_type`: The model type you select, default is `None`. `model_type` specifies default information like `lora_target_modules`, `template_type` for the corresponding model. You can fine-tune by only specifying `model_type`. The corresponding `model_id_or_path` will use the default settings, download from ModelScope, and use the default cache path. One of `model_type` or `model_id_or_path` must be specified. Available `model_type` options can be found in [Supported Models](Supported-models-datasets.md#Models).
- `--model_id_or_path`: The `model_id` of the model on ModelScope Hub or a local path, default is `None`. If the passed `model_id_or_path` has been registered, `model_type` will be inferred based on `model_id_or_path`. If not registered, you need to also specify `model_type`, e.g. `--model_type <model_type> --model_id_or_path <model_id_or_path>`.
- `--model_revision`: The version number corresponding to `model_id` on ModelScope Hub, default is `None`. If `model_revision` is `None`, use the revision registered in `MODEL_MAPPING`. Otherwise, force use of the `model_revision` passed from command line.
- `--sft_type`: Fine-tuning method, default is `'lora'`. Options include: 'lora', 'full', 'longlora', 'qalora'. If using qlora, you need to set `--sft_type lora --quantization_bit 4`.
- `--freeze_parameters`: When sft_type is set to 'full', freeze the bottommost parameters of the model. Range is 0. ~ 1., default is `0.`. This provides a compromise between lora and full fine-tuning.
- `--additional_trainable_parameters`: In addition to freeze_parameters, only allowed when sft_type is 'full', default is `[]`. For example, if you want to train embedding layer in addition to 50% of parameters, you can set `--freeze_parameters 0.5 --additional_trainable_parameters transformer.wte`, all parameters starting with `transformer.wte` will be activated.
- `--tuner_backend`: Backend support for lora, qlora, default is `'swift'`. Options include: 'swift', 'peft'.
- `--template_type`: Type of dialogue template used, default is `'AUTO'`, i.e. look up `template` in `MODEL_MAPPING` based on `model_type`. Available `template_type` options can be found in `TEMPLATE_MAPPING.keys()`.
- `--output_dir`: Directory to store ckpt, default is `'output'`. We will append `model_type` and fine-tuning version number to this directory, allowing users to do multiple comparative experiments on different models without changing the `output_dir` command line argument. If you don't want to append this content, specify `--add_output_dir_suffix false`.
- `--add_output_dir_suffix`: Default is `True`, indicating that a suffix of `model_type` and fine-tuning version number will be appended to the `output_dir` directory. Set to `False` to avoid this behavior.
- `--ddp_backend`: Backend support for distributed training, default is `None`. Options include: 'nccl', 'gloo', 'mpi', 'ccl'.
- `--seed`: Global seed, default is `42`. Used to reproduce training results.
- `--resume_from_checkpoint`: For resuming training from checkpoint, default is `None`. You can set this to the path of a checkpoint, e.g. `'output/qwen-7b-chat/vx-xxx/checkpoint-xxx'`, to resume training.
- `--dtype`: torch_dtype when loading base model, default is `'AUTO'`, i.e. intelligently select dtype: if machine does not support bf16, use fp16; if `MODEL_MAPPING` specifies torch_dtype for corresponding model, use its dtype; otherwise use bf16. Options include: 'bf16', 'fp16', 'fp32'.
- `--dataset`: Datasets to use for training, default is `[]`. Available datasets can be found in [Supported Datasets](Supported-models-datasets.md#Datasets). To use multiple datasets for training, separate them with ',' or ' ', e.g. `--dataset alpaca-en,alpaca-zh` or `--dataset alpaca-en alpaca-zh`.
- `--dataset_seed`: Seed for dataset processing, default is `42`. Exists as random_state, does not affect global seed.
- `--dataset_test_ratio`: Ratio for splitting subdataset into train and validation sets, default is `0.01`. If the subdataset is already split into train and validation, this parameter has no effect.
- `--train_dataset_sample`: The number of samples for training set, default is `-1`, which means using the entire training set for training.
- `--val_dataset_sample`: Sampling of validation set, default is `None`, automatically selects appropriate dataset size for validation. If set to `-1`, use the full validation set.
- `--system`: System used in dialogue template, default is `None`, i.e. use the model's default system. If set to '', no system is used.
- `--max_length`: Maximum token length, default is `2048`. Avoids OOM issues caused by individual overly long samples. When `--truncation_strategy delete` is specified, samples exceeding max_length will be deleted. When `--truncation_strategy truncation_left` is specified, the leftmost tokens will be truncated: `input_ids[-max_length:]`. If set to -1, no limit.
- `--truncation_strategy`: Default is `'delete'` which removes sentences exceeding max_length from dataset. `'truncation_left'` will truncate excess text from the left, which may truncate special tokens and affect performance, not recommended.
- `--check_dataset_strategy`: Default is `'none'`, i.e. no checking. If training an LLM model, `'warning'` is recommended as data check strategy. If your training target is sentence classification etc., setting to `'none'` is recommended.
- `--custom_train_dataset_path`: Default is `[]`. See [Customization](Customization.md) for details.
- `--custom_val_dataset_path`: Default is `[]`. See [Customization](Customization.md) for details.
- `--self_cognition_sample`: Sampling number for self-cognition dataset. Default is `0`. When setting this to >0, you also need to specify `--model_name`, `--model_author`. See [Self-Cognition Fine-tuning Best Practices](Self-cognition-best-practice.md) for more info.
- `--model_name`: Default is `[None, None]`. If self-cognition dataset sampling is enabled (i.e. self_cognition_sample>0), you need to pass two values, representing the model's Chinese and English names respectively. E.g. `--model_name 小黄 'Xiao Huang'`.
- `--model_author`: Default is `[None, None]`. If self-cognition dataset sampling is enabled, you need to pass two values, representing the author's Chinese and English names respectively. E.g. `--model_author 魔搭 ModelScope`.
- `--quantization_bit`: Specifies whether to quantize and number of quantization bits, default is `0`, i.e. no quantization. To use 4bit qlora, set `--sft_type lora --quantization_bit 4`
- `--bnb_4bit_comp_dtype`: When doing 4bit quantization, we need to dequantize during model forward and backward passes. This specifies the torch_dtype after dequantization. Default is `'AUTO'`, i.e. consistent with `dtype`. Options: 'fp16', 'bf16', 'fp32'. Has no effect when quantization_bit is 0.
- `--bnb_4bit_quant_type`: Quantization method for 4bit quantization, default is `'nf4'`. Options: 'nf4', 'fp4'. Has no effect when quantization_bit is 0.
- `--bnb_4bit_use_double_quant`: Whether to enable double quantization for 4bit quantization, default is `True`. Has no effect when quantization_bit is 0.
- `--bnb_4bit_quant_storage`: Default vlaue `None`.This sets the storage type to pack the quanitzed 4-bit prarams. Has no effect when quantization_bit is 0.
- `--lora_target_modules`: Specify lora modules, default is `['DEFAULT']`. If lora_target_modules is passed `'DEFAULT'` or `'AUTO'`, look up `lora_target_modules` in `MODEL_MAPPING` based on `model_type` (default specifies qkv). If passed `'ALL'`, all Linear layers (excluding head) will be specified as lora modules. If passed `'EMBEDDING'`, Embedding layer will be specified as lora module. If memory allows, setting to 'ALL' is recommended. You can also set `['ALL', 'EMBEDDING']` to specify all Linear and embedding layers as lora modules. This parameter only takes effect when `sft_type` is 'lora'.
- `--lora_rank`: Default is `8`. Only takes effect when `sft_type` is 'lora'.
- `--lora_alpha`: Default is `32`. Only takes effect when `sft_type` is 'lora'.
- `--lora_dropout_p`: Default is `0.05`, only takes effect when `sft_type` is 'lora'.
- `--lora_bias_trainable`: Default is `'none'`, options: 'none', 'all'. Set to `'all'` to make all biases trainable.
- `--lora_modules_to_save`: Default is `[]`. If you want to train embedding, lm_head, or layer_norm, you can set this parameter, e.g. `--lora_modules_to_save EMBEDDING LN lm_head`. If passed `'EMBEDDING'`, Embedding layer will be added to `lora_modules_to_save`. If passed `'LN'`, `RMSNorm` and `LayerNorm` will be added to `lora_modules_to_save`.
- `--lora_dtype`: Default is `'fp32'`, specifies dtype for lora modules. If `AUTO`, follow dtype of original module. Options: 'fp16', 'bf16', 'fp32', 'AUTO'.
- `--use_dora`: Default is `False`, whether to use `DoRA`.
- `--use_rslora`: Default is `False`, whether to use `RS-LoRA`.
- `--neftune_noise_alpha`: The noise coefficient added by `NEFTune` can improve performance of instruction fine-tuning, default is `None`. Usually can be set to 5, 10, 15. See [related paper](https://arxiv.org/abs/2310.05914).
- `--neftune_backend`: The backend of `NEFTune`, default uses `transformers` library, may encounter incompatibility when training VL models, in which case it's recommended to specify as `swift`.
- `--gradient_checkpointing`: Whether to enable gradient checkpointing, default is `True`. This can be used to save memory, although it slightly reduces training speed. Has significant effect when max_length and batch_size are large.
- `--deepspeed`: Specifies the path to the deepspeed configuration file or directly passes in configuration information in json format, default is `None`, i.e. deepspeed is not enabled. Deepspeed can save memory. We have written a default [ZeRO-2 configuration file](https://github.com/modelscope/swift/blob/main/swift/llm/ds_config/zero2.json), [ZeRO-3 configuration file](https://github.com/modelscope/swift/blob/main/swift/llm/ds_config/zero3.json). You only need to specify 'default-zero2' to use the default zero2 config file; specify 'default-zero3' to use the default zero3 config file.
- `--batch_size`: Batch_size during training, default is `1`. Increasing batch_size can improve GPU utilization, but won't necessarily improve training speed, because within a batch, shorter sentences need to be padded to the length of the longest sentence in the batch, introducing invalid computations.
- `--eval_batch_size`: Batch_size during evaluation, default is `None`, i.e. set to 1 when `predict_with_generate` is True, set to `batch_size` when False.
- `--num_train_epochs`: Number of epochs to train, default is `1`. If `max_steps >= 0`, this overrides `num_train_epochs`. Usually set to 3 ~ 5.
- `--max_steps`: Max_steps for training, default is `-1`. If `max_steps >= 0`, this overrides `num_train_epochs`.
- `--optim`: Default is `'adamw_torch'`.
- `--learning_rate`: Default is `None`, i.e. set to 1e-4 if `sft_type` is lora, set to 1e-5 if `sft_type` is full.
- `--weight_decay`: Default is `0.01`.
- `--gradient_accumulation_steps`: Gradient accumulation, default is `None`, set to `math.ceil(16 / self.batch_size / world_size)`. `total_batch_size =  batch_size * gradient_accumulation_steps * world_size`.
- `--max_grad_norm`: Gradient clipping, default is `0.5`.
- `--predict_with_generate`: Whether to use generation for evaluation, default is `False`. If set to False, evaluate using `loss`. If set to True, evaluate using `ROUGE-L` and other metrics. Generative evaluation takes a long time, choose carefully.
- `--lr_scheduler_type`: Default is `'linear'`, options: 'linear', 'cosine', 'constant', etc.
- `--warmup_ratio`: Proportion of warmup in total training steps, default is `0.05`.
- `--eval_steps`: Evaluate every this many steps, default is `50`.
- `--save_steps`: Save every this many steps, default is `None`, i.e. set to `eval_steps`.
- `--save_only_model`: Whether to save only model parameters, without saving intermediate states needed for checkpoint resuming, default is `None`, i.e. if `sft_type` is 'lora' and not using deepspeed (`deepspeed` is `None`), set to False, otherwise set to True (e.g. using full fine-tuning or deepspeed).
- `--save_total_limit`: Number of checkpoints to save, default is `2`, i.e. save best and last checkpoint. If set to -1, save all checkpoints.
- `--logging_steps`: Print training information (e.g. loss, learning_rate, etc.) every this many steps, default is `5`.
- `--dataloader_num_workers`: Default is `1`.
- `--push_to_hub`: Whether to sync push trained checkpoint to ModelScope Hub, default is `False`.
- `--hub_model_id`: Model_id to push to on ModelScope Hub, default is `None`, i.e. set to `f'{model_type}-{sft_type}'`. You can set this to model_id or repo_name. We will infer user_name based on hub_token. If the remote repository to push to does not exist, a new repository will be created, otherwise the previous repository will be reused. This parameter only takes effect when `push_to_hub` is set to True.
- `--hub_token`: SDK token needed for pushing. Can be obtained from [https://modelscope.cn/my/myaccesstoken](https://modelscope.cn/my/myaccesstoken), default is `None`, i.e. obtained from environment variable `MODELSCOPE_API_TOKEN`. This parameter only takes effect when `push_to_hub` is set to True.
- `--hub_private_repo`: Whether to set the permission of the pushed model repository on ModelScope Hub to private, default is `False`. This parameter only takes effect when `push__to_hub` is set to True.
- `--push_hub_strategy`: Push strategy, default is `'push_best'`. Options include: 'end', 'push_best', 'push_last', 'checkpoint', 'all_checkpoints'. 'push_best' means when saving weights each time, push and overwrite the best model from before, 'push_last' means when saving weights each time, push and overwrite the last weights from before, 'end' means only push the best model at the end of training. This parameter only takes effect when `push_to_hub` is set to True.
- `--test_oom_error`: Used to detect whether training will cause OOM, default is `False`. If set to True, will sort the training set in descending order by max_length, easy for OOM testing. This parameter is generally used for testing, use carefully.
- `--disable_tqdm`: Whether to disable tqdm, useful when launching script with `nohup`. Default is `False`, i.e. enable tqdm.
- `--lazy_tokenize`: If set to False, preprocess all text before `trainer.train()`. If set to True, delay encoding text, reducing preprocessing wait and memory usage, useful when processing large datasets. Default is `None`, i.e. we intelligently choose based on template type, usually set to False for LLM models, set to True for multimodal models (to avoid excessive memory usage from loading images and audio).
- `--preprocess_num_proc`: Use multiprocessing when preprocessing dataset (tokenizing text). Default is `1`. Same as `lazy_tokenize` command line argument, used to solve slow preprocessing issue. But this strategy cannot reduce memory usage, so if dataset is huge, `lazy_tokenize` is recommended. Recommended values: 4, 8. Note: When using qwen-audio, this parameter will be forced to 1, because qwen-audio's preprocessing function uses torch's multiprocessing, which will cause compatibility issues.
- `--use_flash_attn`: Whether to use flash attn, default is `None`. Installation steps for flash_attn can be found at [https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention). Models supporting flash_attn can be found in [LLM Supported Models](Supported-models-datasets.md).
- `--ignore_args_error`: Whether to ignore Error thrown by command line parameter errors, default is `False`. Set to True if need to copy code to notebook to run.
- `--check_model_is_latest`: Check if model is latest, default is `True`. Set this to `False` if you need to train offline.
- `--logging_dir`: Default is `None`. I.e. set to `f'{self.output_dir}/runs'`, representing path to store tensorboard files.
- `--report_to`: Default is `['tensorboard']`.
- `--acc_strategy`: Default is `'token'`, options include: 'token', 'sentence'.
- `--save_on_each_node`: Takes effect during multi-machine training, default is `True`.
- `--save_strategy`: Strategy for saving checkpoint, default is `'steps'`, options include: 'steps', 'no'.
- `--save_safetensors`: Default is `True`.
- `--max_new_tokens`: Default is `2048`. This parameter only takes effect when `predict_with_generate` is set to True.
- `--do_sample`: Default is `True`. This parameter only takes effect when `predict_with_generate` is set to True.
- `--temperature`: Default is `0.3`. This parameter only takes effect when `do_sample` is set to True. This parameter will be used as default value in deployment parameters.
- `--top_k`: Default is `20`. This parameter only takes effect when `do_sample` is set to True. This parameter will be used as default value in deployment parameters.
- `--top_p`: Default is `0.7`. This parameter only takes effect when `do_sample` is set to True. This parameter will be used as default value in deployment parameters.
- `--repetition_penalty`: Default is `1.`. This parameter will be used as default value in deployment parameters.
- `--num_beams`: Default is `1`. This parameter only takes effect when `predict_with_generate` is set to True.
- `--gpu_memory_fraction`: Default is `None`. This parameter aims to run training under a specified maximum available GPU memory percentage, used for extreme testing.
- `--train_dataset_mix_ratio`: Default is `0`. This defines how to mix datasets for training. When specifying this, training set will be mixed `train_dataset_mix_ratio` times with the general knowledge dataset specified by `train_dataset_mix_ds`, making total dataset length reach `train_dataset_sample`.
- `--train_dataset_mix_ds`: Default is `ms-bench`. General knowledge dataset used to prevent knowledge forgetting.
- `--use_loss_scale`: Default is `False`. When taking effect, strengthens loss weight of some Agent fields (Action/Action Input part) to enhance CoT, has no effect in regular SFT scenarios.

### FSDP Parameters

- `--fsdp`: Default value `''`, the FSDP type, please check [this documentation](https://huggingface.co/docs/transformers/v4.39.3/en/main_classes/trainer#transformers.TrainingArguments.fsdp) for details.

- `--fsdp_config`: Default value `None`, the FSDP config file path.

### LoRA+ Fine-tuning Parameters

- `--lora_lr_ratio`: Default `None`, recommended value `10~16`, specify this parameter when using lora to enable lora+.

### GaLore Fine-tuning Parameters

- `--use_galore: bool` : Default False, whether to use GaLore.
- `--galore_target_modules: Union[str, List[str]]` : Default None, apply GaLore to attention and mlp when not passed.
- `--galore_rank: int` : Default 128, rank value for GaLore.
- `--galore_update_proj_gap: int` : Default 50, update interval for decomposition matrix.
- `--galore_scale: int` : Default 1.0, matrix weight coefficient.
- `--galore_proj_type: str` : Default `std`, GaLore matrix decomposition type.
- `--galore_optim_per_parameter: bool` : Default False, whether to set a separate optimizer for each Galore target Parameter.
- `--galore_with_embedding: bool` : Default False, whether to apply GaLore to embedding.

### LLaMA-PRO Fine-tuning Parameters

- `--llamapro_num_new_blocks`: Default `4`, total number of new layers inserted.
- `--llamapro_num_groups`: Default `None`, how many groups to insert new_blocks into, if `None` then equals `llamapro_num_new_blocks`, i.e. each new layer is inserted into original model separately.

### AdaLoRA Fine-tuning Parameters

The following parameters take effect when `sft_type` is set to `adalora`. AdaLoRA's `target_modules` and other parameters inherit from lora's corresponding parameters, but the `lora_dtype` parameter has no effect.

- `--adalora_target_r`: Default `8`, AdaLoRA's average rank.
- `--adalora_init_r`: Default `12`, AdaLoRA's initial rank.
- `--adalora_tinit`: Default `0`, AdaLoRA's initial warmup.
- `--adalora_tfinal`: Default `0`, AdaLoRA's final warmup.
- `--adalora_deltaT`: Default `1`, AdaLoRA's step interval.
- `--adalora_beta1`: Default `0.85`, AdaLoRA's EMA parameter.
- `--adalora_beta2`: Default `0.85`, AdaLoRA's EMA parameter.
- `--adalora_orth_reg_weight`: Default `0.5`, AdaLoRA's regularization parameter.

### IA3 Fine-tuning Parameters

The following parameters take effect when `sft_type` is set to `ia3`.

- `--ia3_target_modules`: Specify IA3 target modules, default is `['DEFAULT']`. See `lora_target_modules` for specific meaning.
- `--ia3_feedforward_modules`: Specify the Linear name of IA3's MLP, this name must be in `ia3_target_modules`.
- `--ia3_modules_to_save`: Additional modules participating in IA3 training. See meaning of `lora_modules_to_save`.

## dpo Parameters

dpo parameters inherit from sft parameters, with the following added parameters:

- `--ref_model_type`: Type of reference model, available `model_type` options can be found in `MODEL_MAPPING.keys()`.
- `--ref_model_id_or_path`: The local cache dir for reference model, default `None`.
- `--max_prompt_length`: Maximum prompt length, this parameter is passed to DPOTrainer, setting prompt length to not exceed this value, default is `1024`.
- `--beta`: Regularization term for DPO logits, default is 0.1.
- `--label_smoothing`: Whether to use DPO smoothing, default is 0, generally set between 0~0.5.
- `--loss_type`: DPOloss type, supports 'sigmoid', 'hinge', 'ipo', 'kto_pair', default is 'sigmoid'.
- `--sft_beta`: Whether to add sft loss in DPO, default is 0.1, supports [0, 1) interval, final loss is `(1-sft_beta)*KL_loss + sft_beta * sft_loss`.

## merge-lora infer Parameters

- `--model_type`: Default is `None`, see `sft.sh command line arguments` for parameter details.
- `--model_id_or_path`: Default is `None`, see `sft.sh command line arguments` for parameter details. Recommended to use model_type to specify.
- `--model_revision`: Default is `None`. See `sft.sh command line arguments` for parameter details. If `model_id_or_path` is None or a local model directory, this parameter has no effect.
- `--sft_type`: Default is `'lora'`, see `sft.sh command line arguments` for parameter details.
- `--template_type`: Default is `'AUTO'`, see `sft.sh command line arguments` for parameter details.
- `--infer_backend`: Options are 'AUTO', 'vllm', 'pt'. Default uses 'AUTO', for intelligent selection, i.e. if `ckpt_dir` is not passed or using full fine-tuning, and vllm is installed and model supports vllm, then use vllm engine, otherwise use native torch for inference. vllm environment setup can be found in [VLLM Inference Acceleration and Deployment](VLLM-inference-acceleration-and-deployment.md), vllm supported models can be found in [Supported Models](Supported-models-datasets.md).
- `--ckpt_dir`: Required, value is the checkpoint path saved in SFT stage, e.g. `'/path/to/your/vx-xxx/checkpoint-xxx'`.
- `--load_args_from_ckpt_dir`: Whether to read model configuration info from `sft_args.json` file in `ckpt_dir`. Default is `True`.
- `--load_dataset_config`: This parameter only takes effect when `--load_args_from_ckpt_dir true`. I.e. whether to read dataset related configuration from `sft_args.json` file in `ckpt_dir`. Default is `False`.
- `--eval_human`: Whether to evaluate using validation set portion of dataset or manual evaluation. Default is `None`, for intelligent selection, if no datasets (including custom datasets) are passed, manual evaluation will be used. If datasets are passed, dataset evaluation will be used.
- `--seed`: Default is `42`, see `sft.sh command line arguments` for parameter details.
- `--dtype`: Default is `'AUTO`, see `sft.sh command line arguments` for parameter details.
- `--dataset`: Default is `[]`, see `sft.sh command line arguments` for parameter details.
- `--dataset_seed`: Default is `42`, see `sft.sh command line arguments` for parameter details.
- `--dataset_test_ratio`: Default is `0.01`, see `sft.sh command line arguments` for parameter details.
- `--val_dataset_sample`: Represents number of validation set samples to evaluate and display, default is `10`.
- `--system`: Default is `None`. See `sft.sh command line arguments` for parameter details.
- `--max_length`: Default is `-1`. See `sft.sh command line arguments` for parameter details.
- `--truncation_strategy`: Default is `'delete'`. See `sft.sh command line arguments` for parameter details.
- `--check_dataset_strategy`: Default is `'none'`, see `sft.sh command line arguments` for parameter details.
- `--custom_train_dataset_path`: Default is `[]`. See README.md `Custom Datasets` module for details.
- `--custom_val_dataset_path`: Default is `[]`. See README.md `Custom Datasets` module for details.
- `--quantization_bit`: Default is 0. See `sft.sh command line arguments` for parameter details.
- `--bnb_4bit_comp_dtype`: Default is `'AUTO'`.  See `sft.sh command line arguments` for parameter details. If `quantization_bit` is set to 0, this parameter has no effect.
- `--bnb_4bit_quant_type`: Default is `'nf4'`.  See `sft.sh command line arguments` for parameter details. If `quantization_bit` is set to 0, this parameter has no effect.
- `--bnb_4bit_use_double_quant`: Default is `True`.  See `sft.sh command line arguments` for parameter details. If `quantization_bit` is set to 0, this parameter has no effect.
- `--bnb_4bit_quant_storage`: Default value `None`.See `sft.sh command line arguments` for parameter details. If `quantization_bit` is set to 0, this parameter has no effect.
- `--max_new_tokens`: Maximum number of new tokens to generate, default is `2048`.
- `--do_sample`: Whether to use greedy generation or sampling generation, default is `True`.
- `--temperature`: Default is `0.3`. This parameter only takes effect when `do_sample` is set to True. This parameter will be used as default value in deployment parameters.
- `--top_k`: Default is `20`. This parameter only takes effect when `do_sample` is set to True. This parameter will be used as default value in deployment parameters.
- `--top_p`: Default is `0.7`. This parameter only takes effect when `do_sample` is set to True. This parameter will be used as default value in deployment parameters.
- `--repetition_penalty`: Default is `1.`. This parameter will be used as default value in deployment parameters.
- `--num_beams`: Default is `1`.
- `--use_flash_attn`: Default is `None`, i.e. 'auto'. See `sft.sh command line arguments` for parameter details.
- `--ignore_args_error`: Default is `False`, see `sft.sh command line arguments` for parameter details.
- `--stream`: Whether to use streaming output, default is `True`. This parameter only takes effect when using dataset evaluation and verbose is True.
- `--merge_lora`: Whether to merge lora weights into base model and save full weights, default is `False`. Weights will be saved in the same level directory as `ckpt_dir`, e.g. `'/path/to/your/vx-xxx/checkpoint-xxx-merged'` directory.
- `--merge_device_map`: device_map used when merge-lora, default is `None`, to reduce memory usage, use `auto` only during merge-lora process, otherwise default is `cpu`.
- `--save_safetensors`: Whether to save as `safetensors` file or `bin` file. Default is `True`.
- `--overwrite_generation_config`: Whether to save the generation_config used for evaluation as `generation_config.json` file, default is `None`. If `ckpt_dir` is specified, set to `True`, otherwise set to `False`. The generation_config file saved during training will be overwritten.
- `--verbose`: If set to False, use tqdm style inference. If set to True, output inference query, response, label. Default is `None`, for auto selection, i.e. when `len(val_dataset) >= 100`, set to False, otherwise set to True. This parameter only takes effect when using dataset evaluation.
- `--gpu_memory_utilization`: Parameter for initializing vllm engine `EngineArgs`, default is `0.9`. This parameter only takes effect when using vllm. VLLM inference acceleration and deployment can be found in [VLLM Inference Acceleration and Deployment](VLLM-inference-acceleration-and-deployment.md).
- `--tensor_parallel_size`: Parameter for initializing vllm engine `EngineArgs`, default is `1`. This parameter only takes effect when using vllm.
- `--max_model_len`: Override model's max_model__len, default is `None`. This parameter only takes effect when using vllm.

## export Parameters

export parameters inherit from infer parameters, with the following added parameters:

- `--merge_lora`: Default is `False`. This parameter is already defined in InferArguments, not a new parameter. Whether to merge lora weights into base model and save full weights. Weights will be saved in the same level directory as `ckpt_dir`, e.g. `'/path/to/your/vx-xxx/checkpoint-xxx-merged'` directory.
- `--quant_bits`: Number of bits for quantization. Default is `0`, i.e. no quantization. If you set `--quant_method awq`, you can set this to `4` for 4bits quantization. If you set `--quant_method gptq`, you can set this to `2`,`3`,`4`,`8` for corresponding bits quantization. If quantizing original model, weights will be saved in `f'{args.model_type}-{args.quant_method}-int{args.quant_bits}'` directory. If quantizing fine-tuned model, weights will be saved in the same level directory as `ckpt_dir`, e.g. `f'/path/to/your/vx-xxx/checkpoint-xxx-{args.quant_method}-int{args.quant_bits}'` directory.
- `--quant_method`: Quantization method, default is `'awq'`. Options are 'awq', 'gptq'.
- `--dataset`: This parameter is already defined in InferArguments, for export it means quantization dataset. Default is `[]`. Recommended to set `--dataset ms-bench-mini`. This dataset contains multilingual content (mainly Chinese) of high quality, with good effect for quantizing Chinese models. You can also set `--dataset pileval`, using autoawq default quantization dataset, the language of this dataset is English. More details: including how to customize quantization dataset, can be found in [LLM Quantization Documentation](LLM-quantization.md).
- `--quant_n_samples`: Quantization parameter, default is `None`, set to `256` if using awq quantization, set to `1024` if using gptq quantization. When set to `--quant_method awq`, if OOM occurs during quantization, you can moderately reduce `--quant_n_samples` and `--quant_seqlen`. `--quant_method gptq` generally does not encounter quantization OOM.
- `--quant_seqlen`: Quantization parameter, default is `2048`.
- `--quant_device_map`: Default is `'cpu'`, to save memory. You can specify 'cuda:0', 'auto', 'cpu', etc., representing the device to load model during quantization.
- `--push_to_hub`: Default is `False`. Whether to push the final `ckpt_dir` to ModelScope Hub. If you specify `merge_lora`, full parameters will be pushed; if you also specify `quant_bits`, quantized model will be pushed.
- `--hub_model_id`: Default is `None`. Model_id to push to on ModelScope Hub. If `push_to_hub` is set to True, this parameter must be set.
- `--hub_token`: Default is `None`. See `sft.sh command line arguments` for parameter details.
- `--hub_private_repo`: Default is `False`. See `sft.sh command line arguments` for parameter details.
- `--commit_message`: Default is `'update files'`.

## eval parameters

The eval parameters inherit from the infer parameters, and additionally include the following parameters:

- `--name`: Default is `None`. The name of the evaluation, the final evaluation results will be stored in a folder named `{{model_type}-{name}}`.

- `--eval_dataset`: The official dataset for evaluation, the default value is `['ceval', 'gsm8k', 'arc']`, and `mmlu` and `bbh` datasets are also supported. If you only need to evaluate a custom dataset, you can set this parameter to `no`.

- `--eval_limit`: The number of samples for each sub-dataset of the evaluation set, default is `None` which means full evaluation.

- `--eval_few_shot`: The number of few-shot instances for each sub-dataset of the evaluation set, default is `None` which means using the default configuration of the dataset.

- `--custom_eval_config`: Use a custom dataset for evaluation, this should be a local file path, the file format is described in [Custom Evaluation Set](./LLM-eval#Custom-Evaluation-Set).

- `--eval_use_cache`: Whether to use the evaluation cache, if True, the eval process will only refresh the eval results. Default `False`.

- `--eval_url`: The url of OpenAI standard model service. For example: `http://127.0.0.1:8000/v1`.

  ```shell
  swift eval --eval_url http://127.0.0.1:8000/v1 --eval_is_chat_model true --model_type gpt4 --eval_token xxx
  ```

- `--eval_is_chat_model`: If `eval_url` is not None, `eval_is_chat_model` must be passed to tell the url calls a chat or a base model.
- `--eval_token`: The token of the `eval_url`, default value `EMPTY` means token is not needed.

## app-ui Parameters

app-ui parameters inherit from infer parameters, with the following added parameters:

- `--server_name`: Default is `'127.0.0.1'`. Passed to the `demo.queue().launch(...)` function of gradio.
- `--server_port`: Default is `7860`. Passed to the `demo.queue().launch(...)` function of gradio.
- `--share`: Default is `False`. Passed to the `demo.queue().launch(...)` function of gradio.

## deploy Parameters

deploy parameters inherit from infer parameters, with the following added parameters:

- `--host`: Default is `'127.0.0.1`.
- `--port`: Default is `8000`.
- `--ssl_keyfile`: Default is `None`.
- `--ssl_certfile`: Default is `None`.
