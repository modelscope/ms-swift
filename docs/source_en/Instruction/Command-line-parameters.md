# Command Line Arguments

## Table of Contents

- [sft Parameters](#sft-parameters)
- [pt Parameters](#pt-parameters)
- [rlhf Parameters](#rlhf-parameters)
- [infer merge-lora Parameters](#infer-merge-lora-parameters)
- [export Parameters](#export-parameters)
- [eval Parameters](#eval-parameters)
- [app-ui Parameters](#app-ui-parameters)
- [deploy Parameters](#deploy-parameters)

## sft Parameters
- `--🔥model_type`: Represents the selected model type, default is `None`. `model_type` specifies the default `target_modules`, `template_type`, and other information for the corresponding model. You can fine-tune by specifying only `model_type`. The corresponding `model_id_or_path` will use default settings, and the model will be downloaded from ModelScope and use the default cache path. One of model_type and model_id_or_path must be specified. You can see the list of available `model_type` [here](Supported-models-datasets.md#Models). You can set the `USE_HF` environment variable to control downloading models and datasets from the HF Hub, see [HuggingFace Ecosystem Compatibility Documentation](../LLM/Compat-HF.md).
- `--🔥model_id_or_path`: Represents the `model_id` in the ModelScope/HuggingFace Hub or a local path for the model, default is `None`. If the provided `model_id_or_path` has already been registered, the `model_type` will be inferred based on the `model_id_or_path`. If it has not been registered, both `model_type` and `model_id_or_path` must be specified, e.g. `--model_type <model_type> --model_id_or_path <model_id_or_path>`.
- `--model_revision`: The version number corresponding to `model_id` on ModelScope Hub, default is `None`. If `model_revision` is `None`, use the revision registered in `MODEL_MAPPING`. Otherwise, force use of the `model_revision` passed from command line.
- `--local_repo_path`: Some models rely on a GitHub repo for loading. To avoid network issues during `git clone`, you can directly use the local repo. This parameter requires input of the local repo path, and defaults to `None`. These models include:
  - mPLUG-Owl model: `https://github.com/X-PLUG/mPLUG-Owl`
  - DeepSeek-VL model: `https://github.com/deepseek-ai/DeepSeek-VL`
  - YI-VL model: `https://github.com/01-ai/Yi`
  - LLAVA model: `https://github.com/haotian-liu/LLaVA.git`
- `--🔥sft_type`: Fine-tuning method, default is `'lora'`. Options include: 'lora', 'full', 'longlora', 'adalora', 'ia3', 'llamapro', 'adapter', 'vera', 'boft', 'fourierft', 'reft'. If using qlora, you need to set `--sft_type lora --quantization_bit 4`.
- `--packing`: pack the dataset length to `max-length`, default `False`.
- `--full_determinism`: Fix all the values in training, default `False`.
- `--auto_find_batch_size`: Auto find batch size according to the GPU memory, default `False`.
- `--streaming`: Whether to use iterable dataset, Default `False`.
- `--freeze_parameters`: When sft_type is specified as 'full', the layers prefixed with freeze_parameters will be frozen. The default value is `[]`. For example: `--freeze_parameters visual`.
- `--🔥freeze_vit`: When sft_type is set to 'full' and a multimodal model is being trained, the parameters of vit can be frozen by setting this parameter to True. The default value is `False`.
- `--freeze_parameters_ratio`: When sft_type is set to 'full', freeze the bottommost parameters of the model. Range is 0. ~ 1., default is `0.`. This provides a compromise between lora and full fine-tuning.
- `--additional_trainable_parameters`: In addition to freeze_parameters, only allowed when sft_type is 'full', default is `[]`. For example, if you want to train embedding layer in addition to 50% of parameters, you can set `--freeze_parameters_ratio 0.5 --additional_trainable_parameters transformer.wte`, all parameters starting with `transformer.wte` will be activated. You can also set `--freeze_parameters_ratio 1 --additional_trainable_parameters xxx` to customize the trainable layers.
- `--tuner_backend`: Backend support for lora, qlora, default is `'peft'`. Options include: 'swift', 'peft', 'unsloth'.
- `--🔥template_type`: Type of dialogue template used, default is `'AUTO'`, i.e. look up `template` in `MODEL_MAPPING` based on `model_type`. Available `template_type` options can be found in `TEMPLATE_MAPPING.keys()`.
- `--🔥output_dir`: Directory to store ckpt, default is `'output'`. We will append `model_type` and fine-tuning version number to this directory, allowing users to do multiple comparative experiments on different models without changing the `output_dir` command line argument. If you don't want to append this content, specify `--add_output_dir_suffix false`.
- `--add_output_dir_suffix`: Default is `True`, indicating that a suffix of `model_type` and fine-tuning version number will be appended to the `output_dir` directory. Set to `False` to avoid this behavior.
- `--ddp_backend`: Backend support for distributed training, default is `None`. Options include: 'nccl', 'gloo', 'mpi', 'ccl'.
- `--ddp_timeout`: DDP timeout. Default `1800` seconds.
- `--seed`: Global seed, default is `42`. Used to reproduce training results.
- `--🔥resume_from_checkpoint`: Used for resuming training from a checkpoint, default is `None`. You can set it to the path of the checkpoint, for example: `--resume_from_checkpoint output/qwen-7b-chat/vx-xxx/checkpoint-xxx`, to resume training from that point. Supports adjusting `--resume_only_model` to only read the model file during checkpoint continuation.
- `--resume_only_model`: Default is `False`, which means strict checkpoint continuation, this will read the weights of the model, optimizer, lr_scheduler, and the random seeds stored on each device, and continue training from the last paused steps. If set to `True`, it will only read the weights of the model.
- `--dtype`: torch_dtype when loading base model, default is `'AUTO'`, i.e. intelligently select dtype: if machine does not support bf16, use fp16; if `MODEL_MAPPING` specifies torch_dtype for corresponding model, use its dtype; otherwise use bf16. Options include: 'bf16', 'fp16', 'fp32'.
- `--model_kwargs`: Used for passing additional parameters to the multimodal model, for example: `'{"hd_num": 16}'`. You can either pass a JSON string or directly pass a dictionary. The default is `None`. In addition to using this parameter, you can also pass it through environment variables, for example: `HD_NUM=16`.
- `--🔥dataset`: Used to select the training dataset, default is `[]`. You can see the list of available datasets [here](Supported-models-datasets.md#Datasets). If you need to train with multiple datasets, you can use ',' or ' ' to separate them, for example: `--dataset alpaca-en,alpaca-zh` or `--dataset alpaca-en alpaca-zh`. It supports Modelscope Hub/HuggingFace Hub/local paths, subset selection, and dataset sampling. The specified format for each dataset is as follows: `[HF or MS::]{dataset_name} or {dataset_id} or {dataset_path}[:subset1/subset2/...][#dataset_sample]`. The simplest case requires specifying only dataset_name, dataset_id, or dataset_path. Customizing datasets can be found in the [Customizing and Extending Datasets document](Customization.md#custom-dataset)
  - Supports MS and HF hub, as well as dataset_sample. For example, 'MS::alpaca-zh#2000', 'HF::jd-sentiment-zh#2000' (the default hub used is controlled by the `USE_UF` environment variable, default is MS).
  - More fine-grained control over subsets: It uses the subsets specified during registration by default (if not specified during registration, it uses 'default'). For example, 'sharegpt-gpt4'. If subsets are specified, it uses the corresponding subset of the dataset. For example, 'sharegpt-gpt4:default/V3_format#2000'. Here, the `default` and `V3_format` sub-datasets are used, separated by '/', and 2000 entries are selected.
  - Support for dataset_id. For example, 'AI-ModelScope/alpaca-gpt4-data-zh#2000', 'HF::llm-wizard/alpaca-gpt4-data-zh#2000', 'hurner/alpaca-gpt4-data-zh#2000', 'HF::shibing624/alpaca-zh#2000'. If the dataset_id has been registered, it will use the preprocessing function, subsets, split, etc. specified during registration. Otherwise, it will use `SmartPreprocessor`, support 5 dataset formats, and use 'default' subsets, with split set to 'train'. The supported dataset formats can be found in the [Customizing and Extending Datasets document](Customization.md#custom-dataset).
  - Support for dataset_path. For example, '1.jsonl#5000' (if it is a relative path, it is relative to the running directory).
- `--val_dataset`: Specify separate validation datasets with the same format of the `dataset` argument, default is `[]`. If using `val_dataset`, the `dataset_test_ratio` will be ignored.
- `--dataset_seed`: The seed used to specify the dataset processing is set by default to `None`, meaning it is designated as the global `seed`. The `dataset_seed` exists in the form of `random_state` and does not affect the global seed.
- `--dataset_test_ratio`: Used to specify the ratio for splitting the sub-dataset into training and validation sets. The default value is `0.01`. If `--val_dataset` is set, this parameter becomes ineffective.
- `--train_dataset_sample`: The number of samples for the training dataset, default is `-1`, which means using the complete training dataset for training. This parameter is deprecated, please use `--dataset {dataset_name}#{dataset_sample}` instead.
- `--val_dataset_sample`: Used to sample the validation set, with a default value of `None`, which automatically selects a suitable number of data samples for validation. If you specify `-1`, the complete validation set is used for validation. This parameter is deprecated and the number of samples in the validation set is controlled by `--dataset_test_ratio` or `--val_dataset {dataset_name}#{dataset_sample}`.
- `--🔥system`: System used in dialogue template, default is `None`, i.e. use the model's default system. If set to '', no system is used.
- `--tools_prompt`: Select the corresponding tools system prompt for the tools field transformation. The options are ['react_en', 'react_zh', 'toolbench'], which correspond to the English version of ReAct format, Chinese version of ReAct format and the toolbench format, respectively. The default is the English version of the ReAct format. For more information, you can refer to the [Agent Deployment Best Practices](Agent-deployment-best-practices.md).
- `--🔥max_length`: Maximum token length, default is `2048`. Avoids OOM issues caused by individual overly long samples. When `--truncation_strategy delete` is specified, samples exceeding max_length will be deleted. When `--truncation_strategy truncation_left` is specified, the leftmost tokens will be truncated: `input_ids[-max_length:]`. If set to -1, no limit.
- `--truncation_strategy`: Default is `'delete'` which removes sentences exceeding max_length from dataset. `'truncation_left'` will truncate excess text from the left, which may truncate special tokens and affect performance, not recommended.
- `--check_dataset_strategy`: Default is `'none'`, i.e. no checking. If training an LLM model, `'warning'` is recommended as data check strategy. If your training target is sentence classification etc., setting to `'none'` is recommended.

- `--custom_train_dataset_path`: Default value is `[]`. This parameter has been deprecated, please use `--dataset {dataset_path}`.
- `--custom_val_dataset_path`: Default value is `[]`. This parameter is deprecated. Please use `--val_dataset {dataset_path}` instead.
- `--self_cognition_sample`: The number of samples for the self-cognition dataset. Default is `0`. If you set this value to >0, you need to specify `--model_name` and `--model_author` at the same time. This parameter has been deprecated, please use `--dataset self-cognition#{self_cognition_sample}` instead.
- `--🔥model_name`: Default value is `[None, None]`. If self-cognition dataset sampling is enabled (i.e., specifying `--dataset self-cognition` or self_cognition_sample>0), you need to provide two values, representing the Chinese and English names of the model, respectively. For example: `--model_name 小黄 'Xiao Huang'`. If you want to learn more, you can refer to the [Self-Cognition Fine-tuning Best Practices](../LLM/Self-cognition-best-practice.md).
- `--🔥model_author`: Default is `[None, None]`. If self-cognition dataset sampling is enabled, you need to pass two values, representing the author's Chinese and English names respectively. E.g. `--model_author 魔搭 ModelScope`.
- `--quant_method`: Quantization method, default is None. You can choose from 'bnb', 'hqq', 'eetq'.
- `--quantization_bit`: Specifies whether to quantize and number of quantization bits, default is `0`, i.e. no quantization. To use 4bit qlora, set `--sft_type lora --quantization_bit 4`.Hqq support 1,2,3,4,8bit, bnb support 4,8bit
- `--hqq_axis`: Hqq argument. Axis along which grouping is performed. Supported values are 0 or 1. default is `0`
- `--hqq_dynamic_config_path`: Parameters for dynamic configuration. The key is the name tag of the layer and the value is a quantization config. If set, each layer specified by its id will use its dedicated quantization configuration.[ref](https://github.com/mobiusml/hqq?tab=readme-ov-file#custom-quantization-configurations-%EF%B8%8F)
- `--bnb_4bit_comp_dtype`: When doing 4bit quantization, we need to dequantize during model forward and backward passes. This specifies the torch_dtype after dequantization. Default is `'AUTO'`, i.e. consistent with `dtype`. Options: 'fp16', 'bf16', 'fp32'. Has no effect when quantization_bit is 0.
- `--bnb_4bit_quant_type`: Quantization method for 4bit quantization, default is `'nf4'`. Options: 'nf4', 'fp4'. Has no effect when quantization_bit is 0.
- `--bnb_4bit_use_double_quant`: Whether to enable double quantization for 4bit quantization, default is `True`. Has no effect when quantization_bit is 0.
- `--bnb_4bit_quant_storage`: Default vlaue `None`.This sets the storage type to pack the quanitzed 4-bit prarams. Has no effect when quantization_bit is 0.
- `--🔥target_modules`: Specify lora modules, default is `['DEFAULT']`. If target_modules is passed `'DEFAULT'` or `'AUTO'`, look up `target_modules` in `MODEL_MAPPING` based on `model_type` (The LLM is defaulted to qkv, while the MLLM defaults to all lines in the llm and projector.). If passed `'ALL'`, all Linear layers (excluding head) will be specified as lora modules. If passed `'EMBEDDING'`, Embedding layer will be specified as lora module. If memory allows, setting to 'ALL' is recommended. You can also set `['ALL', 'EMBEDDING']` to specify all Linear and embedding layers as lora modules. This parameter only takes effect when `sft_type` is 'lora'. This argument works when sft_type in lora/vera/boft/ia3/adalora/fourierft.
- `--target_regex`: The lora target regex in `Optional[str]`. default is `None`. If this argument is specified, the `target_modules` will have no effect. This argument works when sft_type in lora/vera/boft/ia3/adalora/fourierft.
- `--🔥lora_rank`: Default is `8`. Only takes effect when `sft_type` is 'lora'.
- `--🔥lora_alpha`: Default is `32`. Only takes effect when `sft_type` is 'lora'.
- `--lora_dropout`: Default is `0.05`, only takes effect when `sft_type` is 'lora'.
- `--init_lora_weights`: Method to initialize LoRA weights, can be specified as `true`, `false`, `gaussian`, `pissa`, or `pissa_niter_[number of iters]`. Default value `true`.
- `--lora_bias_trainable`: Default is `'none'`, options: 'none', 'all'. Set to `'all'` to make all biases trainable.
- `--modules_to_save`: Default is `[]`. If you want to train embedding, lm_head, or layer_norm, you can set this parameter, e.g. `--modules_to_save EMBEDDING LN lm_head`. If passed `'EMBEDDING'`, Embedding layer will be added to `modules_to_save`. If passed `'LN'`, `RMSNorm` and `LayerNorm` will be added to `modules_to_save`. This argument works when sft_type in lora/vera/boft/ia3/adalora/fourierft.
- `--lora_dtype`: Default is `'AUTO'`, specifies dtype for lora modules. If `AUTO`, follow dtype of original module. Options: 'fp16', 'bf16', 'fp32', 'AUTO'.
- `--use_dora`: Default is `False`, whether to use `DoRA`.
- `--use_rslora`: Default is `False`, whether to use `RS-LoRA`.
- `--neftune_noise_alpha`: The noise coefficient added by `NEFTune` can improve performance of instruction fine-tuning, default is `None`. Usually can be set to 5, 10, 15. See [related paper](https://arxiv.org/abs/2310.05914).
- `--neftune_backend`: The backend of `NEFTune`, supported values are `transformers`, `swift`, default is `transformers`.
- `--🔥gradient_checkpointing`: Whether to enable gradient checkpointing, default is `True`. This can be used to save memory, although it slightly reduces training speed. Has significant effect when max_length and batch_size are large.
- `--🔥deepspeed`: Used to specify the path to the deepspeed configuration file or directly pass JSON formatted configuration information. By default, it is set to `None`, which means deepspeed is not enabled. Deepspeed can save GPU memory. We have written default [ZeRO-2 configuration file](https://github.com/modelscope/swift/blob/main/swift/llm/ds_config/zero2_offload.json), [ZeRO-3 configuration file](https://github.com/modelscope/swift/blob/main/swift/llm/ds_config/zero3.json), [ZeRO-2 Offload configuration file](https://github.com/modelscope/swift/blob/main/swift/llm/ds_config/zero2_offload.json ), and [ZeRO-3 Offload configuration file](https://github.com/modelscope/swift/blob/main/swift/llm/ds_config/zero3_offload.json). You only need to specify 'default-zero2', 'default-zero3', 'zero2-offload', 'zero3-offload'.
- `--batch_size`: Batch_size during training, default is `1`. Increasing batch_size can improve GPU utilization, but won't necessarily improve training speed, because within a batch, shorter sentences need to be padded to the length of the longest sentence in the batch, introducing invalid computations.
- `--eval_batch_size`: Batch_size during evaluation, default is `None`, i.e. set to 1 when `predict_with_generate` is True, set to `batch_size` when False.
- `--🔥num_train_epochs`: Number of epochs to train, default is `1`. If `max_steps >= 0`, this overrides `num_train_epochs`. Usually set to 3 ~ 5.
- `--max_steps`: Max_steps for training, default is `-1`. If `max_steps >= 0`, this overrides `num_train_epochs`.
- `--optim`: Default is `'adamw_torch'`.
- `--adam_beta1`: Default is `0.9`.
- `--adam_beta2`: Default is `0.95`.
- `--adam_epsilon`: Default is `1e-8`.
- `--🔥learning_rate`: Default is `None`, i.e. set to 1e-4 if `sft_type` is lora, set to 1e-5 if `sft_type` is full.
- `--weight_decay`: Default is `0.01`.
- `--🔥gradient_accumulation_steps`: Gradient accumulation, default is `None`, set to `math.ceil(16 / self.batch_size / world_size)`. `total_batch_size =  batch_size * gradient_accumulation_steps * world_size`.
- `--max_grad_norm`: Gradient clipping, default is `1`.
- `--predict_with_generate`: Whether to use generation for evaluation, default is `False`. If set to False, evaluate using `loss`. If set to True, evaluate using `ROUGE-L` and other metrics. Generative evaluation takes a long time, choose carefully.
- `--lr_scheduler_type`: Default is `'cosine'`, options: 'linear', 'cosine', 'constant', etc.
- `--warmup_ratio`: Proportion of warmup in total training steps, default is `0.05`.
- `--warmup_steps`: The number of warmup steps, default is `0`. If warmup_steps > 0 is set, it overrides warmup_ratio.
- `--🔥eval_steps`: Evaluate every this many steps, default is `50`.
- `--save_steps`: Save every this many steps, default is `None`, i.e. set to `eval_steps`.
- `--🔥save_only_model`: Whether to save only model parameters, without saving intermediate states needed for checkpoint resuming, default is `False`.
- `--save_total_limit`: Number of checkpoints to save, default is `2`, i.e. save best and last checkpoint. If set to -1, save all checkpoints.
- `--logging_steps`: Print training information (e.g. loss, learning_rate, etc.) every this many steps, default is `5`.
- `--dataloader_num_workers`: Default value is `None`. If running on a Windows machine, set it to `0`; otherwise, set it to `1`.
- `--push_to_hub`: Whether to sync push trained checkpoint to ModelScope Hub, default is `False`.
- `--hub_model_id`: Model_id to push to on ModelScope Hub, default is `None`, i.e. set to `f'{model_type}-{sft_type}'`. You can set this to model_id or repo_name. We will infer user_name based on hub_token. If the remote repository to push to does not exist, a new repository will be created, otherwise the previous repository will be reused. This parameter only takes effect when `push_to_hub` is set to True.
- `--hub_token`: SDK token needed for pushing. Can be obtained from [https://modelscope.cn/my/myaccesstoken](https://modelscope.cn/my/myaccesstoken), default is `None`, i.e. obtained from environment variable `MODELSCOPE_API_TOKEN`. This parameter only takes effect when `push_to_hub` is set to True.
- `--hub_private_repo`: Whether to set the permission of the pushed model repository on ModelScope Hub to private, default is `False`. This parameter only takes effect when `push__to_hub` is set to True.
- `--hub_strategy`: Push strategy, default is `'every_save'`. Options include: 'end', 'every_save', 'checkpoint', 'all_checkpoints'. This parameter shares the same meaning from transformers, and only takes effect when `push_to_hub` is set to True.
- `--test_oom_error`: Used to detect whether training will cause OOM, default is `False`. If set to True, will sort the training set in descending order by max_length, easy for OOM testing. This parameter is generally used for testing, use carefully.
- `--disable_tqdm`: Whether to disable tqdm, useful when launching script with `nohup`. Default is `False`, i.e. enable tqdm.
- `--🔥lazy_tokenize`: If set to False, preprocess all text before `trainer.train()`. If set to True, delay encoding text, reducing preprocessing wait and memory usage, useful when processing large datasets. Default is `None`, i.e. we intelligently choose based on template type, usually set to False for LLM models, set to True for multimodal models (to avoid excessive memory usage from loading images and audio).
- `--🔥preprocess_num_proc`: Use multiprocessing when preprocessing dataset (tokenizing text). Default is `1`. Same as `lazy_tokenize` command line argument, used to solve slow preprocessing issue. But this strategy cannot reduce memory usage, so if dataset is huge, `lazy_tokenize` is recommended. Recommended values: 4, 8.
- `--🔥use_flash_attn`: Whether to use flash attn, default is `None`. Installation steps for flash_attn can be found at [https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention). Models supporting flash_attn can be found in [LLM Supported Models](Supported-models-datasets.md).
- `--ignore_args_error`: Whether to ignore Error thrown by command line parameter errors, default is `False`. Set to True if need to copy code to notebook to run.
- `--🔥check_model_is_latest`: Check if model is latest, default is `True`. Set this to `False` if you need to train offline.
- `--logging_dir`: Default is `None`. I.e. set to `f'{self.output_dir}/runs'`, representing path to store tensorboard files.
- `--report_to`: Default is `['tensorboard']`. You can set `--report_to all` to report to all installed integrations.
- `--acc_strategy`: Default is `'token'`, options include: 'token', 'sentence'.
- `--save_on_each_node`: Takes effect during multi-machine training, default is `False`.
- `--save_strategy`: Strategy for saving checkpoint, default is `'steps'`, options include: 'steps', 'epoch', no'.
- `--evaluation_strategy`: Strategy for evaluation, default is `'steps'`, options include: 'steps', 'epoch', no'.
- `--save_safetensors`: Default is `True`.
- `--include_num_input_tokens_seen`: Default is `False`. Tracks the number of input tokens seen throughout training.
- `--max_new_tokens`: Default is `2048`. This parameter only takes effect when `predict_with_generate` is set to True.
- `--do_sample`: Reference document: [https://huggingface.co/docs/transformers/main_classes/text_generation](https://huggingface.co/docs/transformers/main_classes/text_generation). Default is `None`, inheriting the model's generation_config. This parameter only takes effect when `predict_with_generate` is set to True.
- `--temperature`: Default is `None`, inheriting the model's generation_config. This parameter only takes effect when `do_sample` is set to True. This parameter will be used as default value in deployment parameters.
- `--top_k`: Default is `None`, inheriting the model's generation_config. This parameter only takes effect when `do_sample` is set to True. This parameter will be used as default value in deployment parameters.
- `--top_p`: Default is `None`, inheriting the model's generation_config. This parameter only takes effect when `do_sample` is set to True. This parameter will be used as default value in deployment parameters.
- `--repetition_penalty`: Default is `None`, inheriting the model's generation_config. This parameter will be used as default value in deployment parameters.
- `--num_beams`: Default is `1`. This parameter only takes effect when `predict_with_generate` is set to True.
- `--gpu_memory_fraction`: Default is `None`. This parameter aims to run training under a specified maximum available GPU memory percentage, used for extreme testing.
- `--train_dataset_mix_ratio`: Default is `0.`. This parameter defines how to mix datasets for training. When this parameter is specified, it will mix the training dataset with a multiple of `train_dataset_mix_ratio` of the general knowledge dataset specified by `train_dataset_mix_ds`. This parameter has been deprecated, please use `--dataset {dataset_name}#{dataset_sample}` to mix datasets.
- `--train_dataset_mix_ds`: Default is `['ms-bench']`. Used for preventing knowledge forgetting, this is the general knowledge dataset. This parameter has been deprecated, please use `--dataset {dataset_name}#{dataset_sample}` to mix datasets.
- `--use_loss_scale`: Default is `False`. When taking effect, strengthens loss weight of some Agent fields (Action/Action Input part) to enhance CoT, has no effect in regular SFT scenarios.
- `--loss_scale_config_path`: option specifies a custom loss_scale configuration, applicable when use_loss_scale is enabled, such as in Agent training to amplify the loss weights for Action and other crucial ReAct fields.
  - In the configuration file, you can set the loss_scale using a dictionary format. Each key represents a specific field name, and its associated value specifies the loss scaling factor for that field and its subsequent content. For instance, setting `"Observation:": [2, 0]` means that when the response contains `xxxx Observation:error`, the loss for the `Observation:` field will be doubled, while the loss for the `error` portion will not be counted. Besides literal matching, the configuration also supports regular expression rules for more flexible matching; for example, the pattern `'<.*?>':[2.0]` doubles the loss for any content enclosed in angle brackets. The loss scaling factors for field matching and regex matching are respectively indicated by lists of length 2 and 1.
  - There is also support for setting loss_scale for the entire response based on matching queries, which is extremely useful in dealing with fixed multi-turn dialogue queries described in the [Agent-Flan paper](https://arxiv.org/abs/2403.12881) paper. If the query includes any of the predefined keys, the corresponding response will use the associated loss_scale value. Refer to swift/llm/agent/agentflan.json for an example.
  - By default, we have preset loss scaling values for fields such as Action:, Action Input:, Thought:, Final Answer:, and Observation:. We also provide default configurations for [alpha-umi](https://arxiv.org/pdf/2401.07324) and [Agent-FLAN](https://arxiv.org/abs/2403.12881), which you can use by setting to alpha-umi and agent-flan respectively. The default configuration files are located under swift/llm/agent.
  - The application priority of matching rules is as follows, from highest to lowest: query fields > specific response fields > regular expression matching rules.
- `--custom_register_path`: Default is `None`. Pass in a `.py` file used to register templates, models, and datasets.
- `--custom_dataset_info`: Default is `None`. Pass in the path to an external `dataset_info.json`, a JSON string, or a dictionary. Used to register custom datasets. The format example: https://github.com/modelscope/swift/blob/main/swift/llm/data/dataset_info.json
- `--device_map_config`: Manually configure the model's device_map, default is `None`. You can pass a local path (.json), a JSON string, or a dict.
- `--device_max_memory`: The max memory of each device can use for `device_map`, `List`, default is `[]`, The number of values must equal to the device count. Like `10GB 10GB`.

### Long Context

- `--rope_scaling`: Default `None`, Support `linear` and `dynamic` to scale positional embeddings. Use when `max_length` exceeds `max_position_embeddings`.
- `--rescale_image`: Whether to rescale input images, the value should be the pixel value, for example 480000(width * height), every image larger than this value will be resized to this value by its original ratio. Note: not every model can get advantages from this parameter.

### FSDP Parameters

- `--fsdp`: Default value `''`, the FSDP type, please check [this documentation](https://huggingface.co/docs/transformers/v4.39.3/en/main_classes/trainer#transformers.TrainingArguments.fsdp) for details.

- `--fsdp_config`: Default value `None`, the FSDP config file path.

### Sequence Parallel Parameters

- `--sequence_parallel_size`: Default value `1`, a positive value can be used to split a sequence to multiple GPU to reduce memory usage. The value should divide the GPU count.

### FourierFt Parameters

FourierFt uses `target_modules`, `target_regex`, `modules_to_save`.

- `--fourier_n_frequency`: Num of learnable frequencies for the Discrete Fourier Transform, `int` type, like `r` in LoRA. Default value `2000`.
- `--fourier_scaling`: The scaling value for the delta W matrix, `float` type, like `lora_alpha` in LoRA. Default value `300.0`.

### BOFT Parameters

BOFT uses `target_modules`, `target_regex`, `modules_to_save`.

- `--boft_block_size`: BOFT block size, default value is 4.
- `--boft_block_num`: Number of BOFT blocks, cannot be used simultaneously with `boft_block_size`.
- `--boft_dropout`: Dropout value for BOFT, default is 0.0.

### Vera Parameters

Vera uses `target_modules`, `target_regex`, `modules_to_save`.

- `--vera_rank`: Size of Vera Attention, default value is 256.
- `--vera_projection_prng_key`: Whether to store the Vera projection matrix, default is True.
- `--vera_dropout`: Dropout value for Vera, default is 0.0.
- `--vera_d_initial`: Initial value for Vera's d matrix, default is 0.1.

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
- `--galore_quantization`: Whether to use q-galore. Default value `False`.
- `--galore_proj_quant`: Whether to quantize the SVD decomposition matrix, default `False`.
- `--galore_proj_bits`: Number of bits for SVD quantization.
- `--galore_proj_group_size`: Number of groups for SVD quantization.
- `--galore_cos_threshold`: Cosine similarity threshold for updating the projection matrix. Default value 0.4.
- `--galore_gamma_proj`: When the projection matrix gradually becomes similar, this parameter is the coefficient for extending the update interval each time, default value 2.
- `--galore_queue_size`: Queue length for calculating projection matrix similarity, default value 5.

### LISA Fine-tuning Parameters

Note: LISA only supports full training, which is `--sft_type full`.

- `--lisa_activated_layers`: Default value`0`, which means use without `LISA`, suggested value is `2` or `8`.
- `--lisa_step_interval`: Default value `20`, how many iters to switch the layers to back-propagate.

### UNSLOTH Fine-tuning Parameters

unsloth has no new parameters，you can use the existing parameters to use unsloth:

```
--tuner_backend unsloth
--sft_type full/lora
--quantization_type 4
```

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

Vera uses `target_modules`, `target_regex`, `modules_to_save`.

The following parameters take effect when `sft_type` is set to `ia3`.

- `--ia3_feedforward_modules`: Specify the Linear name of IA3's MLP, this name must be in `ia3_target_modules`.

### ReFT Fine-tuning Parameters

The following parameters take effect when the `sft_type` is set to `reft`.

> 1. ReFT tuner cannot be merged
> 2. ReFT and gradient_checkpointing are not compatible
> 3. If error happens when using ReFT and DeepSpeed, please uninstall DeepSpeed

- `--reft_layers`: Specifies which layers ReFT is applied to; defaults to `None`, meaning all layers. You can input a list of layer numbers, for example: `--reft_layers 1 2 3 4`.
- `--reft_rank`: The rank of the ReFT matrix; defaults to `4`.
- `--reft_intervention_type`: The type of ReFT intervention, supporting 'NoreftIntervention', 'LoreftIntervention', 'ConsreftIntervention', 'LobireftIntervention', 'DireftIntervention', and 'NodireftIntervention'; defaults to `LoreftIntervention`.
- `--reft_args`: Other supporting parameters in the ReFT intervention, provided in JSON string format.

### Liger Parameters

- `--use_liger`: Use liger-kernel to train.

## PT Parameters

PT parameters inherit from the SFT parameters with some modifications to the default values:

- `--sft_type`: Default value is `'full'`.
- `--target_modules`: Default value is `'ALL'`.
- `--lazy_tokenize`: Default value is `True`.
- `--eval_steps`: Default value is `500`.

## RLHF Parameters
RLHF parameters are an extension of the sft parameters, with the addition of the following options:
- `--🔥rlhf_type`: Choose the alignment algorithm, with options such as 'dpo', 'orpo', 'simpo', 'kto', 'cpo', default is 'dpo'. For training scripts with  different algorithms, please refer to [document](../LLM/Human-Preference-Alignment-Training-Documentation.md)
- `--ref_model_type`: Select reference model, same as the model_type parameter, default is None, consistent with the training model. For `cpo`, `simpo`, and `orpo` algorithms, this selection is not required. Typically, no setup is needed.
- `--ref_model_id_or_path`: Local cache path for the reference model, default is `None`.
- `--ref_model_revision`: Model revision for the reference model, default is `None`.
- `--beta`: KL regularization term coefficient, default is `None`, meaning that for the simpo algorithm, the default is `2`., and for other algorithms, it is `0.1`. For detail please check[document](../LLM/Human-Preference-Alignment-Training-Documentation.md)
- `--label_smoothing`: Whether to use DPO smoothing, the default value is `0`, normally set between 0 and 0.5.
- `--loss_type`: Type of loss, default is `None`. If it's dpo or cpo, it is `'sigmoid'`, and if it's simpo, it is `'simpo'`.

### DPO Parameters
- `--🔥rpo_alpha`: Controls the weight of sft_loss added in DPO, default is `1.` The final loss is `KL_loss + rpo_alpha * sft_loss`.

### CPO/SimPO Parameters
- `cpo_alpha`: Coefficient for nll loss in CPO/SimPO loss, default is `1.`.
- `--simpo_gamma`: The reward margin term in the SimPO algorithm, the paper recommends setting it to 0.5-1.5, the default is `1.`.

### KTO Parameters
- `--desirable_weight`: The loss weight for desirable responses $\lambda_D$ in the KTO algorithm, default is `1.`.
- `--undesirable_weight`: The loss weight for undesirable responses $\lambda_U$ in the KTO paper, default is `1.`. Let $n_d$ and $n_u$ represent the number of desirable and undesirable examples in the dataset, respectively. The paper recommends controlling $\frac{\lambda_D n_D}{\lambda_Un_U} \in [1,\frac{4}{3}]$.

### PPO Parameters
- `--reward_model_id_or_path` : The local cache path for the reward model, which must include the weights of value_head (`value_head.safetensors` or `value_head.bin`).
- `--reward_model_type`: Select reward model, same as the model_type parameter, default is None, consistent with the training model.
- `--reward_model_revision`: Model revision for the reference model, default is `None`.
- `--local_rollout_forward_batch_size`: Per rank no grad forward pass in the rollout phase, default is 64
- `--whiten_rewards`: Whether to whiten the rewards, default is False
- `--kl_coef`: KL coefficient, default is 0.05
- `--cliprange`: Clip range in the PPO policy loss funtion, default is 0.2
- `--vf_coef`: Coefficient for the value loss function, default is 0.1
- `--cliprange_value`: Clip range in the PPO value loss function, default is 0.2
- `--gamma`: Discount factor for cumulative rewards, default is 1.0
- `--lam`: Lambda value for [GAE](https://arxiv.org/abs/1506.02438), default is 0.95

## infer merge-lora Parameters

- `--🔥model_type`: Default is `None`, see `sft command line arguments` for parameter details.
- `--🔥model_id_or_path`: Default is `None`, see `sft command line arguments` for parameter details. Recommended to use model_type to specify.
- `--model_revision`: Default is `None`. See `sft command line arguments` for parameter details. If `model_id_or_path` is None or a local model directory, this parameter has no effect.
- `--🔥sft_type`: Default is `'lora'`, see `sft command line arguments` for parameter details.
- `--🔥template_type`: Default is `'AUTO'`, see `sft command line arguments` for parameter details.
- `--🔥infer_backend`: Options are 'AUTO', 'vllm', 'pt'. Default uses 'AUTO', for intelligent selection, i.e. if `ckpt_dir` is not passed or using full fine-tuning, and vllm is installed and model supports vllm, then use vllm engine, otherwise use native torch for inference. vllm environment setup can be found in [VLLM Inference Acceleration and Deployment](../LLM/VLLM-inference-acceleration-and-deployment.md), vllm supported models can be found in [Supported Models](Supported-models-datasets.md).
- `--🔥ckpt_dir`: Required, value is the checkpoint path saved in SFT stage, e.g. `'/path/to/your/vx-xxx/checkpoint-xxx'`.
- `--load_args_from_ckpt_dir`: Whether to read model configuration info from `sft_args.json` file in `ckpt_dir`. Default is `True`.
- `--🔥load_dataset_config`: This parameter only takes effect when `--load_args_from_ckpt_dir true`. I.e. whether to read dataset related configuration from `sft_args.json` file in `ckpt_dir`. Default is `False`.
- `--eval_human`: Whether to evaluate using validation set portion of dataset or manual evaluation. Default is `None`, for intelligent selection, if no datasets (including custom datasets) are passed, manual evaluation will be used. If datasets are passed, dataset evaluation will be used.
- `--device_map_config`: Default is `None`, see `sft command line arguments` for parameter details.
- `--device_max_memory`: Default is `[]`, see `sft command line arguments` for parameter details.
- `--seed`: Default is `42`, see `sft command line arguments` for parameter details.
- `--dtype`: Default is `'AUTO`, see `sft command line arguments` for parameter details.
- `--model_kwargs`: Default is `None`, see `sft command line arguments` for parameter details.
- `--🔥dataset`: Default is `[]`, see `sft command line arguments` for parameter details.
- `--🔥val_dataset`: Default is `[]`, see `sft command line arguments` for parameter details.
- `--dataset_seed`: Default is `None`, see `sft command line arguments` for parameter details.
`--dataset_test_ratio`: Default value is `0.01`. For specific parameter details, refer to the `sft command line arguments`.
- `--🔥show_dataset_sample`: Represents number of validation set samples to evaluate and display, default is `-1`.
- `--system`: Default is `None`. See `sft command line arguments` for parameter details.
- `--tools_prompt`: Default is `react_en`. See `sft command line arguments` for parameter details.
- `--max_length`: Default is `-1`. See `sft command line arguments` for parameter details.
- `--truncation_strategy`: Default is `'delete'`. See `sft command line arguments` for parameter details.
- `--check_dataset_strategy`: Default is `'none'`, see `sft command line arguments` for parameter details.
- `--custom_train_dataset_path`: Default value is `[]`. This parameter has been deprecated, please use `--dataset {dataset_path}`.
- `--custom_val_dataset_path`: Default value is `[]`. This parameter is deprecated. Please use `--val_dataset {dataset_path}` instead.
- `--quantization_bit`: Default is 0. See `sft command line arguments` for parameter details.
- `--quant_method`: Quantization method, default is None. You can choose from 'bnb', 'hqq', 'eetq'.
- `--hqq_axis`: Hqq argument. Axis along which grouping is performed. Supported values are 0 or 1. default is `0`
- `--hqq_dynamic_config_path`: Parameters for dynamic configuration. The key is the name tag of the layer and the value is a quantization config. If set, each layer specified by its id will use its dedicated quantization configuration.[ref](https://github.com/mobiusml/hqq?tab=readme-ov-file#custom-quantization-configurations-%EF%B8%8F)
- `--bnb_4bit_comp_dtype`: Default is `'AUTO'`.  See `sft command line arguments` for parameter details. If `quantization_bit` is set to 0, this parameter has no effect.
- `--bnb_4bit_quant_type`: Default is `'nf4'`.  See `sft command line arguments` for parameter details. If `quantization_bit` is set to 0, this parameter has no effect.
- `--bnb_4bit_use_double_quant`: Default is `True`.  See `sft command line arguments` for parameter details. If `quantization_bit` is set to 0, this parameter has no effect.
- `--bnb_4bit_quant_storage`: Default value `None`.See `sft command line arguments` for parameter details. If `quantization_bit` is set to 0, this parameter has no effect.
- `--🔥max_new_tokens`: Maximum number of new tokens to generate, default is `2048`. If using deployment, please control the maximum number of generated tokens by passing `max_tokens` in the client.
- `--🔥do_sample`: Reference document: [https://huggingface.co/docs/transformers/main_classes/text_generation](https://huggingface.co/docs/transformers/main_classes/text_generation). Default is `None`, inheriting the model's generation_config.
- `--temperature`: Default is `None`, inheriting the model's generation_config. This parameter only takes effect when `do_sample` is set to True. This parameter will be used as default value in deployment parameters.
- `--top_k`: Default is `None`, inheriting the model's generation_config. This parameter only takes effect when `do_sample` is set to True. This parameter will be used as default value in deployment parameters.
- `--top_p`: Default is `None`, inheriting the model's generation_config. This parameter only takes effect when `do_sample` is set to True. This parameter will be used as default value in deployment parameters.
- `--repetition_penalty`: Default is `None`, inheriting the model's generation_config. This parameter will be used as default value in deployment parameters.
- `--num_beams`: Default is `1`.
- `--use_flash_attn`: Default is `None`, i.e. 'auto'. See `sft command line arguments` for parameter details.
- `--ignore_args_error`: Default is `False`, see `sft command line arguments` for parameter details.
- `--stream`: Whether to use streaming output, default is `True`. This parameter only takes effect when using dataset evaluation and verbose is True.
- `--🔥merge_lora`: Whether to merge lora weights into base model and save full weights, default is `False`. Weights will be saved in the same level directory as `ckpt_dir`, e.g. `'/path/to/your/vx-xxx/checkpoint-xxx-merged'` directory.
- `--merge_device_map`: device_map used when merge-lora, default is `None`, to reduce memory usage, use `auto` only during merge-lora process, otherwise default is `cpu`.
- `--save_safetensors`: Whether to save as `safetensors` file or `bin` file. Default is `True`.
- `--overwrite_generation_config`: Whether to save the generation_config used for evaluation as a `generation_config.json` file, default is `False`.
- `--🔥verbose`: If set to False, use tqdm style inference. If set to True, output inference query, response, label. Default is `None`, for auto selection, i.e. when `len(val_dataset) >= 100`, set to False, otherwise set to True. This parameter only takes effect when using dataset evaluation.
- `--lora_modules`: Default`[]`, the input format is `'{lora_name}={lora_path}'`, e.g. `--lora_modules lora_name1=lora_path1 lora_name2=lora_path2`. `ckpt_dir` will be added with `f'default-lora={args.ckpt_dir}'` by default.
- `--custom_register_path`: Default is `None`. Pass in a `.py` file used to register templates, models, and datasets.
- `--custom_dataset_info`: Default is `None`. Pass in the path to an external `dataset_info.json`, a JSON string, or a dictionary. Used for expanding datasets.
- `--rope_scaling`: Default `None`, Support `linear` and `dynamic` to scale positional embeddings. Use when `max_length` exceeds `max_position_embeddings`. Specify `--max_length` when using this parameter.


### vLLM Parameters
Reference document: [https://docs.vllm.ai/en/latest/models/engine_args.html](https://docs.vllm.ai/en/latest/models/engine_args.html)

- `--🔥gpu_memory_utilization`: Parameter for initializing vllm engine `EngineArgs`, default is `0.9`. This parameter only takes effect when using vllm. vLLM inference acceleration and deployment can be found in [vLLM Inference Acceleration and Deployment](../LLM/VLLM-inference-acceleration-and-deployment.md).
- `--🔥tensor_parallel_size`: Parameter for initializing vllm engine `EngineArgs`, default is `1`. This parameter only takes effect when using vllm.
- `--max_num_seqs`: The parameter for initializing the `EngineArgs` of the vllm engine, with a default value of `256`. This parameter is only effective when using vllm.
- `--🔥max_model_len`: Override model's max_model__len, default is `None`. This parameter only takes effect when using vllm.
- `--disable_custom_all_reduce`: Whether to disable the custom all-reduce kernel and fallback to NCCL. The default is `True`, which is different from the default value of vLLM.
- `--enforce_eager`: vllm uses the PyTorch eager mode or builds the CUDA graph. Default is `False`. Setting to True can save memory, but it may affect efficiency.
- `--limit_mm_per_prompt`: Control vllm to use multiple images. Default is `None`. For example, pass `--limit_mm_per_prompt '{"image": 10, "video": 5}'`.
- `--vllm_enable_lora`: Default `False`. Whether to support vllm with lora.
- `--vllm_max_lora_rank`: Default `16`.  Lora rank in vLLM.
- `--lora_modules`: Introduced.


### lmdeploy Parameters
Reference document: [https://lmdeploy.readthedocs.io/en/latest/api/pipeline.html#turbomindengineconfig](https://lmdeploy.readthedocs.io/en/latest/api/pipeline.html#turbomindengineconfig)

- `--🔥tp`: Tensor parallelism, a parameter for initializing the lmdeploy engine, default value is `1`.
- `--cache_max_entry_count`: Parameter to initialize the lmdeploy engine, default value is `0.8`.
- `--quant_policy`: Quantization of Key-Value Cache, parameters for initializing the lmdeploy engine, default value is `0`, you can set it to 4 or 8.
- `--vision_batch_size`: Parameter to initialize the lmdeploy engine, default value is `1`. This parameter is effective only when using multimodal models.

## export Parameters

export parameters inherit from infer parameters, with the following added parameters:
- `--to_peft_format`: Default is `False`. Convert the swift format of LoRA (`--tuner_backend swift`) to peft format.
- `--🔥merge_lora`: Default is `False`. This parameter is already defined in InferArguments, not a new parameter. Whether to merge lora weights into base model and save full weights. Weights will be saved in the same level directory as `ckpt_dir`, e.g. `'/path/to/your/vx-xxx/checkpoint-xxx-merged'` directory.
- `--🔥quant_bits`: Number of bits for quantization. Default is `0`, i.e. no quantization. If you set `--quant_method awq`, you can set this to `4` for 4bits quantization. If you set `--quant_method gptq`, you can set this to `2`,`3`,`4`,`8` for corresponding bits quantization. If quantizing original model, weights will be saved in `f'{args.model_type}-{args.quant_method}-int{args.quant_bits}'` directory. If quantizing fine-tuned model, weights will be saved in the same level directory as `ckpt_dir`, e.g. `f'/path/to/your/vx-xxx/checkpoint-xxx-{args.quant_method}-int{args.quant_bits}'` directory.
- `--🔥quant_method`: Quantization method, default is `'awq'`. Options are 'awq', 'gptq', 'bnb'.
- `--🔥dataset`: This parameter is already defined in InferArguments, for export it means quantization dataset. Default is `[]`. More details: including how to customize quantization dataset, can be found in [LLM Quantization Documentation](LLM-quantization-and-export.md).
- `--quant_n_samples`: Quantization parameter, default is `256`. When set to `--quant_method awq`, if OOM occurs during quantization, you can moderately reduce `--quant_n_samples` and `--quant_seqlen`. `--quant_method gptq` generally does not encounter quantization OOM.
- `--quant_seqlen`: Quantization parameter, default is `2048`.
- `--quant_batch_size`: Calibrating batch_size，Default `1`.
- `--quant_device_map`: Default is `None`, to save memory. You can specify 'cuda:0', 'auto', 'cpu', etc., representing the device to load model during quantization.
- `quant_output_dir`: Default is `None`, the default quant_output_dir will be printed in the command line.
- `--push_to_hub`: Default is `False`. Whether to push the final `ckpt_dir` to ModelScope Hub. If you specify `merge_lora`, full parameters will be pushed; if you also specify `quant_bits`, quantized model will be pushed.
- `--hub_model_id`: Default is `None`. Model_id to push to on ModelScope Hub. If `push_to_hub` is set to True, this parameter must be set.
- `--hub_token`: Default is `None`. See `sft command line arguments` for parameter details.
- `--hub_private_repo`: Default is `False`. See `sft command line arguments` for parameter details.
- `--commit_message`: Default is `'update files'`.
- `--to_ollama`: Export to ollama modelfile.
- `--ollama_output_dir`: ollama output dir. Default is `<modeltype>-ollama`.

## eval parameters

The eval parameters inherit from the infer parameters, and additionally include the following parameters: (Note: The generation_config parameter in infer will be invalid, controlled by [evalscope](https://github.com/modelscope/eval-scope).)

- `--🔥eval_dataset`: The official evaluation dataset, default is `None`, means all datasets. if `custom_eval_config` is specified, this arg will be ignored. [Check all supported eval datasets](LLM-eval.mdntroduction).
- `--eval_few_shot`: The few-shot number of sub-datasets for each evaluation set, with a default value of `None`, meaning to use the default configuration of the dataset. **This parameter is currently deprecated.**
- `--eval_limit`: The sampling quantity for each sub-dataset of the evaluation set, with a default value of `None` indicating full-scale evaluation. You can pass integer(number of samples from each eval dataset) or str(`[10:20]`, slice).
- `--name`: Used to differentiate the result storage path for evaluating the same configuration. Like: `{eval_output_dir}/{name}`, default will be `eval_outputs/defaults`, in which a timestamp named folder will hold each eval result.
- `--eval_url`: The standard model invocation interface for OpenAI, for example, `http://127.0.0.1:8000/v1`. This needs to be set when evaluating in a deployed manner, usually not needed. Default is `None`.
  ```shell
  swift eval --eval_url http://127.0.0.1:8000/v1 --eval_is_chat_model true --model_type gpt4 --eval_token xxx
  ```
- `--eval_token`: The token for the standard model invocation interface for OpenAI, with a default value of `'EMPTY'`, indicating no token.
- `--eval_is_chat_model`: If `eval_url` is not empty, this value needs to be passed to determine if it is a "chat" model. False represents a "base" model. Default is `None`.
- `--custom_eval_config`: Used for evaluating with custom datasets, and needs to be a locally existing file path. For details on file format, refer to [Custom Evaluation Set](LLM-eval.mdustom-Evaluation-Set). Default is `None`.
- `--eval_use_cache`: Whether to use already generated evaluation cache, so that previously evaluated results won't be rerun but only the evaluation results regenerated. Default is `False`.
- `--eval_output_dir`: Output path for evaluation results, default is `eval_outputs` in the current folder.
- `--eval_batch_size`: Input batch size for evaluation, default is 8.
- `--eval_nproc`: Concurrent number, a bigger value means a faster evaluation and more cost of GPU memory, default 16. This only takes effects when running multi-modal evaluations.
- `--deploy_timeout`: The timeout duration for waiting for model deployment before evaluation, default is `1800`, which means 30 minutes.

## app-ui Parameters

app-ui parameters inherit from infer parameters, with the following added parameters:

- `--host`: Default is `'127.0.0.1'`. Passed to the `demo.queue().launch(...)` function of gradio.
- `--port`: Default is `7860`. Passed to the `demo.queue().launch(...)` function of gradio.
- `--share`: Default is `False`. Passed to the `demo.queue().launch(...)` function of gradio.

## deploy Parameters

deploy parameters inherit from infer parameters, with the following added parameters:

- `--host`: Default is `'0.0.0.0'`.
- `--port`: Default is `8000`.
- `--api_key`: The default is `None`, meaning that the request will not be subjected to api_key verification.
- `--ssl_keyfile`: Default is `None`.
- `--ssl_certfile`: Default is `None`.
- `--verbose`: Whether to print the request content. Defaults to `True`.
- `--log_interval`: The interval for printing statistics, in seconds. Default is `10`. If set to `0`, it means statistics will not be printed.

## web-ui Parameters

- `--🔥host`: Default `'127.0.0.1'`. To make it accessible on the local network, you can set it to '0.0.0.0'.
- `--port`: Default `7860`.
- `--lang`: Default `'zh'`.
- `--share`: Default `False`.
