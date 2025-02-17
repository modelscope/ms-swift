# ReleaseNote 3.0

> If you encounter any issues while using version 3.x, please submit an issue to us. If something works in version 2.x but not in 3.x, please use version 2.x temporarily while we complete the fixes.

## New Features

1. Dataset module refactoring. The dataset loading speed has improved by 2-20 times, and encoding speed has improved by 2-4 times, with support for streaming mode.
    - Removed the dataset_name mechanism; now use dataset_id, dataset_dir, or dataset_path to specify the dataset.
    - Use `--dataset_num_proc` to support multi-process acceleration.
    - Use `--streaming` to support streaming loading of hub and local datasets.
    - Support `--packing` command for more stable training efficiency.
    - Use `--dataset <dataset_dir>` to support local loading of open-source datasets.

2. Model refactoring:
    - Removed model_type mechanism; use `--model <model_id>/<model_path>` for training and inference.
    - For new models, directly use `--model <model_id>/<model_path> --template xxx --model_type xxx` without needing to write a Python script for model registration.

3. Template module refactoring:
    - Use `--template_backend jinja` for Jinja mode inference.
    - Utilize messages format as the input parameter interface.

4. Supported plugin mechanism for customizing the training process. Current plugins include:
    - callback  to customize training callbacks,
    - loss  to customize the loss method,
    - loss_scale  to customize the weight of each token,
    - metric  to customize cross-validation metrics,
    - optimizer  to customize the optimizer and lr_scheduler used in training,
    - tools  to customize agent training system format,
    - tuner  to customize new tuners.

5. Training module refactoring:
    - Supports a single command to launch multi-machine training. See details [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/multi-node/deepspeed/README.md).
    - Supports PreTraining for all multi-modal LLMs.
    - In training, predict_with_generate now uses the infer module, supporting multi-modal LLM and multi-card setups.
    - Human alignment KTO algorithm supports multi-modal LLMs.

6. Inference and deployment module refactoring:
    - Supports batch inference under pt backend and multi-card inference.
    - Inference and deployment modules are unified using the OpenAI format interface.
    - Supports asynchronous inference interface.

7. Merged app-ui into web-ui, with app-ui supporting multi-modal inference.

8. Supports All-to-All models, such as Emu3-Gen and Janus for text-to-image or all-modal model training and deployment.

9. Enhanced the functionality of the examples, so that they can now fully reflect the capabilities of SWIFT and have stronger usability.

10. Use `--use_hf true/false` to switch between downloading/uploading datasets and models from HuggingFace and ModelScope communities.

11. Improved support for training and inference through code. The code structure is clearer, and extensive code comments have been added.

## BreakChanges

This document lists the BreakChanges between version 3.x and 2.x. Developers should note these differences when using them.

### Parameter Differences

- Version 3.0 only requires specifying --model. The model_type only needs to be specified additionally when the model is not supported by SWIFT.
- sft_type is renamed to train_type.
- model_id_or_path is renamed to model.
- template_type is renamed to template.
- quantization_bit is renamed to quant_bits.
- check_model_is_latest is renamed to check_model.
- batch_size is renamed to per_device_train_batch_size, following the transformers naming convention.
- eval_batch_size is renamed to per_device_eval_batch_size, following the transformers naming convention.
- tuner_backend has removed the swift option.
- use_flash_attn is renamed to attn_impl.
- bnb_4bit_comp_dtype is renamed to bnb_4bit_compute_dtype.
- Removed train_dataset_sample and val_dataset_sample.
- The term 'dtype' has been renamed to 'torch_dtype', and the option names have been changed from 'bf16' to the standard 'bfloat16', 'fp16' to 'float16', and 'fp32' to 'float32'.
- Removed eval_human option.
- The dataset option has removed the HF:: usage; use the new --use_hf to control downloading and uploading.
- Removed the do_sample option, and now use temperature for control.
- add_output_dir_suffix is renamed to add_version.
- Removed eval_token; API key is now supported.
- target_modules (lora_target_modules) ALL is changed to all-linear, retaining the same meaning.
- The parameters --ckpt_dir have been removed from infer/deploy/export, and control is now done using --model and --adapters.

The parameters marked as compatible in version 2.0 have been entirely removed.

### Functionality

1. For pre-training, please use the swift pt command. This command will default to using the generation template, while the swift sft command will default to the template preset by model_type.

2. Completely removed the examples directory from version 2.x, and added new examples categorized by functionality.

3. The dataset format is now fully compatible with messages format; query/response/history formats are no longer supported.

4. The storage directory for merge_lora can be specified using `--output_dir`, and merge_lora and quantization cannot be executed in the same command; at least two commands are required.

5. Use `swift app --model xxx` to launch the app-ui interface, which supports multimodal interface inference.

6. Removed dependencies for AIGC along with corresponding examples and training code.

## Pending Tasks

1. Custom dataset evaluation is not supported in version 3.0. Please use version 2.6.1.
2. Megatron pre-training capabilities are not supported in version 3.0. Please use version 2.6.1.
3. Documentation and README are temporarily incomplete and will be updated.
