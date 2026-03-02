# Frequently-asked-questions

Here are some common questions encountered during the use of SWIFT.

## Training

SWIFT supports training methods including pre-training, instruction-supervised fine-tuning (SFT), preference learning, GRPO, Embedding, Reranker, sequence classification tasks, etc. For details, please see the [homepage](https://github.com/modelscope/ms-swift/blob/main/README.md).

### Q1: What models does Swift support? How do I set a local model path?
For supported models, please refer to the [Supported Models and Datasets documentation](https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html). If the model has already been downloaded locally, simply set `--model <path_to_model>`. For training in an offline environment, set both `--model <local_path>` and `--check_model false`. If you encounter a git clone-related error, you need to clone the repository and then specify it using `local_repo_path`. For details, see [Command-line Parameters](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html). For models downloaded from ModelScope, you can configure the `MODELSCOPE_CACHE=your_path` environment variable to store the original models in a specified directory. If using the ModelScope SDK, use `cache_dir="local_address"`. You can also use the `modelscope download` command-line tool or `git` to download. For details, refer to the ModelScope documentation on [Model Download](https://modelscope.cn/docs/models/download). If you need to download models from Hugging Face, set the environment variable `USE_HF=1`.
Swift automatically matches the model_type, or you can manually specify it by referring to the [Supported Models and Datasets documentation](https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html).

### Q2: What datasets does Swift support? How do I use a custom dataset?
For supported datasets, see the [Supported Models and Datasets documentation](https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html). For the format and usage of custom datasets, see the [Custom Dataset documentation](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html). Datasets that conform to these formats will automatically use Swift's built-in data preprocessors. If your dataset does not match the format in the documentation, please convert it yourself or refer to how currently supported datasets are integrated. If your custom dataset contains extra fields, they will not be used by default. You can configure this using the [remove_unused_columns command-line parameter](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html#data-arguments).
You need to download the dataset locally and then specify its path. Please see the [Custom Dataset documentation](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html). git clone it locally, then specify the path using the `dataset_path` field in the dataset_info.json file.
For data shuffling, see the [dataset_shuffle command-line parameter](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html).
To force re-downloading a dataset, set the `--download_mode` command-line parameter. To perform error checking on the dataset, set the `strict` command-line parameter. If you need a dataset quality inspection tool, you can check out another library, [data-juicer](https://github.com/modelscope/data-juicer).
Due to the strict type checking in the underlying PyArrow of the datasets library, parts like objects in image grounding datasets and tools in agent datasets must be strings. Otherwise, PyArrow will report an error indicating inconsistent types between rows.
If you encounter the error `AttributeError:’TrainerState’ object has no attribute ’last_model_checkpoint’` during training, it's because the dataset is too small (the number of data points is less than one step). Increase the amount of data. A similar error can also occur when the split validation set is very small.
Below is an error caused by an empty assistant field:
```text
File "/mnt/workspace/swift/swift/1lm/dataset/preprocessor/core. py", line 69, in _check_messages raise
ValueError(f'assistant_message; {assistant_message}')
ValueError: assistant_message: {'role' :'assistant', 'content': ''}
```
```shell
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 MAX_PIXELS=1003520 swift sft --model Qwen/Qwen2.5-VL-7B-Instruct --tuner_type lora --dataset /mnt/workspace/data.json --deepspeed zero2 --max_length 16384
```
If the assistant field in the dataset is empty, remove this empty string for inference, as it can cause NaN values during training and will be checked for.

### Q3: Issues related to loading datasets from cache
Setting the command-line parameter `--load_from_cache_file true` can speed up dataset loading, especially for multimodal datasets or large datasets. When debugging or modifying a preprocessor, set it to false. For more information, search for this parameter in the [Command-line Parameters documentation](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html).

### Q4: How do I set up the Swift environment? Are there Docker images available?
For environment setup, see the [Swift Installation documentation](https://swift.readthedocs.io/en/latest/GetStarted/SWIFT-installation.html). Recommended versions for some common dependencies can be found on the [homepage](https://github.com/modelscope/ms-swift/blob/main/README.md). The documentation provides a Docker image. You can start a container using the `docker run` command, for example: `docker run --gpus all -p 8000:8000 -it -d --name ms modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.9.0-vllm0.13.0-modelscope1.33.0-swift3.12.5 /bin/bash`. After starting the container, pull the latest code and install Swift.

### Q5: Questions about multimodal model training data formats, parameter freezing, and optimizer settings
See [examples](https://github.com/modelscope/ms-swift/tree/main/examples/train/multimodal) for multimodal model training. It supports training with text-only data, image-text data, or a mixture of both. For parameters related to images, videos, and audio, such as max pixels, fps, etc., please see [Model-specific Parameters](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html#specific-model-arguments).
In Grounding tasks, the general data format supports one object corresponding to multiple bboxes. Refer to the [Custom Dataset documentation](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html). videos can be a list of images specified via a file directory.
Swift resizes images based on max_pixels and saves both the pre-processed and post-processed images, then adjusts the bboxes accordingly. However, this adjustment is not done during inference, so you need to manually process the images beforehand.
To reduce GPU memory usage when training VLM models, configure `--freeze_vit true` and the `--max_pixels` parameter to limit the maximum pixels. For details on parameters like `--freeze_vit`, `--freeze_aligner`, and `--freeze_llm`, see the Tuner section in the [Command-line Parameters documentation](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html). If the ViT is not being trained, it is normal to see a warning like `"none of the inputs have requires_grad=True"`. If it is being trained, this warning should not appear.
To perform full-parameter fine-tuning on the visual encoder while using LoRA to fine-tune the LLM, refer to this [example](https://github.com/modelscope/ms-swift/tree/main/examples/train/multimodal/lora_llm_full_vit).

### Q6: Issues related to templates
Since Jinja chat templates do not have labels, they are not supported for training.
For multimodal datasets, if you need to perform dynamic data augmentation after loading the data (e.g., randomly adding noise to the input), please modify the encode method in the template.

### Q7: How to debug Swift training?
See the [Pre-training and Fine-tuning documentation](https://swift.readthedocs.io/en/latest/Instruction/Pre-training-and-Fine-tuning.html).

### Q8: How to use a Python script for Swift training?
Refer to the [notebook examples]((https://github.com/modelscope/ms-swift/tree/main/examples/notebook)).

### Q9: How to use the UI for Swift training?
Use the `swift web-ui` command. Training via the UI is consistent with the command line; for UI parameters, please refer to the command-line parameter documentation. The usage of custom datasets is the same as described in Q2 above. Megatron-swift does not support UI training.

### Q10: Issues related to single-node, multi-GPU training
Swift's multi-GPU training relies on torchrun. `deepspeed` and `device_map` are incompatible; you can only choose one. For more details, see the [examples](https://github.com/modelscope/ms-swift/tree/main/examples/train/multi-gpu) in the code repository.

### Q11: Issues related to multi-node, multi-GPU training
Please see the [multi-node, multi-GPU examples](https://github.com/modelscope/ms-swift/tree/main/examples/train/multi-node). During multi-node, multi-GPU training, only the main node produces logs.
If multi-node training is slow (e.g., a significant speed drop when using DeepSpeed ZeRO3), please check this [issue](https://github.com/modelscope/ms-swift/issues/1825).

### Q12: Issues related to large-scale datasets
If the dataset is very large and tokenization takes a long time each run, use lazy_tokenize or streaming. See the [Command-line Parameters](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html) for details.

### Q13: Issues related to resuming from a checkpoint
Keep the parameters from the previous training script and add `--resume_from_checkpoint output/xxx/vx-xxx/checkpoint-xxx`. For details, see [Command-line Parameters](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html). If the dataset has changed and you only want to load the model, also set `--resume_only_model`. For more complex scenarios, search for "resume" in the command-line parameters documentation.

### Q14: Issues related to streaming dataset loading
For streaming (`--streaming true`), data is loaded while training. You must set max_steps. For details, see the documentation for the streaming parameter in the [Command-line Parameters documentation](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html#data-arguments).
Note: Streaming does not shuffle the data and does not automatically create a validation set. The validation set must be specified using the `--val_dataset` command-line parameter.
When resuming training with streaming, the data can only be indexed forward, not randomly. Skipping already trained data is very time-consuming, so using streaming for resuming is not recommended.

### Q15: Issues related to packing
Packing should be used with FlashAttention; otherwise, there will be discrepancies, and the attention_mask will have issues. The packing_cache parameter needs to be set to a shared disk path for multi-node training.
The linear-attention in the Qwen3.5 model does not support var_len, so enabling packing is not recommended.
When packing is enabled, multimodal data will undergo two map operations: one for the dataset and one for the template. If this is very slow, you can set `OMP_NUM_THREADS=14` to accelerate it, or disable packing to avoid the second mapping.

### Q16: Dataset Multiprocessing
When the dataset mapping process is slow, you can enable multiprocessing by setting the `--dataset_num_proc` parameter. It is normal for the mapping process of multimodal datasets to be slow.

### Q17: How many checkpoints are saved by default after training?
By default, all checkpoints are saved. For details, see the [command-line parameter save_total_limit](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html).

### Q18: Loss and Accuracy During Training
Custom loss functions can be added in the plugin. If you need loss curves for different datasets, set `--enable_channel_loss`.
If the accuracy (acc) from evaluation does not match the accuracy calculated by re-running inference on the corresponding saved checkpoint (ckpt), it is because the calculation methods for eval_acc during training and acc during inference are different. `acc_strategy`: The default is `'token'`. Available options include `'token'` and `'seq'`.
token_acc might not be available during training because for some models, the number of `logits` and `labels` do not match, so it is not calculated.
You can view the currently supported losses or add new ones [here](https://github.com/modelscope/ms-swift/blob/main/swift/loss/mapping.py).
To check if special tokens like `<image>` are included in the loss calculation, you can inspect the printed labels in the command-line log.
When training an agent, tool_call should be included in the loss calculation, while tool_response should not.

### Q19: Issues Related to Freezing Model Parameters
During training, if freezing certain layers causes some parameters to not participate in gradient backpropagation, please configure the parameter `--ddp_find_unused_parameters true`.
Regarding freeze_parameters and freeze_vit/freeze_aligner/freeze_llm: The freezing logic is applied first, and then trainable parameters are activated. The three parameters `freeze_vit`, `freeze_aligner`, and `freeze_llm` will adjust the sets of frozen and trainable parameters. Because the ViT in some models includes an `aligner`, the `aligner` will be added to trainable_parameters separately.
The mechanism of the freeze_parameters_ratio parameter is to freeze layers from the bottom up, starting from the embeddings.

### Q20: Issues Related to Sequence Parallelism
Sequence parallelism supports PT (Pre-Training), SFT (Supervised Fine-Tuning), DPO, and GRPO. Refer to this example: [sequence_parallel](https://github.com/modelscope/ms-swift/tree/main/examples/train/sequence_parallel).
For VLM models, only FlashAttention is currently supported. For text-only models, both FlashAttention and SDPA are supported.
Sequence parallelism can be used simultaneously with the Liger kernel.
If there is a conflict between sequence parallelism and a custom loss, it is because sequence parallelism has its own custom loss implementation. You can modify it yourself [here](https://github.com/modelscope/ms-swift/blob/main/swift/trainers/sequence_parallel/ulysses.py).

### Q21: Expanding the Vocabulary
To expand the vocabulary using the Swift framework, you need to set the command-line parameters `new_special_tokens` and `--modules_to_save embed_tokens lm_head`. For details, see this [example](https://github.com/modelscope/ms-swift/tree/main/examples/train/new_special_tokens).

### Q22: Issues Related to Tuners
LlamaPro in Swift has been adapted for multimodal tasks.
LongLoRA can only be used with LLaMA-series models.
LoRA training is incompatible with the `--trainable_parameters` parameter. For other trainable parameters outside the LoRA modules, use modules_to_save.

### Q23: Embedding/Reranker Training
[Example](https://github.com/modelscope/ms-swift/blob/main/examples/train/embedding) for training embeddings.
[Example](https://github.com/modelscope/ms-swift/tree/main/examples/train/reranker) for training a reranker.
For the data format, see [Custom Datasets](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html).

### Q24: Training for Classification Tasks
Swift supports multi-label classification. The data format is described in the custom dataset documentation. Search for `problem_type` in the command-line parameter documentation. Other aspects are the same as for regression tasks.
Note: The label field should be at the same level as the message field.

### Q25: Training a 'Thinking' Model
See this [issue](https://github.com/modelscope/ms-swift/issues/4030).

### Q26: Does Swift support distillation?
Yes. Refer to this [example](https://github.com/modelscope/ms-swift/blob/main/examples/sampler/distill/distill.sh).

### Q27: For GKD training, do the student and teacher models need to have the same model_type? For example, can one be a dense model and the other a MoE model?
Yes, this is possible, as long as their vocabularies are the same. However, using a MoE (Mixture of Experts) model will be slower.

### Q28: Issues Related to GRPO Training
Swift now supports GRPO training for multimodal tasks.It is normal for the loss to approach 0 during GRPO training. See this [issue](https://github.com/huggingface/open-r1/issues/239#issuecomment-2646297851) for reference.
Set sleep_mode to make the VllmEngine release GPU memory after inference is complete. The engine will be reloaded on the next call instead of occupying memory continuously.
If you do not want to include the KL term during GRPO training, you can configure it using the beta command-line parameter.
To continue with GRPO training after LoRA fine-tuning, search for `--adapters` in the command-line parameter documentation.
Because calculating entropy incurs some extra overhead, the entropy curve is not logged by default. If you need it, set `--log_entropy true`.
The colocate mode does not support use_async_engine.
GRPO does not support channel_loss.
The Liger kernel and padding-free cannot be enabled simultaneously during the GRPO phase. Doing so would require modifying the Liger GRPO loss implementation within the Liger kernel library, which is not convenient.
If your training set contains different tasks, please refer to [Multi-task Training](https://swift.readthedocs.io/en/latest/Instruction/GRPO/DeveloperGuide/multi_task.html).

### Q29: Issues Related to Reward Functions/Models
reward_model and reward_funcs can be used together.
For custom reward functions, refer to [examples/train/grpo/plugin](https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo/plugin).
For math problems, you need to pass the solution from the dataset; otherwise, it is difficult to calculate accuracy.
If you need to pass a specific column from the dataset into a custom reward function for ORM, place that column outside of the messages field.
If you need to specify an LLM-judge model for scoring during GRPO training, please refer to the reward model documentation.

### Q30: Issues Related to Rollout
Rollout is likely incompatible with pipeline parallelism.
The vLLM inference engine has trust_remote_code set to true by default.

### Q31: I have a question: in the GRPO script, does save_steps refer to the "step" or the "global step"? Currently, my local training shows global step as 18, while wandb shows step as 628.
It refers to the `global_step` displayed by the local tqdm progress bar.

### Q32: If num_iterations=1 is used by default, the clip function becomes ineffective, right? DAPO's clip_higher also doesn't work. I've seen that veRL has a micro-batch setting to update the policy model in smaller batches within a single epoch to make the clip term effective. Looking at the source code, it seems ms-swift's mini-batch only performs gradient accumulation?
Yes, num_iterations needs to be greater than 1.

### Q33: Does GSPO training support the top_entropy_quantile parameter? After passing --importance_sampling_level sequence, can I still optimize the top x% of tokens based on the entropy distribution?
Yes, it's supported. The order of operations is: first, the loss is calculated normally (affected by importance_sampling_level), and then the loss is masked based on top_entropy_quantile.

### Q34: FAQ in the GRPO documentation
For more GRPO-related FAQs, please refer to the [GRPO documentation](https://swift.readthedocs.io/en/latest/Instruction/GRPO/GetStarted/GRPO.html#faq).

### Q35: Issues Related to PPO and Other Preference Training Methods
PPO training does not support gradient clipping.
Currently, PPO only supports scenarios where the reward model (RM) and the policy model belong to the same model series (i.e., same tokenizer/template).
Multi-turn DPO is not supported.

### Q36: Issues Related to MoE Model Training
When training a Mixture-of-Experts (MoE) model with LoRA, if the aux_loss barely changes, add all_router to target_modules.
During LoRA training, whether the router modules are trained depends on whether their gates are implemented as nn.Linear. If they are nn.Parameter, they are not trained. For details, see the command-line parameter [target_parameters](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html#tuner-arguments).

### Q37: Issues Related to Megatron-SWIFT Training
For checkpoint saving, refer to the command-line parameter [save_strategy](https://swift.readthedocs.io/en/latest/Megatron-SWIFT/Command-line-parameters.html).
During Megatron multi-node training, logs are printed on the last pipeline parallelism (PP) rank, not the master node, because only the last PP rank has the complete information.
Megatron-SWIFT now supports save_total_limit and SwanLab for monitoring training. See the [Megatron-SWIFT command-line parameters documentation](https://swift.readthedocs.io/en/latest/Megatron-SWIFT/Command-line-parameters.html) for details.
ViT uses the Transformers model architecture and currently does not support parallelism. If you encounter an Out-Of-Memory (OOM) error during training, reduce `decoder_first_pipeline_num_layers`.
Megatron-SWIFT supports new models. There is no tutorial available at the moment; please refer to the Pull Requests (PRs) for adding new models.
The degree of parallelism for sequence_parallel is equal to the tensor parallelism (TP) degree.
FP8 training supports block-wise implementation. Refer to the FP8 example in [examples/megatron/fp8](https://github.com/modelscope/ms-swift/tree/main/examples/megatron/fp8).

### Q38: How do I configure resuming from a checkpoint in Megatron-SWIFT?
To resume training, use `--mcore_model` to load the checkpoint. Additionally, configure these parameters as needed: `--finetune`, `--no_load_optim`, `--no_load_rng`. To resume LoRA training from a checkpoint, configure `--mcore_adapter`; other settings are the same as for full-parameter training. For details, see the [Megatron-SWIFT command-line parameters documentation](https://swift.readthedocs.io/en/latest/Megatron-SWIFT/Command-line-parameters.html).

### Q39: Issues Related to MTP
To enable MTP training, set the command-line parameter `mtp_num_layers`.
If the base model does not include an MTP structure, you can initialize and train the MTP from scratch.
Multi-modal MTP is not yet supported.

### Q40: I have a question about Megatron GKD. If the teacher is Qwen3-235B and the student is Qwen3-30BA3B, for SFT on the 235B model, I used pp=8 and set decoder_first and decoder_last to 11. If I also set decoder_first/last during GKD, will it affect the student's parallelism?
Currently, the parallelism parameters are shared between the teacher and student models. Support for different parallelism settings for each model will be introduced in a version after v4.

### Q41: Issues Related to Quantized Model Training
For QLoRA fine-tuning, refer to this [example](https://github.com/modelscope/ms-swift/tree/main/examples/train/qlora).
Quantized models cannot be fully fine-tuned. The integer-type parameters in GPTQ models cannot be used for gradient calculation, so only attached structures like LoRA can be updated.
For merging the model after QLoRA training, refer to the [QLoRA example](https://github.com/modelscope/ms-swift/tree/main/examples/train/qlora).
Megatron-SWIFT does not support QLoRA training.

### Q42: Training Specific Models
SWIFT currently does not support training MiniCPM-O with audio modal input.
To fine-tune DeepSeek-VL-2, use a transformers version earlier than 4.42 and `peft==0.11.*`.
Fine-tuning Moonlight-16B-A3B-Instruct: Training is disabled in the model files. Refer to the solution for DeepSeek-VL-2, which can be found by searching the issues.
Fine-tuning the Ovis-2 model is special; it requires padding to max_length. Set the `--max_length` argument.
Qwen2.5-Omni currently only supports "thinker" mode for training, not "talker" mode.
SFT for Qwen2-Audio does not support packing.

### Q43: On devices that do not support Flash Attention, what is the default attention_implementation? The documentation says the default is none.
The default implementation used is sdpa.

### Q44: Is left padding the default for model training?
You can choose either left or right padding for training. The default is right padding, while `batch infer` always uses left padding.

### Q45: What are the parameters for MoE? I can't find them by searching for keywords in the parameter list. How do I configure parameters like the number of experts and expert routing?
Use the parameters directly from the config.json file.

### Q46: Can swift support setting a minimum learning rate? I feel like it decreases too much towards the end.
Yes, you can set it with `--lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs '{"min_lr": 1e-6}'`.

### Q47: Does it currently support configuring GRPO and SFT using YAML files?
Yes, both are supported. The configuration is directly processed into command-line arguments in main.py.

### Q48: Is it true that use_liger_kernel and log_entropy cannot be used together right now?
They are not supported together.

### Q49: How can I handle this error? Installing apex didn't help.
```text
RuntimeError: ColumnParallelLinear was called with gradient_accumulation_fusion set to True but the custom CUDA extension fused_weight_gradient_mlp_cuda module is not found. To use gradient_accumulation_fusion you must install APEX with --cpp_ext and --cuda_ext. For example: pip install --global-option="--cpp_ext" --global-option="--cuda_ext ." Note that the extension requires CUDA>=11. Otherwise, you must turn off gradient accumulation fusion.
```
Set `--gradient_accumulation_fusion false`.

### Q50: If I finetune a VLM on several tasks together, and the video sampling rules for different tasks are inconsistent, does ms-swift support this? Where can I configure it?
Check `interleave_prob` in the [Command-line Parameters Documentation](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html).

### Q51: I have a question. During multimodal packing pre-training, it seems the GPU memory usage increases slightly after each "pytorch allocator cache flushes since last step," leading to an OOM error after many steps.
Add the environment variable `PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'`.

### Q52: Can use_logits_to_keep be used with large multimodal models now?
It will cause an error if the expansion of multimodal tokens happens within the model's forward pass.

### Q53: Why does the GPU memory usage increase significantly several times during training, for example, at step 50 or 100?
Set the `PYTORCH_CUDA_ALLOC_CONF` environment variable. For details, please refer to the PyTorch documentation.

### Q54: Is there a practical guide for fine-tuning a Qwen base model into a chat model? Are there any special configurations needed?
Use `swift sft`. No other special configurations are needed. Refer to the [example](https://github.com/modelscope/ms-swift/tree/main/examples/train/base_to_chat).

### Q55: After training, the model's responses contain a lot of repetitive content.
Please refer to [Pre-training and Fine-tuning](https://swift.readthedocs.io/en/latest/Instruction/Pre-training-and-Fine-tuning.html). If repetition occurs during training, try training for more epochs, cleaning the data, performing full-parameter fine-tuning, or using RLHF to mitigate the issue.

### Q56: Why does using --torch_dtype float16 (my GPU doesn't support bf16) result in the error: lib/python3.12/site-packages/torch/amp/grad_scaler.py", line 260, in _unscale_grads_ raise ValueError("Attempting to unscale FP16 gradients.") ValueError: Attempting to unscale FP16 gradients.
For full-parameter fine-tuning, you cannot train with fp16.

### Q57: I'm getting an error when merging LoRA parameters. My current PEFT version is 0.11.0. Is this because the PEFT version needs to be upgraded?
```text
File "/opt/conda/lib/python3.9/site-packages/peft/config.py", line 118, in from_peft_type
  return config_cls(**kwargs)
TypeError: __init__() got an unexpected keyword argument 'corda_config'
```
This is caused by a mismatch between the PEFT versions used for training and merging.

### Q58: How to solve this problem? safetensors_rust.SafetensorError: Error while deserializing header: HeaderTooLarge
You are running out of disk space, and the model was not saved completely.

### Q59: Why does this error appear here? I can't find where numpy.object is.
Try `numpy==1.26.3`.

### Q60: Training with unsloth, I get the error: assert(type(target_modules) in (list,tuple,)). The parameter I configured is --target_modules all-linear.
Don't use `all-linear`. Change it to a specific list of modules, for example, `--target_modules q_proj k_proj v_proj`.

### Q61: For qwen2.5-omni, does --freeze_vit false mean that both the visual encoder and the audio encoder are unfrozen? Is there a way to unfreeze only the audio encoder and not the visual encoder?
Use the `--target_regex` argument.

## Inference

Swift supports inference via Python scripts, command line, and UI interfaces. For details, see [Inference and Deployment](https://swift.readthedocs.io/en/latest/Instruction/Inference-and-deployment.html).

### Q1: How to set up a model for inference in Swift?
For models from full-parameter training, models merged after LoRA training, or models downloaded from the Model Hub, set the command-line argument `--model <model_id_or_path>`. For unmerged models after LoRA training, use `--adapters`, and you can also specify the base model path with `--model`.

### Q2: How to use a dataset for inference in Swift? Where are the inference results saved?
Use `--val_dataset <your-val-dataset>` to specify the dataset. For trained models, you can also set the `--load_data_args true` argument. The path to save inference results is set via `--result_path your_path`, and the path will be printed in the logs. For details, see the documentation on [Command-line parameters](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html).
If you need to keep extra fields from the inference dataset, set `--remove_unused_columns false`.

### Q3: How to set up batch inference in Swift?
If the infer_backend is `transformers`, set the command-line argument `--max_batch_size 16`, or use a [Python script](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo.py). Here, max_batch_size refers to the batch size per GPU card.

### Q4: How to set up streaming inference in Swift?
Set `--stream true`. In this case, the inference results will be written to a jsonl file line by line. Note that streaming inference does not support DDP.

### Q5: Questions related to vLLM and SGLang inference backends
For models trained with LoRA, please check the vLLM and SGLang documentation. If they support LoRA inference, merging the adapters is not required. Additionally, SGLang inference does not currently support multi-modality.

### Q6: Questions related to generation parameters
Parameters like temperature are read from generation_config.json by default. Set `--temperature 0` or `--top_k 1` to disable randomness in inference.

### Q7: How to set the system prompt to empty? If I don't set the system parameter in the command line, a default one is still added.
Set `--system ''`.

### Q8: How to compute metrics like ACC/ROUGE during inference?
Refer to the documentation on inference parameters for [metrics](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html#inference-arguments).

### Q9: During model inference, which parameter should be set to continue generation from a specific prefix?
Use the `--response_prefix` parameter.

### Q10: The 'answer' in my data already contains part of the prompt. How should I modify the inference to complete the 'answer'?
```text
{"messages": [{"role": "system", "content": "<system>"}, {"role": "user", "content": "<query1>"}, {"role": "assistant", "content": "answer1, "}]}
```
This is supported in Swift versions 3.0 and later. Refer to [examples/infer/demo_agent](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo_agent.py).

### Q11: During multimodal model inference, how can I limit the maximum number of pixels to reduce GPU memory usage?
Set the command-line argument `--max_pixels xxx`, the environment variable `MAX_PIXELS=xxx`, or the specific model argument `--model_kwargs '{"max_pixels": xxx}'`. The environment variable only affects the models specified in the documentation. For details, see the documentation on [Specific Model Arguments](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html#specific-model-arguments).

### Q12: How to output probability values (logprobs) during Swift inference?
For command-line inference, set `--logprobs true`. For Python script inference, set `request_config = RequestConfig(..., logprobs=True, top_logprobs=2)`. Refer to [test_logprobs.py](https://github.com/modelscope/ms-swift/blob/main/tests/infer/test_logprobs.py).

### Q13: How to output last_hidden_state during Swift inference?
There is no direct example, but you can refer to the `_get_last_hidden_state` method in the GRPO trainer.

### Q14: Issues with inconsistent inference results between Transformers, vLLM, Ollama, etc.
Swift's templates are aligned with those of Transformers. Check if the inference parameters are consistent. Additionally, there are differences between VllmEngine and TransformersEngine.

### Q15: Inference for embedding/reranker models
For embedding model inference, refer to the [example](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo_embedding.py) here. For reranker model inference, refer to the [example](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo_reranker.py) here.

### Q16: When using a Python script for inference, how can I use the CPU?
Set the environment variable: `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'`.

### Q17: Does the swift infer command support multi-machine inference?
If the model can fit on a single node, you can orchestrate it using Kubernetes. If the model does not fit on a single node, multi-machine inference is not supported.

### Q18: When sampling with Swift, it seems batching is not supported? It looks like it samples one by one in a for loop, which is a bit slow.
There is a [script](https://github.com/modelscope/ms-swift/blob/main/examples/train/rft/rft.py) that can use multiple processes to split the dataset for sampling.

### Q19: Issues related to specific model dependency versions
If Qwen2-Audio inference results are garbled, please use transformers==4.48.
LoRA models trained with transformers==4.55.2 can no longer be loaded by versions older than 4.52. See [issue#5440](https://github.com/modelscope/ms-swift/issues/5440) for details.
Swift is compatible with different versions of qwen-vl-utils, so you do not need to switch its version when using qwen2.5-vl and qwen3-vl models.

### Q20: I got an error: safetensors_rust.SafetensorError: Error while deserializing header:MetadataIncompleteBuffer
The model weights are corrupted.

## Export

### Q1: Errors related to autoawq
If you encounter autoawq-related errors during inference without using an AWQ-quantized model, try uninstalling autoawq and running the inference again. For models that do not support AWQ quantization, you can try using GPTQ for quantization instead.

### Q2: The model does not fit on a single GPU during Swift quantization
Try setting the `--device_map cpu` flag. Alternatively, you can load the model across multiple GPUs but perform the quantization on a single GPU.

### Q3: I am trying to quantize the Qwen2.5 72B model to GPTQ int4 using swift export. I'm using the default max_model_length=32768 and a calibration dataset with 128 samples. However, the quantization process failed with the following error log: factorization could not be completed because the input is not positive-definite (the leading minor of order 18145 is not positive-definite). What could be the reason for this?
This is an issue caused by the Hessian matrix not being positive-definite. Please try using a different calibration dataset.

### Q4: If I pass a custom template_type when using swift export (e.g., swift export --template_type custom), will this permanently change the template associated with the model?
No, it will not be permanently modified. In Swift, templates are defined within the framework's internal code; they are not saved as part of the model files, for instance, in a Jinja format.

### Q5: Can the model be directly converted to GGUF format after training?
Currently, only exporting to ModelFile is supported. For details, please refer to the documentation on [Command-line parameters](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html).

## Deployment

### Q1: How do I set up a model for Swift deployment?
This is the same as Q1 for inference.

### Q2: How do I perform multi-GPU deployment with Swift?
For details, see the [example](https://github.com/modelscope/ms-swift/tree/main/examples/deploy). If you are using the transformers engine, it does not support DDP, so multi-GPU deployment is not possible. Furthermore, heterogeneous deployment (e.g., using different GPU models or assigning different memory ratios to each GPU) is not supported.

### Q3: Regarding the system prompt, you can specify it via the --system parameter, prepend it to each data entry in the dataset, or define it in the template. Is it sufficient to use just one of these methods? And are they all treated the same way by the model?
System prompt priority: The one in the dataset > The one from the command line > The default one in the template.

### Q4: Questions about multimodal input from the client.
To pass images, audio, and other media from the client, please see the [client example](https://github.com/modelscope/ms-swift/tree/main/examples/deploy/client/mllm). If you encounter an invalid image URL, you can set a request timeout either through the `SWIFT_TIMEOUT` environment variable or as a parameter in `InferClient`.

### Q5: Questions about setting generation parameters.
For inference, parameters like temperature can only be set before launching. For deployment, you can set default values at launch, which can later be overridden by client-side settings.

### Q6: How do I enable streaming generation for a model deployed with Swift?
This is controlled on the client side. Please refer to the examples in [examples/deploy/client](https://github.com/modelscope/ms-swift/tree/main/examples/deploy/client).

### Q7: How can I output token probabilities in a Swift deployment?
Set `--logprobs true` on the server side. Then, the client needs to pass the corresponding parameters, for example: `request_config = RequestConfig(..., logprobs=True, top_logprobs=2)`.

### Q8: Questions about the "thinking" process during model deployment.
If you need to disable the "thinking" step, it can currently only be done at launch time with swift deploy. See this [issue](https://github.com/modelscope/ms-swift/issues/4030) for more information.

### Q9: During deployment, which parameter should be set to generate multiple outputs in a single request?
The `n` parameter in `RequestConfig`.

### Q10: Questions regarding deployment with Swift using --infer_backend vllm compared to deploying directly with VLLM.
If there's a significant difference in inference results, it's likely due to a template mismatch. If there's a significant difference in inference speed, it might be caused by inconsistent image resolutions. Swift uses the V1 engine by default; you can control this with the environment variable `VLLM_USE_V1=1`.

### Q11: Questions about specific models and dependency versions.
If you get an error message about a missing "model.language_model.embed_tokens.weight", it indicates a version mismatch in the transformers library between training and inference. If you encounter garbled text when running inference with Qwen2.5 using FP16, try using BF16.

### Q12: I have a question: After deploying Qwen2-7B, when I use the client, I have to call the OpenAI API with client.completions.create and cannot use client.chat.completions.create. However, when using the qwen2-7b-instruct-q5_k_m.gguf model, I can use client.chat.completions.create. Why is this?
Base models can use client.chat.completions.create, but this is intended as a compatibility feature.

## Evaluation

### Q1: What evaluation datasets does Swift support? And how can I use a custom evaluation dataset?
For details on using standard and custom evaluation datasets, please refer to the documentation on [Evaluation](https://swift.readthedocs.io/en/latest/Instruction/Evaluation.html).

### Q2: After manually downloading an officially supported evaluation dataset, can swift eval be configured to evaluate using a local path?
For offline evaluation, please refer to the EvalScope documentation's [Quick Start](https://evalscope.readthedocs.io/en/latest/get_started/basic_usage.html).

### Q3: The model after eval fine-tuning keeps stopping at a fixed percentage, but the vllm service seems to be running normally. The larger the model, the sooner it disconnects.
Set the `SWIFT_TIMEOUT` environment variable to -1.

### Q4: Can I control the number of dataset entries during evaluation? It takes over an hour to evaluate an MMLU, which is too slow.
Use the configuration parameter `--eval_limit`. This `--eval_limit` controls the number of entries in each subset. For example, if MMLU has over 50 subsets, and each limit is set to 10 entries, then that would be over 500 entries in total.

### Q5: In swift eval, the model stops generating after 1024 tokens. How can I modify this? Setting --max_new_tokens 5000 doesn't seem to work.
Check the command-line parameter [eval_generation_config](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html#evaluation-arguments).

### Q6: How can I load downloaded datasets locally when using opencompass backend for evaluation?
The opencompass backend doesn't support setting `data_args`.

### Q7: Does swift eval with --eval_backend OpenCompass not support custom datasets?
```text
ValueError: eval_dataset: /mnt/workspace/data.jsonl is not supported.
eval_backend: OpenCompass supported datasets: ['C3', 'summedits', 'WiC', 'csl', 'lambada', 'mbpp', 'hellaswag', 'ARC_e', 'math', 'nq', 'race', 'MultiRC', 'cmb', 'ceval', 'GaokaoBench', 'mmlu', 'winogrande', 'tnews', 'triviaqa', 'CB', 'cluewsc', 'humaneval', 'AX_g', 'DRCD', 'RTE', 'ocnli_fc', 'gsm8k', 'obqa', 'ReCoRD', 'Xsum', 'ocnli', 'WSC', 'siqa', 'agieval', 'piqa', 'cmnli', 'cmmlu', 'eprstmt', 'storycloze', 'AX_b', 'afqmc', 'strategyqa', 'bustm', 'BoolQ', 'COPA', 'ARC_c', 'PMMEval', 'chid', 'CMRC', 'lcsts']
```
OpenCompass doesn't support custom datasets; use native mode for custom datasets.

### Q8: Evalscope can natively generate reports, but other backends like OpenCompass do not support report visualization, correct?
Currently, only native visualization is supported; other backends are not yet supported.

### Q9: Could you explain what causes the following error when using ifeval for evaluation?
```text
[Errno 20] Not a directory: '/root/nltk_data/tokenizers/punkt_tab.zip/punkt_tab/english/collocations.tab'
```
Unzip the file using `unzip /path/to/nltk_data/tokenizers/punkt_tab.zip`.

### Q10: When evaluating with eval_backend='OpenCompass', how can I specify the path to offline datasets?
Check the [data preparation guide](https://evalscope.readthedocs.io/en/latest/user_guides/backend/opencompass_backend.html#data-preparation), download and unzip the dataset. You don't need to specify `dataset-args`; just place the dataset folder (the data folder) in the current working directory.

### Q11: What causes the following error when using evalscope?
```text
unzip: cannot find or open /root/nltk_data/tokenizers/punkt_tab.zip, /root/nltk_data/tokenizers/punkt_tab.zip.zip or /root/nltk_data/tokenizers/punkt_tab.zip.ZIP
```
This occurs during the download of nltk dependencies. Manually download [punkt_tab.zip](https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/open_data/nltk_data/punkt_tab.zip) and unzip it under `~/nltk_data/tokenizers`.

### Q12: Why is there no issue with plain text, but when testing multi-modal data, even though we specify the path, it still fails to detect the dataset and attempts to download it?
The VLMEvalKit process differs from native; it will automatically download data to `~/LMUData/`. For details, see the [documentation](https://evalscope.readthedocs.io/en/latest/user_guides/backend/vlmevalkit_backend.html#data-preparation).

### Q13: When using swift eval for benchmark evaluation, can I specify an llm as a judge and how should I pass in the parameters?
Yes, you can use swift to pass `judge-model-args` parameters from `extra_eval_args`, which include `api_key, api_url, and model_id`, as a JSON string.

### Q14: What could be the reason for uneven GPU memory allocation across multiple cards when running eval?
```shell
NPROC_PER_NODE=8
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\ MAX_PIXELS=802816\ swift eval\
--model "$MODEL_PATH” \$EXTRA_ARGS \
--eval_backend Native \ --infer_backend transformers\ --device_map auto \
--eval_limit"$EVAL_LIMIT"\ --eval_dataset general_qa\
--dataset_args "{\"general_qa\": {\"local_path\": \"${DATA_PATH}\", \"subset_list\": [\"${SUBSET_NAME}\"]}}" \ --host 127.0.0.1\> "$LOG_FILE" 2>&1
```
swift eval does not support being launched in DDP mode.

### Q15: Where can I see what extra fields, besides the question itself, are included in the query sent during a Swift evaluation?
The simplest way is to check the input field in the output reviews file. It contains the content sent to the model, converted to Markdown format. Note that this is not available if you are using the opencompass backend; you need to use the native backend.

The evaluation capabilities of ms-swift utilize the ModelScope community's evaluation framework, EvalScope. For more advanced features, please use the [EvalScope](https://evalscope.readthedocs.io/en/latest/get_started/introduction.html) directly.
