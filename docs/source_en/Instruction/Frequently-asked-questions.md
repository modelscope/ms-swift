# Frequently-asked-questions

Here are some common questions encountered during the use of SWIFT.

## Dataset
SWIFT comes with 150+ built-in datasets for various tasks such as pre-training, fine-tuning, human eye alignment, and multimodal simulations, and also supports custom datasets. See [Homepage](https://github.com/modelscope/ms-swift/blob/main/README_CN.md) for details.

### Q1: What datasets does SWIFT support? How do I use a custom dataset? How do I download a dataset? How do I inspect a dataset?
- For a list of supported datasets, please see [Supported Models and Datasets](https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html).
- For details on custom dataset formats and usage, please refer to [Custom Datasets](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html). Datasets that conform to the format will automatically call Swift's built-in data preprocessor. If the format does not match the documentation requirements, please refer to the supported datasets and convert the format yourself. If your custom dataset contains additional fields, these fields will not be used by default. You can configure them using `--remove_unused_columns`.
- When you need to download the dataset and then use it by specifying the path, you can download it locally through `git clone` and specify it through the `dataset_path` field in the dataset_info.json file. For details, please see the [Customized Dataset Document](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html#dataset-info-json). The download mode of the data set can choose to re-download or reuse the last download, specified by `--download_mode`.
- To perform error checking on the data set, please set the command line parameter `--strict True`. When you need a data set quality inspection tool, you can check out another library [data-juicer](https://github.com/modelscope/data-juicer). Data can be randomized using `--dataset_shuffle true`.
For more instructions, please search for the corresponding parameters in [Command Line Parameter Document](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html).

### Q2: Common Dataset Errors
- Due to PyArrow's strict type control over datasets, the `objects` section of the image grounding dataset and the `tools` section of the agent dataset must use the `str` type; otherwise, an error will occur indicating inconsistent data types across rows.
- If you encounter the error `AttributeError: 'TrainerState' object has no attribute 'last_model_checkpoint'` during training, it may be because the dataset is too small, resulting in insufficient data for one step. Try expanding the dataset to resolve this. Similarly, a similar error will occur if the split validation set data is too small.
- Below is an error caused by an empty assistant field:
```shell
File "/your_workspace/ms-swift/swift/1lm/dataset/preprocessor/core.py", line 69, in _check_messages raise
ValueError(f'assistant_message; {assistant_message}')
ValueError: assistant_message: {'role' :'assistant', 'content': ''}
```
If it's for inference, you can simply delete the empty assistant message.

### Q3: Issues related to loading datasets from the cache
Setting the command-line argument `--load_from_cache_file True` can speed up dataset loading (excluding initial loading), especially in scenarios with multimodal datasets or large datasets. When debugging or modifying the preprocessor, set it to false to ensure code changes take effect. For more information, please search for this parameter in the [Command-line Arguments documentation](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html).

### Q4: Multimodal Model Dataset Related Issues
- Examples of multimodal model training are available at [https://github.com/modelscope/ms-swift/tree/main/examples/train/multimodal]. Training with plain text or image/text data is supported, as well as training with a mixture of both. For parameters related to images, videos, and audio, such as maximum pixels and FPS, please refer to [Specific Model Parameters](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html#specific-model-arguments).
- The common data format in the Grounding task supports multiple bounding boxes for one object. Refer to the documentation on [Custom Datasets](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html#grounding). During training, SWIFT will adjust images exceeding `--max_pixels` and save the images before and after preprocessing, while also adjusting the bounding boxes. No adjustments are made during the inference stage; manual image processing is required beforehand.

### Q5: Issues related to large-scale data sets
The data set is too large, and each tokenize takes a long time. Please use `--lazy_tokenize True` or streaming reading `--streaming True`. For details, see [Command Line Parameter Document](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html).

### Q6: Issues related to streaming loading of data sets
Streaming loading `--streaming True`, loading while training, needs to set max_steps. For detailed instructions, please search for this parameter in [Command Line Parameter Document](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html#id4). <br>
Note:
- Streaming is not random and does not divide the verification set. The verification set is specified through the command line parameter `val_dataset`.
- When resuming training from a breakpoint, streaming can only index forward and cannot index randomly. It takes a long time to skip the data that has been trained. It is not recommended to use streaming.

### Q7: Multi-process processing of data sets
It is normal for multi-modal dataset map to be slow. You can set the parameter `--dataset_num_proc` to open multiple processes to speed up.

## Training

SWIFT supports training methods including pre-training, instruction-supervised fine-tuning, preference learning, GRPO, Embedding, Reranker, sequence classification tasks, etc. See [Homepage](https://github.com/modelscope/ms-swift/blob/main/README_CN.md) for details.

### Q1: How to set up a SWIFT environment? How to install SWIFT offline? Are there any mirrors available?
- For detailed environment setup instructions, please refer to the [SWIFT Installation Documentation](https://swift.readthedocs.io/en/latest/GetStarted/SWIFT-installation.html). Recommended versions of some common dependencies can be found on the [GitHub homepage](https://github.com/modelscope/ms-swift/blob/main/README_CN.md).
- SWIFT Offline Installation Process:
```text
1. Clone the image using git (internet connection required)
2. Install locally using pip -e .
```
- The [SWIFT Installation Documentation](https://swift.readthedocs.io/en/latest/GetStarted/SWIFT-installation.html) provides the image address. Use `docker pull` to pull the image and `docker run` to start the container, for example:
```shell
# Pull the image
docker pull modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda13.0.3-py312-torch2.11.0-vllm0.23.0-modelscope1.38.1-swift4.4.1
# Start the container in the background; -d will make the container run in the background for a long time
docker run --gpus all -p 8000:8000 -dit --name ms modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda13.0.3-py312-torch2.11.0-vllm0.23.0-modelscope1.38.1-swift4.4.1 /bin/bash
# Enter the container
docker exec -it ms /bin/bash
```
After starting the container, pull the latest SWIFT code and install it.

### Q2: What models does SWIFT support? How do I download models? How do I set the model storage path?
- For a list of supported models, see [Supported Models and Datasets](https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html).
- If the model has already been downloaded locally, you can use it by setting `--model <model_path>`. For offline training, you need to set both `--model <local_path_to_model>` and `--check_model false`. If you encounter errors related to git clone, you can specify the local repository using `--local_repo_path <local_repo_path>`.
- Models downloaded from ModelScope can be stored in a specified path by configuring the environment variable `MODELSCOPE_CACHE=your_path`. If you download using the ModelScope SDK, you can also specify the model storage path using `--cache_dir="local_path"`. Models can also be downloaded using the `modelscope download` command-line tool or `git`. See the [Model Download](https://modelscope.cn/docs/models/download) section of the Modelscope documentation for details. If you need to download models from Hugging Face, you need to set the environment variable `USE_HF=1`.
- SWIFT will automatically match `model_type`. You can also manually specify it by checking the [Supported Models and Datasets](https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html).
For more information, please search for the corresponding parameters in the [Command-line Parameters Documentation](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html).

### Q3: Template-related issues
- Because the Jinja chat template does not have labels, training this template is currently not supported.
- For multimodal datasets, if dynamic data augmentation (e.g., randomly adding noise to the input data) is required after data loading, please modify the `encode` method in the template.

### Q4: How to debug SWIFT training?
Debugging can be done in the following way, which is equivalent to fine-tuning using the command line, but this method does not support distributed debugging. The fine-tuning command-line entry point can be found [here](https://github.com/modelscope/ms-swift/blob/main/swift/cli/sft.py).
```shell
from swift import sft_main, SftArguments
result = sft_main(SftArguments(
    model='Qwen/Qwen2.5-7B-Instruct',
    tuner_type='lora',
    dataset=['AI-ModelScope/alpaca-gpt4-data-zh#500',
             'AI-ModelScope/alpaca-gpt4-data-en#500',
             'swift/self-cognition#500'],
    torch_dtype='bfloat16',
    # ...
))
```

### Q5: How to use python script to train SWIFT?
Refer to [notebook example](https://github.com/modelscope/ms-swift/tree/main/examples/notebook).

### Q6: How to use UI interface training for SWIFT?
- Use the `swift web-ui` command to start the UI interface. Interface training and custom data set usage are consistent with the command line. For parameters on the interface, please see the [Command Line Parameter Documentation](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html).
- Megatron-SWIFT does not support UI interface training.

### Q7: Issues related to multi-modal model training
- If you need to reduce video memory usage during VLM model training, please configure `--freeze_vit true` and limit the maximum pixels `--max_pixels`. If VIT is not trained, it is normal to throw `warning: none of the inputs have requires_grad=True`, but if it is trained, it should not be thrown.
- `--freeze_vit`, `--freeze_aligner`, `--freeze_llm` parameters are detailed in [Command Line Parameter Document](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html#tuner).
- Full parameter fine-tuning visual encoder + LoRA fine-tuning LLM, refer to [Example](https://github.com/modelscope/ms-swift/tree/main/examples/train/multimodal/lora_llm_full_vit).

### Q8: Issues related to single-machine multi-card training
The bottom layer of SWIFT multi-card training relies on torchrun. `deepspeed` and `device_map` are incompatible, you can only choose one of the two. For more details, please see the [Single-machine multi-card example](https://github.com/modelscope/ms-swift/tree/main/examples/train/multi-gpu) in the code base.

### Q9: Issues related to multi-machine multi-card training
- When training on multiple machines and multiple cards, only the master node has logs. For more details, please see the [Multi-machine and multi-card example](https://github.com/modelscope/ms-swift/tree/main/examples/train/multi-node) in the code base.
- Multi-machine training is slow. For example, training with DeepSpeed ​​ZeRO3 will cause serious speed drops. Please check [issue](https://github.com/modelscope/ms-swift/issues/1825).

### Q10: Issues related to breakpoint continuation of training
- The parameters in the previous training script remain unchanged, just add `--resume_from_checkpoint output/xxx/vx-xxx/checkpoint-xxx`, and the weights and other related information will be read in the trainer.
- If you wish to load only the model, set also `--resume_only_model` to ignore the optimizer state and random seed.
- For more complex scenarios, please search for the parameter keyword `resume` in [Command Line Parameter Documentation](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html).

### Q11: packing related questions
- Packing must be used together with flash_attn, otherwise attention_mask will have problems and cause errors.
- The linear-attention in the Qwen3.5 model does not support var_len, and it is not recommended to enable packing.
- When packaging is turned on, multi-modal data will be mapped twice. After mapping the data set, template mapping will also be performed. If the speed is very slow, you can set `OMP_NUM_THREADS=14` to speed up, or you can remove the packing so that it will not be mapped a second time.

### Q12: How many checkpoints are saved by default after training?
All checkpoints are saved by default. For details, see save_total_limit in the [command line parameter documentation](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html).

### Q13: Loss related issues during training process
- The .py file of the custom loss function is imported through the `--external_plugins` parameter when used:
```shell
swift sft \
    --external_plugins /path/to/plugin.py \
    --loss_type my_loss \
    # ...
```
The value corresponding to `--loss_type` is the key corresponding to the custom loss function registered in loss_map; similarly, loss_scale needs to be registered in loss_scale_map.
- If you need loss curves for different data sets, please set `--enable_channel_loss`. For more information, please search for this parameter in [Command Line Parameter Document](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html).
- You can view the currently supported losses or add new losses in [loss_map](https://github.com/modelscope/ms-swift/blob/main/swift/loss/mapping.py)
- If you need to check whether special tokens such as `<image>` are involved in loss calculation, you can check the printed labels in the command line log.
- When training the agent, `tool_call` counts as loss, and `tool_response` does not count as loss.

### Q14: Acc related issues during training process
- If the acc obtained by eval is inconsistent with the acc calculated by re-inference of the corresponding ckpt, it may be caused by the different calculation methods of eval_acc during training and acc during inference. Check the `--acc_strategy` parameter, the default is `'token'`, optional values ​​include: `'token'`, `'seq'`.
- Some models do not have token_acc during training because the numbers of `logits` and `labels` do not match.

### Q15: Issues related to model parameter freeze
- During the DDP multi-card training process, freezing some layers results in some parameters not participating in gradient return. Please configure `--ddp_find_unused_parameters True` to automatically skip parameters without gradients.
- `--freeze_parameters/--freeze_vit/--freeze_aligner/--freeze_llm`: The freeze_parameters set by these four parameters are allowed to be overwritten by the activate_parameters executed later during use, that is, the parameter unfreezing priority is higher.
- `--freeze vit/--freeze aligner/--freeze llm` will adjust the freeze parameters. Because the ViT of some models contains aligner, trainable parameters will be adjusted synchronously when `--freeze aligner False` is used, and `aligner` is added separately to ensure that it is not frozen.
- The mechanism of `--freeze_parameters_ratio` is to freeze parameters from bottom to top starting from embedding.

### Q16: Sequence Parallel Related Issues
- pt, sft, dpo, and grpo all support sequence parallel. For command-line examples, please refer to [sequence_parallel](https://github.com/modelscope/ms-swift/tree/main/examples/train/sequence_parallel).
- VLM models currently only support flash-attn; plain text models support both flash-attn and sdpa.
- Sequence parallel can be used simultaneously with the Liger kernel.
- When custom loss functions are ineffective under sequence parallel, it may be because sequence parallel uses its own loss function. You can modify [per_token_loss_func_sp](https://github.com/modelscope/ms-swift/blob/main/swift/trainers/utils.py) as needed.

### Q17: Expanding the Vocabulary
Expanding the vocabulary using the SWIFT framework requires setting the command-line argument `--new_special_tokens <path/to/tokens.txt>` in conjunction with `--modules_to_save embed_tokens lm_head` to unfreeze the corresponding parameters for training. See [Example](https://github.com/modelscope/ms-swift/tree/main/examples/train/new_special_tokens) for details.

### Q18: Tuners Related Issues
- SWIFT's LlamaPro is adapted for multimodal training.
- LongLoRA depends on specific components in the architecture, so only LLaMA series models can use it.
- LoRA training is incompatible with the `--trainable_parameters` parameter; additional parameters besides the LoRA module need to be trained using `--modules_to_save`.

### Q19: Embedding/Reranker Training Related Issues
- [Embedding Training Example](https://github.com/modelscope/ms-swift/blob/main/examples/train/embedding).
- [Reranker Training Example](https://github.com/modelscope/ms-swift/tree/main/examples/train/reranker).
- See [Custom Dataset](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html) for embedding/reranker data format.

### Q20: Classification Training Related Issues
- Requires setting `--num_labels` and `--problem_type`. Detailed explanations can be found in the [Command Line Parameters Documentation](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html).
- For the multi-label classification data format, see [Custom Dataset](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html). <br>
Note: The label and message fields are at the same level in the dataset.

### Q21: Thinking Model Training
See this [issue](https://github.com/modelscope/ms-swift/issues/4030).

### Q22: Does SWIFT support distillation?
Refer to this [example](https://github.com/modelscope/ms-swift/blob/main/examples/sampler/distill/distill.sh).

### Q23: GKD Training Related Issues
- GKD training supports different model_types for the student model and teacher model, as long as the vocabulary is the same (using MoE will be slower).
- SWIFT v4 and later versions support different parallel configurations for the teacher model and student model. See the [example](https://github.com/modelscope/ms-swift/tree/main/examples/ray/gkd) for details.

### Q24: GRPO Training Related Issues
- SWIFT now supports multimodal GRPO training. During GRPO training, a loss close to 0 is normal; see [issue](https://github.com/huggingface/open-r1/issues/239#issuecomment-2646297851).
- To avoid introducing the KL term during GRPO training, you can set the KL regularization coefficient `--beta=0` to prevent loading the ref model.
- To continue GRPO training after LoRA fine-tuning, use `--adapters sft_ckpt --ref_adapters sft_ckpt`.
- Due to the additional overhead of calculating entropy, curves are not recorded by default. If needed, set `--log_entropy True`.
- Colocate mode does not support `--vllm_use_async_engine`.
- GRPO does not support channel loss.
- GRPO cannot use Liger kernel and padding-free simultaneously. Using them simultaneously requires modifying the Liger GRPO loss in the Liger kernel library.
- In the GRPO/PPO code implementation, mini_batch is only used for gradient accumulation. Activating the Clip mechanism requires num_iterations > 1. Setting num_iterations = 1 will cause it to fail.
- If the training set has different tasks, please refer to the [Multi-Task Training Documentation](https://swift.readthedocs.io/en/latest/Instruction/GRPO/DeveloperGuide/multi_task.html).
- For more GRPO-related FAQs, please refer to the [GRPO FAQ](https://swift.readthedocs.io/en/latest/Instruction/GRPO/GetStarted/GRPO.html#faq).

### Q25: Reward Function (Model) Related Questions
- `--reward_model` and `--reward_funcs` can be used together, ultimately resulting in a total reward through weighted summation. Weights can be specified using `--reward_weights`, with the weights in the order of reward_func1, reward_func2, ..., reward_funcn, reward_model.
- For custom reward functions, refer to [examples/train/grpo/plugin/plugin.py](https://github.com/modelscope/ms-swift/blob/main/examples/train/grpo/plugin/plugin.py).
- For math problems, the dataset must contain a solution field; otherwise, accuracy calculations will be affected.
- If your ORM's custom reward function requires a field from the dataset, place that field at the same level as messages. You can then retrieve that field from reward_kwargs.
- If you need to specify an LLM-judge model for scoring during GRPO training, please refer to the [Reward Model Documentation](https://swift.readthedocs.io/en/latest/Instruction/GRPO/DeveloperGuide/reward_model.html) to implement it.

### Q26: Rollout Related Issues
- Rollout is not compatible with Pipeline Parallel. For multi-GPU inference acceleration, use Tensor Parallel.
- The vLLM inference engine defaults to `trust_remote_code` being true.

### Q27: Does save_steps in the training script refer to the step or the global step?
It refers to the global_step, which is what the local TQDM displays.

### Q28: After passing `--importance_sampling_level sequence` to GSPO training, does it also support passing the parameter `--top_entropy_quantile`? That is, can it still optimize for the top x% of tokens in the entropy distribution?
Yes, it is supported. The order is to first calculate the sequence loss normally (affected by importance_sampling_level), and then mask the loss based on top_entropy_quantile.

### Q29: PPO and other preference training related issues
- PPO training does not support the `--max_grad_norm` parameter. If gradient explosion occurs, you need to tune parameters such as the learning rate and reward scale.
- Currently, PPO only supports models where the RM and policy are from the same series (tokenizer/template). Otherwise, it will lead to problems such as inconsistent prompt formats and inconsistent token sequence segmentation, affecting the performance.
- Currently, multi-turn DPO is not supported. You can use GRPO + multi-turn (multi-turn inference + reward function scoring) as an alternative.

### Q30: MoE model training related issues
In LoRA training, whether the router module participates in training depends on whether the gate/router is implemented as nn.Linear. If it is implemented as nn.Parameter, it will not participate in LoRA training. In this case, the aux-loss will remain basically unchanged. Under this premise, if you want to train the router, you need to add all-router to `--target_modules`.
```shell
--target_modules all-linear all-router
```
all-router is not a wildcard matching module name, but a special keyword that tells the framework to "include routers in the trainable scope." <br>
You can also specify specific LoRA replacement parameters using `--target_parameters`, see command-line parameters [--target_parameters](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html#tuner-arguments).

### Q31: Megatron-SWIFT Training Related Issues
- Checkpoint saves and searches for command line parameters [--save_strategy](https://swift.readthedocs.io/en/latest/Megatron-SWIFT/Command-line-parameters.html).
- When Megatron trains on a multi-machine pipeline in parallel, only the last rank holds the complete output, so logs are printed on the last rank, not from the master node.
- Megatron-SWIFT supports `--save_total_limit`; it also supports SwanLab monitoring of training. See the [Megatron-SWIFT command-line parameter documentation](https://swift.readthedocs.io/en/latest/Megatron-SWIFT/Command-line-parameters.html) for details.
- ViT uses a transformers model structure and currently does not use Megatron parallelism. When encountering an OutOfMemoryError (OOM) during training, the number of LLM decoder layers can be reduced using the `--decoder_first_pipeline_num_layers` parameter, freeing up more GPU memory for ViT to alleviate the issue.
- Megatron-SWIFT supports adding new models, but there are currently no tutorials available. Please refer to the PR for adding new models to understand the configuration method.
- Megatron-SWIFT's sequence parallelism is not set independently; the degree of parallelism is equal to the degree of tensor parallelism, i.e., set via `--tensor_parallel_size`.
- Block-wise FP8 is supported. See [examples/megatron/fp8 examples](https://github.com/modelscope/ms-swift/tree/main/examples/megatron/fp8).
- Resuming training with breaks requires configuring the following parameters.
```shell
--mcore_model <path/to/checkpoint-xxx>  # Load model weights
--finetune false                        # Mark as fine-tuning mode (instead of continuing training)
--no_load_optim                         # Do not load optimizer state (optional)
--no_load_rng                           # Do not restore random number state (optional)
```
For LoRA breakpoint resume training, you need to additionally set `--mcore_adapter`. Otherwise, it's the same as full parameter training. See [Megatron-SWIFT command line parameter documentation](https://swift.readthedocs.io/en/latest/Megatron-SWIFT/Command-line-parameters.html) for details.
- Megatron-SWIFT does not support QLoRA training.

### Q32: MTP Related Issues
- MTP training is required. Please manually set the command-line parameter `--mtp_num_layers`. Refer to `num_nextn_predict_layers` in config.json and fill in this value in the `mtp_num_hidden_layers` field.
- If the base model does not include an MTP structure, you can initialize and train the MTP from scratch.
- Multimodal MTP is not currently supported.

### Q33: Quantization Model Training Related Issues
- QLoRA fine-tuning reference [example](https://github.com/modelscope/ms-swift/tree/main/examples/train/qlora).
- Quantization methods such as GPTQ (int type) prevent parameters from participating in differentiation, thus full parameter fine-tuning is not possible. Only additional structures like LoRA can be attached for updates.
- Merging models trained with QLoRA reference [QLoRA example](https://github.com/modelscope/ms-swift/tree/main/examples/train/qlora).
- Megatron-SWIFT does not support QLoRA training.

### Q34: Training of Some Special Models
- SWIFT currently does not support training MiniCPM-O using audio modal input.
- Fine-tuning DeepSeek-VL-2 requires `transformers<4.42` and `peft==0.11.*`.
- Moonlight-16B-A3B-Instruct fine-tuning is hampered by training being disabled in the model file. Refer to the DeepSeek-VL-2 [solution](https://github.com/modelscope/ms-swift/issues/543) for a workaround.
- Ovis2 is a special model; fine-tuning requires padding to max_length, so `--max_length` must be explicitly set.
- Qwen2.5-Omni currently only supports thinker training and does not support talker training.
- Qwen2-Audio's SFT does not support packing.

### Q35: What is the default attention implemment on devices that do not support flash attention?
SDPA is used by default.

### Q36: Is left padding the default for model training?
Training can choose to use either left or right padding. The default is right padding, and batch inferring always uses left padding.

### Q37: Can SWIFT support setting a minimum learning rate? It seems like the final result is too small.
Yes, it can be set via:
```shell
--lr_scheduler_type cosine_with_min_lr
--lr_scheduler_kwargs '{"min_lr": 1e-6}'
```

### Q38: Is it possible to configure grpo and sft using a YAML file?
Yes, this configuration will be processed into a command line in main.py.

### Q39: Is it possible to use `--use_liger_kernel` and `--log_entropy` together?
No, it is not supported. liger does not instantiate logits, so it cannot obtain entropies.

### Q40: Encountering errors related to gradient_accumulation_fusion, even installing APEX doesn't resolve the issue.
```shell
RuntimeError: ColumnParallelLinear was called with gradient_accumulation_fusion set to True but the custom CUDA extension fused_weight_gradient_mlp_cuda module is not found. To use gradient_accumulation_fusion you must install APEX with --cpp_ext and --cuda_ext. For example: pip install --global-option="--cpp_ext" --global-option="--cuda_ext ." Note that the extension requires CUDA>=11. Otherwise, you must turn off gradient accumulation fusion.
```
Disable gradient accumulation fusion by using `--gradient_accumulation_fusion false`.

### Q41: When fine-tuning VLM for several tasks simultaneously, how to configure it when the video sampling rules for different tasks are inconsistent?
Search for `--interleave_prob` in the [Command Line Parameters documentation](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html).

### Q42: During multimodal packing pre-training, memory usage seems to increase slightly after each PyTorch allocator cache flushes since the last step, which can easily lead to OutOfMemoryError (OOM) with many steps.
Add the environment variable `PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'` to reduce memory fragmentation.

### Q43: Can `--use_logits_to_keep` be used on large multimodal models?
It works if multimodal token expansion occurs outside the model; it throws an error if it occurs inside the model's forward pass.

### Q44: Is there any practical documentation on fine-tuning a qwen base model to a chat model? Are there any special configurations required?
Use `swift sft`. There are no other special configurations required. Refer to the [example](https://github.com/modelscope/ms-swift/tree/main/examples/train/base_to_chat).

### Q45: What if the model receives many duplicate responses after training?
Please refer to [Pre-training and Fine-tuning](https://swift.readthedocs.io/en/latest/Instruction/Pre-training-and-Fine-tuning.html). If duplicates occur during training, consider training for several epochs, cleaning the data, performing full parameter training, or using RLHF to mitigate the issue.

### Q46: During full-parameter training, because the card cannot use bf16, I set `--torch_dtype float16`, and the following error occurred:
```shell
lib/python3.12/site-packages/torch/amp/grad_scaler.py", line 260, in _unscale_grads_ raise ValueError("Attempting to unscale FP16 gradients.") ValueError: Attempting to unscale FP16 gradients.
```
The value range of fp16 is very small (maximum 65504), and gradient overflow is easy during full-parameter training. You can try using `--torch_dtype fp32` instead.

### Q47: The following error occurred when merging LoRa parameters. Currently, Peft is version 0.11.0. Is this because the Peft version needs to be upgraded?
```shell
File "/opt/conda/lib/python3.9/site-packages/peft/config.py", line 118, in from_peft_type
  return config_cls(**kwargs)
TypeError: __init__() got an unexpected keyword argument 'corda_config'
```
This is caused by a mismatch between the Peft versions of the training and merging ends. The merging end needs to upgrade Peft to the same (or higher) version as the training end.

### Q48: safetensors_rust.SafetensorError: Error while deserializing header: HeaderTooLarge
Insufficient disk space; the model was not fully saved, and the weight data was truncated.

### Q49: AttributeError: module 'numpy' has no attribute 'object'
Try `numpy==1.26.3`.

### Q50: unsloth training, error: assert(type(target modules) in (list,tuple,)). The configured parameter is `--target modules all-linear`
Change `all-linear` to a specific module list, such as `--target_modules q k v`, and the unsloth LoRA implementation path will not expand the specific module name.

### Q51: For qwen2.5-omni, --freeze_vit false means that both the visual encoder and the audio encoder are turned on. Is there any way to turn on only the audio encoder but not the visual encoder?
Use `--target_regex` to match only the module paths you want to train. For example:
```shell
--target_regex ".*audio.*"    # Only match modules containing audio
```

## Inference

Swift supports inference via Python scripts, command line, and UI interfaces. For details, see [Inference and Deployment](https://swift.readthedocs.io/en/latest/Instruction/Inference-and-deployment.html).

### Q1: How to set up a model for SWIFT inference?
- For models trained with all parameters, models merged after LoRA training, or models downloaded from Model Hub, set the command-line argument `--model <model/id/or/path>`.
- For models trained with LoRA but not merged, specify the base model path with `--model <model/id/or/path>` and set `--adapters <path/to/adapter>`.

### Q2: How does SWIFT use datasets for inference? Where are the inference results stored?
- Specify the dataset using `--val_dataset <path/to/val_dataset>`. If you want to perform inference on the validation set split during training, set the argument `--load_data_args true`.
- Set the path to save the inference results using `--result_path <your/path>`. The path will be printed in the logs. See the documentation [Command Line Parameters Documentation](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html).
- To retain additional fields other than messages in the inference dataset, set `--remove_unused_columns false`.

### Q3: How to set up batch inference in SWIFT?
If infer_backend is transformers, set the command-line parameter `--max_batch_size 16`. Note that this parameter sets the batch size per card, not globally. Or refer to the [demo](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo.py).

### Q4: How to set up streaming inference in SWIFT?
Use `--stream true`. The inference results will be written to a JSONL file line by line. <br>
Note:
- Streaming inference does not support DDP.

### Q5: vLLM and SGLang Inference Backend Related Issues
- For whether LoRA-trained models need to be merged, please refer to the vLLM and SGLang documentation. If LoRA inference is supported, merging before inference is not necessary.
- SGLang inference currently does not support multimodal inference.

### Q6: Issues related to generating parameters
Parameters such as temperature are read from generation_config.json by default. Inference randomness can also be disabled by explicitly setting `--temperature 0` or `--top_k 1`.

### Q7: How to set system_prompt to empty? The command line does not set the system parameter, but it adds the default system.
Explicitly set `--system ''`.

### Q8: How to compute metrics like ACC/ROUGE during inference?
Use `--metric`. For specific details, search for this parameter in [Command Line Parameter Document](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html#id14).

### Q9: During model inference, which parameter should be set to continue generation from a specific prefix?
Use the `--response_prefix` parameter.

### Q10: The 'answer' in my data already contains part of the prompt. How should I modify the inference to complete the 'answer'?
```text
{"messages": [{"role": "system", "content": "<system>"}, {"role": "user", "content": "<query1>"}, {"role": "assistant", "content": "answer1, "}]}
```
This is supported in Swift versions 3.0 and later. Refer to [examples/infer/demo_agent](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo_agent.py).

### Q11: During multimodal model inference, how can I limit the maximum number of pixels to reduce GPU memory usage?
Set the command-line argument `--max_pixels xxx`, the environment variable `MAX_PIXELS=xxx`, or the specific model argument `--model_kwargs '{"max_pixels": xxx}'`. The environment variable only affects the models specified in the documentation. For details, see the documentation on [Specific Model Arguments](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html#id19).

### Q12: How to output the probability value logprobs parameter in SWIFT inference?
Command line inference setting: `--logprobs true`; Python script inference setting:
```shell
request_config = RequestConfig(..., logprobs=True, top_logprobs=2)
```
See [test_logprobs.py](https://github.com/modelscope/ms-swift/blob/main/tests/infer/test_logprobs.py) for details.

### Q13: How to output last_hidden_state in SWIFT inference?
No parameter is required. You can refer to the `_get_last_hidden_state` method of the GRPO trainer [here](https://github.com/modelscope/ms-swift/blob/main/swift/rlhf_trainers/grpo_trainer.py).

### Q14: Issues with inconsistent inference results between Transformers, vLLM, Ollama, etc.
Swift's templates are aligned with those of Transformers. Check if the inference parameters are consistent. Additionally, there are differences between VllmEngine and TransformersEngine.

### Q15: Inference for embedding/reranker models
- For embedding model inference, refer to the [example](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo_embedding.py) here. For - reranker model inference, refer to the [example](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo_reranker.py) here.

### Q16: When using a Python script for inference, how can I use the CPU?
Set the environment variable: `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'`.

### Q17: Does the swift infer command support multi-machine inference?
If the model can fit on a single node, you can orchestrate it using Kubernetes. If the model does not fit on a single node, multi-machine inference is not supported.

### Q18: Does Swift support batch sampling?
This [script](https://github.com/modelscope/ms-swift/blob/main/examples/train/rft/rft.py) allows for multi-process sampling of the dataset.

### Q19: Special Model Dependency Version Issues
- Qwen2-Audio inference results are corrupted; please use transformers 4.48.
- LoRA trained with transformers 4.55.2 cannot be loaded with versions lower than 4.52. See [issue#5440](https://github.com/modelscope/ms-swift/issues/5440) for details.
- Swift is compatible with different versions of qwen-vl-utils; switching this dependency version is not required when using qwen2.5-vl and qwen3-vl models.

### Q20: safetensors_rust.SafetensorError: Error while deserializing header:MetadataIncompleteBuffer
Model weights are corrupted.

### Q21: vLLM error message:
```shell
ValueError: the decoder prompt contains a(n) video item with length 16758, which exceeds the pre-allocated encoder cache size 16384. Please reduce the input size or increase the encoder cache size by setting --limit-mm-per-prompt at startup.
```
This is usually caused by an excessively long multimodal input, exceeding the pre-allocated encoder cache size of vLLM. The encoder cache size can be adjusted using `--limit_mm_per_prompt`; another possible solution is to pass the following in the Swift CLI:
```shell
--vllm_engine_kwargs '{"max_num_batched_tokens": 20000}'
```
This increases `max_num_batched_tokens`, indirectly affecting the encoder cache size allocation.

## Export

### Q1: AutoAWQ related errors
- If the inference doesn't involve the AWQ quantized model but you encounter AutoAWQ related errors, try uninstalling AutoAWQ before inference.
- For models that don't support AWQ quantization, try using GPTQ.

### Q2: When quantizing a model using SWIFT, the model may not fit on a single GPU.
Try setting `--device_map cpu`; or load the model across multiple GPUs and quantize on a single GPU.

### Q3: Using Swift export to perform GPTQ int4 quantization on a qwen2.5 72B model, with the default max model length of 32768, and the provided calibration dataset containing 128 samples, but an error occurred during quantization. The error log is as follows:
```shell
factorization could not be completed because the input is not positive-definite (the leading minor of order 18145 is not pisitive-definite)
```
This is due to the Hessian matrix being non-positive definite. Try using a different dataset.

### Q4: When exporting in Swift, can the custom template_type be permanently changed?
No, it won't be modified. Templates in Swift are defined internally by Swift and are not saved using Jinja.

### Q5: Can a trained model be directly converted to GGFU format?
Currently, only ModelFile is supported for export. See the [Command Line Parameters Documentation](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html#id17) for details on export parameters.

## Deployment

### Q1: How to set up the model for SWIFT deployment?
- For a model trained with full parameters, a model merged after LoRA training, or a model downloaded from model hub, set the command line parameter `--model <model/id/or/path>`.
- For unmerged models after LoRA training, `--modelmodel/id/or/path>` specifies the base model path and sets `--adapters <path/to/adapter>` at the same time.

### Q2: How does SWIFT deploy multiple cards?
See [Examples](https://github.com/modelscope/ms-swift/tree/main/examples/deploy) for details. If it is a transformers engine, it does not support DDP and cannot be deployed with multiple cards. In addition, heterogeneous deployment is not supported, such as different models of graphics cards, different storage ratios for each graphics card, etc.

### Q3: Can I select one operation by specifying the system prompt through the --system parameter and adding system prompt and template before each data in the data set? Do these methods have the same priority for the model?
System priority: The default in the data set>command line>template.

### Q4: Issues related to multi-modal input on the client side
- The client passes in images, audio, etc., see [Client Example](https://github.com/modelscope/ms-swift/tree/main/examples/deploy/client/mllm).
- If the image URL is illegal, you can set the request timeout by setting the environment variable `SWIFT_TIMEOUT` or passing parameters in `InferClient`.

### Q5: Issues related to generating parameter settings
- Inference generation parameters (temperature, etc.) can be set to default values when deployed, and can be dynamically overwritten with each client request;
- Engine/deployment parameters (number of TPs, memory ratio, maximum length) can only be set when the deployment is started and cannot be changed after running.

### Q6: How to set up streaming generation for models deployed by SWIFT?
It is controlled by the client. For details, please see [examples/deploy/client](https://github.com/modelscope/ms-swift/tree/main/examples/deploy/client).

### Q7: How does SWIFT deployment output the probability of token?
First, the server needs to set `--logprobs true`, and secondly, the client needs to pass the following parameters:
```shell
request_config = RequestConfig(..., logprobs=True, top_logprobs=2)
```

### Q8: Questions related to thinking
If you need to disable thinking, currently you can only disable thinking when swift deploy is started. You can check out this [issue](https://github.com/modelscope/ms-swift/issues/4030).

### Q9: How to output multiple results at one time?
Pass in the parameter `n` in `RequestConfig`, as shown below:
```shell
response = client.infer([request], request_config=RequestConfig(
    n=3,              # Generate 3 items
    temperature=0.8,  # Needs randomness to produce different results
))
# response contains 3 different answers
```

### Q10: There is a difference between specifying --infer_backend vllm and directly using vllm to deploy inference results.
- The inference results are quite different, possibly because the templates are not aligned.
- The inference speed varies greatly, possibly because the image resolution is inconsistent.
- SWIFT uses the V1 engine by default, and the switch can be controlled through the environment variable `VLLM_USE_V1=1`.

### Q11: Issues related to special models and dependent versions
- If you encounter an error without `model.language_model.embed_tokens.weight`, it may be caused by inconsistent versions of transformers in training and inference.
- If qwen2.5 uses fp16 inference and encounters garbled code returned, try bf16.

### Q12: Why can’t I use chat.completions but must use completions after the Qwen2-7B base model is deployed?
The base model has not been trained in conversation format, and it does not recognize chat special tokens such as <|im_start|>user<|im_end|>. The SWIFT framework has done the processing, and the base model can also use client.chat.completions.create, but this is a compatible behavior. In essence, messages are spelled into plain text for continuation.

## Evaluation

The eval capability of ms-swift uses the magic community evaluation framework EvalScope. For complex capabilities, please use the [EvalScope framework](https://evalscope.readthedocs.io/en/latest/get_started/introduction.html) directly.

### Q1: What evaluation datasets does Swift support? And how can I use a custom evaluation dataset?
For details on using the standard evaluation set and user-defined evaluation sets, please refer to the [Evaluation Documentation](https://swift.readthedocs.io/en/latest/Instruction/Evaluation.html).

### Q2: After manually downloading an officially supported evaluation dataset, can swift eval be configured to evaluate using a local path?
For offline evaluation, please refer to the EvalScope documentation's [Quick Start](https://evalscope.readthedocs.io/en/latest/get_started/basic_usage.html).

### Q3: The model, after fine-tuning with eval, always stops at a fixed percentage, but the VLLM service continues to run normally.
Client requests exceed the default timeout, and the connection is dropped. You can set the `SWIFT_TIMEOUT` environment variable to -1 to disable timeout-based disconnections.

### Q4: Can the number of data items in the dataset be controlled during evaluation?
The configuration parameter `--eval_limit` controls the number of data items per subset. For example, if MMLU has more than 50 subsets, and each subset has a limit of 10 data items, the total number of data items is over 500.

### Q5: The model generates a maximum of 1024 tokens before stopping. How can this be modified? Trying to set `--max_new_tokens` to 5000 doesn't work.
`--max_new_tokens` is an inference parameter, not an evaluation parameter. The generation length during evaluation is controlled by `--eval_generation_config`, which requires setting `max_new_tokens` within this parameter.
```shell
--eval_generation_config '{"max_new_tokens": 5000}'
```

### Q6: Doesn't `--eval_backend OpenCompass` support custom datasets? The error is reported as follows:
```shell
ValueError: eval_dataset: /mnt/workspace/data.jsonl is not supported.
eval_backend: OpenCompass supported datasets: ['C3', 'summedits', 'WiC', 'csl', 'lambada', 'mbpp', 'hellaswag', 'ARC_e', 'math', 'nq', 'race', 'MultiRC', 'cmb', 'ceval', 'GaokaoBench', 'mmlu', 'winogrande', 'tnews', 'triviaqa', 'CB', 'cluewsc', 'humaneval', 'AX_g', 'DRCD', 'RTE', 'ocnli_fc', 'gsm8k', 'obqa', 'ReCoRD', 'Xsum', 'ocnli', 'WSC', 'siqa', 'agieval', 'piqa', 'cmnli', 'cmmlu', 'eprstmt', 'storycloze', 'AX_b', 'afqmc', 'strategyqa', 'bustm', 'BoolQ', 'COPA', 'ARC_c', 'PMMEval', 'chid', 'CMRC', 'lcsts']
```
OpenCompass only supports its predefined standard evaluation sets and does not support custom datasets. Custom datasets can be defined using native methods.

### Q7: Evalscope can generate reports natively. Do other backends like OpenCompass support this as well?
Currently, only native visualization is supported. Other backends are not yet supported.

### Q8: Ifeval evaluation error:
```shell
[Errno 20] Not a directory: '/root/nltk_data/tokenizers/punkt_tab.zip/punkt_tab/english/collocations.tab'
```
You need to unzip `unzip /path/to/nltk_data/tokenizers/punkt_tab.zip`.

### Q9: How do I specify the offline dataset path for eval_backend='OpenCompass'?
See the [Data Preparation Tutorial](https://evalscope.readthedocs.io/en/latest/user_guides/backend/opencompass_backend.html#id3), download the dataset, and unzip it. No need to specify `dataset-args`. Simply place the dataset folder (i.e., the data folder) in the current working directory, and OpenCompass will automatically recognize it.

### Q10: Error:
```shell
unzip: cannot find or open /root/nltk_data/tokenizers/punkt_tab.zip, /root/nltk_data/tokenizers/punkt_tab.zip.zip or /root/nltk_data/tokenizers/punkt_tab.zip.ZIP
```
This indicates a failure to download nltk dependencies. Manually download [punkt_tab.zip](https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/open_data/nltk_data/punkt_tab.zip) and extract it to `~/nltk_data/tokenizers`.

### Q11: Can LLM be specified as the judge? How should the parameters be passed in?
Supported. Parameter passing is as follows:
```shell
--extra_eval_args '{"judge-model-args": {"api_key": "xxx", "api_url": "http://xxx/v1", "model_id": "qwen-72b"}}'
```

### Q12: When executing `eval`, uneven memory allocation across multiple GPUs occurred, with the following error:
```shell
NPROC_PER_NODE=8
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\ MAX_PIXELS=802816\ swift eval\
--model "$MODEL_PATH” \$EXTRA_ARGS \
--eval_backend Native \ --infer_backend transformers\ --device_map auto \
--eval_limit"$EVAL_LIMIT"\ --eval_dataset general_qa\
--dataset_args "{\"general_qa\": {\"local_path\": \"${DATA_PATH}\", \"subset_list\": [\"${SUBSET_NAME}\"]}}" \ --host 127.0.0.1\> "$LOG_FILE" 2>&1
```
swift eval does not support DDP startup.

### Q13: Where can I see what additional fields are included in the query besides the question during Swift evaluation?
The simplest way is to look at the input field in the output reviews file; it's the Markdown format of the content input to the model. <br>
If the backend is OpenCompass, these outputs won't be available, and you'll need to use a native backend.
