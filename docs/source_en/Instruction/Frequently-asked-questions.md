# Frequently-asked-questions

Here are some common questions encountered during the use of Swift.

## Training

### Q1: What models and datasets are supported for fine-tuning in Swift?
Please refer to the documentation on [Supported Models and Datasets](https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html).

### Q2: What data formats are supported when training with custom datasets?
For custom dataset formats, see the documentation on [Custom Dataset](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html).

### Q3: What is the format for dataset_info.json for custom datasets, and how can I use it?
The dataset_info.json format can be found in the documentation on [Custom Dataset](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html). Use the command line with `--custom_dataset_info xxx.json`, `--dataset <dataset_id_or_path>`.

### Q4: How can I train with a custom dataset using the interface?
Using a custom dataset through the interface is the same as using the command line. Refer to the documentation on [Custom Dataset](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html).

### Q5: Can I write a line in the jsonl file like this? {"index": "00000", "query": "11111", "response": "22222", 'source':'qqq'}
You can have additional fields that won't be used.

### Q6: Where can I find the command line parameters?
Please refer to the documentation on [Command Line Parameters](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html).

### Q7: What parameters need to be configured for training in an offline environment?
Use `--model local_path`, `--check_model false`. For more details, see the [Command Line Parameters](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html).

### Q8: Where can I check model_type?
Check the [Supported Models and Datasets](https://swift.readthedocs.io/en/latest/Instruction/Supported-models-and-datasets.html).

### Q9: Can I directly convert the model to gguf format after training?
Currently, only export to ModelFile is supported. See the [Command Line Parameters](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html).

### Q10: Does Swift support pre-training? I only see SFT.
Yes, it supports it. Use the command line `swift pt`, [pt example](https://github.com/modelscope/ms-swift/tree/main/examples/train/pretrain). The dataset format is detailed in [Custom Dataset](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html).

### Q11: For models fine-tuned with LoRA, should I merge them into one model for resuming training, or can I specify the original model and LoRA block by path directly?
You do not need to merge. Use `--resume_from_checkpoint output/xxx/vx-xxx/checkpoint-xxx`. See the [Command Line Parameters](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html).

### Q12: I would like to control the location where the original model weights downloaded from the internet are stored. How can I place the original model in a specific folder?
You can set the environment variable `MODELSCOPE_CACHE=your_path` to store the original model in the specified path. For SDK downloads, use `cache_dir="local_path"`. You can also use the `modelscope download` command-line tool or `git` to download it. For details, refer to the [Download Model](https://modelscope.cn/docs/Models/Download-Model). During training, set `--model` to the local path. For offline training, configure `--check_model false`. See the [Command Line Parameters](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html).

### Q13: Has anyone encountered this issue with ms-swift?
```text
[rank6]: pydantic_core._pydantic_core.ValidationError: 1 validation error for DeepSpeedZeroConfig
[rank6]: stage3_prefetch_bucket_size
[rank6]: Input should be a valid integer, got a number with a fractional part [type=int_from_float,input_value=11560550.4，in put_type=float]
[rank6]: For further information visit https://errors.pydantic.dev/2.8/v/int_fro_float
```
Downgrade `deepspeed` to `0.14.*`.

### Q14: Is there a complete tutorial and command line for fine-tuning Qwen-2-VL?
Reference the [example](https://github.com/modelscope/ms-swift/tree/main/examples/train/multimodal) for multimodal model training.

### Q15: Are there any tricks supported for fine-tuning multi-modal large models, similar to the LLM's neftune?
You can try variations of `lora` like `piassa/olora/dora`, or `fourierft`. Refer to the tricks in the `sft` parameters, as some may not apply to multi-modal.

### Q16: The accuracy from eval during training and the accuracy computed from re-inference with the saved checkpoint are not consistent.
The methods for calculating eval accuracy during training and inference are different. The default `acc_strategy` is `token`, and the selectable values are: `token`, `seq`.

### Q17: Official Magic Mirror image and Swift environment.
You can start the container using `docker run`, for example: `docker run --gpus all -p 8000:8000 -it -d --name ms modelscope-registry.cn-beijing.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.4.0-py311-torch2.6.0-1.26.0-LLM /bin/bash`. After starting the container, pull the latest code to install Swift. Additionally, for large model training scenarios, the `ms-swift` image is provided, which includes additional dependencies for `Megatron-SWIFT`, such as: `modelscope-registry.cn-beijing.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.4.0-py311-torch2.6.0-vllm0.8.5.post1-modelscope1.26.0-swift3.4.1.post1`. For more details, refer to the [Swift installation documentation](https://swift.readthedocs.io/en/latest/GetStarted/SWIFT-installation.html).

### Q18: Command line for multi-machine multi-card training.
For details, see the [Multi-node Example](https://github.com/modelscope/ms-swift/tree/main/examples/train/multi-node).

### Q19: How to choose a template?
See [issue](https://github.com/modelscope/ms-swift/issues/1813).

### Q20: How to use torchrun and swift sft for multi-card training?
`swift sft` uses `torchrun`.

### Q21: I have a question about my SFT dataset being too large; tokenizing takes a long time. Is there a solution?
Use `lazy_tokenize`or stream reading (`streaming`). See [Command Line Parameters documentation](https://swift.readthedocs.io/zh-cn/latest/Instruction/%E5%91%BD%E4%BB%A4%E8%A1%8C%E5%8F%82%E6%95%B0.html).

### Q22: When two datasets are simply appended together in the training set, does the model shuffle internally during training, or does it take data in order to train?
Command-line parameter `dataset_shuffle`. For more details, see the [command-line parameters documentation](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html).

### Q23: If the model is on two cards and the data is not parallelized, deepspeed will throw an error. How to handle this?
`deepspeed` and `device_map` are incompatible; you can only choose one.

### Q24: Why does it need to download again when retraining offline, despite having already downloaded the dataset online?
The data file contains URLs, which do not support offline training.

### Q25: How to reduce GPU memory usage when training VLM models?
Set `--freeze_vit true` and the parameter `--max_pixels` to limit the maximum pixels.

### Q26: Why are there fewer models supported in the WEB-UI interface than in the documentation?
Upgrade `ms-swift`.

### Q27: For models that do not have a suitable model_type, can I customize special_tokens and chat_template during SFT?
Yes, you can. Refer to the PR for model integration and the custom model dataset documentation.

### Q28: Can I use DPO to train Qwen2-VL in a Python script?
Yes, import `rlhf_main` and `RLHFArguments` from `swift.llm`.

### Q29: Can I pre-train with pure text before fine-tuning on a VQA dataset for MLLM?
Yes, you can mix training as well.

### Q30: When conducting DPO training based on the qwen2 SFT model on a V100 machine, the training shows NaN?
Use fp32 for training with the V100 machine.

### Q31: Does Swift support distillation?
Refer to this [example](https://github.com/modelscope/ms-swift/blob/main/examples/sampler/distill/distill.sh).

### Q32: The default maximum number of checkpoints saved after training is two. How can I increase this number?
Use `--save_total_limit`. See the [Command Line Parameters](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html).

### Q33: In grounding tasks, does the universal data format support multiple instances for one category?
Currently, it supports one object corresponding to multiple bounding boxes. Refer to the documentation on [Custom Dataset](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html#grounding).

### Q34: Why am I getting the error that numpy.object cannot be found?
Try using `numpy==1.26.3`.

### Q35: Does the Swift framework support sequence parallelism now?
Yes, it supports it. Refer to the [example](https://github.com/modelscope/ms-swift/tree/main/examples/train/long_text) here.

### Q36: When fine-tuning qwen2-1.5B on a V100, I see `loss': 0.0, 'acc': 0.0, 'grad_norm': nan`. What is the issue?
Try using fp32.

### Q37: Is it possible to fully fine-tune GPTQ quantized models?
No, GPTQ model's int-type parameters cannot participate in gradients; they can only be updated with additional structures like LoRA.

### Q38: What parameters should I set for fine-tuning using QLoRA on glm4-chat?
Refer to the QLoRA [example](https://github.com/modelscope/ms-swift/tree/main/examples/train/qlora).

### Q39: How do I expand my vocabulary within the Swift framework?
Swift currently does not support vocabulary expansion.

### Q40: Can I directly use models with the same name from Hugging Face?
Set the environment variable `USE_HF=1`.

### Q41: Can Qwen2-VL-2B conduct incremental pre-training? Is there guidance available?
Yes, it supports incremental pre-training. Just include all the content in the response.

### Q42: When training with videos, how can I control the frame sampling rate in the parameters? The `frame_rate` setting doesn't seem to work, and I'm using MiniCPMV.
Set the environment variable `MAX_NUM_FRAMES`.

### Q43: Can I save the inference results of the validation set during training in Swift?
After training, run `swift infer` to save the results.

### Q44: Why is the saved checkpoint larger than the original model file after full parameter DPO?
Using V100 for fine-tuning stores the data in fp32 format.

### Q45: Training speed slows down when using multi-machine training; using Swift framework for LLM training with deepspeed zero3 causes significant performance drop.
See the [issue](https://github.com/modelscope/ms-swift/issues/1825).

### Q46: Does Swift now support multi-stage pre-training for qwen2-vl? It looks like the official best practices only show SFT training with vit+llm together, not sure if separate fine-tuning is supported.
You can control this using the parameters `--freeze_vit`, `--freeze_aligner`, and `--freeze_llm`. For more details, see the [Command Line Parameters Documentation](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html#tuner-arguments).

### Q47: Does qwen2-vl support mixing pure text data?
It supports both mixed visual-text and pure text data.

### Q48: Can I plot loss curves for different datasets during fine-tuning?
Channel loss is supported. Please refer to this [example](https://github.com/modelscope/ms-swift/blob/main/examples/train/plugins/channel_loss.sh).

### Q49: After model training, the responses have a lot of repeated content.
Refer to the [Pre-training and Fine-tuning](https://swift.readthedocs.io/en/latest/Instruction/Pre-training-and-Fine-tuning.html). If you notice repetitions during training, try training for more epochs, cleaning the data, and conducting full parameter training, using RLHF to mitigate this issue.

### Q50: Does Swift currently support prompt tuning or prefix tuning?
No, it does not support these methods, as both methods suffer from serious forgetting issues and are not currently recommended.

### Q51: I encountered the following error when training with two A10s:
```text
[rank0]: torch.distributed.DistBackendError: NCCL error in:../torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:1970， unhandled system error (run with NCCL_DEBUG=INFO for details),NCCL version 2.20.5
[rank0]:ncclSystemError: System call (e.g. socket,malloc) or external library call failed or device error.
```
Please check if shared memory is too small; NCCL requires shared memory.

### Q52: How to solve the issue of certain parameters not participating in backpropagation when freezing layers during DDP fine-tuning?
Set the parameter `--ddp_find_unused_parameters true`.

### Q53: Does Swift have a dataset quality inspection tool?
[data-juicer](https://github.com/modelscope/data-juicer).

### Q54: Where to start model parallelism on the web? I only found the option to check for data parallelism.
You can specify visible GPUs to enable model parallelism.

### Q55: How can I set a fixed location for dataset downloads when using --dataset? I can't find this in command line parameters. How can I read from the download location next time?
`dataset_path` supports folders, typically for datasets downloaded via `git clone`. See [Custom Dataset Documentation](https://swift.readthedocs.io/en/latest/Customization/Custom-dataset.html#dataset-info-json).

### Q56: When using --streaming true, I get an error asking me to set max_steps when setting num_train_epochs. Can't I just set num_train_epochs?
See the streaming parameter description, [Command Line Parameters Documentation](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html#data-arguments).

### Q57: Why is tools in "[]" format rather than directly using []? Could you explain why tools uses this "[]" format instead of direct [] notation?
This is because the underlying pyarrow in datasets has strict type control. For the same reason, the objects part in our official grounding dataset also uses str, otherwise pyarrow would report errors about inconsistent types across rows.

### Q58: Can't this parameter be used? check_dataset_strategy==discard
This parameter no longer exists in swift3.0, use the `strict` parameter instead.

### Q59: Getting this error when running sft command:
```text
RuntimeError: Expected to mark a variable ready only once.This error is caused by one of the following reasons: 1) Use of a module parameter outsid forward function. Please make sure model parameters are not shared across multiple concurrent forward-backward passes. or try to use _set_static_graph( ) as round if this module graph does not change during training loop.2) Reused parameters in multiple reentrant backward passes. For example, if you use multiple oint` functions to wrap the same part of your model, it would result in the same set of parameters been used by different reentrant backward passes multiple and hence marking a variable ready multiple times. DDP does not support such use cases in default. You can try to use _set_static_graph( ) as a workaround if dule graph does not change over iterations.
```
Add this parameter: `--gradient_checkpointing_kwargs '{"use_reentrant": false}'`.

### Q60: Have you encountered this issue? AttributeError:'TrainerState' object has no attribute 'last_model_checkpoint'
Dataset is too small, need to add more data. Error occurs when data quantity is less than one step.

### Q61: I see preprocess can be defined in CustomPreprocessor. Is this processed all at once before training starts, or loaded during training?
If `--streaming true` is set, it loads while training. By default, it processes everything before training.

### Q62: For full-parameter training of internvl2_5, why do vision_model and mlp1 appear in freeze parameters by default? Documentation shows freeze_parameters defaults to [], and command line settings for freeze vit, freeze aligner, freeze llm are all False. It prints trainable parameters: ['mlp1'] - unclear if only mlp1 is trainable or all parameters
First freeze parameters then active parameters. The three parameters `freeze vit/freeze aligner/freeze llm` adjust freeze parameters and trainable parameters. Since some models' `vit` contains `aligner`, aligner is separately added to trainable_parameters.

### Q63: Does LlamaPro in swift support multimodal adaptation?
Yes, it's supported.

### Q64: I noticed 2.x supports MAX_PIXELS. Is the --max_pixel parameter in 3.x documentation the same thing? What's the processing logic? Using 12000*9000 images with internvl still crashes in 2.x even with resacle_image
Environment variable parameters correspond to model parameters. `MAX_PIXELS` only supports qwen2vl, internvl has its own environment variables. See [Specific Model Parameters](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html#specific-model-argumen).

### Q65: Is there documentation for fine-tuning qwen base model to chat model? Any special configurations needed?
Use `swift sft`, no special configuration needed. See [example](https://github.com/modelscope/ms-swift/tree/main/examples/train/base_to_chat).

### Q66:  Where can I find sequence parallel examples?
See this example: [sequence_parallel](https://github.com/modelscope/ms-swift/tree/main/examples/train/long_text).

### Q67: Can swift support training custom model structures?
Yes, just customize the `get_model_tokenizer_xxx` function to return `model` and `tokenizer`.

### Q68: Getting an error using longlora with "name_or_path": "/mnt/workspace/model/Qwen2.5-14B-Instruct". Is longlora only for llama series?
Yes, `longlora` only works with llama series.

### Q69: How to add custom special tokens in swift?
Add them in the `get_model_tokenizer` function.

### Q70: For --freeze_parameters_ratio parameter, if set to 0.7, does it mean only 30% of llm parameters are updated during training? Is it random 30%? What's the update mechanism?
Freezes from bottom to top.

### Q71: Why is the map process so slow? Is this normal?
```text
Map: 4%|██ | 9000/203823 [02:18<50:34, 64.19 examples/s]
```
Use `--dataset_num_proc` parameter to enable multiple processes.

### Q72: How can I delete and redownload a dataset? I think there might be an issue with the dataset.
Set the `--download_mode` parameter.

### Q73: How to solve this error: safetensors_rust.SafetensorError: Error while deserializing header: HeaderTooLarge?
The disk space is insufficient, and the model wasn't saved completely.

### Q74: Does swift3.0 not support get_default_template_type?
Please check `model.model_meta.template`, the information is available in `model.model_meta` and `model.model_info`.

### Q75: Does ModelScope Swift support hermes format agent fine-tuning? I see qwen2.5 uses vllm with native support for hermes format tool calling, why don't I see it in Swift?
Currently, `hermes` format is not supported. We mainly support `toolbench` and `react` formats, as `react` is more widely used. Swift's deploy currently supports parsing these two formats and provides `openai tool calling`.

### Q76: Is the default model training using left padding?
Training can use either left or right padding. The default is right padding, while `batch infer` uses left padding.

### Q77: Does it support grounding tasks now?
Yes, there's an [example](https://github.com/modelscope/ms-swift/blob/main/examples/train/multimodal/grounding.sh) under examples.

### Q78: Does ms-swift support contrastive learning for training llm_emb?
Yes, here's an [example](https://github.com/modelscope/ms-swift/blob/main/examples/train/embedding).

### Q79: Is there a big difference in performance between manually coding fine-tuning and GRPO using peft and trl libraries compared to Swift official training with the same parameters?
The difference is minimal, with Swift additionally supporting multimodality.

### Q80: Does Swift currently not support audio modal input training for minicpmo2_6? It shows error: assert media_type in {'image', 'video'}
Audio is not currently supported.

### Q81: Can Swift fine-tune deepseek R1 671B?
Yes, the template is integrated, but the process is complicated as it requires converting fp8 to bf16 first.

### Q82: Isn't the latest Swift framework supposed to specify the model location using this command? This is the location of the model I've already downloaded, but I don't know why it still tries to download and fails with a git clone error
```shell
--model /mnt/workspace/.cache/modelscope/hub/deepseek-ai/deepseek-vl2/ \
```
Some models require cloning the repo and then specifying through `local_repo_path`.

### Q83: Does Swift now support multimodal GRPO?
Yes, it does.

### Q84: Can the GRPO reward function be customized?
Yes, refer to [examples/train/grpo/plugin](https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo/plugin).

### Q85: Why do I get the error when using --torch_dtype float16 (card cannot use bf16): lib/python3.12/site-packages/torch/amp/grad_scaler.py", line 260, in unscale_grads raise ValueError("Attempting to unscale FP16 gradients.") ValueError: Attempting to unscale FP16 gradients.
FP16 does not support full-parameter training.

### Q86: I have a question. I trained a reward model using Swift (baseline is qwen2.5-7b), but when loading it in PPO or GRPO, it shows an error. The reward model was trained using LoRA.
```shell
--rlhf_type ppo \
--model Qwen/Qwen2.5-14B-Instruct \
--reward_model /mnt/workspace/output/rm/model --train_type lora \
--dataset 'AI-ModelScope/alpaca-gpt4-data-zh#20000' --torch_dtype float32 --num_train_epochs 1 \
--per_device_train_batch_size 1 --per_device_eval_batch_size 1 --learning_rate 1e-5 --lora_rank 8 --lora_alpha 32 \
--target_modules all-linear \
--gradient_accumulation_steps 16 --eval_steps 100 --save_steps 100 \
```
The LoRA-trained reward model needs to be merged.

### Q87: What version of transformers is needed to fine-tune deepseek_vl2? Official docs say <4.42, but it also shows errors with 4.42 and below. Does the peft version need to be lowered too?
Use `peft==0.11.*`.

### Q88: Generate train split is too slow (about 30+ datasets with around a million total data points). Previously Swift 2.x wasn't this slow. Lazy tokenize is already enabled.
Set `--dataset_num_proc 16`.

### Q89: How can I full-parameter fine-tune the visual encoder while using LoRA to fine-tune LLM when fine-tuning qwen2.5vl?
Refer to this [example](https://github.com/modelscope/ms-swift/tree/main/examples/train/multimodal/lora_llm_full_vit).

### Q90: How to use custom loss functions in Swift?
Add it in the plugin.

### Q91: What are the parameters for MoE? Can't find keywords in the parameter table. How to set expert numbers and expert routing parameters?
Use parameters directly from `config.json`.

### Q92: Using lmdeploy in grpo training reports missing functions. The load_weights function isn't found in lmdeployengine class.
Only supported under turbomind engine.

### Q93: Getting errors when fine-tuning Moonlight-16B-A3B-Instruct model. Seems ms-swift doesn't support fine-tuning this model?
Training is disabled in model files. Refer to deepseek_vl2's solution in the issues.

### Q94: How to solve this error: RuntimeError: "triu_tril_cuda_template" not implemented for 'BFloat16'?
```shell
CUDA_VISIBLE_DEVICES=01,2,3,4,5,6,7 \
swift sft \
    --model Internlm3-8b \
    --dataset train.json \
    --train_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --deepspeed zero3 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4
```
Upgrade torch.

### Q95: Is it normal that both loss and grad_norm are 0 during GRPO training?
```text
{'loss':    0.0.    'grad norm':0.0,    'learning_rate':9e-08,    'memory(GiB)':88.1，    'train_speed(iter/s)':0.009252，    'completion_length':    150.00000763，    'response_clip ratio': 0.0,    'rewards/Format':1.0,    'reward
: 1.0,    'reward std':0.0，    'kl': 0.0, 'clip_ratio': 0.0,    'epoch': 0.0， 'qlobal step/max steps':'1/1052'，    'percentage':'0.10%    'elapsed time':    '36s    'remaining time': '10h 43m 54s'}
{'loss': 0.0，'grad_norm':0.0，'learning_rate': 1.8e-07,'memory(GiB)':94.15，'train_speed(iter/s)':0.014782，'completion_length': 133.25000763，'response_clip_ratio': 0.0，'rewards/Format': 1.0, 'rewa rd': 1.0，'reward_std': 0.0, 'kl': 0.0，'clip_ratio': 0.0,'epoch': 0.0, 'global_step/max_steps': '2/1052'，'percentage': '0.19%', 'elapsed_time': '1m 3s'， 'remaining_time': '9h 19m 49s'}
{'loss': 0.0， 'qrad norm': 0.0, 'learning rate': 2.7e-07,'memory(GiB)': 94.15，'train_speed(iter/s)': 0.018695，'completion_length': 123.08333969，，'response_clip_ratio': 0.0，'rewards/Format': 1.0, 'rewa rd': 1.0， 'reward_ std': 0.0,'kl': 0.0,'clip_ratio': 0.0， 'epoch': 0.0， 'global_step/max_steps': '3/1052'，'percentage': '0.29%，'elapsed_time': '1m 29s'，'remaining_time': '8h 39m 34s'}
```
Training with loss close to 0 is normal, refer to this [issue](https://github.com/huggingface/open-r1/issues/239#issuecomment-2646297851).

### Q96: Where can I pass in accuracy_orm for GRPO's built-in reward function?
Currently it requires modifying the code directly.

### Q97: I notice the reward function has a solution parameter, does it need to be passed from the dataset? Does my dataset must have a solution field?
Yes, it's necessary for math problems to calculate accuracy.

### Q98: Why is there no token_acc during training?
Some models have mismatched `logits` and `labels` counts, so token accuracy isn't calculated.

### Q99: When fine-tuning Ovis2, LoRA parameters don't seem to work? Memory usage doesn't change with or without --train_type lora.
Limit `--max_length`, this model is special and needs padding to max_length.

### Q100: Getting ValueError when running classification task with Qwen2.5: The model did not return a loss from the inputs, only the following keys: logits. For reference, the inputs it received are input_ids,attention_mask.
dataset format: {"messages": [{"role": "user", "content": "xxxxx"}, {"label": 1}]}
Put `label` at the same level as `messages`, not inside it.

### Q101: How to exit VllmEngine? I want to release GPU memory after inference rather than keeping it occupied.
Use sleep mode: `engine.sleep(level=1)/engine.wake_up()` with `enable_sleep_mode=True` during initialization.

### Q102: Does trainer_sampler_random have no effect in streaming mode?
Streaming is not random.

### Q103: Can trust_remote_code be set when using VLLM for GRPO training?
It's true by default.

### Q104: For large dataset pretraining using streaming and packing, is there a way to calculate total steps based on epochs, batch size etc when setting max_steps?
Set `--max_steps` or `--max_epochs`. For more details, see the streaming parameter description in the [Command Line Parameters Documentation](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html#data-arguments).

### Q105: Unsloth training error: "assert(type(target modules) in (list,tuple,))" when using --target_modules all-linear
Don't use `all-linear`, specify concrete module list like `--target_modules q k v`.

### Q106: Does Swift support multi-label classification now?
Yes. Check custom dataset docs for format and search for `problem_type` in command line parameter docs.

### Q107: How does flash_attn handle packing - separately or merged?
Flash attention is required to avoid errors, otherwise attention_mask will have issues.

### Q108: For qwen2.5-omni, does setting --freeze_vit false mean both the visual encoder and the audio encoder are enabled? Is there a way to enable only the audio encoder without enabling the visual encoder?
Use `--target_regex`.

### Q109: Does swift currently support sequence parallelism for those reinforcement learning training methods?
It supports pt, sft, dpo, and grpo.

### Q110: After using lora sft, is tokenizer.json not saved?
Lora doesn't save it; it is migrated after merging because the lora directory needs to work with the original model.

### Q111: Can the reward_model and reward_funcs of GRPO be used together?
Yes, they can.

### Q112: I want to ask if there is a parameter that can be adjusted to avoid introducing the KL term in GRPO?
Search for `beta` in the command line parameters.

### Q113: When doing GRPO, how can I access the original labels in the orm custom reward function? I printed the messages field in kwargs, and the value of assistant's content in each item is replaced by the generated result.
Place it in another column.

### Q114: If you use the default num_iterations=1, does clip become ineffective? The clip higher in dapo is also useless. I see that veRL has a micro batch setting to update the policy model in small batches for the clip term to take effect. In ms-swift, it seems mini batch only does gradient accumulation according to the source code?
Yes, num_iterations needs to be >1.

### Q115: Does qwen2.5-omni training support full parameter training, and does it support talker training?
Currently, it does not support talker training, only thinker.

### Q116: Can sequence parallel be enabled at the same time as the liger kernel?
Yes, it can.

### Q117: What are the requirements for rm and policy in ppo training?
PPO currently only supports rm and policy being from the same model series (tokenizer/template).

### Q118: I want to use the 3.2 1B model for fine-tuning because llama3.1 doesn't have models smaller than 8B. Can I still use the Llama-3.1 reward model?
The requirement is that template and tokenizer must be the same, so 3.1 and 3.2 should be fine.

### Q119: Can swift cache a mapped version of data for troubleshooting training data issues?
Set `--load_from_cache_file false`.

### Q120: Why is there a warning: none of the inputs have requires_grad=True during full parameter training?
If vit is not being trained, getting this warning is normal; if it is being trained, then it should not occur.

### Q121: Does qwen2.5vl ulysses currently support sdpa?
The vl model currently only supports flash-attn, but both are supported for pure text.

### Q122: Is the image list format for videos now supported? The format is as follows:
```json
{"messages": [{"role": "assistant", "content": "<video>是一只狮子在跑步"}], "videos": [["1.jpg","2.jpg"]]}
```
It is supported, using the file directory method.

### Q123: In the grpo script, does save_steps refer to step or global step? The local training shows a global step of 18, while wandb shows a step of 628.
It refers to global_step, as shown by local tqdm.

### Q124: Can use_logits_to_keep be used on large multimodal models now?
If the expansion of multimodal tokens occurs within the model's forward, it will cause an error.

### Q125: Why does memory increase significantly multiple times during training, even after 50 or 100 steps?
Set the environment variable `PYTORCH_CUDA_ALLOC_CONF`, and check the torch documentation for details.

### Q126: With the packing_cache parameter set, I am encountering errors when training on multiple machines, even after setting the folder path. Are there any special requirements?
The path must be set to a shared disk directory.

### Q127: For Qwen3, are there differences in datasets and parameter settings between non-thinking and thinking modes?
Check this [issue](https://github.com/modelscope/ms-swift/issues/4030).

## Inference

### Q1: Is there documentation for Swift inference?
Swift supports inference via Python scripts, command line, and UI interface. See the [Inference and Deployment](https://swift.readthedocs.io/en/latest/Instruction/Inference-and-deployment.html).

### Q2: How to use the trained model for inference with a dataset?
Use the parameters `--load_data_args true` or `--val_dataset <your-val-dataset>`. Refer to the [Command Line Parameters](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html).

### Q3: Can I specify a locally saved model during Swift inference?
Set `--model` to the local path. See [Command Line Parameters](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html).

### Q4: How do I infer on a dataset without labels? I see that the dataset format in the documentation is all for the training set.
Configure the parameter `--val_dataset <your-val-dataset>`.

### Q5: How to resolve the error `ValueError: Input length of input_ids is 35, but max_length is set to 20`?
```text
raise ValueError(
ValueError: Input length of input_ids is 35, but `max_length` is set to 20. This can lead to unexpected behavior. You should consider increasing `max_length` or, better yet, setting `max_new_tokens`.
```
Set `model.generation_config.max_new_tokens`.

### Q6: qwen2-vl inference (training) runs out of memory
Set the command line parameter `--max_pixels xxx`, environment variable `MAX_PIXELS=xxx`, or specific model parameter `--model_kwargs '{"max_pixels": xxx}'`. Note that the environment variable only takes effect for the corresponding models in the documentation. For more details, please refer to the documentation [Specific Model Parameters](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html#specific-model-arguments).

### Q7: On a V100 GPU, in a Python virtual environment, following the environment setup instructions from https://swift2x.readthedocs.io/zh-cn/latest/Multi-Modal/qwen2-vl%E6%9C%80%E4%BD%B3%E5%AE%9E%E8%B7%B5.html, when testing the inference command: `CUDA_VISIBLE_DEVICES=0,1,2,3 swift infer --model_type qwen2-vl-7b-instruct`, an error occurs: `RuntimeError: probability tensor contains either 'inf', 'nan' or element < 0`.
Try inference on A10 or 3090 machines.

### Q8: After running the prediction command, where are the results saved? CUDA_VISIBLE_DEVICES=0 swift infer --ckpt_dir output/glm4v-9b-chat/vx-xxx/checkpoint-xxx-merged --load_data_args true
Results will be printed in the log.

### Q9: For the latest version of swift, can the infer command output probability values through the logprobs parameter?
Yes, logprobs can be output. For command line inference, set `--logprobs true`. For Python script inference, set `request_config = RequestConfig(..., logprobs=True, top_logprobs=2)`. Please refer to [test_logprobs.py](https://github.com/modelscope/ms-swift/blob/main/tests/infer/test_logprobs.py).

### Q10: In the latest version of Swift, while loading the qwen2-32b-instruct-awq quantized model, I was advised to add merge-lora true. After doing this, it throws an error. When I omit it, inference works but slowly.
Models trained with QLoRA do not support merge-lora; it is recommended to merge-lora after fine-tuning and then quantize.

### Q11: Getting the error `assert factor in rope_scaling` with vllm?
For more details, see qwen2-vl [issue#96](https://github.com/QwenLM/Qwen2.5-VL/issues/96).

### Q12: Does vllm require the models to be merged before calling them during inference?
Models do not have to be merged. See the documentation on [Command Line Parameters](https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html).

Q13: How to use CPU when performing inference with Python scripts?
Set the environment variable: `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'`.

### Q14: Has anyone encountered the error `RuntimeError: "triu_tril_cuda_template" not implemented for'BFloat16'`?
Upgrade Torch, as the current version may not have implemented this operator.

### Q15: Does qwen2-audio support streaming inference?
Yes, see the [issue](https://github.com/modelscope/ms-swift/issues/1653).

### Q16: Where to set `do_sample` for multi-modal inference using inference client?
Set `temperature=0`.

### Q17: Does ms-swift support batch processing for large models?
Supported.  See the [demo](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo.py).

### Q18: When quantizing models with ms-swift, there is an insufficient memory display. Can we reduce resource usage during quantization, even if it's slower?
Try setting `--device_map cpu`.

### Q19: Does Swift support quantization for multi-modal models?
Yes, it supports quantization.

### Q20: Encountering the following error while using GPTQ, what is the cause?
```text
if llm_config['architectures'][0] == 'LlamaForCausalLM':
KeyError: 'architectures'
```
Try using `transformers==4.44.*`.

### Q21: How can I specify where to save evaluation results during swift infer? I can't find where the results are saved.
Set `--result_path your_path`. See [InferArguments](https://github.com/modelscope/ms-swift/blob/main/swift/llm/argument/infer_args.py).

### Q22: I get an error while using AWQ quantized yi-vl-6b:
```text
TypeError: swift.llm.utils.model.get_model_tokenizer_with_flash_attn() got multiple values for keyword argument 'automodel_class'.
```
Please use GPTQ quantization.

### Q23: I would like to ask about using swift export to perform GPTQ INT4 quantization on the qwen2.5 72B model with a max model length of 32768, which is the default value. The calibration dataset provided has 128 samples, but an error occurred during quantization. The error log is: "factorization could not be completed because the input is not positive-definite (the leading minor of order 18145 is not positive-definite)." What is the cause?
This indicates a problem with the Hessian matrix being non-positive definite. Try using a different dataset.

### Q24: Can batch inference only be done through custom code? Can't it be done like SFT with script parameters?
Yes, it can be done using `swift infer --val_dataset xxx --max_batch_size 16 ...`.

### Q25: What's the default temperature value when using swift app for inference?
It's read from `generation_config.json` by default.

### Q26: Can export and quantization be done using multiple GPUs?
Model loading can use multiple GPUs, but quantization is single-GPU only.

### Q27: When using swift export with a custom template_type, does it permanently change the template_type? If we use swift export --template_type custom, does it change the model's template?
No, it won't be modified. Templates in swift are defined internally, not saved in jinja format.

### Q28: AWQ quantization for Qwen2VL gives error: TypeError: Qwen2VLForConditionalGeneration.init() got an unexpected keyword argument 'use_cache'
Use `gptq` quantization instead.

### Q29: For DDP inference, does max_batch_size in infer refer to batch size per GPU or total batch size?
It refers to batch size per GPU.

### Q30: Does swift.inference now support messages format input? It seems to only support query format currently. The answer contains part of the prompt, how should I modify the inference to complete the answer?
```text
{"messages": [{"role": "system", "content": "<system>"}, {"role": "user", "content": "<query1>"}, {"role": "assistant", "content": "answer1, "}]}
```
This is supported in swift3, refer to [examples/infer/demo_agent](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo_agent.py).

### Q31: How can I make swift infer write results to result_path in real-time instead of writing everything at once at the end?
```shell
swift infer \
--ckpt_dir model_dir \
--streaming true \
--val_dataset dataset.jsonl \
--result_path result.jsonl
```
Use `--stream true`. This will write results one by one, but it's non-batch inference.

### Q32: When I trained and did inference in Swift it worked, but after merge_lora when using Ollama's API the effect disappeared.
Try loading with transformers, Swift's template is aligned with transformers.

### Q33: Which parameter should I set if I need to continue inference under a specific prefix during model inference?
The parameter `--response_prefix`.

### Q34: How do I fix this error that keeps appearing?
```text
File "/mnt/workspace/swift/swift/1lm/dataset/preprocessor/core. py", line 69, in _check_messages raise
ValueError(f'assistant_message; {assistant_message}')
ValueError: assistant_message: {'role' :'assistant', 'content': ''}
```
```shell
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 MAX_PIXELS=1003520 swift sft --model Qwen/Qwen2.5-VL-7B-Instruct --train_type lora --dataset /mnt/workspace/data.json --deepspeed zero2 --max_length 16384
```
The assistant field in the dataset is empty. If this is for inference, delete this empty string because it will cause NaN during training and will be checked.

### Q35: Inference error, ImportError: cannot import name 'shard_checkpoint' from 'transformers.modeling_utils' (/usr/local/lib/python3.10/dist-packages/transformers/modeling_utils.py)
Try uninstalling autoawq.

### Q36: When using swift sample, it seems that batch processing is not supported? It appears to sample examples one by one in a loop, which is somewhat slow.
There is a [script](https://github.com/modelscope/ms-swift/blob/main/examples/train/rft/rft.py) that can use multiprocessing to split and sample the dataset.

### Q37: Does swift support inference of embedding models? The following error occurred:
```text
[rank0]:[W511 17:18:01.815062493ProcessGroupNCCL.cpp:1250]Warning: WARNING: process group has NOT been destroyed before we destruct ProcessGroupNCCL. On normal program exit, the application should call destroy_process_group to ensure that any pending NCCL operations have finished in this process. In rare cases this process can exit before this point and block the progress of another member of the process group. This constraint has always been present, but this warning has only been added since PyTorch 2.4 (function operator( ))
```
For embedding model inference, please use the official model code, as swift does not yet support it.

### Q38: Does the swift framework support model or tensor parallelism for inference? There is no OOM during training, but OOM occurs during inference.
```shell
CUDA_VISIBLE_DEVICES=0,1 \
MAX_PIXELS=1003520 \
swift infer \
    --adapters /path/to/checkpoint-xxx \
    --merge_lora true \
    --infer_backend vllm \
    --load_data_args true \
    --gpu_memory_utilization 0.9 \
    --tensor_parallel_size 2 \
    --max_model_len 32768 \
    --max_new_tokens 15536 \
    --limit_mm_per_prompt '{"image": 8, "video": 2}'
```
```text
Failed: Cuda error /workspace/csrc/custom_all_reduce.cuh:368 'invalid argument'
```
Add the option `--disable_custom_all_reduce true`.

### Q39: Does streaming inference support DDP?
Streaming does not support DDP.

## Deployment

### Q1: How to deploy a trained model?
Use `swift deploy --adapters xxx`. Refer to the documentation on [Inference and Deployment](https://swift.readthedocs.io/en/latest/Instruction/Inference-and-deployment.html).

### Q2: How to use vllm for multi-card deployment?
For details, see the [example](https://github.com/modelscope/ms-swift/tree/main/examples/deploy).

### Q3: How can clients pass images during vllm deployment?
See [client examples](https://github.com/modelscope/ms-swift/tree/main/examples/deploy/client/mllm) for details..

### Q4: I have a question about deploying qwen2-7b and using it with a client. When calling the OpenAI API, should I use `client.completions.create` instead of `client.chat.completions.create`, but when using `qwen2-7b-instruct-q5_k_m.gguf`, I can use `client.chat.completions.create`. Why is that?
The base model can use `client.chat.completions.create`, but this is a compatibility behavior.

### Q5: After launching the server with swift deploy using two cards, when I exit with Ctrl+C, there is always a Python process that continues to occupy the memory of one card. Is this a normal phenomenon?
You may need to kill it. This is an issue with vllm.

### Q6: Where to check if a model supports lmdeploy or vllm acceleration?
Vllm and lmdeploy have their own range of supported models. Please check their respective official documentation to determine availability.

### Q7: Why does Tongyi Qianwen 2.5-Math-7B-Instruct sometimes return garbled characters when using vllm deployment? Using vllm to deploy,fp16
Try using bf16.

### Q8: After starting the swift inference service, how can I set configurations like temperature interactively?
Inference only has preset configurations at startup, while deployment can set defaults initially and allow overriding them later on the client side.

### Q9: When deploying qwen2vl model locally, how can I input videos during inference? Can I use base64? How to call video with curl?
base64, see [mllm client example](https://github.com/modelscope/ms-swift/tree/main/examples/deploy/client/mllm) for details.

### Q10: When deploying qwen2-vl, I encounter an error about the vllm version not being correct?
```text
Unrecognized keys in `rope_scaling`for 'rope_type'='default': {'mrope_section'}
```
Refer to the [issue](https://github.com/QwenLM/Qwen2.5-VL/issues/209).

### Q11: When using Swift deploy for inference, I want to output token probabilities. I added logprobs True, but it outputs null. What's the reason?
```shell
RAY_memory_monitor_refresh_ms=0 CUDA_VISIBLE_DEVICES=1 nohup swift deploy --ckpt_dir /mnt/workspace/checkpoint_600 --infer_backend vllm --logprobs True --load_data_args false --host 0.0.0.0 --port 8000 &
```
Parameters need to be passed from the client side, `request_config = RequestConfig(..., logprobs=True, top_logprobs=2)`.

### Q12: Can we set request timeout time for Swift3.0 deployment inference? What happens if the image URL is invalid?
You can set the `TIMEOUT` environment variable, which defaults to 300 seconds. Alternatively, you can pass parameters in `InferClient`.

### Q13: Why can't I get streaming generation with Swift deployed models? I've set stream to True on both server and client side, but it's still not streaming
It's controlled by the client side. Please check [examples/deploy/client](https://github.com/modelscope/ms-swift/tree/main/examples/deploy/client).

### Q14: After deploying a multimodal model with Swift, is there an example of passing PIL.Image from the client?
Check this [client example](https://github.com/modelscope/ms-swift/blob/main/examples/deploy/client/mllm/openai_client.py).

### Q15: When deploying, which parameter should be set to output multiple results in a single response?
The parameter `n` in `RequestConfig`.

### Q16: When deploying using swift deploy with the parameter --infer_backend vllm, the performance was nearly 10 points worse compared to deploying directly with vllm: vllm serve. Does anyone know the reason for this?
It's likely that the template did not match.

### Q17: How can I disable the deep thinking mode of qwem3 in the deployment command?
Check this [issue](https://github.com/modelscope/ms-swift/issues/4030).

### Q18: When I use ms-swift for vllm deployment inference, it is much slower compared to native vllm. Is this a problem with the swift framework?
The main branch should be using the V1 engine by default. Try adding `VLLM_USE_V1=1`. Also, make sure to align the image resolution.

## Evaluation

### Q1: What evaluation datasets are supported by Swift?
Pure text evaluation:
```text
'obqa', 'cmb', 'AX_b', 'siqa', 'nq', 'mbpp', 'winogrande', 'mmlu', 'BoolQ', 'cluewsc', 'ocnli', 'lambada',
'CMRC', 'ceval', 'csl', 'cmnli', 'bbh', 'ReCoRD', 'math', 'humaneval', 'eprstmt', 'WSC', 'storycloze',
'MultiRC', 'RTE', 'chid', 'gsm8k', 'AX_g', 'bustm', 'afqmc', 'piqa', 'lcsts', 'strategyqa', 'Xsum', 'agieval',
'ocnli_fc', 'C3', 'tnews', 'race', 'triviaqa', 'CB', 'WiC', 'hellaswag', 'summedits', 'GaokaoBench',
'ARC_e', 'COPA', 'ARC_c', 'DRCD'
```

Multimodal evaluation:
```text
'COCO_VAL', 'MME', 'HallusionBench', 'POPE', 'MMBench_DEV_EN', 'MMBench_TEST_EN', 'MMBench_DEV_CN', 'MMBench_TEST_CN',
'MMBench', 'MMBench_CN', 'MMBench_DEV_EN_V11', 'MMBench_TEST_EN_V11', 'MMBench_DEV_CN_V11',
'MMBench_TEST_CN_V11', 'MMBench_V11', 'MMBench_CN_V11', 'SEEDBench_IMG', 'SEEDBench2',
'SEEDBench2_Plus', 'ScienceQA_VAL', 'ScienceQA_TEST', 'MMT-Bench_ALL_MI', 'MMT-Bench_ALL',
'MMT-Bench_VAL_MI', 'MMT-Bench_VAL', 'AesBench_VAL', 'AesBench_TEST', 'CCBench', 'AI2D_TEST', 'MMStar',
'RealWorldQA', 'MLLMGuard_DS', 'BLINK', 'OCRVQA_TEST', 'OCRVQA_TESTCORE', 'TextVQA_VAL', 'DocVQA_VAL',
'DocVQA_TEST', 'InfoVQA_VAL', 'InfoVQA_TEST', 'ChartQA_TEST', 'MathVision', 'MathVision_MINI',
'MMMU_DEV_VAL', 'MMMU_TEST', 'OCRBench', 'MathVista_MINI', 'LLaVABench', 'MMVet', 'MTVQA_TEST',
'MMLongBench_DOC', 'VCR_EN_EASY_500', 'VCR_EN_EASY_100', 'VCR_EN_EASY_ALL', 'VCR_EN_HARD_500',
'VCR_EN_HARD_100', 'VCR_EN_HARD_ALL', 'VCR_ZH_EASY_500', 'VCR_ZH_EASY_100', 'VCR_ZH_EASY_ALL',
'VCR_ZH_HARD_500', 'VCR_ZH_HARD_100', 'VCR_ZH_HARD_ALL', 'MMDU', 'MMBench-Video', 'Video-MME'
```

See the document [Evaluation](https://swift.readthedocs.io/en/latest/Instruction/Evaluation.html) for details.

### Q2: How to use a custom evaluation dataset?
Custom evaluation datasets, both plain text and multimodal, must match the data format (pattern) of an official dataset. See the document [Evaluation](https://swift.readthedocs.io/en/latest/Instruction/Evaluation.html) for details.

### Q3: Error with mmengine in python3.11 environment during evaluation
Try using the Python 3.10 environment. Or first install all dependencies:
`pip3 install evalscope[all]`,
then apply the patch:
`pip3 install https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/package/evalscope-0.5.3.post1-py3-none-any.whl`.

### Q4: After manually downloading the official evaluation dataset, can Swift eval be configured for local path evaluation?
First, download the evaluation dataset [eval.zip](https://modelscope.cn/datasets/swift/evalscope_resource/files), extract it, and place its contents in the `~/.cache/modelscope/media_resources/evalscope/data` folder. Then execute the `swift eval` command to use the local data.

### Q5: Is there a bug with custom evaluation? I modified the standard example to English, but it doesn't work?
```shell
swift eval --model_type 'qwen2_5-1_5b-instruct' --eval_dataset no --custom_eval_config '/mnt/workspace/test_data/config_eval.json'
```
This relies on the nltk package, which needs to download a punkt_tab zip file. Some environments in China have unstable or failed downloads. The code has been modified to handle this issue; reference [issue](https://github.com/nltk/nltk/issues/3293).

### Q6: The model after eval fine-tuning keeps stopping at a fixed percentage, but the vllm service seems to be running normally. The larger the model, the sooner it disconnects.
Set the `TIMEOUT` environment variable to -1.

### Q7: Does evalscope support multi-model comparison?
See the [documentation](https://evalscope.readthedocs.io/en/latest/user_guides/arena.html) for details.

### Q8: Is there a custom evaluation for multimodal datasets?
Custom evaluation for multimodal datasets can be referenced in the [documentation](https://evalscope.readthedocs.io/en/latest/advanced_guides/custom_dataset/index.html).

### Q9: Does ms-swift have methods to test QPS, latency, and tokens/s?
You can try using evalscope's [Model Inference Stress Testing](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/index.html).

### Q10: Can I control the number of dataset entries during evaluation? It takes over an hour to evaluate an MMLU, which is too slow.
Use the configuration parameter `--eval_limit`. This `--eval_limit` controls the number of entries in each subset. For example, if MMLU has over 50 subsets, and each limit is set to 10 entries, then that would be over 500 entries in total.

### Q11: When evaluating, isn't it just having the model output an answer once and checking if it's correct? Is there a way to record or see the complete answer each time?
For multiple-choice evaluations like ceval, the evaluation is done by calculating the logits for each option, without outputting the actual answer content. If you want to see the answer content, you can deploy the model as a service with a specified API URL for evaluation, which will evaluate based on parsing the model's output. See the [documentation](https://evalscope.readthedocs.io/en/latest/get_started/basic_usage.html#model-api-service-evaluation) for details. Both methods can be made optional.

### Q12: I want to stress test my model using evalscope and would like to use a prompt.txt file format. What should the format of this file look like?
Configure line_by_line, see the [documentation](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/parameters.html#dataset-configuration) for details.

### Q13: How should I use the 'parallel' and 'number' parameters when conducting model inference performance testing using evalscope perf?
`number` is the total number of requests, while `parallel` is the number of concurrent requests.

### Q14: In swift eval, the model stops generating after 1024 tokens. How can I modify this? Setting --max_new_tokens 5000 doesn't seem to work.
This parameter hasn't been exposed in swift yet. You can use evalscope to run it, and configure max_tokens in the model according to the [documentation](https://evalscope.readthedocs.io/en/latest/user_guides/backend/vlmevalkit_backend.html#configure-model-evaluation-parameters).

### Q15: Does evalscope currently support benchmarks like AIME and MATH-500 for deepseek-r1?
Yes, it does. Here are the [best practices](https://evalscope.readthedocs.io/en/latest/best_practice/deepseek_r1_distill.html).

### Q16: I'm getting this error when using a local path for gpqa evaluation in evalscope: ValueError: BuildingConfig 'gpqa_extended' not found. Available: ['default']
Parameter configuration:
```shell
 --datasets gpqa --dataset-args '{"gpqa": {"local_path": "/mnt/workspace/gpqa"} }'
 ```
If you want to use datasets locally, it's recommended to clone the repository from modelscope and then specify the path.

### Q17: When evaluating the arc dataset with evalscope, I get this error. What's the reason? I'm using the local data path method.
```text
KeyError: 'RequestId'
```
```shell
--datasets arc --dataset-args '{"arc": {"local_path": "/mnt/workspace/arc"}}'
```
According to the [documentation](https://evalscope.readthedocs.io/en/latest/get_started/basic_usage.html#using-local-datasets-and-models), the arc dataset needs to be downloaded using a Python script; directly cloning the repository won't work.

### Q18: How can I load downloaded datasets locally when using opencompass backend for evaluation?
The opencompass backend doesn't support setting `data_args`.

### Q19: Does swift eval with --eval_backend OpenCompass not support custom datasets?
```text
ValueError: eval_dataset: /mnt/workspace/data.jsonl is not supported.
eval_backend: OpenCompass supported datasets: ['C3', 'summedits', 'WiC', 'csl', 'lambada', 'mbpp', 'hellaswag', 'ARC_e', 'math', 'nq', 'race', 'MultiRC', 'cmb', 'ceval', 'GaokaoBench', 'mmlu', 'winogrande', 'tnews', 'triviaqa', 'CB', 'cluewsc', 'humaneval', 'AX_g', 'DRCD', 'RTE', 'ocnli_fc', 'gsm8k', 'obqa', 'ReCoRD', 'Xsum', 'ocnli', 'WSC', 'siqa', 'agieval', 'piqa', 'cmnli', 'cmmlu', 'eprstmt', 'storycloze', 'AX_b', 'afqmc', 'strategyqa', 'bustm', 'BoolQ', 'COPA', 'ARC_c', 'PMMEval', 'chid', 'CMRC', 'lcsts']
```
OpenCompass doesn't support custom datasets; use native mode for custom datasets.

### Q20: When I run the [RAGAS evaluation task](https://evalscope.readthedocs.io/zh-cn/latest/user_guides/backend/rageval_backend/ragas.html) from the evalscope official documentation locally on a single A100, it takes 10 minutes to run the two examples in the documentation. Is this normal? Are there ways to optimize the running speed?
RAG evaluation itself is resource-intensive, and using a local critic LLM will indeed be slower as it can't handle batch requests. It's recommended to use frameworks like vllm to launch tasks.

### Q21: I'm using evalscope to evaluate RAG, but I also want to use the API method to call the embedded model. Is this supported? I don't see it mentioned in the documentation.
Currently, embedding models do not support API calls, but this will be supported in the future.

### Q22: When testing a locally trained model using evalscope, the output for the test data is very simple, but the data was constructed in an inferential way during model training, leading to lower test results. How can evalscope be used to test only the data within xxx from the model's output?
Set `{"filters": {"remove_until": "</think>"}}` in dataset-args, and refer to this [documentation](https://evalscope.readthedocs.io/en/latest/get_started/parameters.html#dataset-parameters). Setting this parameter will remove `<think>` when calculating metrics.

### Q23: Evalscope can natively generate reports, but other backends like OpenCompass do not support report visualization, correct?
Currently, only native visualization is supported; other backends are not yet supported.

### Q24: Could you explain what causes the following error when using ifeval for evaluation?
```text
[Errno 20] Not a directory: '/root/nltk_data/tokenizers/punkt_tab.zip/punkt_tab/english/collocations.tab'
```
Unzip the file using `unzip /path/to/nltk_data/tokenizers/punkt_tab.zip`.

### Q25: When evaluating with eval_backend='OpenCompass', how can I specify the path to offline datasets?
Check the [data preparation guide](https://evalscope.readthedocs.io/en/latest/user_guides/backend/opencompass_backend.html#data-preparation), download and unzip the dataset. You don't need to specify `dataset-args`; just place the dataset folder (the data folder) in the current working directory.

### Q26: What causes the following error when using evalscope?
```text
unzip: cannot find or open /root/nltk_data/tokenizers/punkt_tab.zip, /root/nltk_data/tokenizers/punkt_tab.zip.zip or /root/nltk_data/tokenizers/punkt_tab.zip.ZIP
```
This occurs during the download of nltk dependencies. Manually download [punkt_tab.zip](https://modelscope-open.oss-cn-hangzhou.aliyuncs.com/open_data/nltk_data/punkt_tab.zip) and unzip it under `~/nltk_data/tokenizers`.

### Q27: Why is there no issue with plain text, but when testing multi-modal data, even though we specify the path, it still fails to detect the dataset and attempts to download it?
The vlmevalkit process differs from native; it will automatically download data to `~/LMUData/`.

### Q28: Could you explain how the score in evalscope is calculated? Is there any documentation about this part?
Please refer to this [issue](https://github.com/modelscope/evalscope/issues/610).

### Q29: When using swift eval for benchmark evaluation, can I specify an llm as a judge and how should I pass in the parameters?
Yes, you can use swift to pass `judge-model-args` parameters from `extra_eval_args`, which include `api_key, api_url, and model_id`, as a JSON string.
