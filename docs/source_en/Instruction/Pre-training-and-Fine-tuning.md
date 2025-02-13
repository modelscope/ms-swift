# Pre-training and Fine-tuning
Training Capability:

| Method                             | Full-Parameter                                               | LoRA                                                         | QLoRA                                                        | Deepspeed                                                    | Multi-Node                                                   | Multi-Modal                                                  |
|------------------------------------|--------------------------------------------------------------|--------------------------------------------------------------|--------------------------------------------------------------|--------------------------------------------------------------|--------------------------------------------------------------|--------------------------------------------------------------|
| Pre-training                       | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/pretrain/train.sh) | ✅                                                            | ✅                                                            | ✅                                                            | ✅                                                            | ✅                                                            |
| Instruction Supervised Fine-tuning | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/full/train.sh) | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/lora_sft.sh) | [✅](https://github.com/modelscope/ms-swift/tree/main/examples/train/qlora) | [✅](https://github.com/modelscope/ms-swift/tree/main/examples/train/multi-gpu/deepspeed) | [✅](https://github.com/modelscope/ms-swift/tree/main/examples/train/multi-node)                                                            | [✅](https://github.com/modelscope/ms-swift/tree/main/examples/train/multimodal) |
| DPO Training                       | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/rlhf/dpo.sh) | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/rlhf/dpo.sh) | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/multimodal/rlhf/dpo.sh) |
| GRPO Training                      | [✅]((https://github.com/modelscope/ms-swift/blob/main/examples/train/grpo/grpo_zero2.sh)) | ✅                                                            | ✅                                                            | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/grpo/multi_node)                      | ✅                                                            |
| Reward Model Training              | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/rlhf/rm.sh) | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/rlhf/rm.sh) | ✅                                                            | ✅                                                            |
| PPO Training                       | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/rlhf/ppo.sh) | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/rlhf/ppo.sh) | ✅                                                            | ❌                                                            |
| KTO Training                       | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/rlhf/kto.sh) | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/rlhf/kto.sh) | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/multimodal/rlhf/kto.sh) |
| CPO Training                       | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/rlhf/cpo.sh) | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/rlhf/cpo.sh) | ✅                                                            | ✅                                                            |
| SimPO Training                     | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/rlhf/simpo.sh) | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/rlhf/simpo.sh) | ✅                                                            | ✅                                                            |
| ORPO Training                      | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/rlhf/orpo.sh) | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/rlhf/orpo.sh) | ✅                                                            | ✅                                                            |
| Classification Model Training      | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/seq_cls/qwen2_5/sft.sh) | ✅                                                            | ✅                                                            | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/seq_cls/qwen2_vl/sft.sh) |
| Embedding Model Training           | ✅                                                            | [✅](https://github.com/modelscope/ms-swift/blob/main/examples/train/embedding/train.sh) | ✅                                                            | ✅                                                            | ✅                                                            | ❌


## Environment Preparation

Refer to the [SWIFT installation documentation](../GetStarted/SWIFT-installation.md) for recommended versions of third-party libraries.

```shell
pip install ms-swift -U

# If using deepspeed zero2/zero3
pip install deepspeed==0.14.5
```

## Pre-training

Pre-training is done using the `swift pt` command, which will automatically use the generative template instead of the conversational template, meaning that `use_chat_template` is set to False (all other commands, such as `swift sft/rlhf/infer`, default `use_chat_template` to True). Additionally, `swift pt` has a different dataset format compared to `swift sft`, which can be referenced in the [Custom Dataset Documentation](../Customization/Custom-dataset.md).

You can refer to the CLI script for pre-training [here](https://github.com/modelscope/swift/blob/main/examples/train/pretrain/train.sh). For more information on training techniques, please refer to the fine-tuning section.

Tips:

- `swift pt` is equivalent to `swift sft --use_chat_template false`.
- `swift pt` typically uses large datasets, and it is recommended to combine it with `--streaming` for streaming datasets.

## Fine-tuning

ms-swift employs a hierarchical design philosophy, allowing users to perform fine-tuning through the command line interface, Web-UI interface, or directly using Python.

### Using CLI

We provide best practices for self-cognition fine-tuning of Qwen2.5-7B-Instruct on a single 3090 GPU in 10 minutes; for details, refer to [here](../GetStarted/Quick-start.md). This can help you quickly understand SWIFT.

Additionally, we offer a series of scripts to help you understand the training capabilities of SWIFT:

- Lightweight Training: Examples of lightweight fine-tuning supported by SWIFT can be found [here](https://github.com/modelscope/swift/blob/main/examples/train/tuners). (Note: These methods can also be used for pre-training, but pre-training typically uses full parameter training.)
- Distributed Training: SWIFT supports distributed training techniques, including: DDP, device_map, DeepSpeed ZeRO2/ZeRO3, and FSDP.
  - device_map: Simplified model parallelism. If multiple GPUs are available, device_map will be automatically enabled. This evenly partitions the model layers across visible GPUs, significantly reducing memory consumption, although training speed may decrease due to serial processing.
  - DDP + device_map: Models will be grouped and partitioned using device_map. Refer to [here](https://github.com/modelscope/ms-swift/blob/main/examples/train/multi-gpu/ddp_device_map/train.sh) for details.
  - DeepSpeed ZeRO2/ZeRO3: Save memory resources but may reduce training speed. ZeRO2 shards optimizer states and model gradients. ZeRO3 further shards model parameters on top of ZeRO2, saving even more memory but reducing training speed further. Refer to [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/multi-gpu/deepspeed) for details.
  - FSDP + QLoRA: Training a 70B model on two 3090 GPUs. Refer to [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/multi-gpu/fsdp_qlora/train.sh).
  - Multi-node Multi-GPU Training: We have provided example shell scripts for launching multi-node runs using swift, torchrun, dlc, deepspeed, and accelerate. Except for dlc and deepspeed, the other launch scripts need to be started on all nodes to run properly. Please refer to [here](https://github.com/modelscope/swift/blob/main/examples/train/multi-node) for details.
- Quantization Training: Supports QLoRA training using quantization techniques such as GPTQ, AWQ, AQLM, BNB, HQQ, and EETQ. Fine-tuning a 7B model only requires 9GB of memory. For more details, refer to [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/qlora).
- Multi-modal Training: SWIFT supports pre-training, fine-tuning, and RLHF for multi-modal models. It supports tasks such as Captioning, VQA, OCR, and [Grounding](https://github.com/modelscope/ms-swift/blob/main/examples/notebook/qwen2_5-vl-grounding/zh.ipynb). It supports three modalities: images, videos, and audio. For more details, refer to [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/multimodal). The format for custom multi-modal datasets can be found in the [Custom Dataset Documentation](../Customization/Custom-dataset.md).
- RLHF Training: Refer to [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/rlhf). For multi-modal models, refer to [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/multimodal/rlhf). For GRPO training, refer to [here](https://github.com/modelscope/ms-swift/blob/main/examples/train/grpo/grpo_zero2.sh). For reinforcement fine-tuning, see [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/rft).
- Sequence Classification Model Training: Refer to [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/seq_cls).
- Embedding Model Training: Refer to [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/embedding).
- Agent Training: Refer to [here](https://github.com/modelscope/swift/blob/main/examples/train/agent).
- Any-to-Any Model Training: Refer to [here](https://github.com/modelscope/swift/blob/main/examples/train/all_to_all).
- Other Capabilities:
  - Streaming Data Reading: Reduces memory usage when handling large datasets. Refer to [here](https://github.com/modelscope/swift/blob/main/examples/train/streaming/train.sh).
  - Sequence Parallelism: Refer to [here](https://github.com/modelscope/swift/blob/main/examples/train/sequence_parallel).
  - Packing: Combines multiple sequences into one, making each training sample as close to max_length as possible to improve GPU utilization. Refer to [here](https://github.com/modelscope/swift/blob/main/examples/train/packing/train.sh).
  - Lazy Tokenize: Performs tokenization during training instead of pre-training (for multi-modal models, this avoids the need to load all multi-modal resources before training), which can reduce preprocessing wait times and save memory. Refer to [here](https://github.com/modelscope/swift/blob/main/examples/train/lazy_tokenize/train.sh).


### Tips:

- When fine-tuning a base model to a chat model using LoRA technology with `swift sft`, you may sometimes need to manually set the template. Add the `--template default` parameter to avoid issues where the base model may fail to stop correctly due to encountering special characters in the dialogue template that it has not seen before. For more details, see [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/base_to_chat).
- If you need to train in an **offline** environment, please set `--model <model_dir>` and `--check_model false`. If the corresponding model requires `git clone` from GitHub repositories, such as `deepseek-ai/Janus-Pro-7B`, please manually download the repository and set `--local_repo_path <repo_dir>`. For specific parameter meanings, refer to the [command line parameter documentation](./Command-line-parameters.md).
- Merging LoRA for models trained with QLoRA is not possible, so it is not recommended to use QLoRA for fine-tuning, as it cannot utilize vLLM/LMDeploy for inference acceleration during inference and deployment. It is recommended to use LoRA or full parameter fine-tuning, merge them into complete weights, and then use GPTQ/AWQ/BNB for [quantization](https://github.com/modelscope/ms-swift/tree/main/examples/export/quantize).
- By default, SWIFT sets `--gradient_checkpointing true` during training to save memory, which may slightly slow down the training speed.
- If you are using DDP for training and encounter the error: `RuntimeError: Expected to mark a variable ready only once.`, please additionally set the parameter `--gradient_checkpointing_kwargs '{"use_reentrant": false}'` or use DeepSpeed for training.
- To use DeepSpeed, you need to install it: `pip install deepspeed==0.14.5`. Using DeepSpeed can save memory but may slightly reduce training speed.
- If your machine has high-performance GPUs like A100 and the model supports flash-attn, it is recommended to install [flash-attn](https://github.com/Dao-AILab/flash-attention/releases) and set `--attn_impl flash_attn`, as this will accelerate training and inference while slightly reducing memory usage.

**How to debug:**

You can use the following method for debugging, which is equivalent to using the command line for fine-tuning, but this method does not support distributed training. You can refer to the entry point for the fine-tuning command line [here](https://github.com/modelscope/ms-swift/blob/main/swift/cli/sft.py).

```python
from swift.llm import sft_main, TrainArguments
result = sft_main(TrainArguments(
    model='Qwen/Qwen2.5-7B-Instruct',
    train_type='lora',
    dataset=['AI-ModelScope/alpaca-gpt4-data-zh#500',
             'AI-ModelScope/alpaca-gpt4-data-en#500',
             'swift/self-cognition#500'],
    torch_dtype='bfloat16',
    # ...
))
```


### Using Web-UI

If you want to use the interface for training, you can refer to the [Web-UI documentation](../GetStarted/Web-UI.md).

### Using Python

- For the Qwen2.5 self-cognition fine-tuning notebook, see [here](https://github.com/modelscope/ms-swift/blob/main/examples/notebook/qwen2_5-self-cognition/self-cognition-sft.ipynb).
- For the Qwen2VL OCR task notebook, see [here](https://github.com/modelscope/ms-swift/blob/main/examples/notebook/qwen2vl-ocr/ocr-sft.ipynb).

## Merge LoRA

- See [here](https://github.com/modelscope/ms-swift/blob/main/examples/export/merge_lora.sh).


## Inference (After Fine-Tuning Model)

To perform inference on a LoRA-trained checkpoint using the CLI:

```shell
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --infer_backend pt \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048
```

- If you are using full parameter training, please replace `--adapters` with `--model` to specify the directory of the trained checkpoint.
- You can choose to merge LoRA (by additionally specifying `--merge_lora true`) and then specify `--infer_backend vllm/lmdeploy` for inference acceleration.
- You can use `swift app` instead of `swift infer` for interface-based inference.

For batch inference on the validation set of the dataset:

```shell
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --infer_backend pt \
    --temperature 0 \
    --max_new_tokens 2048 \
    --load_data_args true \
    --max_batch_size 1
```

- You can set `--max_batch_size 8` to enable batch processing with `--infer_backend pt`. If you use `infer_backend vllm/lmdeploy`, it will automatically handle batching without needing to specify.
- `--load_data_args true` will read the parameter file `args.json` that was stored during training.

If you want to perform inference on an additional test set instead of using the training validation set, use `--val_dataset <dataset_path>` for inference:

```shell
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters output/vx-xxx/checkpoint-xxx \
    --infer_backend pt \
    --temperature 0 \
    --max_new_tokens 2048 \
    --val_dataset <dataset-path> \
    --max_batch_size 1
```


Example of Inference on LoRA-Trained Model Using Python:

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    PtEngine, RequestConfig, safe_snapshot_download, get_model_tokenizer, get_template, InferRequest
)
from swift.tuners import Swift
# Please adjust the following lines
model = 'Qwen/Qwen2.5-7B-Instruct'
lora_checkpoint = safe_snapshot_download('swift/test_lora')  # Change to your checkpoint_dir
template_type = None  # None: use the default template_type of the corresponding model
default_system = "You are a helpful assistant."  # None: use the default system prompt of the corresponding model

# Load model and dialogue template
model, tokenizer = get_model_tokenizer(model)
model = Swift.from_pretrained(model, lora_checkpoint)
template_type = template_type or model.model_meta.template
template = get_template(template_type, tokenizer, default_system=default_system)
engine = PtEngine.from_model_template(model, template, max_batch_size=2)
request_config = RequestConfig(max_tokens=512, temperature=0)

# Using 2 infer_requests to demonstrate batch inference
infer_requests = [
    InferRequest(messages=[{'role': 'user', 'content': 'who are you?'}]),
    InferRequest(messages=[{'role': 'user', 'content': 'Where is the capital of Zhejiang?'},
                           {'role': 'assistant', 'content': 'Where is the capital of Zhejiang?'},
                           {'role': 'user', 'content': 'What is good to eat here?'},]),
]
resp_list = engine.infer(infer_requests, request_config)
query0 = infer_requests[0].messages[0]['content']
print(f'query0: {query0}')
print(f'response0: {resp_list[0].choices[0].message.content}')
print(f'response1: {resp_list[1].choices[0].message.content}')
```

Example of LoRA Inference for Multi-Modal Model:

```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import (
    PtEngine, RequestConfig, safe_snapshot_download, get_model_tokenizer, get_template, InferRequest
)
from swift.tuners import Swift
# Please adjust the following lines
model = 'Qwen/Qwen2.5-VL-7B-Instruct'
lora_checkpoint = safe_snapshot_download('swift/test_grounding')  # Change to your checkpoint_dir
template_type = None  # None: use the default template_type of the corresponding model
default_system = None  # None: use the default system prompt of the corresponding model

# Load model and dialogue template
model, tokenizer = get_model_tokenizer(model)
model = Swift.from_pretrained(model, lora_checkpoint)
template_type = template_type or model.model_meta.template
template = get_template(template_type, tokenizer, default_system=default_system)
engine = PtEngine.from_model_template(model, template, max_batch_size=2)
request_config = RequestConfig(max_tokens=512, temperature=0)

# Using 2 infer_requests to demonstrate batch inference
infer_requests = [
    InferRequest(messages=[{'role': 'user', 'content': 'who are you?'}]),
    InferRequest(messages=[{'role': 'user', 'content': '<image>Task: Object Detection'}],
                 images=['http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/animal.png']),
]
resp_list = engine.infer(infer_requests, request_config)
query0 = infer_requests[0].messages[0]['content']
print(f'query0: {query0}')
print(f'response0: {resp_list[0].choices[0].message.content}')
print(f'response1: {resp_list[1].choices[0].message.content}')
```

- To perform inference on a full parameter-trained checkpoint, you can refer to the [large model inference example](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo.py) and just change the `model`.
- For streaming inference and acceleration using `VllmEngine` and `LmdeployEngine`, you can refer to the inference examples for [large models](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo.py) and [multi-modal large models](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo_mllm.py).
- For inference on fine-tuned models using the Hugging Face transformers/PEFT ecosystem, you can see [here](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo_hf.py).
- If you have trained multiple LoRAs and need to switch among them, refer to the [inference](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo_lora.py) and [deployment](https://github.com/modelscope/ms-swift/tree/main/examples/deploy/lora) examples.
- For grounding tasks in multi-modal models, you can refer to [here](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo_grounding.py).
- For inference on a LoRA fine-tuned BERT model, see [here](https://github.com/modelscope/ms-swift/blob/main/examples/infer/demo_bert.py).
