# OLLaMA Export Documentation

SWIFT now supports exporting OLLaMA Model files, integrated into the `swift export` command.

## Contents

- [Environment Setup](#environment-setup)
- [Export](#export)
- [Points to Note](#points-to-note)

## Environment Setup

```shell
# Set pip global mirror (to speed up downloads)
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# Install ms-swift
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'
```

No additional modules are needed for OLLaMA export, as SWIFT only exports the ModelFile. Users can handle subsequent operations.

## Export

The OLLaMA export command line is as follows:

```shell
# model_type
swift export --model_type llama3-8b-instruct --to_ollama true --ollama_output_dir llama3-8b-instruct-ollama
# ckpt_dir, note that for lora training, add --merge_lora true
swift export --ckpt_dir /mnt/workspace/yzhao/tastelikefeet/swift/output/qwen-7b-chat/v141-20240331-110833/checkpoint-10942 --to_ollama true --ollama_output_dir qwen-7b-chat-ollama --merge_lora true
```

After execution, the following log will be printed:
```shell
[INFO:swift] Exporting to ollama:
[INFO:swift] If you have a gguf file, try to pass the file by :--gguf_file /xxx/xxx.gguf, else SWIFT will use the original(merged) model dir
[INFO:swift] Downloading the model from ModelScope Hub, model_id: LLM-Research/Meta-Llama-3-8B-Instruct
[WARNING:modelscope] Authentication has expired, please re-login with modelscope login --token "YOUR_SDK_TOKEN" if you need to access private models or datasets.
[WARNING:modelscope] Using branch: master as version is unstable, use with caution
[INFO:swift] Loading the model using model_dir: /mnt/workspace/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B-Instruct
[INFO:swift] Save Modelfile done, you can start ollama by:
[INFO:swift] > ollama serve
[INFO:swift] In another terminal:
[INFO:swift] > ollama create my-custom-model -f /mnt/workspace/yzhao/tastelikefeet/swift/llama3-8b-instruct-ollama/Modelfile
[INFO:swift] > ollama run my-custom-model
[INFO:swift] End time of running main: 2024-08-09 17:17:48.768722
```

Check the Modelfile:

```text
FROM /mnt/workspace/.cache/modelscope/hub/LLM-Research/Meta-Llama-3-8B-Instruct
TEMPLATE """{{ if .System }}<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ else }}<|begin_of_text|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{ end }}{{ .Response }}<|eot_id|>"""
PARAMETER stop "<|eot_id|>"
PARAMETER temperature 0.3
PARAMETER top_k 20
PARAMETER top_p 0.7
PARAMETER repeat_penalty 1.0
```

Users can modify the generated file for subsequent inference.

### Using OLLaMA

To use the above file, install OLLaMA:

```shell
# https://github.com/ollama/ollama
curl -fsSL https://ollama.com/install.sh | sh
```

Start OLLaMA:

```shell
ollama serve
```

In another terminal, run:

```shell
ollama create my-custom-model -f /mnt/workspace/yzhao/tastelikefeet/swift/llama3-8b-instruct-ollama/Modelfile
```

The following log will be printed after execution:

```text
transferring model data
unpacking model metadata
processing tensors
converting model
creating new layer sha256:37b0404fb276acb2e5b75f848673566ce7048c60280470d96009772594040706
creating new layer sha256:2ecd014a372da71016e575822146f05d89dc8864522fdc88461c1e7f1532ba06
creating new layer sha256:ddc2a243c4ec10db8aed5fbbc5ac82a4f8425cdc4bd3f0c355373a45bc9b6cb0
creating new layer sha256:fc776bf39fa270fa5e2ef7c6782068acd858826e544fce2df19a7a8f74f3f9df
writing manifest
success
```

You can then use the command name for inference:

```shell
ollama run my-custom-model
```

```shell
>>> who are you?
I'm LLaMA, a large language model trained by a team of researchers at Meta AI. My primary function is to understand and respond to human
input in a helpful and informative way. I'm a type of AI designed to simulate conversation, answer questions, and even generate text based
on a given prompt or topic.

I'm not a human, but rather a computer program designed to mimic human-like conversation. I don't have personal experiences, emotions, or
physical presence, but I'm here to provide information, answer your questions, and engage in conversation to the best of my abilities.

I'm constantly learning and improving my responses based on the interactions I have with users like you, so please bear with me if I make
any mistakes or don't quite understand what you're asking. I'm here to help and provide assistance, so feel free to ask me anything!
```

## Points to Note

1. Some models may report an error during:

```shell
ollama create my-custom-model -f /mnt/workspace/yzhao/tastelikefeet/swift/qwen-7b-chat-ollama/Modelfile
```

Error message:

```shell
Error: Models based on 'QWenLMHeadModel' are not yet supported
```

This is because the conversion in OLLaMA does not support all types of models. You can perform gguf export yourself and modify the FROM field in the Modelfile:

```shell
# Detailed conversion steps can be found at: https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
# The model directory can be found in the `swift export` command log, similar to:
# Using model_dir: /mnt/workspace/yzhao/tastelikefeet/swift/output/qwen-7b-chat/v141-20240331-110833/checkpoint-10942-merged
python convert_hf_to_gguf.py /mnt/workspace/yzhao/tastelikefeet/swift/output/qwen-7b-chat/v141-20240331-110833/checkpoint-10942-merged
```

Then re-execute:

```shell
ollama create my-custom-model -f /mnt/workspace/yzhao/tastelikefeet/swift/qwen-7b-chat-ollama/Modelfile
```
