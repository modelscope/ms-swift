# OLLaMA导出文档

SWIFT已经支持了OLLaMA Modelfile的导出能力，该能力合并到了`swift export`命令中。

## 目录

- [环境准备](#环境准备)
- [导出](#导出)
- [需要注意的问题](#需要注意的问题)

## 环境准备

```shell
# 设置pip全局镜像 (加速下载)
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
# 安装ms-swift
git clone https://github.com/modelscope/swift.git
cd swift
pip install -e '.[llm]'
```

OLLaMA导出不需要其他模块支持，因为SWIFT仅会导出ModelFile，后续的运行用户可以自行处理。

## 导出

OLLaMA导出命令行如下：

```shell
# model_type
swift export --model_type llama3-8b-instruct --to_ollama true --ollama_output_dir llama3-8b-instruct-ollama
# ckpt_dir，注意lora训练需要增加--merge_lora true
swift export --ckpt_dir /mnt/workspace/yzhao/tastelikefeet/swift/output/qwen-7b-chat/v141-20240331-110833/checkpoint-10942 --to_ollama true --ollama_output_dir qwen-7b-chat-ollama --merge_lora true
```

执行后会打印如下log：
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

提示可以运行，此时打开ModelFile查看：

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

用户可以改动生成的文件，用于后续推理。

### OLLaMA使用

使用上面的文件，需要安装OLLaMA：
```shell
# https://github.com/ollama/ollama
curl -fsSL https://ollama.com/install.sh | sh
```

启动OLLaMA:

```shell
ollama serve
```

在另一个terminal运行：

```shell
ollama create my-custom-model -f /mnt/workspace/yzhao/tastelikefeet/swift/llama3-8b-instruct-ollama/Modelfile
```

执行后会打印如下log：

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

之后就可以用命令的名字来推理：

```shell
ollama run my-custom-model
```

```shell
>>> who are you?
I'm LLaMA, I'm a large language model trained by a team of researcher at Meta AI. My primary function is to understand and respond to human
input in a helpful and informative way. I'm a type of AI designed to simulate conversation, answer questions, and even generate text based
on a given prompt or topic.

I'm not a human, but rather a computer program designed to mimic human-like conversation. I don't have personal experiences, emotions, or
physical presence, but I'm here to provide information, answer your questions, and engage in conversation to the best of my abilities.

I'm constantly learning and improving my responses based on the interactions I have with users like you, so please bear with me if I make
any mistakes or don't quite understand what you're asking. I'm here to help and provide assistance, so feel free to ask me anything!
```

## 需要注意的问题

1. 部分模型在

```shell
ollama create my-custom-model -f /mnt/workspace/yzhao/tastelikefeet/swift/qwen-7b-chat-ollama/Modelfile
```

的时候会报错:

```shell
Error: Models based on 'QWenLMHeadModel' are not yet supported
```

这是因为ollama的转换并不支持所有类型的模型，此时可以自行进行gguf导出并修改Modelfile的FROM字段：

```shell
# 详细转换步骤可以参考：https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
# 模型目录可以在`swift export`命令的日志中找到，类似：
# Using model_dir: /mnt/workspace/yzhao/tastelikefeet/swift/output/qwen-7b-chat/v141-20240331-110833/checkpoint-10942-merged
python convert_hf_to_gguf.py /mnt/workspace/yzhao/tastelikefeet/swift/output/qwen-7b-chat/v141-20240331-110833/checkpoint-10942-merged
```

之后重新执行：

```shell
ollama create my-custom-model -f /mnt/workspace/yzhao/tastelikefeet/swift/qwen-7b-chat-ollama/Modelfile
```
