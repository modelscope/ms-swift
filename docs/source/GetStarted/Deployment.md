# 部署

经过训练的模型可以使用各开源推理框架进行部署。下面介绍SWIFT框架如何对接开源推理框架进行部署。

## VLLM

[VLLM](https://github.com/vllm-project/vllm) 是针对transformer结构的推理加速框架，支持的Paged Attention和Continuous Batching等技术可以有效提升推理效率并减低显存占用。

使用VLLM的条件为：

1. 使用全参数微调或LoRA微调
2. 模型类型符合VLLM支持的模型类型

目前VLLM支持的模型系列为：

> - Aquila & Aquila2 (`BAAI/AquilaChat2-7B`, `BAAI/AquilaChat2-34B`, `BAAI/Aquila-7B`, `BAAI/AquilaChat-7B`, etc.)
> - Baichuan (`baichuan-inc/Baichuan-7B`, `baichuan-inc/Baichuan-13B-Chat`, etc.)
> - BLOOM (`bigscience/bloom`, `bigscience/bloomz`, etc.)
> - Falcon (`tiiuae/falcon-7b`, `tiiuae/falcon-40b`, `tiiuae/falcon-rw-7b`, etc.)
> - GPT-2 (`gpt2`, `gpt2-xl`, etc.)
> - GPT BigCode (`bigcode/starcoder`, `bigcode/gpt_bigcode-santacoder`, etc.)
> - GPT-J (`EleutherAI/gpt-j-6b`, `nomic-ai/gpt4all-j`, etc.)
> - GPT-NeoX (`EleutherAI/gpt-neox-20b`, `databricks/dolly-v2-12b`, `stabilityai/stablelm-tuned-alpha-7b`, etc.)
> - InternLM (`internlm/internlm-7b`, `internlm/internlm-chat-7b`, etc.)
> - LLaMA & LLaMA-2 (`meta-llama/Llama-2-70b-hf`, `lmsys/vicuna-13b-v1.3`, `young-geng/koala`, `openlm-research/open_llama_13b`, etc.)
> - Mistral (`mistralai/Mistral-7B-v0.1`, `mistralai/Mistral-7B-Instruct-v0.1`, etc.)
> - MPT (`mosaicml/mpt-7b`, `mosaicml/mpt-30b`, etc.)
> - OPT (`facebook/opt-66b`, `facebook/opt-iml-max-30b`, etc.)
> - Qwen (`Qwen/Qwen-7B`, `Qwen/Qwen-7B-Chat`, etc.)

首先需要安装vllm:

```shell
pip install vllm
```

如果是全参数微调，则可以使用vllm直接启动API服务，方法如下：

```shell
python -m vllm.entrypoints.openai.api_server --model /dir/to/your/trained/model --trust-remote-code
```

如果是LoRA微调，需要先执行下面的脚本将LoRA weights合并到原始模型中：

```shell
python merge_lora_weights_to_model.py --model_type /dir/to/your/base/model --ckpt_dir /dir/to/your/lora/model
```

合并后的模型会输出到`{ckpt_dir}-merged`文件夹中, 将该文件夹传入上述vllm命令中即可拉起服务。

调用服务：

```shell
curl http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "/dir/to/your/trained/model",
"prompt": "San Francisco is a",
"max_tokens": 7,
"temperature": 0
}'

# Response:
{"id":"cmpl-90329ab1eba24d02934b38f2edbb26a8","object":"text_completion","created":11506341,"model":"/dir/to/your/trained/model","choices":[{"index":0,"text":" city in the United States of America","logprobs":null,"finish_reason":"length"}],"usage":{"prompt_tokens":4,"total_tokens":11,"completion_tokens":7}}
```

vllm也支持使用python代码拉起模型并调用，具体可以查看[vllm官方文档](https://vllm.readthedocs.io/en/latest/getting_started/quickstart.html)。
