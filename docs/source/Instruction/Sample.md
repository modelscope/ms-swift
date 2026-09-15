# 采样

采样是SWIFT新支持的重要能力之一，这部分可以理解为`test-time compute`的落地实现。同时，该能力对RFT（强化微调）的实现也至关重要。

## 能力介绍

SWIFT的sample能力可以使用下面的例子进行：
```shell
swift sample --model LLM-Research/Meta-Llama-3.1-8B-Instruct --sampler_engine transformers --num_return_sequences 5 --dataset AI-ModelScope/alpaca-gpt4-data-zh#5
```
在当前文件夹的`sample_output`目录下，会生成以时间戳为文件名的jsonl文件，该文件应该包含25行，每一行都是一个完整`messages`格式的数据。

采样的参数列表请参考[这里](Command-line-parameters.md)。

## 环境准备

```shell
pip install ms-swift[llm] -U
```

或从源代码安装：

```shell
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e '.[llm]'
```

## 使用PRM和ORM进行结果过滤

采样重要的能力就是对过程和结果进行监督，这可以通过设置额外参数来支持。

```shell
swift sample --model LLM-Research/Meta-Llama-3.1-8B-Instruct --sampler_engine lmdeploy --num_return_sequences 5 --n_best_to_keep 2 --dataset tastelikefeet/competition_math#5 --prm_model AI-ModelScope/GRM-llama3.2-3B-rewardmodel-ft --orm_model math
```

在当前文件夹的`sample_output`目录下，会生成以时间戳为文件名的jsonl文件，该文件**至多包含**10行，每一行都是一个完整`messages`格式的数据。
> 之所以至多包含10行，是因为虽然设置了共处理5个数据，每个数据保留2个（n_best_to_keep），但是orm可能会校验失败，失败数据不会保留到文件中。
> 另外，增加了--prm_model或--orm_model后文件格式有所不同，包含了rejected_response key，内容来自于prm评分最低的行。

## 自定义PRM或ORM

PRM和ORM的自定义可以在plugin中按照现有代码增加一个新的实现。例如：
```python
class CustomPRM:

    # 构造需要是无参的
    def __init__(self):
        # init here
        pass

    def __call__(self, infer_requests: List[InferRequest], ground_truths: List[str], **kwargs) -> List[Union[float, List[float]]]:
        ...


prms = {'custom': CustomPRM}
```

之后在命令行中使用`--prm_model custom`即可。

## 显存控制

如果被采样模型和PRM共同加载进显存，则可能出现OOM的问题。因此采样可以分为两段进行：

- 第一段指定`--model`和``--sampler_engine`，同时不指定`--orm_model`和`--prm_model`，仅进行采样，并存储为文件
- 第二段指定`--sampler_engine no`，指定`--orm_model`和`--prm_model`，并同时指定`--cache_files`，仅进行RM数据过滤，不重新采样

通过两段方式可以每次仅加载一个模型，防止OOM。

## 实际例子

请参考[强化微调脚本](https://github.com/modelscope/ms-swift/tree/main/examples/train/rft/rft.py)。该脚本给出了使用采样进行强化微调的实际例子。

> 注意：该脚本的实际效果和模型、数据、RM的质量强相关，因此仅作为样例出现，用户请自行修改该脚本并训练自己的RM和generator模型。

## 大模型蒸馏采样

SWIFT的sample支持使用OpenAI API的方式，用大模型蒸馏数据，如下示例：
```shell
OPENAI_API_KEY="your_api_key" \
swift sample \
    --sampler_type distill \
    --sampler_engine client \
    --model deepseek-r1 \
    --stream true \
    --dataset tastelikefeet/competition_math#5 \
    --num_return_sequences 1 \
    --temperature 0.6 \
    --top_p 0.95 \
    --engine_kwargs '{"base_url":"https://dashscope.aliyuncs.com/compatible-mode/v1"}'
```
在以上示例中，base_url和model分别是api地址和模型名称，stream表示发起请求的stream参数。

注意，对于Deepseek-R1系列模型，输出会被格式化为：`<think>{reasoning_content}</think>\n\n<answer>{content}</answer>`。

也可以使用 OpenAI 兼容的 AI 网关作为 provider，例如 [OrcaRouter](https://www.orcarouter.ai)。与 OpenRouter 类似，OrcaRouter 通过单一的 `https://api.orcarouter.ai/v1` 端点暴露跨多家模型的 provider/model 命名空间，并同时结合自适应路由、自动故障转移、零加成推理、可观测性、护栏与 Agent 工具治理。它还在同一端点上运行网关级、零信任的 AI Agent 安全能力——以默认拒绝的方式审查每一次 prompt/response 并治理每一次工具调用，无需任何应用代码改动。

```shell
OPENAI_API_KEY="your_api_key" \
swift sample \
    --sampler_type distill \
    --sampler_engine client \
    --model deepseek/deepseek-reasoner \
    --stream true \
    --dataset tastelikefeet/competition_math#5 \
    --num_return_sequences 1 \
    --temperature 0.6 \
    --top_p 0.95 \
    --engine_kwargs '{"base_url":"https://api.orcarouter.ai/v1"}'
```

将 `OPENAI_API_KEY` 设置为你的 OrcaRouter API key。由于 OrcaRouter 提供统一的 provider/model 命名空间，`model` 可以填网关暴露的任意模型，例如 `deepseek/deepseek-reasoner`。
