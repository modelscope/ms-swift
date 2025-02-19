# Sampling

Sampling is one of the newly supported key capabilities of SWIFT. This feature can be understood as the practical implementation of `test-time compute`. Additionally, this capability is crucial for the implementation of RFT (Reinforcement Fine-Tuning).

## Capability Introduction

The sampling capability of SWIFT can be demonstrated with the following example:

```shell
swift sample --model LLM-Research/Meta-Llama-3.1-8B-Instruct --sampler_engine pt --num_return_sequences 5 --dataset AI-ModelScope/alpaca-gpt4-data-zh#5
```

A `jsonl` file with a timestamp as the filename will be generated in the `sample_output` directory of the current folder. This file should contain 25 lines, each representing a complete `messages` format data.

For a list of sampling parameters, please refer to [here](Command-line-parameters.md).

## Environment Setup

```shell
pip install ms-swift[llm] -U
```

Or install swift from source:

```shell
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e '.[llm]'
```

## Using PRM and ORM for Result Filtering

An important capability of sampling is supervising the process and results, which can be supported by setting additional parameters.

```shell
swift sample --model LLM-Research/Meta-Llama-3.1-8B-Instruct --sampler_engine lmdeploy --num_return_sequences 5 --n_best_to_keep 2 --dataset tastelikefeet/competition_math#5 --prm_model AI-ModelScope/GRM-llama3.2-3B-rewardmodel-ft --orm_model math
```

A `jsonl` file with a timestamp as the filename will be generated in the `sample_output` directory of the current folder. This file **will contain at most** 10 lines, each representing a complete `messages` format data.
> The reason it contains at most 10 lines is that although 5 data points are processed in total, and 2 are kept for each data point (`n_best_to_keep`), ORM may fail some validations, and failed data will not be retained in the file.
> Additionally, after adding `--prm_model` or `--orm_model`, the file format is slightly different and includes a `rejected_response` key, which contains the responses with the lowest PRM scores.

## Customizing PRM or ORM

PRM and ORM can be customized by adding a new implementation in the plugin according to the existing code. For example:

```python
class CustomPRM:

    # The constructor should be parameterless
    def __init__(self):
        # Initialize here
        pass

    def __call__(self, infer_requests: List[InferRequest],  ground_truths: List[str], **kwargs) -> List[Union[float, List[float]]]:
        ...


prms = {'custom': CustomPRM}
```

Afterward, use `--prm_model custom` in the command line.

## Memory Control

If the sampled model and PRM are loaded into memory simultaneously, it may lead to an OOM (Out of Memory) issue. To address this, sampling can be divided into two stages:

- **Stage 1**: Specify `--model` and `--sampler_engine` without specifying `--orm_model` and `--prm_model`. Perform sampling only and save the results to a file.
- **Stage 2**: Specify `--sampler_engine no`, along with `--orm_model` and `--prm_model`, and also specify `--cache_files`. Perform only RM data filtering without re-sampling.

By dividing the process into two stages, only one model is loaded at a time, avoiding OOM issues.

## Practical Example

Please refer to the [Reinforcement Fine-Tuning Script](https://github.com/modelscope/ms-swift/tree/main/examples/train/rft/rft.py). This script provides a practical example of using sampling for reinforcement fine-tuning.

> **Note:** The actual effectiveness of this script is strongly related to the quality of the model, data, and RM. Therefore, it is presented only as an example. Users should modify this script and train their own RM and generator models accordingly.

## Sampling From Large Model

SWIFT's sample supports using the OpenAI API to distill data with large models. Example:

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
In this example:

`base_url` and `model` represent the API endpoint and model name, respectively. `stream` indicates the stream parameter for the request.

Note: For Deepseek-R1 series models, the output will be formatted as:`<thinking>{reasoning_content}</thinking>\n\n<answer>{content}</answer>`.
