# Architecture Introduction

ms-swift 4.0 adopts a modular design, with functional modules distributed in first-level directories, making it convenient for developers to perform custom extensions. This document will provide a detailed introduction to the functions of each module and customization methods.

## Agent Template

The mapping file for agent templates can be found [here](https://github.com/modelscope/ms-swift/blob/main/swift/agent_template/mapping.py). The design goal of agent template is to flexibly switch between different models for training based on a unified Agent dataset format, without modifying the data. During training, use `--agent_template` to specify the corresponding agent template.

All AgentTemplates need to inherit from `BaseAgentTemplate` and implement several methods: `_format_tools`, `_format_tool_calls`, `_format_tool_responses`, `get_toolcall`.
- _format_tools: Format `tools` and `system` to compose a complete system.
- _format_tool_calls: Format the tool_call part `[{"role": "tool_call", "content": "..."}, {"role": "tool_call", "content": "..."}]` and finally return a string.
- _format_tool_responses: Format the tool (also called tool_response) part `[{"role": "tool", "content": "..."}, {"role": "tool", "content": "..."}]`.
- get_toolcall: Used during deployment to parse the tool name and parameters from the model output content, returning `List[Function]`.


How to debug:
```python
data = {"tools": "[{\"type\": \"function\", \"function\": {\"name\": \"realtime_aqi\", \"description\": \"天气预报。获取实时空气质量。当前空气质量，PM2.5，PM10信息\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\", \"description\": \"城市名，例如：上海\"}}, \"required\": [\"city\"]}}}]", "messages": [{"role": "user", "content": "北京和上海今天的天气情况"}, {"role": "tool_call", "content": "{\"name\": \"realtime_aqi\", \"arguments\": {\"city\": \"北京\"}}"}, {"role": "tool_call", "content": "{\"name\": \"realtime_aqi\", \"arguments\": {\"city\": \"上海\"}}"}, {"role": "tool_response", "content": "{\"city\": \"北京\", \"aqi\": \"10\", \"unit\": \"celsius\"}"}, {"role": "tool_response", "content": "{\"city\": \"上海\", \"aqi\": \"72\", \"unit\": \"fahrenheit\"}"}, {"role": "assistant", "content": "根据天气预报工具，北京今天的空气质量指数为10，属于良好水平；上海今天的空气质量指数为72，属于轻度污染水平。"}]}


from swift import get_processor, get_template

tokenizer = get_processor('Qwen/Qwen3.5-2B')
template = get_template(tokenizer)  # Use default agent template
# template = get_template(tokenizer, agent_template='qwen3_5')
print(f'agent_template: {template._agent_template}')
template.set_mode('train')
encoded = template.encode(data)
print(f'[INPUT_IDS] {template.safe_decode(encoded["input_ids"])}\n')
print(f'[LABELS] {template.safe_decode(encoded["labels"])}')
```

If you want to provide us with a PR, please refer to [here](https://github.com/modelscope/ms-swift/blob/main/tests/test_align/test_template/test_agent.py) to write your test cases.

## Callbacks

The mapping file for callbacks can be found [here](https://github.com/modelscope/ms-swift/blob/main/swift/callbacks/mapping.py). Callbacks can customize the behavior at key points in the trainer. After customization, you need to register them in the mapping and use `--callbacks` to specify the corresponding callback class during training. For example, you can customize:

```python
class CustomCallback(TrainerCallback):

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Doing something when the training begins.
        pass

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Doing something when save checkpoint
        pass
```

All callback classes need to inherit from `TrainerCallback` in base.py and override its methods. The interface is consistent with transformers' `TrainerCallback`, please refer to transformers' [callback documentation](https://huggingface.co/docs/transformers/main_classes/callback).

## Loss

The mapping file for Loss can be found [here](https://github.com/modelscope/ms-swift/blob/main/swift/loss/mapping.py).
Swift supports custom loss (currently only supports sft/pretrain/reranker/embedding tasks). After registration, set `--loss_type <loss-name>` during training to use your custom loss method.

Custom Loss needs to inherit from `BaseLoss` and implement the `__call__` method, returning a scalar Tensor. You can refer to [CustomCrossEntropyLoss](https://github.com/modelscope/ms-swift/blob/0d7c9f5bc0e7e7d67d914ce6edeb9ce24f60746f/swift/loss/causal_lm.py#L5) for customization. For example:

```python
class CustomLoss(BaseLoss):

    def __call__(self, outputs, labels, **kwargs) -> torch.Tensor:
        pass
```

## Loss Scale

The mapping file for loss scale can be found [here](https://github.com/modelscope/ms-swift/blob/main/swift/loss_scale/mapping.py). In pretrain and sft tasks, the loss of trainable tokens is averaged, meaning each token is treated equally. However, in some cases, certain tokens need extra attention and should be assigned higher weights, or some tokens should not be trained. loss_scale allows developers to freely define their own token weights. (Pretrain and SFT support using loss_scale to control whether tokens participate in training and their weight sizes, while in RLHF, it only supports controlling whether tokens participate in training)

You can customize loss scale by inheriting the LossScale base class and implementing the `get_loss_scale` method.

```python
class CustomLossScale(LossScale):

    def get_loss_scale(self, context: str, **kwargs) -> Tuple[List[str], List[float]]:
        ...
```

The `get_loss_scale` function returns a Tuple. The first return is a list of decomposed strings, and the second parameter is a list of loss_scales corresponding to the strings. The float value represents the weight. For example, the following weight setting:

```text
["学习", "好", "数学", "是", "重要", "的"]
[1.0, 0.5, 2.0, 0.5, 2.0, 0.1]
```
In the example, we place more emphasis on the words "数学" and "重要" because their loss_scale is 2.0.

Of course, we also need to pay attention to the core logic of the `__call__` method, namely the influence of the loss_scale base strategy (base_strategy) all/default/last_round on loss_scale. For details, refer to the introduction in the [Command-line Parameters Documentation](../Instruction/Command-line-parameters.md). Also, refer to the influence of the 'loss' field in the dataset on loss_scale in the [Custom Dataset Documentation](../Customization/Custom-dataset.md).

```python
if loss or loss is None and (self.base_strategy == 'all' or
                            (self.base_strategy == 'default' and is_assistant) or
                            (self.base_strategy == 'last_round' and is_assistant and is_last_round)):
    new_context, loss_scale = self.get_loss_scale(context, query=query)
else:
    new_context, loss_scale = [context], [0.]
```

In addition, you can also use [JSON configuration files](https://github.com/modelscope/ms-swift/tree/main/swift/loss_scale/config) and inherit the built-in ConfigLossScale class to customize loss_scale. Currently, two configuration methods are supported: exact string matching and regular expression matching. You can refer to the content in [Agent Support Documentation](../Instruction/Agent-support.md#usage-of-loss_scale) for understanding.

- Exact string matching, for example, refer to `react.json`, `qwen.json`. The JSON needs to contain a mapping of `Dict[str, List[float]]`. The string represents a keyword, and the list needs to have two values. We will split the string into multiple segments based on the keyword. The first value in the list represents the weight of the keyword, and the second value represents the weight of the content after this keyword and before the next keyword.
- Regular expression matching, for example, refer to `ignore_empty_think.json`, `hermes.json`. The JSON needs to contain a mapping of `Dict[str, float]`. The string represents a regular expression pattern, and the float represents the weight of the matching string.

How to debug:

```python
from swift import get_processor, get_template

data = {"messages": [
    {"role": "user", "content": "What is today's date?"},
    {"role": "assistant", "content": (
        "<think>\nI can get the current time by calling the `get_date` function.\n</think>\n"
        '<tool_call>\n{"name": "get_date", "arguments": {}}\n</tool_call>'
    )}
]}

template = get_template(get_processor('Qwen/Qwen3-8B'), loss_scale='hermes')
template.set_mode('train')
inputs = template.encode(data)

print(template.safe_decode(inputs['labels']))
print(inputs['loss_scale'])
```

## Metrics

The mapping file for metrics can be found [here](https://github.com/modelscope/ms-swift/blob/main/swift/metrics/mapping.py). This component is used in both ms-swift and Megatron-SWIFT.

- If used in ms-swift, you need to inherit the `EvalMetrics` base class from base.py and implement the `compute_metrics` function, returning a dictionary `Dict[str, float]`. You can refer to [NlgMetrics](https://github.com/modelscope/ms-swift/blob/0d7c9f5bc0e7e7d67d914ce6edeb9ce24f60746f/swift/metrics/nlg.py#L33) for customization.
- If used in Megatron-SWIFT, you need to inherit the `Metric` base class from utils.py and implement the `update` and `compute` methods. The compute method should return a dictionary `Dict[str, float]`.

You can customize metrics (currently only supports sft/pretrain/reranker/embedding tasks) and set `--eval_metric <metric-name>` during training to use your custom metrics.

## Optimizers

The mapping file for optimizers can be found [here](https://github.com/modelscope/ms-swift/blob/main/swift/optimizers/mapping.py). If you need to customize an optimizer, you need to inherit the `OptimizerCallback` base class and override the `create_optimizer` function. Use `--optimizer <optimizer-name>` during training to specify the custom optimizer.

- You can refer to [MultimodalOptimizerCallback](https://github.com/modelscope/ms-swift/blob/0d7c9f5bc0e7e7d67d914ce6edeb9ce24f60746f/swift/optimizers/multimodal.py#L43) for implementation. This class implements the functionality of vit_lr and aligner_lr, which uses different learning rates for vit, aligner, and LLM respectively.

## Tuner Plugin

The mapping file for Tuner plugins can be found [here](https://github.com/modelscope/ms-swift/blob/main/swift/tuner_plugin/mapping.py). If you need to customize a tuner, you need to inherit the `Tuner` base class and override the `prepare_model`, `save_pretrained`, `from_pretrained` functions.

- prepare_model: This function is called before training to process and prepare the original model, wrap it with the tuner, and set trainable parameters. For example: you can attach LoRA to certain layers and freeze certain layers.
- save_pretrained: This function is called during training to save the model.
- from_pretrained: This function is called during inference/resuming training to prepare the model and load weights.

You can refer to [LoRALLMTuner](https://github.com/modelscope/ms-swift/blob/0d7c9f5bc0e7e7d67d914ce6edeb9ce24f60746f/swift/tuner_plugin/lora_llm.py#L24) for implementation. This class implements the functionality of performing LoRA training on LLM and full parameter training on ViT.

## ORM

Examples can be found [here](https://github.com/modelscope/ms-swift/blob/main/swift/rewards/orm.py).

ORM is an Outcome Reward Model. ORM is generally implemented using regular expressions. ORM determines whether a response is correct. For example:

```python
class MathORM(ORM):

    @staticmethod
    def extract_boxed_result(text):
        pattern = r'\\boxed{([^}]*)}'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        else:
            return None

    def __call__(self, infer_requests: List[InferRequest], ground_truths: List[str],
                **kwargs) -> List[float]:
        rewards = []
        predictions = [request.messages[-1]['content'] for request in infer_requests]
        for prediction, ground_truth in zip(predictions, ground_truths):
            res1 = MathORM.extract_boxed_result(prediction) or ''
            res2 = MathORM.extract_boxed_result(ground_truth) or ''
            rewards.append(float(res1.strip() == res2.strip()))

        return rewards


orms = {
    'math': MathORM,
}
```

In the code above, we define a process for parsing mathematical responses. If the results are the same, it returns a score of 1.0, otherwise 0.0. Unlike PRM, this class has an additional parameter `ground_truths` in infer,
which contains the actual labels (standard responses defined in the dataset) of the corresponding infer_requests.

## PRM

Examples can be found [here](https://github.com/modelscope/ms-swift/blob/main/swift/rewards/prm.py).

PRM is a Process Reward Model, which will be used in the `swift sample` command. The interface that PRM needs to support is relatively simple:

```python
class PRM:

    def __init__(self):
        # init here
        pass

    def __call__(self, infer_requests: List[InferRequest], **kwargs) -> List[Union[float, List[float]]]:
        raise NotImplementedError
```

The InferRequest comes from `swift.infer_engine`, and the returned `List[Union[float, List[float]]]` can contain either a reward or multiple rewards. Developers can obtain queries and responses from infer_requests and split them according to their own methods. For example:

```text
Let's think step by step.

Step1: xxx

Step2: xxx

So, the answer is ...
```

## Introduction to Other Directory Structures

- arguments: Command-line parameter definitions, such as: `SftArguments`, `RLHFArguments`, etc.
- cli: Swift command-line mechanism and startup files. For example, `swift sft ...` is equivalent to `python swift/cli/main.py sft ...` and also equivalent to `python swift/cli/sft.py ...`.
- config: deepspeed/fsdp2 configuration files.
- dataloader: Implementation of dataloader, including shard/dispatcher methods.
- dataset: Dataset-related module implementation, including data preprocessing, packing, streaming data, etc. Registration of built-in datasets is in the `dataset/dataset` and `dataset/data` folders. For details, refer to [Custom Dataset Documentation](Custom-dataset.md).
- infer_engine: Inference engine implementation. Includes inference engine implementations with transformers/vllm/sglang/lmdeploy as backends.
- megatron: Megatron-SWIFT implementation.
- model: Model loading and registration. For details, refer to [Custom Model Documentation](Custom-model.md), [Multimodal Model Registration Best Practices](../BestPractices/MLLM-Registration.md).
- pipelines: Main function pipeline implementations for `swift sft/rlhf/infer`, etc., including `sft_main/rlhf_main/infer_main`, etc.
- rlhf_trainers: Trainer implementations for algorithms such as GRPO/GKD/DPO/KTO/RM.
- rollout: Sampling implementation of the rollout process in RL algorithms.
- rewards: Reward function implementation in RL algorithms, supporting custom reward calculation logic.
- template: Implementation and registration of dialogue templates, including the logic for converting messages to input_ids for various tasks, as well as data_collator-related logic. For details, refer to [Custom Model Documentation](Custom-model.md), [Multimodal Model Registration Best Practices](../BestPractices/MLLM-Registration.md).
- trainers: Trainer implementations for pretrain/SFT/Embedding/Reranker/sequence classification tasks.
- ui: `swift web-ui` interface training and inference implementation.
