# 架构介绍

ms-swift 4.0 采用模块化设计，各功能模块分布在一级目录下，便于开发者进行自定义扩展。本文档将详细介绍各模块的功能及自定义方法。

## Agent Template

agent模板的mapping文件可以参考[这里](https://github.com/modelscope/ms-swift/blob/main/swift/agent_template/mapping.py)。agent template设计目标是，基于统一的Agent数据集格式，可以灵活切换不同模型进行训练，无需修改数据。训练时使用`--agent_template`指定对应的agent模板。

所有的AgentTemplate需要继承自`BaseAgentTemplate`，并实现其中的几个方法: `_format_tools`, `_format_tool_calls`, `_format_tool_responses`, `get_toolcall`。
- _format_tools: 将`tools`和`system`格式化，组成完整的system。
- _format_tool_calls: 将tool_call部分 `[{"role": "tool_call", "content": "..."}, {"role": "tool_call", "content": "..."}]`进行格式化，最后返回字符串。
- _format_tool_responses: 对tool（也称为tool_response）部分 `[{"role": "tool", "content": "..."}, {"role": "tool", "content": "..."}]`进行格式化。
- get_toolcall: 在部署的时候使用，用于解析模型输出内容中的工具名和参数，返回`List[Function]`。


如何debug：
```python
data = {"tools": "[{\"type\": \"function\", \"function\": {\"name\": \"realtime_aqi\", \"description\": \"天气预报。获取实时空气质量。当前空气质量，PM2.5，PM10信息\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\", \"description\": \"城市名，例如：上海\"}}, \"required\": [\"city\"]}}}]", "messages": [{"role": "user", "content": "北京和上海今天的天气情况"}, {"role": "tool_call", "content": "{\"name\": \"realtime_aqi\", \"arguments\": {\"city\": \"北京\"}}"}, {"role": "tool_call", "content": "{\"name\": \"realtime_aqi\", \"arguments\": {\"city\": \"上海\"}}"}, {"role": "tool_response", "content": "{\"city\": \"北京\", \"aqi\": \"10\", \"unit\": \"celsius\"}"}, {"role": "tool_response", "content": "{\"city\": \"上海\", \"aqi\": \"72\", \"unit\": \"fahrenheit\"}"}, {"role": "assistant", "content": "根据天气预报工具，北京今天的空气质量指数为10，属于良好水平；上海今天的空气质量指数为72，属于轻度污染水平。"}]}


from swift import get_processor, get_template

tokenizer = get_processor('Qwen/Qwen3.5-2B')
template = get_template(tokenizer)  # 使用默认agent模板
# template = get_template(tokenizer, agent_template='qwen3_5')
print(f'agent_template: {template._agent_template}')
template.set_mode('train')
encoded = template.encode(data)
print(f'[INPUT_IDS] {template.safe_decode(encoded["input_ids"])}\n')
print(f'[LABELS] {template.safe_decode(encoded["labels"])}')
```

如果你想要给我们提供PR，请参考[这里](https://github.com/modelscope/ms-swift/blob/main/tests/test_align/test_template/test_agent.py)书写你的测试案例。

## Callbacks

callbacks的mapping文件可以参考[这里](https://github.com/modelscope/ms-swift/blob/main/swift/callbacks/mapping.py)。callbacks可以对trainer中的关键节点的行为进行自定义。自定义后，你需要在mapping中进行注册，训练时使用`--callbacks`指定对应的回调类。例如，你可以自定义：

```python
class CustomCallback(TrainerCallback):

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Doing something when the training begins.
        pass

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Doing something when save checkpoint
        pass
```

所有的回调类需继承自base.py中的`TrainerCallback`，并覆盖其方法。接口与transformers的`TrainerCallback`一致，请参考transformers的[callback文档](https://huggingface.co/docs/transformers/main_classes/callback)。


## Loss

Loss的mapping文件可以参考[这里](https://github.com/modelscope/ms-swift/blob/main/swift/loss/mapping.py)。
swift支持自定义loss（当前只支持sft/pretrain/reranker/embedding任务），注册后在训练时设置`--loss_type <loss-name>`使用你定制的loss方法。

自定义Loss需继承自`BaseLoss`，并实现`__call__`方法，返回标量Tensor。你可以参考[CustomCrossEntropyLoss](https://github.com/modelscope/ms-swift/blob/0d7c9f5bc0e7e7d67d914ce6edeb9ce24f60746f/swift/loss/causal_lm.py#L5)进行定制。例如：

```python
class CustomLoss(BaseLoss):

    def __call__(self, outputs, labels, **kwargs) -> torch.Tensor:
        pass
```

## Loss Scale

loss scale的mapping文件可以参考[这里](https://github.com/modelscope/ms-swift/blob/main/swift/loss_scale/mapping.py)。在pretrain和sft任务中，可训练token的loss是平均的，即每个token平等地对待。但在某些情况下，某些token需要被额外关注，并设置更高的权重或者对某些token不进行训练。loss_scale可以让开发者自由地定义自己的token权重。（预训练和SFT支持使用loss_scale控制token是否参与训练以及和其权重大小，RLHF中只支持控制token是否参与训练）

你可以通过继承LossScale基类，并实现`get_loss_scale`方法来自定义loss scale。
```python
class CustomLossScale(LossScale):

    def get_loss_scale(self, context: str, **kwargs) -> Tuple[List[str], List[float]]:
        ...
```
`get_loss_scale`函数需要返回了一个Tuple，第一个返回是拆解后的字符串的列表，第二个参数是字符串对应的loss_scale的列表，float值代表了权重。例如下面的权重设置：
```text
["学习", "好", "数学", "是", "重要", "的"]
[1.0, 0.5, 2.0, 0.5, 2.0, 0.1]
```
例子中，我们更看重数学和重要两个词，因为其loss_scale为2.0。


当然我们也需要关注`__call__`方法的核心逻辑，即loss_scale基本策略（base_strategy）all/default/last_round 对loss_scale的影响，具体参考[命令行参数文档](../Instruction/Command-line-parameters.md)的介绍。以及数据集中的'loss'字段对loss_scale的影响，参考[自定义数据集文档](../Customization/Custom-dataset.md)。
```python
if loss or loss is None and (self.base_strategy == 'all' or
                            (self.base_strategy == 'default' and is_assistant) or
                            (self.base_strategy == 'last_round' and is_assistant and is_last_round)):
    new_context, loss_scale = self.get_loss_scale(context, query=query)
else:
    new_context, loss_scale = [context], [0.]
```

此外你也可以使用[json配置文件](https://github.com/modelscope/ms-swift/tree/main/swift/loss_scale/config)，继承内置的ConfigLossScale类，来自定义loss_scale。目前支持两种配置方式：字符串精确匹配和正则表达式匹配。你可以参考[Agent支持文档](../Instruction/Agent-support.md#loss_scale的使用)的内容进行理解。

- 字符串精确匹配，例如参考`react.json`, `qwen.json`。json中需要书写`Dict[str, List[float]]`的映射。字符串代表关键词，列表中需要有两个值。我们会根据关键词，将字符串切分成多段字符串。列表的第一个值代表关键词的权重，列表的第二个值代表该关键值后，下一关键词前的内容的权重。

- 正则表达式匹配，例如参考`ignore_empty_think.json`, `hermes.json`。json中需要书写`Dict[str, float]`的映射。字符串代表正则表达式pattern，浮点数代表匹配字符串的权重。


如何debug：
```python
from swift import get_processor, get_template

data = {"messages": [
    {"role": "user", "content": "今天的日期是多少？"},
    {"role": "assistant", "content": (
        "<think>\n我可以通过调用`get_date`函数来获取当前时间。\n</think>\n"
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

metrics的mapping文件可以参考[这里](https://github.com/modelscope/ms-swift/blob/main/swift/metrics/mapping.py)。该组件在ms-swift/Megatron-SWIFT中都有被使用。
- 如果是在ms-swift中被使用，你需要继承 base.py 中`EvalMetrics`基类，并实现`compute_metrics`函数，返回字典`Dict[str, float]`。你可以参考[NlgMetrics](https://github.com/modelscope/ms-swift/blob/0d7c9f5bc0e7e7d67d914ce6edeb9ce24f60746f/swift/metrics/nlg.py#L33)进行定制。
- 如果是在Megatron-SWIFT中被使用，你需要继承 utils.py 中`Metric`基类，并实现`update`和`compute`方法，compute方法需返回字典`Dict[str, float]`。

你可以自定义metrics（当前只支持sft/pretrain/reranker/embedding任务），在训练时设置`--eval_metric <metric-name>`使用你定制的metrics。

## Optimizers

optimizer的mapping文件可以参考[这里](https://github.com/modelscope/ms-swift/blob/main/swift/optimizers/mapping.py)。如果你需要自定义优化器，你需要继承`OptimizerCallback`基类，并覆盖`create_optimizer`函数。训练时使用`--optimizer <optimizer-name>`指定自定义的优化器。
- 你可以参考[MultimodalOptimizerCallback](https://github.com/modelscope/ms-swift/blob/0d7c9f5bc0e7e7d67d914ce6edeb9ce24f60746f/swift/optimizers/multimodal.py#L43)进行实现，该类实现了vit_lr, aligner_lr的功能，即对vit, aligner和LLM分别使用不同的学习率。



## Tuner Plugin

Tuner插件的mapping文件可以参考[这里](https://github.com/modelscope/ms-swift/blob/main/swift/tuner_plugin/mapping.py)。如果你需要自定义tuner，你需要继承`Tuner`基类，并覆盖`prepare_model`, `save_pretrained`, `from_pretrained`函数。
- prepare_model: 该函数在训练前被调用，将原始模型进行处理与准备，使用tuner封装，并设置可训练参数。例如：你可以对某些层附加LoRA，对某些层进行冻结等。
- save_pretrained: 该函数在训练中被调用，对模型进行保存。
- from_pretrained: 该函数在推理/断点续训时被调用，准备模型并读取权重。

你可以参考[LoRALLMTuner](https://github.com/modelscope/ms-swift/blob/0d7c9f5bc0e7e7d67d914ce6edeb9ce24f60746f/swift/tuner_plugin/lora_llm.py#L24)进行实现，该类实现了对LLM进行LoRA训练，对ViT进行全参数训练的功能。


## ORM

example参考[这里](https://github.com/modelscope/ms-swift/blob/main/swift/rewards/orm.py)。

ORM是结果奖励模型。ORM一般使用正则表达式来进行，ORM决定了response是否是正确的。例如：

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

在上面的代码中，我们定义了一个对数学response进行解析的过程，如果结果相同则返回score为1.0，否则为0.0。和PRM不同，这个类的infer中有一个额外参数`ground_truths`，
该参数是对应的infer_requests的实际label（数据集中定义的标准response）。


## PRM

example参考[这里](https://github.com/modelscope/ms-swift/blob/main/swift/rewards/prm.py)。

PRM是过程奖励模型，PRM会在`swift sample`命令中使用。PRM需要支持的接口比较简单：
```python
class PRM:

    def __init__(self):
        # init here
        pass

    def __call__(self, infer_requests: List[InferRequest], **kwargs) -> List[Union[float, List[float]]]:
        raise NotImplementedError
```

其中的InferRequest来自于`swift.infer_engine`，返回的`List[Union[float, List[float]]]`，列表中可能是reward也可能是若干reward。开发者可以在infer_requests中拿到queries和responses，并按照自己的方式进行切分，例如：
```text
Let's think step by step.

Step1: xxx

Step2: xxx

So, the answer is ...
```
开发者可以在这里对过程进行切分，并按batch传入PRM中进行推理并返回rewards。更通用来说，开发者可以在这里调用一个远端URL，例如一个闭源PRM大模型并返回rewards。


## 其他目录结构介绍

- arguments: 命令行参数定义，例如：`SftArguments`, `RLHFArguments`等。
- cli: swift命令行机制以及启动文件。例如`swift sft ...`等价于`python swift/cli/main.py sft ...`也等价于`python swift/cli/sft.py ...`。
- config: deepspeed/fsdp2配置文件。
- dataloader: dataloader的实现，包括shard/dispatcher两种方式。
- dataset: 数据集相关模块实现，包括数据预处理、packing、流式数据等。内置数据集的注册在`dataset/dataset`和`dataset/data`文件夹内。具体参考[自定义数据集文档](Custom-dataset.md)。
- infer_engine: 推理引擎实现。包括transformers/vllm/sglang/lmdeploy为后端的推理引擎实现。
- megatron: Megatron-SWIFT 实现。
- model: 模型加载与注册。具体参考[自定义模型文档](Custom-model.md)，[多模态模型注册最佳实践](../BestPractices/MLLM-Registration.md)。
- pipelines: `swift sft/rlhf/infer`等主函数pipeline实现，包括`sft_main/rlhf_main/infer_main`等。
- rlhf_trainers: GRPO/GKD/DPO/KTO/RM等算法的Trainer实现。
- rollout: RL算法中rollout过程的采样实现。
- rewards: RL算法中的奖励函数实现，支持自定义奖励计算逻辑。
- template: 对话模板的实现与注册，包含各个任务将messages转换成input_ids的逻辑，以及data_collator相关逻辑。具体参考[自定义模型文档](Custom-model.md)，[多模态模型注册最佳实践](../BestPractices/MLLM-Registration.md)。
- trainers: 预训练/SFT/Embedding/Reranker/序列分类任务的Trainer实现。
- ui: `swift web-ui`界面训练与推理实现。
