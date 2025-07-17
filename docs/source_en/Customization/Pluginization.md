# Pluginization

Pluginization is a significant new feature introduced in SWIFT 3.0. We aim to make the customization of the development process more natural for developers through a plugin-based approach.

## Callback Mechanism

An example can be found [here](https://github.com/modelscope/ms-swift/blob/main/swift/plugin/callback.py).

The `callback` mechanism is a customization feature in the Transformers Trainer that allows developers to control the training process. Typically, customizing a callback looks like the following:

```python
class CustomCallback(TrainerCallback):

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Doing something when the training begins.
        pass

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # Doing something when saving a checkpoint.
        pass
```

Callbacks are registered with the trainer before it is instantiated. The example provided demonstrates a simple version of an EarlyStopping mechanism. Registering your own callback is straightforward:

```python
extra_callbacks = [CustomCallback()]
```

Developers can add new callbacks in `plugin/callback.py` and customize their training process. For detailed parameters of callbacks, refer to [this documentation](https://huggingface.co/docs/transformers/main_classes/callback).

## Customizing Loss

An example can be found [here](https://github.com/modelscope/ms-swift/blob/main/swift/plugin/loss.py).

SWIFT supports customizing the loss function through plugins. If this feature is not utilized, the default Cross Entropy Loss (CE Loss) is used. Developers can write code in this file to register their custom loss functions, and the trainer will automatically use the customized loss method.

For example, adding the following code in `plugin/loss.py`:

```python
@register_loss_func("custom_loss")
def loss_scale_func(outputs, labels, loss_scale=None, num_items_in_batch=None) -> torch.Tensor:
    # Write your own loss calculation here
    return loss
```

It is important to note that the loss function is strongly related to the training task. Currently, loss customization supports PT and SFT tasks. For human alignment tasks (e.g., DPO, PPO) or classification tasks (seq_cls), loss customization through plugins is not supported.

## Customizing Loss Scale

An example can be found [here](https://github.com/modelscope/ms-swift/blob/main/swift/plugin/loss_scale/loss_scale.py).

The `loss_scale` mechanism is one of the crucial features in SWIFT. In PT and SFT tasks, the loss for trainable tokens is uniform, meaning each token is equally involved in backpropagation. However, in certain situations, some tokens require higher weights and extra attention. In such cases, `loss_scale` allows developers to define custom token weights.

```python
class LastRoundLossScale(LossScale):

    def get_loss_scale(self, context: str, context_type: ContextType, is_last_round: bool, **kwargs):
        if context_type == ContextType.RESPONSE:
            return [context], [float(is_last_round)]
        return super().get_loss_scale(context, context_type, is_last_round)
```

In the above code, a `Tuple` is returned where the first element is the `context` (or its split parts), and the second element is the corresponding `loss_scale`. The float value represents the weight. For example, the following weight settings:

```text
["学习", "好", "数学", "是", "重要", "的"]
[1.0, 0.5, 2.0, 0.5, 2.0, 0.1]
```

Here, we place more emphasis on the words "数学" (mathematics) and "重要" (important) by increasing their weights to 2.0.

Referring back to the code, we check if the provided `context` is a response. If it is a response and is the last round in a multi-turn dialogue, we return a `loss_scale` of `[1]`. In other cases, we use the base implementation (which sets `loss_scale` to `[0]`). This approach ensures that only the responses from the last round participate in training, while other responses do not. Using this method, we can make all tokens (prompts and responses) participate in training or focus on specific special characters of the agent for training, etc.

In PT and SFT, `loss_scale` is uniformly supported (whether to participate in training and the size of the weights). However, in human alignment tasks, only the participation of certain tokens in training is supported, not the size of the weights.

## Customizing Metrics

An example can be found [here](https://github.com/modelscope/ms-swift/blob/main/swift/plugin/metric.py).

Metrics can be customized to evaluate the training process:

```python
METRIC_MAPPING = {
    'acc': (compute_acc_metrics, preprocess_logits_for_acc),
    'nlg': (compute_nlg_metrics, None),
    'custom': (custom_metric, custom_preprocess),
}

def get_metric(metric: str):
    return METRIC_MAPPING[metric]
```

In the above definition, we added a new `custom` metric. Its value consists of two parts: the first is the metric computation process, which returns a dictionary containing metric key-value pairs, and the second is the preprocessing step for logits, which returns the actual predictions.

## Customizing Optimizers

An example can be found [here](https://github.com/modelscope/ms-swift/blob/main/swift/plugin/optimizer.py).
- Apply different learning rates to different parts of the model. For example, use separate learning rates for ViT and LLM, as referenced [here](https://github.com/modelscope/ms-swift/blob/main/examples/train/multimodal/lora_llm_full_vit/custom_plugin.py).

Users can add their own optimizers and learning rate schedulers here:

```python
def create_custom_optimizers(args, model, dataset):
    # Create your own optimizer
    return CustomOptimizer(optimizer_grouped_parameters, **optimizer_kwargs), CustomScheduler(...)

optimizers_map = {
    'custom': create_custom_optimizers,
    ...
}
```

When developers need to use other optimizers, such as those defined in new research papers, they can define their creation process here and specify the parameter:

```shell
--optimizer custom
```

This will invoke the custom optimizer.


## Customizing Agent Template

The example is [here](https://github.com/modelscope/ms-swift/blob/main/swift/plugin/agent_template).

## Customizing Tuners

An example can be found [here](https://github.com/modelscope/ms-swift/blob/main/swift/plugin/tuner.py).
- For the multimodal model, full-parameter training is applied to the ViT part, while LoRA training is used for the LLM part. Refer to [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/multimodal/lora_llm_full_vit).
- For Phi4-multimodal, train its existing LoRA directly without adding extra LoRA. Refer to [here](https://github.com/modelscope/ms-swift/blob/main/examples/train/plugins/tuner_phi4_mm.sh).

Tuner customization is another unique feature of SWIFT. Developers can bypass the complex tuner initialization process and code integration costs by registering new tuners here:

```python
class IA3(Tuner):

    @staticmethod
    def prepare_model(args: 'TrainArguments', model: torch.nn.Module) -> torch.nn.Module:
        model_arch: ModelKeys = MODEL_ARCH_MAPPING[model.model_meta.model_arch]
        ia3_config = IA3Config(
            target_modules=find_all_linears(model), feedforward_modules='.*' + model_arch.mlp.split('{}.')[1] + '.*')
        return get_peft_model(model, ia3_config)

    @staticmethod
    def save_pretrained(
        model: torch.nn.Module,
        save_directory: str,
        state_dict: Optional[dict] = None,
        safe_serialization: bool = True,
        **kwargs,
    ) -> None:
        model: PeftModel
        model.save_pretrained(save_directory, state_dict=state_dict, safe_serialization=safe_serialization, **kwargs)

    @staticmethod
    def from_pretrained(model: torch.nn.Module, model_id: str, **kwargs) -> torch.nn.Module:
        return PeftModel.from_pretrained(model, model_id, **kwargs)
```

In the above example, we apply PEFT's IA3 to model training. This class includes three methods:

- `prepare_model`: How to wrap the original model using the tuner and set up trainable parameters.
- `save_pretrained`: How to save the model during training.
- `from_pretrained`: How to reload checkpoints saved earlier for subsequent training and inference.

These three methods are invoked during the SWIFT training process, allowing developers to use their tuners without reading the complex training code.

## PRM (Process Reward Model)

An example can be found [here](https://github.com/modelscope/ms-swift/blob/main/swift/plugin/prm.py).

PRM stands for Process Reward Model, which is used in the `swift sample` command. PRM needs to support simple interfaces:

```python
class PRM:

    def __init__(self):
        # init here
        pass

    def __call__(self, infer_requests: List[InferRequest], **kwargs) -> List[Union[float, List[float]]]:
        raise NotImplementedError
```

The InferRequest comes from `swift.llm`, and the returned `List[Union[float, List[float]]]` may contain a reward or several rewards. Developers can access queries and responses in infer_requests and split them according to their own methods, for example:

```text
Let's think step by step.

Step1: xxx

Step2: xxx

So, the answer is ...
```

Developers can split the process here, batch them into PRM for inference, and return rewards. More generally, developers can call a remote URL here, such as a closed-source PRM large model, and return rewards.

## ORM (Outcome Reward Model)

An example can be found [here](https://github.com/modelscope/ms-swift/blob/main/swift/plugin/orm.py).

ORM stands for Outcome Reward Model. ORM typically uses regular expressions to determine whether a response is correct. For example:

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

In the above code, we define a process to parse mathematical responses. If the results are the same, it returns a score of `1.0`; otherwise, it returns `0.0`. Unlike PRM, this class's `infer` method includes an additional parameter `ground_truths`, which corresponds to the actual labels (standard responses defined in the dataset) for the `infer_requests`.
