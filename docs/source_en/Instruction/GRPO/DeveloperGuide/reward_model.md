# Reward Model

By default, a reward model refers to a model with a classification head that outputs numeric values, usually called an Output Reward Model (ORM). These models score the outputs from other models and produce a scalar value representing the quality of the model response.

You can load reward models with a classification head using the `reward_models` parameter, or load reward models trained by [reward modeling](../../RLHF.md#rm), and then use the model's logits as rewards.

## Custom Reward Models

For generative reward models, there are two common ways to use them: one is by directly defining the reward model logic inside the Trainer via the `reward_model_plugin`, and then using PTEngine for inference; the other is to call an externally deployed model service.

- Using `reward_model_plugin`, the reward model will be embedded within the Trainer and does not require additional computational resources. The advantage of this approach is ease of integration, but generation speed is relatively slow, making it more suitable for small-parameter reward models.
- When deploying reward models externally, you can use commands like `swift deploy` or `vllm serve` to deploy the model service on an independent device to greatly improve inference speed, which is more suitable for large models. However, this approach requires reserving extra hardware resources.

### Internal Plugin

You can flexibly customize the reward model processing logic inside `reward_model_plugin`. This enables implementations such as generative reward models, including:

- Custom model system prompts: define specific instructions and context to guide the evaluation process.
- Handling model interaction history: manage dialog context to allow meaningful and context-aware evaluation.
- Defining custom evaluation metrics: set unique criteria and measures for response evaluation, beyond the default accuracy and relevance checks.

With `reward_model_plugin`, developers can tailor the reward evaluation process for specific application needs. This flexibility allows for more fine-grained and effective reward-based training strategies.

The reward model is called via the plugin's `__call__` method, which takes `inputs` as a parameter. `inputs` contains the messages of model input/output and other columns from the dataset.

```python
    def __call__(self, inputs):
        print(inputs)
        """
[
    {
        'messages': [
                {'role': 'system', 'content': 'system prompt'},
                {'role': 'query', 'content': 'query'},
                {'role': 'user', 'content': 'completions1'},
            ],
        'solution': "abc",
    },
    {
        'messages': [
                {'role': 'system', 'content': 'system prompt'},
                {'role': 'query', 'content': 'query'},
                {'role': 'user', 'content': 'completions2'},
            ],
        'solution': "abc",
    }
]
        """
```

When using PTEngine in the plugin for reward model inference, you only need to construct messages and call the infer interface:

```python
class RMPlugin(DefaultRMPlugin):

    def __init__(self, model, template):

        super().__init__(model, template)
        # initilize PTEngine to infer
        self.engine = PtEngine.from_model_template(self.model, self.template, max_batch_size=0)

    def __call__(self, inputs):
        system_prompt = ...
        query = ...
        messages = [{'role': 'system', 'content': system_prompt}, {'role': 'query', 'content': query}]
        result = self.engine.infer([messages], self.request_config, use_tqdm=False)
        rewards = ...
        return rewards
```

We provide a simple example of a generative reward model (`GenRMPlugin`) in [rm_plugin.py](https://github.com/modelscope/ms-swift/blob/main/swift/plugin/rm_plugin.py).

You can customize your reward model plugin in [plugin.py](https://github.com/modelscope/ms-swift/blob/main/examples/train/grpo/plugin/plugin.py) and register it using the `external_plugins` parameter.

Note:
1. In `GRPOTrainer`, the reward_model will be appended to reward_funcs one by one. Therefore, the order of `reward_weights` corresponds to `[reward_funcs, reward_model]`.
2. The default for `reward_model_plugin` is `default`, which uses ORM logic.
3. For models with a large number of parameters, PTEngine generation is slow. Please use [external deployment](#external-deployment).

For models like BERT that cannot be loaded by `reward_model`, you can load them inside `reward_function`, see [issue](https://github.com/modelscope/ms-swift/issues/4580).

### External Deployment

This approach does not require the `reward_model_plugin` and can be called directly in the reward function.

First, use the following command to start the model service:

```bash
# Note: Do not overlap deployment devices with training devices
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift deploy \
    --model Qwen/Qwen2.5-72B-Instruct \
    --vllm_tensor_parallel_size 4

# [INFO:swift] model_list: ['Qwen2.5-72B-Instruct']
# INFO:     Started server process [xxxxxx]
# INFO:     Waiting for application startup.
# INFO:     Application startup complete.
# INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

In the reward function, initialize the client using the OpenAI library and specify the address and port of the model service. Example:

```python
from openai import OpenAI

class RMReward(ORM):

    def __init__(self):
        super().__init__()
        try:
            self.client = OpenAI(
                api_key='EMPTY',
                base_url='http://127.0.0.1:8000/v1', # 127.0.0.1 if deployed locally
            )
            self.verify_model_name = self.client.models.list().data[0].id
        except Exception as e:
            raise RuntimeError('Failed to connect to the model service. Please deploy the model '
                               "using 'swift deploy' or 'vllm serve'.") from e

    def __call__(self, completions, messages, **kwargs) -> List[float]:
        rewards = []
        for completion, message in zip(completions, messages):
            rm_prompt = ... # Construct the prompt for the reward model
            chat_response = self.client.chat.completions.create(
                model=self.verify_model_name,
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a helpful assistant.'
                    },
                    {
                        'role': 'user',
                        'content': rm_prompt
                    },
                ],
            )
            response = chat_response.choices[0].message.content.strip()
            reward = ... # Extract the reward value from the result
            rewards.append(reward)
        return rewards
```
