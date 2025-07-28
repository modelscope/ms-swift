# Reward Model

By default, a reward model refers to a model with a classification head that outputs numerical values, commonly known as an Output Reward Model (ORM). These models score the outputs of other models, generating a scalar value that represents the quality of the model's response.

We can load reward models with classification heads using the `reward_models` parameter, or load reward models trained via [Reward Modeling](../../human-alignment.md#rm) and use the model's logits as rewards.

## Custom Reward Model

Currently, we can flexibly customize the processing logic of reward models using the `reward_model_plugin`. This enables the implementation of techniques such as generative reward models, including:
- Custom model system prompts: Define specific instructions and contexts to guide the evaluation process.
- Handling model interaction history: Manage dialogue context to provide meaningful and context-aware evaluations.
- Defining custom evaluation criteria: Set unique standards and metrics for assessing model responses, going beyond default measures of accuracy and relevance.

With the `reward_model_plugin`, developers can tailor the reward evaluation process to the specific needs of their applications. This flexibility allows for more nuanced and effective reward-based training strategies.

The reward model is invoked via the plugin's `__call__` method, which takes `inputs` as a parameter, containing the model's input-output messages and other columns from the dataset.

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

```
For generative reward models, there are two common invocation methods: one is to run the reward model directly within the Trainer using PTEngine, and the other is to call an externally deployed model service.

- When deploying the reward model with PTEngine, the model is embedded inside the Trainer and does not require additional computational resources. The advantage of this method is easy integration, but the generation speed is relatively slow, making it more suitable for reward models with a small number of parameters.

- When deploying the reward model externally, you can use commands like swift deploy or vllm serve to deploy the model service on dedicated devices, which greatly improves inference speed and is more suitable for larger models with more parameters. However, this requires allocating additional hardware resources.

**Example 1: Using PTEngine for model inference**

Within a plugin, you can use PTEngine to directly run inference with the reward model. After constructing the messages as input, make a call via the infer interface:
```python
class RMlugin(DefaultRMPlugin):

    def __init__(self, model, template):

        super().__init__(model, template)
        # Initialize PTEngine for inference
        self.engine = PtEngine.from_model_template(self.model, self.template, max_batch_size=0)

    def __call__(self, inputs):
        system_prompt = ...
        query = ...
        messages = [{'role': 'system', 'content': system_prompt}, {'role': 'query', 'content': query}]
        result = self.engine.infer([messages], self.request_config, use_tqdm=False)
        rewards = ...
        return rewards

```

**Example 2: Deploying the reward model with swift deploy and making remote calls**

With this method, you do not need to use the reward_model_plugin, and can simply make calls directly within the reward function.

First, start the model service using the following command:
```bash
# Note: Do not use the same devices for deployment and training
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

In the reward function, use the OpenAI library to initialize the client, specify the address and port of the model service, and call the model as shown below:

```python
from openai import OpenAI

class RMReward(ORM):

    def __init__(self):
        super().__init__()
        try:
            self.client = OpenAI(
                api_key='EMPTY',
                base_url='http://127.0.0.1:8000/v1', # Use 127.0.0.1 if deployed locally
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
            reward = ... # Extract the reward value from the reward model output
            rewards.append(reward)
        return rewards
```


We provide a simple generative reward model example (GenRMPlugin) in [rm_plugin.py](https://github.com/modelscope/ms-swift/blob/main/swift/plugin/rm_plugin.py).

Customize the reward model plugin in [plugin.py](https://github.com/modelscope/ms-swift/blob/main/examples/train/grpo/plugin/plugin.py) and register it using the `external_plugins` parameter.

Here is an example training script for GRPO training using two reward models, including an ORM and a Gen-RM (here using qwen2.5-3B-Instruct):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift rlhf \
    --rlhf_type grpo \
    --model Qwen/Qwen2.5-7B \
    --dataset AI-MO/NuminaMath-TIR#5000 \
    --external_plugins examples/train/grpo/plugin/plugin.py \
    --reward_funcs format \
    --reward_model Qwen/Qwen2.5-3B-Instruct Shanghai_AI_Laboratory/internlm2-7b-reward \
    --reward_model_plugin genrm my_rmplugin \
    --reward_weights 0.1 1 1 \
    --vllm_gpu_memory_utilization 0.5 \
    --sleep_level 1 \
    --offload_model true \
    --offload_optimizer true \
    --log_completions true \
    --deepspeed zero2
```

Note:
1. In GRPOTrainer, the `reward_model` will be sequentially appended to `reward_funcs`. Therefore, the order of `reward_weights` corresponds to [reward_funcs, reward_model].
2. The default `reward_model_plugin` is set to "default," meaning it uses ORM processing logic.
3. For models with large parameter sizes, PTEngine generation may be slow. In such cases, the model can be deployed externally and called within `reward_funcs`.

For models like BERT that cannot be loaded via `reward_model`, we can embed them within `reward_function` for loading. Refer to [this issue](https://github.com/modelscope/ms-swift/issues/4580) for details.
