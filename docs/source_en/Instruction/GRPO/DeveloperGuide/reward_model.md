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

For generative reward models, it is recommended to use PTEngine for model inference. After constructing the model's input messages, use the `infer` interface for inference.
```python
class RMlugin(DefaultRMPlugin):

    def __init__(self, model, template):

        super().__init__(model, template)
        # initilize PTEngine to infer
        self.engine = PtEngine.from_model_template(self.model, self.template, max_batch_size=0)

        ...
        messages = [{'role': 'system', 'content': 'system prompt'}, {'role': 'query', 'content': 'query'}]
        result = self.engine.infer([messages], self.request_config, use_tqdm=False)
        print(result.message.content)
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
