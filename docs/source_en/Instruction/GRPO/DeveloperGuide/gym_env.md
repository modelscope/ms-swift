# GYM Environment Training

Note: The GYM environment training logic was refactored in ms-swift 3.8. If you are using an earlier version of ms-swift, please refer to the documentation for that specific version.

## Gym Interface

GYM originates from [OpenAI Gym](https://github.com/openai/gym) and is an abstract interface for reinforcement learning environments. Based on the current "Model as Agent" trend, we have defined a similar interface in swift to provide end-to-end reinforcement learning training for Agents.
```python
class Env(ABC):

    def __init__(self, env_config):
        """

        Args:
            env_config: Environment configuration, such as available tools, etc.
        """
        self.env_config = env_config

    @abstractmethod
    async def reset(self, config: RolloutInferRequest) -> Tuple[str, Dict[str, Any], str]:
        """

        Args:
            config: Environment initialization information.

        Returns:
            - observation: The first user message as the initial observation or environment information, which will be treated as a user message.
            - info: Extra information for DEBUG and logging, which will be recorded in completions.jsonl.
            - system_message: The system prompt sampled for the user's current environment.
        """
        pass

    @abstractmethod
    async def step(self, action: Messages) -> Tuple[str, float, bool, Dict[str, Any]]:
        """

        Args:
            action: All dialogue messages, with the last message being the current sampled response.

        Returns:
            - next_observation: The environment's response, which will be returned as a user message.
            - reward: The reward.
            - done: Whether the episode has finished.
            - info: Extra information for DEBUG and logging, which will be recorded in completions.jsonl.
        """
        pass
    @abstractmethod
    async def close(self):
        """Clean up environment resources."""
        pass
```
Additionally, based on the practices of [Kimi-Researcher](https://moonshotai.github.io/Kimi-Researcher/), we also provide an extra `ContextManager` interface to help you dynamically manage the current Agent's context.

**Specifying the ContextManager (Optional)**
1. In the dataset, specify it using the `name` key in the [`ctx_config`](#Notes) column. Place related initialization parameters in other keys.
2. Use the parameter `--context_manager ctx_name` to specify it.


```python
class ContextManager(ABC):
    def __init__(self,ctx_config):
        self.ctx_config = ctx_config

    @abstractmethod
    def manage_context(self, history: Messages,trajectory_id:str) -> Messages:
        """Dynamically adjusts the current agent's context.

        Args:
            history: The current message history.

        Returns:
            The adjusted message history.
        """
        pass
```

Input Parameter Example

```python
infer_request
"""
RolloutInferRequest(
    messages=[
        {'role': 'system', 'content': 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>\n'}, {'role': 'user', 'content': 'What is the value of $\\sqrt{36 \\times \\sqrt{16}}$?'},
        {'role': 'assistant', 'content': 'To find the value of \\(\\sqrt{36 \\times \\sqrt{16}}\\), we will break down the problem step-by-step.\n\nFirst, we need to evaluate the inner square root:\n\\[\n\\sqrt{16}\n\\]\nWe know that:\n\\[\n4^2 = 16 \\implies \\sqrt{16} = 4\n\\]\n\nNext, we substitute this result back into the original expression:\n\\[\n\\sqrt{36 \\times \\sqrt{16}} = \\sqrt{36 \\times 4}\n\\]\n\nNow, we need to evaluate the product inside the square root:\n\\[\n36 \\times 4 = 144\n\\]\n\nSo, the expression simplifies to:\n\\[\n\\sqrt{144}\n\\]\n\nFinally, we determine the square root of 144:\n\\[\n\\sqrt{144} = 12\n\\]\n\nThus, the value of \\(\\sqrt{36 \\times \\sqrt{16}}\\) is:\n\\[\n\\boxed{12}\n\\]'}
    ],
    images=[],
    audios=[],
    videos=[],
    tools=None,
    objects={},
    data_dict={
        'problem': 'What is the value of $\\sqrt{36 \\times \\sqrt{16}}$?',
        'solution': "To solve the problem, we need to evaluate the expression \\(\\sqrt{36 \\times \\sqrt{16}}\\).\n\nWe can break down the steps as follows:\n\n1. Evaluate the inner square root: \\(\\sqrt{16}\\).\n2. Multiply the result by 36.\n3. Take the square root of the product obtained in step 2.\n\nLet's compute this step by step using Python code for accuracy.\n```python\nimport math\n\n# Step 1: Evaluate the inner square root\ninner_sqrt = math.sqrt(16)\n\n# Step 2: Multiply the result by 36\nproduct = 36 * inner_sqrt\n\n# Step 3: Take the square root of the product\nfinal_result = math.sqrt(product)\nprint(final_result)\n```\n```output\n12.0\n```\nThe value of \\(\\sqrt{36 \\times \\sqrt{16}}\\) is /\\(\\boxed{12}\\)."
        }
    )
"""
result
"""
RolloutResponseChoice(
    index=0,
    message=ChatMessage(
        role='assistant',
        content='To find the value of \\(\\sqrt{36 \\times \\sqrt{16}}\\), we will break down the problem step-by-step.\n\nFirst, we need to evaluate the inner square root:\n\\[\n\\sqrt{16}\n\\]\nWe know that:\n\\[\n4^2 = 16 \\implies \\sqrt{16} = 4\n\\]\n\nNext, we substitute this result back into the original expression:\n\\[\n\\sqrt{36 \\times \\sqrt{16}} = \\sqrt{36 \\times 4}\n\\]\n\nNow, we need to evaluate the product inside the square root:\n\\[\n36 \\times 4 = 144\n\\]\n\nSo, the expression simplifies to:\n\\[\n\\sqrt{144}\n\\]\n\nFinally, we determine the square root of 144:\n\\[\n\\sqrt{144} = 12\n\\]\n\nThus, the value of \\(\\sqrt{36 \\times \\sqrt{16}}\\) is:\n\\[\n\\boxed{12}\n\\]', tool_calls=None),
        finish_reason='stop',
        logprobs=None,
        messages=None)
"""
```

Training with a GYM environment can be considered a special form of multi-turn training, the difference being that reward signals are obtained directly from the environment.

To enable this mode, add the use_gym_env argument to the rollout command, which instructs the system to use GYM as the training environment interface.
We also provide a multi-turn planner example compatible with GYM; see the GymScheduler class in the [built-in multi-turn scheduler implementation](https://github.com/modelscope/ms-swift/blob/main/swift/plugin/multi_turn.py)

```bash
CUDA_VISIBLE_DEVICES=0 \
swift rollout \
    --model xxx \
    --use_gym_env true \
    --multi_turn_scheduler gym_scheduler \
    --max_turns xxx
```


**Environment Selection**
1. In the dataset, you need to specify it using the `name` key in the [`env_config`](#Notes) column. Place related initialization parameters in other keys.
2. Use the parameter `--gym_env env_name` to specify it.


## Best Practices

- [Training Script](../../../../../examples/train/grpo/external/vllm_gym.sh)

Using the `external_plugins` parameter, we can register local `Env` and `ContextManager` classes into ms-swift. For the specific implementation, refer to the [code](https://github.com/modelscope/ms-swift/blob/main/examples/train/grpo/plugin/plugin.py).

## Notes

1. Reference Training Data Format
```json
{"messages": [{"role": "system", "content": "You are a helpful and harmless assistant"}, {"role": "user", "content": "Tell me tomorrow's weather"}],"env_config":{"name":"custom_env","other_config":"xxxx"},"ctx_config":{"name":"custom_ctx","other_config":"xxxx"}}
```

2. By default, only the response from the last round is used for training. If the gym involves generating multi-turn responses, use the parameter `--loss_scale default` to train on the responses from all rounds. For more details, please refer to the [documentation](./multi_turn.md#loss-masking).

3. Data Flow
The entire gym data flow is as follows:
<img src="../../../../resources/gym_env.png" width="400" />

4. Reward Logging
Since the gym reward is calculated within the `step` function, you need to manually return the log via `info`. The final record will be placed in the `trajectory_infos` field of `completions.jsonl`.
