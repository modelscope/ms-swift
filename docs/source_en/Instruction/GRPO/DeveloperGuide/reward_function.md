# Reward Function
## Custom Reward Function
The reward function takes the model-generated text `completions` and other columns from the dataset as parameters (`kwargs`) and scores the model-generated text. Below is an example demonstrating how to implement a simple length-based reward function. This function assigns a reward signal of 1.0 if the length of the model-generated text exceeds 1024; otherwise, the reward signal is 0.0.

```python
from swift.plugin import ORM, orms
class DummyLengthRewardFunction(ORM)
    def __call__(completions, **kwargs):
        return [1.0 if len(completion) > 1024 else 0.0 for completion in completions]

orms['dummy']= DummyLengthRewardFunction
```

**Accessing Other Columns in the Dataset**

For example, if the reward function needs to access the solution column from the dataset for auxiliary calculations, here are two ways to achieve this:

Explicitly define the solution column name in the __call__ parameters:
```python
    def __call__(completions, solution, **kwargs):
        print(solution)
        ...
```

Retrieve it from kwargs:
```python
    def __call__(completions, **kwargs):
        solution = kwargs.get('solution')
        ...
```

Note: Columns related to messages (e.g., query, response) will be processed, and the original assistant responses in the dataset will be discarded. Use additional columns to retain such information.

**Using Custom Reward Functions**

You can add the reward function in [plugin program](https://github.com/modelscope/ms-swift/blob/main/examples/train/grpo/plugin/plugin.py), register it using the parameter `--external_plugins examples/train/grpo/plugin/plugin.py`, and specify it via the `reward_funcs` parameter.

For execution scripts, refer to [here](https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo/plugin/run_external_reward_func.sh).

## Built-in Reward Functions
Swift includes five rule-based reward functions (code can be found in swift/plugin/orm.py).

| Reward Function | Paper |
|----------------|----------------------------------------------------------------------------|
| accuracy       | [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL](https://arxiv.org/abs/2501.12948) |
| format         | Same as above |
| cosine         | [Demystifying Long Chain-of-Thought Reasoning in LLMs](https://arxiv.org/abs/2502.03373) |
| repetition     | Same as above |
| soft_overlong  | [Decoupled Clip and Dynamic sAmpling Policy Optimization (DAPO)](https://arxiv.org/abs/2503.14476) |

### 1. **accuracy**

This function compares the model's generated output with the solution column in the dataset to calculate an accuracy score. If the generated output matches the reference answer, the score is 1.0; otherwise, it is 0.0.

Note: This reward function uses the `math_verify` library to parse the generated output and the solution, which may only be applicable to specific mathematical datasets.

### 2. **format**

The paper uses the following system prompt to require the model to return responses in a fixed format:
```
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>
```

This function checks whether the model generates text in the format `<think>think content</think><answer>answer content</answer>`. If the generated text meets the format requirements, the score is 1.0; otherwise, it is 0.0.

### 3. **cosine**

The paper found that using only the accuracy reward function for training could lead to excessively long generated outputs, thereby affecting training effectiveness. The cosine reward function optimizes the training process by controlling the length of the model's outputs:

- For texts with correct answers, the reward decreases as the length increases, encouraging the model to generate concise responses.
- For texts with incorrect answers, the reward increases as the length increases, encouraging the model to think more deeply.

A cosine function is used to smoothly adjust the reward value, ensuring the changes remain within a reasonable range. The parameters of the cosine function include the length of the generated text, the maximum length limit, and the minimum and maximum reward values.

Parameters:
- cosine_min_len_value_wrong (default: -0.5): The reward value for the minimum length when the answer is incorrect.
- cosine_max_len_value_wrong (default: 0.0): The reward value for the maximum length when the answer is incorrect.
- cosine_min_len_value_correct (default: 1.0): The reward value for the minimum length when the answer is correct.
- cosine_max_len_value_correct (default: 0.5): The reward value for the maximum length when the answer is correct.
- cosine_max_len (default equals the model's maximum generation length): The maximum length limit for the generated text.

### 4. **repetition**

Penalizes repetitive content in the model's generated text by detecting repeated n-gram patterns and applying corresponding penalties.

The function splits the generated text into words and extracts n-grams of a specified size (default: 3-grams). By calculating the ratio of unique n-grams to the total number of n-grams, it determines the repetition rate. If the repetition rate is high, a larger negative reward (penalty) is applied. The penalty value is calculated based on the repetition rate and the maximum penalty value (default: -1.0).

Parameters:
- repetition_n_grams (default: 3): The size of n-grams used to detect repetition.
- repetition_max_penalty (default: -1.0): The maximum penalty value, controlling the penalty strength.

### 5. **soft overlong punishment**
Defines a length penalty interval. Within this interval, a linear penalty in the range [-1, 0] is applied.

Parameters:
- soft_max_length: L_max in the paper, the model's maximum generation length, defaulting to max_completion_length.
- soft_cache_length: L_cache in the paper, controlling the length penalty interval, which is [soft_max_length - soft_cache_length, soft_max_length].

## Notes

If a model needs to be loaded in the reward function, the training DeepSpeed plugin (transformers logic) will be used by default. Under Zero3, this may cause the model to fail to perform inference properly. Refer to this [issue](https://github.com/modelscope/ms-swift/issues/4580) to skip the DeepSpeed initialization environment.
