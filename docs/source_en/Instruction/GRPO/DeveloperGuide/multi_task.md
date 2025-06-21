# Multi-Task Training
We can add a column to the dataset that indicates the task type, and then use this information in the reward function or reward model plugin to determine which task is being processed. This allows us to implement multi-task training. For example, suppose our dataset contains both math and programming tasks like the following:

```json
[
    {"query": "Solve the equation x + 2 = 5", "solution": "3", "task": "math"},
    {"query": "Write a function to calculate the Fibonacci sequence", "solution": "xxx", "task": "code"},
    {"query": "What is the integral of x^2?", "solution": "xxx", "task": "math"},
    {"query": "Implement a sorting algorithm in Python", "solution": "xxx", "task": "code"}
]
```


We can set up different reward functions to handle math and code data separately. Note that the columns in the dataset will be passed to the reward function, so we can use the `task` column to distinguish between tasks.

Below are examples of reward functions tailored for different tasks:

```python
from swift.plugin import ORM, orms
import random

# Math-specific reward function
class MathRandomReward(ORM):
  def __call__(self, completions, task, **kwargs):
      rewards = []
      for completion, t in zip(completions, task):
          if t == "math":
              import random
              # Implement math accuracy logic
              reward = random.random()
              rewards.append(reward)
          else:
              # Return None for non-math tasks
              rewards.append(None)
      return rewards

# Coding-specific reward function
class CodeRandomReward(ORM):
  def __call__(self, completions, task, **kwargs):
      rewards = []
      for prompt, completion, t in zip(prompts, completions, task):
          if t == "code":
              # Implement coding accuracy logic
              reward = random.random()
              rewards.append(reward)
          else:
              # Return None for non-coding tasks
              rewards.append(None)
      return rewards

orms['math_reward'] = MathRandomReward
orms['code_reward'] = CodeRandomReward
```

For data that does not belong to the current task, we handle it by returning None, ensuring that the reward calculation only applies to data within the designated task.
