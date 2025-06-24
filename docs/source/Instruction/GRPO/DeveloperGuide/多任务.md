# 多任务训练
我们可以在数据集中添加一个用于标识任务类型的列，并在奖励函数/奖励模型插件中根据任务类型进行判断，从而实现多任务训练。假设数据集中包含数学和编程任务，比如：

```
    {"query": "Solve the equation x + 2 = 5", "solution": "3", "task": "math"},
    {"query": "Write a function to calculate the Fibonacci sequence", "solution": "xxx", "task": "code"},
    {"query": "What is the integral of x^2?", "solution": "xxx", "task": "math"},
    {"query": "Implement a sorting algorithm in Python", "solution": "xxx", "task": "code"},
```

我们可以设置不同的奖励函数来分别处理数学数据和代码数据，注意数据集中的列会传入奖励函数，所以我们可以通过 `task` 列

下面是针对不同任务的奖励函数的示例：

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
              # imple math accuracy logic
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
              # imple coding accuracy logic
              reward = random.random()
              rewards.append(reward)
          else:
              # Return None for non-coding tasks
              rewards.append(None)
      return rewards

orms['math_reward'] = MathRandomReward
orms['code_reward'] = CodeRandomReward
```
对于非当前任务的数据， 通过返回 None 来处理，从而使得奖励相关仅计算任务内的数据。
