# Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning

> 注意，使用该功能需要安装 ms-swift 和 trl 源码版本
>```
>git clone https://github.com/modelscope/ms-swift.git
>cd ms-swift
>pip install -e .
>pip install git+https://github.com/huggingface/trl.git
>```


[论文](https://arxiv.org/abs/2503.14476)发现在以 RLVR等方法训练大型语言模型推理能力时，驱动学习进步的关键在于一小部分高熵“少数 token”，而并非大多数信息熵低的 token。

论文发现，在模型推理的 token 分布中，只有极少数信息熵较高的 token 起到了主导作用。这些 token 往往出现在推理和决策路径分歧最大的关键节点（如 "wait"、"since" 等），决定了模型能否习得复杂推理任务。而大多数熵低的 token 对模型推理能力的提升作用有限。论文提出只对高熵 token 计算策略梯度、舍弃低熵 token 的梯度。


token 熵公式如下

$
H_t := -∑_{j=1}^{V} p_{t,j} \log p_{t,j}, where (p_{t,1}, ···, p_{t,V}) = \mathbf{p}_t = π_θ(\cdot | \mathbf{q}, \mathbf{o}_{<t}) = \text{Softmax}(\frac{\mathbf{z}_t}{T}) \
$

在实践中，我们可以在 GRPO 训练中通过参数 `token_entropy_percentile_threshold` 控制熵过滤的分位数。论文实验设置该参数为 0.8，即每次仅对处于熵分布前 20% 的 token 进行训练优化。
