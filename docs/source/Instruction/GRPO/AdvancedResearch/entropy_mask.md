# Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning

**版本依赖**：ms-swift>=3.7

[论文](https://arxiv.org/abs/2506.01939)发现在以 RLVR等方法训练大型语言模型推理能力时，驱动学习进步的关键在于一小部分高熵“少数 token”，而并非大多数信息熵低的 token。

论文指出，在模型推理的 token 分布中，只有极少数信息熵较高的 token 起到了主导作用。这些 token 往往出现在推理和决策路径分歧最大的关键节点（如 "wait"、"since" 等），决定了模型能否习得复杂推理任务。而大多数熵低的 token 对模型推理能力的提升作用有限。论文提出只对高熵 token 计算策略梯度、舍弃低熵 token 的梯度。


token 熵公式如下

$
H_t := -∑_{j=1}^{V} p_{t,j} \log p_{t,j}, \qquad where (p_{t,1}, ···, p_{t,V}) = \mathbf{p}_t = π_θ(\cdot | \mathbf{q}, \mathbf{o}_{<t}) = \text{Softmax}(\frac{\mathbf{z}_t}{T})
$

其中
- $\pi_\theta$：参数为 $\theta$ 的模型
- $\mathbf{q}$：输入查询（input query）。
- $\mathbf{o}_{<t} = (o_1, o_2, \cdots, o_{t-1})$：时间步 $t$ 之前已生成的 token 序列。
- $V$：词表大小（vocabulary size）。
- $\mathbf{z}_t \in \mathbb{R}^V$：时间步 $t$ 的 pre-softmax 逻辑值（logits）。
- $\mathbf{p}_t \in \mathbb{R}^V$：模型对词表的概率分布。
- $T \in \mathbb{R}$：解码温度（decoding temperature），控制分布的平滑程度。

熵的计算对象：$H_t$ 是 token 生成分布 $\mathbf{p}_t$ 的熵，用于衡量训练策略 $\pi_\theta$ 在给定上下文 $(\mathbf{q}, \mathbf{o}_{<t})$ 下的不确定性。

> "Token entropy" $H_t$ 始终指向位置 $t$ 的生成分布 $\mathbf{p}_t$ 的不确定性，而非 token $o_t$ 本身的属性。即$H_t$ 是位置 $t$ 对应分布 $\mathbf{p}_t$ 的熵，与采样得到的 token $o_t$ 无关。


在实践中，我们可以在 GRPO 训练中通过参数 `top_entropy_quantile` 控制训练范围。论文实验设置该参数为 0.2，即每次仅对处于熵分布前 20% 的 token 进行训练优化。

同时使用参数`log_entropy`，可以记录训练过程中的熵值变化，参考[文档](../GetStarted/GRPO.md#logged-metrics)
