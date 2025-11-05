# Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning

**Version Requirement**: ms-swift>=3.7

The [paper](https://arxiv.org/abs/2506.01939) finds that when training large language models for reasoning abilities with methods such as RLVR, the key to learning progress lies in a small fraction of high-entropy "minority tokens," rather than the majority of low-entropy tokens.

The paper demonstrates that within the token distribution during model reasoning, only a few high-entropy tokens play a dominant role. These tokens typically appear at critical junctures where the reasoning or decision path diverges the most (e.g., tokens like "wait," "since," etc.), determining whether the model can master complex reasoning tasks. In contrast, most low-entropy tokens contribute little to the model's reasoning ability. The paper proposes computing policy gradients exclusively on high-entropy tokens, discarding gradients for low-entropy tokens.

The formula for token entropy is as follows:

$
H_t := -\sum_{j=1}^{V} p_{t,j} \log p_{t,j}, \qquad \text{where } (p_{t,1}, \cdots, p_{t,V}) = \mathbf{p}_t = \pi_\theta(\cdot | \mathbf{q}, \mathbf{o}_{<t}) = \text{Softmax}\left(\frac{\mathbf{z}_t}{T}\right)
$

Where:
- $\pi_\theta$: The model parameterized by $\theta$;
- $\mathbf{q}$: The input query;
- $\mathbf{o}_{<t} = (o_1, o_2, \cdots, o_{t-1})$: The sequence of tokens generated prior to timestep $t$;
- $V$: Vocabulary size;
- $\mathbf{z}_t \in \mathbb{R}^V$: The pre-softmax logits at timestep $t$;
- $\mathbf{p}_t \in \mathbb{R}^V$: The model's output probability distribution over the vocabulary;
- $T \in \mathbb{R}$: The decoding temperature, controlling the smoothness of the distribution.

Object of entropy computation: $H_t$ is the entropy of the token generation distribution $\mathbf{p}_t$, which measures the uncertainty in the policy $\pi_\theta$ under the given context $(\mathbf{q}, \mathbf{o}_{<t})$.

> "Token entropy" $H_t$ always refers to the uncertainty of the generation distribution $\mathbf{p}_t$ at position $t$, rather than a property of the token $o_t$ itself. In other words, $H_t$ is the entropy of the distribution $\mathbf{p}_t$ at position $t$, and is independent of the sampled token $o_t$.

In practice, during GRPO training, the top_entropy_quantile parameter can be used to control the percentile threshold for entropy filtering. In the experiments from the paper, this parameter is set to 0.2, meaning that only the top 20% of tokens (with the highest entropy) at each sequence position are used for optimization in each batch.

By setting the parameter `log_entropy`, you can record the changes in entropy during training; see the [documentation](../GetStarted/GRPO.md#logged-metrics) for reference.
