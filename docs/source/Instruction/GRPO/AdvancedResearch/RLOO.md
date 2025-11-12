# REINFORCE Leave-One-Out (RLOO)

**版本依赖**：ms-swift>=3.10

[REINFORCE Leave-One-Out (RLOO)](https://arxiv.org/abs/2402.14740) 基于经典的 REINFORCE 策略梯度方法，通过留一法（Leave-One-Out）构造无偏的优势函数基线。

## 算法原理

为便于理解，我们基于 GRPO（Group Relative Policy Optimization）算法进行对比说明。

GRPO 和 RLOO 都采用组内对比的方式来估计优势函数，避免了全局基线估计带来的高方差问题。两者的核心区别主要体现在以下两个方面：

### 区别1：优势函数基线的构造方法

**1. GRPO (Group Relative Policy Optimization)**

GRPO 对每个 prompt 生成 $G$ 个响应样本，使用**组内所有样本的均值和标准差**进行标准化：

$$
\hat{A}_{i} = \frac{R_i - \text{mean}(\{R_j\}_{j=1}^G)}{\text{std}(\{R_j\}_{j=1}^G)}
$$

其中：
- $R_i$ 是第 $i$ 个样本的奖励值
- $\text{mean}(\{R_j\}_{j=1}^G) = \frac{1}{G}\sum_{j=1}^G R_j$ 是组内均值
- $\text{std}(\{R_j\}_{j=1}^G)$ 是组内标准差

**2. RLOO (REINFORCE Leave-One-Out)**

RLOO 对每个 prompt 生成 $K$ 个响应样本，使用 **留一法（Leave-One-Out）** 构造基线，即第 $i$ 个样本的基线为除自己外的其他 $K-1$ 个样本的均值：

$$
\hat{A}_{i} = R_i - \frac{1}{K-1}\sum_{j \neq i} R_j
$$

这个公式可以等价地改写为：

$$
\hat{A}_{i} = \frac{K}{K-1} \left(R_i - \bar{R}\right)
$$

其中 $\bar{R} = \frac{1}{K}\sum_{j=1}^K R_j$ 是组内所有样本的均值。

> **说明**：这里使用 $K$ 对齐论文符号，与 GRPO 中的 $G$ 含义一致，均对应配置参数 `num_generations`

**为什么使用留一法？**

留一法的关键优势在于**无偏性**。对于第 $i$ 个样本，其奖励 $R_i$ 和基线 $\frac{1}{K-1}\sum_{j \neq i} R_j$ 是独立的，因此优势估计是无偏的。相比之下，如果使用包含自身的均值作为基线，会引入偏差。

### 区别2：KL 散度正则化项的处理方式

为防止策略偏离参考策略过远，两种算法都引入了 KL 散度正则化，但处理方式不同：

**GRPO**：将 KL 散度作为独立的正则化项添加到[损失函数](../GetStarted/GRPO.md#算法原理)中：

$$
\mathcal{L}(\theta) = -\mathbb{E}\left[\hat{A}_i \log \pi_\theta(a_i|s_i)\right] + \beta \cdot \text{KL}(\pi_\theta || \pi_{\text{ref}})
$$

**RLOO**：将 KL 散度直接整合到奖励项中，构造修正后的奖励：

$$
R'_i = R_i - \beta \cdot \text{KL}(\pi_\theta || \pi_{\text{ref}})
$$

其中 $\beta$ 是 KL 散度的权重系数（对应参数 `beta`），$\pi_{\text{ref}}$ 是参考策略（通常是 SFT 模型或初始策略）。

## 参数设置

我们可以基于 `GRPOTrainer`，通过设置以下参数实现 RLOO 训练：
```bash
# 基本 RLOO 配置
--advantage_estimator rloo  # 使用 RLOO 的留一法优势函数计算
--kl_in_reward true         # 将 KL 散度项整合到奖励中（RLOO 默认方式）
```

训练可以参考该[脚本](https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo/internal/rloo.sh)

### 重要参数说明

- **`--advantage_estimator`**：选择优势函数估计方法
  - `grpo`（默认）：使用组内均值和标准差进行标准化
  - `rloo`：使用留一法（Leave-One-Out）构造基线

- **`--kl_in_reward`**：控制 KL 散度正则化项的处理位置
  - `false`：KL 散度作为损失函数的独立正则化项（GRPO 方式）
  - `true`：KL 散度直接从奖励中扣除，构造修正后的奖励（RLOO 方式）

- **`--num_generations`**：每个 prompt 生成的样本数量 $K$

- **`--beta`**：KL 散度正则化系数 $\beta$
  - 控制策略更新的保守程度

其他参数与 [GRPO参数](../../Command-line-parameters.md#grpo参数)一致
