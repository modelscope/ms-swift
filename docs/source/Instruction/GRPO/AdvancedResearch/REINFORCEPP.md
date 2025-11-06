# REINFORCE++: An Efficient RLHF Algorithm with Robustness to Both Prompt and Reward Models

**版本依赖**：ms-swift>=3.10

[REINFORCE++ Baseline](https://arxiv.org/abs/2501.03262) 是 REINFORCE++ 算法的简化版本，适用于 outcome rewards（response-level 标量奖励）。它与 GRPO 类似，对每个prompt输入采样多条模型输出，并使用组内 baseline 来估计优势函数，主要区别在于标准化时使用的统计量不同。


## 算法原理
为便于理解，我们基于 GRPO（Group Relative Policy Optimization）算法进行对比说明。

GRPO 和 REINFORCE++ Baseline 都采用组内对比的方式来估计优势函数，主要区别在于：

### 区别1：标准化时使用的统计量不同

**GRPO (Group Relative Policy Optimization)**

对每个 prompt 生成 $G$ 个响应样本，使用**组内所有样本的均值和标准差**进行标准化：

$$
\hat{A}_{i} = \frac{R_i - \text{mean}(\{R_j\}_{j=1}^G)}{\text{std}(\{R_j\}_{j=1}^G)}
$$

当设置 `scale_rewards='batch'` 时，使用**原始奖励的批次 std**：

$$
\hat{A}_{i} = \frac{R_i - \text{mean}(\{R_j\}_{j=1}^G)}{\text{std}(\{R_j\}_{j=1}^{N})}
$$

其中 $N$ 是批次中所有样本数。

**REINFORCE++ Baseline**

对每个 prompt 生成 $G$ 个响应样本，先减去组内均值，再使用**减去组内均值后的奖励**的标准差进行标准化：

$$
\begin{align}
\tilde{A}_{i} &= R_i - \text{mean}(\{R_j\}_{j=1}^G) \\
\hat{A}_{i} &= \frac{\tilde{A}_{i}}{\text{std}(\{\tilde{A}_k\}_{k=1}^{N})}
\end{align}
$$

其中 $N$ 是批次中所有样本数。

**关键区别**：
- **GRPO**：标准化时使用**原始奖励 $R$** 的标准差
- **REINFORCE++**：标准化时使用**减去组内均值后的奖励 $\tilde{A}$** 的标准差

### 区别2: KL 散度正则化

与 RLOO 类似，REINFORCE++ Baseline 将 KL 散度整合到奖励项中：

$$
R'_i = R_i - \beta \cdot \text{KL}(\pi_\theta || \pi_{\text{ref}})
$$

其中 $\beta$ 是 KL 散度的权重系数（对应参数 `beta`），$\pi_{\text{ref}}$ 是参考策略。

## 参数设置

我们可以基于 `GRPOTrainer`，通过设置以下参数实现 REINFORCE++ Baseline 训练：

```bash
--advantage_estimator reinforce_plus_plus
--scale_rewards batch
--kl_in_reward true
```

训练可以参考该[脚本](https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo/internal/reinforce_plus_plus.sh)

### 重要参数说明

- **`--advantage_estimator`**：选择优势函数估计方法
  - `grpo`（默认）：标准化时使用原始奖励的标准差
  - `reinforce_plus_plus`：标准化时使用减去组内均值后的奖励的标准差

- **`--kl_in_reward`**：控制 KL 散度正则化项的处理位置
  - `false`：KL 散度作为损失函数的独立正则化项（GRPO 默认）
  - `true`：KL 散度直接从奖励中扣除（REINFORCE++ 原始实现）

- **`--scale_rewards`**：控制标准化方式
  - `group`（默认）：组内标准化
  - `batch`：全局批次标准化（REINFORCE++原始实现）
  - `none`：不进行标准化

- **`--num_generations`**：每个 prompt 生成的样本数量 $G$

- **`--beta`**：KL 散度正则化系数 $\beta$

其他参数参考 [GRPO参数](../../Command-line-parameters.md#grpo参数)
