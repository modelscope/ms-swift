# REINFORCE++ Baseline

**版本依赖**：ms-swift>=3.10

[REINFORCE++ Baseline](https://arxiv.org/abs/2501.03262) 是 REINFORCE++ 算法的简化版本，适用于 outcome rewards（response-level 标量奖励）。它通过组内 baseline 和全局批次级别标准化来估计优势函数，无需价值网络（Critic）。

## 算法原理

### GRPO vs REINFORCE++ Baseline 的主要区别

GRPO 和 REINFORCE++ Baseline 都采用组内对比的方式来估计优势函数，主要区别在于**标准化方式**：

#### 区别：标准化对象和方式不同

**GRPO (Group Relative Policy Optimization)**

对每个 prompt 生成 $G$ 个响应样本，使用**组内所有样本的均值和标准差**进行标准化：

$$
\hat{A}_{i} = \frac{R_i - \text{mean}(\{R_j\}_{j=1}^G)}{\text{std}(\{R_j\}_{j=1}^G)}
$$

当设置 `scale_rewards='batch'` 时，使用**原始奖励的批次 std**：

$$
\hat{A}_{i} = \frac{R_i - \text{mean}(\{R_j\}_{j=1}^G)}{\text{std}(\{R_k\}_{k=1}^{N})}
$$

其中 $N$ 是批次中所有样本数。

**REINFORCE++ Baseline**

对每个 prompt 生成 $K$ 个响应样本，使用**全局白化（Global Whitening）**：

$$
\begin{align}
\tilde{A}_{i} &= R_i - \text{mean}(\{R_j\}_{j=1}^K) \\
\hat{A}_{i} &= \frac{\tilde{A}_{i} - \text{mean}(\{\tilde{A}_k\}_{k=1}^{N})}{\text{std}(\{\tilde{A}_k\}_{k=1}^{N})}
\end{align}
$$

关键区别：
1. **标准化对象**：REINFORCE++ 使用 $\tilde{A}$ (减去组内均值后的优势) 的统计量，而 GRPO 使用原始奖励 $R$ 的统计量
2. **二次中心化**：REINFORCE++ 会再次减去 $\tilde{A}$ 的均值，GRPO 不进行此操作

### 数值示例

假设有 2 个 prompts，每个生成 2 个 responses：

```python
Group 1: rewards = [100, 90], group_mean = 95
Group 2: rewards = [10, 0],   group_mean = 5
```

**GRPO with scale_rewards='batch':**
```python
advantages = [5, -5, 5, -5]  # rewards - group_mean
std(rewards) = 52.28
result = [5, -5, 5, -5] / 52.28 = [0.096, -0.096, 0.096, -0.096]
```

**REINFORCE++ with scale_rewards='batch':**
```python
advantages = [5, -5, 5, -5]  # rewards - group_mean
mean(advantages) = 0
std(advantages) = 5.77
result = ([5, -5, 5, -5] - 0) / 5.77 = [0.866, -0.866, 0.866, -0.866]
```

**梯度大小差异约 9 倍！** REINFORCE++ 的梯度更加稳定，不受原始奖励绝对值范围的影响。

### KL 散度正则化

与 RLOO 类似，REINFORCE++ Baseline 将 KL 散度直接整合到奖励项中：

$$
R'_i = R_i - \beta \cdot \text{KL}(\pi_\theta || \pi_{\text{ref}})
$$

其中 $\beta$ 是 KL 散度的权重系数（对应参数 `beta`），$\pi_{\text{ref}}$ 是参考策略（通常是 SFT 模型或初始策略）。

## 参数设置

我们可以基于 `GRPOTrainer`，通过设置以下参数实现 REINFORCE++ Baseline 训练：

```bash
# 基本 REINFORCE++ 配置
--advantage_estimator reinforce_plus_plus  # 使用 REINFORCE++ 优势函数计算
--scale_rewards batch                       # 使用全局批次级别标准化（推荐）
```

训练可以参考该[脚本](https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo/internal/reinforce_plus_plus.sh)

### 重要参数说明

- **`--advantage_estimator`**：选择优势函数估计方法
  - `grpo`（默认）：使用组内均值和标准差进行标准化
  - `rloo`：使用留一法（Leave-One-Out）构造基线
  - `reinforce_plus_plus`：使用组内 baseline + 全局白化

- **`--kl_in_reward`**：控制 KL 散度正则化项的处理位置（自动设置）
  - `false`：KL 散度作为损失函数的独立正则化项（GRPO 默认）
  - `true`：KL 散度直接从奖励中扣除（RLOO/REINFORCE++ 默认）
  - **注意**：当设置 `advantage_estimator=reinforce_plus_plus` 时，`kl_in_reward` 会自动设置为 `true`

- **`--scale_rewards`**：控制标准化方式
  - `group`（默认）：组内标准化
  - `batch`：全局批次标准化（**推荐用于 REINFORCE++**）
  - `none`：不进行标准化

- **`--num_generations`**：每个 prompt 生成的样本数量 $K$
  - 推荐设置为 4-8

- **`--beta`**：KL 散度正则化系数 $\beta$
  - REINFORCE++ 论文推荐值：0.04
  - 控制策略更新的保守程度

其他参数与 [GRPO参数](../../命令行参数.md#grpo参数)一致

### 完整示例

```bash
# REINFORCE++ Baseline 完整配置
swift rlhf \
    --rlhf_type grpo \
    --model_id_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset alpaca-zh \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 5e-7 \
    --beta 0.04 \
    --advantage_estimator reinforce_plus_plus \
    --scale_rewards batch \
    --num_generations 4
```

## 算法对比

### 三种算法的完整对比

| 特性 | GRPO | RLOO | REINFORCE++ Baseline |
|------|------|------|---------------------|
| **优势函数基线** | 组内均值 | Leave-One-Out 均值 | 组内均值 |
| **标准化对象** | 原始奖励 $R$ | 原始奖励 $R$ | 优势 $\tilde{A}$ |
| **二次中心化** | ❌ | ❌ | ✅ |
| **KL 散度位置** | 损失函数中 | 奖励中 | 奖励中 |
| **num_generations** | ≥1 | >1（推荐4-8） | ≥1（推荐4-8） |
| **推荐 scale_rewards** | `group` / `batch` | `group` / `batch` | `batch` |
| **参数设置** | `--advantage_estimator grpo`<br>`--kl_in_reward false` | `--advantage_estimator rloo`<br>`--kl_in_reward true` | `--advantage_estimator reinforce_plus_plus`<br>`--scale_rewards batch` |
