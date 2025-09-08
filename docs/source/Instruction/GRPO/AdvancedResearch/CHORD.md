# On-Policy RL Meets Off-Policy Experts: Harmonizing SFT and RL via Dynamic Weighting (CHORD)

**版本依赖**：ms-swift>=3.9

本文档介绍论文 [On-Policy RL Meets Off-Policy Experts: Harmonizing SFT and RL via Dynamic Weighting](https://arxiv.org/abs/2508.11408) 中提出的 CHORD 算法。CHORD 的核心思想是在强化学习过程中，动态融合专家数据（SFT），通过 全局权重 μ + token 级别权重 φ 的双重控制机制，在模仿与探索之间实现平衡。

## 算法概述
CHORD 算法通过在 GRPO loss 中引入 **SFT loss**，实现动态混合训练。总体目标函数为：

$$
    \mathcal{L}_{\text{CHORD}} = (1 - \mu) \cdot \mathcal{L}_{\text{GRPO}} + \mu \cdot \mathcal{L}_{\text{SFT}}
$$

其中：
- $\mathcal{L}_{\text{GRPO}}$：基于 on-policy 采样的强化学习损失（类似 PPO）。
- $\mathcal{L}_{\text{SFT}}$：监督微调损失。
- $\mu \in [0, 1]$：全局平衡系数，控制 SFT 信号在总梯度中的贡献。

### 参数配置（数据与批量大小）
我们可以基于 GRPO 训练实现 CHORD 训练。

CHORD 需要在训练时指定额外的 SFT 数据集和批量大小：
- `chord_sft_dataset`: 用于提供专家数据的 SFT 数据集。
- `chord_sft_per_device_train_batch_size`: 每个设备的 SFT mini-batch 大小。

---

## 两种 CHORD 变体

论文提出了两种算法变体：**CHORD-µ** 和 **CHORD-ϕ**。

### CHORD-µ
通过在训练过程中逐步 **衰减 μ**，实现从模仿专家到自主探索的过渡。

**参数：**
- `chord_mu_peak`：μ 的峰值。
- `chord_mu_valley` μ 的衰减终值。
- `chord_mu_warmup_steps` μ 值上升至峰值的训练步数。
- `chord_mu_decay_steps` μ 从峰值衰减到谷值的训练步数。

### CHORD-ϕ（Token 级加权）
**CHORD-ϕ** 通过 **token-wise 权重函数 φ** 动态控制每个专家 token 的梯度贡献。

**φ 定义：**
$$
    \phi(y_t^\star, \pi_\theta) = p_t \cdot (1 - p_t)
$$

其中：
- $p_t = \pi_\theta(y_t^\star \mid x, y_{<t}^\star)$：模型当前预测专家 token 的概率。
- 当 $p_t ≈ 0.5$（模型不确定时），φ 取最大值 → 强化学习不确定的 token。
- 当 $p_t ≈ 0$ 或 $p_t ≈ 1$，φ → 0 → 避免对过于确定或完全不会的 token 过度学习。

**开启 φ 加权的参数**：
- `chord_enable_phi_function: bool = False`
  - 设置为 `True` 即启用 token-wise 权重 φ。

注：如果使用常数 μ 值 ，设置 chord_mu_peak 与 chord_mu_valley 相同

<details>
<summary>mu值衰减与loss计算代码实现</summary>
请参考`GRPOTrainer`的`_compute_chord_loss`方法：
</details>

训练参考该[脚本](https://github.com/modelscope/ms-swift/tree/main/examples/train/grpo/internal/chord.sh)
