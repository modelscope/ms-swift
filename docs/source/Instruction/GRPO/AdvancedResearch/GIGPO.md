# Group-in-Group Policy Optimization (GIGPO)

**版本依赖**：ms-swift>=3.10

[Group-in-Group Policy Optimization (GIGPO)](https://arxiv.org/abs/2505.10978) 是一种改进的策略优化算法，基于分组对比的思想，通过两级分组结构（轨迹级别和步骤级别）提供更细粒度的优势估计。

## 算法原理

GIGPO 基于 GRPO（Group Relative Policy Optimization）算法扩展而来，两者都采用组内对比的方式来估计优势函数，但 GIGPO 引入了更细粒度的步骤级别优势估计，以解决长期序列决策中的信用分配问题。

### 核心创新：两级分组优势估计

GIGPO 的核心创新在于同时使用轨迹级别和步骤级别的相对优势来指导策略优化：

#### 1. 轨迹级别相对优势

轨迹级别相对优势捕获了整个决策过程中智能体的整体表现：

$$
A^E(\tau_i) = \frac{R(\tau_i) - \text{mean}(\{R(\tau_j)\})}{F_{\text{norm}}(\{R(\tau_j)\})}
$$

其中：
- $\tau_i$ 是第 $i$ 个轨迹
- $R(\tau_i) = \sum_t r_t^{(i)}$ 是轨迹的总回报
- $\text{mean}(\{R(\tau_j)\})$ 是组内所有轨迹的平均回报
- $F_{\text{norm}}$ 是归一化因子（可以是标准差或固定值1）

#### 2. 步骤级别相对优势

GIGPO 的关键创新在于**锚点状态分组**机制：
- 识别并分组不同轨迹中重复出现的环境状态，称为**锚点状态**
- 在每个锚点状态组内计算相对优势，提供细粒度的信用分配

步骤级别相对优势的计算过程：

1. **识别锚点状态**：收集所有轨迹中出现的唯一环境状态 $\mathcal{U} = \{\tilde{s}_1, \tilde{s}_2, \ldots, \tilde{s}_U\}$
2. **构建步骤级分组**：
   $$G^S(\tilde{s}) = \{(a_t^{(i)}, r_t^{(i)}) \mid s_t^{(i)} = \tilde{s}, 1 \leq i \leq N, 1 \leq t \leq T\}$$
3. **计算折扣回报**：
   $$R_t^{(i)} = \sum_{k=t}^T \gamma^{k-t} r_k^{(i)}$$
4. **计算步骤相对优势**：
   $$A^S(a_t^{(i)}) = \frac{R_t^{(i)} - \text{mean}(\{R_t^{(j)} \mid (a_t^{(j)}, R_t^{(j)}) \in G^S(\tilde{s})\})}{F_{\text{norm}}(\{R_t^{(j)} \mid (a_t^{(j)}, R_t^{(j)}) \in G^S(\tilde{s})\})}$$

#### 3. 组合优势信号

GIGPO 将轨迹级别和步骤级别的优势信号加权组合，形成最终的优势估计：

$$A(a_t^{(i)}) = A^E(\tau_i) + \omega \cdot A^S(a_t^{(i)})$$

其中 $\omega$ 是平衡两种优势信号的权重系数（对应参数 `gigpo_step_advantage_weight`）。

### 与 GRPO 的主要区别

| 对比维度 | GRPO | GIGPO |
|---------|------|-------|
| **优势估计粒度** | 仅轨迹级别 | 轨迹级别 + 步骤级别 |
| **信用分配** | 粗粒度（整个轨迹） | 细粒度（每个动作步骤） |
| **环境状态利用** | 不利用 | 利用锚点状态分组 |
| **适用场景** | 通用序列生成 | 复杂长期决策任务 |
| **额外参数** | 无 | `gigpo_step_advantage_weight` |

## 参数设置

我们可以基于 `GRPOTrainer`，通过设置以下参数实现 GIGPO 训练：
```bash
# 基本 GIGPO 配置
--advantage_estimator gigpo  # 使用 GIGPO 的两级优势函数计算
--use_gym_env true          # 启用 Gym 环境支持（GIGPO 必需）
--gigpo_step_advantage_weight 1.0  # 步骤级优势的权重系数
```

### 重要参数说明

- **`--advantage_estimator`**：选择优势函数估计方法
  - `grpo`（默认）：仅使用轨迹级别优势
  - `rloo`：使用留一法构造基线
  - `gigpo`：同时使用轨迹级别和步骤级别优势

- **`--use_gym_env`**：是否启用 Gym 环境支持
  - `true`：启用（GIGPO 必需，因为需要环境状态信息）
  - `false`：禁用

- **`--gigpo_step_advantage_weight`**：步骤级优势的权重系数 $\omega$
  - 控制步骤级优势在组合优势中的贡献
  - 取值范围：[0, +∞)
  - 默认值：1.0

- **`--num_generations`**：每个 prompt 生成的样本数量
  - 增加样本数量可以提高优势估计的稳定性

- **`--beta`**：KL 散度正则化系数
  - 控制策略偏离参考策略的程度

其他参数与 [GRPO参数](../../Command-line-parameters.md#grpo参数) 一致
