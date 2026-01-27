# Group-in-Group Policy Optimization (GIGPO)

**Version Dependency**: ms-swift>=3.10

[Group-in-Group Policy Optimization (GIGPO)](https://arxiv.org/abs/2505.10978) is an improved policy optimization algorithm based on the idea of group comparison, providing more fine-grained advantage estimation through a two-level grouping structure (trajectory level and step level).

## Algorithm Principles

GIGPO is extended from GRPO (Group Relative Policy Optimization). Both algorithms use in-group comparison to estimate advantage functions, but GIGPO introduces finer-grained step-level advantage estimation to solve the credit assignment problem in long-term sequential decision-making.

### Core Innovation: Two-Level Group Advantage Estimation

GIGPO's core innovation lies in simultaneously using trajectory-level and step-level relative advantages to guide policy optimization:

#### 1. Trajectory-Level Relative Advantage

Trajectory-level relative advantage captures the overall performance of the agent in the entire decision-making process:

$$
A^E(\tau_i) = \frac{R(\tau_i) - \text{mean}(\{R(\tau_j)\})}{F_{\text{norm}}(\{R(\tau_j)\})}
$$

Where:
- $\tau_i$ is the i-th trajectory
- $R(\tau_i) = \sum_t r_t^{(i)}$ is the total return of the trajectory
- $\text{mean}(\{R(\tau_j)\})$ is the average return of all trajectories in the group
- $F_{\text{norm}}$ is the normalization factor (can be standard deviation or fixed value 1)

#### 2. Step-Level Relative Advantage

The key innovation of GIGPO is the **anchor state grouping** mechanism:
- Identify and group repeated environmental states across different trajectories, called **anchor states**
- Calculate relative advantages within each anchor state group to provide fine-grained credit assignment

The calculation process of step-level relative advantage:

1. **Identify anchor states**: Collect all unique environmental states from all trajectories $\mathcal{U} = \{\tilde{s}_1, \tilde{s}_2, \ldots, \tilde{s}_U\}$
2. **Construct step-level groups**:
   $$G^S(\tilde{s}) = \{(a_t^{(i)}, r_t^{(i)}) \mid s_t^{(i)} = \tilde{s}, 1 \leq i \leq N, 1 \leq t \leq T\}$$
3. **Calculate discounted returns**:
   $$R_t^{(i)} = \sum_{k=t}^T \gamma^{k-t} r_k^{(i)}$$
4. **Calculate step relative advantages**:
   $$A^S(a_t^{(i)}) = \frac{R_t^{(i)} - \text{mean}(\{R_t^{(j)} \mid (a_t^{(j)}, R_t^{(j)}) \in G^S(\tilde{s})\})}{F_{\text{norm}}(\{R_t^{(j)} \mid (a_t^{(j)}, R_t^{(j)}) \in G^S(\tilde{s})\})}$$

#### 3. Combined Advantage Signal

GIGPO weightedly combines trajectory-level and step-level advantage signals to form the final advantage estimation:

$$A(a_t^{(i)}) = A^E(\tau_i) + \omega \cdot A^S(a_t^{(i)})$$

Where $\omega$ is the weight coefficient that balances the two advantage signals (corresponding to the parameter `gigpo_step_advantage_weight`).

### Main Differences from GRPO

| Comparison Dimension | GRPO | GIGPO |
|---------------------|------|-------|
| **Advantage Estimation Granularity** | Trajectory level only | Trajectory level + Step level |
| **Credit Assignment** | Coarse-grained (entire trajectory) | Fine-grained (each action step) |
| **Environmental State Utilization** | Not utilized | Utilizes anchor state grouping |
| **Applicable Scenarios** | General sequence generation | Complex long-term decision tasks |
| **Additional Parameters** | None | `gigpo_step_advantage_weight` |

## Parameter Settings

We can implement GIGPO training based on `GRPOTrainer` by setting the following parameters:
```bash
# Basic GIGPO configuration
--advantage_estimator gigpo  # Use GIGPO's two-level advantage function calculation
--use_gym_env true          # Enable Gym environment support (required for GIGPO)
--gigpo_step_advantage_weight 1.0  # Weight coefficient for step-level advantage
```

### Important Parameter Descriptions

- **`--advantage_estimator`**：Selects the advantage function estimation method
  - `grpo` (default): Uses only trajectory-level advantage
  - `rloo`: Uses leave-one-out method to construct baseline
  - `gigpo`: Uses both trajectory-level and step-level advantages

- **`--use_gym_env`**：Whether to enable Gym environment support
  - `true`: Enabled (required for GIGPO, as it needs environmental state information)
  - `false`: Disabled

- **`--gigpo_step_advantage_weight`**：Weight coefficient $\omega$ for step-level advantage
  - Controls the contribution of step-level advantage in the combined advantage
  - Range: [0, +∞)
  - Default value: 1.0

- **`--num_generations`**：Number of samples generated per prompt
  - Increasing the number of samples can improve the stability of advantage estimation

- **`--beta`**：KL divergence regularization coefficient
  - Controls the degree of policy deviation from the reference policy

Other parameters are the same as [GRPO parameters](../../Command-line-parameters.md#grpo-parameters)